"""Shared-encoder G2PW converter.

The upstream G2PW inference path runs the full BERT encoder once per
polyphonic *position*, each time on a +/-16 character window around that
position. The monolithic graph can be split at the encoder's final hidden
states: the encoder runs once per sentence on the whole token sequence,
and the small head scores every polyphonic position of the sentence from
that shared encoding. Sentences that exceed BERT's token limit fall back
to the upstream per-position window layout, batched through the same two
graphs.

Sharing the encoder changes the effective context from the training-time
window to the whole sentence, so predictions can differ from the
monolithic graph on longer inputs.
"""

from __future__ import annotations

import contextlib
import os
from typing import Optional, Sequence

from ..errors import BackendError

MONO_FILENAME = "g2pw.onnx"
MONO_INT8_FILENAME = "g2pw_int8.onnx"
ENCODER_FILENAME = "g2pw_encoder.onnx"
ENCODER_INT8_FILENAME = "g2pw_encoder_int8.onnx"
HEAD_FILENAME = "g2pw_head.onnx"
HEAD_INT8_FILENAME = "g2pw_head_int8.onnx"

_MAX_LEN = 512


@contextlib.contextmanager
def _redirect_g2pw_session(target_path: Optional[str]):
    """Skip parsing ``g2pw.onnx`` inside upstream ``G2PWConverter.__init__``.

    The upstream constructor always builds an ONNX session for the 606 MB
    monolithic graph, even when the caller replaces (or never uses) that
    session right after. While this context is active its
    ``onnxruntime.InferenceSession`` construction is redirected to
    ``target_path`` — or short-circuited to ``None`` when only the
    converter metadata (tokenizer, dictionaries) is needed.
    """
    import onnxruntime

    real_session_class = onnxruntime.InferenceSession
    if target_path is None:
        onnxruntime.InferenceSession = lambda *args, **kwargs: None
    else:
        onnxruntime.InferenceSession = lambda _path, **kwargs: real_session_class(target_path, **kwargs)
    try:
        yield
    finally:
        onnxruntime.InferenceSession = real_session_class


class SplitG2PWConverter:
    """Drop-in replacement for ``g2pw.G2PWConverter`` with a shared encoder.

    ``base`` supplies the tokenizer, label metadata, and monophonic-char
    handling (typically an upstream ``G2PWConverter`` instance whose
    monolithic session is never used). ``encoder_session`` and
    ``head_session`` are ONNX Runtime sessions for the two split graphs.
    ``mapper`` overrides the (tokenizer, text) -> tokens/mapping helper;
    when omitted, ``g2pw.utils.tokenize_and_map`` is imported lazily.
    ``boundary_name`` is the head's encoder-output input; it defaults to
    the name used by ``scripts/prepare_g2pw_split.py``.
    """

    def __init__(self, base, encoder_session, head_session, mapper=None, boundary_name: str = "1663") -> None:
        self._base = base
        self._encoder = encoder_session
        self._head = head_session
        self._mapper = mapper
        self._boundary = boundary_name
        self._window = getattr(getattr(base, "config", None), "window_size", 32)
        self._polyphonic = set(base.chars)
        self._monophonic = dict(base.monophonic_chars)

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str,
        *,
        model_source: str,
        enable_non_tradional_chinese: bool = False,
        onnx_threads: Optional[int] = None,
        encoder_name: str = ENCODER_FILENAME,
        head_name: str = HEAD_FILENAME,
    ) -> "SplitG2PWConverter":
        try:
            import onnxruntime
            from g2pw import G2PWConverter
        except ImportError as error:
            raise BackendError(
                "G2PW backend is unavailable; install it with 'pip install g2p-mix[g2pw]'"
            ) from error

        # Only the converter metadata (tokenizer, pronunciation
        # dictionaries) is needed here; its monolithic session is never
        # used, so skip parsing the 606 MB graph entirely.
        with _redirect_g2pw_session(None):
            base = G2PWConverter(
                model_dir=model_dir,
                style="pinyin",
                model_source=model_source,
                num_workers=0,
                enable_non_tradional_chinese=enable_non_tradional_chinese,
            )

        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = onnx_threads if onnx_threads is not None else 2
        sessions = []
        for filename in (encoder_name, head_name):
            path = os.path.join(model_dir, filename)
            if not os.path.exists(path):
                raise BackendError(f"split G2PW model file missing: {path}")
            sessions.append(
                onnxruntime.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
            )
        # Discover the head input that carries the encoder output instead
        # of assuming the exporter's tensor name.
        query_inputs = {"input_ids", "position_ids", "char_ids", "phoneme_mask"}
        boundary = [item.name for item in sessions[1].get_inputs() if item.name not in query_inputs]
        if len(boundary) != 1:
            raise BackendError(f"cannot identify the encoder-output input of the G2PW head graph: {boundary}")
        return cls(base, sessions[0], sessions[1], boundary_name=boundary[0])

    def _tokenize_and_map(self, tokenizer, text):
        if self._mapper is not None:
            return self._mapper(tokenizer, text)
        from g2pw.utils import tokenize_and_map

        return tokenize_and_map(tokenizer, text)

    def __call__(self, sentences):
        base = self._base
        if isinstance(sentences, str):
            sentences = [sentences]
        if base.enable_non_tradional_chinese:
            sentences = [base._convert_s2t(sent) for sent in sentences]

        results = []
        for sentence in sentences:
            partial = [None] * len(sentence)
            positions = []
            for index, char in enumerate(sentence):
                if char in self._polyphonic:
                    positions.append(index)
                elif char in self._monophonic:
                    partial[index] = base.style_convert_func(self._monophonic[char])
                elif char in base.char_bopomofo_dict:
                    partial[index] = base.style_convert_func(base.char_bopomofo_dict[char][0])
            if positions:
                lowered = sentence.lower()
                try:
                    tokens, text2token, _ = self._tokenize_and_map(base.tokenizer, lowered)
                except Exception:
                    tokens = []
                if not tokens:
                    pass  # leave every position to the backend fallback
                elif len(tokens) + 2 <= _MAX_LEN:
                    self._predict_shared(lowered, positions, tokens, text2token, partial)
                else:
                    self._predict_windowed(lowered, positions, partial)
            results.append(partial)
        return results

    def _label_for(self, label_index) -> Optional[str]:
        return self._base.style_convert_func(self._base.labels[int(label_index)])

    def _predict_shared(self, lowered, positions, tokens, text2token, partial) -> None:
        import numpy as np

        base = self._base
        ids = base.tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"])
        input_ids = np.array([ids], dtype=np.int64)
        hidden = self._encoder.run(
            [],
            {
                "input_ids": input_ids,
                "token_type_ids": np.zeros_like(input_ids),
                "attention_mask": np.ones_like(input_ids),
            },
        )[0]
        count = len(positions)
        probs = self._head.run(
            [],
            {
                self._boundary: np.repeat(hidden, count, axis=0),
                "input_ids": np.repeat(input_ids, count, axis=0),
                "position_ids": np.array([text2token[q] + 1 for q in positions], dtype=np.int64),
                "char_ids": np.array([base.chars.index(lowered[q]) for q in positions], dtype=np.int64),
                "phoneme_mask": np.array(
                    [self._phoneme_mask(lowered[q]) for q in positions],
                    dtype=np.float32,
                ),
            },
        )[0]
        for index, row in zip(positions, probs.argmax(-1)):
            partial[index] = self._label_for(row)

    def _predict_windowed(self, lowered, positions, partial) -> None:
        import numpy as np

        base = self._base
        half = self._window // 2
        batch_ids, batch_positions, batch_chars, batch_masks, kept = [], [], [], [], []
        for position in positions:
            start = max(0, position - half)
            window = lowered[start : min(len(lowered), position + half)]
            try:
                tokens, text2token, _ = self._tokenize_and_map(base.tokenizer, window)
            except Exception:
                continue
            query = position - start
            batch_ids.append(base.tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"]))
            batch_positions.append(text2token[query] + 1)
            batch_chars.append(base.chars.index(window[query]))
            batch_masks.append(self._phoneme_mask(window[query]))
            kept.append(position)
        if not kept:
            return
        width = max(len(ids) for ids in batch_ids)
        input_ids = np.zeros((len(kept), width), dtype=np.int64)
        attention_mask = np.zeros((len(kept), width), dtype=np.int64)
        for row, ids in enumerate(batch_ids):
            input_ids[row, : len(ids)] = ids
            attention_mask[row, : len(ids)] = 1
        hidden = self._encoder.run(
            [],
            {
                "input_ids": input_ids,
                "token_type_ids": np.zeros_like(input_ids),
                "attention_mask": attention_mask,
            },
        )[0]
        probs = self._head.run(
            [],
            {
                self._boundary: hidden,
                "input_ids": input_ids,
                "position_ids": np.array(batch_positions, dtype=np.int64),
                "char_ids": np.array(batch_chars, dtype=np.int64),
                "phoneme_mask": np.array(batch_masks, dtype=np.float32),
            },
        )[0]
        for index, row in zip(kept, probs.argmax(-1)):
            partial[index] = self._label_for(row)

    def _phoneme_mask(self, query_char: str) -> Sequence[float]:
        base = self._base
        return [1.0 if i in base.char2phonemes[query_char] else 0.0 for i in range(len(base.labels))]
