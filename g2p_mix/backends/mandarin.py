from __future__ import annotations

import os
import unicodedata
from collections.abc import Sequence as SequenceABC
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from ..errors import AlignmentError, BackendError, G2PError
from ..models import (
    ChineseDialect,
    Language,
    PhoneAlphabet,
    Pronunciation,
    PronunciationUnit,
    UnknownPolicy,
)
from ..phonetics import split_pinyin
from ..resources import install_pypinyin_overrides
from .base import (
    BackendCapabilities,
    PronunciationRequest,
    encode_character_projection,
    unknown_unit,
)

G2PWPredictor = Callable[[str], Sequence[Sequence[Optional[str]]]]
G2PW_PINYIN_CORRECTIONS = {
    ("和", "han4"): "he2",
    ("崖", "yai2"): "ya2",
    ("儿", "er1"): "er2",
    ("兒", "er1"): "er2",
    ("削", "xue4"): "xue1",
    ("嵌", "qian1"): "qian4",
    ("炔", "jue2"): "que1",
    ("综", "zong4"): "zong1",
    ("綜", "zong4"): "zong1",
    ("缉", "qi4"): "ji1",
    ("緝", "qi4"): "ji1",
    ("识", "shi4"): "shi2",
    ("識", "shi4"): "shi2",
    ("朴", "pu2"): "pu3",
    ("湮", "yin1"): "yan1",
    ("晟", "cheng2"): "sheng4",
}
G2PW_MAINLAND_LIU2_PHRASES = (
    "分馏",
    "分餾",
    "干馏",
    "干餾",
    "乾馏",
    "乾餾",
    "精馏",
    "精餾",
    "蒸馏",
    "蒸餾",
    "馏出",
    "餾出",
    "馏分",
    "餾分",
    "馏程",
    "餾程",
)
G2PW_MAINLAND_JUN1_PHRASES = (
    "古菌",
    "抗菌",
    "杆菌",
    "桿菌",
    "杀菌",
    "殺菌",
    "球菌",
    "病菌",
    "真菌",
    "细菌",
    "細菌",
    "腐菌",
    "菌丝",
    "菌絲",
    "菌体",
    "菌體",
    "菌株",
    "菌核",
    "菌根",
    "菌盖",
    "菌蓋",
    "菌种",
    "菌種",
    "菌类",
    "菌類",
    "菌群",
    "菌落",
    "菌褶",
    "菌门",
    "菌門",
    "菌齿",
    "菌齒",
    "菌管",
    "菌酶",
)
G2PW_BAI4_COMPOUNDS = ("梵呗", "梵唄", "美呗", "美唄", "赞呗", "讚唄")


def _index_is_in_phrase(text: str, index: int, phrases: Sequence[str]) -> bool:
    for phrase in phrases:
        start = text.find(phrase)
        while start >= 0:
            if start <= index < start + len(phrase):
                return True
            start = text.find(phrase, start + 1)
    return False


def _only_spacing_or_punctuation(text: str) -> bool:
    return all(char.isspace() or unicodedata.category(char).startswith("P") for char in text)


def _normalize_g2pw_pinyin(text: str, index: int, syllable: str) -> str:
    char = text[index]
    corrected = G2PW_PINYIN_CORRECTIONS.get((char, syllable))
    if corrected is not None:
        return corrected

    if char in "馏餾" and syllable == "liu4" and _index_is_in_phrase(text, index, G2PW_MAINLAND_LIU2_PHRASES):
        return "liu2"
    if char == "菌" and syllable == "jun4" and _index_is_in_phrase(text, index, G2PW_MAINLAND_JUN1_PHRASES):
        return "jun1"
    if char == "耶" and syllable == "ye2":
        following = text[index + 1 : index + 2]
        if following and following != "非" and not _only_spacing_or_punctuation(following):
            return "ye1"
    if char in "呗唄" and syllable == "bai4":
        if _only_spacing_or_punctuation(text[index + 1 :]) and not _index_is_in_phrase(
            text, index, G2PW_BAI4_COMPOUNDS
        ):
            return "bei5"
    return syllable


def _flatten_syllables(values: Iterable) -> List[str]:
    result = []
    for value in values:
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise AlignmentError(f"Expected one pronunciation per character, got {value!r}")
            value = value[0]
        if not isinstance(value, str):
            raise AlignmentError(f"Invalid pronunciation value: {value!r}")
        result.append(value)
    return result


def _build_pronunciation(
    token,
    syllables: Sequence[str],
    backend: str,
    strict: bool,
    unknown_policy: UnknownPolicy,
) -> Pronunciation:
    if len(syllables) != len(token.text):
        raise AlignmentError(
            f"{backend} returned {len(syllables)} syllables for {token.text!r} ({len(token.text)} characters)"
        )

    units = []
    for index, (char, syllable) in enumerate(zip(token.text, syllables)):
        try:
            initial, final, tone = split_pinyin(syllable, strict=strict)
        except (TypeError, ValueError) as error:
            if unknown_policy is UnknownPolicy.PRESERVE and _is_unpronounced_character(char, syllable):
                units.append(unknown_unit(token, index, PhoneAlphabet.PINYIN))
                continue
            raise BackendError(
                f"{backend} returned an invalid Mandarin pronunciation for {char!r}: {syllable!r}"
            ) from error
        phones = tuple(phone for phone in (initial, final) if phone)
        units.append(
            PronunciationUnit(
                text=char,
                source_spans=(token.source_spans[index],),
                phones=phones,
                tone=tone,
                alphabet=PhoneAlphabet.PINYIN,
                native=syllable,
            )
        )
    return Pronunciation(token_id=token.id, units=tuple(units), backend=backend)


def _is_unpronounced_character(char: str, syllable: object) -> bool:
    return isinstance(syllable, str) and syllable.rstrip("12345") == char


class PypinyinBackend:
    name = "pypinyin"
    capabilities = BackendCapabilities(
        language=Language.CHINESE,
        dialect=ChineseDialect.MANDARIN,
        alphabet=PhoneAlphabet.PINYIN,
    )

    def __init__(
        self,
        strict: bool = False,
        converter=None,
        unknown_policy: UnknownPolicy = UnknownPolicy.STRICT,
    ) -> None:
        self._strict = strict
        self._converter = converter
        self._engine = None
        self._unknown_policy = UnknownPolicy(unknown_policy)

    def _get_converter(self):
        if self._converter is None:
            from pypinyin.converter import UltimateConverter

            self._converter = UltimateConverter(neutral_tone_with_five=True)
        return self._converter

    def _get_engine(self):
        if self._engine is None:
            from pypinyin.core import Pinyin

            install_pypinyin_overrides()
            self._engine = Pinyin(converter=self._get_converter())
        return self._engine

    def _convert(self, text: str) -> List[str]:
        try:
            from pypinyin import Style

            values = self._get_engine().pinyin(
                text,
                style=Style.TONE3,
                heteronym=False,
                errors="default",
                strict=True,
            )
            return _flatten_syllables(values)
        except G2PError:
            raise
        except Exception as error:
            raise BackendError(f"Pypinyin prediction failed for {text!r}") from error

    def predict(
        self,
        request: PronunciationRequest,
    ) -> Mapping[int, Pronunciation]:
        return {
            token.id: _build_pronunciation(
                token,
                self._convert(token.text),
                backend=self.name,
                strict=self._strict,
                unknown_policy=self._unknown_policy,
            )
            for token in request.target_tokens
        }


def _resolve_g2pw_model_path(model_dir: str, fp32_name: str, int8_name: str, quantize: bool) -> str:
    """Pick the fp32 or dynamic-INT8 variant of a G2PW graph file.

    The fp32 variant is the default: it scored marginally better on clean
    CPP (96.0716% vs 96.0269% target exact), so INT8 stays an explicit
    opt-in and a missing INT8 file is an error rather than a silent swap.
    """
    if not quantize:
        return os.path.join(model_dir, fp32_name)
    int8_path = os.path.join(model_dir, int8_name)
    if not os.path.exists(int8_path):
        raise BackendError(f"quantized G2PW model file missing: {int8_path}")
    return int8_path


_G2PW_COMMON_PATTERNS = [
    "G2PWModel/version",
    "G2PWModel/config.py",
    "G2PWModel/POLYPHONIC_CHARS.txt",
    "G2PWModel/MONOPHONIC_CHARS.txt",
    "bert-base-chinese/vocab.txt",
    "bert-base-chinese/tokenizer.json",
]


def _g2pw_download_patterns(use_split_model: bool, quantize: bool) -> List[str]:
    """Repo files a given G2PW configuration actually downloads.

    Everything else in the snapshot (the monolithic graph on split/INT8
    paths, the unused ``pytorch_model.bin`` tokenizer checkpoint, the
    derived graphs on the default path) stays on the hub.
    """
    from .g2pw_split import (
        ENCODER_FILENAME,
        ENCODER_INT8_FILENAME,
        HEAD_FILENAME,
        HEAD_INT8_FILENAME,
        MONO_FILENAME,
        MONO_INT8_FILENAME,
    )

    if use_split_model:
        graph_names = [
            ENCODER_INT8_FILENAME if quantize else ENCODER_FILENAME,
            HEAD_INT8_FILENAME if quantize else HEAD_FILENAME,
        ]
    else:
        graph_names = [MONO_INT8_FILENAME if quantize else MONO_FILENAME]
    return [*_G2PW_COMMON_PATTERNS, *(f"G2PWModel/{name}" for name in graph_names)]


class G2PWBackend:
    name = "g2pw"
    capabilities = BackendCapabilities(
        language=Language.CHINESE,
        dialect=ChineseDialect.MANDARIN,
        alphabet=PhoneAlphabet.PINYIN,
    )

    def __init__(
        self,
        strict: bool = False,
        converter=None,
        foreign_placeholder: str = "，",
        num_workers: int = 0,
        onnx_threads: Optional[int] = None,
        use_split_model: bool = False,
        quantize: bool = False,
        unknown_policy: UnknownPolicy = UnknownPolicy.STRICT,
    ) -> None:
        if len(foreign_placeholder) != 1:
            raise ValueError("The model placeholder must be exactly one character")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if onnx_threads is not None and onnx_threads < 1:
            raise ValueError("onnx_threads must be a positive integer or None")
        if not isinstance(quantize, bool):
            raise ValueError("quantize must be a bool")
        self._strict = strict
        self._converter = converter
        self._placeholder = foreign_placeholder
        self._num_workers = num_workers
        self._onnx_threads = onnx_threads
        self._use_split_model = use_split_model
        self._quantize = quantize
        self._unknown_policy = UnknownPolicy(unknown_policy)
        self._fallback = PypinyinBackend(
            strict=strict,
            unknown_policy=self._unknown_policy,
        )

    def _get_converter(self) -> G2PWPredictor:
        if self._converter is None:
            try:
                from g2pw import G2PWConverter
                from modelscope import snapshot_download
            except ImportError as error:
                raise BackendError(
                    "G2PW backend is unavailable; install it with 'pip install g2p-mix[g2pw]'"
                ) from error

            try:
                repo_dir = snapshot_download(
                    "pengzhendong/g2pw",
                    allow_patterns=_g2pw_download_patterns(self._use_split_model, self._quantize),
                )
            except G2PError:
                raise
            except Exception as error:
                raise BackendError("G2PW model snapshot download failed") from error
            try:
                model_dir = f"{repo_dir}/G2PWModel"
                if self._use_split_model:
                    from .g2pw_split import (
                        ENCODER_FILENAME,
                        ENCODER_INT8_FILENAME,
                        HEAD_FILENAME,
                        HEAD_INT8_FILENAME,
                        SplitG2PWConverter,
                    )

                    encoder_path = _resolve_g2pw_model_path(
                        model_dir, ENCODER_FILENAME, ENCODER_INT8_FILENAME, self._quantize
                    )
                    head_path = _resolve_g2pw_model_path(model_dir, HEAD_FILENAME, HEAD_INT8_FILENAME, self._quantize)
                    converter = SplitG2PWConverter.from_model_dir(
                        model_dir,
                        model_source=f"{repo_dir}/bert-base-chinese",
                        enable_non_tradional_chinese=True,
                        onnx_threads=self._onnx_threads,
                        encoder_name=os.path.basename(encoder_path),
                        head_name=os.path.basename(head_path),
                    )
                else:
                    from .g2pw_split import MONO_FILENAME, MONO_INT8_FILENAME, _redirect_g2pw_session

                    model_path = _resolve_g2pw_model_path(model_dir, MONO_FILENAME, MONO_INT8_FILENAME, self._quantize)
                    if model_path != os.path.join(model_dir, MONO_FILENAME):
                        # Upstream __init__ always parses the fp32 graph
                        # only to have its session replaced; build it on
                        # the quantized file from the start instead.
                        with _redirect_g2pw_session(model_path):
                            converter = G2PWConverter(
                                model_dir=model_dir,
                                style="pinyin",
                                model_source=f"{repo_dir}/bert-base-chinese",
                                num_workers=self._num_workers,
                                enable_non_tradional_chinese=True,
                            )
                    else:
                        converter = G2PWConverter(
                            model_dir=model_dir,
                            style="pinyin",
                            model_source=f"{repo_dir}/bert-base-chinese",
                            num_workers=self._num_workers,
                            enable_non_tradional_chinese=True,
                        )
                    converter.num_workers = self._num_workers
                    if self._onnx_threads is not None:
                        converter.session_g2pw = self._build_onnx_session(model_path)
            except G2PError:
                raise
            except Exception as error:
                raise BackendError("G2PW converter initialization failed") from error
            self._converter = converter
        return self._converter

    def _build_onnx_session(self, model_path: str):
        # Upstream g2pw hardcodes intra_op_num_threads=2 on the session it
        # builds at __init__; a quantized model file is swapped in during
        # that construction, so a session is rebuilt only when onnx_threads
        # asks for a different thread count: 6 threads measured ~10% lower
        # isolated converter p50 on an Apple M5, but the gain vanishes
        # end-to-end through the public API.
        try:
            import onnxruntime
        except ImportError as error:
            raise BackendError("onnxruntime is required for the G2PW backend") from error

        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = self._onnx_threads if self._onnx_threads is not None else 2
        return onnxruntime.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])

    def _convert(self, text: str) -> Sequence[Optional[str]]:
        try:
            values = self._get_converter()(text)
        except G2PError:
            raise
        except Exception as error:
            raise BackendError("G2PW prediction failed") from error

        try:
            if not isinstance(values, SequenceABC) or isinstance(values, (str, bytes)):
                raise BackendError(
                    f"{self.name} converter outer result for projection {text!r} must be a non-string sequence"
                )
            sentence_count = len(values)
            if sentence_count != 1:
                raise BackendError(
                    f"{self.name} converter returned {sentence_count} sentences for one projection {text!r}"
                )
            syllables = values[0]
            if not isinstance(syllables, SequenceABC) or isinstance(syllables, (str, bytes)):
                raise BackendError(
                    f"{self.name} converter sentence result for projection {text!r} must be a non-string sequence"
                )
            corrected = []
            for index, syllable in enumerate(syllables):
                if isinstance(syllable, str) and index < len(text):
                    syllable = _normalize_g2pw_pinyin(text, index, syllable)
                corrected.append(syllable)
            return tuple(corrected)
        except G2PError:
            raise
        except Exception as error:
            raise BackendError(
                f"{self.name} prediction failed while validating converter output for projection {text!r}"
            ) from error

    def predict(
        self,
        request: PronunciationRequest,
    ) -> Mapping[int, Pronunciation]:
        projection = encode_character_projection(request, self._placeholder)
        syllables = self._convert(projection.text)
        if len(syllables) != len(projection.sources):
            raise BackendError(
                f"{self.name} converter returned {len(syllables)} positions for a "
                f"{len(projection.sources)}-character projection {projection.text!r}"
            )

        by_token: Dict[int, Dict[int, str]] = {}
        for position, source in enumerate(projection.sources):
            syllable = syllables[position]
            if syllable is not None and not isinstance(syllable, str):
                raise BackendError(
                    f"{self.name} converter returned a non-string syllable at position {position} "
                    f"for projection {projection.text!r}"
                )
            if source is None:
                continue
            if syllable is None:
                syllable = self._fallback._convert(projection.text[position])[0]
            token_id, char_index = source
            by_token.setdefault(token_id, {})[char_index] = syllable

        result = {}
        for token in request.target_tokens:
            token_syllables = [by_token.get(token.id, {}).get(index) for index in range(len(token.text))]
            if any(syllable is None for syllable in token_syllables):
                raise AlignmentError(f"{self.name} did not predict every character in {token.text!r}")
            result[token.id] = _build_pronunciation(
                token,
                token_syllables,
                backend=self.name,
                strict=self._strict,
                unknown_policy=self._unknown_policy,
            )
        return result
