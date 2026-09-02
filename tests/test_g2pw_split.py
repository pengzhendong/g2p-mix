"""Unit tests for SplitG2PWConverter using fake base metadata and sessions.

These tests avoid the optional g2pw/onnxruntime dependencies: the token
mapper is injected and both ONNX sessions are faked, so they run in the
default test environment.
"""

import numpy as np
import pytest

from g2p_mix.backends.g2pw_split import SplitG2PWConverter
from g2p_mix.errors import BackendError


def _one_token_per_char(tokenizer, text):
    if "�" in text:
        raise ValueError("invalid text")
    tokens = list(text)
    text2token = list(range(len(text)))
    token2text = [(index, index + 1) for index in range(len(text))]
    return tokens, text2token, token2text


class FakeTokenizer:
    def convert_tokens_to_ids(self, tokens):
        special = {"[CLS]": 101, "[SEP]": 102}
        return [special[token] if token in special else ord(token) for token in tokens]


class FakeBase:
    chars = ["行", "重", "长"]
    labels = ["chang2", "chong2", "hang2", "xing2", "zhang3", "zhong4"]
    char2phonemes = {
        "行": [labels.index("hang2"), labels.index("xing2")],
        "重": [labels.index("chong2"), labels.index("zhong4")],
        "长": [labels.index("chang2"), labels.index("zhang3")],
    }
    monophonic_chars = [("你", "ni3"), ("好", "hao3")]
    char_bopomofo_dict = {"的": ["de5"]}
    tokenizer = FakeTokenizer()
    enable_non_tradional_chinese = False

    @staticmethod
    def style_convert_func(label):
        return f"converted:{label}"


class FakeEncoder:
    def __init__(self, width=4):
        self.calls = []
        self._width = width

    def run(self, _outputs, feeds):
        self.calls.append(feeds)
        batch, length = feeds["input_ids"].shape
        return [np.zeros((batch, length, self._width), dtype=np.float32)]


class FakeHead:
    def __init__(self, labels_count, decide):
        self.calls = []
        self._labels_count = labels_count
        self._decide = decide

    def run(self, _outputs, feeds):
        self.calls.append(feeds)
        count = len(feeds["position_ids"])
        probs = np.zeros((count, self._labels_count), dtype=np.float32)
        for row in range(count):
            chosen = self._decide(int(feeds["position_ids"][row]), int(feeds["char_ids"][row]))
            probs[row, chosen] = 1.0
        return [probs]


def _make_converter(decide):
    base = FakeBase()
    encoder = FakeEncoder()
    head = FakeHead(len(base.labels), decide)
    converter = SplitG2PWConverter(base, encoder, head, mapper=_one_token_per_char)
    return converter, base, encoder, head


def test_monophonic_and_bopomofo_chars_skip_the_model():
    converter, _, encoder, head = _make_converter(lambda _pos, _char: 0)
    assert converter(["你好，的"])[0] == ["converted:ni3", "converted:hao3", None, "converted:de5"]
    assert encoder.calls == []
    assert head.calls == []


def test_polyphonic_positions_share_one_encoder_call():
    label = FakeBase.labels.index("hang2")
    converter, base, encoder, head = _make_converter(lambda _pos, _char: label)
    (result,) = converter(["银行"])
    assert result == [None, "converted:hang2"]
    assert len(encoder.calls) == 1
    assert encoder.calls[0]["input_ids"].shape == (1, 4)  # [CLS] 银 行 [SEP]
    assert len(head.calls) == 1
    feeds = head.calls[0]
    # The query char 行 sits at text index 1 -> token position 1 + [CLS] offset.
    assert feeds["position_ids"].tolist() == [2]
    assert feeds["char_ids"].tolist() == [base.chars.index("行")]
    # Encoder output is repeated once per polyphonic position.
    assert feeds["1663"].shape == (1, 4, 4)
    mask = feeds["phoneme_mask"][0].tolist()
    assert mask == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]


def test_multiple_polyphonics_repeat_hidden_states():
    label = FakeBase.labels.index("zhang3")
    converter, _, encoder, head = _make_converter(lambda _pos, _char: label)
    (result,) = converter(["长长长"])
    assert result == ["converted:zhang3"] * 3
    assert len(encoder.calls) == 1
    feeds = head.calls[0]
    assert feeds["1663"].shape == (3, 5, 4)
    assert feeds["input_ids"].shape == (3, 5)
    assert feeds["position_ids"].tolist() == [1, 2, 3]


def test_long_sentences_fall_back_to_batched_windows():
    label = FakeBase.labels.index("zhang3")
    converter, _, encoder, head = _make_converter(lambda _pos, _char: label)
    sentence = "长" * 520  # one token per char -> 520 tokens > limit
    (result,) = converter([sentence])
    assert result == ["converted:zhang3"] * len(sentence)
    assert len(encoder.calls) == 1
    feeds = encoder.calls[0]
    # Every window spans at most window_size(32) tokens plus [CLS]/[SEP].
    assert feeds["input_ids"].shape == (520, 34)
    head_feeds = head.calls[0]
    assert head_feeds["1663"].shape == (520, 34, 4)
    assert min(head_feeds["position_ids"].tolist()) >= 1
    assert max(head_feeds["position_ids"].tolist()) <= 33


def test_tokenize_failure_leaves_positions_to_the_fallback():
    label = FakeBase.labels.index("hang2")
    converter, _, encoder, head = _make_converter(lambda _pos, _char: label)
    (result,) = converter(["行�行"])
    assert result == [None, None, None]
    assert encoder.calls == []
    assert head.calls == []


def test_from_model_dir_requires_optional_dependencies(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", None)
    with pytest.raises(BackendError):
        SplitG2PWConverter.from_model_dir("unused", model_source="unused")


def test_redirect_g2pw_session(monkeypatch):
    import sys
    import types

    calls = []

    class FakeSession:
        def __init__(self, path, **kwargs):
            calls.append((path, kwargs))

    fake_ort = types.SimpleNamespace(InferenceSession=FakeSession)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    from g2p_mix.backends.g2pw_split import _redirect_g2pw_session

    with _redirect_g2pw_session("/tmp/int8.onnx"):
        fake_ort.InferenceSession("/model/g2pw.onnx", sess_options=1)
    assert calls == [("/tmp/int8.onnx", {"sess_options": 1})]

    fake_ort.InferenceSession("/model/g2pw.onnx")
    assert calls[-1] == ("/model/g2pw.onnx", {})  # restored after the context

    with _redirect_g2pw_session(None):
        assert fake_ort.InferenceSession("/model/g2pw.onnx") is None
