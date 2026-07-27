import pytest

from g2p_mix.backends.base import PronunciationRequest
from g2p_mix.backends.english import EnglishBackend
from g2p_mix.backends.mandarin import G2PWBackend, PypinyinBackend
from g2p_mix.errors import AlignmentError
from g2p_mix.models import ChineseDialect, Language, NormalizedText
from g2p_mix.text import ProjectionBuilder, TextAnalyzer


class WholeChineseSegmenter:
    def segment(self, text):
        return [(text, "x")]


class FakePinyinConverter:
    def __init__(self, pronunciations):
        self.pronunciations = pronunciations
        self.calls = []

    def convert(self, text, **kwargs):
        self.calls.append(text)
        return [[self.pronunciations.get(char, char)] for char in text]


def make_request(text, target):
    tokens = TextAnalyzer(WholeChineseSegmenter()).analyze(NormalizedText.identity(text))
    projection = ProjectionBuilder().build(tokens, target)
    return PronunciationRequest(
        tokens=tokens,
        projection=projection,
        dialect=(ChineseDialect.MANDARIN if target is Language.CHINESE else None),
    )


def test_pypinyin_backend_validates_character_alignment():
    converter = FakePinyinConverter({"你": "ni3", "好": "hao3"})
    backend = PypinyinBackend(converter=converter)
    result = backend.predict(make_request("你好", Language.CHINESE))

    assert converter.calls == ["你好"]
    assert [unit.native for unit in result[0].units] == ["ni3", "hao3"]
    assert [unit.tone for unit in result[0].units] == ["3", "3"]

    class BrokenConverter:
        def convert(self, text, **kwargs):
            return [["ni3"]]

    with pytest.raises(AlignmentError):
        PypinyinBackend(converter=BrokenConverter()).predict(make_request("你好", Language.CHINESE))


def test_g2pw_backend_encodes_foreign_island_once():
    pronunciations = {
        "银": "yin2",
        "行": "hang2",
        "不": "bu4",
    }

    class ContextConverter(FakePinyinConverter):
        def convert(self, text, **kwargs):
            self.calls.append(text)
            values = []
            for index, char in enumerate(text):
                if char == "行" and index > text.index("\ue000"):
                    value = "xing2"
                else:
                    value = pronunciations.get(char, char)
                values.append([value])
            return values

    converter = ContextConverter(pronunciations)
    backend = G2PWBackend(converter=converter)
    result = backend.predict(make_request("银行 ATM 行不行", Language.CHINESE))

    assert len(converter.calls) == 1
    assert converter.calls[0] == "银行\ue000行不行"
    assert "ATM" not in converter.calls[0]
    assert [unit.native for unit in result[0].units] == ["yin2", "hang2"]
    assert [unit.native for unit in result[4].units] == [
        "xing2",
        "bu4",
        "xing2",
    ]


def test_english_backend_decision_tree_is_dependency_injectable():
    dictionary = {
        "a": [["AH0"], ["EY1"]],
        "i": [["AY1"]],
        "idea": [["AY0", "D", "IY1", "AH0"]],
    }
    backend = EnglishBackend(
        dictionary=dictionary,
        segmenter=lambda word: [word],
        predictor=lambda word: ["T", "EH1", "S", "T"],
    )

    assert backend.convert("AI") == ["EY1", "AY1"]
    assert backend.convert("idea") == ["AY0", "D", "IY1", "AH0"]
    assert backend.convert("testing") == ["T", "EH1", "S", "T"]


def test_english_backend_handles_standalone_negative_clitics():
    sentence = (
        "He was as thick as my leg, and looked as if millstones could n't crush the disgusting vitality out of him."
    )
    words = (
        "he",
        "was",
        "as",
        "thick",
        "my",
        "leg",
        "and",
        "looked",
        "if",
        "millstones",
        "could",
        "crush",
        "the",
        "disgusting",
        "vitality",
        "out",
        "of",
        "him",
    )
    dictionary = {word: [[word.upper()]] for word in words}
    backend = EnglishBackend(
        dictionary=dictionary,
        segmenter=lambda word: [word],
        predictor=lambda word: ["OOV"],
    )

    request = make_request(sentence, Language.ENGLISH)
    result = backend.predict(request)
    clitic = next(token for token in request.target_tokens if token.text == "n't")

    assert result[clitic.id].units[0].phones == ("AH0", "N", "T")
    assert backend.convert("n’t") == ["AH0", "N", "T"]
