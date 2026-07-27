import json
from pathlib import Path

import pytest

from g2p_mix.backends.base import PronunciationRequest
from g2p_mix.backends.cantonese import PyCantoneseBackend, ToJyutpingBackend
from g2p_mix.backends.english import EnglishBackend
from g2p_mix.backends.mandarin import G2PWBackend, PypinyinBackend
from g2p_mix.errors import AlignmentError, BackendError
from g2p_mix.models import ChineseDialect, Language, NormalizedText
from g2p_mix.text import ProjectionBuilder, TextAnalyzer

CASE_FILE = Path(__file__).parent / "cases" / "backend_integration.json"
CASE_GROUPS = json.loads(CASE_FILE.read_text(encoding="utf-8"))


def cases(group):
    return [pytest.param(case, id=case["id"]) for case in CASE_GROUPS[group]]


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


def make_request(text, target, dialect=ChineseDialect.MANDARIN):
    tokens = TextAnalyzer(WholeChineseSegmenter()).analyze(NormalizedText.identity(text))
    projection = ProjectionBuilder().build(tokens, target)
    return PronunciationRequest(
        tokens=tokens,
        projection=projection,
        dialect=(dialect if target is Language.CHINESE else None),
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


@pytest.mark.parametrize("case", cases("g2pw_projection"))
def test_g2pw_backend_encodes_foreign_island_once(case):
    class ContextConverter:
        def __init__(self):
            self.calls = []

        def __call__(self, text):
            self.calls.append(text)
            return [case["converter_values"]]

    converter = ContextConverter()
    backend = G2PWBackend(converter=converter)
    result = backend.predict(make_request(case["text"], Language.CHINESE))

    assert len(converter.calls) == 1
    assert converter.calls[0] == case["projected_text"]
    assert {
        str(token_id): [unit.native for unit in pronunciation.units] for token_id, pronunciation in result.items()
    } == case["expected_by_token"]


@pytest.mark.parametrize("case", cases("tojyutping_projection"))
def test_tojyutping_backend_processes_projection_once(case):
    calls = []

    def converter(text):
        calls.append(text)
        return case["converter_values"]

    backend = ToJyutpingBackend(converter=converter)
    result = backend.predict(
        make_request(
            case["text"],
            Language.CHINESE,
            dialect=ChineseDialect.CANTONESE,
        )
    )

    assert calls == [case["projected_text"]]
    assert {
        str(token_id): [unit.native for unit in pronunciation.units] for token_id, pronunciation in result.items()
    } == case["expected_by_token"]
    assert {
        str(token_id): [unit.text for unit in pronunciation.units] for token_id, pronunciation in result.items()
    } == case["expected_unit_text"]
    assert {
        str(token_id): ["".join(span.slice(case["text"]) for span in unit.source_spans) for unit in pronunciation.units]
        for token_id, pronunciation in result.items()
    } == case["expected_unit_text"]


@pytest.mark.parametrize("case", cases("tojyutping_errors"))
def test_tojyutping_backend_validates_converter_output(case):
    error_types = {
        "alignment": AlignmentError,
        "backend": BackendError,
    }
    backend = ToJyutpingBackend(converter=lambda text: case["converter_values"])

    with pytest.raises(error_types[case["error_type"]], match=case["message"]):
        backend.predict(
            make_request(
                case["text"],
                Language.CHINESE,
                dialect=ChineseDialect.CANTONESE,
            )
        )


@pytest.mark.parametrize("case", cases("pycantonese_smoke"))
def test_pycantonese_backend_remains_available(case):
    result = PyCantoneseBackend().predict(
        make_request(
            case["text"],
            Language.CHINESE,
            dialect=ChineseDialect.CANTONESE,
        )
    )

    assert [unit.native for pronunciation in result.values() for unit in pronunciation.units] == case["expected"]


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
