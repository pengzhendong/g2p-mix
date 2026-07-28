import json
from pathlib import Path

import pytest

from g2p_mix.backends.base import FallbackBackend, PronunciationRequest
from g2p_mix.backends.cantonese import PyCantoneseBackend, ToJyutpingBackend
from g2p_mix.backends.english import (
    CmuLexicon,
    EnglishBackend,
    G2pEnOovPredictor,
    HomographRule,
    PosHomographResolver,
)
from g2p_mix.backends.mandarin import G2PWBackend, PypinyinBackend
from g2p_mix.errors import AlignmentError, BackendError
from g2p_mix.models import (
    ChineseDialect,
    Language,
    NormalizedText,
    UnknownPolicy,
)
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


@pytest.mark.parametrize("case", cases("pinyin_errors"))
def test_pypinyin_backend_surfaces_invalid_syllables_with_context(case):
    converter = FakePinyinConverter({case["text"]: case["converter_value"]})

    with pytest.raises(BackendError, match=case["message"]):
        PypinyinBackend(converter=converter).predict(make_request(case["text"], Language.CHINESE))


@pytest.mark.parametrize("case", cases("backend_exception_boundaries"))
def test_builtin_backends_wrap_dependency_runtime_errors(case, monkeypatch):
    def fail(_text, **_kwargs):
        raise RuntimeError("dependency failed")

    if case["backend"] == "pypinyin":

        class Converter:
            convert = staticmethod(fail)

        backend = PypinyinBackend(converter=Converter())
        dialect = ChineseDialect.MANDARIN
    elif case["backend"] == "tojyutping":
        backend = ToJyutpingBackend(converter=fail)
        dialect = ChineseDialect.CANTONESE
    else:
        import pycantonese

        monkeypatch.setattr(pycantonese, "characters_to_jyutping", fail)
        backend = PyCantoneseBackend()
        dialect = ChineseDialect.CANTONESE

    with pytest.raises(BackendError, match=case["message"]) as captured:
        backend.predict(make_request("你", Language.CHINESE, dialect=dialect))

    assert type(captured.value.__cause__) is RuntimeError


@pytest.mark.parametrize("case", cases("pinyin_unknown"))
def test_pypinyin_backend_can_preserve_unregistered_characters(case):
    converter = FakePinyinConverter(case["converter_values"])
    backend = PypinyinBackend(
        converter=converter,
        unknown_policy=UnknownPolicy.PRESERVE,
    )

    result = backend.predict(make_request(case["text"], Language.CHINESE))
    units = result[0].units

    assert [unit.native for unit in units] == case["expected_native"]
    assert [unit.is_unknown for unit in units] == case["expected_unknown"]


@pytest.mark.parametrize("case", cases("pinyin_unknown"))
def test_g2pw_backend_preserves_a_character_missing_from_model_and_pypinyin(case):
    class MissingConverter:
        def __call__(self, text):
            return [[None] * len(text)]

    backend = G2PWBackend(
        converter=MissingConverter(),
        unknown_policy=UnknownPolicy.PRESERVE,
    )

    result = backend.predict(make_request(case["text"][-1], Language.CHINESE))
    unit = result[0].units[0]

    assert unit.is_unknown is True
    assert unit.phones == ()


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


@pytest.mark.parametrize("case", cases("tojyutping_real_supplementary"))
def test_tojyutping_backend_supports_supplementary_han_inventory(case):
    result = ToJyutpingBackend().predict(
        make_request(
            case["text"],
            Language.CHINESE,
            dialect=ChineseDialect.CANTONESE,
        )
    )
    units = [unit for pronunciation in result.values() for unit in pronunciation.units]

    assert [unit.native for unit in units] == case["expected_native"]
    assert [list(unit.phones) for unit in units] == case["expected_phones"]
    assert [unit.tone for unit in units] == case["expected_tones"]


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


@pytest.mark.parametrize("case", cases("pycantonese_errors"))
def test_pycantonese_backend_rejects_partially_parsed_raw_output(case, monkeypatch):
    import pycantonese

    monkeypatch.setattr(
        pycantonese,
        "characters_to_jyutping",
        lambda text: [(case["source_chars"], case["raw_value"])],
    )

    with pytest.raises(BackendError) as captured:
        PyCantoneseBackend().predict(
            make_request(
                case["text"],
                Language.CHINESE,
                dialect=ChineseDialect.CANTONESE,
            )
        )

    message = str(captured.value)
    assert all(fragment in message for fragment in case["message_fragments"])
    assert type(captured.value.__cause__).__name__ == case["cause_type"]


@pytest.mark.parametrize("case", cases("cantonese_unknown"))
def test_cantonese_backends_can_preserve_missing_characters(case, monkeypatch):
    if case["backend"] == "tojyutping":
        backend = ToJyutpingBackend(
            converter=lambda _text: case["converter_values"],
            unknown_policy=UnknownPolicy.PRESERVE,
        )
    else:
        import pycantonese

        monkeypatch.setattr(
            pycantonese,
            "characters_to_jyutping",
            lambda _text: case["converter_values"],
        )
        backend = PyCantoneseBackend(
            unknown_policy=UnknownPolicy.PRESERVE,
        )

    result = backend.predict(
        make_request(
            case["text"],
            Language.CHINESE,
            dialect=ChineseDialect.CANTONESE,
        )
    )
    units = result[0].units

    assert [unit.native for unit in units] == case["expected_native"]
    assert [unit.is_unknown for unit in units] == case["expected_unknown"]


def test_fallback_backend_uses_compatible_secondary_backend():
    class FailingBackend(PypinyinBackend):
        name = "failing"

        def predict(self, request):
            raise BackendError("primary failed")

    fallback = PypinyinBackend(converter=FakePinyinConverter({"你": "ni3"}))
    backend = FallbackBackend(FailingBackend(), fallback)

    result = backend.predict(make_request("你", Language.CHINESE))

    assert result[0].backend == "pypinyin"
    assert result[0].units[0].native == "ni3"


def test_english_backend_decision_tree_is_dependency_injectable():
    dictionary = {
        "a": [["AH0"], ["EY1"]],
        "i": [["AY1"]],
        "idea": [["AY0", "D", "IY1", "AH0"]],
    }
    backend = EnglishBackend(
        lexicon=CmuLexicon(dictionary),
        segmenter=lambda word: [word],
        oov_predictor=G2pEnOovPredictor(lambda word: ["T", "EH1", "S", "T"]),
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
    dictionary = {word: [["T", "EH1", "S", "T"]] for word in words}
    backend = EnglishBackend(
        lexicon=CmuLexicon(dictionary),
        segmenter=lambda word: [word],
        oov_predictor=G2pEnOovPredictor(lambda word: ["OOV"]),
    )

    request = make_request(sentence, Language.ENGLISH)
    result = backend.predict(request)
    clitic = next(token for token in request.target_tokens if token.text == "n't")

    assert result[clitic.id].units[0].phones == ("AH", "N", "T")
    assert result[clitic.id].units[0].stress_marks == ((0, 0),)
    assert result[clitic.id].units[0].native == "AH0 N T"
    assert backend.convert("n’t") == ["AH0", "N", "T"]


@pytest.mark.parametrize("case", cases("english_context_projection"))
def test_english_backend_analyzes_shared_context_once(case):
    class RecordingAnalyzer:
        def __init__(self):
            self.calls = []

        def analyze(self, request):
            self.calls.append(tuple(token.text for token in request.projection.tokens if not token.text.isspace()))
            return {
                token.id: (case["target_pos"] if token.text == case["target"] else "NN")
                for token in request.target_tokens
            }

    analyzer = RecordingAnalyzer()
    resolver = PosHomographResolver(
        {
            case["target"]: HomographRule(
                matching_pronunciation=tuple(case["matching_pronunciation"]),
                other_pronunciation=tuple(case["other_pronunciation"]),
                pos_prefix="V",
            )
        }
    )
    backend = EnglishBackend(
        lexicon=CmuLexicon(
            {
                "i": [["AY1"]],
                case["target"]: [case["other_pronunciation"]],
                "music": [["M", "Y", "UW1", "Z", "IH0", "K"]],
            }
        ),
        context_analyzer=analyzer,
        resolver=resolver,
    )

    request = make_request(case["text"], Language.ENGLISH)
    result = backend.predict(request)

    assert analyzer.calls == [tuple(case["context_tokens"])]
    target = next(token for token in request.target_tokens if token.text == case["target"])
    assert result[target.id].units[0].native == case["expected_native"]


@pytest.mark.parametrize("case", cases("english_context_fast_path"))
def test_english_backend_skips_context_analysis_for_unambiguous_words(case):
    class FailingAnalyzer:
        def analyze(self, request):
            raise AssertionError("POS analysis should not run")

    request = make_request(case["text"], Language.ENGLISH)
    result = EnglishBackend(context_analyzer=FailingAnalyzer()).predict(request)

    assert [result[token.id].units[0].native for token in request.target_tokens] == case["expected_native"]


@pytest.mark.parametrize("case", cases("english_homographs"))
def test_english_backend_resolves_real_pos_homographs(case):
    request = make_request(case["text"], Language.ENGLISH)
    result = EnglishBackend().predict(request)
    target = next(token for token in request.target_tokens if token.text.lower() == case["target"])

    assert result[target.id].units[0].native == case["expected_native"]
