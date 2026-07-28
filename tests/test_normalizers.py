import json
from pathlib import Path

import pytest

from g2p_mix import G2P
from g2p_mix.errors import NormalizationError
from g2p_mix.models import Language, NormalizedText
from g2p_mix.text import (
    AsciiLatinValidator,
    NormalizationPipeline,
    TraditionalChineseNormalizer,
    UnicodeCompatibilityNormalizer,
    WeTextNormalizer,
)

CASE_FILE = Path(__file__).parent / "cases" / "text_normalization.json"
CASE_GROUPS = json.loads(CASE_FILE.read_text(encoding="utf-8"))


def cases(group):
    return [pytest.param(case, id=case["id"]) for case in CASE_GROUPS[group]]


@pytest.mark.parametrize("case", cases("normalization"))
def test_wetext_normalization_preserves_source_alignment(case):
    result = WeTextNormalizer().normalize(NormalizedText.identity(case["text"]))

    assert result.original == case["text"]
    assert result.text == case["normalized"]
    assert [(span.start, span.end) for span in result.char_sources] == [tuple(span) for span in case["source_spans"]]


@pytest.mark.parametrize("case", cases("unicode_compatibility"))
def test_unicode_compatibility_normalization_preserves_source_alignment(case):
    result = UnicodeCompatibilityNormalizer().normalize(NormalizedText.identity(case["text"]))

    assert result.original == case["text"]
    assert result.text == case["normalized"]
    assert [(span.start, span.end) for span in result.char_sources] == [tuple(span) for span in case["source_spans"]]


@pytest.mark.parametrize("case", cases("unsupported_latin"))
def test_unsupported_latin_text_fails_with_its_original_source(case):
    pipeline = NormalizationPipeline(
        (
            UnicodeCompatibilityNormalizer(),
            AsciiLatinValidator(),
        )
    )

    with pytest.raises(NormalizationError) as captured:
        pipeline.normalize(case["text"])

    assert f"character={case['character']!r}" in str(captured.value)
    assert f"source={case['source']!r}" in str(captured.value)
    assert f"span=[{case['span'][0]}, {case['span'][1]})" in str(captured.value)


@pytest.mark.parametrize("case", cases("g2p"))
def test_g2p_pronounces_normalized_numbers(case):
    result = G2P(case["mode"])(case["text"])

    assert result.normalized_text == case["normalized"]
    assert [unit.text for unit in result.units] == case["unit_text"]
    if "native" in case:
        assert [unit.native for unit in result.units] == case["native"]
    assert result.phones
    assert all(output.token.language is not Language.NUMBER for output in result.tokens)


class BrokenNormalizer:
    def normalize(self, text):
        raise RuntimeError("broken")


def test_wetext_failure_uses_the_normalization_error_boundary():
    normalizer = WeTextNormalizer(normalizer=BrokenNormalizer())

    with pytest.raises(NormalizationError, match="WeText normalization failed"):
        normalizer.normalize(NormalizedText.identity("3"))


@pytest.mark.parametrize("case", cases("traditional_exception_boundaries"))
def test_traditional_chinese_failure_uses_the_normalization_error_boundary(case):
    class Converter:
        def convert(self, text):
            if case["id"] == "runtime-failure":
                raise RuntimeError("dependency failed")
            if case["id"] == "non-string-result":
                return None
            return text + "話"

    normalizer = TraditionalChineseNormalizer(converter=Converter())

    with pytest.raises(NormalizationError, match=case["message"]):
        normalizer.normalize(NormalizedText.identity(case["text"]))
