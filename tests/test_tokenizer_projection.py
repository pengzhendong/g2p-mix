import json
from pathlib import Path

import pytest

from g2p_mix.lexicons import MandarinLexicon
from g2p_mix.models import Language, NormalizedText, ProjectionKind
from g2p_mix.text import LosslessTokenizer, ProjectionBuilder, TextAnalyzer

CASE_FILE = Path(__file__).parent / "cases" / "unicode_tokenization.json"
CASE_GROUPS = json.loads(CASE_FILE.read_text(encoding="utf-8"))


def cases(group):
    return [pytest.param(case, id=case["id"]) for case in CASE_GROUPS[group]]


class WholeChineseSegmenter:
    def segment(self, text):
        return [(text, "x")]


def analyze(text):
    return TextAnalyzer(WholeChineseSegmenter()).analyze(NormalizedText.identity(text))


def test_analysis_is_lossless_for_unicode_and_mixed_text():
    text = "café_测试𠀀🙂 rock'n'roll"
    tokens = analyze(text)

    assert "".join(token.text for token in tokens) == text
    assert [token.text for token in tokens] == [
        "café",
        "_",
        "测试𠀀",
        "🙂",
        " ",
        "rock'n'roll",
    ]
    assert [token.language for token in tokens] == [
        Language.ENGLISH,
        Language.SYMBOL,
        Language.CHINESE,
        Language.SYMBOL,
        Language.SPACE,
        Language.ENGLISH,
    ]
    assert tuple(span.slice(text) for span in tokens[2].source_spans) == (
        "测",
        "试",
        "𠀀",
    )


def test_projection_collapses_language_islands_but_preserves_punctuation():
    tokens = analyze("这个 make sense, idea 不错")
    builder = ProjectionBuilder()

    chinese = builder.build(tokens, Language.CHINESE)
    english = builder.build(tokens, Language.ENGLISH)

    assert chinese.text == "这个 <EN>, <EN> 不错"
    assert english.text == "<ZH> make sense, idea <ZH>"
    placeholders = [token for token in chinese.tokens if token.kind is ProjectionKind.PLACEHOLDER]
    assert len(placeholders) == 2
    assert len(placeholders[0].source_ids) == 3  # make + space + sense


def test_projection_never_creates_false_chinese_adjacency():
    tokens = analyze("银行 ATM 行不行")
    chinese = ProjectionBuilder().build(tokens, Language.CHINESE)

    assert chinese.text == "银行 <EN> 行不行"
    assert "银行行不行" not in chinese.text


@pytest.mark.parametrize("case", cases("tokenization"))
def test_unicode_script_tokenization_is_lossless_and_aligned(case):
    value = NormalizedText.identity(case["text"])
    tokens = LosslessTokenizer().scan(value)

    assert [
        {
            "text": token.text,
            "language": token.language.value,
            "start": token.normalized_span.start,
            "end": token.normalized_span.end,
        }
        for token in tokens
    ] == case["tokens"]
    assert "".join(token.text for token in tokens) == case["text"]
    assert all("".join(span.slice(case["text"]) for span in token.source_spans) == token.text for token in tokens)


@pytest.mark.parametrize("case", cases("latin_codepoints"))
def test_latin_classification_uses_versioned_script_data(case):
    char = chr(int(case["codepoint"], 16))
    token = LosslessTokenizer().scan(NormalizedText.identity(char))

    assert len(token) == 1
    assert token[0].language.value == case["language"]


@pytest.mark.parametrize("case", cases("han_ranges"))
def test_tokenizer_and_lexicon_share_current_han_ranges(case):
    boundaries = (chr(int(case["start"], 16)), chr(int(case["end"], 16)))
    tokenizer = LosslessTokenizer()
    lexicon = MandarinLexicon(lookup=lambda char: ("zi4",))

    for char in boundaries:
        token = tokenizer.scan(NormalizedText.identity(char))
        assert len(token) == 1
        assert token[0].language is Language.CHINESE
        assert lexicon.pronunciations(char) == ("zi4",)

    for key in ("negative_before", "negative_after"):
        if key not in case:
            continue
        char = chr(int(case[key], 16))
        token = tokenizer.scan(NormalizedText.identity(char))
        assert len(token) == 1
        assert token[0].language is Language.SYMBOL
        with pytest.raises(ValueError, match="exactly one Han character"):
            lexicon.pronunciations(char)
