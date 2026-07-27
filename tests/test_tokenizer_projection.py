from g2p_mix.models import Language, NormalizedText, ProjectionKind
from g2p_mix.projection import ProjectionBuilder
from g2p_mix.tokenizer import TextAnalyzer


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
