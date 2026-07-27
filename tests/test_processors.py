from g2p_mix.models import (
    Language,
    PhoneAlphabet,
    Pronunciation,
    PronunciationUnit,
    Span,
    TextToken,
    TokenKind,
)
from g2p_mix.processors import MandarinToneSandhi


def make_token(token_id, text, tones, *, pos="x", language=Language.CHINESE):
    token = TextToken(
        id=token_id,
        text=text,
        normalized_span=Span(token_id, token_id + len(text)),
        source_spans=tuple(Span(token_id + index, token_id + index + 1) for index in range(len(text))),
        language=language,
        kind=(TokenKind.HAN if language is Language.CHINESE else TokenKind.LATIN),
        pos=pos,
    )
    if language is not Language.CHINESE:
        return token, None

    units = tuple(
        PronunciationUnit(
            text=char,
            source_spans=(token.source_spans[index],),
            phones=(char,),
            tone=tone,
            alphabet=PhoneAlphabet.PINYIN,
            native=char + tone,
        )
        for index, (char, tone) in enumerate(zip(text, tones))
    )
    return token, Pronunciation(token_id, units, "fake")


def process(*items):
    tokens = tuple(token for token, _ in items)
    pronunciations = {token.id: pronunciation for token, pronunciation in items if pronunciation is not None}
    result = MandarinToneSandhi().process(tokens, pronunciations)
    return {token_id: [unit.tone for unit in pronunciation.units] for token_id, pronunciation in result.items()}


def test_bu_and_third_tone_rules_cross_adjacent_chinese_tokens():
    bu = make_token(0, "不", ["4"], pos="d")
    pa = make_token(1, "怕", ["4"], pos="v")
    ni = make_token(2, "你", ["3"], pos="r")
    hao = make_token(3, "好", ["3"], pos="a")

    result = process(bu, pa, ni, hao)

    assert result[0] == ["2"]
    assert result[2] == ["2"]


def test_sandhi_does_not_cross_an_english_island():
    bu = make_token(0, "不", ["4"], pos="d")
    english = make_token(
        1,
        "AI",
        [],
        language=Language.ENGLISH,
    )
    pa = make_token(2, "怕", ["4"], pos="v")

    assert process(bu, english, pa)[0] == ["4"]


def test_numeric_yi_and_neutral_suffix_regressions():
    numeric = make_token(0, "一二三", ["1", "4", "1"], pos="m")
    plural = make_token(3, "我们", ["3", "2"], pos="r")
    reduplication = make_token(5, "看看", ["4", "4"], pos="v")

    result = process(numeric, plural, reduplication)

    assert result[0][0] == "1"
    assert result[3][-1] == "5"
    assert result[5][-1] == "5"
