import pytest

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


def make_token(
    token_id,
    text,
    tones,
    *,
    start=None,
    pos="x",
    language=Language.CHINESE,
):
    start = token_id if start is None else start
    token = TextToken(
        id=token_id,
        text=text,
        normalized_span=Span(start, start + len(text)),
        source_spans=tuple(Span(start + index, start + index + 1) for index in range(len(text))),
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
    return token, Pronunciation(token_id, units, "test")


def make_tokens(parts):
    items = []
    cursor = 0
    for token_id, (text, tones, pos) in enumerate(parts):
        items.append(make_token(token_id, text, tones, start=cursor, pos=pos))
        cursor += len(text)
    return items


def process(*items, word_splits=None):
    splits = {} if word_splits is None else word_splits

    def split_word(word):
        return splits.get(word, (word,))

    tokens = tuple(token for token, _ in items)
    pronunciations = {token.id: pronunciation for token, pronunciation in items if pronunciation is not None}
    result = MandarinToneSandhi(word_splitter=split_word).process(tokens, pronunciations)
    return {token_id: tuple(unit.tone for unit in pronunciation.units) for token_id, pronunciation in result.items()}


@pytest.mark.parametrize(
    ("text", "tones", "expected"),
    [
        pytest.param("看不懂", ("4", "4", "3"), ("4", "5", "3"), id="kan-bu-dong"),
        pytest.param("去不去", ("4", "4", "4"), ("4", "5", "4"), id="qu-bu-qu"),
        pytest.param("不怕", ("4", "4"), ("2", "4"), id="bu-before-fourth-tone"),
    ],
)
def test_bu_sandhi(text, tones, expected):
    assert process(make_token(0, text, tones))[0] == expected


@pytest.mark.parametrize(
    ("text", "tones", "pos", "expected"),
    [
        pytest.param("看一看", ("4", "1", "4"), "v", ("4", "5", "4"), id="verb-reduplication"),
        pytest.param("一月", ("1", "4"), "m", ("1", "4"), id="month"),
        pytest.param("一段", ("1", "4"), "m", ("2", "4"), id="before-fourth-tone"),
        pytest.param("一天", ("1", "1"), "m", ("4", "1"), id="before-non-fourth-tone"),
        pytest.param("第一", ("4", "1"), "m", ("4", "1"), id="ordinal"),
        pytest.param("一二三", ("1", "4", "1"), "m", ("1", "4", "1"), id="digits"),
    ],
)
def test_yi_sandhi(text, tones, pos, expected):
    assert process(make_token(0, text, tones, pos=pos))[0] == expected


@pytest.mark.parametrize(
    ("text", "tones", "pos", "split", "expected"),
    [
        pytest.param("我们", ("3", "2"), "r", ("我们",), ("3", "5"), id="plural-men"),
        pytest.param("花儿", ("1", "2"), "n", ("花儿",), ("1", "5"), id="erhua"),
        pytest.param("女儿", ("3", "2"), "n", ("女儿",), ("3", "2"), id="erhua-whitelist"),
        pytest.param("好吧", ("3", "1"), "x", ("好吧",), ("3", "5"), id="interjection"),
        pytest.param("看看", ("4", "4"), "v", ("看看",), ("4", "5"), id="verb-reduplication"),
        pytest.param("看了", ("4", "4"), "v", ("看", "了"), ("4", "5"), id="aspect-particle"),
        pytest.param(
            "很漂亮",
            ("3", "4", "4"),
            "a",
            ("很", "漂亮"),
            ("3", "4", "5"),
            id="lexical-neutral-tone",
        ),
    ],
)
def test_neutral_tone_sandhi(text, tones, pos, split, expected):
    assert (
        process(
            make_token(0, text, tones, pos=pos),
            word_splits={text: split},
        )[0]
        == expected
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("很漂亮", ("很", "漂亮"), id="hen-piaoliang"),
        pytest.param("与好姐妹", ("与", "好姐妹"), id="yu-haojiemei"),
    ],
)
def test_default_word_splitter_preserves_sandhi_subwords(text, expected):
    assert tuple(MandarinToneSandhi._split_word(text)) == expected


@pytest.mark.parametrize(
    ("text", "tones", "split", "expected"),
    [
        pytest.param("你好", ("3", "3"), ("你好",), ("2", "3"), id="two-syllable"),
        pytest.param("小火车", ("3", "3", "1"), ("小", "火车"), ("2", "3", "1"), id="one-plus-two"),
        pytest.param("也很好", ("3", "3", "3"), ("也", "很好"), ("3", "2", "3"), id="three-thirds"),
        pytest.param("纸老虎", ("3", "3", "3"), ("纸", "老虎"), ("3", "2", "3"), id="zhi-laohu"),
        pytest.param(
            "与好姐妹",
            ("3", "3", "3", "4"),
            ("与", "好姐妹"),
            ("3", "2", "3", "4"),
            id="yu-haojiemei",
        ),
        pytest.param("好喜欢", ("3", "3", "1"), ("好", "喜欢"), ("2", "3", "5"), id="hao-xihuan"),
        pytest.param("广场舞", ("3", "3", "3"), ("广场", "舞"), ("2", "2", "3"), id="guangchang-wu"),
        pytest.param("字母表", ("4", "3", "3"), ("字母", "表"), ("4", "2", "3"), id="zimu-biao"),
        pytest.param("所有人", ("3", "3", "2"), ("所有", "人"), ("2", "3", "2"), id="suoyou-ren"),
        pytest.param(
            "坎坎坷坷",
            ("3", "3", "3", "3"),
            ("坎坎", "坷坷"),
            ("2", "3", "2", "3"),
            id="kankan-keke",
        ),
        pytest.param(
            "期待已久",
            ("1", "4", "3", "3"),
            ("期待", "已久"),
            ("1", "4", "2", "3"),
            id="qidai-yijiu",
        ),
        pytest.param(
            "省委党校",
            ("3", "3", "3", "4"),
            ("省委", "党校"),
            ("2", "2", "3", "4"),
            id="shengwei-dangxiao",
        ),
    ],
)
def test_third_tone_sandhi(text, tones, split, expected):
    assert (
        process(
            make_token(0, text, tones),
            word_splits={text: split},
        )[0]
        == expected
    )


@pytest.mark.parametrize(
    ("parts", "word_splits", "expected"),
    [
        pytest.param(
            (("不", ("4",), "d"), ("怕", ("4",), "v")),
            {},
            {0: ("2",), 1: ("4",)},
            id="bu-plus-following-word",
        ),
        pytest.param(
            (("你", ("3",), "r"), ("好", ("3",), "a")),
            {},
            {0: ("2",), 1: ("3",)},
            id="third-tones",
        ),
        pytest.param(
            (("听", ("1",), "v"), ("一", ("1",), "m"), ("听", ("1",), "v")),
            {},
            {0: ("1",), 1: ("5",), 2: ("1",)},
            id="verb-yi-verb",
        ),
        pytest.param(
            (("小", ("3",), "a"), ("火车", ("3", "1"), "n")),
            {"小火车": ("小", "火车")},
            {0: ("2",), 1: ("3", "1")},
            id="merge-one-plus-two",
        ),
        pytest.param(
            (("也", ("3",), "d"), ("很", ("3",), "d"), ("好", ("3",), "a")),
            {"也很好": ("也", "很好")},
            {0: ("3",), 1: ("2",), 2: ("3",)},
            id="merge-three-thirds",
        ),
    ],
)
def test_sandhi_crosses_adjacent_chinese_tokens(parts, word_splits, expected):
    assert process(*make_tokens(parts), word_splits=word_splits) == expected


def test_sandhi_does_not_cross_an_english_island():
    bu = make_token(0, "不", ("4",), pos="d")
    english = make_token(
        1,
        "AI",
        (),
        start=1,
        language=Language.ENGLISH,
    )
    pa = make_token(2, "怕", ("4",), start=3, pos="v")

    assert process(bu, english, pa)[0] == ("4",)
