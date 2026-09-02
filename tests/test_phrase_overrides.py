import pytest

from g2p_mix import G2P
from g2p_mix.backends.mandarin import G2PWBackend
from g2p_mix.models import (
    Language,
    PhoneAlphabet,
    Pronunciation,
    PronunciationUnit,
    Span,
    TextToken,
    TokenKind,
)
from g2p_mix.phonetics import split_pinyin
from g2p_mix.processors import MandarinPhraseOverrides
from g2p_mix.resources import load_pinyin_phrases


def make_token(token_id, text, natives, *, start, language=Language.CHINESE):
    token = TextToken(
        id=token_id,
        text=text,
        normalized_span=Span(start, start + len(text)),
        source_spans=tuple(Span(start + index, start + index + 1) for index in range(len(text))),
        language=language,
        kind=(TokenKind.HAN if language is Language.CHINESE else TokenKind.PUNCTUATION),
    )
    if language is not Language.CHINESE:
        return token, None
    units = tuple(
        PronunciationUnit(
            text=char,
            source_spans=(token.source_spans[index],),
            phones=(native,),
            tone=native[-1],
            alphabet=PhoneAlphabet.PINYIN,
            native=native,
        )
        for index, (char, native) in enumerate(zip(text, natives))
    )
    return token, Pronunciation(token_id, units, "test")


def run_processor(text_runs, phrases):
    """Build tokens from a chunked segmentation and run the processor on them."""
    tokens = []
    pronunciations = {}
    cursor = 0
    for language, chunks, natives in text_runs:
        for chunk in chunks:
            token, pronunciation = make_token(
                len(tokens),
                chunk,
                natives[: len(chunk)] if language is Language.CHINESE else (),
                start=cursor,
                language=language,
            )
            if language is Language.CHINESE:
                natives = natives[len(chunk) :]
                pronunciations[token.id] = pronunciation
            tokens.append(token)
            cursor += len(chunk)
    processor = MandarinPhraseOverrides(phrases)
    result = processor.process(tuple(tokens), pronunciations)
    ordered = []
    for token in tokens:
        pronunciation = result.get(token.id)
        if pronunciation is None:
            continue
        ordered.extend(unit.native for unit in pronunciation.units)
    return ordered


def test_split_phrase_is_still_overridden():
    # jieba may cut "出壳" apart (e.g. "散出" + "壳"); the override must still win.
    ordered = run_processor(
        [(Language.CHINESE, ("散", "出", "壳"), ["san4", "chu1", "ke2"])],
        (("出壳", ("chu1", "qiao4")),),
    )
    assert ordered == ["san4", "chu1", "qiao4"]


def test_longest_phrase_wins():
    phrases = (
        ("量量", ("liang2", "liang2")),
        ("量量看", ("liang2", "liang5", "kan4")),
    )
    ordered = run_processor(
        [(Language.CHINESE, ("量量", "看"), ["liang4", "liang4", "kan4"])],
        phrases,
    )
    assert ordered == ["liang2", "liang5", "kan4"]


def test_phrases_do_not_cross_punctuation_or_foreign_runs():
    phrases = (("和泥", ("huo2", "ni2")),)
    ordered = run_processor(
        [
            (Language.CHINESE, ("和",), ["he2"]),
            (Language.SYMBOL, ("，",), ()),
            (Language.CHINESE, ("泥",), ["ni2"]),
        ],
        phrases,
    )
    assert ordered == ["he2", "ni2"]


def test_invalid_phrase_entry_is_rejected():
    with pytest.raises(ValueError):
        MandarinPhraseOverrides((("落差", ("luo4",)),))


def test_bundled_lexicon_is_well_formed():
    phrases = load_pinyin_phrases()
    assert len(phrases) == len({word for word, _ in phrases})
    for word, syllables in phrases:
        assert len(word) == len(syllables)
        for syllable in syllables:
            initial, final, tone = split_pinyin(syllable)
            assert initial or final
            assert tone in {"1", "2", "3", "4", "5"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("他说差旅的事情。", ["ta1", "shuo1", "chai1", "lv3", "de5", "shi4", "qing2"], id="chai-lv"),
        pytest.param(
            "散出壳去往黄泉。", ["san4", "chu1", "qiao4", "qu4", "wang3", "huang2", "quan2"], id="split-chu-ke"
        ),
        pytest.param("他就好打架。", ["ta1", "jiu4", "hao4", "da3", "jia4"], id="split-hao-da-jia"),
        pytest.param(
            "风吹草低见牛羊。", ["feng1", "chui1", "cao3", "di1", "xian4", "niu2", "yang2"], id="xian-niu-yang"
        ),
        pytest.param("扁鹊见蔡桓公。", ["bian3", "que4", "jian4", "cai4", "huan2", "gong1"], id="no-over-override"),
    ],
)
def test_overrides_apply_through_pypinyin_pipeline(text, expected):
    result = G2P("mandarin", tone_sandhi=False)(text)

    assert [unit.native for unit in result.units] == expected


@pytest.mark.parametrize("tone_sandhi", [True, False])
def test_overrides_apply_with_and_without_tone_sandhi(tone_sandhi):
    result = G2P("mandarin", tone_sandhi=tone_sandhi)("和点儿水泥把窟窿堵上。")

    natives = [unit.native for unit in result.units]
    assert natives[:3] == ["huo2", "dian3", "er5"]


class FakeG2PWConverter:
    def __init__(self, syllables):
        self.syllables = syllables

    def __call__(self, text):
        return [self.syllables[: len(text)]]


def test_overrides_apply_on_top_of_g2pw_output():
    fake = FakeG2PWConverter(["ta1", "shuo1", "cha4", "lv2", "zhe4", "jian4", "shi4"])
    backend = G2PWBackend(converter=fake)
    result = G2P("mandarin", backend=backend, tone_sandhi=False)("他说差旅这件事")

    assert [unit.native for unit in result.units] == [
        "ta1",
        "shuo1",
        "chai1",
        "lv3",
        "zhe4",
        "jian4",
        "shi4",
    ]
