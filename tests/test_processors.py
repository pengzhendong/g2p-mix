import json
from pathlib import Path

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

CASE_FILE = Path(__file__).parent / "cases" / "mandarin_tone_sandhi.json"
CASE_GROUPS = json.loads(CASE_FILE.read_text(encoding="utf-8"))
LANGUAGES = {
    "chinese": Language.CHINESE,
    "english": Language.ENGLISH,
}


def cases(*groups):
    return [pytest.param(case, id=f"{group}-{case['id']}") for group in groups for case in CASE_GROUPS[group]]


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
    if len(text) != len(tones):
        raise ValueError(f"{text!r} has {len(text)} characters but {len(tones)} tones")

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


def make_tokens(specs):
    items = []
    cursor = 0
    for token_id, spec in enumerate(specs):
        text = spec["text"]
        items.append(
            make_token(
                token_id,
                text,
                spec["tones"],
                start=cursor,
                pos=spec.get("pos", "x"),
                language=LANGUAGES[spec.get("language", "chinese")],
            )
        )
        cursor += len(text)
    return items


def process(*items, word_splits=None):
    splits = {} if word_splits is None else word_splits

    def split_word(word):
        parts = splits.get(word, (word,))
        if "".join(parts) != word:
            raise ValueError(f"split for {word!r} is not lossless: {parts!r}")
        return parts

    tokens = tuple(token for token, _ in items)
    pronunciations = {token.id: pronunciation for token, pronunciation in items if pronunciation is not None}
    result = MandarinToneSandhi(word_splitter=split_word).process(tokens, pronunciations)
    return {token_id: tuple(unit.tone for unit in pronunciation.units) for token_id, pronunciation in result.items()}


@pytest.mark.parametrize("case", cases("bu", "yi", "neutral", "third_tone"))
def test_word_sandhi(case):
    text = case["text"]
    split = case.get("split", [text])
    result = process(
        make_token(0, text, case["tones"], pos=case.get("pos", "x")),
        word_splits={text: split},
    )

    assert result[0] == tuple(case["expected"])


@pytest.mark.parametrize("case", cases("word_splitter"))
def test_default_word_splitter_preserves_sandhi_subwords(case):
    assert MandarinToneSandhi._split_word(case["text"]) == case["expected"]


@pytest.mark.parametrize("case", cases("token_runs"))
def test_sandhi_across_token_runs(case):
    result = process(
        *make_tokens(case["tokens"]),
        word_splits=case.get("word_splits"),
    )
    expected = {int(token_id): tuple(tones) for token_id, tones in case["expected"].items()}

    assert result == expected
