from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from ..models import Language, PhoneAlphabet, Pronunciation, PronunciationUnit, TextToken
from ..phonetics import split_pinyin
from ..resources import load_pinyin_phrases

_Phrases = Sequence[Tuple[str, Tuple[str, ...]]]


class MandarinPhraseOverrides:
    """Apply lexicon phrase readings independent of word segmentation.

    Backends are consulted per token (pypinyin) or per projection (G2PW), so a
    phrase override from ``dict/phrases.txt`` silently disappears whenever the
    segmenter splits the phrase. This processor re-matches the phrase lexicon
    against the surviving tokens — left to right, longest match first — and
    pins the covered units to the lexicon readings, for every Mandarin backend.
    """

    def __init__(self, phrases: Optional[_Phrases] = None) -> None:
        entries = tuple(phrases) if phrases is not None else load_pinyin_phrases()
        by_first_char: Dict[str, List[Tuple[str, Tuple[str, ...]]]] = {}
        for word, syllables in entries:
            if not word or len(word) != len(syllables):
                raise ValueError(f"Invalid pinyin override for {word!r}")
            by_first_char.setdefault(word[0], []).append((word, syllables))
        for candidates in by_first_char.values():
            candidates.sort(key=lambda item: len(item[0]), reverse=True)
        self._by_first_char = by_first_char

    def process(
        self,
        tokens: Sequence[TextToken],
        pronunciations: Mapping[int, Pronunciation],
    ) -> Mapping[int, Pronunciation]:
        result: MutableMapping[int, Pronunciation] = dict(pronunciations)
        run: List[TextToken] = []

        def flush() -> None:
            if not run:
                return
            self._apply_run(run, result)
            run.clear()

        for token in tokens:
            pronunciation = result.get(token.id)
            if (
                token.language is not Language.CHINESE
                or pronunciation is None
                or len(pronunciation.units) != len(token.text)
            ):
                flush()
                continue
            run.append(token)
        flush()
        return result

    def _apply_run(
        self,
        run: Sequence[TextToken],
        result: MutableMapping[int, Pronunciation],
    ) -> None:
        text = "".join(token.text for token in run)
        refs = [(token.id, index) for token in run for index in range(len(token.text))]

        changes: Dict[Tuple[int, int], str] = {}
        position = 0
        while position < len(text):
            match = None
            for word, syllables in self._by_first_char.get(text[position], ()):  # longest first
                if text.startswith(word, position):
                    match = (word, syllables)
                    break
            if match is None:
                position += 1
                continue
            word, syllables = match
            for offset, syllable in enumerate(syllables):
                changes[refs[position + offset]] = syllable
            position += len(word)

        if not changes:
            return
        for token in run:
            pronunciation = result.get(token.id)
            if pronunciation is None:
                continue
            units: List[PronunciationUnit] = []
            changed = False
            for index, unit in enumerate(pronunciation.units):
                syllable = changes.get((token.id, index))
                if syllable is None:
                    units.append(unit)
                    continue
                initial, final, tone = split_pinyin(syllable)
                phones = tuple(phone for phone in (initial, final) if phone)
                units.append(
                    replace(
                        unit,
                        phones=phones,
                        tone=tone,
                        alphabet=PhoneAlphabet.PINYIN,
                        native=syllable,
                        is_unknown=False,
                    )
                )
                changed = True
            if changed:
                result[token.id] = replace(pronunciation, units=tuple(units))
