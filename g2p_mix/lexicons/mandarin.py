from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from ..resources import install_pypinyin_overrides
from ..text.unicode_script import is_han_character

PronunciationLookup = Callable[[str], Sequence[str]]


class MandarinLexicon:
    def __init__(self, lookup: Optional[PronunciationLookup] = None) -> None:
        self._lookup = lookup if lookup is not None else self._pypinyin_lookup
        self._cache: Dict[str, Tuple[str, ...]] = {}

    @staticmethod
    def _pypinyin_lookup(char: str) -> Sequence[str]:
        from pypinyin import Style, pinyin

        install_pypinyin_overrides()
        values = pinyin(
            char,
            style=Style.TONE3,
            heteronym=True,
            errors="default",
            strict=True,
            neutral_tone_with_five=True,
        )
        return values[0] if len(values) == 1 else ()

    def pronunciations(self, char: str) -> Tuple[str, ...]:
        if not is_han_character(char):
            raise ValueError(f"Expected exactly one Han character, got {char!r}")
        if char not in self._cache:
            values = self._lookup(char)
            self._cache[char] = tuple(
                dict.fromkeys(value for value in values if isinstance(value, str) and value and value[-1] in "12345")
            )
        return self._cache[char]

    def scan(self, text: str) -> Mapping[str, Tuple[str, ...]]:
        return {char: self.pronunciations(char) for char in dict.fromkeys(text) if is_han_character(char)}

    def clear_cache(self) -> None:
        self._cache.clear()
