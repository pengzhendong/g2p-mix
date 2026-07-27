from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from ..resources import install_pinyin_overrides

PronunciationLookup = Callable[[str], Sequence[str]]


def _is_han(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
    )


class MandarinLexicon:
    def __init__(self, lookup: Optional[PronunciationLookup] = None) -> None:
        self._lookup = lookup if lookup is not None else self._pypinyin_lookup
        self._cache: Dict[str, Tuple[str, ...]] = {}

    @staticmethod
    def _pypinyin_lookup(char: str) -> Sequence[str]:
        from pypinyin import Style, pinyin

        install_pinyin_overrides()
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
        if len(char) != 1 or not _is_han(char):
            raise ValueError(f"Expected exactly one Han character, got {char!r}")
        if char not in self._cache:
            values = self._lookup(char)
            self._cache[char] = tuple(
                dict.fromkeys(value for value in values if isinstance(value, str) and value and value[-1] in "12345")
            )
        return self._cache[char]

    def scan(self, text: str) -> Mapping[str, Tuple[str, ...]]:
        return {char: self.pronunciations(char) for char in dict.fromkeys(text) if _is_han(char)}

    def clear_cache(self) -> None:
        self._cache.clear()
