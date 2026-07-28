from __future__ import annotations

import unicodedata

from .unicode_script import is_latin_character


def fold_english_spelling(value: str) -> str:
    """Fold decomposable Latin diacritics while rejecting lossy transliteration."""

    folded = []
    for char in unicodedata.normalize("NFC", value):
        if char.isascii():
            folded.append(char)
            continue
        if not is_latin_character(char):
            raise ValueError(f"Unsupported character in English spelling: {char!r}")

        candidate = "".join(part for part in unicodedata.normalize("NFKD", char) if not unicodedata.combining(part))
        if not candidate or not candidate.isascii() or not candidate.isalpha():
            raise ValueError(f"Latin character cannot be folded safely: {char!r}")
        folded.append(candidate)
    return "".join(folded)
