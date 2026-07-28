from __future__ import annotations

import re
from typing import Sequence, Tuple

CodepointRange = Tuple[int, int]

# Keep this table explicit so classification is stable across Python versions.
# These are assigned Han ideographs rather than the maxima of their Unicode
# blocks, which also contain reserved codepoints.
HAN_CODEPOINT_RANGES: Tuple[CodepointRange, ...] = (
    (0x3007, 0x3007),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFA6D),
    (0xFA70, 0xFAD9),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81D),
    (0x2B820, 0x2CEAD),
    (0x2CEB0, 0x2EBE0),
    (0x2EBF0, 0x2EE5D),
    (0x2F800, 0x2FA1D),
    (0x30000, 0x3134A),
    (0x31350, 0x323AF),
    (0x323B0, 0x33479),
)

# Unicode 17.0 Script=Latin codepoints, with adjacent property ranges merged.
# This deliberately vendors Scripts.txt instead of inheriting the host
# Python's Unicode database or treating whole Latin-named blocks as Latin.
LATIN_CODEPOINT_RANGES: Tuple[CodepointRange, ...] = (
    (0x0041, 0x005A),
    (0x0061, 0x007A),
    (0x00AA, 0x00AA),
    (0x00BA, 0x00BA),
    (0x00C0, 0x00D6),
    (0x00D8, 0x00F6),
    (0x00F8, 0x02B8),
    (0x02E0, 0x02E4),
    (0x1D00, 0x1D25),
    (0x1D2C, 0x1D5C),
    (0x1D62, 0x1D65),
    (0x1D6B, 0x1D77),
    (0x1D79, 0x1DBE),
    (0x1E00, 0x1EFF),
    (0x2071, 0x2071),
    (0x207F, 0x207F),
    (0x2090, 0x209C),
    (0x212A, 0x212B),
    (0x2132, 0x2132),
    (0x214E, 0x214E),
    (0x2160, 0x2188),
    (0x2C60, 0x2C7F),
    (0xA722, 0xA787),
    (0xA78B, 0xA7DC),
    (0xA7F1, 0xA7FF),
    (0xAB30, 0xAB5A),
    (0xAB5C, 0xAB64),
    (0xAB66, 0xAB69),
    (0xFB00, 0xFB06),
    (0xFF21, 0xFF3A),
    (0xFF41, 0xFF5A),
    (0x10780, 0x10785),
    (0x10787, 0x107B0),
    (0x107B2, 0x107BA),
    (0x1DF00, 0x1DF1E),
    (0x1DF25, 0x1DF2A),
)

# Combining marks are continuations only: LATIN_SEGMENT always requires a
# Script=Latin codepoint before this class can match.
COMBINING_MARK_RANGES: Tuple[CodepointRange, ...] = (
    (0x0300, 0x036F),
    (0x1AB0, 0x1AFF),
    (0x1DC0, 0x1DFF),
    (0x20D0, 0x20FF),
    (0xFE20, 0xFE2F),
)


def _contains(codepoint: int, ranges: Sequence[CodepointRange]) -> bool:
    return any(start <= codepoint <= end for start, end in ranges)


def is_han_character(value: str) -> bool:
    return len(value) == 1 and _contains(ord(value), HAN_CODEPOINT_RANGES)


def is_latin_character(value: str) -> bool:
    return len(value) == 1 and _contains(ord(value), LATIN_CODEPOINT_RANGES)


def is_combining_mark(value: str) -> bool:
    return len(value) == 1 and _contains(ord(value), COMBINING_MARK_RANGES)


def _regex_character_class(ranges: Sequence[CodepointRange]) -> str:
    return "".join(
        re.escape(chr(start)) if start == end else f"{re.escape(chr(start))}-{re.escape(chr(end))}"
        for start, end in ranges
    )


# Imported by the tokenizer when it builds its compiled pattern.
HAN_CHARACTER_CLASS = _regex_character_class(HAN_CODEPOINT_RANGES)
LATIN_CHARACTER_CLASS = _regex_character_class(LATIN_CODEPOINT_RANGES)
COMBINING_MARK_CHARACTER_CLASS = _regex_character_class(COMBINING_MARK_RANGES)
