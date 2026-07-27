from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping, Optional, Tuple


class Language(str, Enum):
    CHINESE = "zh"
    ENGLISH = "en"
    NUMBER = "num"
    SYMBOL = "sym"
    SPACE = "space"


class ChineseDialect(str, Enum):
    MANDARIN = "mandarin"
    CANTONESE = "cantonese"


class TokenKind(str, Enum):
    HAN = "han"
    LATIN = "latin"
    NUMBER = "number"
    PUNCTUATION = "punctuation"
    SYMBOL = "symbol"
    SPACE = "space"


class Boundary(str, Enum):
    NONE = "none"
    SOFT = "soft"
    CODE_SWITCH = "code_switch"
    PAUSE = "pause"
    SENTENCE = "sentence"


class ProjectionKind(str, Enum):
    TARGET = "target"
    PLACEHOLDER = "placeholder"
    CONTEXT = "context"


class PhoneAlphabet(str, Enum):
    PINYIN = "pinyin"
    JYUTPING = "jyutping"
    ARPABET = "arpabet"
    IPA = "ipa"


@dataclass(frozen=True, order=True)
class Span:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid span: [{self.start}, {self.end})")

    def slice(self, text: str) -> str:
        return text[self.start : self.end]


@dataclass(frozen=True)
class NormalizedText:
    original: str
    text: str
    char_sources: Tuple[Span, ...]

    def __post_init__(self) -> None:
        if len(self.text) != len(self.char_sources):
            raise ValueError("Every normalized character must have a source span")

    @classmethod
    def identity(cls, text: str) -> "NormalizedText":
        return cls(
            original=text,
            text=text,
            char_sources=tuple(Span(index, index + 1) for index in range(len(text))),
        )

    def sources_for(self, span: Span) -> Tuple[Span, ...]:
        return self.char_sources[span.start : span.end]


@dataclass(frozen=True)
class TextToken:
    id: int
    text: str
    normalized_span: Span
    source_spans: Tuple[Span, ...]
    language: Language
    kind: TokenKind
    boundary_before: Boundary = Boundary.NONE
    pos: Optional[str] = None


@dataclass(frozen=True)
class ProjectionToken:
    text: str
    source_ids: Tuple[int, ...]
    kind: ProjectionKind
    boundary_before: Boundary

    @property
    def is_target(self) -> bool:
        return self.kind is ProjectionKind.TARGET


@dataclass(frozen=True)
class LanguageProjection:
    target: Language
    tokens: Tuple[ProjectionToken, ...]

    @property
    def text(self) -> str:
        return "".join(token.text for token in self.tokens)

    @property
    def target_mask(self) -> Tuple[bool, ...]:
        return tuple(token.is_target for token in self.tokens)


@dataclass(frozen=True)
class PronunciationUnit:
    text: str
    source_spans: Tuple[Span, ...]
    phones: Tuple[str, ...]
    alphabet: PhoneAlphabet
    native: str
    tone: Optional[str] = None
    stress: Optional[int] = None

    def with_tone(self, tone: str) -> "PronunciationUnit":
        return replace(self, tone=tone)


@dataclass(frozen=True)
class Pronunciation:
    token_id: int
    units: Tuple[PronunciationUnit, ...]
    backend: str
    confidence: Optional[float] = None


@dataclass(frozen=True)
class OutputToken:
    token: TextToken
    pronunciation: Optional[Pronunciation]

    @property
    def units(self) -> Tuple[PronunciationUnit, ...]:
        if self.pronunciation is None:
            return ()
        return self.pronunciation.units


@dataclass(frozen=True)
class G2PResult:
    original_text: str
    normalized_text: str
    tokens: Tuple[OutputToken, ...]
    projections: Mapping[Language, LanguageProjection]
    warnings: Tuple[str, ...] = ()

    def reconstruct_original(self) -> str:
        return self.original_text

    @property
    def units(self) -> Tuple[PronunciationUnit, ...]:
        return tuple(unit for token in self.tokens for unit in token.units)

    @property
    def phones(self) -> Tuple[str, ...]:
        return tuple(phone for unit in self.units for phone in unit.phones)
