from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Protocol, Tuple

from ..errors import BackendError, ConfigurationError
from ..models import (
    ChineseDialect,
    Language,
    LanguageProjection,
    PhoneAlphabet,
    ProjectionKind,
    Pronunciation,
    PronunciationUnit,
    TextToken,
)


@dataclass(frozen=True)
class BackendCapabilities:
    language: Language
    alphabet: PhoneAlphabet
    dialect: Optional[ChineseDialect] = None
    ascii_latin_only: bool = False


@dataclass(frozen=True)
class PronunciationRequest:
    tokens: Tuple[TextToken, ...]
    projection: LanguageProjection
    dialect: Optional[ChineseDialect] = None

    @property
    def target_tokens(self) -> Tuple[TextToken, ...]:
        return tuple(token for token in self.tokens if token.language is self.projection.target)

    @property
    def tokens_by_id(self) -> Mapping[int, TextToken]:
        return {token.id: token for token in self.tokens}


@dataclass(frozen=True)
class CharacterProjection:
    text: str
    sources: Tuple[Optional[Tuple[int, int]], ...]


def encode_character_projection(
    request: PronunciationRequest,
    placeholder: str,
    *,
    preserve_context: bool = False,
) -> CharacterProjection:
    if len(placeholder) != 1:
        raise ValueError("The model placeholder must be exactly one character")

    tokens_by_id = request.tokens_by_id
    characters: List[str] = []
    sources: List[Optional[Tuple[int, int]]] = []

    for projected in request.projection.tokens:
        if projected.kind is ProjectionKind.TARGET:
            token = tokens_by_id[projected.source_ids[0]]
            for index, char in enumerate(token.text):
                characters.append(char)
                sources.append((token.id, index))
        elif preserve_context and projected.kind is ProjectionKind.CONTEXT:
            characters.extend(projected.text)
            sources.extend([None] * len(projected.text))
        elif preserve_context or not sources or sources[-1] is not None:
            characters.append(placeholder)
            sources.append(None)

    return CharacterProjection(text="".join(characters), sources=tuple(sources))


class PronunciationBackend(Protocol):
    name: str
    capabilities: BackendCapabilities

    def predict(
        self,
        request: PronunciationRequest,
    ) -> Mapping[int, Pronunciation]:
        pass


def unknown_unit(
    token: TextToken,
    index: int,
    alphabet: PhoneAlphabet,
) -> PronunciationUnit:
    return PronunciationUnit(
        text=token.text[index],
        source_spans=(token.source_spans[index],),
        phones=(),
        alphabet=alphabet,
        native="",
        is_unknown=True,
    )


class FallbackBackend:
    """Use a second compatible backend when the primary backend fails."""

    def __init__(
        self,
        primary: PronunciationBackend,
        fallback: PronunciationBackend,
    ) -> None:
        primary_identity = (
            primary.capabilities.language,
            primary.capabilities.dialect,
            primary.capabilities.alphabet,
        )
        fallback_identity = (
            fallback.capabilities.language,
            fallback.capabilities.dialect,
            fallback.capabilities.alphabet,
        )
        if primary_identity != fallback_identity:
            raise ConfigurationError("Fallback backend capabilities must match the primary backend")
        if primary.name == fallback.name:
            raise ConfigurationError("Fallback backend must differ from the primary backend")
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}->{fallback.name}"
        self.capabilities = primary.capabilities

    def predict(
        self,
        request: PronunciationRequest,
    ) -> Mapping[int, Pronunciation]:
        try:
            return self.primary.predict(request)
        except BackendError as primary_error:
            try:
                return self.fallback.predict(request)
            except BackendError as fallback_error:
                raise BackendError(
                    f"Primary backend {self.primary.name!r} failed: "
                    f"{primary_error}; fallback backend "
                    f"{self.fallback.name!r} also failed: {fallback_error}"
                ) from fallback_error
