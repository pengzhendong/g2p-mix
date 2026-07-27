from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Tuple

from ..models import (
    ChineseDialect,
    Language,
    LanguageProjection,
    PhoneAlphabet,
    Pronunciation,
    TextToken,
)


@dataclass(frozen=True)
class BackendCapabilities:
    language: Language
    alphabet: PhoneAlphabet
    dialect: Optional[ChineseDialect] = None
    contextual: bool = False
    supports_batch: bool = False
    supports_projection: bool = False


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


class PronunciationBackend(Protocol):
    name: str
    capabilities: BackendCapabilities

    def predict(
        self,
        request: PronunciationRequest,
    ) -> Mapping[int, Pronunciation]:
        pass
