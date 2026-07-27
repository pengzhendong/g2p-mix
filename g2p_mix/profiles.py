from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .backends import (
    EnglishBackend,
    PronunciationBackend,
    PyCantoneseBackend,
    PypinyinBackend,
)
from .errors import ConfigurationError
from .models import ChineseDialect, Language
from .normalizers import IdentityNormalizer, TextNormalizer, TraditionalChineseNormalizer
from .processors import MandarinToneSandhi, PronunciationProcessor
from .tokenizer import (
    ChineseSegmenter,
    JiebaSegmenter,
    PyCantoneseSegmenter,
)


@dataclass(frozen=True)
class ChineseProfile:
    dialect: ChineseDialect
    backend: PronunciationBackend
    segmenter: ChineseSegmenter
    normalizers: Tuple[TextNormalizer, ...]
    processors: Tuple[PronunciationProcessor, ...]

    def __post_init__(self) -> None:
        capabilities = self.backend.capabilities
        if capabilities.language is not Language.CHINESE:
            raise ConfigurationError("ChineseProfile requires a Chinese backend")
        if capabilities.dialect is not self.dialect:
            raise ConfigurationError(f"{self.backend.name} supports {capabilities.dialect}, not {self.dialect}")


class MandarinProfile(ChineseProfile):
    def __init__(
        self,
        backend=None,
        *,
        tone_sandhi: bool = True,
    ) -> None:
        super().__init__(
            dialect=ChineseDialect.MANDARIN,
            backend=PypinyinBackend() if backend is None else backend,
            segmenter=JiebaSegmenter(),
            normalizers=(IdentityNormalizer(),),
            processors=(MandarinToneSandhi(),) if tone_sandhi else (),
        )


class CantoneseProfile(ChineseProfile):
    def __init__(
        self,
        backend=None,
        *,
        traditional: bool = True,
        tagset: str = "universal",
    ) -> None:
        normalizers = (TraditionalChineseNormalizer(),) if traditional else (IdentityNormalizer(),)
        super().__init__(
            dialect=ChineseDialect.CANTONESE,
            backend=PyCantoneseBackend() if backend is None else backend,
            segmenter=PyCantoneseSegmenter(tagset=tagset),
            normalizers=normalizers,
            processors=(),
        )


@dataclass(frozen=True)
class EnglishProfile:
    backend: PronunciationBackend
    processors: Tuple[PronunciationProcessor, ...] = ()

    def __post_init__(self) -> None:
        if self.backend.capabilities.language is not Language.ENGLISH:
            raise ConfigurationError("EnglishProfile requires an English backend")

    @classmethod
    def default(cls) -> "EnglishProfile":
        return cls(backend=EnglishBackend())
