from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .backends import (
    EnglishBackend,
    PronunciationBackend,
    PypinyinBackend,
    ToJyutpingBackend,
)
from .errors import ConfigurationError
from .models import ChineseDialect, Language
from .processors import MandarinToneSandhi, PronunciationProcessor
from .text import (
    AsciiLatinValidator,
    ChineseSegmenter,
    JiebaSegmenter,
    PyCantoneseSegmenter,
    TextNormalizer,
    TraditionalChineseNormalizer,
    UnicodeCompatibilityNormalizer,
    WeTextNormalizer,
)


def _default_normalizers(*, traditional: bool = False) -> Tuple[TextNormalizer, ...]:
    normalizers = [
        UnicodeCompatibilityNormalizer(),
        WeTextNormalizer(),
    ]
    if traditional:
        normalizers.append(TraditionalChineseNormalizer())
    return tuple(normalizers)


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
            normalizers=_default_normalizers(),
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
        super().__init__(
            dialect=ChineseDialect.CANTONESE,
            backend=ToJyutpingBackend() if backend is None else backend,
            segmenter=PyCantoneseSegmenter(tagset=tagset),
            normalizers=_default_normalizers(traditional=traditional),
            processors=(),
        )


@dataclass(frozen=True)
class EnglishProfile:
    backend: PronunciationBackend
    normalizers: Tuple[TextNormalizer, ...] = ()
    processors: Tuple[PronunciationProcessor, ...] = ()

    def __post_init__(self) -> None:
        if self.backend.capabilities.language is not Language.ENGLISH:
            raise ConfigurationError("EnglishProfile requires an English backend")

    @classmethod
    def for_backend(cls, backend: PronunciationBackend) -> "EnglishProfile":
        normalizers = (AsciiLatinValidator(),) if backend.capabilities.ascii_latin_only else ()
        return cls(backend=backend, normalizers=normalizers)

    @classmethod
    def default(cls) -> "EnglishProfile":
        return cls.for_backend(EnglishBackend())
