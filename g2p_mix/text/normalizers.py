from __future__ import annotations

from typing import Iterable, Protocol

from ..models import NormalizedText


class TextNormalizer(Protocol):
    def normalize(self, value: NormalizedText) -> NormalizedText:
        pass


class IdentityNormalizer:
    def normalize(self, value: NormalizedText) -> NormalizedText:
        return value


class TraditionalChineseNormalizer:
    def __init__(self) -> None:
        self._converter = None

    def _get_converter(self):
        if self._converter is None:
            from pyopenhc import OpenHC

            self._converter = OpenHC("s2t")
        return self._converter

    def normalize(self, value: NormalizedText) -> NormalizedText:
        converted = self._get_converter().convert(value.text)
        if len(converted) != len(value.text):
            raise ValueError(
                "The traditional Chinese converter changed text length and cannot preserve source alignment"
            )
        return NormalizedText(
            original=value.original,
            text=converted,
            char_sources=value.char_sources,
        )


class NormalizationPipeline:
    def __init__(self, normalizers: Iterable[TextNormalizer] = ()) -> None:
        self._normalizers = tuple(normalizers)

    def normalize(self, text: str) -> NormalizedText:
        value = NormalizedText.identity(text)
        for normalizer in self._normalizers:
            value = normalizer.normalize(value)
        return value
