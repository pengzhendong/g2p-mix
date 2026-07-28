from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from typing import Iterable, Protocol

from ..errors import NormalizationError
from ..models import NormalizedText, Span
from .latin import fold_english_spelling
from .unicode_script import is_combining_mark, is_latin_character


class TextNormalizer(Protocol):
    def normalize(self, value: NormalizedText) -> NormalizedText:
        pass


class IdentityNormalizer:
    def normalize(self, value: NormalizedText) -> NormalizedText:
        return value


class UnicodeCompatibilityNormalizer:
    """Normalize compatibility letters and numbers without rewriting punctuation."""

    def normalize(self, value: NormalizedText) -> NormalizedText:
        characters = []
        sources = []
        for char, source in zip(value.text, value.char_sources):
            converted = unicodedata.normalize("NFKC", char) if unicodedata.category(char)[:1] in {"L", "N"} else char
            characters.append(converted)
            sources.extend(source for _ in converted)

        decomposed = NormalizedText(
            original=value.original,
            text="".join(characters),
            char_sources=tuple(sources),
        )
        composed = unicodedata.normalize("NFC", decomposed.text)
        if composed == decomposed.text:
            return decomposed
        return NormalizedText(
            original=value.original,
            text=composed,
            char_sources=_align_normalized_characters(decomposed, composed),
        )


class WeTextNormalizer:
    """Expand written forms while preserving their relationship to the source text."""

    def __init__(self, normalizer=None) -> None:
        self._normalizer = normalizer

    def _get_normalizer(self):
        if self._normalizer is None:
            try:
                from wetext import Normalizer

                self._normalizer = Normalizer(lang="auto", operator="tn")
            except Exception as error:
                raise NormalizationError("WeText normalizer initialization failed") from error
        return self._normalizer

    def normalize(self, value: NormalizedText) -> NormalizedText:
        try:
            converted = self._get_normalizer().normalize(value.text)
        except NormalizationError:
            raise
        except Exception as error:
            raise NormalizationError(f"WeText normalization failed for {value.text!r}") from error
        if not isinstance(converted, str):
            raise NormalizationError("WeText normalizer returned a non-string value")
        if converted == value.text:
            return value
        return NormalizedText(
            original=value.original,
            text=converted,
            char_sources=_align_normalized_characters(value, converted),
        )


class AsciiLatinValidator:
    """Require Latin text to be ASCII or safely foldable for the English backend."""

    def normalize(self, value: NormalizedText) -> NormalizedText:
        inside_latin_segment = False
        for index, char in enumerate(value.text):
            if is_latin_character(char):
                inside_latin_segment = True
                if char.isascii():
                    continue
                try:
                    fold_english_spelling(char)
                except ValueError:
                    self._raise_unsupported(value, index, char)
            elif is_combining_mark(char) and inside_latin_segment:
                self._raise_unsupported(value, index, char)
            else:
                inside_latin_segment = False
        return value

    @staticmethod
    def _raise_unsupported(value: NormalizedText, index: int, char: str) -> None:
        source = value.char_sources[index]
        original = source.slice(value.original)
        raise NormalizationError(
            "Unsupported Latin text after Unicode normalization: "
            f"character={char!r}, source={original!r}, span=[{source.start}, {source.end})"
        )


def _align_normalized_characters(value: NormalizedText, converted: str) -> tuple[Span, ...]:
    aligned = []
    matcher = SequenceMatcher(a=value.text, b=converted, autojunk=False)
    for operation, source_start, source_end, target_start, target_end in matcher.get_opcodes():
        if operation == "equal":
            aligned.extend(value.char_sources[source_start:source_end])
            continue
        if operation == "delete":
            continue

        source = _source_span_for_change(value, source_start, source_end)
        aligned.extend(source for _ in range(target_start, target_end))
    return tuple(aligned)


def _source_span_for_change(value: NormalizedText, start: int, end: int) -> Span:
    sources = value.char_sources[start:end]
    if not sources and value.char_sources:
        adjacent_start = max(0, start - 1)
        adjacent_end = min(len(value.char_sources), start + 1)
        sources = value.char_sources[adjacent_start:adjacent_end]
    if not sources:
        return Span(0, 0)
    return Span(
        min(source.start for source in sources),
        max(source.end for source in sources),
    )


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
