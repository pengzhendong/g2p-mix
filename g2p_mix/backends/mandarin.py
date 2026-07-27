from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from ..errors import AlignmentError, BackendError, G2PError
from ..models import (
    ChineseDialect,
    Language,
    PhoneAlphabet,
    Pronunciation,
    PronunciationUnit,
)
from ..phonetics import split_pinyin
from ..resources import install_pypinyin_overrides
from .base import BackendCapabilities, PronunciationRequest, encode_character_projection

G2PWPredictor = Callable[[str], Sequence[Sequence[Optional[str]]]]


def _flatten_syllables(values: Iterable) -> List[str]:
    result = []
    for value in values:
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise AlignmentError(f"Expected one pronunciation per character, got {value!r}")
            value = value[0]
        if not isinstance(value, str):
            raise AlignmentError(f"Invalid pronunciation value: {value!r}")
        result.append(value)
    return result


def _build_pronunciation(
    token,
    syllables: Sequence[str],
    backend: str,
    strict: bool,
) -> Pronunciation:
    if len(syllables) != len(token.text):
        raise AlignmentError(
            f"{backend} returned {len(syllables)} syllables for {token.text!r} ({len(token.text)} characters)"
        )

    units = []
    for index, (char, syllable) in enumerate(zip(token.text, syllables)):
        try:
            initial, final, tone = split_pinyin(syllable, strict=strict)
        except (TypeError, ValueError) as error:
            raise BackendError(
                f"{backend} returned an invalid Mandarin pronunciation for {char!r}: {syllable!r}"
            ) from error
        phones = tuple(phone for phone in (initial, final) if phone)
        units.append(
            PronunciationUnit(
                text=char,
                source_spans=(token.source_spans[index],),
                phones=phones,
                tone=tone,
                alphabet=PhoneAlphabet.PINYIN,
                native=syllable,
            )
        )
    return Pronunciation(token_id=token.id, units=tuple(units), backend=backend)


class PypinyinBackend:
    name = "pypinyin"
    capabilities = BackendCapabilities(
        language=Language.CHINESE,
        dialect=ChineseDialect.MANDARIN,
        alphabet=PhoneAlphabet.PINYIN,
    )

    def __init__(self, strict: bool = False, converter=None) -> None:
        self._strict = strict
        self._converter = converter

    def _get_converter(self):
        if self._converter is None:
            from pypinyin.converter import UltimateConverter

            install_pypinyin_overrides()
            self._converter = UltimateConverter(neutral_tone_with_five=True)
        return self._converter

    def _convert(self, text: str) -> List[str]:
        from pypinyin import Style

        values = self._get_converter().convert(
            text,
            style=Style.TONE3,
            heteronym=False,
            errors="default",
            strict=True,
        )
        return _flatten_syllables(values)

    def predict(
        self,
        request: PronunciationRequest,
    ) -> Mapping[int, Pronunciation]:
        return {
            token.id: _build_pronunciation(
                token,
                self._convert(token.text),
                backend=self.name,
                strict=self._strict,
            )
            for token in request.target_tokens
        }


class G2PWBackend:
    name = "g2pw"
    capabilities = BackendCapabilities(
        language=Language.CHINESE,
        dialect=ChineseDialect.MANDARIN,
        alphabet=PhoneAlphabet.PINYIN,
        contextual=True,
        supports_projection=True,
    )

    def __init__(
        self,
        strict: bool = False,
        converter=None,
        foreign_placeholder: str = "，",
        num_workers: int = 0,
    ) -> None:
        if len(foreign_placeholder) != 1:
            raise ValueError("The model placeholder must be exactly one character")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        self._strict = strict
        self._converter = converter
        self._placeholder = foreign_placeholder
        self._num_workers = num_workers
        self._fallback = PypinyinBackend(strict=strict)

    def _get_converter(self) -> G2PWPredictor:
        if self._converter is None:
            try:
                from g2pw import G2PWConverter
                from modelscope import snapshot_download
            except ImportError as error:
                raise BackendError(
                    "G2PW backend is unavailable; install it with 'pip install g2p-mix[g2pw]'"
                ) from error

            try:
                repo_dir = snapshot_download("pengzhendong/g2pw")
            except G2PError:
                raise
            except Exception as error:
                raise BackendError("G2PW model snapshot download failed") from error
            try:
                converter = G2PWConverter(
                    model_dir=f"{repo_dir}/G2PWModel",
                    style="pinyin",
                    model_source=f"{repo_dir}/bert-base-chinese",
                    num_workers=self._num_workers,
                    enable_non_tradional_chinese=True,
                )
                converter.num_workers = self._num_workers
            except G2PError:
                raise
            except Exception as error:
                raise BackendError("G2PW converter initialization failed") from error
            self._converter = converter
        return self._converter

    def _convert(self, text: str) -> Sequence[Optional[str]]:
        try:
            values = self._get_converter()(text)
        except G2PError:
            raise
        except Exception as error:
            raise BackendError("G2PW prediction failed") from error

        try:
            if not isinstance(values, SequenceABC) or isinstance(values, (str, bytes)):
                raise BackendError(
                    f"{self.name} converter outer result for projection {text!r} must be a non-string sequence"
                )
            sentence_count = len(values)
            if sentence_count != 1:
                raise BackendError(
                    f"{self.name} converter returned {sentence_count} sentences for one projection {text!r}"
                )
            syllables = values[0]
            if not isinstance(syllables, SequenceABC) or isinstance(syllables, (str, bytes)):
                raise BackendError(
                    f"{self.name} converter sentence result for projection {text!r} must be a non-string sequence"
                )
            return tuple(syllables)
        except G2PError:
            raise
        except Exception as error:
            raise BackendError(
                f"{self.name} prediction failed while validating converter output for projection {text!r}"
            ) from error

    def predict(
        self,
        request: PronunciationRequest,
    ) -> Mapping[int, Pronunciation]:
        projection = encode_character_projection(request, self._placeholder)
        syllables = self._convert(projection.text)
        if len(syllables) != len(projection.sources):
            raise BackendError(
                f"{self.name} converter returned {len(syllables)} positions for a "
                f"{len(projection.sources)}-character projection {projection.text!r}"
            )

        by_token: Dict[int, Dict[int, str]] = {}
        for position, source in enumerate(projection.sources):
            syllable = syllables[position]
            if syllable is not None and not isinstance(syllable, str):
                raise BackendError(
                    f"{self.name} converter returned a non-string syllable at position {position} "
                    f"for projection {projection.text!r}"
                )
            if source is None:
                continue
            if syllable is None:
                syllable = self._fallback._convert(projection.text[position])[0]
            token_id, char_index = source
            by_token.setdefault(token_id, {})[char_index] = syllable

        result = {}
        for token in request.target_tokens:
            token_syllables = [by_token.get(token.id, {}).get(index) for index in range(len(token.text))]
            if any(syllable is None for syllable in token_syllables):
                raise AlignmentError(f"{self.name} did not predict every character in {token.text!r}")
            result[token.id] = _build_pronunciation(
                token,
                token_syllables,
                backend=self.name,
                strict=self._strict,
            )
        return result
