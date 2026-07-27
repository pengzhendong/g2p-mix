from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..errors import AlignmentError
from ..models import (
    ChineseDialect,
    Language,
    PhoneAlphabet,
    ProjectionKind,
    Pronunciation,
    PronunciationUnit,
)
from ..phonetics import split_pinyin
from ..resources import install_pinyin_overrides
from .base import BackendCapabilities, PronunciationRequest


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
        initial, final, tone = split_pinyin(syllable, strict=strict)
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

            install_pinyin_overrides()
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
        foreign_placeholder: str = "\ue000",
    ) -> None:
        if len(foreign_placeholder) != 1:
            raise ValueError("The model placeholder must be exactly one character")
        self._strict = strict
        self._converter = converter
        self._placeholder = foreign_placeholder

    def _get_converter(self):
        if self._converter is None:
            from modelscope import snapshot_download
            from pypinyin_g2pw import G2PWPinyin

            repo_dir = snapshot_download("pengzhendong/g2pw")
            self._converter = G2PWPinyin(
                model_dir=f"{repo_dir}/G2PWModel",
                model_source=f"{repo_dir}/bert-base-chinese",
                neutral_tone_with_five=True,
            )._converter
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

    def _encode_projection(
        self,
        request: PronunciationRequest,
    ) -> Tuple[str, Tuple[Optional[Tuple[int, int]], ...]]:
        tokens_by_id = request.tokens_by_id
        characters: List[str] = []
        alignment: List[Optional[Tuple[int, int]]] = []

        for projected in request.projection.tokens:
            if projected.kind is ProjectionKind.TARGET:
                token = tokens_by_id[projected.source_ids[0]]
                for index, char in enumerate(token.text):
                    characters.append(char)
                    alignment.append((token.id, index))
            else:
                if not alignment or alignment[-1] is not None:
                    characters.append(self._placeholder)
                    alignment.append(None)
        return "".join(characters), tuple(alignment)

    def predict(
        self,
        request: PronunciationRequest,
    ) -> Mapping[int, Pronunciation]:
        projected_text, alignment = self._encode_projection(request)
        syllables = self._convert(projected_text)
        if len(syllables) != len(alignment):
            raise AlignmentError(
                f"{self.name} returned {len(syllables)} positions for a {len(alignment)}-character projection"
            )

        by_token: Dict[int, Dict[int, str]] = {}
        for position, source in enumerate(alignment):
            if source is None:
                continue
            token_id, char_index = source
            by_token.setdefault(token_id, {})[char_index] = syllables[position]

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
