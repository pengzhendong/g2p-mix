from __future__ import annotations

import re
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..errors import AlignmentError, BackendError
from ..models import (
    ChineseDialect,
    Language,
    PhoneAlphabet,
    Pronunciation,
    PronunciationUnit,
)
from ..phonetics import split_jyutping
from .base import BackendCapabilities, PronunciationRequest, encode_character_projection

JyutpingList = Sequence[Tuple[str, Optional[str]]]
JyutpingConverter = Callable[[str], JyutpingList]
JYUTPING_PATTERN = re.compile(r"[a-z]+[1-6]")


def _parse_jyutping_sequence(
    raw: str,
    *,
    backend: str,
    source_chars: str,
    source_text: str,
) -> List[str]:
    try:
        syllables = raw.split()
        if not syllables:
            raise ValueError("The pronunciation is empty")
        for syllable in syllables:
            split_jyutping(syllable)
    except (AttributeError, TypeError, ValueError) as error:
        raise BackendError(
            f"Invalid Cantonese pronunciation from {backend} for source "
            f"{source_chars!r} in text {source_text!r}: raw={raw!r}"
        ) from error
    return syllables


def _build_units(
    token,
    pronunciations: Sequence[Sequence[str]],
    backend: str,
) -> Tuple[PronunciationUnit, ...]:
    units = []
    for index, (char, syllables) in enumerate(zip(token.text, pronunciations)):
        for syllable in syllables:
            try:
                onset, final, tone = split_jyutping(syllable)
            except (TypeError, ValueError) as error:
                raise BackendError(
                    f"Invalid Cantonese pronunciation from {backend} for {char!r}: {syllable!r}"
                ) from error
            units.append(
                PronunciationUnit(
                    text=char,
                    source_spans=(token.source_spans[index],),
                    phones=tuple(phone for phone in (onset, final) if phone),
                    tone=tone,
                    alphabet=PhoneAlphabet.JYUTPING,
                    native=syllable,
                )
            )
    return tuple(units)


class ToJyutpingBackend:
    name = "tojyutping"
    capabilities = BackendCapabilities(
        language=Language.CHINESE,
        dialect=ChineseDialect.CANTONESE,
        alphabet=PhoneAlphabet.JYUTPING,
        contextual=True,
        supports_projection=True,
    )

    def __init__(
        self,
        converter: Optional[JyutpingConverter] = None,
        foreign_placeholder: str = "\ue000",
    ) -> None:
        if len(foreign_placeholder) != 1:
            raise ValueError("The model placeholder must be exactly one character")
        self._converter = converter
        self._placeholder = foreign_placeholder

    def _get_converter(self) -> JyutpingConverter:
        if self._converter is None:
            from ToJyutping import get_jyutping_list

            self._converter = get_jyutping_list
        return self._converter

    def _convert(self, text: str) -> JyutpingList:
        return self._get_converter()(text)

    def predict(
        self,
        request: PronunciationRequest,
    ) -> Mapping[int, Pronunciation]:
        projection = encode_character_projection(
            request,
            self._placeholder,
            preserve_context=True,
        )
        values = self._convert(projection.text)
        if len(values) != len(projection.sources):
            raise AlignmentError(
                f"{self.name} returned {len(values)} positions for a {len(projection.sources)}-character projection"
            )

        by_token: Dict[int, Dict[int, Tuple[str, ...]]] = {}
        for position, (value, source) in enumerate(zip(values, projection.sources)):
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise AlignmentError(f"{self.name} returned an invalid entry at position {position}: {value!r}")
            char, jyutping = value
            expected_char = projection.text[position]
            if char != expected_char:
                raise AlignmentError(
                    f"{self.name} returned {char!r} for position {position}, expected {expected_char!r}"
                )
            if source is None:
                continue
            if not isinstance(jyutping, str):
                raise BackendError(f"No Cantonese pronunciation for {char!r}")
            syllables = tuple(jyutping.split())
            if not syllables or any(JYUTPING_PATTERN.fullmatch(syllable) is None for syllable in syllables):
                raise BackendError(f"Invalid Cantonese pronunciation for {char!r}: {jyutping!r}")
            token_id, char_index = source
            by_token.setdefault(token_id, {})[char_index] = syllables

        result = {}
        for token in request.target_tokens:
            pronunciations = [by_token.get(token.id, {}).get(index) for index in range(len(token.text))]
            if any(syllables is None for syllables in pronunciations):
                raise AlignmentError(f"{self.name} did not predict every character in {token.text!r}")
            result[token.id] = Pronunciation(
                token_id=token.id,
                units=_build_units(token, pronunciations, self.name),
                backend=self.name,
            )
        return result


class PyCantoneseBackend:
    name = "pycantonese"
    capabilities = BackendCapabilities(
        language=Language.CHINESE,
        dialect=ChineseDialect.CANTONESE,
        alphabet=PhoneAlphabet.JYUTPING,
    )

    @staticmethod
    def _convert(text: str) -> List[str]:
        import pycantonese

        result = []
        for chars, jyutping in pycantonese.characters_to_jyutping(text):
            if jyutping is None:
                raise BackendError(f"No Cantonese pronunciation for {chars!r}")
            result.extend(
                _parse_jyutping_sequence(
                    jyutping,
                    backend=PyCantoneseBackend.name,
                    source_chars=chars,
                    source_text=text,
                )
            )
        return result

    def predict(
        self,
        request: PronunciationRequest,
    ) -> Mapping[int, Pronunciation]:
        result = {}
        for token in request.target_tokens:
            syllables = self._convert(token.text)
            if len(syllables) != len(token.text):
                raise AlignmentError(
                    f"{self.name} returned {len(syllables)} syllables for {token.text!r} ({len(token.text)} characters)"
                )

            result[token.id] = Pronunciation(
                token_id=token.id,
                units=_build_units(token, tuple((syllable,) for syllable in syllables), self.name),
                backend=self.name,
            )
        return result
