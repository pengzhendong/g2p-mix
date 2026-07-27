from __future__ import annotations

import re
from typing import List, Mapping

from ..errors import AlignmentError, BackendError
from ..models import (
    ChineseDialect,
    Language,
    PhoneAlphabet,
    Pronunciation,
    PronunciationUnit,
)
from ..phonetics import split_jyutping
from .base import BackendCapabilities, PronunciationRequest


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
            result.extend(re.findall(r"[a-z]+[1-6]", jyutping))
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

            units = []
            for index, (char, syllable) in enumerate(zip(token.text, syllables)):
                onset, final, tone = split_jyutping(syllable)
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
            result[token.id] = Pronunciation(
                token_id=token.id,
                units=tuple(units),
                backend=self.name,
            )
        return result
