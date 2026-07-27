from __future__ import annotations

from typing import Tuple

from .errors import UnsupportedFeatureError
from .models import G2PResult, PhoneAlphabet, PronunciationUnit
from .phonetics import arpabet_to_ipa, pinyin_to_ipa


class NativeRenderer:
    def render_unit(self, unit: PronunciationUnit) -> Tuple[str, ...]:
        if unit.tone and unit.phones:
            return unit.phones[:-1] + (unit.phones[-1] + unit.tone,)
        return unit.phones

    def render(self, result: G2PResult) -> Tuple[str, ...]:
        return tuple(phone for unit in result.units for phone in self.render_unit(unit))


class IpaRenderer:
    def render_unit(self, unit: PronunciationUnit) -> Tuple[str, ...]:
        if unit.alphabet is PhoneAlphabet.ARPABET:
            return tuple(arpabet_to_ipa(phone) for phone in unit.phones)
        if unit.alphabet is PhoneAlphabet.PINYIN:
            initial = unit.phones[0] if len(unit.phones) == 2 else ""
            final = unit.phones[-1]
            return pinyin_to_ipa(initial, final, unit.tone or "5")
        if unit.alphabet is PhoneAlphabet.JYUTPING:
            raise UnsupportedFeatureError("Cantonese IPA rendering requires an explicit Jyutping-to-IPA map")
        return unit.phones

    def render(self, result: G2PResult) -> Tuple[str, ...]:
        return tuple(phone for unit in result.units for phone in self.render_unit(unit))
