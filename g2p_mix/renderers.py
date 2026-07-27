from __future__ import annotations

from typing import Tuple

from .errors import RenderingError, UnsupportedFeatureError
from .models import G2PResult, PhoneAlphabet, PronunciationUnit
from .phonetics import arpabet_to_ipa, canonical_pinyin_phones, pinyin_to_ipa


class NativeRenderer:
    def render_unit(self, unit: PronunciationUnit) -> Tuple[str, ...]:
        if unit.alphabet in {PhoneAlphabet.PINYIN, PhoneAlphabet.JYUTPING} and unit.tone and unit.phones:
            return unit.phones[:-1] + (unit.phones[-1] + unit.tone,)
        return unit.phones

    def render(self, result: G2PResult) -> Tuple[str, ...]:
        return tuple(phone for unit in result.units for phone in self.render_unit(unit))


class IpaRenderer:
    def render_unit(self, unit: PronunciationUnit) -> Tuple[str, ...]:
        if unit.alphabet is PhoneAlphabet.JYUTPING:
            raise UnsupportedFeatureError("Cantonese IPA rendering requires an explicit Jyutping-to-IPA map")
        try:
            if unit.alphabet is PhoneAlphabet.ARPABET:
                return tuple(arpabet_to_ipa(phone) for phone in unit.phones)
            if unit.alphabet is PhoneAlphabet.PINYIN:
                if unit.tone is None:
                    raise ValueError("Pinyin IPA rendering requires a tone")
                initial, final = canonical_pinyin_phones(unit.phones)
                return pinyin_to_ipa(initial, final, unit.tone)
            return unit.phones
        except (KeyError, TypeError, ValueError) as error:
            spans = [(span.start, span.end) for span in unit.source_spans]
            raise RenderingError(
                "Could not render pronunciation unit as IPA: "
                f"alphabet={unit.alphabet.value!r}, native={unit.native!r}, "
                f"text={unit.text!r}, source_spans={spans!r}"
            ) from error

    def render(self, result: G2PResult) -> Tuple[str, ...]:
        return tuple(phone for unit in result.units for phone in self.render_unit(unit))
