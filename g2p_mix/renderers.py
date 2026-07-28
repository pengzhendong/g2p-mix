from __future__ import annotations

from collections.abc import Callable
from typing import Tuple

from .errors import RenderingError
from .models import G2PResult, PhoneAlphabet, PronunciationUnit
from .phonetics import render_tone_contour


def _render_result(
    result: G2PResult,
    render_unit: Callable[[PronunciationUnit], Tuple[str, ...]],
) -> Tuple[str, ...]:
    return tuple(phone for unit in result.units for phone in render_unit(unit))


class NativeRenderer:
    def render_unit(self, unit: PronunciationUnit) -> Tuple[str, ...]:
        if unit.alphabet in {PhoneAlphabet.PINYIN, PhoneAlphabet.JYUTPING} and unit.tone and unit.phones:
            return unit.phones[:-1] + (unit.phones[-1] + unit.tone,)
        if unit.alphabet is PhoneAlphabet.ARPABET:
            stress_by_index = dict(unit.stress_marks)
            return tuple(
                phone + str(stress_by_index[index]) if index in stress_by_index else phone
                for index, phone in enumerate(unit.phones)
            )
        return unit.phones

    def render(self, result: G2PResult) -> Tuple[str, ...]:
        return _render_result(result, self.render_unit)


class IpaRenderer:
    def render_unit(self, unit: PronunciationUnit) -> Tuple[str, ...]:
        if unit.alphabet is not PhoneAlphabet.IPA:
            spans = [(span.start, span.end) for span in unit.source_spans]
            raise RenderingError(
                "IPA rendering requires an IPA pronunciation unit: "
                f"alphabet={unit.alphabet.value!r}, native={unit.native!r}, "
                f"text={unit.text!r}, source_spans={spans!r}"
            )

        phones = list(unit.phones)
        for index, stress in unit.stress_marks:
            if stress in {1, 2}:
                phones[index] = {1: "ˈ", 2: "ˌ"}[stress] + phones[index]
        if unit.tone_contour:
            phones[-1] += render_tone_contour(unit.tone_contour)
        return tuple(phones)

    def render(self, result: G2PResult) -> Tuple[str, ...]:
        return _render_result(result, self.render_unit)
