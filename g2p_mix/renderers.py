from __future__ import annotations

from typing import Tuple

from .errors import RenderingError, TranscriptionError
from .models import G2PResult, PhoneAlphabet, PronunciationUnit
from .phonetics import render_tone_contour
from .transcription import IpaTranscriber


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
        return tuple(phone for unit in result.units for phone in self.render_unit(unit))


class IpaRenderer:
    def __init__(self, transcriber=None) -> None:
        self._transcriber = transcriber or IpaTranscriber()

    def render_unit(self, unit: PronunciationUnit) -> Tuple[str, ...]:
        try:
            transcribed = self._transcriber.transcribe_unit(unit)
        except TranscriptionError as error:
            spans = [(span.start, span.end) for span in unit.source_spans]
            raise RenderingError(
                "Could not render pronunciation unit as IPA: "
                f"alphabet={unit.alphabet.value!r}, native={unit.native!r}, "
                f"text={unit.text!r}, source_spans={spans!r}"
            ) from (error.__cause__ or error)

        phones = list(transcribed.phones)
        for index, stress in transcribed.stress_marks:
            if stress in {1, 2}:
                phones[index] = {1: "ˈ", 2: "ˌ"}[stress] + phones[index]
        if transcribed.tone_contour:
            phones[-1] += render_tone_contour(transcribed.tone_contour)
        return tuple(phones)

    def render(self, result: G2PResult) -> Tuple[str, ...]:
        return tuple(phone for unit in result.units for phone in self.render_unit(unit))
