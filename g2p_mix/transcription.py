from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from .errors import TranscriptionError
from .models import (
    G2PResult,
    OutputToken,
    PhoneAlphabet,
    Pronunciation,
    PronunciationUnit,
)
from .phonetics import (
    canonical_pinyin_phones,
    tone_contour,
    transcribe_arpabet_phone,
    transcribe_jyutping,
    transcribe_pinyin,
)


class ResultTranscriber(Protocol):
    target_alphabet: PhoneAlphabet

    def transcribe(self, result: G2PResult) -> G2PResult:
        pass


class IpaTranscriber:
    target_alphabet = PhoneAlphabet.IPA

    def transcribe_unit(self, unit: PronunciationUnit) -> PronunciationUnit:
        if unit.alphabet is PhoneAlphabet.IPA:
            return unit

        try:
            if unit.alphabet is PhoneAlphabet.PINYIN:
                initial, final = canonical_pinyin_phones(unit.phones)
                phones = transcribe_pinyin(initial, final)
                contour = tone_contour("cmn", self._require_tone(unit))
                stress_marks = ()
            elif unit.alphabet is PhoneAlphabet.JYUTPING:
                onset, final = self._onset_and_final(unit)
                phones = transcribe_jyutping(onset, final)
                contour = tone_contour("yue-HK", self._require_tone(unit))
                stress_marks = ()
            elif unit.alphabet is PhoneAlphabet.ARPABET:
                phones, stress_marks = self._transcribe_arpabet(unit)
                contour = ()
            else:
                raise ValueError(f"Unsupported source alphabet: {unit.alphabet.value}")
        except (KeyError, TypeError, ValueError) as error:
            spans = [(span.start, span.end) for span in unit.source_spans]
            raise TranscriptionError(
                "Could not transcribe pronunciation unit as IPA: "
                f"alphabet={unit.alphabet.value!r}, native={unit.native!r}, "
                f"text={unit.text!r}, source_spans={spans!r}"
            ) from error

        return replace(
            unit,
            phones=phones,
            alphabet=PhoneAlphabet.IPA,
            source_alphabet=unit.source_alphabet or unit.alphabet,
            source_phones=unit.source_phones or unit.phones,
            tone_contour=contour,
            stress_marks=stress_marks,
        )

    def transcribe_pronunciation(self, pronunciation: Pronunciation) -> Pronunciation:
        return replace(
            pronunciation,
            units=tuple(self.transcribe_unit(unit) for unit in pronunciation.units),
        )

    def transcribe(self, result: G2PResult) -> G2PResult:
        return replace(
            result,
            tokens=tuple(
                OutputToken(
                    token=output.token,
                    pronunciation=(
                        self.transcribe_pronunciation(output.pronunciation)
                        if output.pronunciation is not None
                        else None
                    ),
                )
                for output in result.tokens
            ),
        )

    @staticmethod
    def _require_tone(unit: PronunciationUnit) -> str:
        if unit.tone is None:
            raise ValueError(f"{unit.alphabet.value} IPA transcription requires a tone")
        return unit.tone

    @staticmethod
    def _onset_and_final(unit: PronunciationUnit) -> tuple[str, str]:
        if len(unit.phones) == 1:
            return "", unit.phones[0]
        if len(unit.phones) == 2:
            return unit.phones
        raise ValueError(f"Invalid Jyutping phones: {unit.phones!r}")

    @staticmethod
    def _transcribe_arpabet(
        unit: PronunciationUnit,
    ) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
        phones = []
        stress_marks = []
        for source_phone in unit.phones:
            segments, stress = transcribe_arpabet_phone(source_phone)
            if stress in {1, 2}:
                stress_marks.append((len(phones), stress))
            phones.extend(segments)
        return tuple(phones), tuple(stress_marks)
