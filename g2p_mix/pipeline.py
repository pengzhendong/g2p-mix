from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import replace
from typing import Dict, Mapping, Optional, Sequence, Union

from .backends.base import PronunciationRequest
from .errors import AlignmentError, ConfigurationError
from .models import (
    G2PResult,
    Language,
    LanguageProjection,
    OutputToken,
    PhoneAlphabet,
    Pronunciation,
    PronunciationUnit,
    Span,
    TextToken,
)
from .phonetics import (
    canonical_pinyin_phones,
    split_jyutping,
    split_pinyin,
    transcribe_arpabet_phone,
)
from .profiles import (
    ChineseProfile,
    EnglishProfile,
)
from .renderers import IpaRenderer, NativeRenderer
from .text import LosslessTokenizer, NormalizationPipeline, ProjectionBuilder, TextAnalyzer
from .transcription import IpaTranscriber, ResultTranscriber


class G2PPipeline:
    def __init__(
        self,
        chinese: ChineseProfile,
        english: Optional[EnglishProfile] = None,
        *,
        tokenizer: Optional[LosslessTokenizer] = None,
        projector: Optional[ProjectionBuilder] = None,
        output_alphabet: Optional[PhoneAlphabet] = None,
        transcriber: Optional[ResultTranscriber] = None,
    ) -> None:
        if transcriber is not None and output_alphabet is None:
            output_alphabet = transcriber.target_alphabet
        if output_alphabet not in {None, PhoneAlphabet.IPA}:
            raise ConfigurationError("Mixed-language output supports only native alphabets or IPA")
        if transcriber is not None and transcriber.target_alphabet is not output_alphabet:
            raise ConfigurationError("The transcriber target does not match output_alphabet")

        self.chinese = chinese
        self.english = english or EnglishProfile.default()
        self._tokenizer = tokenizer or LosslessTokenizer()
        self._projector = projector or ProjectionBuilder()
        self.output_alphabet = output_alphabet
        self._transcriber = transcriber or (IpaTranscriber() if output_alphabet is PhoneAlphabet.IPA else None)
        self._renderer = IpaRenderer(self._transcriber) if output_alphabet is PhoneAlphabet.IPA else NativeRenderer()

    def __call__(self, text: str) -> G2PResult:
        return self.convert(text)

    def convert(self, text: str) -> G2PResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        normalized = NormalizationPipeline(self.chinese.normalizers + self.english.normalizers).normalize(text)
        analyzer = TextAnalyzer(
            chinese_segmenter=self.chinese.segmenter,
            tokenizer=self._tokenizer,
        )
        tokens = analyzer.analyze(normalized)
        projections = {
            language: self._projector.build(tokens, target=language)
            for language in (Language.CHINESE, Language.ENGLISH)
        }

        pronunciations: Dict[int, Pronunciation] = {}
        pronunciations.update(
            self._predict(
                profile=self.chinese,
                tokens=tokens,
                projection=projections[Language.CHINESE],
                source_text=text,
            )
        )
        pronunciations.update(
            self._predict(
                profile=self.english,
                tokens=tokens,
                projection=projections[Language.ENGLISH],
                source_text=text,
            )
        )

        predicted_tokens = tuple(token for token in tokens if token.id in pronunciations)
        expected_alphabets = {
            Language.CHINESE: self.chinese.backend.capabilities.alphabet,
            Language.ENGLISH: self.english.backend.capabilities.alphabet,
        }
        for processor in self.chinese.processors:
            baseline = dict(pronunciations)
            processed = processor.process(tokens, pronunciations)
            self._validate_prediction(
                producer=type(processor).__name__,
                result=processed,
                expected_tokens=predicted_tokens,
                expected_alphabets=expected_alphabets,
                source_text=text,
            )
            self._validate_processor_identity(
                producer=type(processor).__name__,
                baseline=baseline,
                result=processed,
            )
            pronunciations = dict(processed)
        for processor in self.english.processors:
            baseline = dict(pronunciations)
            processed = processor.process(tokens, pronunciations)
            self._validate_prediction(
                producer=type(processor).__name__,
                result=processed,
                expected_tokens=predicted_tokens,
                expected_alphabets=expected_alphabets,
                source_text=text,
            )
            self._validate_processor_identity(
                producer=type(processor).__name__,
                baseline=baseline,
                result=processed,
            )
            pronunciations = dict(processed)

        result = G2PResult(
            original_text=text,
            normalized_text=normalized.text,
            tokens=tuple(
                OutputToken(
                    token=token,
                    pronunciation=pronunciations.get(token.id),
                )
                for token in tokens
            ),
            projections=projections,
        )
        if self._transcriber is not None:
            result = self._transcriber.transcribe(result)
        return replace(
            result,
            phones=self._renderer.render(result),
            output="ipa" if self.output_alphabet is PhoneAlphabet.IPA else "native",
        )

    def _predict(
        self,
        profile: Union[ChineseProfile, EnglishProfile],
        tokens: Sequence[TextToken],
        projection: LanguageProjection,
        source_text: str,
    ) -> Mapping[int, Pronunciation]:
        target_tokens = tuple(token for token in tokens if token.language is projection.target)
        if not target_tokens:
            return {}

        request = PronunciationRequest(
            tokens=tuple(tokens),
            projection=projection,
            dialect=getattr(profile, "dialect", None),
        )
        result = profile.backend.predict(request)
        self._validate_prediction(
            producer=profile.backend.name,
            result=result,
            expected_tokens=target_tokens,
            expected_alphabets={
                projection.target: profile.backend.capabilities.alphabet,
            },
            source_text=source_text,
        )
        return dict(result)

    @staticmethod
    def _validate_prediction(
        *,
        producer: str,
        result: Mapping[int, Pronunciation],
        expected_tokens: Sequence[TextToken],
        expected_alphabets: Mapping[Language, PhoneAlphabet],
        source_text: str,
    ) -> None:
        if not isinstance(result, MappingABC):
            raise AlignmentError(f"{producer} must return a mapping of token IDs to pronunciations")

        actual_keys = tuple(result.keys())
        if any(type(token_id) is not int for token_id in actual_keys):
            raise AlignmentError(f"{producer} returned a mapping key that is not an integer token ID")

        tokens_by_id = {token.id: token for token in expected_tokens}
        expected = set(tokens_by_id)
        actual = set(actual_keys)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise AlignmentError(f"{producer} coverage mismatch: missing={missing}, extra={extra}")

        for token_id, pronunciation in result.items():
            token = tokens_by_id[token_id]
            if not isinstance(pronunciation, Pronunciation):
                raise AlignmentError(f"{producer} returned a value that is not a Pronunciation for token {token_id}")
            if type(pronunciation.token_id) is not int:
                raise AlignmentError(f"{producer} returned a non-integer token_id for token {token_id}")
            if pronunciation.token_id != token_id:
                raise AlignmentError(
                    f"{producer} token_id mismatch for mapping key {token_id}: {pronunciation.token_id}"
                )
            if not isinstance(pronunciation.units, tuple):
                raise AlignmentError(f"{producer} returned non-tuple units for token {token_id}")
            if not isinstance(pronunciation.backend, str) or not pronunciation.backend:
                raise AlignmentError(f"{producer} returned an invalid backend field for token {token_id}")
            if pronunciation.confidence is not None and (
                isinstance(pronunciation.confidence, bool) or not isinstance(pronunciation.confidence, (int, float))
            ):
                raise AlignmentError(f"{producer} returned an invalid confidence field for token {token_id}")

            expected_alphabet = expected_alphabets[token.language]
            occurrences = tuple(zip(token.text, token.source_spans))
            token_span_values = set(token.source_spans)
            next_position = 0
            previous_positions = ()
            units_by_positions = {}
            for unit in pronunciation.units:
                G2PPipeline._validate_unit_structure(producer, token_id, unit)
                if unit.alphabet is not expected_alphabet:
                    raise AlignmentError(
                        f"{producer} alphabet mismatch for token {token_id}: "
                        f"expected {expected_alphabet.value}, got {unit.alphabet.value}"
                    )

                for span in unit.source_spans:
                    if span.end > len(source_text) or span not in token_span_values:
                        raise AlignmentError(
                            f"{producer} returned a foreign or out-of-range source span "
                            f"[{span.start}, {span.end}) for token {token_id}"
                        )

                desired = tuple(zip(unit.text, unit.source_spans))
                width = len(desired)
                candidate = occurrences[next_position : next_position + width]
                if len(candidate) == width and candidate == desired:
                    unit_positions = tuple(range(next_position, next_position + width))
                    next_position += width
                elif (
                    unit.alphabet is PhoneAlphabet.JYUTPING
                    and width == 1
                    and previous_positions == (next_position - 1,)
                    and occurrences[previous_positions[0]] == desired[0]
                ):
                    unit_positions = previous_positions
                else:
                    expected_text = "".join(char for char, _ in candidate)
                    expected_spans = tuple(span for _, span in candidate)
                    if len(candidate) == width and unit.source_spans == expected_spans:
                        raise AlignmentError(
                            f"{producer} unit text {unit.text!r} does not match its normalized occurrence "
                            f"{expected_text!r} for token {token_id}"
                        )
                    raise AlignmentError(
                        f"{producer} returned unordered source spans or reuses normalized position for token {token_id}"
                    )

                seen_units = units_by_positions.setdefault(unit_positions, set())
                if unit in seen_units:
                    raise AlignmentError(f"{producer} returned a duplicate pronunciation unit for token {token_id}")
                if seen_units and unit.alphabet is not PhoneAlphabet.JYUTPING:
                    raise AlignmentError(f"{producer} reuses normalized position for token {token_id}")
                seen_units.add(unit)
                previous_positions = unit_positions
                G2PPipeline._validate_unit_phonetics(producer, token_id, unit)

            if next_position != len(occurrences):
                raise AlignmentError(f"{producer} source coverage mismatch for token {token_id}")

    @staticmethod
    def _validate_unit_structure(producer: str, token_id: int, unit: object) -> None:
        if not isinstance(unit, PronunciationUnit):
            raise AlignmentError(f"{producer} returned a value that is not a PronunciationUnit for token {token_id}")
        if not isinstance(unit.text, str) or not unit.text:
            raise AlignmentError(f"{producer} returned a unit with an invalid text field for token {token_id}")
        if not isinstance(unit.source_spans, tuple) or not unit.source_spans:
            raise AlignmentError(f"{producer} returned a unit with invalid source spans for token {token_id}")
        if len(unit.text) != len(unit.source_spans):
            raise AlignmentError(f"{producer} returned unit text/source span cardinality mismatch for token {token_id}")
        if any(not isinstance(span, Span) for span in unit.source_spans):
            raise AlignmentError(f"{producer} returned a source span that is not a Span for token {token_id}")
        if not isinstance(unit.phones, tuple) or not unit.phones:
            raise AlignmentError(f"{producer} returned invalid phones for token {token_id}")
        if any(not isinstance(phone, str) or not phone for phone in unit.phones):
            raise AlignmentError(f"{producer} returned invalid phones for token {token_id}")
        if not isinstance(unit.alphabet, PhoneAlphabet):
            raise AlignmentError(f"{producer} returned an alphabet that is not a PhoneAlphabet for token {token_id}")
        if not isinstance(unit.native, str):
            raise AlignmentError(f"{producer} returned an invalid native field for token {token_id}")
        if unit.tone is not None and not isinstance(unit.tone, str):
            raise AlignmentError(f"{producer} returned an invalid tone field for token {token_id}")
        if unit.source_alphabet is not None and not isinstance(unit.source_alphabet, PhoneAlphabet):
            raise AlignmentError(f"{producer} returned an invalid source alphabet for token {token_id}")
        if not isinstance(unit.source_phones, tuple) or any(
            not isinstance(phone, str) or not phone for phone in unit.source_phones
        ):
            raise AlignmentError(f"{producer} returned invalid source phones for token {token_id}")
        if not isinstance(unit.tone_contour, tuple) or any(
            isinstance(value, bool) or not isinstance(value, int) or value not in range(1, 6)
            for value in unit.tone_contour
        ):
            raise AlignmentError(f"{producer} returned an invalid tone contour for token {token_id}")
        if not isinstance(unit.stress_marks, tuple):
            raise AlignmentError(f"{producer} returned invalid stress marks for token {token_id}")
        previous_stress_index = -1
        for mark in unit.stress_marks:
            if (
                not isinstance(mark, tuple)
                or len(mark) != 2
                or isinstance(mark[0], bool)
                or not isinstance(mark[0], int)
                or mark[0] not in range(len(unit.phones))
                or mark[0] <= previous_stress_index
                or isinstance(mark[1], bool)
                or mark[1] not in {0, 1, 2}
            ):
                raise AlignmentError(f"{producer} returned invalid stress marks for token {token_id}")
            previous_stress_index = mark[0]

    @staticmethod
    def _validate_unit_phonetics(producer, token_id, unit) -> None:
        if unit.alphabet is PhoneAlphabet.ARPABET:
            if unit.tone is not None:
                raise AlignmentError(f"{producer} returned a tone for ARPABET token {token_id}")
            if any(phone[-1].isdigit() for phone in unit.phones):
                raise AlignmentError(
                    f"{producer} returned stressed ARPABET instead of base phones for token {token_id}"
                )
            stress_by_index = dict(unit.stress_marks)
            try:
                for index, phone in enumerate(unit.phones):
                    transcribe_arpabet_phone(phone, stress=stress_by_index.get(index))
            except (TypeError, ValueError) as error:
                raise AlignmentError(f"{producer} returned invalid ARPABET phones for token {token_id}") from error
            if unit.native != " ".join(NativeRenderer().render_unit(unit)):
                raise AlignmentError(f"{producer} native/phones mismatch for ARPABET token {token_id}")
            return
        if unit.stress_marks:
            raise AlignmentError(f"{producer} returned stress marks for a non-ARPABET token {token_id}")

        try:
            if unit.alphabet is PhoneAlphabet.PINYIN:
                initial, final, tone = split_pinyin(unit.native)
                native_phones = canonical_pinyin_phones(tuple(phone for phone in (initial, final) if phone))
                unit_phones = canonical_pinyin_phones(unit.phones)
            elif unit.alphabet is PhoneAlphabet.JYUTPING:
                initial, final, tone = split_jyutping(unit.native)
                native_phones = tuple(phone for phone in (initial, final) if phone)
                unit_phones = unit.phones
            else:
                return
        except (TypeError, ValueError) as error:
            raise AlignmentError(f"{producer} returned invalid native or phones for token {token_id}") from error

        if unit.tone != tone:
            raise AlignmentError(f"{producer} tone mismatch for token {token_id}: expected {tone!r}, got {unit.tone!r}")
        if unit_phones != native_phones:
            raise AlignmentError(f"{producer} native/phones mismatch for token {token_id}")

    @staticmethod
    def _validate_processor_identity(
        *,
        producer: str,
        baseline: Mapping[int, Pronunciation],
        result: Mapping[int, Pronunciation],
    ) -> None:
        for token_id, before in baseline.items():
            after = result[token_id]
            if after.backend != before.backend:
                raise AlignmentError(
                    f"{producer} changed pronunciation backend for token {token_id}: "
                    f"{before.backend!r} -> {after.backend!r}"
                )

            before_alignment = tuple((unit.text, unit.source_spans, unit.alphabet) for unit in before.units)
            after_alignment = tuple((unit.text, unit.source_spans, unit.alphabet) for unit in after.units)
            if after_alignment != before_alignment:
                raise AlignmentError(f"{producer} changed unit alignment identity for token {token_id}")
