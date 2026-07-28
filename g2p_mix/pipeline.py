from __future__ import annotations

from dataclasses import replace
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

from .backends.base import PronunciationRequest
from .errors import ConfigurationError
from .models import (
    G2PResult,
    Language,
    LanguageProjection,
    OutputToken,
    PhoneAlphabet,
    Pronunciation,
    TextToken,
)
from .profiles import (
    ChineseProfile,
    EnglishProfile,
)
from .renderers import IpaRenderer, NativeRenderer
from .text import LosslessTokenizer, NormalizationPipeline, ProjectionBuilder, TextAnalyzer
from .transcription import IpaTranscriber, ResultTranscriber
from .validation import validate_prediction, validate_processor_identity


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
        self._normalizer = NormalizationPipeline(self.chinese.normalizers + self.english.normalizers)
        self._analyzer = TextAnalyzer(
            chinese_segmenter=self.chinese.segmenter,
            tokenizer=self._tokenizer,
        )
        self.output_alphabet = output_alphabet
        self._transcriber = transcriber or (IpaTranscriber() if output_alphabet is PhoneAlphabet.IPA else None)
        self._renderer = IpaRenderer() if output_alphabet is PhoneAlphabet.IPA else NativeRenderer()

    def __call__(self, text: str) -> G2PResult:
        return self.convert(text)

    def convert(self, text: str) -> G2PResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        normalized = self._normalizer.normalize(text)
        tokens = self._analyzer.analyze(normalized)
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
        for processor in self.chinese.processors + self.english.processors:
            baseline = dict(pronunciations)
            processed = processor.process(tokens, pronunciations)
            validate_prediction(
                producer=type(processor).__name__,
                result=processed,
                expected_tokens=predicted_tokens,
                expected_alphabets=expected_alphabets,
                source_text=text,
            )
            validate_processor_identity(
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
            warnings=self._unknown_warnings(
                pronunciations,
                source_text=text,
            ),
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
        validate_prediction(
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
    def _unknown_warnings(
        pronunciations: Mapping[int, Pronunciation],
        *,
        source_text: str,
    ) -> Tuple[str, ...]:
        warnings = []
        for pronunciation in pronunciations.values():
            for unit in pronunciation.units:
                if not unit.is_unknown:
                    continue
                spans = ", ".join(f"[{span.start}, {span.end})" for span in unit.source_spans)
                source = "".join(span.slice(source_text) for span in unit.source_spans)
                warnings.append(
                    f"{pronunciation.backend} preserved unknown character "
                    f"{unit.text!r} (source={source!r}, spans={spans}) "
                    "without phones"
                )
        return tuple(warnings)
