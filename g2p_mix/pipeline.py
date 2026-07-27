from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Union

from .backends.base import PronunciationRequest
from .errors import AlignmentError
from .models import (
    G2PResult,
    Language,
    LanguageProjection,
    OutputToken,
    Pronunciation,
    TextToken,
)
from .normalizers import NormalizationPipeline
from .profiles import (
    CantoneseProfile,
    ChineseProfile,
    EnglishProfile,
    MandarinProfile,
)
from .projection import ProjectionBuilder
from .tokenizer import LosslessTokenizer, TextAnalyzer


class MixedG2P:
    def __init__(
        self,
        chinese: ChineseProfile,
        english: Optional[EnglishProfile] = None,
        *,
        tokenizer: Optional[LosslessTokenizer] = None,
        projector: Optional[ProjectionBuilder] = None,
    ) -> None:
        self.chinese = chinese
        self.english = english or EnglishProfile.default()
        self._tokenizer = tokenizer or LosslessTokenizer()
        self._projector = projector or ProjectionBuilder()

    @classmethod
    def mandarin(
        cls,
        *,
        chinese_backend=None,
        english_backend=None,
        tone_sandhi: bool = True,
    ) -> "MixedG2P":
        return cls(
            chinese=MandarinProfile(
                backend=chinese_backend,
                tone_sandhi=tone_sandhi,
            ),
            english=(EnglishProfile(english_backend) if english_backend is not None else None),
        )

    @classmethod
    def cantonese(
        cls,
        *,
        chinese_backend=None,
        english_backend=None,
        traditional: bool = True,
        tagset: str = "universal",
    ) -> "MixedG2P":
        return cls(
            chinese=CantoneseProfile(
                backend=chinese_backend,
                traditional=traditional,
                tagset=tagset,
            ),
            english=(EnglishProfile(english_backend) if english_backend is not None else None),
        )

    def __call__(self, text: str) -> G2PResult:
        return self.convert(text)

    def convert(self, text: str) -> G2PResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        normalized = NormalizationPipeline(self.chinese.normalizers).normalize(text)
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
            )
        )
        pronunciations.update(
            self._predict(
                profile=self.english,
                tokens=tokens,
                projection=projections[Language.ENGLISH],
            )
        )

        for processor in self.chinese.processors:
            pronunciations = dict(processor.process(tokens, pronunciations))
        for processor in self.english.processors:
            pronunciations = dict(processor.process(tokens, pronunciations))

        return G2PResult(
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

    def _predict(
        self,
        profile: Union[ChineseProfile, EnglishProfile],
        tokens: Sequence[TextToken],
        projection: LanguageProjection,
    ) -> Mapping[int, Pronunciation]:
        target_tokens = tuple(token for token in tokens if token.language is projection.target)
        if not target_tokens:
            return {}

        request = PronunciationRequest(
            tokens=tuple(tokens),
            projection=projection,
            dialect=getattr(profile, "dialect", None),
        )
        result = dict(profile.backend.predict(request))
        expected = {token.id for token in target_tokens}
        actual = set(result)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise AlignmentError(f"{profile.backend.name} coverage mismatch: missing={missing}, extra={extra}")
        return result
