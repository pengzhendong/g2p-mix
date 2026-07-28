from __future__ import annotations

from typing import Literal

from .backends import (
    EnglishBackend,
    G2PWBackend,
    PyCantoneseBackend,
    PypinyinBackend,
    ToJyutpingBackend,
)
from .errors import ConfigurationError
from .models import G2PResult, PhoneAlphabet
from .pipeline import G2PPipeline
from .profiles import CantoneseProfile, EnglishProfile, MandarinProfile
from .similarity import PhoneticMatcher, SimilarityResult

Mode = Literal["mandarin", "cantonese"]
Output = Literal["native", "ipa"]

_CHINESE_BACKENDS = {
    "mandarin": {
        "pypinyin": PypinyinBackend,
        "g2pw": G2PWBackend,
    },
    "cantonese": {
        "tojyutping": ToJyutpingBackend,
        "pycantonese": PyCantoneseBackend,
    },
}
_DEFAULT_BACKENDS = {
    "mandarin": "pypinyin",
    "cantonese": "tojyutping",
}
_ENGLISH_BACKENDS = {
    "g2p-en": EnglishBackend,
}


class G2P:
    """Simple public interface for Mandarin-English or Cantonese-English G2P."""

    def __init__(
        self,
        mode: Mode = "mandarin",
        *,
        output: Output = "native",
        backend=None,
        english_backend=None,
        tone_sandhi: bool = True,
        traditional: bool = True,
    ) -> None:
        if mode not in _CHINESE_BACKENDS:
            raise ConfigurationError("mode must be 'mandarin' or 'cantonese'")
        if output not in {"native", "ipa"}:
            raise ConfigurationError("output must be 'native' or 'ipa'")

        chinese_backend = self._resolve_backend(mode, backend)
        resolved_english_backend = self._resolve_english_backend(english_backend)
        if mode == "mandarin":
            chinese = MandarinProfile(
                backend=chinese_backend,
                tone_sandhi=tone_sandhi,
            )
        else:
            chinese = CantoneseProfile(
                backend=chinese_backend,
                traditional=traditional,
            )

        self.mode = mode
        self.output = output
        self.backend = chinese_backend.name
        self.english_backend = resolved_english_backend.name
        self._pipeline = G2PPipeline(
            chinese=chinese,
            english=EnglishProfile(resolved_english_backend),
            output_alphabet=PhoneAlphabet.IPA if output == "ipa" else None,
        )
        self._matcher = None

    def __call__(self, text: str) -> G2PResult:
        return self._pipeline(text)

    def compare(self, left: str, right: str) -> SimilarityResult:
        if not isinstance(left, str) or not isinstance(right, str):
            raise TypeError("left and right must be strings")
        if self._matcher is None:
            self._matcher = PhoneticMatcher()
        return self._matcher.compare(self(left), self(right))

    @staticmethod
    def _resolve_backend(mode: Mode, backend):
        if backend is None:
            backend = _DEFAULT_BACKENDS[mode]
        if not isinstance(backend, str):
            return backend
        choices = _CHINESE_BACKENDS[mode]
        try:
            return choices[backend]()
        except KeyError as error:
            available = ", ".join(choices)
            raise ConfigurationError(f"backend for {mode} must be one of: {available}") from error

    @staticmethod
    def _resolve_english_backend(backend):
        if backend is None:
            backend = "g2p-en"
        if not isinstance(backend, str):
            return backend
        try:
            return _ENGLISH_BACKENDS[backend]()
        except KeyError as error:
            available = ", ".join(_ENGLISH_BACKENDS)
            raise ConfigurationError(f"English backend must be one of: {available}") from error
