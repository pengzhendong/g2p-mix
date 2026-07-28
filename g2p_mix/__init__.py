from importlib.metadata import PackageNotFoundError, version

from .backends import (
    EnglishBackend,
    G2PWBackend,
    PyCantoneseBackend,
    PypinyinBackend,
    ToJyutpingBackend,
)
from .errors import (
    AlignmentError,
    BackendError,
    ConfigurationError,
    G2PError,
    RenderingError,
    SimilarityError,
    TranscriptionError,
    UnsupportedFeatureError,
)
from .lexicons import MandarinLexicon, PronunciationLookup
from .models import (
    Boundary,
    ChineseDialect,
    G2PResult,
    Language,
    PhoneAlphabet,
    Pronunciation,
    PronunciationUnit,
    Span,
    TextToken,
)
from .pipeline import MixedG2P
from .profiles import CantoneseProfile, EnglishProfile, MandarinProfile
from .renderers import IpaRenderer, NativeRenderer
from .similarity import (
    AlignmentStep,
    DistanceResult,
    EditOperation,
    PanPhonDistanceBackend,
    PhoneticDistanceBackend,
    PhoneticMatcher,
    SimilarityResult,
)
from .transcription import IpaTranscriber, ResultTranscriber

try:
    __version__ = version("g2p-mix")
except (PackageNotFoundError, TypeError):
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "MixedG2P",
    "MandarinProfile",
    "CantoneseProfile",
    "EnglishProfile",
    "PypinyinBackend",
    "G2PWBackend",
    "ToJyutpingBackend",
    "PyCantoneseBackend",
    "EnglishBackend",
    "NativeRenderer",
    "IpaRenderer",
    "IpaTranscriber",
    "ResultTranscriber",
    "PhoneticMatcher",
    "PhoneticDistanceBackend",
    "PanPhonDistanceBackend",
    "DistanceResult",
    "SimilarityResult",
    "AlignmentStep",
    "EditOperation",
    "MandarinLexicon",
    "PronunciationLookup",
    "G2PResult",
    "TextToken",
    "Pronunciation",
    "PronunciationUnit",
    "Span",
    "Language",
    "ChineseDialect",
    "PhoneAlphabet",
    "Boundary",
    "G2PError",
    "ConfigurationError",
    "BackendError",
    "AlignmentError",
    "RenderingError",
    "TranscriptionError",
    "SimilarityError",
    "UnsupportedFeatureError",
]
