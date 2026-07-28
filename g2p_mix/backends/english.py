from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Callable, List, Mapping, Optional, Protocol, Sequence, Tuple

from ..errors import BackendError
from ..models import Language, PhoneAlphabet, ProjectionKind, Pronunciation, PronunciationUnit
from ..phonetics import split_arpabet_phone
from ..resources import ensure_bundled_nltk_data, load_cmudict, load_json
from ..text.latin import fold_english_spelling
from .base import BackendCapabilities, PronunciationRequest

APOSTROPHE_TRANSLATION = str.maketrans({"’": "'", "‘": "'"})
CLITIC_PRONUNCIATIONS = {
    "n't": ("AH0", "N", "T"),
}


class EnglishLexicon(Protocol):
    def lookup(self, word: str) -> Tuple[Tuple[str, ...], ...]:
        pass


class EnglishContextAnalyzer(Protocol):
    def analyze(self, request: PronunciationRequest) -> Mapping[int, str]:
        pass


class EnglishPronunciationResolver(Protocol):
    def needs_context(self, words: Sequence[str]) -> bool:
        pass

    def resolve(
        self,
        word: str,
        candidates: Sequence[Sequence[str]],
        pos: Optional[str],
    ) -> Sequence[str]:
        pass


class EnglishOovPredictor(Protocol):
    def predict(self, word: str) -> Sequence[str]:
        pass


class CmuLexicon:
    """Lazy CMUdict adapter that exposes every pronunciation candidate."""

    def __init__(self, dictionary: Optional[Mapping[str, Sequence[Sequence[str]]]] = None) -> None:
        self._dictionary = dictionary

    @property
    def dictionary(self) -> Mapping[str, Sequence[Sequence[str]]]:
        if self._dictionary is None:
            self._dictionary = load_cmudict()
        return self._dictionary

    def lookup(self, word: str) -> Tuple[Tuple[str, ...], ...]:
        return tuple(tuple(pronunciation) for pronunciation in self.dictionary.get(word.lower(), ()))


class NltkContextAnalyzer:
    """Tag the English projection once so every target token shares context."""

    def analyze(self, request: PronunciationRequest) -> Mapping[int, str]:
        tagged_text = []
        target_ids = []
        for projected in request.projection.tokens:
            if projected.kind is ProjectionKind.TARGET:
                tagged_text.append(projected.text)
                target_ids.append(projected.source_ids[0])
            elif projected.kind is ProjectionKind.PLACEHOLDER:
                tagged_text.append(projected.text)
                target_ids.append(None)
            elif projected.text and not projected.text.isspace():
                tagged_text.append(projected.text)
                target_ids.append(None)

        if not tagged_text:
            return {}
        ensure_bundled_nltk_data()
        try:
            import nltk

            tagged = nltk.pos_tag(tagged_text)
        except (ImportError, LookupError, OSError) as error:
            raise BackendError("English POS resources are unavailable") from error
        except Exception as error:
            raise BackendError("English context analysis failed") from error

        if len(tagged) != len(tagged_text):
            raise BackendError(
                f"English context analyzer returned {len(tagged)} tags for {len(tagged_text)} projection tokens"
            )
        result = {}
        for text, token_id, tagged_item in zip(tagged_text, target_ids, tagged):
            if (
                not isinstance(tagged_item, tuple)
                or len(tagged_item) != 2
                or tagged_item[0] != text
                or not isinstance(tagged_item[1], str)
                or not tagged_item[1]
            ):
                raise BackendError(f"English context analyzer returned a malformed tag for {text!r}")
            if token_id is not None:
                result[token_id] = tagged_item[1]
        return result


@dataclass(frozen=True)
class HomographRule:
    matching_pronunciation: Tuple[str, ...]
    other_pronunciation: Tuple[str, ...]
    pos_prefix: str


@lru_cache(maxsize=1)
def load_g2p_en_homographs() -> Mapping[str, HomographRule]:
    """Load g2p-en's data and apply project-reviewed corrections."""

    try:
        resource = files("g2p_en").joinpath("homographs.en")
        lines = resource.read_text(encoding="utf-8").splitlines()
    except (ImportError, ModuleNotFoundError, OSError) as error:
        raise BackendError("English homograph rules are unavailable") from error

    rules = {}
    try:
        for line in lines:
            if not line or line.startswith("#"):
                continue
            word, matching, other, pos_prefix = line.split("|")
            rules[word.lower()] = HomographRule(
                matching_pronunciation=tuple(matching.split()),
                other_pronunciation=tuple(other.split()),
                pos_prefix=pos_prefix,
            )
    except ValueError as error:
        raise BackendError("English homograph rules are malformed") from error

    try:
        overrides = load_json("english_homograph_overrides.json")
        if not isinstance(overrides, dict):
            raise TypeError("overrides must be an object")
        for word, rule in overrides.items():
            if not isinstance(word, str) or not word or not isinstance(rule, dict):
                raise TypeError("override keys and values must be non-empty words and objects")
            matching = rule["matching_pronunciation"]
            other = rule["other_pronunciation"]
            pos_prefix = rule["pos_prefix"]
            if (
                not isinstance(matching, list)
                or not matching
                or any(not isinstance(phone, str) or not phone for phone in matching)
                or not isinstance(other, list)
                or not other
                or any(not isinstance(phone, str) or not phone for phone in other)
                or not isinstance(pos_prefix, str)
                or not pos_prefix
            ):
                raise TypeError("override pronunciations and POS prefixes must be non-empty strings")
            rules[word.lower()] = HomographRule(
                matching_pronunciation=tuple(matching),
                other_pronunciation=tuple(other),
                pos_prefix=pos_prefix,
            )
    except (KeyError, TypeError, ValueError) as error:
        raise BackendError("English homograph overrides are malformed") from error
    return rules


class PosHomographResolver:
    """Resolve covered homographs by POS and otherwise preserve lexicon priority."""

    def __init__(self, rules: Optional[Mapping[str, HomographRule]] = None) -> None:
        self._rules = rules

    @property
    def rules(self) -> Mapping[str, HomographRule]:
        if self._rules is None:
            self._rules = load_g2p_en_homographs()
        return self._rules

    def needs_context(self, words: Sequence[str]) -> bool:
        return any(word.lower() in self.rules for word in words)

    def resolve(
        self,
        word: str,
        candidates: Sequence[Sequence[str]],
        pos: Optional[str],
    ) -> Sequence[str]:
        rule = self.rules.get(word.lower())
        if rule is not None:
            if pos is not None and pos.startswith(rule.pos_prefix):
                return rule.matching_pronunciation
            return rule.other_pronunciation
        if not candidates:
            raise BackendError(f"No English lexicon pronunciation for {word!r}")
        return candidates[0]


class G2pEnOovPredictor:
    """Lazy neural fallback used only after lexicon and segmentation misses."""

    def __init__(self, predictor: Optional[Callable[[str], Sequence[str]]] = None) -> None:
        self._predictor = predictor

    def _ensure_predictor(self) -> Callable[[str], Sequence[str]]:
        if self._predictor is None:
            try:
                ensure_bundled_nltk_data()
                import g2p_en

                self._predictor = g2p_en.G2p().predict
            except BackendError:
                raise
            except Exception as error:
                raise BackendError("English OOV pronunciation resources are unavailable; install g2p-en") from error
        return self._predictor

    def predict(self, word: str) -> Sequence[str]:
        try:
            phones = self._ensure_predictor()(word)
            if isinstance(phones, (str, bytes)):
                raise TypeError("predictor returned text instead of a sequence of phones")
            result = list(phones)
            if not result or any(not isinstance(phone, str) or not phone.strip() for phone in result):
                raise ValueError("predictor returned an empty or non-string phone")
            return result
        except BackendError:
            raise
        except (TypeError, ValueError) as error:
            raise BackendError(f"English OOV predictor returned malformed phones for {word!r}") from error
        except Exception as error:
            raise BackendError(f"English backend failed to pronounce {word!r}") from error


class EnglishBackend:
    name = "cmudict-g2p-en"
    capabilities = BackendCapabilities(
        language=Language.ENGLISH,
        alphabet=PhoneAlphabet.ARPABET,
        ascii_latin_only=True,
    )

    def __init__(
        self,
        *,
        lexicon: Optional[EnglishLexicon] = None,
        context_analyzer: Optional[EnglishContextAnalyzer] = None,
        resolver: Optional[EnglishPronunciationResolver] = None,
        oov_predictor: Optional[EnglishOovPredictor] = None,
        segmenter: Optional[Callable[[str], Sequence[str]]] = None,
    ) -> None:
        self._lexicon = lexicon or CmuLexicon()
        self._context_analyzer = context_analyzer or NltkContextAnalyzer()
        self._resolver = resolver or PosHomographResolver()
        self._oov_predictor = oov_predictor or G2pEnOovPredictor()
        self._segmenter = segmenter

    def _ensure_segmenter(self) -> Callable[[str], Sequence[str]]:
        if self._segmenter is None:
            try:
                import wordsegment

                wordsegment.load()
                self._segmenter = wordsegment.segment
            except Exception as error:
                raise BackendError("English OOV segmentation requires wordsegment") from error
        return self._segmenter

    def _character(self, char: str) -> List[str]:
        char = char.lower()
        candidates = self._lexicon.lookup(char)
        if len(char) != 1 or not candidates:
            raise BackendError(f"Cannot spell English character {char!r}")
        index = 1 if char == "a" and len(candidates) > 1 else 0
        return list(candidates[index])

    def _abbreviation(self, word: str) -> List[str]:
        return [phone for char in word for phone in self._character(char)]

    def _normalize(self, word: str) -> str:
        try:
            return fold_english_spelling(word.translate(APOSTROPHE_TRANSLATION))
        except ValueError as error:
            raise BackendError(f"Cannot normalize English spelling {word!r}") from error

    def convert(self, word: str, *, pos: Optional[str] = None) -> List[str]:
        return self._convert_normalized(self._normalize(word), original=word, pos=pos)

    def _convert_normalized(
        self,
        word: str,
        *,
        original: str,
        pos: Optional[str] = None,
    ) -> List[str]:
        if word.lower() in CLITIC_PRONUNCIATIONS:
            return list(CLITIC_PRONUNCIATIONS[word.lower()])
        if word.isalpha() and word.isupper() and len(word) <= 3:
            return self._abbreviation(word)

        candidates = self._lexicon.lookup(word)
        if candidates:
            return list(self._resolver.resolve(word, candidates, pos))
        if word.isalpha() and ((word.islower() and len(word) <= 3) or (word.isupper() and len(word) <= 4)):
            return self._abbreviation(word)

        try:
            raw_segments = self._ensure_segmenter()(word.lower())
            if isinstance(raw_segments, (str, bytes)):
                raise TypeError("segmenter returned text instead of a sequence of segments")
            segments = list(raw_segments)
            if not segments or any(not isinstance(segment, str) or not segment for segment in segments):
                raise ValueError("segmenter returned an empty or non-string segment")
        except BackendError:
            raise
        except Exception as error:
            raise BackendError(f"English backend failed to segment {original!r}") from error

        if len(segments) == 1:
            return list(self._oov_predictor.predict(segments[0]))

        phones = []
        for segment in segments:
            if len(segment) == 1:
                phones.extend(self._character(segment))
            else:
                phones.extend(self._convert_normalized(segment, original=segment))
        return phones

    def predict(
        self,
        request: PronunciationRequest,
    ) -> Mapping[int, Pronunciation]:
        target_tokens = request.target_tokens
        normalized_words = tuple(self._normalize(token.text) for token in target_tokens)
        if self._resolver.needs_context(normalized_words):
            try:
                pos_by_id = self._context_analyzer.analyze(request)
            except BackendError:
                raise
            except Exception as error:
                raise BackendError("English context analysis failed") from error
        else:
            pos_by_id = {token.id: None for token in target_tokens}
        if set(pos_by_id) != {token.id for token in target_tokens}:
            raise BackendError("English context analyzer did not cover every target token")

        result = {}
        for token, normalized_word in zip(target_tokens, normalized_words):
            rendered_phones = tuple(
                self._convert_normalized(
                    normalized_word,
                    original=token.text,
                    pos=pos_by_id[token.id],
                )
            )
            if not rendered_phones:
                raise BackendError(f"English backend returned no phones for {token.text!r}")
            phones = []
            stress_marks = []
            try:
                for rendered_phone in rendered_phones:
                    phone, stress = split_arpabet_phone(rendered_phone)
                    if stress is not None:
                        stress_marks.append((len(phones), stress))
                    phones.append(phone)
            except (TypeError, ValueError) as error:
                raise BackendError(f"English backend returned malformed ARPABET for {token.text!r}") from error
            unit = PronunciationUnit(
                text=token.text,
                source_spans=token.source_spans,
                phones=tuple(phones),
                alphabet=PhoneAlphabet.ARPABET,
                native=" ".join(rendered_phones),
                stress_marks=tuple(stress_marks),
            )
            result[token.id] = Pronunciation(
                token_id=token.id,
                units=(unit,),
                backend=self.name,
            )
        return result
