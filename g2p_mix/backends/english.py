from __future__ import annotations

from typing import Callable, List, Mapping, Optional, Sequence

from ..errors import BackendError
from ..models import Language, PhoneAlphabet, Pronunciation, PronunciationUnit
from ..phonetics import split_arpabet_phone
from ..resources import ensure_bundled_nltk_data, load_cmudict
from .base import BackendCapabilities, PronunciationRequest

APOSTROPHE_TRANSLATION = str.maketrans({"’": "'", "‘": "'"})
CLITIC_PRONUNCIATIONS = {
    "n't": ("AH0", "N", "T"),
}


class EnglishBackend:
    name = "cmudict-g2p-en"
    capabilities = BackendCapabilities(
        language=Language.ENGLISH,
        alphabet=PhoneAlphabet.ARPABET,
    )

    def __init__(
        self,
        dictionary=None,
        segmenter: Optional[Callable[[str], Sequence[str]]] = None,
        predictor: Optional[Callable[[str], Sequence[str]]] = None,
    ) -> None:
        self._dictionary = dictionary
        self._segmenter = segmenter
        self._predictor = predictor

    @property
    def dictionary(self):
        if self._dictionary is None:
            self._dictionary = load_cmudict()
        return self._dictionary

    def _ensure_fallbacks(self) -> None:
        try:
            if self._segmenter is None:
                import wordsegment

                wordsegment.load()
                self._segmenter = wordsegment.segment
            if self._predictor is None:
                ensure_bundled_nltk_data()
                import g2p_en

                self._predictor = g2p_en.G2p()
        except BackendError:
            raise
        except Exception as error:
            raise BackendError(
                "English OOV pronunciation resources are unavailable; install g2p-en, wordsegment, and NLTK"
            ) from error

    def _character(self, char: str) -> List[str]:
        char = char.lower()
        if len(char) != 1 or char not in self.dictionary:
            raise BackendError(f"Cannot spell English character {char!r}")
        index = 1 if char == "a" and len(self.dictionary[char]) > 1 else 0
        return list(self.dictionary[char][index])

    def _abbreviation(self, word: str) -> List[str]:
        return [phone for char in word for phone in self._character(char)]

    def convert(self, word: str) -> List[str]:
        word = word.translate(APOSTROPHE_TRANSLATION)
        if word.lower() in CLITIC_PRONUNCIATIONS:
            return list(CLITIC_PRONUNCIATIONS[word.lower()])
        if word.isalpha() and word.isupper() and len(word) <= 3:
            return self._abbreviation(word)
        if word.lower() in self.dictionary:
            return list(self.dictionary[word.lower()][0])
        if word.isalpha() and ((word.islower() and len(word) <= 3) or (word.isupper() and len(word) <= 4)):
            return self._abbreviation(word)

        self._ensure_fallbacks()
        try:
            raw_segments = self._segmenter(word.lower())
            if isinstance(raw_segments, (str, bytes)):
                raise TypeError("segmenter returned text instead of a sequence of segments")
            segments = list(raw_segments)
            if not segments or any(not isinstance(segment, str) or not segment for segment in segments):
                raise ValueError("segmenter returned an empty or non-string segment")
        except BackendError:
            raise
        except Exception as error:
            raise BackendError(f"English backend failed to segment {word!r}") from error
        if len(segments) == 1:
            try:
                raw_phones = self._predictor(segments[0])
                if isinstance(raw_phones, (str, bytes)):
                    raise TypeError("predictor returned text instead of a sequence of phones")
                phones = list(raw_phones)
            except BackendError:
                raise
            except Exception as error:
                raise BackendError(f"English backend failed to pronounce {word!r}") from error
            try:
                filtered = [phone for phone in phones if isinstance(phone, str) and phone.strip()]
                if len(filtered) != len(phones) or not filtered:
                    raise ValueError("predictor returned an empty or non-string phone")
            except Exception as error:
                raise BackendError(f"English backend returned malformed phones for {word!r}") from error
            return filtered

        phones = []
        for segment in segments:
            if len(segment) == 1:
                phones.extend(self._character(segment))
            else:
                phones.extend(self.convert(segment))
        return [phone for phone in phones if phone.strip()]

    def predict(
        self,
        request: PronunciationRequest,
    ) -> Mapping[int, Pronunciation]:
        result = {}
        for token in request.target_tokens:
            rendered_phones = tuple(self.convert(token.text))
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
