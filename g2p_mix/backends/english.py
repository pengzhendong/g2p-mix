from __future__ import annotations

from typing import Callable, List, Mapping, Optional, Sequence

from ..errors import BackendError
from ..models import Language, PhoneAlphabet, Pronunciation, PronunciationUnit
from ..resources import load_cmudict
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
        if self._segmenter is None:
            import wordsegment

            wordsegment.load()
            self._segmenter = wordsegment.segment
        if self._predictor is None:
            import g2p_en

            self._predictor = g2p_en.G2p()

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
        segments = list(self._segmenter(word.lower()))
        if len(segments) == 1:
            try:
                phones = list(self._predictor(segments[0]))
            except TypeError as error:
                raise BackendError(f"English backend failed to pronounce {word!r}") from error
            return [phone for phone in phones if phone.strip()]

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
            phones = tuple(self.convert(token.text))
            if not phones:
                raise BackendError(f"English backend returned no phones for {token.text!r}")
            unit = PronunciationUnit(
                text=token.text,
                source_spans=token.source_spans,
                phones=phones,
                alphabet=PhoneAlphabet.ARPABET,
                native=" ".join(phones),
            )
            result[token.id] = Pronunciation(
                token_id=token.id,
                units=(unit,),
                backend=self.name,
            )
        return result
