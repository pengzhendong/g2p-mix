from __future__ import annotations

import re
import unicodedata
from dataclasses import replace
from typing import List, Protocol, Sequence, Tuple

from .models import (
    Boundary,
    Language,
    NormalizedText,
    Span,
    TextToken,
    TokenKind,
)

HAN_RANGES = "\u3400-\u4dbf" "\u4e00-\u9fff" "\uf900-\ufaff" "\U00020000-\U0002fa1f"
TOKEN_PATTERN = re.compile(
    rf"(?P<space>\s+)"
    rf"|(?P<han>[{HAN_RANGES}]+)"
    rf"|(?P<number>\d+(?:[.:/-]\d+)*)"
    rf"|(?P<latin>[^\W\d_]+(?:['’-][^\W\d_]+)*)"
    rf"|(?P<other>[^\s])",
    re.UNICODE,
)

PAUSE_MARKS = frozenset(",，、;；:：")
SENTENCE_MARKS = frozenset(".。!?！？")


class ChineseSegmenter(Protocol):
    def segment(self, text: str) -> Sequence[Tuple[str, str]]:
        pass


class JiebaSegmenter:
    def segment(self, text: str) -> Sequence[Tuple[str, str]]:
        from jieba.posseg import cut
        from pypinyin.seg.simpleseg import seg

        words = []
        for word, pos in cut(text):
            if len(word) < 4:
                words.append((word, pos))
            else:
                words.extend((subword, pos) for subword in seg(word))
        return words


class PyCantoneseSegmenter:
    def __init__(self, tagset: str = "universal") -> None:
        if tagset not in {"universal", "hkcancor"}:
            raise ValueError(f"Unsupported Cantonese tagset: {tagset!r}")
        self._tagset = tagset

    def segment(self, text: str) -> Sequence[Tuple[str, str]]:
        import pycantonese

        words = pycantonese.segment(text)
        return pycantonese.pos_tag(words, self._tagset)


class LosslessTokenizer:
    def scan(self, value: NormalizedText) -> Tuple[TextToken, ...]:
        tokens: List[TextToken] = []
        previous_non_space = None
        saw_space = False

        for match in TOKEN_PATTERN.finditer(value.text):
            text = match.group()
            language, kind = self._classify(match.lastgroup, text)
            boundary = self._boundary_before(
                language=language,
                text=text,
                previous=previous_non_space,
                saw_space=saw_space,
            )
            normalized_span = Span(match.start(), match.end())
            token = TextToken(
                id=len(tokens),
                text=text,
                normalized_span=normalized_span,
                source_spans=value.sources_for(normalized_span),
                language=language,
                kind=kind,
                boundary_before=boundary,
            )
            tokens.append(token)

            if language is Language.SPACE:
                saw_space = True
            else:
                previous_non_space = token
                saw_space = False

        if "".join(token.text for token in tokens) != value.text:
            raise RuntimeError("Tokenizer lost input characters")
        return tuple(tokens)

    @staticmethod
    def _classify(group: str, text: str) -> Tuple[Language, TokenKind]:
        if group == "space":
            return Language.SPACE, TokenKind.SPACE
        if group == "han":
            return Language.CHINESE, TokenKind.HAN
        if group == "number":
            return Language.NUMBER, TokenKind.NUMBER
        if group == "latin":
            return Language.ENGLISH, TokenKind.LATIN

        category = unicodedata.category(text[0])
        kind = TokenKind.PUNCTUATION if category.startswith("P") else TokenKind.SYMBOL
        return Language.SYMBOL, kind

    @staticmethod
    def _boundary_before(
        language: Language,
        text: str,
        previous: TextToken,
        saw_space: bool,
    ) -> Boundary:
        if previous is None or language is Language.SPACE:
            return Boundary.NONE
        if previous.text and previous.text[-1] in SENTENCE_MARKS:
            return Boundary.SENTENCE
        if previous.text and previous.text[-1] in PAUSE_MARKS:
            return Boundary.PAUSE
        if {
            previous.language,
            language,
        } == {Language.CHINESE, Language.ENGLISH}:
            return Boundary.CODE_SWITCH
        if saw_space:
            return Boundary.SOFT
        return Boundary.NONE


class TextAnalyzer:
    def __init__(
        self,
        chinese_segmenter: ChineseSegmenter,
        tokenizer: LosslessTokenizer = None,
    ) -> None:
        self._segmenter = chinese_segmenter
        self._tokenizer = tokenizer or LosslessTokenizer()

    def analyze(self, value: NormalizedText) -> Tuple[TextToken, ...]:
        coarse_tokens = self._tokenizer.scan(value)
        analyzed: List[TextToken] = []

        for coarse in coarse_tokens:
            if coarse.language is not Language.CHINESE:
                analyzed.append(replace(coarse, id=len(analyzed)))
                continue

            words = tuple(self._segmenter.segment(coarse.text))
            if "".join(word for word, _ in words) != coarse.text:
                raise ValueError(f"Chinese segmenter did not preserve text: {coarse.text!r}")

            cursor = coarse.normalized_span.start
            for index, (word, pos) in enumerate(words):
                span = Span(cursor, cursor + len(word))
                analyzed.append(
                    TextToken(
                        id=len(analyzed),
                        text=word,
                        normalized_span=span,
                        source_spans=value.sources_for(span),
                        language=Language.CHINESE,
                        kind=TokenKind.HAN,
                        boundary_before=(coarse.boundary_before if index == 0 else Boundary.NONE),
                        pos=pos,
                    )
                )
                cursor = span.end

        if "".join(token.text for token in analyzed) != value.text:
            raise RuntimeError("Analyzer lost input characters")
        return tuple(analyzed)
