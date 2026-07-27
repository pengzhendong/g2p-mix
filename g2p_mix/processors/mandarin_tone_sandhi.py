from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from ..models import Language, Pronunciation, TextToken
from ..resources import load_lines

WordSplitter = Callable[[str], Sequence[str]]


@dataclass
class _SandhiWord:
    text: str
    pos: str
    refs: List[Tuple[int, int]]
    tones: List[str]

    def tone(self, index: int) -> str:
        if not -len(self.tones) <= index < len(self.tones):
            return ""
        return self.tones[index]

    def set_tone(self, tone: str, index: int = 0) -> None:
        self.tones[index] = tone

    def append(self, other: "_SandhiWord") -> "_SandhiWord":
        self.text += other.text
        self.refs.extend(other.refs)
        self.tones.extend(other.tones)
        return self


class MandarinToneSandhi:
    def __init__(self, word_splitter: Optional[WordSplitter] = None) -> None:
        self._digits = set(load_lines("digits.txt"))
        self._interjections = set(load_lines("interjections.txt"))
        self._neutral_tone_words = set(load_lines("neural_tone_words.txt"))
        self._whitelist = set(load_lines("whitelist.txt"))
        sandhi_words = self._neutral_tone_words | self._whitelist
        self._sandhi_words = sandhi_words
        self._sandhi_word_prefixes = {word[:end] for word in sandhi_words for end in range(1, len(word) + 1)}
        self._word_splitter = word_splitter or self._split_word

    def process(
        self,
        tokens: Sequence[TextToken],
        pronunciations: Mapping[int, Pronunciation],
    ) -> Mapping[int, Pronunciation]:
        changes: Dict[Tuple[int, int], str] = {}
        run: List[_SandhiWord] = []

        def flush() -> None:
            if not run:
                return
            for word in self._modify_run(run):
                for ref, tone in zip(word.refs, word.tones):
                    changes[ref] = tone
            run.clear()

        for token in tokens:
            pronunciation = pronunciations.get(token.id)
            if token.language is not Language.CHINESE or pronunciation is None:
                flush()
                continue

            run.append(
                _SandhiWord(
                    text=token.text,
                    pos=token.pos or "",
                    refs=[(token.id, index) for index in range(len(pronunciation.units))],
                    tones=[unit.tone or "" for unit in pronunciation.units],
                )
            )
        flush()

        result: MutableMapping[int, Pronunciation] = dict(pronunciations)
        for token_id, pronunciation in pronunciations.items():
            units = []
            changed = False
            for index, unit in enumerate(pronunciation.units):
                tone = changes.get((token_id, index), unit.tone)
                if tone != unit.tone and tone is not None:
                    units.append(unit.with_tone(tone))
                    changed = True
                else:
                    units.append(unit)
            if changed:
                result[token_id] = replace(pronunciation, units=tuple(units))
        return result

    def _modify_run(self, words: Sequence[_SandhiWord]) -> List[_SandhiWord]:
        merged = self._merge(
            [
                _SandhiWord(
                    text=word.text,
                    pos=word.pos,
                    refs=list(word.refs),
                    tones=list(word.tones),
                )
                for word in words
            ]
        )
        for word in merged:
            if len(word.text) < 2:
                continue
            self._bu_sandhi(word)
            self._yi_sandhi(word)
            if word.text in self._whitelist:
                subwords = (word.text,)
            else:
                subwords = self._split_subwords(word.text)
                self._neutral_sandhi(word, subwords)
            self._third_tone_sandhi(word, subwords)
        return merged

    def _merge(self, words: Sequence[_SandhiWord]) -> List[_SandhiWord]:
        words = self._merge_sandhi_words(words)
        merged: List[_SandhiWord] = []
        index = 0
        while index < len(words):
            word = words[index]
            index += 1
            if not merged:
                merged.append(word)
                continue

            previous = merged[-1]
            if previous.text in {"不", "很", "一"} or word.text == "儿":
                previous.append(word)
            elif (
                previous.tone(-1) == "3"
                and word.tone(0) == "3"
                and len(previous.text) <= 2
                and len(word.text) <= 2
                and len(previous.text) + len(word.text) <= 4
            ):
                previous.append(word)
            elif (
                word.text == "一"
                and previous.pos.startswith("v")
                and index < len(words)
                and previous.text == words[index].text
            ):
                previous.append(word).append(words[index])
                index += 1
            else:
                merged.append(word)
        return merged

    def _merge_sandhi_words(self, words: Sequence[_SandhiWord]) -> List[_SandhiWord]:
        merged = []
        index = 0
        while index < len(words):
            word = words[index]
            candidate = word.text
            best_end = index + 1
            end = index + 1

            while candidate in self._sandhi_word_prefixes and end < len(words):
                candidate += words[end].text
                end += 1
                if candidate in self._sandhi_words:
                    best_end = end

            for other in words[index + 1 : best_end]:
                word.append(other)
            merged.append(word)
            index = best_end
        return merged

    @staticmethod
    def _bu_sandhi(word: _SandhiWord) -> None:
        if len(word.text) == 3 and word.text[1] == "不":
            word.set_tone("5", 1)
            return
        for index, char in enumerate(word.text):
            if char == "不" and index + 1 < len(word.text) and word.tone(index + 1) == "4":
                word.set_tone("2", index)

    def _yi_sandhi(self, word: _SandhiWord) -> None:
        if len(word.text) == 3 and word.text[1] == "一" and word.text[0] == word.text[-1]:
            word.set_tone("5", 1)
            return

        for index, char in enumerate(word.text):
            if char != "一":
                continue
            previous = word.text[index - 1] if index else ""
            following = word.text[index + 1] if index + 1 < len(word.text) else ""
            if previous in self._digits | {"第", "初"} or following in self._digits:
                word.set_tone("1", index)
            elif following in {"", "月", "班", "连", "楼"}:
                word.set_tone("1", index)
            elif word.tone(index + 1) == "4":
                word.set_tone("2", index)
            else:
                word.set_tone("4", index)

    @staticmethod
    def _split_word(word: str) -> List[str]:
        import jieba

        candidates = sorted(jieba.cut_for_search(word), key=len)
        if not candidates:
            return [word]
        first = candidates[0]
        if not first or first == word:
            return [word]
        begin = word.find(first)
        if begin == 0:
            return [first, word[len(first) :]]
        return [word[:begin], word[begin:]]

    def _split_subwords(self, word: str) -> Tuple[str, ...]:
        subwords = tuple(self._word_splitter(word))
        if not subwords or any(not subword for subword in subwords) or "".join(subwords) != word:
            raise ValueError(f"Word splitter did not preserve {word!r}: {subwords!r}")
        return subwords

    def _neutral_sandhi(self, word: _SandhiWord, subwords: Sequence[str]) -> None:
        if word.text in self._neutral_tone_words:
            word.set_tone("5", -1)

        offset = 0
        for subword in subwords:
            offset += len(subword)
            if subword in self._whitelist:
                continue
            if subword in self._neutral_tone_words:
                word.set_tone("5", offset - 1)
            if subword[-1] in {"儿", "们"}:
                word.set_tone("5", offset - 1)
            if subword[-1] in self._interjections:
                word.set_tone("5", offset - 1)
            if len(subword) == 2 and subword[0] == subword[1] and word.pos.startswith("v"):
                word.set_tone("5", offset - 1)
            if subword in {"了", "着", "过"} and word.pos[:1] in {"n", "v", "a"}:
                word.set_tone("5", offset - 1)

    @staticmethod
    def _all_third(tones: Sequence[str]) -> bool:
        return bool(tones) and all(tone == "3" for tone in tones)

    def _third_tone_sandhi(self, word: _SandhiWord, subwords: Sequence[str]) -> None:
        if len(word.text) == 2:
            if self._all_third(word.tones):
                word.set_tone("2", 0)
            return
        if len(word.text) < 3:
            return

        if len(subwords[0]) == 1:
            if self._all_third(word.tones[1:3]):
                word.set_tone("2", 1)
            elif self._all_third(word.tones[:2]):
                word.set_tone("2", 0)
            return

        if self._all_third(word.tones[:2]):
            word.set_tone("2", 0)
        if self._all_third(word.tones[1:3]):
            if len(word.text) <= 3 or not self._all_third(word.tones[2:4]):
                word.set_tone("2", 1)
        if len(word.text) > 3 and self._all_third(word.tones[2:4]):
            word.set_tone("2", 2)
