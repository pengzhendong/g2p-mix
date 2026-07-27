from __future__ import annotations

import json
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, List

PACKAGE_DIR = Path(__file__).resolve().parent
DICT_DIR = PACKAGE_DIR / "dict"
NLTK_DATA_DIR = PACKAGE_DIR / "nltk_data"


@lru_cache(maxsize=None)
def load_json(name: str) -> dict:
    with (DICT_DIR / name).open(encoding="utf-8") as source:
        return json.load(source)


@lru_cache(maxsize=None)
def load_lines(name: str) -> tuple[str, ...]:
    with (DICT_DIR / name).open(encoding="utf-8") as source:
        return tuple(line.strip() for line in source if line.strip())


@contextmanager
def _temporary_nltk_path(path: str) -> Iterator[None]:
    import nltk

    inserted = path not in nltk.data.path
    if inserted:
        nltk.data.path.insert(0, path)
    try:
        yield
    finally:
        if inserted:
            nltk.data.path.remove(path)


@lru_cache(maxsize=1)
def load_cmudict() -> Dict[str, List[List[str]]]:
    from nltk.corpus import cmudict

    with _temporary_nltk_path(str(NLTK_DATA_DIR)):
        dictionary = cmudict.dict()

    for word in ("AE", "AI", "AR", "IOS", "HUD", "OS"):
        dictionary.pop(word.lower(), None)
    return dictionary


@lru_cache(maxsize=1)
def install_pinyin_overrides() -> None:
    import jieba
    from pypinyin import load_phrases_dict, load_single_dict

    for line in load_lines("single.txt"):
        char, pinyins = line.split(maxsplit=1)
        load_single_dict({ord(char): pinyins})

    for line in load_lines("phrases.txt"):
        word, pinyins = line.split(maxsplit=1)
        syllables = pinyins.split()
        if len(word) != len(syllables):
            raise ValueError(f"Invalid pinyin override for {word!r}")
        jieba.add_word(word)
        load_phrases_dict({word: [[syllable] for syllable in syllables]})
