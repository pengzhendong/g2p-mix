from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from threading import Condition, Lock, get_ident
from typing import Dict, FrozenSet, List, NamedTuple, Optional, Tuple

from .errors import BackendError

PACKAGE_DIR = Path(__file__).resolve().parent
DICT_DIR = PACKAGE_DIR / "dict"
NLTK_DATA_DIR = PACKAGE_DIR / "nltk_data"
_JIEBA_INSTALL_PENDING = "pending"
_JIEBA_INSTALL_INSTALLING = "installing"
_JIEBA_INSTALL_INSTALLED = "installed"
_JIEBA_PHRASE_INSTALL_CONDITION = Condition()
_NLTK_DATA_PATH_LOCK = Lock()


class _JiebaPhraseInstallState(NamedTuple):
    status: str
    owner: Optional[int]
    next_index: int
    installed: Optional[FrozenSet[str]]


_jieba_phrase_install_state = _JiebaPhraseInstallState(_JIEBA_INSTALL_PENDING, None, 0, None)


@lru_cache(maxsize=None)
def load_json(name: str) -> dict:
    with (DICT_DIR / name).open(encoding="utf-8") as source:
        return json.load(source)


@lru_cache(maxsize=None)
def load_lines(name: str) -> tuple[str, ...]:
    with (DICT_DIR / name).open(encoding="utf-8") as source:
        return tuple(line.strip() for line in source if line.strip())


def load_english_force_spellings() -> Tuple[str, ...]:
    return load_lines("english_force_spellings.txt")


def load_pinyin_phrases() -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    phrases = []
    for line in load_lines("phrases.txt"):
        word, pinyins = line.split(maxsplit=1)
        syllables = tuple(pinyins.split())
        if len(word) != len(syllables):
            raise ValueError(f"Invalid pinyin override for {word!r}")
        phrases.append((word, syllables))
    return tuple(phrases)


def ensure_bundled_nltk_data() -> None:
    try:
        import nltk
    except ImportError as error:
        raise BackendError("English pronunciation requires NLTK and the bundled NLTK resources") from error

    path = str(NLTK_DATA_DIR)
    with _NLTK_DATA_PATH_LOCK:
        nltk.data.path[:] = [candidate for candidate in nltk.data.path if candidate != path]
        nltk.data.path.append(path)


@lru_cache(maxsize=1)
def load_cmudict() -> Dict[str, List[List[str]]]:
    ensure_bundled_nltk_data()
    try:
        from nltk.corpus import cmudict

        dictionary = cmudict.dict()
    except (ImportError, LookupError, OSError) as error:
        raise BackendError("Bundled English CMUdict resources could not be loaded") from error

    for spelling in load_english_force_spellings():
        dictionary.pop(spelling.lower(), None)
    return dictionary


def install_jieba_phrases() -> FrozenSet[str]:
    global _jieba_phrase_install_state

    state = _jieba_phrase_install_state
    if state.installed is not None:
        return state.installed

    import jieba

    owner = get_ident()
    claimed = False
    try:
        with _JIEBA_PHRASE_INSTALL_CONDITION:
            while True:
                observed_state = _jieba_phrase_install_state
                if observed_state.status != _JIEBA_INSTALL_INSTALLING:
                    break
                if observed_state.owner == owner:
                    raise RuntimeError("install_jieba_phrases() re-entered on its installing thread")
                _JIEBA_PHRASE_INSTALL_CONDITION.wait()

            if observed_state.installed is not None:
                return observed_state.installed

            claimed_state = _JiebaPhraseInstallState(
                _JIEBA_INSTALL_INSTALLING,
                owner,
                observed_state.next_index,
                None,
            )
            claimed = True
            _jieba_phrase_install_state = claimed_state
            next_index = claimed_state.next_index

        phrases = load_pinyin_phrases()
        for index in range(next_index, len(phrases)):
            jieba.add_word(phrases[index][0])
            with _JIEBA_PHRASE_INSTALL_CONDITION:
                _jieba_phrase_install_state = _JiebaPhraseInstallState(
                    _JIEBA_INSTALL_INSTALLING, owner, index + 1, None
                )

        installed = frozenset(word for word, _ in phrases)
        with _JIEBA_PHRASE_INSTALL_CONDITION:
            _jieba_phrase_install_state = _JiebaPhraseInstallState(
                _JIEBA_INSTALL_INSTALLED, None, len(phrases), installed
            )
            _JIEBA_PHRASE_INSTALL_CONDITION.notify_all()
    except BaseException:
        if claimed:
            with _JIEBA_PHRASE_INSTALL_CONDITION:
                observed_state = _jieba_phrase_install_state
                if observed_state.status == _JIEBA_INSTALL_INSTALLING and observed_state.owner == owner:
                    _jieba_phrase_install_state = _JiebaPhraseInstallState(
                        _JIEBA_INSTALL_PENDING,
                        None,
                        observed_state.next_index,
                        None,
                    )
                    _JIEBA_PHRASE_INSTALL_CONDITION.notify_all()
                elif observed_state.status == _JIEBA_INSTALL_INSTALLED:
                    _JIEBA_PHRASE_INSTALL_CONDITION.notify_all()
        raise

    return installed


@lru_cache(maxsize=1)
def install_pypinyin_overrides() -> None:
    from pypinyin import load_phrases_dict, load_single_dict

    for line in load_lines("single.txt"):
        char, pinyins = line.split(maxsplit=1)
        load_single_dict({ord(char): pinyins})

    for word, syllables in load_pinyin_phrases():
        load_phrases_dict({word: [[syllable] for syllable in syllables]})
