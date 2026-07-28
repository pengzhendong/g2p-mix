from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, FrozenSet, Optional, Sequence, Tuple

from .resources import load_json

PINYIN_BODY_PATTERN = re.compile(r"[a-zvê]+")
JYUTPING_BODY_PATTERN = re.compile(r"[a-z]+")
SPLIT_CACHE_SIZE = 4096


@lru_cache(maxsize=2)
def _legal_syllable_bodies(alphabet: str) -> FrozenSet[str]:
    return frozenset(load_json("syllables.json")[alphabet].split())


@lru_cache(maxsize=SPLIT_CACHE_SIZE)
def split_pinyin(pinyin: str, strict: bool = False) -> Tuple[str, str, str]:
    from pypinyin.contrib.tone_convert import to_finals, to_initials

    phones = load_json("phones.json")
    inventory = phones["ZH"]
    postnasals = set(inventory["postnasals"])
    tones = set(inventory["tones"])

    if len(pinyin) < 2 or pinyin[-1] not in tones:
        raise ValueError(f"Invalid numbered pinyin syllable: {pinyin!r}")

    base, tone = pinyin[:-1], pinyin[-1]
    if PINYIN_BODY_PATTERN.fullmatch(base) is None:
        raise ValueError(f"Invalid numbered pinyin syllable: {pinyin!r}")
    if base not in _legal_syllable_bodies("pinyin"):
        raise ValueError(f"Invalid numbered pinyin syllable: {pinyin!r}")
    if base in postnasals:
        return "", base, tone
    if base in {"hm", "hng"}:
        return "", base, tone

    initial = to_initials(base, strict=strict)
    final = to_finals(base, strict=strict)
    valid_initials = set(inventory["initials"]) | set(inventory["strict"]["initials"])
    valid_finals = set(inventory["finals"]) | set(inventory["strict"]["finals"])
    if initial not in valid_initials or final not in valid_finals:
        raise ValueError(f"Invalid numbered pinyin syllable: {pinyin!r}")
    return initial, final, tone


@lru_cache(maxsize=SPLIT_CACHE_SIZE)
def split_jyutping(jyutping: str) -> Tuple[str, str, str]:
    inventory = load_json("phones.json")["ZH"]["jyut"]
    tones = set(inventory["tones"])
    if len(jyutping) < 2 or jyutping[-1] not in tones:
        raise ValueError(f"Invalid Jyutping syllable: {jyutping!r}")

    body, tone = jyutping[:-1], jyutping[-1]
    if JYUTPING_BODY_PATTERN.fullmatch(body) is None:
        raise ValueError(f"Invalid Jyutping syllable: {jyutping!r}")
    if body not in _legal_syllable_bodies("jyutping"):
        raise ValueError(f"Invalid Jyutping syllable: {jyutping!r}")

    finals = set(inventory["finals"])
    for onset in sorted(inventory["onsets"], key=len, reverse=True):
        if body.startswith(onset) and body[len(onset) :] in finals:
            return onset, body[len(onset) :], tone
    raise ValueError(f"Invalid Jyutping syllable: {jyutping!r}")


@lru_cache(maxsize=SPLIT_CACHE_SIZE)
def split_jyutping_final(final: str) -> Tuple[str, str]:
    inventory = load_json("phones.json")["ZH"]["jyut"]
    if final not in set(inventory["finals"]):
        raise ValueError(f"Invalid Jyutping final: {final!r}")

    codas = set(inventory["codas"])
    for nucleus in sorted(inventory["nuclei"], key=len, reverse=True):
        if final.startswith(nucleus) and final[len(nucleus) :] in codas:
            return nucleus, final[len(nucleus) :]
    raise ValueError(f"Invalid Jyutping final: {final!r}")


@lru_cache(maxsize=1)
def _canonical_pinyin_phone_map() -> Dict[Tuple[str, str], Tuple[str, str]]:
    from pypinyin.contrib.tone_convert import to_finals, to_initials

    special_finals = set(load_json("phones.json")["ZH"]["postnasals"]) | {"hm", "hng"}
    result = {}
    for body in _legal_syllable_bodies("pinyin"):
        if body in special_finals:
            strict_phones = ("", body)
            result[strict_phones] = strict_phones
            continue

        strict_phones = (
            to_initials(body, strict=True),
            to_finals(body, strict=True),
        )
        loose_phones = (
            to_initials(body, strict=False),
            to_finals(body, strict=False),
        )
        result[strict_phones] = strict_phones
        result[loose_phones] = strict_phones
    return result


def canonical_pinyin_phones(phones: Sequence[str]) -> Tuple[str, str]:
    if len(phones) == 1:
        key = ("", phones[0])
    elif len(phones) == 2:
        key = (phones[0], phones[1])
    else:
        raise ValueError(f"Invalid Pinyin phones: {tuple(phones)!r}")

    try:
        return _canonical_pinyin_phone_map()[key]
    except KeyError as error:
        raise ValueError(f"Invalid Pinyin phones: {tuple(phones)!r}") from error


@lru_cache(maxsize=3)
def _transcription_profile(name: str):
    profiles = load_json("transcriptions.json")
    if profiles.get("schema_version") != 1:
        raise ValueError("Unsupported transcription resource schema")
    return profiles[name]


def tone_contour(profile: str, tone: str) -> Tuple[int, ...]:
    return tuple(_transcription_profile(profile)["tones"][tone])


def render_tone_contour(contour: Sequence[int]) -> str:
    tone_letters = {1: "˩", 2: "˨", 3: "˧", 4: "˦", 5: "˥"}
    try:
        return "".join(tone_letters[value] for value in contour)
    except KeyError as error:
        raise ValueError(f"Invalid tone contour: {tuple(contour)!r}") from error


def transcribe_pinyin(initial: str, final: str) -> Tuple[str, ...]:
    profile = _transcription_profile("cmn")
    pinyin = initial + final
    if pinyin in profile["interjections"]:
        return tuple(profile["interjections"][pinyin])
    if pinyin in profile["syllabic_consonants"]:
        return tuple(profile["syllabic_consonants"][pinyin])

    result = tuple(profile["initials"][initial])
    if final == "i" and initial in {"z", "c", "s"}:
        return result + tuple(profile["apical_finals"]["alveolar"])
    if final == "i" and initial in {"zh", "ch", "sh", "r"}:
        return result + tuple(profile["apical_finals"]["retroflex"])
    return result + tuple(profile["finals"][final])


def transcribe_jyutping(onset: str, final: str) -> Tuple[str, ...]:
    profile = _transcription_profile("yue-HK")
    nucleus, coda = split_jyutping_final(final)
    nucleus_segments = profile["nucleus_overrides"].get(f"{nucleus}+{coda}", profile["nuclei"][nucleus])
    coda_segments = profile["coda_overrides"].get(f"{nucleus}+{coda}", profile["codas"][coda])
    return tuple(profile["onsets"][onset]) + tuple(nucleus_segments) + tuple(coda_segments)


def transcribe_arpabet_phone(phone: str) -> Tuple[Tuple[str, ...], Optional[int]]:
    profile = _transcription_profile("en-US")
    base = phone
    stress = None
    if phone and phone[-1].isdigit():
        base = phone[:-1]
        stress = int(phone[-1])

    if base in profile["consonants"]:
        if stress is not None:
            raise KeyError(phone)
        return tuple(profile["consonants"][base]), None

    phones = profile["vowels"][base]
    if stress == 0:
        phones = profile["unstressed_vowels"].get(base, phones)
    return tuple(phones), stress
