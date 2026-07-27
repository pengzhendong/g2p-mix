from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, FrozenSet, List, Sequence, Tuple

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


def _apply_tone(phones, tone: str) -> List[str]:
    if isinstance(phones, str):
        phones = [phones]
    return [phone.replace("0", tone) for phone in phones]


def pinyin_to_ipa(initial: str, final: str, tone: str) -> Tuple[str, ...]:
    ipa = load_json("ipa.json")["ZH"]
    tone_mark = ipa["tones"][tone]
    pinyin = initial + final

    if pinyin in ipa["interjections"]:
        return tuple(_apply_tone(ipa["interjections"][pinyin], tone_mark))
    if pinyin in ipa["syllabic_consonants"]:
        return tuple(_apply_tone(ipa["syllabic_consonants"][pinyin], tone_mark))

    result = []
    if initial:
        result.append(ipa["initials"][initial])
    if initial in {"zh", "ch", "sh", "r", "z", "c", "s"} and final == "i":
        result.extend(_apply_tone(ipa["finals"]["-i"], tone_mark))
    else:
        result.extend(_apply_tone(ipa["finals"][final], tone_mark))
    return tuple(result)


def arpabet_to_ipa(phone: str) -> str:
    ipa = load_json("ipa.json")["EN"]
    consonants = dict(ipa["consonants"])
    consonants.update({"JH": "ʤ", "CH": "ʧ"})
    mapping = {**consonants, **ipa["vowels"]}

    if phone and phone[-1].isdigit():
        base, stress = phone[:-1], phone[-1]
        return ipa["stress"][stress] + mapping[base]
    return mapping[phone]
