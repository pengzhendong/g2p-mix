from __future__ import annotations

from typing import List, Tuple

from .resources import load_json


def split_pinyin(pinyin: str, strict: bool = False) -> Tuple[str, str, str]:
    from pypinyin.contrib.tone_convert import to_finals, to_initials

    phones = load_json("phones.json")
    postnasals = phones["ZH"]["postnasals"]
    tones = set(phones["ZH"]["tones"])

    if len(pinyin) < 2 or pinyin[-1] not in tones:
        raise ValueError(f"Invalid numbered pinyin syllable: {pinyin!r}")

    base, tone = pinyin[:-1], pinyin[-1]
    if base in postnasals:
        return "", base, tone
    return (
        to_initials(pinyin, strict=strict),
        to_finals(pinyin, strict=strict),
        tone,
    )


def split_jyutping(jyutping: str) -> Tuple[str, str, str]:
    import pycantonese

    parsed = pycantonese.parse_jyutping(jyutping)
    if len(parsed) != 1:
        raise ValueError(f"Invalid Jyutping syllable: {jyutping!r}")
    syllable = parsed[0]
    return syllable.onset, syllable.nucleus + syllable.coda, syllable.tone


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
    vowels = dict(ipa["vowels"])
    vowels.update({"EY": "A", "AY": "I", "AW": "W", "OY": "Y"})
    mapping = {**consonants, **vowels}

    if phone and phone[-1].isdigit():
        base, stress = phone[:-1], phone[-1]
        return ipa["stress"][stress] + mapping[base]
    return mapping[phone]
