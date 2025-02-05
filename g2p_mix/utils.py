# Copyright (c) 2024, Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re

import jieba
import pycantonese
from jieba.posseg import cut
from pycantonese import pos_tag
from pycantonese.jyutping.characters import _get_words_characters_to_jyutping
from pypinyin import load_phrases_dict, load_single_dict
from pypinyin.constants import RE_HANS
from pypinyin.contrib.tone_convert import to_finals, to_initials
from pypinyin.seg.simpleseg import seg

from .constants import CMUDICT, IPA, POSTNASALS_MAP


def g2p_ch(ch):
    """
    Convert a single English character to phonemes.
    """
    ch = ch.lower()
    assert len(ch) == 1
    # In abbreviations, "A" should be pronounced as "EY1", not "AH0".
    return CMUDICT[ch][1] if ch == "a" else CMUDICT[ch][0]


def g2p_abbr(word):
    """
    Convert an English abbreviation to phonemes.
    """
    return [phone for ch in word for phone in g2p_ch(ch)]


def get_language(word):
    """
    Get the language of a word:
    - "ZH" for Chinese
    - "EN" for English
    - "NUM" for number
    - "SYM" for symbols
    """
    if RE_HANS.match(word):
        return "ZH"
    elif word.isdigit():
        return "NUM"
    elif word.replace("'", "").isalnum():
        return "EN"
    return "SYM"


def load_dict():
    """
    Load dictionary.
    """
    # from pypinyin.constants import PINYIN_DICT
    # print(hex(ord("为")), PINYIN_DICT[ord("为")])
    dirname = os.path.dirname(__file__)
    for line in open(f"{dirname}/dict/single.txt", encoding="utf-8"):
        char, pinyins = line.strip().split(maxsplit=1)
        load_single_dict({ord(char): pinyins})
    for line in open(f"{dirname}/dict/phrases.txt", encoding="utf-8"):
        word, pinyins = line.strip().split(maxsplit=1)
        jieba.add_word(word, freq=None, tag=None)
        pinyins = pinyins.split()
        assert len(word) == len(pinyins)
        load_phrases_dict({word: [[pinyin] for pinyin in pinyins]})


def parse_jyutping(jyutping):
    """
    Parse an jyutping (for Cantonese Chinese) into initial, final and tone.
    """
    jyutping = pycantonese.parse_jyutping(jyutping)
    assert len(jyutping) == 1
    initial = jyutping[0].onset
    final = jyutping[0].nucleus + jyutping[0].coda
    tone = jyutping[0].tone
    return initial, final, tone


def parse_pinyin(pinyin, strict=False):
    """
    Parse a pinyin (for Mandarin Chinese) into initial, final and tone.
    """
    if pinyin[:-1] in POSTNASALS_MAP:
        initial = ""
        final = POSTNASALS_MAP[pinyin[:-1]]
    else:
        initial = to_initials(pinyin, strict=strict)
        final = to_finals(pinyin, strict=strict)
    tone = pinyin[-1]
    return initial, final, tone


def segment(text):
    """
    Segment a multilingual text into:
    - zh: Chinese text
    - m: numbers
    - x: punctuations
    - eng: English text
    """
    pattern = r"([\u4e00-\u9fff]+)|(\d+)|([^\w\s])|([a-zA-Z]+('[a-zA-Z]+)?)"
    matches = re.finditer(pattern, text)
    for match in matches:
        for i, value in enumerate(match.groups()):
            if value is not None:
                yield value, ["zh", "m", "x", "eng"][i]
                break


def posseg_cut(text, jyut=False, tagset="universal"):
    """
    Cut the text into Chinese text (for g2pw), words and pos tags.
    """
    assert tagset in ("universal", "hkcancor")
    chinese_text = ""
    words = []
    for text, pos in segment(text):
        if pos == "zh":
            chinese_text += text
            if jyut:
                words.extend(pos_tag(pycantonese.segment(text), tagset))
            else:
                # jieba cut
                for word, pos in cut(text):
                    # pypinyin cut
                    words.extend([(w, pos) for w in seg(word)])
        else:
            if jyut and pos == "x":
                pos = "PUNCT"
            words.append((text, pos))
    words = [(word, pos) for word, pos in words]
    return chinese_text, words


def convert_jyut(word):
    """
    Convert the word into jyutping.
    """
    words_to_jyutping, chars_to_jyutping = _get_words_characters_to_jyutping()
    try:
        jyutping = words_to_jyutping[word]
    except KeyError:
        jyutping = ""
        for char in word:
            try:
                jyutping += chars_to_jyutping[char]
            except KeyError:
                jyutping = None
                break
    if jyutping is not None:
        jyutping = re.findall(r"[\D]+\d|\D", jyutping)
    return jyutping


def apply_tone(phones, tone):
    if isinstance(phones, str):
        phones = [phones]
    return [phone.replace("0", tone) for phone in phones]


def pinyin2ipa(initial, final, tone):
    tone = IPA["ZH"]["tones"][tone]
    pinyin = initial + final
    if pinyin in IPA["ZH"]["interjections"]:
        phones = IPA["ZH"]["interjections"][pinyin]
        return apply_tone(phones, tone)
    if pinyin in IPA["ZH"]["syllabic_consonants"]:
        phones = IPA["ZH"]["syllabic_consonants"][pinyin]
        return apply_tone(phones, tone)

    phones = [IPA["ZH"]["initials"][initial]]
    if initial in {"zh", "ch", "sh", "r"} and final == "i":
        phones.append(IPA["ZH"]["finals"]["-i"]["ZH_CH_SH_R"].replace("0", tone))
    elif initial in {"z", "c", "s"} and final == "i":
        phones.append(IPA["ZH"]["finals"]["-i"]["Z_C_S"].replace("0", tone))
    else:
        phones.extend(apply_tone(IPA["ZH"]["finals"][final], tone))
    return phones
