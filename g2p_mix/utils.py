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

import pycantonese
from jieba import posseg
from pypinyin import load_phrases_dict, load_single_dict
from pypinyin.contrib.tone_convert import to_initials, to_finals

from .constants import CMUDICT, POSTNASALS


def g2p_ch(ch):
    ch = ch.lower()
    assert len(ch) == 1
    # In abbreviations, "A" should be pronounced as "EY1", not "AH0".
    return CMUDICT[ch][1] if ch == "a" else CMUDICT[ch][0]


def g2p_abbr(word):
    return [phone for ch in word for phone in g2p_ch(ch)]


def load_dict():
    # from pypinyin.constants import PINYIN_DICT
    # print(hex(ord("为")), PINYIN_DICT[ord("为")])
    dirname = os.path.dirname(__file__)
    for line in open(f"{dirname}/dict/single.txt", encoding="utf-8"):
        char, pinyins = line.strip().split(maxsplit=1)
        load_single_dict({ord(char): pinyins})
    for line in open(f"{dirname}/dict/phrases.txt", encoding="utf-8"):
        word, pinyins = line.strip().split(maxsplit=1)
        pinyins = pinyins.split()
        assert len(word) == len(pinyins)
        load_phrases_dict({word: [[pinyin] for pinyin in pinyins]})


def parse_jyutping(jyutping):
    jyutping = pycantonese.parse_jyutping(jyutping)
    assert len(jyutping) == 1
    initial = jyutping[0].onset
    final = jyutping[0].nucleus + jyutping[0].coda
    tone = jyutping[0].tone
    return initial, final, tone


def parse_pinyin(pinyin, strict=False):
    if pinyin[:-1] in POSTNASALS:
        initial = ""
        final = POSTNASALS[pinyin[:-1]]
    else:
        initial = to_initials(pinyin, strict=strict)
        final = to_finals(pinyin, strict=strict)
    tone = pinyin[-1]
    return initial, final, tone


def split_text(s):
    pattern = pattern = r"([\u4e00-\u9fff]+)|(\d+)|([^\w\s])|([a-zA-Z]+('[a-zA-Z]+)?)"
    matches = re.finditer(pattern, s)
    for match in matches:
        for i, value in enumerate(match.groups()):
            if value is not None:
                yield value, ["zh", "m", "x", "eng"][i]
                break


def posseg_cut(text):
    chinese_text = ""
    words = []
    for text, pos in split_text(text):
        if pos == "zh":
            chinese_text += text
            words += [(word, pos) for word, pos in posseg.cut(text)]
        else:
            words.append((text, pos))
    return chinese_text, words

