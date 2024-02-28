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

import re

import jieba.posseg as psg
from pypinyin import lazy_pinyin, Style

from g2p_mix.g2p_eng import G2pEn
from g2p_mix.tone_sandhi import ToneSandhi


g2p_en = G2pEn()
sandhier = ToneSandhi()


def cut(text):
    return sandhier.pre_merge_for_modify(psg.cut(text))


def get_initials_finals(word, strict):
    initials = lazy_pinyin(
        word, strict=strict, neutral_tone_with_five=True, style=Style.INITIALS
    )
    finals = lazy_pinyin(
        word, strict=strict, neutral_tone_with_five=True, style=Style.FINALS_TONE3
    )
    return initials, finals


def retokenize(words, pinyins):
    i = 0
    tokens = []
    while i < len(pinyins):
        pinyin = pinyins[i]
        if pinyin == " ":
            i += 1
            continue
        # he ' ve => he've
        suffixes = ["d", "s", "m", "re", "ve", "t", "clock", "em", "cause"]
        if pinyin == "'" and i + 1 < len(pinyins) and words[i + 1] in suffixes:
            tokens[-1]["word"] += words[i] + words[i + 1]
            tokens[-1]["phones"] += pinyin + pinyins[i + 1]
            i += 1
        else:
            tokens.append({"word": words[i], "phones": pinyin})
        i += 1
    return tokens


def g2p(text, sandhi=False, strict=True):
    # g2p zh
    words = []
    initials = []
    finals = []
    # add space between chinese char and alphabet
    text = re.sub(
        r"(?<=[\u4e00-\u9fa5])(?=[a-zA-Z])|(?<=[a-zA-Z])(?=[\u4e00-\u9fa5])", " ", text
    )
    for word, pos in cut(text):
        sub_initials, sub_finals = get_initials_finals(word, strict)
        # process english word and punctuations
        if len(sub_initials) == 1 and word == sub_initials[0]:
            if word.encode("UTF-8").isalnum():
                words.append(word)
            # split the continuous punctuations into single, e.g. "……" => "…", "…"
            else:
                words.extend(word)
                sub_initials = list(sub_initials[0])
                sub_finals = list(sub_finals[0])
        else:
            words.extend(word)
            if sandhi:
                sub_finals = sandhier.modified_tone(word, pos, sub_finals)
        initials.extend(sub_initials)
        finals.extend(sub_finals)

    pinyins = []
    for initial, final in zip(initials, finals):
        if initial == final:
            pinyins.append(initial)
        else:
            pinyins.append([initial, final])
    tokens = retokenize(words, pinyins)

    # g2p en
    for token in tokens:
        if token["word"].replace("'", "").encode("UTF-8").isalnum():
            token["phones"] = g2p_en.g2p(token["phones"])
    return tokens
