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

import jieba.posseg as psg
from pypinyin import lazy_pinyin, Style
from pypinyin.contrib.tone_convert import to_initials, to_finals

from g2p_mix.tone_sandhi import ToneSandhi


sandhier = ToneSandhi()

# 将后鼻音 n, m 当成韵母，同时与声母的 n, m 区分开
postnasals = {
    "n1": "ng1",
    "n2": "ng2",
    "n4": "ng4",
    "m2": "mg2",
    "m3": "mg3",
    "m4": "mg4",
}


def cut(text):
    return sandhier.pre_merge_for_modify(psg.cut(text))


def get_initials_finals(word, strict):
    pinyins = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.TONE3)
    initials = []
    finals = []
    for pinyin in pinyins:
        if pinyin in postnasals:
            initials.append("")
            finals.append(postnasals[pinyin])
        else:
            initials.append(to_initials(pinyin, strict=strict))
            finals.append(to_finals(pinyin, strict=strict) + pinyin[-1])
    return initials, finals
