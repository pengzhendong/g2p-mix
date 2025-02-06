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

import json
import os

import nltk
from jieba import posseg
from nltk.corpus import cmudict
from pycantonese.pos_tagging.hkcancor_to_ud import _MAP

# cmudict
dirname = os.path.dirname(__file__)
nltk.data.path.insert(0, f"{dirname}/nltk_data")
CMUDICT = cmudict.dict()
# Remove abbreviations badcase like "HUD" in cmudict.
for word in ["AE", "AI", "AR", "IOS", "HUD", "OS"]:
    del CMUDICT[word.lower()]

POS = json.load(open(f"{dirname}/dict/pos.json"))
PHONES = json.load(open(f"{dirname}/dict/phones.json"))
# https://github.com/stefantaubert/pinyin-to-ipa/blob/master/src/pinyin_to_ipa/transcription.py
# first = True
# 0 是声调的占位符
# h: [x, h], r: [ɻ, ʐ], y: [j, ɥ], er: [ɚ0, aɚ̯0], i: ɨ0
IPA = json.load(open(f"{dirname}/dict/ipa.json"))
IPA_ZH = IPA["ZH"]
# https://github.com/hexgrad/misaki/blob/main/EN_PHONES.md
consonants = IPA["EN"]["consonants"]
consonants["JH"] = "ʤ"
consonants["CH"] = "ʧ"
vowels = IPA["EN"]["vowels"]
vowels["EY"] = "A"
vowels["AY"] = "I"
vowels["AW"] = "W"
vowels["OY"] = "Y"
IPA_EN = consonants | vowels | IPA["EN"]["stress"]

JIEBA_FREQ = posseg.dt.FREQ  # jieba 词典频率
JIEBA_POS = POS["ZH"]["jieba"]
# https://universaldependencies.org/u/pos/index.html
UNIVERSAL_POS = POS["ZH"]["universal"]
# https://github.com/fcbond/hkcancor
# The HKCanCor paper describes 46 tags in its tagset, but the actual data has 112 tags.
_MAP["G1"] = "VERB"  # https://github.com/jacksonllee/pycantonese/issues/48
HKCANCOR_POS = {**{key: UNIVERSAL_POS[value] for key, value in _MAP.items()} | POS["ZH"]["hkcancor"]}

# pypinyin to_initials(strict=True): 21
# 对于零声母拼音(没有声母)，声母为空字符串
STRICT_INITIALS = PHONES["ZH"]["strict"]["initials"]
# 严格意义来说，y 和 w 不算声母。对于零声母拼音（没有声母）：
# 1. 韵母 i 后面有其他韵母，把 i 换成 y
# 2. 韵母 u 后面有其他韵母，把 u 换成 w
# 3. 韵母 i 后面没有其他韵母，i 前面加 y
# 4. 韵母 u 后面没有其他韵母，u 前面加 w
# 5. 韵母 ü 后面有其他韵母，去掉两点写成 u，且前面加 y。如 yue
# pypinyin to_initials(strict=False)
INITIALS = PHONES["ZH"]["initials"]
# 39 个韵母：
# a o e i -i(前) -i(后) u ü er ai ei ao ou ia ie ua uo
# üe iao iou uai uei an ian uan üan en in uen ün ang
# iang uang eng ing ueng ong iong ê
# 舌尖前元音 -i(前) 舌尖要靠前，是前高不圆唇舌尖元音，只能与舌尖前音 z c s 构成整体认读音节
# 舌尖后元音 -i(后) 舌尖要靠后，是后高不圆唇舌尖元音，只能与舌尖后音 zh ch sh r 后构成整体认读音节
# ê 只在语气词"欸"(现用"唉")中单用
# pypinyin to_finals(strict=False): 37
# - 不区分 i, -i(前) 和 -i(后)，统一成 i
# - 少了 uen 和 ueng (只出现在零声母拼音中，会写成 wen 和 weng)
# - 少了 üan (在拼音中均写成 uan)
# - üe 多了一种写法 ue (根据《汉语拼音方案》，ü 跟 n, l 以外的声母相拼时，写成 u)
# - 简写：iou => iu, uei => ui, ün => un
STRICT_FINALS = PHONES["ZH"]["strict"]["finals"]
# pypinyin to_finals(strict=True): 40
# - 不区分 i, -i(前) 和 -i(后)，统一成 i
# - 多了 io (《现代汉语词典》中，“哟”读 io1)
FINALS = PHONES["ZH"]["finals"]
# 3 个 后鼻音韵母：n, ng, m
POSTNASALS = PHONES["ZH"]["postnasals"]
# 1: level, 2: rising, 3: falling-rising, 4: falling, 5: neutral
TONES = PHONES["ZH"]["tones"]

# https://www.ilc.cuhk.edu.hk/workshop/Chinese/Cantonese/Romanization/ch1_intro/1_history.aspx
ONSETS = PHONES["ZH"]["jyut"]["onsets"]
NUCLEI = PHONES["ZH"]["jyut"]["nuclei"]
CODAS = PHONES["ZH"]["jyut"]["codas"]
JYUT_TONES = PHONES["ZH"]["jyut"]["tones"]
# 60 + 2
# 韵母分类：
# - 单元音（单韵母）：i yu u e oe o a aa
# - 复元音收 i、u 韵尾（复韵母）：iu ui ei eu eoi oi ou ai au aai aau
# - 单元音收 m、n、ng 鼻音韵尾（鼻音韵母）：im in ing yun um un ung em en eng eon oeng on ong am an ang aam aan aang
# - 单元音收 p、t、k 塞音韵尾（塞音韵母）：ip it ik yut up ut uk ep et ek eot oet oek ot ok ap at ak aap aat aak
# - 鼻音 m、ng 单独成韵：m ng
JYUT_FINALS = PHONES["ZH"]["jyut"]["finals"]

# http://www.speech.cs.cmu.edu/cgi-bin/cmudict
# 0: no stress, 1: primary stress, 2: secondary stress
STRESS = PHONES["EN"]["stress"]
VOWELS = PHONES["EN"]["vowels"]
CONSONANTS = PHONES["EN"]["consonants"]
