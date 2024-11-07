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

import jieba
import nltk
from jieba import posseg
from nltk.corpus import cmudict
from pycantonese.jyutping import parse_jyutping


# cmudict
dirname = os.path.dirname(__file__)
nltk.data.path.insert(0, f"{dirname}/nltk_data")
CMUDICT = cmudict.dict()
# Remove abbreviations badcase like "HUD" in cmudict.
for word in ["AE", "AI", "AR", "IOS", "HUD", "OS"]:
    del CMUDICT[word.lower()]

# jieba 词典频率
jieba.initialize()
FREQ = posseg.dt.FREQ

# jiba 词性
# fmt: off
FLAGS = {
    "a": "形容词", "ad": "副形词", "ag": "形语素", "an": "名形词",
    "b": "区别词", "c": "连词", "d": "副词", "df": "不要", "dg": "副语素",
    "e": "叹词", "f": "方位词", "g": "语素", "h": "前接成分", "i": "成语",
    "j": "简称略称", "k": "后接成分", "l": "习用语", "m": "数词", "mg": "数语素",
    "mq": "数量词", "n": "名词", "ng": "名语素", "nr": "人名", "nrfg": "古近代人名",
    "nrt": "音译人名", "ns": "地名", "nt": "机构团体", "nz": "其他专名",
    "o": "拟声词", "p": "介词", "q": "量词", "r": "代词", "rg": "代语素",
    "rr": "人称代词", "rz": "指示代词", "s": "处所词", "t": "时间词", "tg": "时语素",
    "u": "助词", "ud": "结构助词-得", "ug": "时态助词-过", "uj": "结构助词-的",
    "ul": "时态助词-了", "uv": "结构助词-地", "uz": "时态助词-着", "v": "动词",
    "vd": "副动词", "vg": "动语素", "vi": "不及物动词", "vn": "名动词", "vq": "去过",
    "x": "非语素", "y": "语气词", "z": "状态词", "zg": "状语素", "eng": "英文单词",
}

# 21
# pypinyin to_initials(strict=True)
# 对于零声母拼音(没有声母)，声母为空字符串
# fmt: off
STRICT_INITIALS = {
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "zh",
    "ch", "sh", "r", "z", "c", "s", "",
}

# 严格意义来说，y 和 w 不算声母。对于零声母拼音（没有声母）：
# 1. 韵母 i 后面有其他韵母，把 i 换成 y
# 2. 韵母 u 后面有其他韵母，把 u 换成 w
# 3. 韵母 i 后面没有其他韵母，i 前面加 y
# 4. 韵母 u 后面没有其他韵母，u 前面加 w
# 5. 韵母 ü 后面有其他韵母，去掉两点写成 u，且前面加 y。如 yue
# pypinyin to_initials(strict=False)
# fmt: off
INITIALS = {
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "zh",
    "ch", "sh", "r", "z", "c", "s", "y", "w",
}

# 3 个 后鼻音韵母：n, ng, m。替换 n 和 m 为 ng 和 mg，避免与声母混淆
POSTNASALS = {"n": "ng", "m": "mg"}

# 39 个韵母：
# a o e i -i(前) -i(后) u ü er ai ei ao ou ia ie ua uo
# üe iao iou uai uei an ian uan üan en in uen ün ang
# iang uang eng ing ueng ong iong ê
# 舌尖前元音 -i(前) 舌尖要靠前，是前高不圆唇舌尖元音，只能与舌尖前音 z c s 构成整体认读音节
# 舌尖后元音 -i(后) 舌尖要靠后，是后高不圆唇舌尖元音，只能与舌尖后音 zh ch sh r 后构成整体认读音节
# ê 只在语气词"欸"(现用"唉")中单用

# 37
# pypinyin to_finals(strict=False)
# - 不区分 i, -i(前) 和 -i(后)，统一成 i
# - 少了 uen 和 ueng (只出现在零声母拼音中，会写成 wen 和 weng)
# - 少了 üan (在拼音中均写成 uan)
# - üe 多了一种写法 ue (根据《汉语拼音方案》，ü 跟 n, l 以外的声母相拼时，写成 u)
# - 简写：iou => iu, uei => ui, ün => un
# fmt: off
STRICT_FINALS = {
    "a", "o", "e", "i", "u", "v", "er", "ai", "ei", "ao", "ou", "ia", "ie",
    "ua", "uo", "ve", "iao", "iu", "uai", "ui", "an", "ian", "uan", "en", "in",
    "ang", "un", "iang", "uang", "eng", "ing", "ong", "iong", "ê", "ng", "mg",
    "ue",
}

# 40
# pypinyin to_finals(strict=True)
# - 不区分 i, -i(前) 和 -i(后)，统一成 i
# - 多了 io (《现代汉语词典》中，“哟”读 io1)
# fmt: off
FINALS = {
    "a", "o", "e", "i", "u", "v", "er", "ai", "ei", "ao", "ou", "ia", "ie",
    "ua", "uo", "ve", "iao", "iou", "uai", "uei", "an", "ian", "uan", "van",
    "en", "in", "uen", "vn", "ang", "iang", "uang", "eng", "ing", "ueng", "ong",
    "iong", "ê", "ng", "mg", "io",
}

# 5
TONES = {"0", "1", "2", "3", "4"}

# https://www.ilc.cuhk.edu.hk/workshop/Chinese/Cantonese/Romanization/ch1_intro/1_history.aspx
ONSETS = parse_jyutping.ONSETS
NUCLEI = parse_jyutping.NUCLEI
CODAS = parse_jyutping.CODAS
JYUT_TONES = parse_jyutping.TONES

# 60 + 2
# 韵母分类：
# - 单元音（单韵母）：i yu u e oe o a aa
# - 复元音收 i、u 韵尾（复韵母）：iu ui ei eu eoi oi ou ai au aai aau
# - 单元音收 m、n、ng 鼻音韵尾（鼻音韵母）：im in ing yun um un ung em en eng eon oeng on ong am an ang aam aan aang
# - 单元音收 p、t、k 塞音韵尾（塞音韵母）：ip it ik yut up ut uk ep et ek eot oet oek ot ok ap at ak aap aat aak
# - 鼻音 m、ng 单独成韵：m ng
# fmt: off
JYUT_FINALS = {
    "i", "yu", "u", "e", "oe", "o", "a", "aa", "iu", "ui", "ei", "eu", "eoi",
    "oi", "ou", "ai", "au", "aai", "aau", "im", "in", "ing", "yun", "um", "un",
    "ung", "em", "en", "eng", "eon", "oeng", "on", "ong", "am", "an", "ang",
    "aam", "aan", "aang", "ip", "it", "ik", "yut", "up", "ut", "uk", "ep", "et",
    "ek", "eot", "oet", "oek", "ot", "ok", "ap", "at", "ak", "aap", "aat",
    "aak", "m", "ng",
}

# http://www.speech.cs.cmu.edu/cgi-bin/cmudict
STRESS = {"", "0", "1", "2"}

# 39
# fmt: off
PHONES = {
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P",
    "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
}
