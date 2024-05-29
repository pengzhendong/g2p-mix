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
from functools import partial

from jieba import posseg
from pypinyin import Style, load_phrases_dict, load_single_dict
from pypinyin.contrib.tone_convert import to_initials, to_finals
from pypinyin.converter import UltimateConverter

from .constants import POSTNASALS
from .tone_sandhi import ToneSandhi


os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


class G2pPth:
    def __init__(self, strict, use_g2pw):
        if use_g2pw:
            from modelscope import snapshot_download
            from pypinyin_g2pw import G2PWPinyin

            repo_dir = snapshot_download("pengzhendong/g2pw")
            self.converter = G2PWPinyin(
                model_dir=f"{repo_dir}/G2PWModel",
                model_source=f"{repo_dir}/bert-base-chinese",
                neutral_tone_with_five=True,
            )._converter
        else:
            # tone_sandhi rules of pypinyin is not perfect
            self.converter = UltimateConverter(neutral_tone_with_five=True)
        self.convert = partial(
            self.converter.convert,
            style=Style.TONE3,
            heteronym=False,
            errors="default",
            strict=True,
        )
        self.strict = strict
        self.sandhi = ToneSandhi().modified_tone
        # load dict to fix some badcase of pypinyin
        self.load_dict()

    def load_dict(self):
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

    def parse(self, pinyin):
        if pinyin[:-1] in POSTNASALS:
            initial = ""
            final = POSTNASALS[pinyin[:-1]]
        else:
            initial = to_initials(pinyin, strict=self.strict)
            final = to_finals(pinyin, strict=self.strict)
        tone = pinyin[-1]
        return initial, final, tone

    def g2p(self, text):
        segs = list(posseg.cut(text))
        pinyins = []
        for word, pos in segs:
            pinyin = sum(self.convert(word), [])
            pinyins.append((word, pos, pinyin))
        return pinyins
