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

import logging
from functools import partial

import jieba
from pyopenhc import OpenHC
from pypinyin import Style
from pypinyin.converter import UltimateConverter
from wetext import Normalizer

from .token import Token
from .tone_sandhi import convert as convert_tone
from .utils import convert_jyut, get_language, parse_jyutping, parse_pinyin, posseg_cut

jieba.setLogLevel(logging.INFO)


class G2pMix:
    def __init__(self, tn=False, jyut=False, g2pw=False, strict=False):
        self.parse = parse_jyutping if jyut else partial(parse_pinyin, strict=strict)
        self.tn = tn
        self.jyut = jyut
        self.g2pw = g2pw
        self.normalize = Normalizer().normalize if tn else None
        if jyut:
            self.s2t_converter = OpenHC("s2t")
            self.convert = convert_jyut
        else:
            if g2pw:
                from modelscope import snapshot_download
                from pypinyin_g2pw import G2PWPinyin

                repo_dir = snapshot_download("pengzhendong/g2pw")
                converter = G2PWPinyin(
                    model_dir=f"{repo_dir}/G2PWModel",
                    model_source=f"{repo_dir}/bert-base-chinese",
                    neutral_tone_with_five=True,
                )._converter
            else:
                # tone_sandhi rules of pypinyin is not perfect
                converter = UltimateConverter(neutral_tone_with_five=True)
            self.convert = partial(
                converter.convert,
                style=Style.TONE3,
                heteronym=False,
                errors="default",
                strict=True,
            )

    def g2p(self, text, sandhi=False, return_seg=False):
        if self.tn:
            text = self.normalize(text)
        if self.jyut:
            text = self.s2t_converter.convert(text)
        chinese_text, words = list(posseg_cut(text, self.jyut))
        # g2p
        seg_tokens = []
        if self.g2pw:
            pinyins = self.convert(chinese_text)
        for word, pos in words:
            phones = None
            if get_language(word) == "ZH":
                if self.g2pw:
                    phones = pinyins[: len(word)]
                    del pinyins[: len(word)]
                else:
                    phones = self.convert(word)
                if not self.jyut:
                    phones = sum(phones, [])
            seg_tokens.append(Token(word, pos, phones))
        # sandhi
        if not self.jyut and sandhi:
            seg_tokens = convert_tone(seg_tokens)
        # split
        tokens = []
        for token in seg_tokens:
            if token.lang == "ZH":
                # split pinyin into initial and final
                for idx, pinyin in enumerate(token.phones):
                    initial, final, tone = self.parse(pinyin)
                    token.phones[idx] = [initial, final + tone]
                    tokens.append(Token(token.word[idx], token.pos, token.phones[idx]))
            else:
                tokens.append(token)
        # 中文保留分词结果 or 中文分词结果拆成单个汉字
        return tokens if not return_seg else seg_tokens
