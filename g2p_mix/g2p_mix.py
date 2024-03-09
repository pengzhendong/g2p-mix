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

import jieba.posseg as psg
from pypinyin import lazy_pinyin, Style
from pypinyin.constants import RE_HANS
from pypinyin.contrib.tone_convert import to_initials, to_finals
from pypinyin.seg import simpleseg

from .token import Token
from .tone_sandhi import ToneSandhi


os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
# 将后鼻音 n, m 当成韵母，同时与声母的 n, m 区分开
postnasals = {
    "n1": "ng1",
    "n2": "ng2",
    "n4": "ng4",
    "m2": "mg2",
    "m3": "mg3",
    "m4": "mg4",
}


class G2pMix:
    def __init__(self, use_g2pw=False, model_dir=None, model_id=None):
        if use_g2pw:
            from modelscope.hub.snapshot_download import snapshot_download
            from pypinyin_g2pw import G2PWPinyin

            model_id = model_id or "pengzhendong/g2pw"
            model_dir = model_dir or snapshot_download(model_id)
            self.lazy_pinyin = G2PWPinyin(
                model_dir=f"{model_dir}/G2PWModel",
                model_source=f"{model_dir}/bert-base-chinese",
                neutral_tone_with_five=True,
            ).lazy_pinyin
        else:
            self.lazy_pinyin = lazy_pinyin
        self.sandhier = ToneSandhi()
        # add space between ascii and chinese characters
        self.pattern = re.compile(
            r"(?<=[\u4e00-\u9fa5])(?=[\x00-\x19\x21-\x7F])|(?<=[\x00-\x19\x21-\x7F])(?=[\u4e00-\u9fa5])"
        )

    def get_pinyins(self, text, sandhi):
        pinyins = self.lazy_pinyin(
            text, tone_sandhi=sandhi, neutral_tone_with_five=True, style=Style.TONE3
        )
        # remove invalid pinyins of english words and punctuations
        idx = 0
        valid_pinyins = []
        for word in simpleseg.seg(text):
            if not RE_HANS.match(word):
                idx += 1
                continue
            valid_pinyins.extend(pinyins[idx + i] for i in range(len(word)))
            idx += len(word)
        return valid_pinyins

    def recut(self, text, pinyins):
        # combine pinyins, english words and punctuations by `psg.cut`'s output
        idx = 0
        tokens = []
        last_ch = ""
        for word, pos in psg.cut(text):
            if word == " ":
                last_ch = word
                continue
            if RE_HANS.match(word):
                tokens.append(Token(word, pos, pinyins[idx : idx + len(word)]))
                idx += len(word)
            # he ' ve => he've
            elif (word[0] == "'" and last_ch != " ") or (
                word.isalnum() and last_ch == "'"
            ):
                word = tokens[-1].word + word
                tokens[-1] = Token(word, pos)
            else:
                tokens.append(Token(word, pos))
            last_ch = word[-1]
        return tokens

    def g2p(self, text, strict=False, sandhi=False):
        text = self.pattern.sub(" ", text)
        pinyins = self.get_pinyins(text, sandhi)
        tokens = self.recut(text, pinyins)
        if sandhi:
            tokens = self.sandhier.modified_tone(tokens)
        for token in tokens:
            if token.lang != "ZH":
                continue
            # split pinyin into initial and final
            for i, pinyin in enumerate(token.phones):
                if pinyin in postnasals:
                    token.phones[i] = ["", postnasals[pinyin]]
                else:
                    tone = pinyin[-1]
                    initial = to_initials(pinyin, strict=strict)
                    final = to_finals(pinyin, strict=strict) + tone
                    token.phones[i] = [initial, final]
        return tokens
