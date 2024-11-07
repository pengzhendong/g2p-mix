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

from pypinyin.constants import RE_HANS

from .token import Token


os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


class G2pMix:
    def __init__(self, jyut=False, strict=False, use_g2pw=False):
        if jyut:
            from .g2p_jyut import G2pJyut

            self.g2per = G2pJyut()
        else:
            from .g2p_pth import G2pPth

            self.g2per = G2pPth(strict, use_g2pw)

        self.jyut = jyut
        # add space between ascii and chinese characters
        self.pattern = re.compile(
            r"(?<=[\u4e00-\u9fa5])(?=[\x00-\x19\x21-\x7F])|(?<=[\x00-\x19\x21-\x7F])(?=[\u4e00-\u9fa5])"
        )

    def get_pinyins(self, text):
        text = self.pattern.sub(" ", text).replace(" ", "▁")
        pinyins = self.g2per.g2p(text)
        # combine pinyins, english words and punctuations by `posseg.cut`'s output
        tokens = []
        last_word = "▁"
        for idx, (word, pos, pinyin) in enumerate(pinyins):
            next_word = pinyins[idx + 1][0] if idx + 1 < len(pinyins) else "▁"
            if RE_HANS.match(word):
                tokens.append(Token(word, pos, pinyin))
            # he' ve => he've, ' cause => ' cause
            elif (
                (word.isalnum() and len(last_word) > 1 and last_word[-1] == "'")
                or (word[0] == "'" and len(word) > 1 and last_word != "▁")
                or (word == "'" and last_word != "▁" and next_word != "▁")
            ):
                word = tokens[-1].word + word
                tokens[-1] = Token(word, pos)
            elif word != "▁":
                tokens.append(Token(word, pos))
            last_word = word
        return tokens

    def g2p(self, text, sandhi=False, return_seg=False):
        seg_tokens = self.get_pinyins(text)
        if not self.jyut and sandhi:
            seg_tokens = self.g2per.sandhi(seg_tokens)
        tokens = []
        for token in seg_tokens:
            if token.lang != "ZH":
                if not return_seg:
                    token.pos = None
                tokens.append(token)
                continue
            # split pinyin into initial and final
            for idx, pinyin in enumerate(token.phones):
                initial, final, tone = self.g2per.parse(pinyin)
                token.phones[idx] = [initial, final + tone]
                tokens.append(Token(token.word[idx], None, token.phones[idx]))
        # 中文保留分词结果 or 中文分词结果拆成单个汉字
        return tokens if not return_seg else seg_tokens
