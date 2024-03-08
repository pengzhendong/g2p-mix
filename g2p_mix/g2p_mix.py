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

from .g2p_eng import G2pEn
from .g2p_han import G2pHan


class G2pMix:
    def __init__(self, use_g2pw=False, model_dir=None):
        self.g2p_en = G2pEn()
        self.g2p_han = G2pHan(use_g2pw=use_g2pw, model_dir=model_dir)

    @staticmethod
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

    def g2p(self, text, sandhi=False, strict=False):
        # g2p zh
        words, pinyins = self.g2p_han.g2p(text, strict=strict, sandhi=sandhi)
        tokens = self.retokenize(words, pinyins)
        for token in tokens:
            if token["word"] != token["phones"]:
                token["lang"] = "ZH"
            elif token["word"].isdigit():
                token["lang"] = "NUM"
            elif token["word"].replace("'", "").encode("UTF-8").isalnum():
                # g2p en
                token["phones"] = self.g2p_en.g2p(token["phones"])
                token["lang"] = "EN"
        return tokens
