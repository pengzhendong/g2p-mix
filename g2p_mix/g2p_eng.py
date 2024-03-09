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

import nltk
from nltk.corpus import cmudict
import wordninja

dirname = os.path.dirname(__file__)
nltk.data.path.insert(0, f"{dirname}/nltk_data")

import g2p_en


class G2pEn:
    def __init__(self):
        self.g2p_e = g2p_en.G2p()
        self.cmudict = cmudict.dict()
        # Remove abbreviations badcase like "HUD" in cmudict.
        for word in ["AE", "AI", "AR", "IOS", "HUD", "OS"]:
            del self.cmudict[word.lower()]

    def g2p_ch(self, ch):
        assert len(ch) == 1
        # In abbreviations, "A" should be pronounced as "EY1", not "AH0".
        if ch.upper() == "A":
            return ["EY1"]
        return self.g2p_e(ch)

    def g2p_abbr(self, word):
        phones = []
        for ch in word:
            phones.extend(self.g2p_ch(ch))
        return phones

    def g2p(self, word):
        if word.isupper() and len(word) <= 4:
            # e.g. "IT" => "I T"
            return self.g2p_abbr(word)

        word = word.lower()
        if word in self.cmudict:
            return self.cmudict[word][0]
        if len(word) <= 3:
            # e.g. "tts" => "t t s"
            return self.g2p_abbr(word)

        bpes = wordninja.split(word)
        if len(bpes) == 1:
            return self.g2p_e(word)
        # e.g. "autojs" => "auto js"
        phones = []
        for bpe in bpes:
            if len(bpe) == 1:
                phones.extend(self.g2p_ch(bpe))
            else:
                phones.extend(self.g2p(bpe))
        return phones
