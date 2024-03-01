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

dirname = os.path.dirname(__file__)
nltk.data.path.insert(0, f"{dirname}/nltk_data")

import g2p_en


class G2pEn:
    def __init__(self):
        self.g2p_e = g2p_en.G2p()
        self.cmudict = cmudict.dict()
        # Remove abbreviations badcase like "HUD" in cmudict.
        for word in ["AI", "HUD"]:
            del self.cmudict[word.lower()]

    def g2p(self, word):
        if word.lower() in self.cmudict:
            return self.cmudict[word.lower()][0]

        is_abbr = (word.isupper() and len(word) <= 4) or (
            word.islower() and len(word) <= 3
        )
        if not is_abbr:
            return self.g2p_e(word)

        phonemes = []
        for ch in word:
            # In abbreviations, "A" should be pronounced as "EY1", not "AH0".
            if ch.upper() == "A":
                phonemes.append("EY1")
            else:
                phonemes.extend(self.g2p_e(ch))
        return phonemes
