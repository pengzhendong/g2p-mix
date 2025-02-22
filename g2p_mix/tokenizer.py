# Copyright (c) 2025, Zhendong Peng (pzd17@tsinghua.org.cn)
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

import re

import pycantonese
from jieba.posseg import cut
from pycantonese import pos_tag
from pypinyin.constants import RE_HANS
from pypinyin.seg.simpleseg import seg


class Tokenizer:
    def __init__(self):
        self.pattern = r"([\u4e00-\u9fff]+)|(\d+)|([^\w\s])|([a-zA-Z]+(?:'[a-zA-Z]+)?)"

    def get_language(self, word):
        """
        Get the language of a word:
        - "ZH" for Chinese
        - "EN" for English
        - "NUM" for number
        - "SYM" for symbols
        """
        if RE_HANS.match(word):
            return "ZH"
        elif word.isdigit():
            return "NUM"
        elif word.replace("'", "").isalnum():
            return "EN"
        return "SYM"

    def tokenize(self, text):
        """
        Segment a multilingual text into:
        - zh: Chinese text
        - m: numbers
        - x: punctuations
        - eng: English text
        """
        matches = re.finditer(self.pattern, text)
        pairs = []
        for match in matches:
            for i, value in enumerate(match.groups()):
                if value is None:
                    continue
                pairs.append((value, ["zh", "m", "x", "en"][i]))
        # TODO: support SpaCy
        return pairs

    def posseg_cut(self, text, jyut=False, tagset="universal"):
        """
        Cut the text into Chinese text (for g2pw), English text, words and pos tags.
        """
        assert tagset in ("universal", "hkcancor")
        chinese_text = ""
        words = []
        for text, pos in self.tokenize(text):
            if pos == "zh":
                chinese_text += text
                if jyut:
                    words.extend(pos_tag(pycantonese.segment(text), tagset))
                else:
                    # jieba cut
                    for word, pos in cut(text):
                        if len(word) < 4:
                            words.append((word, pos))
                        else:
                            # pypinyin cut: 市场行情 => 市场 行情
                            words.extend([(w, pos) for w in seg(word)])
            else:
                if jyut and pos == "x":
                    pos = "PUNCT"
                words.append((text, pos))
        words = [(word, pos) for word, pos in words]
        return chinese_text, words
