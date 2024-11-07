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

from pypinyin.constants import RE_HANS

from .g2p_eng import G2pEn


g2p_en = G2pEn()


class Token:
    def __init__(self, word, pos=None, phones=None):
        self.word = word
        self.pos = pos
        self.phones = phones if phones else word
        if RE_HANS.match(word):
            self.lang = "ZH"
        elif word.isdigit():
            self.lang = "NUM"
        elif word.replace("'", "").isalnum():
            self.phones = [phone for phone in g2p_en.g2p(self.word) if phone != " "]
            self.pos = "eng"
            self.lang = "EN"
        else:
            self.lang = "SYM"

    def __getitem__(self, item):
        return self.__dict__[item]

    def tone(self, index):
        if self.lang != "ZH" or index >= len(self.phones):
            return -1
        return self.phones[index][-1]

    def sandhi(self, tone, index=0):
        assert index < len(self.phones)
        self.phones[index] = self.phones[index][:-1] + str(tone)

    def push_front(self, token):
        self.word = token.word + self.word
        self.phones = token.phones + self.phones
        return self

    def push_back(self, token):
        self.word = self.word + token.word
        self.phones = self.phones + token.phones
        return self

    def __str__(self):
        items = {k: v for k, v in self.__dict__.items() if v is not None}
        return json.dumps(items, ensure_ascii=False)

    def __repr__(self):
        return self.__str__()
