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

import re

from pycantonese import characters_to_jyutping, pos_tag
from pyopenhc import OpenHC

from .utils import parse_jyutping


class G2pJyut:
    def __init__(self):
        self.converter = OpenHC("s2t")
        self.parse = parse_jyutping

    def g2p(self, text):
        text = self.converter.convert(text)
        words = []
        jyutpings = []
        for word, jyutping in characters_to_jyutping(text):
            words.append(word)
            if jyutping is not None:
                jyutping = re.findall(r"[\D]+\d|\D", jyutping)
            jyutpings.append(jyutping)
        for (word, pos), jyutping in zip(pos_tag(words), jyutpings):
            yield word, pos, jyutping
