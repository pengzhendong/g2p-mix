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

import pycantonese
from pyopenhc import OpenHC


class G2pJyut:
    def __init__(self):
        self.converter = OpenHC("s2t")

    def parse(self, jyutping):
        jyutping = pycantonese.parse_jyutping(jyutping)
        assert len(jyutping) == 1
        initial = jyutping[0].onset
        final = jyutping[0].nucleus + jyutping[0].coda
        tone = jyutping[0].tone
        return initial, final, tone

    def g2p(self, text):
        text = self.converter.convert(text)
        words = []
        jyutpings = []
        for word, jyutping in pycantonese.characters_to_jyutping(text):
            words.append(word)
            if jyutping is not None:
                jyutping = re.findall(r"[\D]+\d|\D", jyutping)
            jyutpings.append(jyutping)
        segs = pycantonese.pos_tag(words)
        return [(word, pos, jyutping) for (word, pos), jyutping in zip(segs, jyutpings)]
