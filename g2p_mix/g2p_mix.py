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

from .g2p_jyut import G2pJyut
from .g2p_pth import G2pPth
from .token import Token


class G2pMix:
    def __init__(self, jyut=False, strict=False, use_g2pw=False):
        self.g2per = G2pJyut() if jyut else G2pPth(strict, use_g2pw)
        self.jyut = jyut

    def g2p(self, text, sandhi=False, return_seg=False):
        seg_tokens = list(map(lambda x: Token(*x), self.g2per.g2p(text)))
        if not self.jyut and sandhi:
            seg_tokens = self.g2per.sandhi(seg_tokens)

        tokens = []
        for token in seg_tokens:
            if token.lang != "ZH":
                tokens.append(token)
                continue
            # split pinyin into initial and final
            for idx, pinyin in enumerate(token.phones):
                initial, final, tone = self.g2per.parse(pinyin)
                token.phones[idx] = [initial, final + tone]
                tokens.append(Token(token.word[idx], token.pos, token.phones[idx]))
        # 中文保留分词结果 or 中文分词结果拆成单个汉字
        return tokens if not return_seg else seg_tokens
