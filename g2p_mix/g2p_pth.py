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

from functools import partial

from pypinyin import Style
from pypinyin.converter import UltimateConverter

from .tone_sandhi import ToneSandhi
from .utils import parse_pinyin, posseg_cut


class G2pPth:
    def __init__(self, strict, use_g2pw):
        if use_g2pw:
            from modelscope import snapshot_download
            from pypinyin_g2pw import G2PWPinyin

            repo_dir = snapshot_download("pengzhendong/g2pw")
            self.converter = G2PWPinyin(
                model_dir=f"{repo_dir}/G2PWModel",
                model_source=f"{repo_dir}/bert-base-chinese",
                neutral_tone_with_five=True,
            )._converter
        else:
            # tone_sandhi rules of pypinyin is not perfect
            self.converter = UltimateConverter(neutral_tone_with_five=True)
        self.convert = partial(
            self.converter.convert,
            style=Style.TONE3,
            heteronym=False,
            errors="default",
            strict=True,
        )
        self.parse = partial(parse_pinyin, strict=strict)
        self.sandhi = ToneSandhi().modified_tone

    def g2p(self, text):
        chinese_text, words = list(posseg_cut(text))
        pinyins = self.convert(chinese_text)
        for word, pos in words:
            if pos in ["eng", "m", "x"]:
                yield word, pos, word
            else:
                yield word, pos, sum(pinyins[:len(word)], [])
                del pinyins[:len(word)]
