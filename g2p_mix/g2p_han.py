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

from pypinyin import lazy_pinyin, Style
from pypinyin.contrib.tone_convert import to_initials, to_finals

from .tone_sandhi import ToneSandhi


os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
# 将后鼻音 n, m 当成韵母，同时与声母的 n, m 区分开
postnasals = {
    "n1": "ng1",
    "n2": "ng2",
    "n4": "ng4",
    "m2": "mg2",
    "m3": "mg3",
    "m4": "mg4",
}


class G2pHan:
    def __init__(self, use_g2pw=False, model_dir=None):
        if use_g2pw:
            from modelscope.hub.snapshot_download import snapshot_download
            from pypinyin_g2pw import G2PWPinyin

            model_dir = model_dir or snapshot_download("pengzhendong/g2pw")
            self.lazy_pinyin = G2PWPinyin(
                model_dir=f"{model_dir}/G2PWModel",
                model_source=f"{model_dir}/bert-base-chinese",
                neutral_tone_with_five=True,
            ).lazy_pinyin
        else:
            self.lazy_pinyin = lazy_pinyin
        self.sandhier = ToneSandhi()

    def get_initials_finals(self, word, strict):
        pinyins = self.lazy_pinyin(word, neutral_tone_with_five=True, style=Style.TONE3)
        initials = []
        finals = []
        for pinyin in pinyins:
            if pinyin and not pinyin[-1].isdigit():
                # not a valid pinyin, e.g. english word or punctuation
                initials.append(pinyin)
                finals.append(pinyin)
            elif pinyin in postnasals:
                initials.append("")
                finals.append(postnasals[pinyin])
            else:
                initials.append(to_initials(pinyin, strict=strict))
                finals.append(to_finals(pinyin, strict=strict) + pinyin[-1])
        return initials, finals

    def g2p(self, text, strict=False, sandhi=False):
        # add space between chinese char and alphabet
        text = re.sub(
            r"(?<=[\u4e00-\u9fa5])(?=[a-zA-Z])|(?<=[a-zA-Z])(?=[\u4e00-\u9fa5])",
            " ",
            text,
        )
        words = []
        initials = []
        finals = []
        segs = self.sandhier.pre_merge_for_modify(text)
        for word, pos in segs:
            sub_initials, sub_finals = self.get_initials_finals(word, strict)
            # process english word and punctuations
            if len(sub_initials) == 1 and word == sub_initials[0]:
                if word.encode("UTF-8").isalnum():
                    words.append(word)
                # split the continuous punctuations into single, e.g. "……" => "…", "…"
                else:
                    words.extend(word)
                    sub_initials = list(sub_initials[0])
                    sub_finals = list(sub_finals[0])
            else:
                words.extend(word)
                if sandhi:
                    sub_finals = self.sandhier.modified_tone(word, pos, sub_finals)
            initials.extend(sub_initials)
            finals.extend(sub_finals)

        pinyins = []
        for initial, final in zip(initials, finals):
            if initial == final:
                pinyins.append(initial)
            else:
                pinyins.append([initial, final])
        return words, pinyins
