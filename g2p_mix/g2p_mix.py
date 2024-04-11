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

from jieba import posseg
from pypinyin import lazy_pinyin, Style
from pypinyin.constants import RE_HANS
from pypinyin.contrib.tone_convert import to_initials, to_finals

from .constants import POSTNASALS
from .token import Token
from .tone_sandhi import ToneSandhi


os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


class G2pMix:
    def __init__(self, use_g2pw=False, repo_id=None, model_dir=None, model_source=None):
        if use_g2pw:
            from modelscope.hub.snapshot_download import snapshot_download
            from pypinyin_g2pw import G2PWPinyin

            if not model_dir or not model_source:
                repo_id = repo_id or "pengzhendong/g2pw"
                repo_dir = snapshot_download(repo_id)
                model_dir = f"{repo_dir}/G2PWModel"
                model_source = f"{repo_dir}/bert-base-chinese"
            self.lazy_pinyin = G2PWPinyin(
                model_dir=model_dir,
                model_source=model_source,
                neutral_tone_with_five=True,
            ).lazy_pinyin
        else:
            self.lazy_pinyin = lazy_pinyin
        self.sandhier = ToneSandhi()
        # add space between ascii and chinese characters
        self.pattern = re.compile(
            r"(?<=[\u4e00-\u9fa5])(?=[\x00-\x19\x21-\x7F])|(?<=[\x00-\x19\x21-\x7F])(?=[\u4e00-\u9fa5])"
        )
        # load dict to fix some badcase of pypinyin
        self.load_dict()

    def load_dict(self):
        # 为、曾、更、长
        # from pypinyin.constants import PINYIN_DICT
        # print(hex(ord("为")), PINYIN_DICT[ord("为")])
        from pypinyin import load_single_dict
        load_single_dict({ord("长"): "cha2ng,zha4ng"}, "tone2")
        load_single_dict({ord("为"): "we4i,we2i"}, "tone2")
        # from pypinyin import load_phrases_dict
        # load_phrases_dict({"长时间": [["cha2ng"], ["shi2"], ["jia1n"]]}, "tone2")
        # from pypinyin_dict.pinyin_data import zdic
        # zdic.load()  # 汉典

    def get_pinyins(self, text):
        segs = list(posseg.cut(text))
        words = [word for word, _ in segs]
        # tone_sandhi rules of pypinyin is not perfect
        pinyins = self.lazy_pinyin(words, neutral_tone_with_five=True, style=Style.TONE3)

        # combine pinyins, english words and punctuations by `posseg.cut`'s output
        idx = 0
        tokens = []
        last_word = " "
        for i, (word, pos) in enumerate(segs):
            next_word = words[i + 1] if i + 1 < len(words) else " "
            if word == " ":
                idx += 1
            elif RE_HANS.match(word):
                tokens.append(Token(word, pos, pinyins[idx : idx + len(word)]))
                idx += len(word)
            # he' ve => he've, ' cause => ' cause
            elif (
                (word.isalnum() and len(last_word) > 1 and last_word[-1] == "'")
                or (word[0] == "'" and len(word) > 1 and last_word != " ")
                or (word == "'" and last_word != " " and next_word != " ")
            ):
                word = tokens[-1].word + word
                tokens[-1] = Token(word, pos)
                idx += 1
            else:
                tokens.append(Token(word, pos))
                idx += 1
            last_word = word
        return tokens

    def g2p(self, text, strict=False, sandhi=False, return_seg=False):
        text = self.pattern.sub(" ", text)
        seg_tokens = self.get_pinyins(text)
        if sandhi:
            seg_tokens = self.sandhier.modified_tone(seg_tokens)
        tokens = []
        for token in seg_tokens:
            if token.lang != "ZH":
                tokens.append(token)
                continue
            # split pinyin into initial and final
            for i, pinyin in enumerate(token.phones):
                tone = pinyin[-1]
                if pinyin[:-1] in POSTNASALS:
                    token.phones[i] = ["", POSTNASALS[pinyin[:-1]] + tone]
                else:
                    initial = to_initials(pinyin, strict=strict)
                    final = to_finals(pinyin, strict=strict) + tone
                    token.phones[i] = [initial, final]
                tokens.append(Token(token.word[i], token.pos, token.phones[i]))
        # 中文保留分词结果 or 中文分词结果拆成单个汉字
        return tokens if not return_seg else seg_tokens
