# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import jieba


class ToneSandhi:
    def __init__(self):
        dirname = os.path.join(os.path.dirname(__file__), "dict")
        self.digits = [line.strip() for line in open(f"{dirname}/digits.txt")]
        interjections = f"{dirname}/interjections.txt"
        self.interjections = [line.strip() for line in open(interjections)]
        neural_tone_words = f"{dirname}/neural_tone_words.txt"
        self.neural_tone_words = [line.strip() for line in open(neural_tone_words)]
        self.whitelist = [line.strip() for line in open(f"{dirname}/whitelist.txt")]

    def _merge(self, tokens):
        new_tokens = []
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            idx += 1
            if not new_tokens or new_tokens[-1].lang != "ZH" or token.lang != "ZH":
                new_tokens.append(token)
                continue
            last_token = new_tokens[-1]
            # 1. merge single 不, 很, 一 and the word behind it.
            # 2. merge single 儿 and the word before it.
            if last_token.word in ("不", "很", "一") or token.word in ("儿"):
                last_token.push_back(token)
            # 3. merge continuous three tones, e.g. 小/火车 => 小火车, 也/很/好 => 也很/好 => 也很好
            elif (
                last_token.tone(-1) == "3"
                and token.tone(0) == "3"
                and len(last_token.word) <= 2
                and len(token.word) <= 2
                and len(last_token.word) + len(token.word) <= 4
            ):
                last_token.push_back(token)
            # 4. merge 一 and reduplication verbs around it, e.g. 听/一/听 => 听一听
            elif (
                token.word == "一"
                and last_token.pos[0] == "v"
                and idx < len(tokens)
                and last_token.word == tokens[idx].word
            ):
                last_token.push_back(token).push_back(tokens[idx])
                idx += 1
            else:
                new_tokens.append(token)
        return new_tokens

    def _bu_sandhi(self, token):
        # e.g. 看不懂, 去不去
        if len(token.word) == 3 and token.word[1] == "不":
            token.sandhi("5", 1)
        else:
            for i, char in enumerate(token.word):
                # 不 before tone4 should be bu2, e.g. 不怕
                if (
                    char == "不"
                    and i + 1 < len(token.word)
                    and token.tone(i + 1) == "4"
                ):
                    token.sandhi("2", i)

    def _yi_sandhi(self, token):
        # 一 between reduplication words should be yi5, e.g. 看一看
        if (
            len(token.word) == 3
            and token.word[1] == "一"
            and token.word[0] == token.word[-1]
        ):
            token.sandhi("5", 1)
        else:
            for i, char in enumerate(token.word):
                if char != "一":
                    continue
                prev_char = token.word[i - 1] if i > 0 else ""
                next_char = token.word[i + 1] if i + 1 < len(token.word) else ""
                if prev_char in self.digits + ["第", "初"]:
                    token.sandhi("1", i)
                # 无法判断是否为序数时，当成序数不变调处理, e.g. 一月 => yi1 yue4
                elif next_char in ["", "月", "班", "连", "楼"]:
                    token.sandhi("1", i)
                # 一 before tone4 should be yi2, e.g. 一段
                elif token.tone(i + 1) == "4":
                    token.sandhi("2", i)
                # 一 before non-tone4 should be yi4, e.g. 一天
                else:
                    token.sandhi("4", i)

    # finer split a long word into twos for neural & three tone sandhi
    # e.g. 很漂亮 => 很/漂亮, 与好姐妹 => 与/好姐妹
    @staticmethod
    def _split_word(word):
        word_list = jieba.cut_for_search(word)
        word_list = sorted(word_list, key=len, reverse=False)
        first_subword = word_list[0]
        first_begin_idx = word.find(first_subword)
        if first_begin_idx == 0:
            second_subword = word[len(first_subword) :]
            new_word_list = [first_subword, second_subword]
        else:
            second_subword = word[: -len(first_subword)]
            new_word_list = [second_subword, first_subword]
        return new_word_list

    def _neural_sandhi(self, token):
        idx = 0
        for sub_word in self._split_word(token.word):
            idx += len(sub_word)
            if not sub_word or sub_word in self.whitelist:
                continue
            if sub_word in self.neural_tone_words:
                token.sandhi("5", idx - 1)
            # 儿话音、们字变调
            if sub_word[-1] == ["儿", "们"]:
                token.sandhi("5", idx - 1)
            # 语气词变调
            if len(sub_word) >= 1 and sub_word[-1] in self.interjections:
                token.sandhi("5", idx - 1)
            if sub_word in ["了", "着", "过"] and token.pos in {"n", "v", "a"}:
                token.sandhi("5", idx - 1)
        return token

    def _all_tone_three(self, pinyins):
        return all(pinyin.endswith("3") for pinyin in pinyins)

    def _three_sandhi(self, token):
        if len(token.word) == 2:
            if self._all_tone_three(token.phones):
                token.sandhi("2", 0)
        elif len(token.word) >= 3:
            sub_words = self._split_word(token.word)
            if len(sub_words[0]) == 1:
                # e.g. 纸/老虎、与/好姐妹
                if self._all_tone_three(token.phones[1:3]):
                    token.sandhi("2", 1)
                # e.g. 好/喜欢
                elif self._all_tone_three(token.phones[:2]):
                    token.sandhi("2", 0)
            else:
                # e.g. 广场/舞、字母/表、所有/人
                if self._all_tone_three(token.phones[:2]):
                    token.sandhi("2", 0)
                if self._all_tone_three(token.phones[1:3]):
                    # e.g. 坎坎/坷坷 => kan2 kan3 ke2 ke3
                    if len(token.word) <= 3 or not self._all_tone_three(
                        token.phones[2:4]
                    ):
                        token.sandhi("2", 1)
                # e.g. 期待/已久、省委/党校
                if len(token.word) > 3 and self._all_tone_three(token.phones[2:4]):
                    token.sandhi("2", 2)

    def modified_tone(self, tokens):
        tokens = self._merge(tokens)
        for token in tokens:
            if len(token.word) < 2 or token.lang != "ZH":
                continue
            self._bu_sandhi(token)
            self._yi_sandhi(token)
            self._neural_sandhi(token)
            self._three_sandhi(token)
        return tokens


convert = ToneSandhi().modified_tone
