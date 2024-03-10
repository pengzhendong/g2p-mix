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

import jieba

from .token import Token


class ToneSandhi:
    def __init__(self):
        self.must_neural_tone_words = (
            "麻烦 麻利 鸳鸯 高粱 骨头 骆驼 马虎 首饰 馒头 馄饨 风筝 难为 队伍 阔气 闺女"
            "门道 锄头 铺盖 铃铛 铁匠 钥匙 里脊 里头 部分 那么 道士 造化 迷糊 连累 这么"
            "这个 运气 过去 软和 转悠 踏实 跳蚤 跟头 趔趄 财主 豆腐 讲究 记性 记号 认识"
            "规矩 见识 裁缝 补丁 衣裳 衣服 衙门 街坊 行李 行当 蛤蟆 蘑菇 薄荷 葫芦 葡萄"
            "萝卜 荸荠 苗条 苗头 苍蝇 芝麻 舒服 舒坦 舌头 自在 膏药 脾气 脑袋 脊梁 能耐"
            "胳膊 胭脂 胡萝 胡琴 胡同 聪明 耽误 耽搁 耷拉 耳朵 老爷 老实 老婆 老头 老太"
            "翻腾 罗嗦 罐头 编辑 结实 红火 累赘 糨糊 糊涂 精神 粮食 簸箕 篱笆 算计 算盘"
            "答应 笤帚 笑语 笑话 窟窿 窝囊 窗户 稳当 稀罕 称呼 秧歌 秀气 秀才 福气 祖宗"
            "砚台 码头 石榴 石头 石匠 知识 眼睛 眯缝 眨巴 眉毛 相声 盘算 白净 痢疾 痛快"
            "疟疾 疙瘩 疏忽 畜生 生意 甘蔗 琵琶 琢磨 琉璃 玻璃 玫瑰 玄乎 狐狸 状元 特务"
            "牲口 牙碜 牌楼 爽快 爱人 热闹 烧饼 烟筒 烂糊 点心 炊帚 灯笼 火候 漂亮 滑溜"
            "溜达 温和 清楚 消息 浪头 活泼 比方 正经 欺负 模糊 槟榔 棺材 棒槌 棉花 核桃"
            "栅栏 柴火 架势 枕头 枇杷 机灵 本事 木头 木匠 朋友 月饼 月亮 暖和 明白 时候"
            "新鲜 故事 收拾 收成 提防 挖苦 挑剔 指甲 指头 拾掇 拳头 拨弄 招牌 招呼 抬举"
            "护士 折腾 扫帚 打量 打算 打点 打扮 打听 打发 扎实 扁担 戒指 懒得 意识 意思"
            "情形 悟性 怪物 思量 怎么 念头 念叨 快活 忙活 志气 心思 得罪 张罗 弟兄 开通"
            "应酬 庄稼 干事 帮手 帐篷 希罕 师父 师傅 巴结 巴掌 差事 工夫 岁数 屁股 尾巴"
            "少爷 小气 小伙 将就 对头 对付 寡妇 家伙 客气 实在 官司 学问 学生 字号 嫁妆"
            "媳妇 媒人 婆家 娘家 委屈 姑娘 姐夫 妯娌 妥当 妖精 奴才 女婿 头发 太阳 大爷"
            "大方 大意 大夫 多少 多么 外甥 壮实 地道 地方 在乎 困难 嘴巴 嘱咐 嘟囔 嘀咕"
            "喜欢 喇嘛 喇叭 商量 唾沫 哑巴 哈欠 哆嗦 咳嗽 和尚 告诉 告示 含糊 吓唬 后头"
            "名字 名堂 合同 吆喝 叫唤 口袋 厚道 厉害 千斤 包袱 包涵 匀称 勤快 动静 动弹"
            "功夫 力气 前头 刺猬 刺激 别扭 利落 利索 利害 分析 出息 凑合 凉快 冷战 冤枉"
            "冒失 养活 关系 先生 兄弟 便宜 使唤 佩服 作坊 体面 位置 似的 伙计 休息 什么"
            "人家 亲戚 亲家 交情 云彩 事情 买卖 主意 丫头 丧气 两口 东西 东家 世故 不由"
            "不在 下水 下巴 上头 上司 丈夫 丈人 一辈 那个 菩萨 父亲 母亲 咕噜 邋遢 费用"
            "冤家 甜头 介绍 荒唐 大人 泥鳅 幸福 熟悉 计划 扑腾 蜡烛 姥爷 照顾 喉咙 吉他"
            "弄堂 蚂蚱 凤凰 拖沓 寒碜 糟蹋 倒腾 报复 逻辑 盘缠 喽啰 牢骚 咖喱 扫把 惦记"
        ).split()
        self.must_not_neural_tone_words = (
            "孔子 墨子 孟子 庄子 荀子 韩非子 男子 女子 分子 原子 量子 莲子 石子 瓜子 电子"
            "人人 虎虎 软软 暖暖 糯糯"
        ).split()
        self.punc = "：，；。？！“”‘’':,;.?!"

    # merge single "不" and the word behind it
    def _merge_bu(self, tokens):
        new_tokens = []
        last_token = Token("")
        for token in tokens:
            if last_token.word == "不":
                token.push_front(last_token)
            if token.word != "不":
                new_tokens.append(token)
            last_token = token
        if last_token.word == "不":
            last_token.pos = "d"
            new_tokens.append(last_token)
        return new_tokens

    def _bu_sandhi(self, token):
        # e.g. 看不懂
        if len(token.word) == 3 and token.word[1] == "不":
            token.sandhi("5", 1)
        else:
            for i, char in enumerate(token.word):
                # "不" before tone4 should be bu2, e.g. 不怕
                if (
                    char == "不"
                    and i + 1 < len(token.word)
                    and token.phones[i + 1].endswith("4")
                ):
                    token.sandhi("2", i)

    def _merge_yi(self, tokens):
        new_tokens = []
        # function 1
        i = 0
        while i < len(tokens):
            # merge "一" and reduplication words around it, e.g. "听","一","听" ->"听一听"
            if (
                i - 1 >= 0
                and tokens[i].word == "一"
                and i + 1 < len(tokens)
                and tokens[i - 1].word == tokens[i + 1].word
                and tokens[i - 1].pos == "v"
            ):
                new_tokens[-1].push_back(tokens[i]).push_back(tokens[i + 1])
                i += 2
            # merge single  "一" and the word behind it
            elif new_tokens and new_tokens[-1].word == "一":
                new_tokens[-1].push_back(tokens[i])
                i += 1
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def _yi_sandhi(self, token):
        # "一" between reduplication words should be yi5, e.g. 看一看
        if (
            len(token.word) == 3
            and token.word[1] == "一"
            and token.word[0] == token.word[-1]
        ):
            token.sandhi("5", 1)
        else:
            for i, char in enumerate(token.word):
                if char == "一" and i + 1 < len(token.word):
                    # "一" before tone4 should be yi2, e.g. 一段
                    if token.phones[i + 1].endswith("4"):
                        token.sandhi("2", i)
                    # "一" before non-tone4 should be yi4, e.g. 一天
                    elif token.word[i + 1] not in self.punc:
                        token.sandhi("4", i)

    def _merge_reduplication(self, tokens):
        new_tokens = []
        for token in tokens:
            if new_tokens and token.word == new_tokens[-1].word:
                new_tokens[-1].push_back(token)
            else:
                new_tokens.append(token)
        return new_tokens

    def _is_reduplication(self, word):
        return len(word) == 2 and word[0] == word[1]

    # the meaning of jieba pos tag: https://blog.csdn.net/weixin_44174352/article/details/113731041
    def _neural_sandhi(self, token):
        # reduplication words for n. and v. e.g. 奶奶, 试试, 旺旺
        if (
            self._is_reduplication(token.word)
            and token.pos in {"n", "v", "a"}
            and token.word not in self.must_not_neural_tone_words
        ):
            token.sandhi("5", 1)
        if (
            (
                token.word[-1] in "们子"
                and token.pos in {"r", "n"}
                and token.word not in self.must_not_neural_tone_words
            )
            # e.g. 桌上, 地下, 家里
            or (token.word[-1] in "上下里" and token.pos in {"s", "l", "f"})
            # e.g. 上来, 下去
            or (token.word[-2] in "上下进出回过起开" and token.word[-1] in "来去")
            or token.word[-2:] in self.must_neural_tone_words
        ):
            token.sandhi("5", -1)
        # 个做量词
        ge_idx = token.word.find("个")
        if (
            ge_idx >= 1
            and (
                token.word[ge_idx - 1].isnumeric()
                or token.word[ge_idx - 1] in "几有两半多各整每做是"
            )
        ) or token.word == "个":
            token.sandhi("5", ge_idx)
        # e.g. 骆驼祥子 => luo4 tuo5 xiang2 zi3
        idx = 0
        for sub_word in jieba.cut_for_search(token.word):
            if len(sub_word) < 2 or sub_word[-2:] not in self.must_neural_tone_words:
                idx += len(sub_word)
                continue
            token.sandhi("5", idx + 1)
        return token

    # the last char of first word and the first char of second word is tone_three
    # e.g. 小 火车 => 小火车
    def _merge_continuous_three_tones(self, tokens):
        new_tokens = []
        merge_last = [False] * len(tokens)
        for i, token in enumerate(tokens):
            if (
                i - 1 >= 0
                and tokens[i - 1].get_tone(-1) == "3"
                and tokens[i].get_tone(0) == "3"
                and not merge_last[i - 1]
                and not self._is_reduplication(tokens[i - 1].word)
                and len(tokens[i - 1].word) + len(tokens[i].word) <= 3
            ):
                new_tokens[-1].push_back(token)
                merge_last[i] = True
            else:
                new_tokens.append(token)
        return new_tokens

    def _all_tone_three(self, pinyins):
        return all(pinyin.endswith("3") for pinyin in pinyins)

    def _three_sandhi(self, token):
        if len(token.word) == 2 and self._all_tone_three(token.phones):
            token.sandhi("2", 0)
        elif len(token.word) == 3:
            sub_words = list(jieba.cut_for_search(token.word))
            # e.g. 广场/舞、字母/表、所有/人、九九九
            if len(sub_words[0]) != 1 and self._all_tone_three(token.phones[:2]):
                token.sandhi("2", 0)
            # e.g. 广场/舞、纸/老虎、好/喜欢
            if self._all_tone_three(token.phones[1:]):
                token.sandhi("2", 1)
        # split idiom into two words who's length is 2
        elif len(token.word) == 4:
            if self._all_tone_three(token.phones[:2]):
                token.sandhi("2", 1)
            if self._all_tone_three(token.phones[2:]):
                token.sandhi("2", 3)

    def _merge_er(self, tokens):
        new_tokens = []
        for token in tokens:
            if new_tokens and token.word == "儿":
                new_tokens[-1].push_back(token)
            else:
                new_tokens.append(token)
        return new_tokens

    def modified_tone(self, tokens):
        tokens = self._merge_bu(tokens)
        tokens = self._merge_yi(tokens)
        tokens = self._merge_reduplication(tokens)
        tokens = self._merge_continuous_three_tones(tokens)
        tokens = self._merge_er(tokens)
        for token in tokens:
            if len(token.word) < 2 or token.lang != "ZH":
                continue
            self._bu_sandhi(token)
            self._yi_sandhi(token)
            self._neural_sandhi(token)
            self._three_sandhi(token)
        return tokens
