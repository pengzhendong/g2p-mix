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


class ToneSandhi:
    def __init__(self):
        self.digits = list("零一二三四五六七八九十")
        self.neural_tone_words = (
            "麻烦 麻利 鸳鸯 高粱 骨头 骆驼 马虎 首饰 馒头 馄饨 风筝 难为 队伍 阔气 闺女 门道 锄头 铺盖 铃铛 铁匠 钥匙 里脊 里头 部分 "
            "那么 道士 造化 迷糊 连累 这么 这个 运气 过去 软和 转悠 踏实 跳蚤 跟头 趔趄 财主 豆腐 讲究 记性 记号 认识 规矩 见识 裁缝 "
            "补丁 衣裳 衣服 衙门 街坊 行李 行当 蛤蟆 蘑菇 薄荷 葫芦 葡萄 萝卜 荸荠 苗条 苗头 苍蝇 芝麻 舒服 舒坦 舌头 自在 膏药 脾气 "
            "脑袋 脊梁 能耐 胳膊 胭脂 胡萝 胡琴 胡同 聪明 耽误 耽搁 耷拉 耳朵 老爷 老实 老婆 戏弄 将军 翻腾 罗嗦 罐头 编辑 结实 红火 "
            "累赘 糨糊 糊涂 精神 粮食 簸箕 篱笆 算计 算盘 答应 笤帚 笑语 笑话 窟窿 窝囊 窗户 稳当 稀罕 称呼 秧歌 秀气 秀才 福气 祖宗 "
            "砚台 码头 石榴 石头 石匠 知识 眼睛 眯缝 眨巴 眉毛 相声 盘算 白净 痢疾 痛快 疟疾 疙瘩 疏忽 畜生 生意 甘蔗 琵琶 琢磨 琉璃 "
            "玻璃 玫瑰 玄乎 狐狸 状元 特务 牲口 牙碜 牌楼 爽快 爱人 热闹 烧饼 烟筒 烂糊 点心 炊帚 灯笼 火候 漂亮 滑溜 溜达 温和 清楚 "
            "消息 浪头 活泼 比方 正经 欺负 模糊 槟榔 棺材 棒槌 棉花 核桃 栅栏 柴火 架势 枕头 枇杷 机灵 本事 木头 木匠 朋友 月饼 月亮 "
            "暖和 明白 时候 新鲜 故事 收拾 收成 提防 挖苦 挑剔 指甲 指头 拾掇 拳头 拨弄 招牌 招呼 抬举 护士 折腾 扫帚 打量 打算 打扮 "
            "打听 打发 扎实 扁担 戒指 懒得 意识 意思 悟性 怪物 思量 怎么 念头 念叨 别人 快活 忙活 志气 心思 得罪 张罗 弟兄 开通 应酬 "
            "庄稼 干事 帮手 帐篷 希罕 师父 师傅 巴结 巴掌 差事 工夫 岁数 屁股 尾巴 少爷 小气 小伙 将就 对头 对付 寡妇 家伙 客气 实在 "
            "官司 学问 字号 嫁妆 媳妇 媒人 婆家 娘家 委屈 姑娘 姐夫 妯娌 妥当 妖精 奴才 女婿 头发 太阳 大爷 大方 大意 大夫 多少 多么 "
            "外甥 壮实 地道 地方 在乎 困难 嘴巴 嘱咐 嘟囔 嘀咕 喜欢 喇嘛 喇叭 商量 唾沫 哑巴 哈欠 哆嗦 咳嗽 和尚 告诉 告示 含糊 吓唬 "
            "后头 名字 名堂 合同 吆喝 叫唤 口袋 厚道 厉害 千斤 包袱 包涵 匀称 勤快 动静 动弹 功夫 力气 前头 刺猬 刺激 别扭 利落 利索 "
            "利害 分析 出息 凑合 凉快 冷战 冤枉 冒失 养活 关系 先生 兄弟 便宜 使唤 佩服 作坊 体面 位置 似的 伙计 休息 什么 人家 亲戚 "
            "亲家 交情 云彩 事情 买卖 主意 丫头 丧气 两口 东西 东家 世故 不由 下水 下巴 上头 上司 丈夫 丈人 一辈 那个 菩萨 父亲 母亲 "
            "咕噜 邋遢 费用 冤家 甜头 介绍 荒唐 大人 泥鳅 幸福 熟悉 计划 扑腾 蜡烛 姥爷 照顾 喉咙 吉他 弄堂 蚂蚱 凤凰 拖沓 寒碜 糟蹋 "
            "倒腾 报复 逻辑 盘缠 喽啰 牢骚 咖喱 扫把 惦记 上来 下来 进来 出来 回来 过来 起来 上去 下去 进去 出去 回去 过去 几个 有个 "
            "两个 半个 多个 各个 整个 每个 做个 是个 房子 车子 胖子 瘦子 矮子 垫子 饺子 妻子 儿子 桌子 椅子 嫂子 屋子 面子 样子 盘子 "
            "瓶子 脑子 旗子 乱子 乐子 小子 法子 票子 圈子 调子 场子 位子 瞎子 镜子 方子 耗子 帽子 爷爷 奶奶 试试 姐姐 哥哥 弟弟 妹妹 "
            "爸爸 妈妈 姥姥 舅舅 婶婶 嫂嫂 宝宝"
        ).split()
        self.not_neural_tone_words = (
            "女儿 妻儿 胎儿 婴儿 幼儿 少儿 小儿 孤儿 新生儿 婴幼儿"
        ).split()

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
                if prev_char in self.digits + ["万", "月", "第", "初"]:
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
            if not sub_word or sub_word in self.not_neural_tone_words:
                continue
            if sub_word in self.neural_tone_words:
                token.sandhi("5", idx - 1)
            # 儿话音、们字变调
            if sub_word[-1] == ["儿", "们"]:
                token.sandhi("5", idx - 1)
            # 语气词变调
            if (
                len(sub_word) >= 1
                and sub_word[-1] in "吧呢啊呐噻嘛吖嗨呐哦哒滴哩哟喽啰耶喔诶"
            ):
                token.sandhi("5", idx - 1)
            if sub_word in "了着过" and token.pos in {"n", "v", "a"}:
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
