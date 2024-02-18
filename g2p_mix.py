import g2p_en
from pypinyin import lazy_pinyin, Style


g2p_e = g2p_en.G2p()


def g2p(text):
    # g2p zh
    chars = list(text)
    initials = lazy_pinyin(chars, neutral_tone_with_five=True, style=Style.INITIALS)
    finals = lazy_pinyin(chars, neutral_tone_with_five=True, style=Style.FINALS_TONE3)

    pinyins = []
    for initial, final in zip(initials, finals):
        if initial == final:
            pinyins.append(initial)
        else:
            pinyins.append([initial, final])

    # tokenize
    i = 0
    items = []
    while i < len(pinyins):
        pinyin = pinyins[i]
        if pinyin != " ":
            items.append({"token": text[i], "phones": pinyin})
        if len(pinyin) == 1 and pinyin.isalpha():
            while i + 1 < len(pinyins) and len(pinyins[i + 1]) == 1:
                i += 1
                pinyin = pinyins[i]
                if pinyin == " ":
                    break
                elif not pinyin.isalpha() and pinyin != "'":
                    items.append({"token": text[i], "phones": pinyin})
                    break
                else:
                    items[-1]["token"] += text[i]
                    items[-1]["phones"] += pinyin
        i += 1
    # g2p en
    for item in items:
        token = item["token"]
        if len(token) > 1 or token.encode("UTF-8").isalpha():
            item["phones"] = g2p_e(item["phones"])
    return items


if __name__ == "__main__":
    print(g2p("钢琴男孩piano-boy，I'm 不 OK。"))
