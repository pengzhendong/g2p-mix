# g2p-mix

- Cantonese: [pycantonese](https://github.com/jacksonllee/pycantonese)
- English: [g2p_en](https://github.com/Kyubyong/g2p)
- Mandarin: [pypinyin](https://github.com/mozillazg/python-pinyin)

## Usage

```bash
$ pip install g2p-mix
$ python
```

### Mandarin

```python
>>> from g2p_mix import G2pMix
>>> G2pMix().g2p("你这个idea, 不太make sense。", sandhi=True, return_seg=True)
```

```json
[
  {"word": "你", "lang": "ZH", "pos": "r", "phones": [["n", "i3"]]},
  {"word": "这个", "lang": "ZH", "pos": "r", "phones": [["zh", "e4"], ["g", "e5"]]},
  {"word": "idea", "lang": "EN", "pos": null, "phones": ["AY0", "D", "IY1", "AH0"]},
  {"word": ",", "lang": "SYM", "pos": "x", "phones": [","]},
  {"word": "不太", "lang": "ZH", "pos": "d", "phones": [["b", "u2"], ["t", "ai4"]]},
  {"word": "make", "lang": "EN", "pos": null, "phones": ["M", "EY1", "K"]},
  {"word": "sense", "lang": "EN", "pos": null, "phones": ["S", "EH1", "N", "S"]},
  {"word": "。", "lang": "SYM", "pos": "x", "phones": ["。"]},
]
```

### Cantonese

```python
>>> G2pMix(jyut=True).g2p("你这个idea, 不太make sense。", return_seg=True)
```

```json
[
  {"word": "你", "lang": "ZH", "pos": "PRON", "phones": [["n", "ei5"]]},
  {"word": "這個", "lang": "ZH", "pos": "PRON", "phones": [["z", "e3"], ["g", "o3"]]},
  {"word": "idea", "lang": "EN", "pos": null, "phones": ["AY0", "D", "IY1", "AH0"]},
  {"word": ",", "lang": "SYM", "pos": "x", "phones": [","]},
  {"word": "不", "lang": "ZH", "pos": "ADV", "phones": [["b", "at1"]]},
  {"word": "太", "lang": "ZH", "pos": "ADV", "phones": [["t", "aai3"]]},
  {"word": "make", "lang": "EN", "pos": null, "phones": ["M", "EY1", "K"]},
  {"word": "sense", "lang": "EN", "pos": null, "phones": ["S", "EH1", "N", "S"]},
  {"word": "。", "lang": "SYM", "pos": "x", "phones": ["。"]},
]
```
