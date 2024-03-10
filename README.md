# g2p_mix

```bash
$ pip install g2p_mix
$ python
```

```python
>>> from g2p_mix import G2pMix
>>> G2pMix().g2p("你这个idea, 不太make sense。")
```

```json
[
  { "word": "你", "phones": ["n", "i3"], "lang": "ZH" },
  { "word": "这", "phones": ["zh", "e4"], "lang": "ZH" },
  { "word": "个", "phones": ["g", "e4"], "lang": "ZH" },
  { "word": "idea", "phones": ["AY0", "D", "IY1", "AH0"], "lang": "EN" },
  { "word": ",", "phones": ",", "lang": "SYM" },
  { "word": "不", "phones": ["b", "u4"], "lang": "ZH" },
  { "word": "太", "phones": ["t", "ai4"], "lang": "ZH" },
  { "word": "make", "phones": ["M", "EY1", "K"], "lang": "EN" },
  { "word": "sense", "phones": ["S", "EH1", "N", "S"], "lang": "EN" },
  { "word": "。", "phones": "。", "lang": "SYM" }
]
```
