# g2p_mix

```bash
$ pip install g2p_mix
$ python
```

```python
>>> from g2p_mix import g2p_mix
>>> g2p_mix.g2p("你这个idea，不太make sense。")
```

```json
[
  { "word": "你", "phones": ["n", "i3"] },
  { "word": "这", "phones": ["zh", "e4"] },
  { "word": "个", "phones": ["g", "e4"] },
  { "word": "idea", "phones": ["AY0", "D", "IY1", "AH0"] },
  { "word": "，", "phones": "，" },
  { "word": "不", "phones": ["b", "u4"] },
  { "word": "太", "phones": ["t", "ai4"] },
  { "word": "make", "phones": ["M", "EY1", "K"] },
  { "word": "sense", "phones": ["S", "EH1", "N", "S"] },
  { "word": "。", "phones": "。" }
]
```
