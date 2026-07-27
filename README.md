# g2p-mix

`g2p-mix` converts either Mandarin–English or Cantonese–English text into a
source-aligned pronunciation model.

The package deliberately exposes two modes only:

- Mandarin + English
- Cantonese + English

It uses a lossless tokenizer and builds two views of every sentence. In the
Chinese view, each continuous English island becomes `<EN>`. In the English
view, each continuous Chinese island becomes `<ZH>`. Contextual backends can
therefore process a sentence once without creating false adjacency between
disconnected spans.

## Mandarin

```python
from g2p_mix import MixedG2P, NativeRenderer

g2p = MixedG2P.mandarin()
result = g2p("你这个 idea，不太 make sense。")

print(result.projections)
print(NativeRenderer().render(result))
```

The default Mandarin backend is `PypinyinBackend`. G2PW is optional:

```bash
pip install "g2p-mix[g2pw]"
```

```python
from g2p_mix import G2PWBackend, MixedG2P

g2p = MixedG2P.mandarin(chinese_backend=G2PWBackend())
```

## Cantonese

```python
from g2p_mix import MixedG2P, NativeRenderer

g2p = MixedG2P.cantonese()
result = g2p("你这个 idea。")

assert result.normalized_text == "你這個 idea。"
print(NativeRenderer().render(result))
```

## Result model

`MixedG2P` returns `G2PResult`, not language-dependent nested phone lists.

```text
G2PResult
├── original_text
├── normalized_text
├── projections
├── tokens
│   ├── TextToken
│   │   ├── normalized span
│   │   ├── source spans
│   │   ├── language
│   │   └── POS
│   └── Pronunciation
│       └── PronunciationUnit[]
│           ├── phones
│           ├── tone
│           ├── stress
│           └── alphabet
└── warnings
```

Whitespace, punctuation, emoji, accented Latin characters, and CJK extension
characters are retained by the analysis layer. Consequently:

```python
result.reconstruct_original() == input_text
```

## Extending backends

Backends are injected through profiles and implement one contract:

```python
class MyMandarinBackend:
    name = "my-mandarin"
    capabilities = BackendCapabilities(
        language=Language.CHINESE,
        dialect=ChineseDialect.MANDARIN,
        alphabet=PhoneAlphabet.PINYIN,
        contextual=True,
        supports_projection=True,
    )

    def predict(self, request):
        ...
```

Use it without modifying the pipeline:

```python
g2p = MixedG2P.mandarin(chinese_backend=MyMandarinBackend())
```

The same contract supports additional Mandarin, Cantonese, and English
implementations. Language-specific tone rules are pronunciation processors;
phone-set conversion is implemented by renderers.

Backends and models are loaded lazily. Importing `g2p_mix` does not download a
model, change offline environment variables, or initialize third-party G2P
libraries.

## CLI

```bash
g2p_mix "你这个 idea。" --mode mandarin
g2p_mix "你这个 idea。" --mode cantonese --format json
```

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest
```
