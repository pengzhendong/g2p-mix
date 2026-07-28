# g2p-mix

`g2p-mix` converts either Mandarin–English or Cantonese–English text into a
source-aligned pronunciation model.

Python 3.10 or newer is required.

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
from g2p_mix import Language, MixedG2P, NativeRenderer

g2p = MixedG2P.mandarin()
result = g2p("你这个 idea，不太 make sense。")

print("Chinese view:", result.projections[Language.CHINESE].text)
print("English view:", result.projections[Language.ENGLISH].text)
print("Phones:", NativeRenderer().render(result))
```

```text
Chinese view: 你这个 <EN>，不太 <EN>。
English view: <ZH> idea，<ZH> make sense。
Phones: ('n', 'i3', 'zh', 'e4', 'g', 'e5', 'AY0', 'D', 'IY1', 'AH0', 'b', 'u2', 't', 'ai4', 'M', 'EY1', 'K', 'S', 'EH1', 'N', 'S')
```

The default Mandarin backend is `PypinyinBackend`. G2PW is optional:

```bash
pip install "g2p-mix[g2pw]"
```

```python
from g2p_mix import G2PWBackend, Language, MixedG2P, PhoneAlphabet

g2p = MixedG2P.mandarin(chinese_backend=G2PWBackend())
result = g2p("银行 ATM 行不行")

print("Chinese view:", result.projections[Language.CHINESE].text)
print(
    "Pinyin:",
    [(unit.text, unit.native) for unit in result.units if unit.alphabet is PhoneAlphabet.PINYIN],
)
```

```text
Chinese view: 银行 <EN> 行不行
Pinyin: [('银', 'yin2'), ('行', 'hang2'), ('行', 'xing2'), ('不', 'bu4'), ('行', 'xing2')]
```

Dictionary candidates are available separately from contextual G2P:

```python
from g2p_mix import MandarinLexicon

lexicon = MandarinLexicon()

assert lexicon.pronunciations("中") == ("zhong1", "zhong4")
print(lexicon.scan("中心"))
```

```text
{'中': ('zhong1', 'zhong4'), '心': ('xin1',)}
```

`MandarinLexicon` queries each unique Han character once and caches the result.
The main G2P pipeline still produces one context-resolved pronunciation, so
rendering and tone sandhi remain unambiguous.

## Cantonese

The default Cantonese backend is
[`ToJyutpingBackend`](https://github.com/CanCLID/ToJyutping). It processes the
whole Cantonese projection once, while each continuous English island occupies
one placeholder position.

```python
from g2p_mix import Language, MixedG2P, PhoneAlphabet

g2p = MixedG2P.cantonese()
result = g2p("你这个 idea。")

print("Normalized:", result.normalized_text)
print("Chinese view:", result.projections[Language.CHINESE].text)
print(
    "Jyutping:",
    [(unit.text, unit.native) for unit in result.units if unit.alphabet is PhoneAlphabet.JYUTPING],
)
```

```text
Normalized: 你這個 idea。
Chinese view: 你這個 <EN>。
Jyutping: [('你', 'nei5'), ('這', 'ze5'), ('個', 'go3')]
```

The previous `PyCantoneseBackend` remains available for explicit use:

```python
from g2p_mix import MixedG2P, PhoneAlphabet, PyCantoneseBackend

g2p = MixedG2P.cantonese(chinese_backend=PyCantoneseBackend())
result = g2p("你好")

print([unit.native for unit in result.units if unit.alphabet is PhoneAlphabet.JYUTPING])
```

```text
['nei5', 'hou2']
```

## IPA output

Backend selection and output alphabet selection are independent. Backends
produce their native phone sets, and a transcriber can then convert the
source-aligned result to IPA:

```python
from g2p_mix import IpaRenderer, MixedG2P, PhoneAlphabet

g2p = MixedG2P.mandarin(output_alphabet=PhoneAlphabet.IPA)
result = g2p("中国 idea")

print(result.phones)
print(IpaRenderer().render(result))
```

```text
('ʈ͡ʂ', 'ʊ', 'ŋ', 'k', 'w', 'o', 'a', 'ɪ', 'd', 'i', 'ə')
('ʈ͡ʂ', 'ʊ', 'ŋ˥˥', 'k', 'w', 'o˧˥', 'a', 'ɪ', 'd', 'ˈi', 'ə')
```

Structured IPA keeps lexical tone contours and phone-level English stress
separate from segment symbols. Every converted unit also retains
`source_alphabet` and `source_phones`. Mandarin, Hong Kong Cantonese, and
General American English are supported.

Use `IpaTranscriber` directly when both native and IPA results are needed:

```python
from g2p_mix import IpaTranscriber

native = MixedG2P.cantonese()("廣東話 idea")
ipa = IpaTranscriber().transcribe(native)
```

## Cross-language phonetic similarity

Install the optional PanPhon backend:

```bash
pip install "g2p-mix[similarity]"
```

Then compare any source-aligned pronunciation results:

```python
from g2p_mix import MixedG2P, PhoneticMatcher

g2p = MixedG2P.mandarin(tone_sandhi=False)
matcher = PhoneticMatcher()

near = matcher.compare(g2p("西"), g2p("she"))
far = matcher.compare(g2p("西"), g2p("key"))

assert near.score > far.score
print(near.score, near.alignment)
```

The matcher converts both inputs to atomic IPA segments, obtains articulatory
feature costs from PanPhon, and performs deterministic weighted sequence
alignment locally. Similarity is currently segmental: tone and stress are
preserved in the IPA result but intentionally excluded from the score.
PanPhon is loaded lazily and is not required for G2P or IPA output.

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
│           ├── alphabet
│           ├── source_alphabet / source_phones
│           └── tone_contour / stress_marks
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

    def predict(self, request): ...
```

Use it without modifying the pipeline:

```python
g2p = MixedG2P.mandarin(chinese_backend=MyMandarinBackend())
```

The same contract supports additional Mandarin, Cantonese, and English
implementations. Language-specific tone rules are pronunciation processors;
phone-set conversion is implemented by transcribers, and renderers only
format results.

Backends and models are loaded lazily. Importing `g2p_mix` does not download a
model, change offline environment variables, or initialize third-party G2P
libraries.

## CLI

```bash
g2p_mix "你这个 idea。" --mode mandarin
g2p_mix "你这个 idea。" --mode cantonese --format json
g2p_mix "你这个 idea。" --mode mandarin --alphabet ipa
```

## Development

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
python -m pytest
```

The normal test suite uses an injected converter to verify G2PW projection and
alignment without downloading a model. Run the opt-in real-model smoke test
with:

```bash
python -m pip install -e ".[g2pw,test]"
G2P_MIX_TEST_G2PW=1 python -m pytest -m g2pw
```
