# Architecture and extension points

The public API is intentionally small:

```python
from g2p_mix import G2P
```

`G2P` is a thin facade. It resolves built-in backend names and configures the
internal pipeline; pronunciation behavior remains separated into focused
components.

```text
G2P facade
  -> language profile
  -> lossless analysis and projections
  -> Chinese and English backends
  -> pronunciation processors
  -> optional IPA transcriber
  -> structured result and final phones
```

## Mixed-text projections

The tokenizer retains whitespace, punctuation, emoji, accented Latin text, and
CJK extension characters. The pipeline builds two views of each sentence:

- Every continuous English island becomes one `<EN>` placeholder in the
  Chinese view.
- Every continuous Chinese island becomes one `<ZH>` placeholder in the
  English view.

Each contextual backend therefore processes its projection once. Disconnected
Chinese or English spans do not become falsely adjacent, and neural backends do
not need to run once per island.

## Result model

```text
G2PResult
├── original_text
├── normalized_text
├── output
├── phones
├── segments
├── projections
├── tokens
│   ├── TextToken
│   └── Pronunciation
│       └── PronunciationUnit[]
│           ├── text / source_spans
│           ├── phones / alphabet
│           ├── tone / stress
│           ├── source_alphabet / source_phones
│           └── tone_contour / stress_marks
└── warnings
```

`phones` is the final public output. `segments` flattens the structured units
without rendering tone or stress.

## Custom backends

Backend types and contracts live under `g2p_mix.backends`:

```python
from g2p_mix.backends import BackendCapabilities
from g2p_mix.models import ChineseDialect, Language, PhoneAlphabet


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

Inject the object through the simple facade:

```python
from g2p_mix import G2P

g2p = G2P("mandarin", backend=MyMandarinBackend())
```

The same contract supports additional Mandarin, Cantonese, and English
implementations. Backend modules and model dependencies are initialized lazily.

## Internal modules

- `g2p_mix.pipeline.G2PPipeline`: orchestration and alignment validation
- `g2p_mix.profiles`: language-specific backend, normalizer, and processor
  composition
- `g2p_mix.text`: lossless normalization, tokenization, and projections
- `g2p_mix.processors`: pronunciation transformations such as Mandarin tone
  sandhi
- `g2p_mix.transcription`: structured phone-alphabet conversion
- `g2p_mix.renderers`: final tone and stress rendering
- `g2p_mix.similarity`: pluggable phonetic distance and alignment
- `g2p_mix.models`: source-aligned domain objects

These modules are extension points rather than compatibility aliases for the
old root-level API.
