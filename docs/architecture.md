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
  -> Unicode compatibility and written-form normalization
  -> lossless analysis and projections
  -> Chinese and English backends
  -> pronunciation processors
  -> optional IPA transcriber
  -> structured result and final phones
```

## Mixed-text projections

The default normalization chain applies compatibility normalization to Unicode
letters and numbers, expands written forms with WeText, rejects Latin text the
built-in English backend cannot pronounce safely, and optionally converts
Chinese to traditional characters for Cantonese. The built-in English backend
folds decomposable diacritics only for dictionary lookup; normalized spelling
and source spans remain unchanged. Every normalized character retains a source
span; characters expanded from one written form share that form's original
span.

The tokenizer then retains whitespace, punctuation, emoji, and CJK extension
characters. The pipeline builds two views of each sentence:

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
├── base_phones
├── projections
├── tokens
│   ├── TextToken
│   └── Pronunciation
│       └── PronunciationUnit[]
│           ├── text / source_spans
│           ├── phones / alphabet
│           ├── tone
│           ├── source_alphabet / source_phones
│           └── tone_contour / stress_marks
└── warnings
```

`phones` is the final public output. `base_phones` flattens the structured
units without rendering tone or stress. This invariant applies to every
alphabet: an English unit stores `IY` plus a `(phone_index, stress)` entry,
while the native renderer produces `IY1`.

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
Backends must keep tone and stress out of `PronunciationUnit.phones`: numeric
Chinese tone belongs in `tone`, and ARPABET stress belongs in `stress_marks`.
Renderers are the only components that attach those values to output symbols.

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
