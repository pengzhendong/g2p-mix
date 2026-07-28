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
not need to run once per island. English POS analysis retains punctuation and
represents each Chinese island as one `<ZH>` token, then maps tags only back to
English target-token IDs.

The built-in English backend composes four replaceable layers:

```text
CMU lexicon candidates
  -> one-pass NLTK context analysis
  -> POS-aware homograph resolver
  -> g2p-en OOV predictor
```

Only the last layer is neural and it runs after dictionary lookup and compound
segmentation miss. The resolver receives all CMUdict candidates, applies
context rules where supported, and preserves dictionary priority otherwise.
The interfaces between these layers keep future Flite or other
deployment-oriented implementations independent of the mixed-text pipeline.

Flite is the preferred native deployment candidate, not a bundled Python
dependency. A production adapter should bind its C lexicon/LTS API behind
`EnglishLexicon` and `EnglishOovPredictor`; invoking the test-only
`lex_lookup` executable once per token would be too slow and too fragile for a
public backend. Until a maintained native binding exists, the Python default
remains CMUdict plus contextual resolution and the g2p-en OOV predictor.

Backends preserve one unit per source character. Under the default `strict`
unknown-character policy, a missing pronunciation is an error. Under
`preserve`, built-in Chinese backends emit an explicitly marked unknown unit
with no phones; the pipeline retains its alignment and adds a diagnostic
warning. Unknown units remain empty when native or IPA output is rendered.
Backend fallback is an explicit composition of two compatible backends rather
than an implicit dictionary guess.

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
│           ├── is_unknown
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
Backends translate dependency failures into `BackendError`, allowing an
explicit `FallbackBackend` to retry without hiding programming errors or
process interruption.

## Internal modules

- `g2p_mix.pipeline.G2PPipeline`: mixed-language orchestration
- `g2p_mix.validation`: backend and processor alignment contracts
- `g2p_mix.profiles`: language-specific backend, normalizer, and processor
  composition
- `g2p_mix.text`: lossless normalization, tokenization, and projections
- `g2p_mix.processors`: pronunciation transformations such as Mandarin tone
  sandhi
- `g2p_mix.transcription`: structured phone-alphabet conversion
- `g2p_mix.renderers`: final tone and stress rendering of units already in the
  requested alphabet
- `g2p_mix.similarity`: pluggable phonetic distance and alignment
- `g2p_mix.models`: source-aligned domain objects

These modules are extension points rather than compatibility aliases for the
old root-level API.
