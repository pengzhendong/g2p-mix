# g2p-mix

[![PyPI](https://img.shields.io/pypi/v/g2p-mix)](https://pypi.org/project/g2p-mix/)
[![License](https://img.shields.io/github/license/pengzhendong/g2p-mix)](LICENSE)

Mixed Chinese–English grapheme-to-phoneme conversion with source alignment.

The package intentionally supports two modes:

- Mandarin + English
- Cantonese + English

Python 3.10 or newer is required.

## Installation

```bash
pip install g2p-mix
```

## Quick start

```python
from g2p_mix import G2P

g2p = G2P()
result = g2p("你这个 idea，不太 make sense。")

print(result.phones)
```

```text
('n', 'i3', 'zh', 'e4', 'g', 'e5', 'AY0', 'D', 'IY1', 'AH0', 'b', 'u2', 't', 'ai4', 'M', 'EY1', 'K', 'S', 'EH1', 'N', 'S')
```

Mandarin is the default. Use Cantonese by changing only the mode:

```python
g2p = G2P("cantonese")
print(g2p("你好 idea").phones)
```

```text
('n', 'ei5', 'h', 'ou2', 'AY0', 'D', 'IY1', 'AH0')
```

`result.phones` is always the final, directly usable output. Numeric tones are
attached in native mode; IPA tone letters and English stress marks are attached
in IPA mode.

Arabic numbers and other written forms are normalized automatically by WeText:

```python
result = G2P()("版本1.0发布于2026年")
print(result.normalized_text)
```

```text
版本一点零发布于二零二六年
```

Unicode compatibility letters and numbers are normalized before TN, so
full-width input such as `ＡＢＣ １２３` is supported without changing Chinese
punctuation. Decomposable English diacritics are folded only for pronunciation
lookup, so `café` retains its spelling and source spans while using the
CMUdict entry for `cafe`. Latin text that cannot be folded safely still raises
`G2PError` instead of silently losing phones.

## IPA

```python
from g2p_mix import G2P

g2p = G2P(output="ipa", tone_sandhi=False)
result = g2p("中国 idea")

print(result.phones)
```

```text
('ʈ͡ʂ', 'ʊ', 'ŋ˥˥', 'k', 'w', 'o˧˥', 'a', 'ɪ', 'd', 'ˈi', 'ə')
```

Base phones without tone or stress remain available separately:

```python
print(result.base_phones)
```

```text
('ʈ͡ʂ', 'ʊ', 'ŋ', 'k', 'w', 'o', 'a', 'ɪ', 'd', 'i', 'ə')
```

## Backends

Built-in Chinese backends are selected by name:

| Mode | Default | Alternatives |
| --- | --- | --- |
| `mandarin` | `pypinyin` | `g2pw` |
| `cantonese` | `tojyutping` | `pycantonese` |

```python
g2p = G2P("mandarin", backend="g2pw")
```

G2PW is optional:

```bash
pip install "g2p-mix[g2pw]"
```

A custom backend object can be passed through the same argument:

```python
g2p = G2P("mandarin", backend=MyMandarinBackend())
```

## Detailed results

Most applications only need `result.phones`. `result.base_phones` always
removes Mandarin and Cantonese tones as well as English stress. Source-aligned
units are available when more detail is required:

```python
for unit in result.units:
    print(unit.text, unit.phones, unit.tone, unit.source_spans)
```

IPA units also retain their original alphabet and phones:

```python
for unit in result.units:
    print(unit.source_alphabet, unit.source_phones)
```

The input remains losslessly reconstructable:

```python
assert result.reconstruct_original() == "中国 idea"
```

## Phonetic similarity

Install the optional PanPhon backend:

```bash
pip install "g2p-mix[similarity]"
```

Then compare text directly through the same `G2P` object:

```python
g2p = G2P("mandarin", tone_sandhi=False)

near = g2p.compare("西", "she")
far = g2p.compare("西", "key")

assert near.score > far.score
print(near.score, near.alignment)
```

Similarity currently uses base phones. Tone and stress remain available in the
structured result but are intentionally excluded from the score. PanPhon is
loaded lazily and is not required for G2P or IPA output.

## CLI

```bash
g2p_mix "你这个 idea。"
g2p_mix "你这个 idea。" --mode cantonese
g2p_mix "你这个 idea。" --output ipa
g2p_mix "银行 ATM" --backend g2pw --format json
```

## Advanced usage

The root package exposes only the simple API:

```python
from g2p_mix import G2P, G2PError, G2PResult
```

Backend protocols, structured models, transcription, projections, and the
internal pipeline live in their respective submodules. See
[Architecture and extension points](https://github.com/pengzhendong/g2p-mix/blob/master/docs/architecture.md).

## Development

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
python -m pytest
```

The normal suite uses an injected converter for G2PW tests and does not
download a model. Run the real-model smoke test explicitly:

```bash
python -m pip install -e ".[g2pw,test]"
G2P_MIX_TEST_G2PW=1 python -m pytest -m g2pw
```

Quality evaluation is separate from unit tests and the published wheel:

```bash
python -m evals
python -m evals --json
python -m evals --fail-under 1.0
python -m evals --corpus cpp --max-cases 100 --seed 42
python -m evals --corpus hkcancor --cantonese-backend tojyutping
python -m evals evals/data/mandarin_normalization_sandhi.json
```

See [evals/README.md](evals/README.md) for the dataset schema, backend
comparison options, reproducible CPP/HKCanCor adapters, metrics, and measured
baselines.
