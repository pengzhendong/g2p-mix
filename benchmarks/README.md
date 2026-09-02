# Benchmarks

`benchmarks` is an offline quality-measurement layer. It is intentionally separate
from unit tests and is not included in the published wheel.

Run the project-authored smoke baseline:

```bash
python -m benchmarks
```

Emit machine-readable results or enforce a regression threshold:

```bash
python -m benchmarks --json
python -m benchmarks --fail-under 1.0
```

Alternative Chinese backends can be measured against the same annotations:

```bash
python -m benchmarks --mandarin-backend g2pw
python -m benchmarks --cantonese-backend pycantonese
```

Run the bundled clean Mandarin and contextual English test sets:

```bash
python -m benchmarks --corpus cpp-clean --mandarin-backend pypinyin
python -m benchmarks --corpus cpp-clean --mandarin-backend g2pw
python -m benchmarks --corpus wikipedia-homographs
```

Long model evaluations can checkpoint every completed case. Re-running the
same dataset and backend configuration with the same file resumes only the
missing cases; a mismatched configuration is rejected:

```bash
python -m benchmarks --corpus cpp-clean --mandarin-backend g2pw \
  --checkpoint /tmp/g2p-mix-cpp-clean-g2pw.jsonl
```

The compact clean CPP test (8,935 cases) and Wikipedia Homograph eval split
(1,615 cases) are checked into the repository and excluded from the wheel. Run
the Cantonese corpus by downloading its pinned source archive into
`~/.cache/g2p-mix/benchmarks` and verifying its SHA-256 checksum:

```bash
python -m benchmarks --corpus hkcancor --cantonese-backend tojyutping
python -m benchmarks --corpus hkcancor --cantonese-backend pycantonese
```

Use `--cache-dir` or `G2P_MIX_EVAL_CACHE` to move the download cache for
downloaded corpora. Downloaded third-party archives are never copied into the
repository or wheel.

Run the project-authored number-normalization and tone-sandhi edge cases:

```bash
python -m benchmarks benchmarks/data/mandarin_normalization_sandhi.json
```

Run the English POS-homograph checks:

```bash
python -m benchmarks benchmarks/data/english_homographs.json
```

The current measured baselines and their limitations are recorded in
[baseline.md](baseline.md).

## Dataset schema

Datasets are UTF-8 JSON objects with `schema_version`, `name`, and `cases`.
Each case contains:

- `id`: stable unique identifier;
- `mode`: `mandarin`, `cantonese`, or benchmark-only `english`;
- `text`: written input;
- `expected_normalized`: optional expected text after Unicode and WeText
  normalization;
- exactly one pronunciation annotation form:
  - `expected_native`: the complete ordered `PronunciationUnit.native` values;
  - `targets`: source-aligned annotations with a half-open `span`, matching
    `text`, and either:
    - the target's exact `expected_native` values; or
    - an `expected_label` and at least two ARPABET `pronunciation_classes` for
      contextual pronunciation-class evaluation;
- `tone_sandhi`: optional Mandarin tone-sandhi setting, defaulting to `true`.

The report includes normalization exact rate, case-level pronunciation exact
rate, target exact rate, pronunciation-class exact rate when applicable, and
unit error rate based on Levenshtein distance.
Normalization only counts cases which provide `expected_normalized`. At most
100 failures are retained in a report, with backend errors prioritized.

The bundled `data/smoke.json` only validates the evaluation machinery. Corpus
data retains a source URL, immutable revision, checksums, and license status
alongside its annotations.

## Corpus provenance

- **Clean CPP** is copied from `NewZsh/polyphone`, revision
  `9fd7524e6a07e94107f681a8ba735f21df79be61`. The upstream README claims 8,925
  cases, while the pinned files contain 8,935 unique marked pairs. That
  repository has no declared license, so its status is recorded as
  `NOASSERTION`.
- **Wikipedia Homograph Data** is copied from `google-research-datasets`,
  revision `8f008f021e88f8b71118a27ae655f1f3121162bc`, under Apache-2.0. The eval
  split contains 1,615 contexts across 162 homographs and is scored by gold
  `wordid` pronunciation class.
- **HKCanCor** comes from `fcbond/hkcancor`, revision
  `39aeadf920e0b5ca93d0ad7792c59e740e7bdd65`, under CC BY 4.0. The adapter
  evaluates character-aligned Jyutping and excludes utterances containing
  Latin letters so the score isolates the Cantonese backend.

The exact archive and member checksums are in
[`corpora/sources.json`](corpora/sources.json).
