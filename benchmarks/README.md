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

Run a fixed open corpus. The first run downloads the pinned source archive into
`~/.cache/g2p-mix/benchmarks` and verifies its SHA-256 checksum:

```bash
python -m benchmarks --corpus cpp --mandarin-backend pypinyin
python -m benchmarks --corpus cpp --max-cases 100 --seed 42 --mandarin-backend g2pw
python -m benchmarks --corpus hkcancor --cantonese-backend tojyutping
python -m benchmarks --corpus hkcancor --cantonese-backend pycantonese
```

`--max-cases` uses a reproducible random sample rather than the first records.
Use `--cache-dir` or `G2P_MIX_EVAL_CACHE` to move the download cache. The
third-party archives are never copied into the repository or wheel.

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
- `mode`: `mandarin` or `cantonese`;
- `text`: written input;
- `expected_normalized`: optional expected text after Unicode and WeText
  normalization;
- exactly one pronunciation annotation form:
  - `expected_native`: the complete ordered `PronunciationUnit.native` values;
  - `targets`: source-aligned annotations with a half-open `span`, matching
    `text`, and the target's `expected_native` values;
- `tone_sandhi`: optional Mandarin tone-sandhi setting, defaulting to `true`.

The report includes normalization exact rate, case-level pronunciation exact
rate, target exact rate, and unit error rate based on Levenshtein distance.
Normalization only counts cases which provide `expected_normalized`. At most
100 failures are retained in a report, with backend errors prioritized.

The bundled `data/smoke.json` only validates the evaluation machinery. Larger
open-source evaluations should live outside the wheel and retain a source URL,
immutable revision, extraction procedure, checksums, and license alongside
their annotations.

## Corpus provenance

- **CPP** comes from `kakaobrain/g2pM`, revision
  `170526efad0a3ef9b55a9ad4579f73218f9be06c`, under Apache-2.0. Its test split
  has one marked polyphonic target per sentence.
- **HKCanCor** comes from `fcbond/hkcancor`, revision
  `39aeadf920e0b5ca93d0ad7792c59e740e7bdd65`, under CC BY 4.0. The adapter
  evaluates character-aligned Jyutping and excludes utterances containing
  Latin letters so the score isolates the Cantonese backend.

The exact archive and member checksums are in
[`corpora/sources.json`](corpora/sources.json).
