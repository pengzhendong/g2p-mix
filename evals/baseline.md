# Backend quality baseline

Measured on 2026-07-28 with the pinned corpora described in
[`README.md`](README.md). Accuracy is reported over source-aligned annotated
targets, not over every output unit.

| Corpus | Backend | Selection | Completed | Target accuracy | Unit error rate |
| --- | --- | ---: | ---: | ---: | ---: |
| CPP test | pypinyin 0.55.0 | 10,254 targets | 10,252 / 10,254 | 86.8929% | 13.1071% |
| CPP test | G2PW 0.1.1 | seed 42, 100 targets | 100 / 100 | 88.00% | 12.00% |
| CPP test | pypinyin 0.55.0 | seed 42, same 100 targets | 100 / 100 | 89.00% | 11.00% |
| HKCanCor | ToJyutping 3.2.0 | 125,711 targets | 9,273 / 9,273 utterances | 92.4740% | 7.5276% |
| HKCanCor | PyCantonese 5.0.0 | 125,711 targets | 9,257 / 9,273 utterances | 90.6277% | 9.3723% |

The full CPP pypinyin run processed about 502 cases/s. G2PW processed about
4.1 cases/s in the 100-case local run because the public API evaluates one
sentence per call; running all 10,254 cases that way would be a poor local
benchmark. The 100-case result only proves the real model path works and is not
large enough to rank Mandarin backends. A future batched backend API should
precede a full G2PW speed/quality run.

Two CPP utterances are incomplete for pypinyin because unrelated rare
characters (`㘃`, `䤈`) are returned without a usable pronunciation. They remain
in the denominator so the baseline does not silently discard backend errors.

HKCanCor contains conversational variants and historical annotations, so its
labels are useful as a consistent corpus reference rather than an absolute
definition of the only acceptable Cantonese pronunciation. Still, the
same-corpus comparison supports keeping ToJyutping as the default Cantonese
backend.
