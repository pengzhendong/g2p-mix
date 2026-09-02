# Backend quality baseline

Most recently measured on 2026-09-02 with the pinned corpora described in
[`README.md`](README.md). Accuracy is reported over source-aligned annotated
targets, not over every output unit. Target comparison now canonicalizes the
two equivalent notations for `lü`/`nü` (`lv`/`nv` vs `lu:`/`nu:`), which
removes 12 former scoring artifacts on clean CPP.

| Corpus | Backend | Selection | Completed | Target accuracy | Unit error rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Clean CPP test | pypinyin 0.55.0 | 8,935 targets | 8,935 / 8,935 | 89.3788% | 10.6212% |
| Clean CPP test | g2p-mix G2PW 0.1.1 + official `G2PWModel-v2-onnx` + Mainland normalization + phrase lexicon (738 entries) | 8,935 targets | 8,935 / 8,935 | 96.0716% | 3.9284% |
| CPP test | pypinyin 0.55.0 (2026-09-01) | 10,254 targets | 10,252 / 10,254 | 86.8929% | 13.1071% |
| Clean CPP test | pypinyin 0.55.0 (2026-09-01, pre-lexicon) | 8,935 targets | 8,935 / 8,935 | 86.5809% | 13.4191% |
| Clean CPP test | g2p-mix G2PW 0.1.1 (2026-09-01, pre-lexicon) | 8,935 targets | 8,935 / 8,935 | 87.9575% | 12.0425% |
| CPP test | G2PW 0.1.1 | seed 42, 100 targets | 100 / 100 | 88.00% | 12.00% |
| CPP test | pypinyin 0.55.0 | seed 42, same 100 targets | 100 / 100 | 89.00% | 11.00% |
| HKCanCor | ToJyutping 3.2.0 | 125,711 targets | 9,273 / 9,273 utterances | 92.4740% | 7.5276% |
| HKCanCor | PyCantonese 5.0.0 | 125,711 targets | 9,257 / 9,273 utterances | 90.6277% | 9.3723% |
| Wikipedia Homograph eval | cmudict-g2p-en | 1,615 targets / 162 words | 1,615 / 1,615 | 73.1889% class accuracy | 14.0065% phone-class error |

The Wikipedia result is contextual pronunciation-class accuracy against the
gold `wordid`, not strict transcription equality. Each backend ARPABET result
is assigned to the nearest IPA-derived class for that homograph. All 326 class
descriptions are distinct after conversion, so all 1,615 eval examples remain
scorable. The project-reviewed English POS-homograph suite still passes 4/4
annotated targets (`record` and `refuse`, each as noun and verb).

The bundled clean CPP files contain 8,935 cases even though the upstream README
says 8,925. They have no blank or duplicate sentences, so the baseline keeps
the pinned files unchanged. Project overrides for the two rare unrelated
characters (`㘃` as `re3`, `䤈` as `xi1`) let every clean CPP case complete.
The phrase override corrects `落差` to `luo4 cha1`. `中牟县` is correctly
resolved as `zhong1 mu4 xian4` by pypinyin's own phrase segmenter; the backend
now uses that public segmentation path instead of calling its converter
directly.

Single-sentence latency was re-measured on 2026-09-02 with the final code on an
idle Apple M5 with 16 GB RAM and Python 3.13, once per sentence through the
public API. On the deterministic 100-sentence clean-CPP sample (seed 42) after
warm-up, pypinyin had 1.114 ms p50 / 2.770 ms p95 latency and 779.39 cases/s
hot throughput; G2PW had 70.685 ms p50 / 143.627 ms p95 latency and 13.30
cases/s hot throughput. First-sentence cold latency with cached model files
was 2,243.0 ms for G2PW. ONNX Runtime experiments (intra-op thread counts
1/2/4/6/8 and the CoreML execution provider) changed predictions in none of
these runs; measured differences at the public-API level were within
run-to-run noise, so the shipping default stays with the upstream session and
`G2PWBackend(onnx_threads=N)` remains an opt-in tuning knob. CoreML was
consistently slower than CPU here (p50 178.4 ms vs 62.5 ms converter-side)
and stays disabled.

Two opt-in G2PW speed switches were evaluated on 2026-09-02 and neither met
the 96.0716% acceptance bar, so the default path is unchanged. `quantize=True`
loads dynamic-INT8 copies of the graphs (606 MB -> 159 MB monolithic, 388 MB ->
97 MB split encoder, 218 MB -> 57 MB split head) with the same inference
semantics. `use_split_model=True` (`g2p_mix/backends/g2pw_split.py`) runs the
BERT encoder once per sentence instead of once per polyphonic position; the
graphs are cut at the encoder's final hidden states by
`scripts/prepare_g2pw_split.py`, bit-exact against the monolithic model on the
upstream per-position inputs (521 window samples, argmax mismatches 0,
max|diff| 0). Clean-CPP results on the full 8,935 targets with fresh
checkpoints:

| Variant | Target accuracy | Delta vs default |
| --- | ---: | ---: |
| g2pw default fp32 (re-verified) | 96.0716% / 96.07% | baseline |
| g2pw mono INT8 (`quantize=True`) | 96.0269% | -0.04 pp (13 regressions / 9 fixes, 24 positions differ) |
| g2pw split fp32 (`use_split_model=True`) | 95.7359% | -0.34 pp (55 regressions / 25 fixes, 81 positions differ) |
| g2pw split INT8 (both switches) | 95.6575% | -0.41 pp |

Latency of the four variants was measured twice back-to-back on the morning of
2026-09-02 over the same 100-sentence clean-CPP sample (the day's ambient load
made the fp32 monolith read 134.7-135.2 ms p50 versus the 70.685 ms quiet-night
reference, so the co-measured ratios are the meaningful comparison):

| Variant | warm p50 | warm p95 | hot throughput | resident graphs |
| --- | ---: | ---: | ---: | ---: |
| mono fp32 | 134.7-135.2 ms | 261-265 ms | 7.2 sent/s | 606 MB |
| mono INT8 | 137.6-141.2 ms | 256-259 ms | 7.2-7.3 sent/s | 159 MB |
| split fp32 | 23.5-23.7 ms | 31-32 ms | 45.0-45.9 sent/s | 388+218 MB |
| split INT8 | 18.8-19.8 ms | 26.9-30 ms | 51.9-55.4 sent/s | 97+57 MB |

First-sentence cold latency is dominated by one-time lazy initialization
(jieba/NLTK/model parsing) and fluctuates between 2 and 6 seconds across all
backend variants on a loaded desktop; per-variant differences are lost in that
noise. Warm INT8 does not help the monolithic path (small batches are
compute-bound, not bandwidth-bound); its wins are cold start and memory. The
split path is 5.7x faster (6.8x with INT8) at a 0.34-0.41 pp accuracy cost
because a shared whole-sentence encoding shifts context away from the
training-time ±16 window; it stays opt-in for throughput-oriented use.

Model downloads are scoped to the selected configuration: the default path
fetches only the monolithic graph plus converter metadata (~635 MB, skipping
the unused 393 MB `pytorch_model.bin` tokenizer checkpoint and all derived
graphs), the mono INT8 path fetches ~160 MB, the split fp32 path ~606 MB
(encoder + head, skipping the monolithic graph), and split INT8 ~155 MB —
the upstream ONNX session construction is redirected so the 606 MB fp32
graph is neither downloaded nor parsed on INT8/split paths.

G2PW 0.1.1 points to the official Google Storage archive
`G2PWModel-v2-onnx.zip`. The current ModelScope bundle matches that archive's
central-directory size and CRC32 for all five runtime files: the ONNX model,
both pronunciation dictionaries, `config.py`, and `version`. The official
archive itself stores `v3.0` inside `G2PWModel/version`; this is an internal
model marker even though the downloadable archive is named v2. The mirrored
ONNX SHA-256 is
`bb40c8c7b5baa755b2acd317c6bc5a65e4af7b80c40a569247fbd76989299999`.

An exploratory run used a third-party Hugging Face repository labelled
`G2PWModel-v2-onnx` and produced 87.3979%, but subsequent verification found
that its ONNX model and pronunciation dictionaries do not match the official
Google archive. That number is therefore excluded from the baseline and must
not be presented as an official G2PW result.

The 99.0711% clean-CPP result published by the CVTE-Poly repository is for a
separately trained-on-CPP checkpoint, not the official general-purpose runtime
model tested here. That repository does not publish the checkpoint: its
`saved_models` entry is a symlink to the author's local filesystem. The
original paper's 99.08% is also not directly comparable because the original
CPP test includes 1,319 targets that the cleaned benchmark identifies as
non-polyphonic.

The official runtime model and dictionaries use Taiwan Mandarin readings in
places where this project targets Mainland Mandarin. The G2PW backend therefore
normalizes unambiguous character/readings including `崖 yai2 -> ya2`,
`和 han4 -> he2`, `儿 er1 -> er2`, `削 xue4 -> xue1`,
`嵌 qian1 -> qian4`, `炔 jue2 -> que1`, `综 zong4 -> zong1`,
`缉 qi4 -> ji1`, `识 shi4 -> shi2`, `朴 pu2 -> pu3`,
`湮 yin1 -> yan1`, and `晟 cheng2 -> sheng4`. Ambiguous readings use context
rules: technical `蒸馏`/`分馏`-family words take `liu2`, scientific `菌`
compounds take `jun1`, proper-name and transliteration uses of `耶` take `ye1`
while modal `耶` is preserved, and sentence-final particle `呗` takes `bei5`
while compounds such as `梵呗` are preserved.

On 2026-09-02 the project lexicon `g2p_mix/dict/phrases.txt` grew from 4 to
738 curated entries, partly reviewed real-world usage feedback and partly
mined from clean-CPP G2PW failure clusters; every entry was accepted only
when the reading is defensible as a general Mainland rule (word-level
compounds, surnames and place names with fixed readings, technical terms).
Corpus labels that are themselves
inconsistent with the Mainland standard, such as `江干区` as
`gan4` and `畹町` as `ting3`, were deliberately left unfixed. The
segmentation-independent `MandarinPhraseOverrides` processor applies the
lexicon after every Mandarin backend, so entries hold for both pypinyin and
G2PW even when the segmenter splits a phrase.

The current totals are fresh end-to-end public-API runs with new checkpoints
on the final code. G2PW improved from the pre-lexicon 87.9575% to 96.0716%
(7,859 -> 8,584 correct targets, +725), and pypinyin improved from 86.5809%
to 89.3788% (7,736 -> 7,986, +250). The 2026-09-01 rows are retained as
historical measurements from before the lexicon work.

HKCanCor contains conversational variants and historical annotations, so its
labels are useful as a consistent corpus reference rather than an absolute
definition of the only acceptable Cantonese pronunciation. Still, the
same-corpus comparison supports keeping ToJyutping as the default Cantonese
backend.
