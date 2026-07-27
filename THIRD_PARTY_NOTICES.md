# Third-party test fixture notices

This repository contains small, fixed upstream excerpts under
`tests/fixtures/third_party`. They are test data only; full corpora, model
weights, download caches, and generated archives are not vendored.

Each source directory includes immutable provenance and hashes in
`SOURCE.json`, the extraction rule and attribution in `NOTICE.md`, the exact
raw excerpt in `raw_excerpt.txt`, normalized expectations in `cases.json`, and
the applicable license in `LICENSE.txt`.

## CC-CEDICT

Four records are extracted from Debian source version
`0.0~repack20260403-1`. CC-CEDICT is a community-maintained dictionary
published by MDBG and references CEDICT, Copyright (C) 1997, 1998 Paul Andrew
Denisowski. The fixture is distributed under CC BY-SA 4.0.

## g2pW README examples

The quick-demo examples come from g2pW commit
`36c3fcce93aebfcb54803d2ad6677023a28ad950`, by Yi-Chang Chen, Yu-Chuan
Chang, Yen-Cheng Chang, and Yi-Ren Yeh. They are licensed under Apache-2.0.
No model weights are included.

## Hong Kong Cantonese Corpus

The selected annotations come from HKCanCor commit
`39aeadf920e0b5ca93d0ad7792c59e740e7bdd65`, created by Luke Kang Kwong.
Users should cite K. K. Luke and May L. Y. Wong (2015), “The Hong Kong
Cantonese Corpus: Design and Uses.” The fixture is licensed under CC BY 4.0.

## CMUdict

Six pronunciation records come from CMUdict commit
`74790861f652b15e4ac49015a90074ad62a27690`, Copyright (C) 1993–2015
Carnegie Mellon University. The upstream BSD-style license is reproduced in
the fixture directory. CMUdict annotations and g2p-mix’s force-spelling policy
are intentionally represented as separate layers.
