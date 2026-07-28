from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Any

from benchmarks.corpora.download import archive_path, provenance, verify_bytes
from benchmarks.corpora.sampling import deterministic_sample


def load_cpp(
    *,
    cache_dir: Path | None = None,
    max_cases: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    archive, metadata = archive_path("cpp", cache_dir=cache_dir)
    root = f"g2pm-{metadata['revision']}"
    with tarfile.open(archive, "r:gz") as bundle:
        sentences = _read_member(
            bundle,
            f"{root}/data/test.sent",
            metadata["files"]["test.sent"],
        ).splitlines()
        labels = _read_member(
            bundle,
            f"{root}/data/test.lb",
            metadata["files"]["test.lb"],
        ).splitlines()
    if len(sentences) != len(labels):
        raise ValueError("CPP test.sent and test.lb have different line counts")

    cases = [_case(index, sentence, label) for index, (sentence, label) in enumerate(zip(sentences, labels))]
    cases = deterministic_sample(cases, max_cases=max_cases, seed=seed)
    return {
        "schema_version": 1,
        "name": "Chinese Polyphones with Pinyin (CPP) test",
        "provenance": provenance(metadata),
        "selection": {
            "split": "test",
            "available_cases": len(sentences),
            "selected_cases": len(cases),
            "seed": seed,
        },
        "cases": cases,
    }


def _read_member(bundle: tarfile.TarFile, name: str, checksum: str) -> str:
    member = bundle.extractfile(name)
    if member is None:
        raise ValueError(f"Missing corpus file: {name}")
    data = member.read()
    verify_bytes(data, checksum, name)
    return data.decode("utf-8")


def _case(index: int, marked: str, label: str) -> dict[str, Any]:
    parts = marked.split("▁")
    if len(parts) != 3 or len(parts[1]) != 1:
        raise ValueError(f"Invalid CPP target marker at test line {index + 1}")
    left, target, right = parts
    text = left + target + right
    return {
        "id": f"cpp-test-{index + 1:05d}",
        "mode": "mandarin",
        "text": text,
        "tone_sandhi": False,
        "targets": [
            {
                "span": [len(left), len(left) + 1],
                "text": target,
                "expected_native": [label],
            }
        ],
    }
