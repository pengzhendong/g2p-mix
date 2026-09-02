from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.corpora.download import provenance, source, verify_file
from benchmarks.corpora.sampling import deterministic_sample

DATA_DIR = Path(__file__).parents[1] / "data" / "corpora" / "cpp_clean"


def load_cpp_clean(
    *,
    cache_dir: Path | None = None,
    max_cases: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    del cache_dir  # The compact test split is intentionally bundled.
    metadata = source("cpp-clean")
    sentence_path = DATA_DIR / "test.sent"
    label_path = DATA_DIR / "test.lb"
    verify_file(sentence_path, metadata["files"]["test.sent"], "cpp_clean/test.sent")
    verify_file(label_path, metadata["files"]["test.lb"], "cpp_clean/test.lb")

    sentences = sentence_path.read_text(encoding="utf-8").splitlines()
    labels = label_path.read_text(encoding="utf-8").splitlines()
    if len(sentences) != len(labels):
        raise ValueError("Clean CPP test.sent and test.lb have different line counts")

    cases = [_case(index, sentence, label) for index, (sentence, label) in enumerate(zip(sentences, labels))]
    cases = deterministic_sample(cases, max_cases=max_cases, seed=seed)
    return {
        "schema_version": 1,
        "name": "Clean CPP test from CVTE-Poly",
        "provenance": provenance(metadata),
        "selection": {
            "split": "test",
            "available_cases": len(sentences),
            "selected_cases": len(cases),
            "seed": seed,
            "upstream_readme_cases": 8925,
        },
        "cases": cases,
    }


def _case(index: int, marked: str, label: str) -> dict[str, Any]:
    parts = marked.split("▁")
    if len(parts) != 3 or len(parts[1]) != 1:
        raise ValueError(f"Invalid clean CPP target marker at test line {index + 1}")
    left, target, right = parts
    return {
        "id": f"cpp-clean-test-{index + 1:05d}",
        "mode": "mandarin",
        "text": left + target + right,
        "tone_sandhi": False,
        "targets": [
            {
                "span": [len(left), len(left) + 1],
                "text": target,
                "expected_native": [label],
            }
        ],
    }
