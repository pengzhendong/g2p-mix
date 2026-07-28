from __future__ import annotations

from pathlib import Path
from typing import Any


def load_corpus(
    name: str,
    *,
    cache_dir: Path | None = None,
    max_cases: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    if name == "cpp":
        from evals.corpora.cpp import load_cpp

        return load_cpp(cache_dir=cache_dir, max_cases=max_cases, seed=seed)
    if name == "hkcancor":
        from evals.corpora.hkcancor import load_hkcancor

        return load_hkcancor(
            cache_dir=cache_dir,
            max_cases=max_cases,
            seed=seed,
        )
    raise ValueError(f"Unknown evaluation corpus: {name!r}")


__all__ = ["load_corpus"]
