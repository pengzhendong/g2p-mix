from __future__ import annotations

from pathlib import Path
from typing import Any


def load_corpus(
    name: str,
    *,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    if name == "cpp-clean":
        from benchmarks.corpora.cpp_clean import load_cpp_clean

        return load_cpp_clean(cache_dir=cache_dir)
    if name == "wikipedia-homographs":
        from benchmarks.corpora.wikipedia_homographs import load_wikipedia_homographs

        return load_wikipedia_homographs(cache_dir=cache_dir)
    if name == "hkcancor":
        from benchmarks.corpora.hkcancor import load_hkcancor

        return load_hkcancor(cache_dir=cache_dir)
    raise ValueError(f"Unknown evaluation corpus: {name!r}")


__all__ = ["load_corpus"]
