from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SOURCES_FILE = Path(__file__).with_name("sources.json")


def source(name: str) -> dict[str, Any]:
    sources = json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
    try:
        return sources[name]
    except KeyError as error:
        raise ValueError(f"Unknown corpus source: {name!r}") from error


def archive_path(
    name: str,
    *,
    cache_dir: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    metadata = source(name)
    root = cache_dir or Path(
        os.environ.get(
            "G2P_MIX_EVAL_CACHE",
            Path.home() / ".cache" / "g2p-mix" / "evals",
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    destination = root / metadata["archive_name"]
    if destination.exists():
        verify_file(
            destination,
            metadata["archive_sha256"],
            destination.name,
        )
        return destination, metadata

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=root,
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            with urllib.request.urlopen(metadata["url"], timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        temporary = Path(temporary_name)
        verify_file(
            temporary,
            metadata["archive_sha256"],
            destination.name,
        )
        temporary.replace(destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination, metadata


def verify_bytes(data: bytes, expected: str, label: str) -> None:
    _verify_digest(hashlib.sha256(data).hexdigest(), expected, label)


def verify_file(path: Path, expected: str, label: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            digest.update(chunk)
    _verify_digest(digest.hexdigest(), expected, label)


def provenance(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: metadata[key]
        for key in (
            "repository",
            "revision",
            "url",
            "archive_sha256",
            "license",
        )
    }


def _verify_digest(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {label}: expected {expected}, got {actual}")
