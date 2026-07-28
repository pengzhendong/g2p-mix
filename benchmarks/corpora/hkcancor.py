from __future__ import annotations

import io
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Any

from benchmarks.corpora.download import archive_path, provenance, verify_bytes
from benchmarks.corpora.sampling import deterministic_sample
from g2p_mix.text.unicode_script import is_han_character

JYUTPING = re.compile(r"[a-z]+[1-6]")


def load_hkcancor(
    *,
    cache_dir: Path | None = None,
    max_cases: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    archive, metadata = archive_path("hkcancor", cache_dir=cache_dir)
    root = f"hkcancor-{metadata['revision']}"
    with tarfile.open(archive, "r:gz") as bundle:
        nested = bundle.extractfile(f"{root}/data/hkcancor-utf8.zip")
        if nested is None:
            raise ValueError("Missing HKCanCor UTF-8 archive")
        nested_data = nested.read()
    verify_bytes(
        nested_data,
        metadata["files"]["hkcancor-utf8.zip"],
        "hkcancor-utf8.zip",
    )

    cases = _parse_archive(nested_data)
    available = len(cases)
    cases = deterministic_sample(cases, max_cases=max_cases, seed=seed)
    return {
        "schema_version": 1,
        "name": "Hong Kong Cantonese Corpus (HKCanCor)",
        "provenance": provenance(metadata),
        "selection": {
            "split": "all annotated utterances without Latin text",
            "available_cases": available,
            "selected_cases": len(cases),
            "seed": seed,
        },
        "cases": cases,
    }


def _parse_archive(data: bytes) -> list[dict[str, Any]]:
    cases = []
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        names = sorted(name for name in bundle.namelist() if name.startswith("utf8/") and not name.endswith("/"))
        for name in names:
            content = bundle.read(name).decode("utf-8-sig")
            cases.extend(_parse_file(Path(name).name, content))
    return cases


def _parse_file(filename: str, content: str) -> list[dict[str, Any]]:
    utterances = []
    current: list[tuple[str, str]] | None = None
    utterance_number = 0
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line == "<sent_tag>":
            current = []
        elif line == "</sent_tag>":
            if current:
                utterance_number += 1
                case = _case(filename, utterance_number, current)
                if case is not None:
                    utterances.append(case)
            current = None
        elif current is not None:
            parsed = _parse_token(line)
            if parsed is not None:
                current.append(parsed)
    return utterances


def _parse_token(line: str) -> tuple[str, str] | None:
    if not line.endswith("/"):
        return None
    fields = line[:-1].rsplit("/", 2)
    if len(fields) != 3:
        return None
    text, _part_of_speech, pronunciation = fields
    return text, pronunciation


def _case(
    filename: str,
    utterance_number: int,
    tokens: list[tuple[str, str]],
) -> dict[str, Any] | None:
    text = "".join(token for token, _pronunciation in tokens)
    if any(character.isascii() and character.isalpha() for character in text):
        return None

    targets = []
    cursor = 0
    for token, raw_pronunciation in tokens:
        syllables = JYUTPING.findall(raw_pronunciation.lower())
        if token and all(is_han_character(character) for character in token):
            if len(syllables) == len(token):
                for offset, (character, syllable) in enumerate(zip(token, syllables)):
                    targets.append(
                        {
                            "span": [cursor + offset, cursor + offset + 1],
                            "text": character,
                            "expected_native": [syllable],
                        }
                    )
        cursor += len(token)
    if not targets:
        return None
    return {
        "id": f"hkcancor-{filename}-{utterance_number:04d}",
        "mode": "cantonese",
        "text": text,
        "targets": targets,
    }
