from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from benchmarks.corpora.download import provenance, source, verify_file

DATA_DIR = Path(__file__).parents[1] / "data" / "corpora" / "wikipedia_homographs"

_CONSONANTS = {
    "b": "B",
    "ʧ": "CH",
    "d": "D",
    "ð": "DH",
    "f": "F",
    "ɡ": "G",
    "h": "HH",
    "ʤ": "JH",
    "k": "K",
    "l": "L",
    "m": "M",
    "n": "N",
    "ŋ": "NG",
    "p": "P",
    "ɹ": "R",
    "s": "S",
    "ʃ": "SH",
    "t": "T",
    "θ": "TH",
    "v": "V",
    "w": "W",
    "j": "Y",
    "z": "Z",
}
_VOWELS = {
    "aɪ": "AY",
    "aʊ": "AW",
    "ɔɪ": "OY",
    "eɪ": "EY",
    "oʊ": "OW",
    "ɑ": "AA",
    "æ": "AE",
    "ʌ": "AH",
    "ə": "AH",
    "ɔ": "AO",
    "ɛ": "EH",
    "ɚ": "ER",
    "ɪ": "IH",
    "i": "IY",
    "ʊ": "UH",
    "u": "UW",
}
_IPA_TOKENS = tuple(sorted((*_CONSONANTS, *_VOWELS), key=len, reverse=True))


def load_wikipedia_homographs(
    *,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    del cache_dir  # The compact eval split is intentionally bundled.
    metadata = source("wikipedia-homographs")
    wordids_path = DATA_DIR / "wordids.tsv"
    eval_dir = DATA_DIR / "eval"
    verify_file(wordids_path, metadata["files"]["wordids.tsv"], "wikipedia_homographs/wordids.tsv")
    _verify_eval_tree(eval_dir, metadata["files"]["eval_tree"])

    classes = _load_pronunciation_classes(wordids_path)
    cases = []
    for path in sorted(eval_dir.glob("*.tsv")):
        with path.open(encoding="utf-8", newline="") as source_file:
            for row_index, row in enumerate(csv.DictReader(source_file, delimiter="\t"), start=1):
                cases.append(_case(path.stem, row_index, row, classes))

    available_cases = len(cases)
    return {
        "schema_version": 1,
        "name": "Wikipedia Homograph Data eval",
        "provenance": provenance(metadata),
        "selection": {
            "split": "eval",
            "available_cases": available_cases,
            "selected_cases": len(cases),
            "homographs": len(classes),
        },
        "cases": cases,
    }


def _load_pronunciation_classes(path: Path) -> dict[str, dict[str, list[str]]]:
    classes: dict[str, dict[str, list[str]]] = {}
    with path.open(encoding="utf-8", newline="") as source_file:
        for row in csv.DictReader(source_file, delimiter="\t"):
            word = row["homograph"].lower()
            classes.setdefault(word, {})[row["wordid"]] = _ipa_to_arpabet(row["pronunciation"])

    for word, pronunciations in classes.items():
        values = [tuple(value) for value in pronunciations.values()]
        if len(values) < 2 or len(values) != len(set(values)):
            raise ValueError(f"Wikipedia homograph {word!r} does not have distinct pronunciation classes")
    return classes


def _ipa_to_arpabet(pronunciation: str) -> list[str]:
    result = []
    pending_stress: int | None = None
    index = 0
    while index < len(pronunciation):
        char = pronunciation[index]
        if char == "'":
            pending_stress = 1
            index += 1
            continue
        if char == "ˌ":
            pending_stress = 2
            index += 1
            continue
        if char in "ː01":
            index += 1
            continue

        token = next((candidate for candidate in _IPA_TOKENS if pronunciation.startswith(candidate, index)), None)
        if token is None:
            raise ValueError(f"Unsupported Wikipedia Homograph IPA at offset {index}: {pronunciation!r}")
        index += len(token)
        if token in _VOWELS:
            stress = 0 if pending_stress is None else pending_stress
            result.append(f"{_VOWELS[token]}{stress}")
            pending_stress = None
        else:
            result.append(_CONSONANTS[token])
    if not result:
        raise ValueError(f"Empty Wikipedia Homograph pronunciation: {pronunciation!r}")
    return result


def _case(
    filename: str,
    row_index: int,
    row: dict[str, str],
    classes: dict[str, dict[str, list[str]]],
) -> dict[str, Any]:
    sentence = row["sentence"]
    encoded = sentence.encode("utf-8")
    start_byte = int(row["start"])
    end_byte = int(row["end"])
    try:
        left = encoded[:start_byte].decode("utf-8")
        target = encoded[start_byte:end_byte].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"Invalid UTF-8 byte span in {filename}.tsv row {row_index}") from error
    if target.lower() != row["homograph"].lower():
        raise ValueError(f"Homograph target mismatch in {filename}.tsv row {row_index}")

    word = row["homograph"].lower()
    wordid = row["wordid"]
    pronunciations = classes[word]
    if wordid not in pronunciations:
        raise ValueError(f"Unknown wordid {wordid!r} in {filename}.tsv row {row_index}")
    start = len(left)
    return {
        "id": f"wikipedia-homograph-eval-{filename}-{row_index:03d}",
        "mode": "english",
        "text": sentence,
        "tone_sandhi": False,
        "targets": [
            {
                "span": [start, start + len(target)],
                "text": target,
                "expected_label": wordid,
                "pronunciation_classes": pronunciations,
            }
        ],
    }


def _verify_eval_tree(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    files = sorted(path.glob("*.tsv"))
    for file_path in files:
        digest.update(file_path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for Wikipedia Homograph eval tree: expected {expected}, got {actual}")
