from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_dataset(path: str | Path) -> dict[str, Any]:
    dataset_path = Path(path)
    try:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load evaluation dataset {dataset_path}: {error}") from error
    return validate_dataset(dataset)


def validate_dataset(dataset: object) -> dict[str, Any]:
    if not isinstance(dataset, dict) or dataset.get("schema_version") != 1:
        raise ValueError("Evaluation dataset must be an object with schema_version=1")
    if not isinstance(dataset.get("name"), str) or not dataset["name"]:
        raise ValueError("Evaluation dataset must have a non-empty name")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation dataset must have a non-empty cases list")

    identifiers = set()
    for index, case in enumerate(cases):
        _validate_case(case, index)
        if case["id"] in identifiers:
            raise ValueError(f"Duplicate evaluation case id: {case['id']!r}")
        identifiers.add(case["id"])
    return dataset


def _validate_case(case: object, index: int) -> None:
    if not isinstance(case, dict):
        raise ValueError(f"Evaluation case {index} must be an object")
    for field in ("id", "text"):
        if not isinstance(case.get(field), str) or not case[field]:
            raise ValueError(f"Evaluation case {index} must have a non-empty {field}")
    if "expected_normalized" in case and (
        not isinstance(case["expected_normalized"], str) or not case["expected_normalized"]
    ):
        raise ValueError(f"Evaluation case {case['id']!r} has an invalid expected_normalized")
    if case.get("mode") not in {"mandarin", "cantonese"}:
        raise ValueError(f"Evaluation case {case['id']!r} has an invalid mode")
    expected_native = case.get("expected_native")
    targets = case.get("targets")
    if (expected_native is None) == (targets is None):
        raise ValueError(f"Evaluation case {case['id']!r} must have exactly one of expected_native or targets")
    if expected_native is not None and (
        not isinstance(expected_native, list)
        or not expected_native
        or any(not isinstance(value, str) or not value for value in expected_native)
    ):
        raise ValueError(f"Evaluation case {case['id']!r} has an invalid expected_native value")
    if targets is not None:
        if not isinstance(targets, list) or not targets:
            raise ValueError(f"Evaluation case {case['id']!r} must have non-empty targets")
        for target_index, target in enumerate(targets):
            _validate_target(case, target, target_index)
    if "tone_sandhi" in case and not isinstance(case["tone_sandhi"], bool):
        raise ValueError(f"Evaluation case {case['id']!r} has an invalid tone_sandhi value")


def _validate_target(
    case: Mapping[str, Any],
    target: object,
    index: int,
) -> None:
    case_id = case["id"]
    if not isinstance(target, dict):
        raise ValueError(f"Evaluation target {case_id!r}[{index}] must be an object")
    span = target.get("span")
    if (
        not isinstance(span, list)
        or len(span) != 2
        or any(not isinstance(value, int) for value in span)
        or not 0 <= span[0] < span[1] <= len(case["text"])
    ):
        raise ValueError(f"Evaluation target {case_id!r}[{index}] has an invalid span")
    if target.get("text") != case["text"][span[0] : span[1]]:
        raise ValueError(f"Evaluation target {case_id!r}[{index}] text does not match its span")
    expected = target.get("expected_native")
    if (
        not isinstance(expected, list)
        or not expected
        or any(not isinstance(value, str) or not value for value in expected)
    ):
        raise ValueError(f"Evaluation target {case_id!r}[{index}] has invalid expected_native")
