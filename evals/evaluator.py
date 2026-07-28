from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from g2p_mix import G2P
from g2p_mix.models import PronunciationUnit, Span

MAX_REPORTED_FAILURES = 100


def evaluate(
    dataset: Mapping[str, Any],
    *,
    mandarin_backend: str | None = None,
    cantonese_backend: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    converters = {}
    results = []

    for case in dataset["cases"]:
        mode = case["mode"]
        backend = mandarin_backend if mode == "mandarin" else cantonese_backend
        tone_sandhi = case.get("tone_sandhi", True)
        converter_key = (mode, backend, tone_sandhi)
        if converter_key not in converters:
            converters[converter_key] = G2P(
                mode,
                backend=backend,
                tone_sandhi=tone_sandhi,
            )
        results.append(_evaluate_case(case, converters[converter_key]))

    failures = [result for result in results if _failed(result)]
    reported_failures = sorted(
        failures,
        key=lambda result: result["error"] is None,
    )[:MAX_REPORTED_FAILURES]
    report = {
        "schema_version": 1,
        "dataset": dataset["name"],
        "backends": {
            "mandarin": mandarin_backend or "pypinyin",
            "cantonese": cantonese_backend or "tojyutping",
            "english": "g2p-en",
        },
        "summary": _summarize(
            results,
            duration=time.perf_counter() - started,
        ),
        "by_mode": {
            mode: _summarize(
                [result for result in results if result["mode"] == mode],
                duration=None,
            )
            for mode in ("mandarin", "cantonese")
            if any(result["mode"] == mode for result in results)
        },
        "failure_count": len(failures),
        "failures": reported_failures,
    }
    for field in ("provenance", "selection"):
        if field in dataset:
            report[field] = dataset[field]
    return report


def _failed(result: Mapping[str, Any]) -> bool:
    return result["error"] is not None or result["normalized_exact"] is False or not result["pronunciation_exact"]


def _evaluate_case(
    case: Mapping[str, Any],
    converter: G2P,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        converted = converter(case["text"])
        actual_normalized = converted.normalized_text
        actual_native = [unit.native for unit in converted.units]
        target_results = _actual_targets(
            case,
            actual_native,
            converted.units,
        )
        error = None
    except Exception as exception:
        actual_normalized = None
        actual_native = []
        target_results = _empty_targets(case)
        error = f"{type(exception).__name__}: {exception}"

    target_results = [_score_target(target) for target in target_results]
    expected_normalized = case.get("expected_normalized")
    normalized_exact = actual_normalized == expected_normalized if expected_normalized is not None else None
    return {
        "id": case["id"],
        "mode": case["mode"],
        "text": case["text"],
        "normalized_exact": normalized_exact,
        "pronunciation_exact": all(target["exact"] for target in target_results),
        "targets": len(target_results),
        "targets_correct": sum(target["exact"] for target in target_results),
        "unit_edit_distance": sum(target["unit_edit_distance"] for target in target_results),
        "expected_units": sum(len(target["expected_native"]) for target in target_results),
        "expected_normalized": expected_normalized,
        "actual_normalized": actual_normalized,
        "actual_native": actual_native,
        "target_results": target_results,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "error": error,
    }


def _actual_targets(
    case: Mapping[str, Any],
    actual_native: list[str],
    units: Sequence[PronunciationUnit],
) -> list[dict[str, Any]]:
    if "targets" in case:
        return [_evaluate_target(target, units) for target in case["targets"]]
    return [
        {
            "span": None,
            "text": case["text"],
            "expected_native": case["expected_native"],
            "actual_native": actual_native,
        }
    ]


def _empty_targets(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    targets = case.get(
        "targets",
        [
            {
                "text": case["text"],
                "expected_native": case.get("expected_native", []),
            }
        ],
    )
    return [
        {
            "span": target.get("span"),
            "text": target.get("text"),
            "expected_native": target["expected_native"],
            "actual_native": [],
        }
        for target in targets
    ]


def _score_target(target: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **target,
        "exact": target["actual_native"] == target["expected_native"],
        "unit_edit_distance": _edit_distance(
            target["expected_native"],
            target["actual_native"],
        ),
    }


def _evaluate_target(
    target: Mapping[str, Any],
    units: Sequence[PronunciationUnit],
) -> dict[str, Any]:
    target_span = Span(*target["span"])
    actual = [
        unit.native
        for unit in units
        if any(_spans_overlap(target_span, source_span) for source_span in unit.source_spans)
    ]
    return {
        "span": target["span"],
        "text": target["text"],
        "expected_native": target["expected_native"],
        "actual_native": actual,
    }


def _spans_overlap(left: Span, right: Span) -> bool:
    return left.start < right.end and right.start < left.end


def _edit_distance(expected: Sequence[str], actual: Sequence[str]) -> int:
    previous = list(range(len(actual) + 1))
    for expected_index, expected_value in enumerate(expected, start=1):
        current = [expected_index]
        for actual_index, actual_value in enumerate(actual, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[actual_index] + 1,
                    previous[actual_index - 1] + (expected_value != actual_value),
                )
            )
        previous = current
    return previous[-1]


def _summarize(
    results: Sequence[Mapping[str, Any]],
    duration: float | None,
) -> dict[str, Any]:
    total = len(results)
    normalization_results = [result for result in results if result["normalized_exact"] is not None]
    expected_units = sum(result["expected_units"] for result in results)
    unit_edits = sum(result["unit_edit_distance"] for result in results)
    targets = sum(result["targets"] for result in results)
    targets_correct = sum(result["targets_correct"] for result in results)
    summary = {
        "cases": total,
        "completed": sum(result["error"] is None for result in results),
        "normalized_cases": len(normalization_results),
        "normalized_exact_rate": _ratio(
            sum(result["normalized_exact"] for result in normalization_results),
            len(normalization_results),
        ),
        "pronunciation_exact_rate": _ratio(
            sum(result["pronunciation_exact"] for result in results),
            total,
        ),
        "targets": targets,
        "targets_correct": targets_correct,
        "target_exact_rate": _ratio(targets_correct, targets),
        "unit_error_rate": _ratio(unit_edits, expected_units),
        "unit_edits": unit_edits,
        "expected_units": expected_units,
    }
    if duration is not None:
        summary["duration_seconds"] = round(duration, 6)
        summary["cases_per_second"] = round(total / duration, 3) if duration else None
    return summary


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0
