from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from g2p_mix import G2P
from g2p_mix.backends import EnglishBackend, PypinyinBackend
from g2p_mix.models import PhoneAlphabet, PronunciationUnit, Span
from g2p_mix.pipeline import G2PPipeline
from g2p_mix.profiles import ChineseProfile, EnglishProfile, MandarinProfile

from .dataset import validate_dataset

MAX_REPORTED_FAILURES = 100


class _EnglishBenchmarkConverter:
    def __init__(self) -> None:
        english = EnglishBackend()
        mandarin = MandarinProfile(backend=PypinyinBackend(), tone_sandhi=False)
        chinese = ChineseProfile(
            dialect=mandarin.dialect,
            backend=mandarin.backend,
            segmenter=mandarin.segmenter,
            normalizers=(),
            processors=(),
        )
        self.backend = "not-applicable"
        self.english_backend = english.name
        self._pipeline = G2PPipeline(
            chinese=chinese,
            english=EnglishProfile.for_backend(english),
        )

    def __call__(self, text: str):
        return self._pipeline(text)


def evaluate(
    dataset: Mapping[str, Any],
    *,
    mandarin_backend: str | None = None,
    cantonese_backend: str | None = None,
    existing_results: Mapping[str, Mapping[str, Any]] | None = None,
    result_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    dataset = validate_dataset(dataset)
    started = time.perf_counter()
    converters = {}
    resolved_backends = {
        "mandarin": mandarin_backend or "pypinyin",
        "cantonese": cantonese_backend or "tojyutping",
        "english": EnglishBackend.name,
    }
    existing_results = existing_results or {}
    case_ids = {case["id"] for case in dataset["cases"]}
    unexpected_ids = set(existing_results) - case_ids
    if unexpected_ids:
        raise ValueError(f"Checkpoint contains unknown case IDs: {sorted(unexpected_ids)!r}")
    existing_duration = sum(float(result.get("duration_ms", 0)) for result in existing_results.values()) / 1000
    results = []

    for case in dataset["cases"]:
        if case["id"] in existing_results:
            results.append(dict(existing_results[case["id"]]))
            continue
        mode = case["mode"]
        backend = mandarin_backend if mode == "mandarin" else cantonese_backend
        tone_sandhi = case.get("tone_sandhi", True)
        converter_key = (mode, backend, tone_sandhi)
        if converter_key not in converters:
            if mode == "english":
                converters[converter_key] = _EnglishBenchmarkConverter()
            else:
                converters[converter_key] = G2P(
                    mode,
                    backend=backend,
                    tone_sandhi=tone_sandhi,
                )
        converter = converters[converter_key]
        resolved_backends[mode] = converter.backend
        resolved_backends["english"] = converter.english_backend
        result = _evaluate_case(case, converter)
        results.append(result)
        if result_callback is not None:
            result_callback(result)

    failures = [result for result in results if _failed(result)]
    reported_failures = sorted(
        failures,
        key=lambda result: result["error"] is None,
    )[:MAX_REPORTED_FAILURES]
    report = {
        "schema_version": 1,
        "dataset": dataset["name"],
        "backends": {
            "mandarin": resolved_backends.get("mandarin", mandarin_backend or "pypinyin"),
            "cantonese": resolved_backends.get("cantonese", cantonese_backend or "tojyutping"),
            "english": resolved_backends["english"],
        },
        "summary": _summarize(
            results,
            duration=existing_duration + time.perf_counter() - started,
        ),
        "by_mode": {
            mode: _summarize(
                [result for result in results if result["mode"] == mode],
                duration=None,
            )
            for mode in ("mandarin", "cantonese", "english")
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
        "label_targets": sum(target["label_target"] for target in target_results),
        "label_targets_correct": sum(target["exact"] for target in target_results if target["label_target"]),
        "unit_edit_distance": sum(target["unit_edit_distance"] for target in target_results),
        "expected_units": sum(target["expected_units"] for target in target_results),
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
    results = []
    for target in targets:
        result = {
            "span": target.get("span"),
            "text": target.get("text"),
            "actual_native": [],
        }
        if "expected_label" in target:
            result.update(
                expected_label=target["expected_label"],
                pronunciation_classes=target["pronunciation_classes"],
                actual_label=None,
                actual_phones=[],
                class_distances={},
            )
        else:
            result["expected_native"] = target["expected_native"]
        results.append(result)
    return results


def _score_target(target: Mapping[str, Any]) -> dict[str, Any]:
    if "expected_label" in target:
        expected_phones = target["pronunciation_classes"][target["expected_label"]]
        return {
            "span": target["span"],
            "text": target["text"],
            "expected_label": target["expected_label"],
            "actual_label": target["actual_label"],
            "actual_native": target["actual_native"],
            "class_distances": target["class_distances"],
            "exact": target["actual_label"] == target["expected_label"],
            "unit_edit_distance": _edit_distance(expected_phones, target["actual_phones"]),
            "expected_units": len(expected_phones),
            "label_target": True,
        }
    return {
        **target,
        "exact": _canonical_native(target["actual_native"]) == _canonical_native(target["expected_native"]),
        "unit_edit_distance": _edit_distance(
            _canonical_native(target["expected_native"]),
            _canonical_native(target["actual_native"]),
        ),
        "expected_units": len(target["expected_native"]),
        "label_target": False,
    }


def _evaluate_target(
    target: Mapping[str, Any],
    units: Sequence[PronunciationUnit],
) -> dict[str, Any]:
    target_span = Span(*target["span"])
    actual_units = [
        unit for unit in units if any(_spans_overlap(target_span, source_span) for source_span in unit.source_spans)
    ]
    actual = [unit.native for unit in actual_units]
    if "expected_label" in target:
        actual_phones = []
        if len(actual_units) == 1 and actual_units[0].alphabet is PhoneAlphabet.ARPABET:
            actual_phones = actual_units[0].native.split()
        distances = {
            label: _edit_distance(phones, actual_phones) for label, phones in target["pronunciation_classes"].items()
        }
        best_distance = min(distances.values())
        closest = [label for label, distance in distances.items() if distance == best_distance]
        return {
            "span": target["span"],
            "text": target["text"],
            "expected_label": target["expected_label"],
            "pronunciation_classes": target["pronunciation_classes"],
            "actual_label": closest[0] if len(closest) == 1 else None,
            "actual_native": actual,
            "actual_phones": actual_phones,
            "class_distances": distances,
        }
    return {
        "span": target["span"],
        "text": target["text"],
        "expected_native": target["expected_native"],
        "actual_native": actual,
    }


def _spans_overlap(left: Span, right: Span) -> bool:
    return left.start < right.end and right.start < left.end


def _canonical_native(syllables: Sequence[str]) -> list[str]:
    # pypinyin writes lü/nü as lv/nv while several corpora write lu:/nu:.
    return [syllable.replace("u:", "v") for syllable in syllables]


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
    label_targets = sum(result["label_targets"] for result in results)
    label_targets_correct = sum(result["label_targets_correct"] for result in results)
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
        "label_targets": label_targets,
        "label_targets_correct": label_targets_correct,
        "label_exact_rate": _ratio(label_targets_correct, label_targets),
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
