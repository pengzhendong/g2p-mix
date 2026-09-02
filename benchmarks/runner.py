from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.dataset import load_dataset, validate_dataset
from benchmarks.evaluator import evaluate

DEFAULT_DATASET = Path(__file__).parent / "data" / "smoke.json"


def _format_report(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"Dataset: {report['dataset']}",
        "Backends: " + ", ".join(f"{language}={backend}" for language, backend in report["backends"].items()),
        f"Cases: {summary['completed']}/{summary['cases']} completed",
        _format_normalization(summary),
        f"Pronunciation exact: {summary['pronunciation_exact_rate']:.2%}",
        f"Target exact: {summary['target_exact_rate']:.2%}",
        *([f"Pronunciation class exact: {summary['label_exact_rate']:.2%}"] if summary["label_targets"] else []),
        f"Unit error rate: {summary['unit_error_rate']:.2%}",
        f"Throughput: {summary['cases_per_second']:.2f} cases/s",
    ]
    if report["failure_count"]:
        lines.append(f"Failures (showing {len(report['failures'])}/{report['failure_count']}):")
        for failure in report["failures"]:
            reason = failure["error"] or (
                f"normalized={failure['normalized_exact']}, pronunciation={failure['pronunciation_exact']}"
            )
            lines.append(f"  - {failure['id']}: {reason}")
    return "\n".join(lines)


def _format_normalization(summary: Mapping[str, Any]) -> str:
    if not summary["normalized_cases"]:
        return "Normalization exact: n/a (0 annotated cases)"
    return (
        f"Normalization exact: {summary['normalized_exact_rate']:.2%} ({summary['normalized_cases']} annotated cases)"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate g2p-mix against an external JSON dataset")
    parser.add_argument("dataset", nargs="?", type=Path)
    parser.add_argument(
        "--corpus",
        choices=("cpp", "cpp-clean", "wikipedia-homographs", "hkcancor"),
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mandarin-backend", choices=("pypinyin", "g2pw"))
    parser.add_argument("--cantonese-backend", choices=("tojyutping", "pycantonese"))
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="append each completed case to this JSONL file and resume it on the next run",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        metavar="RATE",
        help="exit with status 1 when pronunciation exact rate is below RATE",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.fail_under is not None and not 0 <= arguments.fail_under <= 1:
        parser.error("--fail-under must be between 0 and 1")
    if arguments.dataset is not None and arguments.corpus is not None:
        parser.error("dataset and --corpus are mutually exclusive")
    if arguments.max_cases is not None and arguments.max_cases <= 0:
        parser.error("--max-cases must be positive")

    try:
        if arguments.corpus:
            from benchmarks.corpora import load_corpus

            dataset = load_corpus(
                arguments.corpus,
                cache_dir=arguments.cache_dir,
                max_cases=arguments.max_cases,
                seed=arguments.seed,
            )
            validate_dataset(dataset)
        else:
            dataset = load_dataset(arguments.dataset or DEFAULT_DATASET)
        existing_results = None
        result_callback = None
        if arguments.checkpoint is not None:
            fingerprint = _checkpoint_fingerprint(
                dataset,
                mandarin_backend=arguments.mandarin_backend,
                cantonese_backend=arguments.cantonese_backend,
            )
            existing_results, result_callback = _open_checkpoint(
                arguments.checkpoint,
                fingerprint,
            )
        report = evaluate(
            dataset,
            mandarin_backend=arguments.mandarin_backend,
            cantonese_backend=arguments.cantonese_backend,
            existing_results=existing_results,
            result_callback=result_callback,
        )
    except ValueError as error:
        parser.error(str(error))

    if arguments.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_format_report(report))

    if arguments.fail_under is not None and report["summary"]["pronunciation_exact_rate"] < arguments.fail_under:
        return 1
    return 0


def _checkpoint_fingerprint(
    dataset: Mapping[str, Any],
    *,
    mandarin_backend: str | None,
    cantonese_backend: str | None,
) -> str:
    payload = {
        "dataset": dataset,
        "mandarin_backend": mandarin_backend or "pypinyin",
        "cantonese_backend": cantonese_backend or "tojyutping",
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _open_checkpoint(path: Path, fingerprint: str):
    metadata = {
        "type": "metadata",
        "schema_version": 1,
        "fingerprint": fingerprint,
    }
    results = {}
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError(f"Checkpoint is empty: {path}")
        try:
            saved_metadata = json.loads(lines[0])
        except json.JSONDecodeError as error:
            raise ValueError(f"Checkpoint metadata is malformed: {path}") from error
        if saved_metadata != metadata:
            raise ValueError(f"Checkpoint does not match this dataset and backend configuration: {path}")
        for index, line in enumerate(lines[1:], start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                if index == len(lines) - 1:
                    break
                raise ValueError(f"Checkpoint entry {index + 1} is malformed: {path}") from error
            if not isinstance(item, dict) or item.get("type") != "result":
                raise ValueError(f"Checkpoint entry {index + 1} is invalid: {path}")
            result = item.get("result")
            if not isinstance(result, dict) or not isinstance(result.get("id"), str):
                raise ValueError(f"Checkpoint entry {index + 1} has no valid result: {path}")
            results[result["id"]] = result

    path.parent.mkdir(parents=True, exist_ok=True)
    _rewrite_checkpoint(path, metadata, results.values())

    def append_result(result: Mapping[str, Any]) -> None:
        entry = json.dumps(
            {"type": "result", "result": result},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with path.open("a", encoding="utf-8") as stream:
            stream.write(entry + "\n")

    return results, append_result


def _rewrite_checkpoint(
    path: Path,
    metadata: Mapping[str, Any],
    results,
) -> None:
    lines = [json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))]
    lines.extend(
        json.dumps(
            {"type": "result", "result": result},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for result in results
    )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)
