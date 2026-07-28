from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evals.dataset import load_dataset, validate_dataset
from evals.evaluator import evaluate

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
    parser.add_argument("--corpus", choices=("cpp", "hkcancor"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mandarin-backend", choices=("pypinyin", "g2pw"))
    parser.add_argument("--cantonese-backend", choices=("tojyutping", "pycantonese"))
    parser.add_argument("--json", action="store_true", dest="as_json")
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
            from evals.corpora import load_corpus

            dataset = load_corpus(
                arguments.corpus,
                cache_dir=arguments.cache_dir,
                max_cases=arguments.max_cases,
                seed=arguments.seed,
            )
            validate_dataset(dataset)
        else:
            dataset = load_dataset(arguments.dataset or DEFAULT_DATASET)
        report = evaluate(
            dataset,
            mandarin_backend=arguments.mandarin_backend,
            cantonese_backend=arguments.cantonese_backend,
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
