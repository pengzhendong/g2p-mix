from __future__ import annotations

import json

import pytest

from benchmarks.corpora.cpp import _case as cpp_case
from benchmarks.corpora.hkcancor import _case as hkcancor_case
from benchmarks.corpora.sampling import deterministic_sample
from benchmarks.dataset import load_dataset
from benchmarks.evaluator import _evaluate_case, evaluate
from g2p_mix.models import PhoneAlphabet, PronunciationUnit, Span


class StubResult:
    normalized_text = "重庆"
    units = (
        PronunciationUnit(
            text="重",
            source_spans=(Span(0, 1),),
            phones=("ch", "ong"),
            alphabet=PhoneAlphabet.PINYIN,
            native="chong2",
        ),
        PronunciationUnit(
            text="庆",
            source_spans=(Span(1, 2),),
            phones=("q", "ing"),
            alphabet=PhoneAlphabet.PINYIN,
            native="qing4",
        ),
    )


def test_target_evaluation_only_scores_annotated_source_span():
    case = {
        "id": "target",
        "mode": "mandarin",
        "text": "重庆",
        "targets": [
            {
                "span": [0, 1],
                "text": "重",
                "expected_native": ["chong2"],
            }
        ],
    }

    result = _evaluate_case(case, lambda _text: StubResult())

    assert result["normalized_exact"] is None
    assert result["pronunciation_exact"] is True
    assert result["targets_correct"] == 1
    assert result["expected_units"] == 1


def test_dataset_requires_exactly_one_pronunciation_annotation(tmp_path):
    dataset = {
        "schema_version": 1,
        "name": "invalid",
        "cases": [
            {
                "id": "both",
                "mode": "mandarin",
                "text": "重",
                "expected_native": ["zhong4"],
                "targets": [
                    {
                        "span": [0, 1],
                        "text": "重",
                        "expected_native": ["zhong4"],
                    }
                ],
            }
        ],
    }
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        load_dataset(path)


def test_evaluator_reports_resolved_backend_identity():
    dataset = {
        "schema_version": 1,
        "name": "identity",
        "cases": [
            {
                "id": "english",
                "mode": "mandarin",
                "text": "hello",
                "expected_native": ["HH AH0 L OW1"],
            }
        ],
    }

    report = evaluate(dataset)

    assert report["backends"]["mandarin"] == "pypinyin"
    assert report["backends"]["english"] == "cmudict-g2p-en"


def test_cpp_marker_becomes_a_source_aligned_target():
    case = cpp_case(11, "重庆银▁行▁", "hang2")

    assert case["text"] == "重庆银行"
    assert case["targets"] == [
        {
            "span": [3, 4],
            "text": "行",
            "expected_native": ["hang2"],
        }
    ]
    assert case["tone_sandhi"] is False


def test_hkcancor_tokens_become_character_targets():
    case = hkcancor_case(
        "FC-test",
        3,
        [("你好", "nei5hou2"), ("！", "VQ1")],
    )

    assert case is not None
    assert case["text"] == "你好！"
    assert case["targets"] == [
        {"span": [0, 1], "text": "你", "expected_native": ["nei5"]},
        {"span": [1, 2], "text": "好", "expected_native": ["hou2"]},
    ]


def test_hkcancor_skips_code_switched_utterances():
    assert hkcancor_case("FC-test", 1, [("OK", "ou1kei1")]) is None


def test_corpus_sampling_is_reproducible_and_keeps_source_order():
    values = list(range(100))

    first = deterministic_sample(values, max_cases=10, seed=42)
    second = deterministic_sample(values, max_cases=10, seed=42)

    assert first == second
    assert first == sorted(first)
    assert len(first) == 10
