from __future__ import annotations

import json

import pytest

from benchmarks.corpora.cpp_clean import _case as cpp_case
from benchmarks.corpora.cpp_clean import load_cpp_clean
from benchmarks.corpora.hkcancor import _case as hkcancor_case
from benchmarks.corpora.wikipedia_homographs import _ipa_to_arpabet, load_wikipedia_homographs
from benchmarks.dataset import load_dataset
from benchmarks.evaluator import _evaluate_case, evaluate
from benchmarks.runner import _checkpoint_fingerprint, _open_checkpoint
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


class EnglishStubResult:
    normalized_text = "I record music."
    units = (
        PronunciationUnit(
            text="record",
            source_spans=(Span(2, 8),),
            phones=("R", "IH", "K", "AO", "R", "D"),
            stress_marks=((1, 0), (3, 1)),
            alphabet=PhoneAlphabet.ARPABET,
            native="R IH0 K AO1 R D",
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


def test_target_evaluation_equates_lv_and_lu_colon_notations():
    case = {
        "id": "lv-notation",
        "mode": "mandarin",
        "text": "率",
        "targets": [
            {
                "span": [0, 1],
                "text": "率",
                "expected_native": ["lu:4"],
            }
        ],
    }

    class LvStubResult:
        normalized_text = "率"
        units = (
            PronunciationUnit(
                text="率",
                source_spans=(Span(0, 1),),
                phones=("l", "v"),
                alphabet=PhoneAlphabet.PINYIN,
                native="lv4",
            ),
        )

    result = _evaluate_case(case, lambda _text: LvStubResult())

    assert result["pronunciation_exact"] is True
    assert result["targets_correct"] == 1
    assert result["unit_edit_distance"] == 0


def test_target_evaluation_scores_pronunciation_class():
    case = {
        "id": "record-verb",
        "mode": "mandarin",
        "text": "I record music.",
        "targets": [
            {
                "span": [2, 8],
                "text": "record",
                "expected_label": "record_vrb",
                "pronunciation_classes": {
                    "record_nou": ["R", "EH1", "K", "ER0", "D"],
                    "record_vrb": ["R", "IH0", "K", "AO1", "R", "D"],
                },
            }
        ],
    }

    result = _evaluate_case(case, lambda _text: EnglishStubResult())

    assert result["pronunciation_exact"] is True
    assert result["label_targets"] == 1
    assert result["label_targets_correct"] == 1
    assert result["target_results"][0]["actual_label"] == "record_vrb"


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


def test_bundled_clean_cpp_has_expected_pinned_size():
    dataset = load_cpp_clean()

    assert dataset["selection"]["available_cases"] == 8935
    assert len(dataset["cases"]) == 8935


def test_wikipedia_homographs_has_expected_pinned_size_and_classes():
    dataset = load_wikipedia_homographs()

    assert dataset["selection"]["available_cases"] == 1615
    assert dataset["selection"]["homographs"] == 162
    assert len(dataset["cases"]) == 1615
    assert all(case["mode"] == "english" for case in dataset["cases"])
    assert all("expected_label" in case["targets"][0] for case in dataset["cases"])


def test_wikipedia_ipa_is_converted_to_stressed_arpabet_classes():
    assert _ipa_to_arpabet("əb'stɹækt") == ["AH0", "B", "S", "T", "R", "AE1", "K", "T"]


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


def test_checkpoint_recovers_a_partial_tail_and_resumes_remaining_cases(tmp_path, monkeypatch):
    cases = [
        {
            "id": case_id,
            "mode": "mandarin",
            "text": "重庆",
            "expected_native": ["chong2", "qing4"],
        }
        for case_id in ("first", "second")
    ]
    dataset = {
        "schema_version": 1,
        "name": "resume",
        "cases": cases,
    }
    checkpoint = tmp_path / "resume.jsonl"
    fingerprint = _checkpoint_fingerprint(
        dataset,
        mandarin_backend="pypinyin",
        cantonese_backend=None,
    )
    _, append_result = _open_checkpoint(checkpoint, fingerprint)
    first_result = _evaluate_case(cases[0], lambda _text: StubResult())
    append_result(first_result)
    with checkpoint.open("a", encoding="utf-8") as stream:
        stream.write('{"type":"result","result":')

    existing_results, append_result = _open_checkpoint(checkpoint, fingerprint)
    calls = []

    class StubConverter:
        backend = "pypinyin"
        english_backend = "cmudict-g2p-en"

        def __call__(self, text):
            calls.append(text)
            return StubResult()

    monkeypatch.setattr("benchmarks.evaluator.G2P", lambda *_args, **_kwargs: StubConverter())
    report = evaluate(
        dataset,
        mandarin_backend="pypinyin",
        existing_results=existing_results,
        result_callback=append_result,
    )

    assert calls == ["重庆"]
    assert report["summary"]["completed"] == 2
    resumed_results, _ = _open_checkpoint(checkpoint, fingerprint)
    assert set(resumed_results) == {"first", "second"}


def test_checkpoint_rejects_a_different_backend_configuration(tmp_path):
    dataset = {
        "schema_version": 1,
        "name": "identity",
        "cases": [],
    }
    checkpoint = tmp_path / "identity.jsonl"
    pypinyin_fingerprint = _checkpoint_fingerprint(
        dataset,
        mandarin_backend="pypinyin",
        cantonese_backend=None,
    )
    _open_checkpoint(checkpoint, pypinyin_fingerprint)
    g2pw_fingerprint = _checkpoint_fingerprint(
        dataset,
        mandarin_backend="g2pw",
        cantonese_backend=None,
    )

    with pytest.raises(ValueError, match="does not match"):
        _open_checkpoint(checkpoint, g2pw_fingerprint)
