import json
from pathlib import Path

import pytest

pytest.importorskip("panphon")

from g2p_mix import G2P
from g2p_mix.resources import load_json
from g2p_mix.similarity import PanPhonDistanceBackend

CASE_FILE = Path(__file__).parent / "cases" / "transcription_similarity.json"
CASE_GROUPS = json.loads(CASE_FILE.read_text(encoding="utf-8"))


def cases(group):
    return [pytest.param(case, id=case["id"]) for case in CASE_GROUPS[group]]


def test_all_builtin_ipa_segments_are_supported_by_panphon_adapter():
    segments = set()

    def collect(value, parent=""):
        if parent == "tones":
            return
        if isinstance(value, dict):
            for key, child in value.items():
                collect(child, key)
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            segments.update(value)

    collect(load_json("transcriptions.json"))
    backend = PanPhonDistanceBackend()

    assert segments
    for segment in segments:
        result = backend.compare((segment,), (segment,))
        assert result.distance == pytest.approx(0)


@pytest.mark.parametrize("case", cases("feature_orderings"))
def test_panphon_feature_distance_orders_nearby_phones(case):
    backend = PanPhonDistanceBackend()

    near = backend.compare(case["source"], case["near"])
    far = backend.compare(case["source"], case["far"])

    assert near.distance < far.distance


@pytest.mark.parametrize("case", cases("similarity_orderings"))
def test_cross_language_matcher_orders_pronunciations(case):
    converter = G2P("mandarin", tone_sandhi=False)

    identical = converter.compare(case["source"], case["source"])
    near = converter.compare(case["source"], case["near"])
    far = converter.compare(case["source"], case["far"])

    assert identical.score == pytest.approx(1.0)
    assert all(step.cost == 0 for step in identical.alignment)
    assert near.score > far.score
    assert near.left
    assert near.right
