import json
from pathlib import Path

import pytest

import g2p_mix
from g2p_mix import G2P, G2PError, G2PResult
from g2p_mix.backends import PypinyinBackend
from g2p_mix.errors import ConfigurationError

CASE_FILE = Path(__file__).parent / "cases" / "transcription_similarity.json"
CASE_GROUPS = json.loads(CASE_FILE.read_text(encoding="utf-8"))


def cases(group):
    return [pytest.param(case, id=case["id"]) for case in CASE_GROUPS[group]]


@pytest.mark.parametrize("case", cases("public_api"))
def test_simple_api_returns_final_phones_and_atomic_segments(case):
    converter = G2P(case["mode"], output=case["output"], **case["options"])

    result = converter(case["text"])

    assert isinstance(result, G2PResult)
    assert converter.backend == case["expected_backend"]
    assert result.output == case["output"]
    assert result.phones == tuple(case["expected_phones"])
    assert result.segments == tuple(case["expected_segments"])


@pytest.mark.parametrize("case", cases("invalid_api"))
def test_simple_api_rejects_invalid_configuration(case):
    with pytest.raises(ConfigurationError, match=case["message"]):
        G2P(**case["arguments"])


def test_root_package_exposes_only_the_simple_api():
    assert g2p_mix.__all__ == [
        "__version__",
        "G2P",
        "G2PResult",
        "G2PError",
    ]
    assert G2PError is g2p_mix.G2PError
    assert not hasattr(g2p_mix, "MixedG2P")
    assert not hasattr(g2p_mix, "IpaRenderer")


def test_custom_backend_uses_the_same_simple_argument():
    converter = G2P("mandarin", backend=PypinyinBackend())

    assert converter.backend == "pypinyin"
