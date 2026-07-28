import json
import os
import re
from pathlib import Path

import pytest

from g2p_mix import G2P
from g2p_mix.models import Language

CASE_FILE = Path(__file__).parent / "cases" / "backend_integration.json"
CASES = json.loads(CASE_FILE.read_text(encoding="utf-8"))["g2pw_smoke"]
UPSTREAM_CASE_FILE = Path(__file__).parent / "fixtures" / "third_party" / "g2pw_readme" / "cases.json"
UPSTREAM_CASES = json.loads(UPSTREAM_CASE_FILE.read_text(encoding="utf-8"))["opt_in_project_regressions"]


@pytest.mark.g2pw
@pytest.mark.skipif(
    os.environ.get("G2P_MIX_TEST_G2PW") != "1",
    reason="set G2P_MIX_TEST_G2PW=1 to run the real G2PW model",
)
@pytest.mark.parametrize("case", [pytest.param(case, id=case["id"]) for case in CASES])
def test_real_g2pw_model_smoke(case):
    result = G2P(
        "mandarin",
        backend="g2pw",
        tone_sandhi=False,
    )(case["text"])
    chinese_units = [
        unit for output in result.tokens if output.token.language is Language.CHINESE for unit in output.units
    ]

    assert "".join(unit.text for unit in chinese_units) == case["expected_chinese_text"]
    assert [unit.native for unit in chinese_units] == case["expected_native"]
    assert all(unit.native and re.fullmatch(r"[a-z]+[1-5]", unit.native) for unit in chinese_units)
    assert all(
        output.pronunciation.backend == "g2pw"
        for output in result.tokens
        if output.token.language is Language.CHINESE and output.pronunciation
    )


@pytest.mark.g2pw
@pytest.mark.skipif(
    os.environ.get("G2P_MIX_TEST_G2PW") != "1",
    reason="set G2P_MIX_TEST_G2PW=1 to run the real G2PW model",
)
@pytest.mark.parametrize(
    "case",
    [pytest.param(case, id=case["id"]) for case in UPSTREAM_CASES],
)
def test_real_g2pw_model_matches_vendored_readme_examples(case):
    converter = G2P(
        "mandarin",
        backend="g2pw",
        tone_sandhi=False,
    )

    assert len(case["texts"]) == len(case["expected_native"])
    for text, expected_native in zip(case["texts"], case["expected_native"]):
        result = converter(text)
        chinese_units = [
            unit for output in result.tokens if output.token.language is Language.CHINESE for unit in output.units
        ]

        assert [unit.native for unit in chinese_units] == expected_native
        assert all(
            output.pronunciation.backend == "g2pw"
            for output in result.tokens
            if output.token.language is Language.CHINESE and output.pronunciation
        )
