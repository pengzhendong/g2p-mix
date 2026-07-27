import json
import os
import re
from pathlib import Path

import pytest

from g2p_mix import G2PWBackend, MixedG2P
from g2p_mix.models import Language

CASE_FILE = Path(__file__).parent / "cases" / "backend_integration.json"
CASES = json.loads(CASE_FILE.read_text(encoding="utf-8"))["g2pw_smoke"]


@pytest.mark.g2pw
@pytest.mark.skipif(
    os.environ.get("G2P_MIX_TEST_G2PW") != "1",
    reason="set G2P_MIX_TEST_G2PW=1 to run the real G2PW model",
)
@pytest.mark.parametrize("case", [pytest.param(case, id=case["id"]) for case in CASES])
def test_real_g2pw_model_smoke(case):
    result = MixedG2P.mandarin(
        chinese_backend=G2PWBackend(),
        tone_sandhi=False,
    )(case["text"])
    chinese_units = [
        unit for output in result.tokens if output.token.language is Language.CHINESE for unit in output.units
    ]

    assert "".join(unit.text for unit in chinese_units) == case["expected_chinese_text"]
    assert all(unit.native and re.fullmatch(r"[a-z]+[1-5]", unit.native) for unit in chinese_units)
    assert all(
        output.pronunciation.backend == "g2pw"
        for output in result.tokens
        if output.token.language is Language.CHINESE and output.pronunciation
    )
