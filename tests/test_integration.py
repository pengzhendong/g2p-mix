import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from g2p_mix import MixedG2P, NativeRenderer
from g2p_mix.models import Language, PhoneAlphabet

CASE_FILE = Path(__file__).parent / "cases" / "mandarin_initialization.json"
CASE_GROUPS = json.loads(CASE_FILE.read_text(encoding="utf-8"))


def cases(group):
    return [pytest.param(case, id=case["id"]) for case in CASE_GROUPS[group]]


def test_mandarin_english_readme_semantics():
    result = MixedG2P.mandarin()("你这个idea, 不太make sense。")

    assert result.reconstruct_original() == "你这个idea, 不太make sense。"
    assert result.projections[Language.CHINESE].text == ("你这个<EN>, 不太<EN>。")
    assert NativeRenderer().render(result)
    assert {unit.alphabet for unit in result.units} == {
        PhoneAlphabet.PINYIN,
        PhoneAlphabet.ARPABET,
    }


def test_cantonese_english_readme_semantics():
    original = "你这个idea。"
    result = MixedG2P.cantonese()(original)

    assert result.normalized_text == "你這個idea。"
    assert result.projections[Language.CHINESE].text == "你這個<EN>。"
    chinese_units = [
        unit for output in result.tokens if output.token.language is Language.CHINESE for unit in output.units
    ]
    assert ["".join(span.slice(original) for span in unit.source_spans) for unit in chinese_units] == ["你", "这", "个"]
    assert all(unit.alphabet is PhoneAlphabet.JYUTPING for unit in chinese_units)
    assert all(unit.native and re.fullmatch(r"[a-z]+[1-6]", unit.native) for unit in chinese_units)
    assert all(unit.phones and unit.tone in set("123456") for unit in chinese_units)
    assert {
        output.pronunciation.backend
        for output in result.tokens
        if output.token.language is Language.CHINESE and output.pronunciation
    } == {"tojyutping"}
    assert NativeRenderer().render(result)


def test_import_has_no_model_or_environment_side_effects():
    code = """
import json
import os
import sys
os.environ["TRANSFORMERS_OFFLINE"] = "custom"
import g2p_mix
print(json.dumps({
        "offline": os.environ["TRANSFORMERS_OFFLINE"],
        "g2p_en": "g2p_en" in sys.modules,
        "pycantonese": "pycantonese" in sys.modules,
        "tojyutping": "ToJyutping" in sys.modules,
        "jieba": "jieba" in sys.modules,
        "nltk": "nltk" in sys.modules,
}))
"""
    environment = os.environ.copy()
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=20,
    )

    assert json.loads(completed.stdout) == {
        "offline": "custom",
        "g2p_en": False,
        "pycantonese": False,
        "tojyutping": False,
        "jieba": False,
        "nltk": False,
    }


@pytest.mark.parametrize("case", cases("phrase_overrides"))
def test_mandarin_phrase_overrides_are_stable_from_the_first_conversion(case):
    lookup_code = """
import json
import sys
from g2p_mix import MandarinLexicon

case = json.loads(sys.argv[1])
jieba_before_lookup = "jieba" in sys.modules
MandarinLexicon().pronunciations(case["lookup_char"])
jieba_after_lookup = "jieba" in sys.modules

print(json.dumps({
    "jieba_before_lookup": jieba_before_lookup,
    "jieba_after_lookup": jieba_after_lookup,
}))
"""
    lookup_completed = subprocess.run(
        [sys.executable, "-c", lookup_code, json.dumps(case)],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=20,
    )
    assert json.loads(lookup_completed.stdout) == {
        "jieba_before_lookup": False,
        "jieba_after_lookup": False,
    }

    conversion_code = """
import json
import sys
from g2p_mix import MixedG2P

case = json.loads(sys.argv[1])
converter = MixedG2P.mandarin(tone_sandhi=False)

def snapshot():
    result = converter(case["text"])
    return {
        "tokens": [output.token.text for output in result.tokens],
        "native": [unit.native for unit in result.units],
    }

print(json.dumps({
    "first": snapshot(),
    "second": snapshot(),
}, ensure_ascii=False))
"""
    conversion_completed = subprocess.run(
        [sys.executable, "-c", conversion_code, json.dumps(case)],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=30,
    )

    payload = json.loads(conversion_completed.stdout)
    expected = {
        "tokens": case["expected_tokens"],
        "native": case["expected_native"],
    }
    assert payload == {
        "first": expected,
        "second": expected,
    }
