import json
import os
import subprocess
import sys

from g2p_mix import MixedG2P, NativeRenderer
from g2p_mix.models import Language


def test_mandarin_english_readme_semantics():
    result = MixedG2P.mandarin()("你这个idea, 不太make sense。")

    assert result.reconstruct_original() == "你这个idea, 不太make sense。"
    assert result.projections[Language.CHINESE].text == ("你这个<EN>, 不太<EN>。")
    assert NativeRenderer().render(result) == (
        "n",
        "i3",
        "zh",
        "e4",
        "g",
        "e5",
        "AY0",
        "D",
        "IY1",
        "AH0",
        "b",
        "u2",
        "t",
        "ai4",
        "M",
        "EY1",
        "K",
        "S",
        "EH1",
        "N",
        "S",
    )


def test_cantonese_english_readme_semantics():
    original = "你这个idea。"
    result = MixedG2P.cantonese()(original)

    assert result.normalized_text == "你這個idea。"
    assert result.projections[Language.CHINESE].text == "你這個<EN>。"
    chinese_units = [
        unit for output in result.tokens if output.token.language is Language.CHINESE for unit in output.units
    ]
    assert ["".join(span.slice(original) for span in unit.source_spans) for unit in chinese_units] == ["你", "这", "个"]
    assert NativeRenderer().render(result) == (
        "n",
        "ei5",
        "z",
        "e3",
        "g",
        "o3",
        "AY0",
        "D",
        "IY1",
        "AH0",
    )


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
    )

    assert json.loads(completed.stdout) == {
        "offline": "custom",
        "g2p_en": False,
        "pycantonese": False,
        "jieba": False,
        "nltk": False,
    }
