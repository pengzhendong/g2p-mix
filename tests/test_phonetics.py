import json
from pathlib import Path

import pytest

from g2p_mix import G2P
from g2p_mix.errors import RenderingError
from g2p_mix.models import PhoneAlphabet, PronunciationUnit, Span
from g2p_mix.phonetics import split_jyutping, split_jyutping_final, split_pinyin
from g2p_mix.renderers import IpaRenderer, NativeRenderer
from g2p_mix.resources import load_json

CASE_FILE = Path(__file__).parent / "cases" / "phonetics.json"
CASE_GROUPS = json.loads(CASE_FILE.read_text(encoding="utf-8"))


def cases(group):
    return [pytest.param(case, id=case["id"]) for case in CASE_GROUPS[group]]


def make_unit(*, phones, alphabet, native="", tone=None, text="x", source_spans=()):
    return PronunciationUnit(
        text=text,
        source_spans=tuple(Span(*span) for span in source_spans),
        phones=tuple(phones),
        alphabet=PhoneAlphabet(alphabet),
        native=native,
        tone=tone,
    )


def make_case_unit(case):
    return make_unit(**case["unit"])


@pytest.mark.parametrize("case", cases("pinyin_valid"))
def test_split_pinyin_accepts_inventory_syllables(case):
    assert split_pinyin(case["value"], strict=case["strict"]) == tuple(case["expected"])


@pytest.mark.parametrize("case", cases("pinyin_invalid"))
def test_split_pinyin_rejects_invalid_bodies(case):
    with pytest.raises(ValueError, match="Invalid numbered pinyin syllable"):
        split_pinyin(case["value"])


@pytest.mark.parametrize("case", cases("jyutping_valid"))
def test_split_jyutping_accepts_inventory_syllables(case):
    assert split_jyutping(case["value"]) == tuple(case["expected"])


@pytest.mark.parametrize("case", cases("jyutping_inventory"))
def test_jyutping_inventory_covers_complete_han_ranges(case):
    bodies = set(load_json("syllables.json")["jyutping"].split())

    assert len(bodies) == case["expected_count"]
    assert set(case["required_bodies"]) <= bodies


@pytest.mark.parametrize("case", cases("jyutping_invalid"))
def test_split_jyutping_rejects_invalid_bodies(case):
    with pytest.raises(ValueError, match="Invalid Jyutping syllable"):
        split_jyutping(case["value"])


def test_phonetic_split_caches_are_bounded():
    assert split_pinyin.cache_info().maxsize == 4096
    assert split_jyutping.cache_info().maxsize == 4096
    assert split_jyutping_final.cache_info().maxsize == 4096


@pytest.mark.parametrize("case", cases("mandarin_ipa"))
def test_default_ipa_renderer_uses_canonical_strict_pinyin(case):
    result = G2P("mandarin", tone_sandhi=False)(case["text"])

    assert [unit.native for unit in result.units] == case["expected_native"]
    assert IpaRenderer().render(result) == tuple(case["expected_ipa"])


@pytest.mark.parametrize("case", cases("ipa_unit_rendering"))
def test_ipa_renderer_uses_structured_phones_instead_of_native(case):
    assert IpaRenderer().render_unit(make_case_unit(case)) == tuple(case["expected"])


@pytest.mark.parametrize("case", cases("ipa_unit_errors"))
def test_ipa_renderer_wraps_phonetic_errors_with_unit_context(case):
    with pytest.raises(RenderingError) as captured:
        IpaRenderer().render_unit(make_case_unit(case))

    message = str(captured.value)
    assert all(fragment in message for fragment in case["message_fragments"])
    assert type(captured.value.__cause__).__name__ == case["cause_type"]


@pytest.mark.parametrize("case", cases("native_tone_policy"))
def test_native_renderer_only_appends_tones_to_numeric_alphabets(case):
    unit = make_unit(
        phones=case["phones"],
        alphabet=case["alphabet"],
        tone=case["tone"],
    )

    assert NativeRenderer().render_unit(unit) == tuple(case["expected"])


@pytest.mark.parametrize("case", cases("tone_updates"))
def test_pronunciation_unit_tone_update_synchronizes_numeric_native(case):
    unit = make_unit(
        phones=("n", "i"),
        alphabet=case["alphabet"],
        native=case["native"],
        tone=case["initial_tone"],
    )

    updated = unit.with_tone(case["new_tone"])

    assert updated.tone == case["new_tone"]
    assert updated.native == case["expected_native"]
