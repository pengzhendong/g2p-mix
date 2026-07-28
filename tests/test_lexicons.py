import json
from pathlib import Path

import pytest

from g2p_mix.lexicons import MandarinLexicon

CASE_FILE = Path(__file__).parent / "cases" / "mandarin_lexicon.json"
CASE_GROUPS = json.loads(CASE_FILE.read_text(encoding="utf-8"))


def cases(group):
    return [pytest.param(case, id=case["id"]) for case in CASE_GROUPS[group]]


@pytest.mark.parametrize("case", cases("pronunciations"))
def test_pronunciations_are_filtered_deduplicated_and_cached(case):
    calls = []

    def lookup(char):
        calls.append(char)
        return case["lookup_values"]

    lexicon = MandarinLexicon(lookup=lookup)

    assert lexicon.pronunciations(case["char"]) == tuple(case["expected"])
    assert lexicon.pronunciations(case["char"]) == tuple(case["expected"])
    assert calls == [case["char"]]

    lexicon.clear_cache()
    assert lexicon.pronunciations(case["char"]) == tuple(case["expected"])
    assert calls == [case["char"], case["char"]]


@pytest.mark.parametrize("case", cases("scan"))
def test_scan_queries_each_unique_han_character_once(case):
    calls = []

    def lookup(char):
        calls.append(char)
        return case["lookup_values"][char]

    result = MandarinLexicon(lookup=lookup).scan(case["text"])
    expected = {char: tuple(values) for char, values in case["expected"].items()}

    assert list(result) == case["expected_keys"]
    assert result == expected
    assert calls == case["expected_keys"]


@pytest.mark.parametrize("case", cases("invalid"))
def test_pronunciations_reject_non_han_input(case):
    with pytest.raises(ValueError, match="exactly one Han character"):
        MandarinLexicon(lookup=lambda char: ()).pronunciations(case["value"])


@pytest.mark.parametrize("case", cases("integration"))
def test_default_lookup_uses_pypinyin_heteronyms(case):
    pronunciations = MandarinLexicon().pronunciations(case["char"])

    assert set(case["contains"]).issubset(pronunciations)
