import json
from pathlib import Path

import pytest

from g2p_mix import (
    ConfigurationError,
    IpaTranscriber,
    MixedG2P,
    PhoneAlphabet,
    PronunciationUnit,
    Span,
)
from g2p_mix.phonetics import (
    canonical_pinyin_phones,
    split_jyutping,
    split_pinyin,
    transcribe_arpabet_phone,
    transcribe_jyutping,
    transcribe_pinyin,
)
from g2p_mix.resources import load_cmudict, load_json

CASE_FILE = Path(__file__).parent / "cases" / "transcription_similarity.json"
CASE_GROUPS = json.loads(CASE_FILE.read_text(encoding="utf-8"))


def cases(group):
    return [pytest.param(case, id=case["id"]) for case in CASE_GROUPS[group]]


@pytest.mark.parametrize("case", cases("structured_ipa"))
def test_pipeline_can_return_source_aligned_structured_ipa(case):
    kwargs = {"output_alphabet": PhoneAlphabet.IPA}
    if case["mode"] == "mandarin":
        kwargs["tone_sandhi"] = case["tone_sandhi"]
        converter = MixedG2P.mandarin(**kwargs)
    else:
        converter = MixedG2P.cantonese(**kwargs)

    result = converter(case["text"])

    assert all(unit.alphabet is PhoneAlphabet.IPA for unit in result.units)
    assert [
        {
            "text": unit.text,
            "phones": list(unit.phones),
            "source_alphabet": unit.source_alphabet.value,
            "source_phones": list(unit.source_phones),
            "tone_contour": list(unit.tone_contour),
            "stress_marks": [list(mark) for mark in unit.stress_marks],
        }
        for unit in result.units
    ] == case["expected"]
    assert result.reconstruct_original() == case["text"]


@pytest.mark.parametrize("case", cases("arpabet_context"))
def test_english_ipa_preserves_phone_level_stress(case):
    source = PronunciationUnit(
        text="word",
        source_spans=(Span(0, 1),),
        phones=tuple(case["phones"]),
        alphabet=PhoneAlphabet.ARPABET,
        native=" ".join(case["phones"]),
    )

    result = IpaTranscriber().transcribe_unit(source)

    assert result.phones == tuple(case["expected_phones"])
    assert result.stress_marks == tuple(tuple(mark) for mark in case["expected_stress_marks"])
    assert result.source_alphabet is PhoneAlphabet.ARPABET
    assert result.source_phones == source.phones
    assert source.alphabet is PhoneAlphabet.ARPABET


def test_mixed_output_rejects_a_single_language_native_alphabet():
    with pytest.raises(ConfigurationError, match="native alphabets or IPA"):
        MixedG2P.mandarin(output_alphabet=PhoneAlphabet.PINYIN)


def test_factory_infers_output_alphabet_from_injected_transcriber():
    result = MixedG2P.mandarin(transcriber=IpaTranscriber())("一")

    assert result.units
    assert all(unit.alphabet is PhoneAlphabet.IPA for unit in result.units)


def test_jyutping_ipa_matches_pycantonese_reference_inventory():
    from pycantonese import jyutping_to_ipa

    for body in load_json("syllables.json")["jyutping"].split():
        onset, final, _ = split_jyutping(body + "1")
        actual = "".join(transcribe_jyutping(onset, final))
        expected = jyutping_to_ipa(body + "1")[0][:-2]
        normalizer = str.maketrans("", "", "̩̍͡")
        assert actual.translate(normalizer) == expected.translate(normalizer), body


def test_mandarin_ipa_covers_every_bundled_pinyin_syllable():
    for body in load_json("syllables.json")["pinyin"].split():
        initial, final, _ = split_pinyin(body + "1", strict=True)
        initial, final = canonical_pinyin_phones(tuple(phone for phone in (initial, final) if phone))
        assert transcribe_pinyin(initial, final), body


def test_english_ipa_covers_every_bundled_cmudict_phone():
    phones = {
        phone
        for pronunciations in load_cmudict().values()
        for pronunciation in pronunciations
        for phone in pronunciation
    }

    assert phones
    for phone in phones:
        segments, _ = transcribe_arpabet_phone(phone)
        assert segments
