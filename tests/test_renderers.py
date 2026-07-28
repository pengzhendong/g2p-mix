from g2p_mix import IpaRenderer, MixedG2P


def test_ipa_renderer_handles_mandarin_and_english_units():
    result = MixedG2P.mandarin(tone_sandhi=False)("中国idea")

    assert IpaRenderer().render(result) == (
        "ʈ͡ʂ",
        "ʊ",
        "ŋ˥˥",
        "k",
        "w",
        "o˧˥",
        "a",
        "ɪ",
        "d",
        "ˈi",
        "ə",
    )


def test_cantonese_ipa_renderer_uses_jyutping_tone_contours():
    result = MixedG2P.cantonese()("廣東話")

    assert IpaRenderer().render(result) == (
        "kʷ",
        "ɔ",
        "ŋ˨˥",
        "t",
        "ʊ",
        "ŋ˥˥",
        "w",
        "aː˨˥",
    )
