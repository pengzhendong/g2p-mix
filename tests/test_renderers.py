from g2p_mix import G2P
from g2p_mix.renderers import IpaRenderer


def test_ipa_renderer_handles_mandarin_and_english_units():
    result = G2P("mandarin", tone_sandhi=False)("中国idea")

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
    result = G2P("cantonese")("廣東話")

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
