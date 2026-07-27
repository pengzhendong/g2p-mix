import pytest

from g2p_mix import IpaRenderer, MixedG2P
from g2p_mix.errors import UnsupportedFeatureError


def test_ipa_renderer_handles_mandarin_and_english_units():
    result = MixedG2P.mandarin(tone_sandhi=False)("中国idea")

    assert IpaRenderer().render(result) == (
        "ʈʂ",
        "ʊ→",
        "ŋ",
        "k",
        "w",
        "o↗",
        "I",
        "d",
        "ˈi",
        "ʌ",
    )


def test_cantonese_ipa_is_an_explicit_unsupported_feature():
    result = MixedG2P.cantonese()("你")

    with pytest.raises(UnsupportedFeatureError):
        IpaRenderer().render(result)
