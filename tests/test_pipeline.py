from dataclasses import dataclass

import pytest

from g2p_mix.backends.base import BackendCapabilities
from g2p_mix.errors import AlignmentError, ConfigurationError
from g2p_mix.models import (
    ChineseDialect,
    Language,
    PhoneAlphabet,
    Pronunciation,
    PronunciationUnit,
)
from g2p_mix.pipeline import G2PPipeline
from g2p_mix.profiles import ChineseProfile, EnglishProfile
from g2p_mix.renderers import NativeRenderer
from g2p_mix.text import IdentityNormalizer


class WholeChineseSegmenter:
    def segment(self, text):
        return [(text, "x")]


@dataclass
class RecordingBackend:
    name: str
    capabilities: BackendCapabilities
    calls: int = 0
    omit_last: bool = False

    def predict(self, request):
        self.calls += 1
        targets = list(request.target_tokens)
        if self.omit_last:
            targets = targets[:-1]

        result = {}
        for token in targets:
            alphabet = self.capabilities.alphabet
            if token.language is Language.CHINESE:
                units = tuple(
                    PronunciationUnit(
                        text=char,
                        source_spans=(token.source_spans[index],),
                        phones=("n", "i"),
                        tone="3",
                        alphabet=alphabet,
                        native="ni3",
                    )
                    for index, char in enumerate(token.text)
                )
            else:
                phones = ("T", "EH", "S", "T")
                units = (
                    PronunciationUnit(
                        text=token.text,
                        source_spans=token.source_spans,
                        phones=phones,
                        alphabet=alphabet,
                        native="T EH1 S T",
                        stress_marks=((1, 1),),
                    ),
                )
            result[token.id] = Pronunciation(
                token_id=token.id,
                units=units,
                backend=self.name,
            )
        return result


def make_pipeline(chinese_backend, english_backend):
    chinese = ChineseProfile(
        dialect=ChineseDialect.MANDARIN,
        backend=chinese_backend,
        segmenter=WholeChineseSegmenter(),
        normalizers=(IdentityNormalizer(),),
        processors=(),
    )
    return G2PPipeline(
        chinese=chinese,
        english=EnglishProfile(english_backend),
    )


def test_each_language_backend_is_called_once_and_results_are_reassembled():
    chinese = RecordingBackend(
        name="zh-fake",
        capabilities=BackendCapabilities(
            language=Language.CHINESE,
            alphabet=PhoneAlphabet.PINYIN,
            dialect=ChineseDialect.MANDARIN,
            contextual=True,
            supports_projection=True,
        ),
    )
    english = RecordingBackend(
        name="en-fake",
        capabilities=BackendCapabilities(
            language=Language.ENGLISH,
            alphabet=PhoneAlphabet.ARPABET,
        ),
    )
    converter = make_pipeline(chinese, english)

    result = converter("这个 make sense 不错")

    assert chinese.calls == 1
    assert english.calls == 1
    assert result.reconstruct_original() == "这个 make sense 不错"
    assert result.projections[Language.CHINESE].text == "这个 <EN> 不错"
    assert result.projections[Language.ENGLISH].text == "<ZH> make sense <ZH>"
    assert [token.token.text for token in result.tokens] == [
        "这个",
        " ",
        "make",
        " ",
        "sense",
        " ",
        "不错",
    ]
    assert NativeRenderer().render(result) == (
        "n",
        "i3",
        "n",
        "i3",
        "T",
        "EH1",
        "S",
        "T",
        "T",
        "EH1",
        "S",
        "T",
        "n",
        "i3",
        "n",
        "i3",
    )


def test_custom_english_backend_can_accept_extended_latin_text():
    chinese = RecordingBackend(
        name="zh-fake",
        capabilities=BackendCapabilities(
            language=Language.CHINESE,
            alphabet=PhoneAlphabet.PINYIN,
            dialect=ChineseDialect.MANDARIN,
        ),
    )
    english = RecordingBackend(
        name="unicode-en-fake",
        capabilities=BackendCapabilities(
            language=Language.ENGLISH,
            alphabet=PhoneAlphabet.ARPABET,
        ),
    )

    result = make_pipeline(chinese, english)("café")

    assert [unit.text for unit in result.units] == ["café"]
    assert result.units[0].source_spans == result.tokens[0].token.source_spans


def test_profile_rejects_a_backend_for_the_wrong_dialect():
    backend = RecordingBackend(
        name="cantonese",
        capabilities=BackendCapabilities(
            language=Language.CHINESE,
            alphabet=PhoneAlphabet.JYUTPING,
            dialect=ChineseDialect.CANTONESE,
        ),
    )

    with pytest.raises(ConfigurationError):
        ChineseProfile(
            dialect=ChineseDialect.MANDARIN,
            backend=backend,
            segmenter=WholeChineseSegmenter(),
            normalizers=(IdentityNormalizer(),),
            processors=(),
        )


def test_pipeline_validates_backend_coverage():
    chinese = RecordingBackend(
        name="broken",
        capabilities=BackendCapabilities(
            language=Language.CHINESE,
            alphabet=PhoneAlphabet.PINYIN,
            dialect=ChineseDialect.MANDARIN,
        ),
        omit_last=True,
    )
    english = RecordingBackend(
        name="en",
        capabilities=BackendCapabilities(
            language=Language.ENGLISH,
            alphabet=PhoneAlphabet.ARPABET,
        ),
    )

    with pytest.raises(AlignmentError, match="coverage mismatch"):
        make_pipeline(chinese, english)("中文")
