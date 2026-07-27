import importlib.abc
import json
import os
import subprocess
import sys
import threading
import types
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from click.testing import CliRunner

from g2p_mix import resources
from g2p_mix.backends.base import BackendCapabilities, PronunciationRequest
from g2p_mix.backends.english import EnglishBackend
from g2p_mix.backends.mandarin import G2PWBackend
from g2p_mix.cli import main
from g2p_mix.errors import AlignmentError, BackendError, G2PError, RenderingError
from g2p_mix.models import (
    ChineseDialect,
    Language,
    NormalizedText,
    PhoneAlphabet,
    Pronunciation,
    PronunciationUnit,
    Span,
    TextToken,
    TokenKind,
)
from g2p_mix.phonetics import split_jyutping, split_pinyin
from g2p_mix.pipeline import MixedG2P
from g2p_mix.profiles import ChineseProfile, EnglishProfile
from g2p_mix.text import IdentityNormalizer, ProjectionBuilder, TextAnalyzer

CASE_FILE = Path(__file__).parent / "cases" / "phase5_contracts.json"
CASE_GROUPS = json.loads(CASE_FILE.read_text(encoding="utf-8"))


def cases(group):
    return [pytest.param(case, id=case["id"]) for case in CASE_GROUPS[group]]


def subprocess_env():
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    return environment


class WholeChineseSegmenter:
    def segment(self, text):
        return [(text, "x")]


def _pinyin_unit(token, index):
    native = "ni3" if token.text[index] == "你" else "hao3"
    initial, final, tone = split_pinyin(native)
    return PronunciationUnit(
        text=token.text[index],
        source_spans=(token.source_spans[index],),
        phones=tuple(phone for phone in (initial, final) if phone),
        tone=tone,
        alphabet=PhoneAlphabet.PINYIN,
        native=native,
    )


class MatrixBackend:
    name = "matrix"
    capabilities = BackendCapabilities(
        language=Language.CHINESE,
        dialect=ChineseDialect.MANDARIN,
        alphabet=PhoneAlphabet.PINYIN,
    )

    def __init__(self, mutation=None):
        self.mutation = mutation

    def predict(self, request):
        result = {}
        for token in request.target_tokens:
            units = tuple(_pinyin_unit(token, index) for index in range(len(token.text)))
            pronunciation = Pronunciation(token_id=token.id, units=units, backend=self.name)
            result[token.id] = pronunciation

        if not result or self.mutation is None:
            return result

        token_id = next(iter(result))
        pronunciation = result[token_id]
        units = list(pronunciation.units)
        first = units[0]
        if self.mutation == "non_mapping":
            return ["not-a-key-value-pair"]
        if self.mutation == "heterogeneous_keys":
            result["foreign"] = pronunciation
            result[None] = pronunciation
        elif self.mutation == "non_pronunciation":
            result[token_id] = {"units": units}
        elif self.mutation == "non_unit":
            result[token_id] = replace(pronunciation, units=(object(),))
        elif self.mutation == "non_span":
            units[0] = replace(first, source_spans=(object(),))
            result[token_id] = replace(pronunciation, units=tuple(units))
        elif self.mutation == "non_enum":
            units[0] = replace(first, alphabet="pinyin")
            result[token_id] = replace(pronunciation, units=tuple(units))
        elif self.mutation == "bad_phones":
            units[0] = replace(first, phones=(None,))
            result[token_id] = replace(pronunciation, units=tuple(units))
        elif self.mutation == "wrong_token_id":
            result[token_id] = replace(pronunciation, token_id=token_id + 100)
        elif self.mutation == "wrong_alphabet":
            units[0] = replace(first, alphabet=PhoneAlphabet.ARPABET)
            result[token_id] = replace(pronunciation, units=tuple(units))
        elif self.mutation == "foreign_span":
            units[0] = replace(first, source_spans=(Span(3, 4),))
            result[token_id] = replace(pronunciation, units=tuple(units))
        elif self.mutation == "out_of_range_span":
            units[0] = replace(first, source_spans=(Span(100, 101),))
            result[token_id] = replace(pronunciation, units=tuple(units))
        elif self.mutation == "reversed_spans":
            result[token_id] = replace(pronunciation, units=tuple(reversed(units)))
        elif self.mutation == "wrong_source_slice":
            units[0] = replace(first, text="好")
            result[token_id] = replace(pronunciation, units=tuple(units))
        elif self.mutation == "missing_coverage":
            result[token_id] = replace(pronunciation, units=(units[0],))
        elif self.mutation == "wrong_tone":
            units[0] = replace(first, tone="4")
            result[token_id] = replace(pronunciation, units=tuple(units))
        elif self.mutation == "wrong_native":
            units[0] = replace(first, native="hao3")
            result[token_id] = replace(pronunciation, units=tuple(units))
        elif self.mutation == "wrong_phones":
            units[0] = replace(first, phones=("x",))
            result[token_id] = replace(pronunciation, units=tuple(units))
        return result


class MatrixEnglishBackend:
    name = "matrix-english"
    capabilities = BackendCapabilities(language=Language.ENGLISH, alphabet=PhoneAlphabet.ARPABET)

    def predict(self, request):
        return {
            token.id: Pronunciation(
                token_id=token.id,
                units=(
                    PronunciationUnit(
                        text=token.text,
                        source_spans=token.source_spans,
                        phones=("EY1",),
                        alphabet=PhoneAlphabet.ARPABET,
                        native="EY1",
                    ),
                ),
                backend=self.name,
            )
            for token in request.target_tokens
        }


class MatrixProcessor:
    def __init__(self, mutation):
        self.mutation = mutation

    def process(self, tokens, pronunciations):
        result = dict(pronunciations)
        token_id = next(iter(result))
        pronunciation = result[token_id]
        if self.mutation == "non_mapping":
            return ["not-a-key-value-pair"]
        if self.mutation == "non_pronunciation":
            result[token_id] = {"units": pronunciation.units}
        elif self.mutation == "add":
            result[999] = pronunciation
        elif self.mutation == "drop":
            result.pop(token_id)
        elif self.mutation == "replace_token_id":
            result[token_id] = replace(pronunciation, token_id=999)
        elif self.mutation == "replace_alphabet":
            unit = replace(pronunciation.units[0], alphabet=PhoneAlphabet.ARPABET)
            result[token_id] = replace(pronunciation, units=(unit,) + pronunciation.units[1:])
        elif self.mutation == "merge_units":
            first, second = pronunciation.units
            unit = replace(
                first,
                text=first.text + second.text,
                source_spans=first.source_spans + second.source_spans,
            )
            result[token_id] = replace(pronunciation, units=(unit,))
        elif self.mutation == "drop_jyutping_unit":
            result[token_id] = replace(pronunciation, units=pronunciation.units[:1])
        elif self.mutation == "rewrite_backend":
            result[token_id] = replace(pronunciation, backend="rewritten")
        elif self.mutation == "legal_tone_change":
            units = (pronunciation.units[0].with_tone("2"),) + pronunciation.units[1:]
            result[token_id] = replace(pronunciation, units=units)
        return result


class JyutpingMatrixBackend:
    name = "matrix-jyutping"
    capabilities = BackendCapabilities(
        language=Language.CHINESE,
        dialect=ChineseDialect.CANTONESE,
        alphabet=PhoneAlphabet.JYUTPING,
    )

    def predict(self, request):
        result = {}
        for token in request.target_tokens:
            units = []
            for native in ("cin1", "ngaa5"):
                initial, final, tone = split_jyutping(native)
                units.append(
                    PronunciationUnit(
                        text=token.text,
                        source_spans=token.source_spans,
                        phones=tuple(phone for phone in (initial, final) if phone),
                        tone=tone,
                        alphabet=PhoneAlphabet.JYUTPING,
                        native=native,
                    )
                )
            result[token.id] = Pronunciation(token_id=token.id, units=tuple(units), backend=self.name)
        return result


def make_matrix_pipeline(*, mutation=None, processor=None):
    return MixedG2P(
        chinese=ChineseProfile(
            dialect=ChineseDialect.MANDARIN,
            backend=MatrixBackend(mutation),
            segmenter=WholeChineseSegmenter(),
            normalizers=(IdentityNormalizer(),),
            processors=(() if processor is None else (processor,)),
        ),
        english=EnglishProfile(MatrixEnglishBackend()),
    )


def make_identity_pipeline(case):
    processor = MatrixProcessor(case["mutation"])
    if case["profile"] == "cantonese":
        chinese = ChineseProfile(
            dialect=ChineseDialect.CANTONESE,
            backend=JyutpingMatrixBackend(),
            segmenter=WholeChineseSegmenter(),
            normalizers=(IdentityNormalizer(),),
            processors=(processor,),
        )
    else:
        chinese = ChineseProfile(
            dialect=ChineseDialect.MANDARIN,
            backend=MatrixBackend(),
            segmenter=WholeChineseSegmenter(),
            normalizers=(IdentityNormalizer(),),
            processors=(processor,),
        )
    return MixedG2P(chinese=chinese, english=EnglishProfile(MatrixEnglishBackend()))


@pytest.mark.parametrize("case", cases("backend_malformed"))
def test_pipeline_rejects_malformed_backend_results(case):
    with pytest.raises(AlignmentError, match=case["message"]):
        make_matrix_pipeline(mutation=case["mutation"])("你好 A")


@pytest.mark.parametrize("case", cases("processor_malformed"))
def test_pipeline_revalidates_processor_results(case):
    with pytest.raises(AlignmentError, match=case["message"]):
        make_matrix_pipeline(processor=MatrixProcessor(case["mutation"]))("你好")


@pytest.mark.parametrize("case", cases("processor_identity"))
def test_processor_preserves_backend_and_unit_alignment_identity(case):
    pipeline = make_identity_pipeline(case)
    text = "瓩" if case["profile"] == "cantonese" else "你好"
    if case["expected"] == "error":
        with pytest.raises(AlignmentError, match=case["message"]):
            pipeline(text)
        return

    result = pipeline(text)
    assert [unit.native for unit in result.units] == case["expected_native"]
    assert [unit.tone for unit in result.units] == case["expected_tones"]


@pytest.mark.parametrize("case", cases("structural_malformed"))
def test_pipeline_structural_guard_always_raises_alignment_error(case):
    kwargs = (
        {"mutation": case["mutation"]}
        if case["producer"] == "backend"
        else {"processor": MatrixProcessor(case["mutation"])}
    )
    with pytest.raises(AlignmentError, match=case["message"]):
        make_matrix_pipeline(**kwargs)("你好")


def _case_pronunciation(case):
    language = Language(case["language"])
    alphabet = PhoneAlphabet(case["alphabet"])
    token_spans = tuple(Span(*values) for values in case["token_spans"])
    token = TextToken(
        id=0,
        text=case["token_text"],
        normalized_span=Span(0, len(case["token_text"])),
        source_spans=token_spans,
        language=language,
        kind=TokenKind.HAN if language is Language.CHINESE else TokenKind.LATIN,
    )
    units = []
    for values in case["units"]:
        native = values["native"]
        if alphabet is PhoneAlphabet.PINYIN:
            initial, final, tone = split_pinyin(native)
            phones = tuple(phone for phone in (initial, final) if phone)
        elif alphabet is PhoneAlphabet.JYUTPING:
            initial, final, tone = split_jyutping(native)
            phones = tuple(phone for phone in (initial, final) if phone)
        else:
            phones = tuple(native.split())
            tone = None
        units.append(
            PronunciationUnit(
                text=values["text"],
                source_spans=tuple(token_spans[index] for index in values["span_indexes"]),
                phones=phones,
                alphabet=alphabet,
                native=native,
                tone=tone,
            )
        )
    return token, Pronunciation(token_id=token.id, units=tuple(units), backend="matrix")


@pytest.mark.parametrize("case", cases("occurrence_alignment_valid"))
def test_validator_accepts_monotonic_occurrence_alignment(case):
    token, pronunciation = _case_pronunciation(case)
    MixedG2P._validate_prediction(
        producer="matrix",
        result={token.id: pronunciation},
        expected_tokens=(token,),
        expected_alphabets={token.language: pronunciation.units[0].alphabet},
        source_text=case["source_text"],
    )


@pytest.mark.parametrize("case", cases("occurrence_alignment_invalid"))
def test_validator_rejects_illegal_occurrence_reuse(case):
    token, pronunciation = _case_pronunciation(case)
    with pytest.raises(AlignmentError, match=case["message"]):
        MixedG2P._validate_prediction(
            producer="matrix",
            result={token.id: pronunciation},
            expected_tokens=(token,),
            expected_alphabets={token.language: pronunciation.units[0].alphabet},
            source_text=case["source_text"],
        )


@pytest.mark.parametrize("case", cases("english_offline"))
def test_fresh_english_conversion_never_downloads_or_opens_sockets(case, tmp_path):
    code = """
import json
import socket
import sys
import nltk
from g2p_mix.backends.english import EnglishBackend

nltk.data.path[:] = [sys.argv[2]]
events = []

def forbidden_download(*args, **kwargs):
    events.append(["download", repr(args)])
    raise AssertionError("nltk.download must not be called")

def forbidden_socket(*args, **kwargs):
    events.append(["socket", repr(args)])
    raise AssertionError("network socket must not be opened")

nltk.download = forbidden_download
socket.create_connection = forbidden_socket
socket.socket.connect = forbidden_socket
socket.socket.connect_ex = forbidden_socket
phones = EnglishBackend().convert(sys.argv[1])
print(json.dumps({"phones": phones, "events": events}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, case["word"], str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        env=subprocess_env(),
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert bool(payload["phones"]) is case["expected_nonempty"]
    assert payload["events"] == []


def test_missing_nltk_is_wrapped_as_backend_error_with_cause():
    code = """
import importlib.abc

class BlockNltk(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "nltk" or fullname.startswith("nltk."):
            raise ModuleNotFoundError("blocked nltk", name=fullname)
        return None

import sys
sys.meta_path.insert(0, BlockNltk())
from g2p_mix.backends.english import EnglishBackend
from g2p_mix.errors import BackendError

try:
    EnglishBackend().convert("idea")
except BackendError as error:
    assert "NLTK" in str(error)
    assert isinstance(error.__cause__, ModuleNotFoundError)
else:
    raise AssertionError("BackendError was not raised")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=subprocess_env(),
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("case", cases("nltk_path_contract"))
def test_bundled_nltk_path_is_concurrent_persistent_deduplicated_fallback(case):
    import g2p_en
    import nltk

    bundled = str(resources.NLTK_DATA_DIR)
    original = list(nltk.data.path)
    nltk.data.path[:] = [case["user_paths"][0], bundled, case["user_paths"][1], bundled]
    barrier = threading.Barrier(case["workers"])

    def ensure_after_barrier(_):
        barrier.wait()
        resources.ensure_bundled_nltk_data()

    try:
        with ThreadPoolExecutor(max_workers=case["workers"]) as executor:
            tuple(executor.map(ensure_after_barrier, range(case["workers"])))
        resources.ensure_bundled_nltk_data()

        assert nltk.data.path == case["user_paths"] + [bundled]
        assert nltk.data.path.count(bundled) == 1
        assert g2p_en is sys.modules["g2p_en"]
        assert hasattr(resources, "_NLTK_DATA_PATH_LOCK")
        assert not hasattr(resources, "_temporary_nltk_path")
    finally:
        nltk.data.path[:] = original


class _SentinelG2PError(G2PError):
    pass


class _SentinelException(Exception):
    pass


class _SentinelBaseException(BaseException):
    pass


def _sentinel(failure):
    return {
        "g2p": _SentinelG2PError,
        "exception": _SentinelException,
        "base": _SentinelBaseException,
    }[failure]("injected boundary failure")


@pytest.mark.parametrize("case", cases("g2pw_boundaries"))
def test_g2pw_boundary_exception_policy(case, monkeypatch):
    failure = _sentinel(case["failure"])

    def fail():
        raise failure

    if case["boundary"] == "predict":
        backend = G2PWBackend(converter=lambda text: fail())

        def invoke():
            return backend._convert("你")

    else:
        modelscope = types.ModuleType("modelscope")
        g2pw = types.ModuleType("g2pw")

        if case["boundary"] == "snapshot":
            modelscope.snapshot_download = lambda name: fail()
            g2pw.G2PWConverter = object
        else:
            modelscope.snapshot_download = lambda name: "/tmp/g2pw-contract"
            g2pw.G2PWConverter = lambda **kwargs: fail()
        monkeypatch.setitem(sys.modules, "modelscope", modelscope)
        monkeypatch.setitem(sys.modules, "g2pw", g2pw)
        invoke = G2PWBackend()._get_converter

    if case["expected"] == "wrapped":
        with pytest.raises(BackendError) as captured:
            invoke()
        assert captured.value is not failure
        assert captured.value.__cause__ is failure
    else:
        with pytest.raises(type(failure)) as captured:
            invoke()
        assert captured.value is failure


class _FailingSequence(Sequence):
    def __init__(self, failure):
        self.failure = failure

    def __getitem__(self, index):
        raise self.failure

    def __len__(self):
        raise self.failure


def _g2pw_shape(case):
    shape = case["shape"]
    if shape == "outer_mapping":
        return {"sentence": ["ni3", "hao3"]}, None
    if shape == "outer_string":
        return "ni3 hao3", None
    if shape == "outer_bytes":
        return b"ni3 hao3", None
    if shape == "two_sentences":
        return [["ni3", "hao3"], ["ni3", "hao3"]], None
    if shape == "inner_integer":
        return [42], None
    if shape == "inner_string":
        return ["ni3 hao3"], None
    if shape == "inner_bytes":
        return [b"ni3 hao3"], None
    if shape == "wrong_position_count":
        return [["ni3"]], None
    if shape == "non_string_syllable":
        return [["ni3", object()]], None
    failure = _sentinel(shape.removeprefix("len_"))
    return _FailingSequence(failure), failure


def _g2pw_request():
    tokens = TextAnalyzer(WholeChineseSegmenter()).analyze(NormalizedText.identity("你好"))
    projection = ProjectionBuilder().build(tokens, Language.CHINESE)
    return PronunciationRequest(
        tokens=tokens,
        projection=projection,
        dialect=ChineseDialect.MANDARIN,
    )


@pytest.mark.parametrize("case", cases("g2pw_return_shapes"))
def test_g2pw_converter_return_shape_contract(case):
    converter_result, failure = _g2pw_shape(case)
    backend = G2PWBackend(converter=lambda text: converter_result)

    if case.get("expected") == "same":
        with pytest.raises(type(failure)) as captured:
            backend.predict(_g2pw_request())
        assert captured.value is failure
        return

    with pytest.raises(BackendError, match=case["message"]) as captured:
        backend.predict(_g2pw_request())
    if failure is not None:
        assert captured.value.__cause__ is failure


@pytest.mark.parametrize("case", cases("english_third_party_failures"))
def test_english_third_party_failures_are_wrapped(case):
    failure = _SentinelException("injected English dependency failure")

    def fail(_):
        raise failure

    def valid_predictor(_):
        return ["T"]

    def malformed_segmenter(_):
        return [object()]

    def identity_segmenter(word):
        return [word]

    def malformed_predictor(_):
        return [None]

    if case["boundary"] == "segmenter":
        segmenter = fail if case["failure"] == "exception" else malformed_segmenter
        predictor = valid_predictor
    else:
        segmenter = identity_segmenter
        predictor = fail if case["failure"] == "exception" else malformed_predictor
    backend = EnglishBackend(dictionary={}, segmenter=segmenter, predictor=predictor)

    with pytest.raises(BackendError, match=case["message"]) as captured:
        backend.convert("testing")
    assert captured.value.__cause__ is not None


@pytest.mark.parametrize("case", cases("cli_success"))
def test_cli_success_matrix(case):
    completed = CliRunner().invoke(main, case["args"])

    assert completed.exit_code == 0, completed.output
    assert "Traceback" not in completed.output
    if case["format"] == "json":
        payload = json.loads(completed.output)
        assert payload["text"] == case["args"][0]
        assert isinstance(payload["tokens"], list)
    elif case["format"] == "empty":
        assert completed.output == ""
    else:
        assert completed.output.strip()


@pytest.mark.parametrize("case", cases("cli_errors"))
def test_cli_error_matrix(case):
    completed = CliRunner().invoke(main, case["args"])

    assert completed.exit_code == case["exit"]
    assert case["message"] in completed.output
    assert "Traceback" not in completed.output


def test_missing_g2pw_cli_has_install_hint_without_traceback():
    class BlockOptionalG2PW(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "g2pw" or fullname.startswith("g2pw."):
                raise ModuleNotFoundError("blocked g2pw", name=fullname)
            return None

    blocker = BlockOptionalG2PW()
    sys.meta_path.insert(0, blocker)
    try:
        completed = CliRunner().invoke(main, ["你", "--mandarin-backend", "g2pw"])
    finally:
        sys.meta_path.remove(blocker)

    assert completed.exit_code != 0
    assert "g2p-mix[g2pw]" in completed.output
    assert "Traceback" not in completed.output


@pytest.mark.parametrize("case", cases("cli_runtime_errors"))
def test_cli_runtime_errors_have_no_traceback(case, monkeypatch):
    class BrokenRenderer:
        def render_unit(self, unit):
            raise RenderingError(case["message"])

    monkeypatch.setattr("g2p_mix.cli.NativeRenderer", BrokenRenderer)
    completed = CliRunner().invoke(main, case["args"])

    assert completed.exit_code == 1
    assert case["message"] in completed.output
    assert "Traceback" not in completed.output


def test_missing_g2pw_backend_error_retains_module_cause(monkeypatch):
    class BlockG2PW(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "g2pw" or fullname.startswith("g2pw."):
                raise ModuleNotFoundError("blocked g2pw", name=fullname)
            return None

    blocker = BlockG2PW()
    sys.meta_path.insert(0, blocker)
    try:
        from g2p_mix import G2PWBackend

        with pytest.raises(BackendError, match=r"g2p-mix\[g2pw\]") as captured:
            G2PWBackend()._get_converter()
    finally:
        sys.meta_path.remove(blocker)

    assert isinstance(captured.value.__cause__, ModuleNotFoundError)


@pytest.mark.parametrize("case", cases("property_matrix"))
def test_projection_and_source_span_property_matrix(case):
    value = NormalizedText.identity(case["text"])
    tokens = TextAnalyzer(WholeChineseSegmenter()).analyze(value)
    builder = ProjectionBuilder()

    assert [token.text for token in tokens] == case["tokens"]
    assert builder.build(tokens, Language.CHINESE).text == case["zh_projection"]
    assert builder.build(tokens, Language.ENGLISH).text == case["en_projection"]
    assert "".join(token.text for token in tokens) == case["text"]
    assert ["".join(span.slice(case["text"]) for span in token.source_spans) for token in tokens] == case["tokens"]
