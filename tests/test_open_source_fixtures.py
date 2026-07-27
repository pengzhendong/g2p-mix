import ast
import hashlib
import json
import re
from pathlib import Path

import pytest

from g2p_mix import resources
from g2p_mix.backends.cantonese import ToJyutpingBackend
from g2p_mix.backends.english import EnglishBackend
from g2p_mix.backends.mandarin import PypinyinBackend

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "third_party"
MANIFEST = json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))
SOURCE_DIRECTORIES = [FIXTURE_ROOT / name for name in MANIFEST["sources"]]
REQUIRED_SOURCE_FILES = {
    "LICENSE.txt",
    "NOTICE.md",
    "SOURCE.json",
    "cases.json",
    "raw_excerpt.txt",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
LICENSE_EXPRESSIONS = {
    "Apache-2.0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "LicenseRef-CMUdict",
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_cc_cedict(text):
    records = []
    pattern = re.compile(r"^(\S+) (\S+) \[([^\]]+)\] /")
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            raise ValueError(f"invalid CC-CEDICT fixture record: {line!r}")
        traditional, simplified, pinyin = match.groups()
        records.append(
            {
                "traditional": traditional,
                "simplified": simplified,
                "pinyin": [syllable.lower() for syllable in pinyin.split()],
            }
        )
    return records


def parse_python_repl(text):
    pending_inputs = None
    records = []
    for line in text.splitlines():
        if line.startswith(">>> ") and " = " in line:
            _, expression = line[4:].split(" = ", 1)
            value = ast.literal_eval(expression)
            pending_inputs = value if isinstance(value, list) else [value]
        elif line.startswith("[["):
            if pending_inputs is None:
                raise ValueError("g2pW output has no preceding input assignment")
            outputs = ast.literal_eval(line)
            if len(pending_inputs) != len(outputs):
                raise ValueError("g2pW README input/output batch sizes do not match")
            records.extend(
                {"input": input_text, "output": output} for input_text, output in zip(pending_inputs, outputs)
            )
            pending_inputs = None
    if pending_inputs is not None:
        raise ValueError("g2pW input assignment has no output")
    return records


def parse_hkcancor(text):
    records = []
    for line in text.splitlines():
        fields = line.strip().split("/")
        if len(fields) != 4 or fields[-1]:
            raise ValueError(f"invalid HKCanCor fixture record: {line!r}")
        token, pos, romanization, _ = fields
        records.append({"token": token, "pos": pos, "romanization": romanization})
    return records


def parse_cmudict(text):
    records = []
    for line in text.splitlines():
        word, *phones = line.split()
        if not phones:
            raise ValueError(f"invalid CMUdict fixture record: {line!r}")
        records.append({"word": word, "phones": phones})
    return records


PARSERS = {
    "cc_cedict": parse_cc_cedict,
    "python_repl": parse_python_repl,
    "hkcancor": parse_hkcancor,
    "cmudict": parse_cmudict,
}


@pytest.mark.parametrize("source_directory", SOURCE_DIRECTORIES, ids=lambda path: path.name)
def test_third_party_fixture_provenance_and_schema(source_directory):
    assert {path.name for path in source_directory.iterdir()} == REQUIRED_SOURCE_FILES
    source = load_json(source_directory / "SOURCE.json")
    cases = load_json(source_directory / "cases.json")

    required_source_fields = {
        "schema_version",
        "name",
        "parser",
        "annotation_group",
        "source_type",
        "revision_type",
        "revision",
        "source_url",
        "source_file",
        "excerpt_sha256",
        "license_spdx",
        "license_source_url",
        "license_source_sha256",
        "license_sha256",
        "license_normalization",
        "extraction_rule",
        "attribution",
    }
    assert required_source_fields <= source.keys()
    assert source["schema_version"] == cases["schema_version"] == 1
    assert source["parser"] in PARSERS
    assert source["annotation_group"] in cases
    assert source["source_type"] in {"archive", "direct_file"}
    assert source["license_spdx"] in LICENSE_EXPRESSIONS

    hash_fields = {"excerpt_sha256", "license_source_sha256", "license_sha256"}
    if source["source_type"] == "archive":
        archive_fields = {"archive_sha256", "archive_member", "archive_member_sha256"}
        assert archive_fields <= source.keys()
        assert "source_sha256" not in source
        assert source["archive_member"] in source["extraction_rule"]
        hash_fields.update({"archive_sha256", "archive_member_sha256"})
    else:
        assert "source_sha256" in source
        assert not {"archive_sha256", "archive_member", "archive_member_sha256"} & source.keys()
        hash_fields.add("source_sha256")

    for field in hash_fields:
        assert SHA256_PATTERN.fullmatch(source[field]), f"{source_directory.name}: invalid {field}"

    if source["revision_type"] == "git_commit":
        assert re.fullmatch(r"[0-9a-f]{40}", source["revision"])
        assert source["revision"] in source["source_url"]
        assert source["revision"] in source["license_source_url"]

    assert sha256(source_directory / "raw_excerpt.txt") == source["excerpt_sha256"]
    assert sha256(source_directory / "LICENSE.txt") == source["license_sha256"]
    assert source["attribution"] in (source_directory / "NOTICE.md").read_text(encoding="utf-8") or source[
        "name"
    ].split(" ")[0] in (source_directory / "NOTICE.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("source_directory", SOURCE_DIRECTORIES, ids=lambda path: path.name)
def test_raw_excerpt_parses_to_externalized_annotations(source_directory):
    source = load_json(source_directory / "SOURCE.json")
    cases = load_json(source_directory / "cases.json")
    raw_excerpt = (source_directory / "raw_excerpt.txt").read_text(encoding="utf-8")

    assert PARSERS[source["parser"]](raw_excerpt) == cases[source["annotation_group"]]


def test_fixture_bundle_is_small_and_not_a_corpus_cache():
    files = [path for path in FIXTURE_ROOT.rglob("*") if path.is_file()]
    assert sum(path.stat().st_size for path in files) <= MANIFEST["maximum_total_bytes"]
    assert not any(path.suffix in {".zip", ".xz", ".tar", ".bin", ".pt", ".onnx"} for path in files)


def test_root_third_party_notice_covers_every_fixture():
    notice = (FIXTURE_ROOT.parents[2] / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for source_directory in SOURCE_DIRECTORIES:
        source = load_json(source_directory / "SOURCE.json")
        assert source["name"].split(" ")[0] in notice
        assert source["revision"] in notice


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, id=case["id"])
        for case in load_json(FIXTURE_ROOT / "cc_cedict" / "cases.json")["project_regressions"]
    ],
)
def test_cc_cedict_regressions_match_pypinyin_backend(case):
    assert PypinyinBackend()._convert(case["text"]) == case["expected_native"]


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, id=case["id"])
        for case in load_json(FIXTURE_ROOT / "hkcancor" / "cases.json")["compatible_project_regressions"]
    ],
)
def test_hkcancor_compatible_annotations_match_tojyutping(case):
    converted = ToJyutpingBackend()._convert(case["text"])
    assert [pronunciation for _, pronunciation in converted] == case["expected_native"]


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, id=case["id"])
        for case in load_json(FIXTURE_ROOT / "hkcancor" / "cases.json")["known_backend_differences"]
    ],
)
def test_hkcancor_known_differences_are_explicit_not_backend_failures(case):
    converted = ToJyutpingBackend()._convert(case["text"])
    actual = [pronunciation for _, pronunciation in converted]

    assert actual == case["tojyutping_native"]
    assert "".join(actual) != case["corpus_romanization"]
    assert case["disposition"]


def test_cmudict_annotations_are_separate_from_force_spelling_policy():
    cases = load_json(FIXTURE_ROOT / "cmudict" / "cases.json")
    corpus_by_word = {case["word"]: case["phones"] for case in cases["corpus_annotations"]}
    policy = cases["project_force_spelling_policy"]
    backend = EnglishBackend()

    assert tuple(case["word"] for case in policy) == resources.load_english_force_spellings()
    for case in policy:
        assert backend.convert(case["word"]) == case["expected_spelled_phones"]
        assert case["expected_spelled_phones"] != corpus_by_word[case["word"].lower()]
