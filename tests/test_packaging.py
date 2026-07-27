import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def test_wheel_contains_only_runtime_code_and_declared_audit_resources(tmp_path):
    project_root = Path(__file__).parents[1]
    source_root = tmp_path / "source"
    source_root.mkdir()

    for name in (
        "g2p_mix",
        "tests/cases",
        "tests/fixtures/third_party",
    ):
        source = project_root / name
        destination = source_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
    for name in (
        "LICENSE",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "VERSION",
        "pyproject.toml",
    ):
        shutil.copy2(project_root / name, source_root / name)

    wheel_dir = tmp_path / "wheel"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheel_dir),
        ],
        cwd=source_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr

    wheels = tuple(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        members = set(wheel.namelist())

    assert not any(member.endswith(".py") and "/tests/" in member for member in members)
    assert any(member.endswith("share/doc/g2p-mix/THIRD_PARTY_NOTICES.md") for member in members)

    expected_cases = {path.name for path in (project_root / "tests/cases").glob("*.json")}
    packaged_cases = {
        Path(member).name for member in members if "/share/g2p-mix/tests/cases/" in member and member.endswith(".json")
    }
    assert len(expected_cases) == 8
    assert packaged_cases == expected_cases

    fixture_root = project_root / "tests/fixtures/third_party"
    expected_fixtures = {
        path.relative_to(fixture_root).as_posix() for path in fixture_root.rglob("*") if path.is_file()
    }
    packaged_fixtures = {
        member.split("/share/g2p-mix/tests/fixtures/third_party/", 1)[1]
        for member in members
        if "/share/g2p-mix/tests/fixtures/third_party/" in member
    }
    assert len(expected_fixtures) == 21
    assert packaged_fixtures == expected_fixtures

    install_root = tmp_path / "install"
    installed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--no-index",
            "--target",
            str(install_root),
            str(wheels[0]),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert installed.returncode == 0, installed.stderr

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(install_root)
    smoke = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path;"
                "import g2p_mix;"
                "from g2p_mix.resources import load_json, load_lines;"
                f"assert Path(g2p_mix.__file__).is_relative_to(Path({str(install_root)!r}));"
                "assert load_json('phones.json')['ZH'];"
                "assert load_lines('phrases.txt');"
                "print('wheel-resource-smoke-ok')"
            ),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=20,
    )
    assert smoke.returncode == 0, smoke.stderr
    assert smoke.stdout.strip() == "wheel-resource-smoke-ok"
