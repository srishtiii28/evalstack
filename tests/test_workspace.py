"""The workspace is the security boundary, so these tests are adversarial."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from evalforge.env.limits import probe_limit_support
from evalforge.env.workspace import (
    PathEscapeError,
    ResourceLimits,
    Workspace,
    workspace_for,
)
from evalforge.schema.case import EvalCase, FileSpec


def make_case(**overrides: object) -> EvalCase:
    defaults: dict[str, object] = {
        "case_id": "demo",
        "prompt": "Fix the bug.",
        "files": (FileSpec(path="pkg/calc.py", contents="def add(a, b):\n    return a + b\n"),),
        "test_command": ("python", "-c", "print('ok')"),
    }
    return EvalCase.model_validate(defaults | overrides)


@pytest.fixture
def workspace(tmp_path: Path):
    with workspace_for(make_case(), base_dir=tmp_path) as ws:
        yield ws


def test_seed_files_are_materialised(workspace: Workspace) -> None:
    assert workspace.list_files() == ("pkg/calc.py",)
    assert "def add" in workspace.read_file("pkg/calc.py")


def test_write_then_read_roundtrip(workspace: Workspace) -> None:
    record = workspace.write_file("pkg/calc.py", "def add(a, b):\n    return a - b\n")

    assert workspace.read_file("pkg/calc.py").endswith("a - b\n")
    assert record.before_hash is not None
    assert record.after_hash != record.before_hash
    assert (record.lines_added, record.lines_removed) == (1, 1)


def test_writing_a_new_file_reports_creation(workspace: Workspace) -> None:
    record = workspace.write_file("pkg/new.py", "x = 1\n")

    assert record.created is True
    assert record.before_hash is None
    assert record.lines_added == 1


def test_parent_traversal_is_refused_and_recorded(workspace: Workspace) -> None:
    with pytest.raises(PathEscapeError):
        workspace.write_file("../escaped.txt", "nope")

    assert [violation.rule for violation in workspace.violations] == ["path_escape"]
    assert workspace.violations[0].attempted == "../escaped.txt"
    assert not (workspace.root.parent / "escaped.txt").exists()


def test_absolute_path_is_refused(workspace: Workspace) -> None:
    with pytest.raises(PathEscapeError):
        workspace.read_file("/etc/passwd")

    assert workspace.violations[0].rule == "path_escape"


def test_symlink_out_of_the_workspace_is_refused(workspace: Workspace, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace.root / "link").symlink_to(outside)

    with pytest.raises(PathEscapeError):
        workspace.write_file("link/pwned.txt", "nope")

    assert not (outside / "pwned.txt").exists()
    assert workspace.violations[0].rule == "path_escape"


def test_violation_hook_is_invoked(tmp_path: Path) -> None:
    seen: list[str] = []
    with workspace_for(
        make_case(), base_dir=tmp_path, on_violation=lambda v: seen.append(v.attempted)
    ) as ws, pytest.raises(PathEscapeError):
        ws.read_file("../../etc/passwd")

    assert seen == ["../../etc/passwd"]


def test_diff_reports_added_modified_and_deleted(workspace: Workspace) -> None:
    workspace.write_file("pkg/calc.py", "def add(a, b):\n    return a * b\n")
    workspace.write_file("pkg/extra.py", "y = 2\n")
    workspace.write_file("pkg/gone.py", "z = 3\n")
    workspace.set_baseline()

    workspace.write_file("pkg/calc.py", "def add(a, b):\n    return a + b\n")
    workspace.write_file("pkg/fresh.py", "w = 4\n")
    workspace.delete_file("pkg/gone.py")

    diff = workspace.diff()
    assert diff.added == ("pkg/fresh.py",)
    assert diff.modified == ("pkg/calc.py",)
    assert diff.deleted == ("pkg/gone.py",)
    assert diff.touched == ("pkg/calc.py", "pkg/fresh.py", "pkg/gone.py")


def test_untouched_workspace_has_empty_diff(workspace: Workspace) -> None:
    assert workspace.diff().is_empty is True


def test_generated_artefacts_are_not_counted_as_edits(workspace: Workspace) -> None:
    cache = workspace.root / "pkg" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "calc.cpython-312.pyc").write_bytes(b"\x00")

    assert workspace.list_files() == ("pkg/calc.py",)
    assert workspace.diff().is_empty is True


async def test_run_executes_inside_the_workspace(workspace: Workspace) -> None:
    result = await workspace.run(("python", "-c", "import os; print(os.getcwd())"), timeout_s=30)

    assert result.ok is True
    assert result.exit_code == 0
    assert result.stdout.strip() == str(workspace.root)


async def test_run_reports_failure_exit_codes(workspace: Workspace) -> None:
    result = await workspace.run(("python", "-c", "raise SystemExit(3)"), timeout_s=30)

    assert result.ok is False
    assert result.exit_code == 3
    assert result.timed_out is False


async def test_run_times_out_and_kills_the_process(workspace: Workspace) -> None:
    result = await workspace.run(("python", "-c", "import time; time.sleep(30)"), timeout_s=1.0)

    assert result.timed_out is True
    assert result.exit_code is None
    assert result.duration_ms < 15_000


async def test_run_environment_is_scrubbed(workspace: Workspace) -> None:
    os.environ["EVALFORGE_LEAK_CANARY"] = "leaked"
    try:
        result = await workspace.run(
            ("python", "-c", "import os; print(os.environ.get('EVALFORGE_LEAK_CANARY', 'absent'))"),
            timeout_s=30,
        )
    finally:
        del os.environ["EVALFORGE_LEAK_CANARY"]

    assert result.stdout.strip() == "absent"


async def test_run_home_points_outside_the_tracked_tree(workspace: Workspace) -> None:
    result = await workspace.run(
        ("python", "-c", "import os; print(os.environ['HOME'])"), timeout_s=30
    )

    assert result.stdout.strip() == str(workspace.home)
    assert workspace.root not in Path(result.stdout.strip()).parents


@pytest.mark.skipif(
    not probe_limit_support().memory,
    reason="platform does not enforce address-space limits (Darwin rejects RLIMIT_AS)",
)
async def test_memory_limit_is_enforced(tmp_path: Path) -> None:
    limits = ResourceLimits(memory_mb=256)
    with workspace_for(make_case(), base_dir=tmp_path, limits=limits) as ws:
        result = await ws.run(
            ("python", "-c", "x = bytearray(1024 * 1024 * 512); print(len(x))"),
            timeout_s=60,
        )

    assert result.ok is False


@pytest.mark.skipif(
    not probe_limit_support().file_size,
    reason="platform does not enforce file-size limits",
)
async def test_file_size_limit_is_enforced(tmp_path: Path) -> None:
    limits = ResourceLimits(file_size_mb=1)
    with workspace_for(make_case(), base_dir=tmp_path, limits=limits) as ws:
        result = await ws.run(
            ("python", "-c", "open('big.bin','wb').write(b'x' * (4 * 1024 * 1024))"),
            timeout_s=60,
        )

    assert result.ok is False


def test_limit_support_is_described_honestly() -> None:
    support = probe_limit_support()
    description = support.describe()

    if support.all_supported:
        assert description == "cpu, memory and file-size limits enforced"
    else:
        assert description.startswith("not enforced on this platform:")
        for name in support.unsupported():
            assert name in description


async def test_empty_argv_is_rejected(workspace: Workspace) -> None:
    with pytest.raises(ValueError, match="argv must not be empty"):
        await workspace.run((), timeout_s=5)
