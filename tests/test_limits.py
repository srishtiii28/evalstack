"""Resource limits: the launcher that applies them, and the probe that reports them.

The launcher only ever runs as a subprocess (it ``execvp``s over itself), so it
is driven as one here. The probe's failure paths are exercised in-process,
because a probe that cannot answer must report "nothing enforced" rather than
crash a run.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from evalforge.env import _launcher, limits
from evalforge.env.limits import LimitSupport, probe_limit_support

LAUNCHER = [sys.executable, "-m", "evalforge.env._launcher"]


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    probe_limit_support.cache_clear()
    yield
    probe_limit_support.cache_clear()


# -- the launcher --------------------------------------------------------


def test_the_launcher_execs_the_command_it_is_given() -> None:
    completed = subprocess.run(
        [*LAUNCHER, "--", sys.executable, "-c", "print('hello from the child')"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "hello from the child"


def test_the_launcher_passes_through_exit_codes() -> None:
    completed = subprocess.run(
        [*LAUNCHER, "--", sys.executable, "-c", "raise SystemExit(7)"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 7


def test_a_cpu_limit_stops_a_runaway_process() -> None:
    completed = subprocess.run(
        [
            *LAUNCHER,
            "--cpu-seconds",
            "1",
            "--",
            sys.executable,
            "-c",
            "while True:\n    pass",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    # Killed by a signal (negative return code) rather than exiting cleanly.
    assert completed.returncode != 0


def test_running_the_launcher_with_no_command_is_an_error() -> None:
    assert _launcher.main([]) == _launcher.EXIT_LAUNCH_FAILED


def test_a_command_that_cannot_be_executed_reports_failure(capsys) -> None:
    exit_code = _launcher.main(["--", "evalforge-definitely-not-a-real-binary"])

    assert exit_code == _launcher.EXIT_LAUNCH_FAILED
    assert "cannot execute" in capsys.readouterr().err


def test_apply_limits_reports_what_the_platform_accepted() -> None:
    """Run in a subprocess: applying limits in-process would lower the test runner's."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json;from evalforge.env._launcher import apply_limits;"
            "print(json.dumps(list(apply_limits(cpu_seconds=3600, file_size_mb=1024))))",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    applied = json.loads(completed.stdout)
    assert "cpu" in applied
    assert "file_size" in applied


def test_requesting_no_limits_applies_none() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json;from evalforge.env._launcher import apply_limits;"
            "print(json.dumps(list(apply_limits())))",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert json.loads(completed.stdout) == []


# -- the probe -----------------------------------------------------------


def test_the_probe_finds_cpu_and_file_size_limits_on_this_platform() -> None:
    support = probe_limit_support()

    assert support.cpu is True
    assert support.file_size is True


def test_the_probe_is_cached() -> None:
    first = probe_limit_support()
    second = probe_limit_support()

    assert first is second


def test_a_probe_that_cannot_run_reports_nothing_enforced(monkeypatch) -> None:
    def explode(*_args, **_kwargs):
        raise OSError("no subprocesses today")

    monkeypatch.setattr(limits.subprocess, "run", explode)

    assert probe_limit_support() == LimitSupport(cpu=False, memory=False, file_size=False)


def test_a_probe_that_fails_reports_nothing_enforced(monkeypatch) -> None:
    monkeypatch.setattr(
        limits.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="x"),
    )

    assert probe_limit_support().all_supported is False


def test_a_probe_returning_junk_reports_nothing_enforced(monkeypatch) -> None:
    monkeypatch.setattr(
        limits.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not json", stderr=""
        ),
    )

    assert probe_limit_support().unsupported() == ("cpu", "memory", "file_size")


def test_full_support_is_described_positively() -> None:
    support = LimitSupport(cpu=True, memory=True, file_size=True)

    assert support.all_supported is True
    assert support.unsupported() == ()
    assert support.describe() == "cpu, memory and file-size limits enforced"


def test_partial_support_names_what_is_missing() -> None:
    support = LimitSupport(cpu=True, memory=False, file_size=True)

    assert support.all_supported is False
    assert support.unsupported() == ("memory",)
    assert "memory" in support.describe()
