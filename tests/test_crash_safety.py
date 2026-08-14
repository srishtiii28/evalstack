"""A run killed mid-flight must leave a readable store, not a corrupt one.

This is the reason results are written as each attempt finishes rather than in
one batch at the end, and the reason the connection runs with WAL journalling
and full synchronous writes. Asserting it with a real SIGKILL rather than a
simulated interruption is the only way to know it holds.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from evalforge.datasets.builder import build_synthetic_dataset
from evalforge.datasets.io import write_dataset
from evalforge.store.db import Store

CASE_COUNT = 24
KILL_AFTER_S = 2.5
SHUTDOWN_GRACE_S = 10.0


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()


@pytest.mark.slow
def test_a_killed_run_leaves_the_finished_attempts_readable(tmp_path: Path) -> None:
    dataset = build_synthetic_dataset(count=CASE_COUNT, seed=7)
    write_dataset(dataset, tmp_path / "datasets" / "synth")
    database = tmp_path / "runs.db"

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "evalforge.cli.main",
            "run",
            "--dataset",
            "synth@v1",
            "--agent",
            "scripted:baseline",
            "--datasets-root",
            str(tmp_path / "datasets"),
            "--db",
            str(database),
            "--trajectories",
            str(tmp_path / "traces"),
            "--concurrency",
            "2",
        ],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    try:
        time.sleep(KILL_AFTER_S)
        assert process.poll() is None, "run finished before it could be interrupted"
        _kill_group(process)
        process.wait(timeout=SHUTDOWN_GRACE_S)
    finally:
        if process.poll() is None:  # pragma: no cover - only on an unexpected hang
            _kill_group(process)
            process.wait(timeout=SHUTDOWN_GRACE_S)

    assert database.is_file(), "the run never got as far as creating a database"

    with Store.open(database) as store:
        summaries = store.list_runs()
        assert len(summaries) == 1, "the run header should have been written before any work"

        run = store.load_run(summaries[0].run_id)

    # Partial, but coherent: fewer attempts than cases, every one fully formed.
    assert 0 < len(run.case_results) < CASE_COUNT
    for result in run.case_results:
        assert result.status == "completed"
        assert result.evaluators, "an attempt was recorded without its evaluator scores"
        assert result.evaluator("tests") is not None
