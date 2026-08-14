"""The command line, exercised the way a user drives it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalforge import __version__
from evalforge.cli.main import EXIT_CHECK_FAILED, EXIT_USAGE_ERROR, app
from evalforge.datasets.io import CASES_FILENAME

runner = CliRunner()

SMALL_DATASET = ("--cases", "4", "--seed", "7")


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch working directory, so default relative paths land in tmp_path."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def build_dataset(*extra: str) -> None:
    result = runner.invoke(app, ["dataset", "build", *SMALL_DATASET, *extra])
    assert result.exit_code == 0, result.output


def run_baseline(*extra: str):
    return runner.invoke(app, ["run", "--agent", "scripted:baseline", *extra])


# -- basics --------------------------------------------------------------


def test_version_is_reported() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_doctor_reports_sandbox_capabilities_and_registries() -> None:
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "sandbox limits" in result.output
    assert "scripted:baseline" in result.output
    assert "default" in result.output


# -- dataset -------------------------------------------------------------


def test_dataset_build_writes_a_verifiable_dataset(project: Path) -> None:
    build_dataset()

    assert (project / "datasets" / "synth" / CASES_FILENAME).is_file()

    result = runner.invoke(app, ["dataset", "verify", "datasets/synth"])
    assert result.exit_code == 0
    assert "ok synth@v1" in result.output


def test_dataset_build_refuses_to_clobber_without_force(project: Path) -> None:
    build_dataset()

    result = runner.invoke(app, ["dataset", "build", *SMALL_DATASET])

    assert result.exit_code == EXIT_USAGE_ERROR
    assert "already holds a dataset" in result.output


def test_dataset_build_overwrites_with_force(project: Path) -> None:
    build_dataset()

    result = runner.invoke(app, ["dataset", "build", *SMALL_DATASET, "--force"])

    assert result.exit_code == 0


def test_dataset_build_rejects_a_zero_case_count(project: Path) -> None:
    result = runner.invoke(app, ["dataset", "build", "--cases", "0"])

    assert result.exit_code == EXIT_USAGE_ERROR
    assert "count must be at least 1" in result.output


def test_dataset_verify_detects_drift(project: Path) -> None:
    build_dataset()
    cases = project / "datasets" / "synth" / CASES_FILENAME
    cases.write_text(cases.read_text(encoding="utf-8").replace("Fix it", "Fix it now"))

    result = runner.invoke(app, ["dataset", "verify", "datasets/synth"])

    assert result.exit_code == EXIT_CHECK_FAILED
    assert "drift" in result.output


def test_dataset_verify_on_a_missing_directory_is_a_clear_error(project: Path) -> None:
    result = runner.invoke(app, ["dataset", "verify", "datasets/nothing"])

    assert result.exit_code == EXIT_USAGE_ERROR
    assert "no dataset manifest" in result.output


# -- run -----------------------------------------------------------------


def test_run_evaluates_and_reports(project: Path) -> None:
    build_dataset()

    result = run_baseline()

    assert result.exit_code == 0, result.output
    assert "success rate" in result.output
    assert "scripted:baseline" in result.output


def test_run_emits_machine_readable_json(project: Path) -> None:
    build_dataset()

    result = run_baseline("--json")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["agent_ref"] == "scripted:baseline"
    assert len(payload["case_results"]) == 4


def test_run_accepts_a_dataset_directory_as_well_as_a_ref(project: Path) -> None:
    build_dataset()

    result = run_baseline("--dataset", "datasets/synth", "--json")

    assert result.exit_code == 0, result.output


def test_run_restricted_to_one_case(project: Path) -> None:
    build_dataset()
    case_id = json.loads(
        (project / "datasets" / "synth" / CASES_FILENAME).read_text().splitlines()[0]
    )["case_id"]

    result = run_baseline("--case", case_id, "--json")

    assert result.exit_code == 0, result.output
    assert len(json.loads(result.output)["case_results"]) == 1


def test_run_against_a_missing_dataset_is_a_usage_error(project: Path) -> None:
    result = run_baseline("--dataset", "ghost@v1")

    assert result.exit_code == EXIT_USAGE_ERROR
    assert "no dataset directory" in result.output


def test_run_with_an_unknown_agent_is_a_usage_error(project: Path) -> None:
    build_dataset()

    result = runner.invoke(app, ["run", "--agent", "scripted:imaginary"])

    assert result.exit_code == EXIT_USAGE_ERROR
    assert "unknown scripted policy" in result.output


def test_run_with_an_unknown_suite_is_a_usage_error(project: Path) -> None:
    build_dataset()

    result = run_baseline("--suite", "imaginary")

    assert result.exit_code == EXIT_USAGE_ERROR
    assert "unknown suite" in result.output


def test_run_rejects_a_version_mismatch_between_ref_and_manifest(project: Path) -> None:
    build_dataset()

    result = run_baseline("--dataset", "synth@v9")

    assert result.exit_code == EXIT_USAGE_ERROR
    assert "not synth@v9" in result.output


# -- inspection ----------------------------------------------------------


def test_runs_are_listed_after_a_run(project: Path) -> None:
    build_dataset()
    expected_run_id = json.loads(run_baseline("--json").output)["run_id"]

    result = runner.invoke(app, ["runs", "--json"])

    assert result.exit_code == 0
    listing = json.loads(result.output)
    assert [entry["run_id"] for entry in listing] == [expected_run_id]
    assert listing[0]["agent_ref"] == "scripted:baseline"
    assert listing[0]["completed"] == 4


def test_runs_renders_a_table_by_default(project: Path) -> None:
    build_dataset()
    run_baseline()

    result = runner.invoke(app, ["runs"])

    assert result.exit_code == 0
    assert "runs" in result.output


def test_listing_an_empty_store_says_so(project: Path) -> None:
    result = runner.invoke(app, ["runs"])

    assert result.exit_code == 0
    assert "no runs recorded yet" in result.output


def test_show_reloads_a_recorded_run(project: Path) -> None:
    build_dataset()
    run_id = json.loads(run_baseline("--json").output)["run_id"]

    result = runner.invoke(app, ["show", run_id, "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output)["run_id"] == run_id


def test_show_renders_tables_by_default(project: Path) -> None:
    build_dataset()
    run_id = json.loads(run_baseline("--json").output)["run_id"]

    result = runner.invoke(app, ["show", run_id])

    assert result.exit_code == 0
    assert "success rate" in result.output


def test_show_on_an_unknown_run_is_a_usage_error(project: Path) -> None:
    result = runner.invoke(app, ["show", "run-nope"])

    assert result.exit_code == EXIT_USAGE_ERROR
    assert "no run" in result.output


def test_trajectory_prints_recorded_events(project: Path) -> None:
    build_dataset()
    payload = json.loads(run_baseline("--json").output)
    case_id = payload["case_results"][0]["case_id"]

    result = runner.invoke(app, ["trajectory", payload["run_id"], case_id])

    assert result.exit_code == 0, result.output
    assert "task_started" in result.output


def test_trajectory_for_an_unknown_case_is_a_usage_error(project: Path) -> None:
    build_dataset()
    run_id = json.loads(run_baseline("--json").output)["run_id"]

    result = runner.invoke(app, ["trajectory", run_id, "no-such-case"])

    assert result.exit_code == EXIT_USAGE_ERROR
    assert "no attempt 0 of case" in result.output
