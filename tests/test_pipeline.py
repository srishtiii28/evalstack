"""End-to-end runs: reproducibility, sampling, persistence and traces."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalforge.datasets.builder import build_synthetic_dataset
from evalforge.evaluators.registry import default_suite
from evalforge.pipeline import RunRequest, execute_run, new_run_id
from evalforge.schema.dataset import Dataset
from evalforge.store.db import Store

SMALL_DATASET_SIZE = 4


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return build_synthetic_dataset(count=SMALL_DATASET_SIZE, seed=7)


def verdicts(run) -> list[tuple[str, int, bool]]:
    return [(r.case_id, r.attempt, r.passed) for r in run.case_results]


async def test_a_run_records_what_it_evaluated(dataset: Dataset, tmp_path: Path) -> None:
    request = RunRequest(
        dataset=dataset, agent_ref="scripted:baseline", trajectory_dir=tmp_path / "traces"
    )

    run = await execute_run(request)

    assert run.dataset_hash == dataset.content_hash
    assert run.suite_hash == default_suite().content_hash
    assert run.agent_hash.startswith("sha256:")
    assert run.dataset_ref == f"{dataset.name}@{dataset.version}"
    assert len(run.case_results) == SMALL_DATASET_SIZE
    assert all(result.status == "completed" for result in run.case_results)


async def test_identical_runs_produce_identical_verdicts(dataset: Dataset) -> None:
    request = RunRequest(dataset=dataset, agent_ref="scripted:baseline")

    first = await execute_run(request)
    second = await execute_run(request)

    assert verdicts(first) == verdicts(second)
    assert first.success_rate == second.success_rate
    # Same inputs, same provenance — only the identity differs.
    assert first.agent_hash == second.agent_hash
    assert first.run_id != second.run_id


async def test_the_oracle_solves_everything_and_the_idle_agent_nothing(
    dataset: Dataset,
) -> None:
    oracle = await execute_run(RunRequest(dataset=dataset, agent_ref="scripted:oracle"))
    idle = await execute_run(RunRequest(dataset=dataset, agent_ref="scripted:idle"))

    assert oracle.success_rate == 1.0
    assert idle.success_rate == 0.0


async def test_the_baseline_lands_between_the_bounds(dataset: Dataset) -> None:
    run = await execute_run(RunRequest(dataset=dataset, agent_ref="scripted:baseline"))

    # A dataset an agent passes 100% or 0% of measures nothing.
    assert 0.0 < run.success_rate < 1.0


async def test_the_regressed_agent_scores_worse_on_every_dimension(dataset: Dataset) -> None:
    baseline = await execute_run(RunRequest(dataset=dataset, agent_ref="scripted:baseline"))
    regressed = await execute_run(RunRequest(dataset=dataset, agent_ref="scripted:regressed"))

    baseline_scores = baseline.evaluator_scores()
    regressed_scores = regressed.evaluator_scores()

    assert regressed_scores["patch_locality"] < baseline_scores["patch_locality"]
    assert regressed_scores["trajectory"] < baseline_scores["trajectory"]


async def test_sampling_runs_each_case_more_than_once(dataset: Dataset) -> None:
    run = await execute_run(
        RunRequest(dataset=dataset, agent_ref="scripted:baseline", samples_per_case=2)
    )

    assert len(run.case_results) == SMALL_DATASET_SIZE * 2
    assert all(tally.total == 2 for tally in run.tallies().values())


async def test_a_case_filter_restricts_the_run(dataset: Dataset) -> None:
    wanted = dataset.cases[0].case_id

    run = await execute_run(
        RunRequest(dataset=dataset, agent_ref="scripted:idle", case_ids=(wanted,))
    )

    assert [result.case_id for result in run.case_results] == [wanted]


async def test_an_unknown_case_filter_is_rejected(dataset: Dataset) -> None:
    request = RunRequest(dataset=dataset, agent_ref="scripted:idle", case_ids=("nope",))

    with pytest.raises(KeyError, match="unknown case ids"):
        await execute_run(request)


async def test_trajectories_are_written_and_reloadable(dataset: Dataset, tmp_path: Path) -> None:
    traces = tmp_path / "traces"

    run = await execute_run(
        RunRequest(
            dataset=dataset,
            agent_ref="scripted:baseline",
            case_ids=(dataset.cases[0].case_id,),
            trajectory_dir=traces,
        )
    )

    path = run.case_results[0].trajectory_path
    assert path is not None
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    assert lines
    assert '"kind":"task_started"' in lines[0]


async def test_no_trajectory_directory_means_no_trace_files(dataset: Dataset) -> None:
    run = await execute_run(
        RunRequest(
            dataset=dataset,
            agent_ref="scripted:idle",
            case_ids=(dataset.cases[0].case_id,),
            trajectory_dir=None,
        )
    )

    assert run.case_results[0].trajectory_path is None


async def test_results_reach_the_store_as_they_finish(dataset: Dataset, tmp_path: Path) -> None:
    with Store.open(tmp_path / "runs.db") as store:
        run = await execute_run(
            RunRequest(dataset=dataset, agent_ref="scripted:baseline"), store=store
        )
        loaded = store.load_run(run.run_id)

    assert verdicts(loaded) == verdicts(run)
    assert loaded.dataset_hash == run.dataset_hash
    assert loaded.suite_hash == run.suite_hash


async def test_results_are_ordered_by_case_and_attempt(dataset: Dataset) -> None:
    run = await execute_run(
        RunRequest(dataset=dataset, agent_ref="scripted:idle", samples_per_case=2)
    )

    keys = [(result.case_id, result.attempt) for result in run.case_results]
    assert keys == sorted(keys)


async def test_a_supplied_run_id_is_honoured(dataset: Dataset, tmp_path: Path) -> None:
    with Store.open(tmp_path / "runs.db") as store:
        run = await execute_run(
            RunRequest(
                dataset=dataset, agent_ref="scripted:idle", case_ids=(dataset.cases[0].case_id,)
            ),
            store=store,
            run_id="run-fixed",
        )

        assert run.run_id == "run-fixed"
        assert store.run_exists("run-fixed") is True


async def test_an_unknown_suite_is_rejected_before_anything_runs(dataset: Dataset) -> None:
    request = RunRequest(dataset=dataset, agent_ref="scripted:idle", suite_name="nope")

    with pytest.raises(KeyError, match="unknown suite"):
        await execute_run(request)


def test_run_ids_are_sortable_and_unique() -> None:
    first = new_run_id()
    second = new_run_id()

    assert first.startswith("run-")
    assert first != second


def test_selected_dataset_defaults_to_the_whole_dataset(dataset: Dataset) -> None:
    request = RunRequest(dataset=dataset, agent_ref="scripted:idle")

    assert request.selected_dataset() is dataset
