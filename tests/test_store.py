"""The results store: migrations, round-tripping, and crash-safe partial runs."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from evalforge.schema.result import CaseResult, EvaluatorResult, RunResult
from evalforge.store.db import Store, StoreError, iter_migration_versions, iter_sql_statements


def make_run(run_id: str = "run-1", **overrides: object) -> RunResult:
    defaults: dict[str, object] = {
        "run_id": run_id,
        "agent_ref": "scripted:baseline",
        "agent_hash": "sha256:agent",
        "dataset_name": "synth",
        "dataset_version": "v1",
        "dataset_hash": "sha256:dataset",
        "suite_name": "default",
        "suite_hash": "sha256:suite",
        "samples_per_case": 1,
        "concurrency": 4,
        "notes": "a note",
    }
    return RunResult.model_validate(defaults | overrides)


def make_case_result(case_id: str, *, passed: bool = True, attempt: int = 0) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        attempt=attempt,
        status="completed",
        passed=passed,
        evaluators=(
            EvaluatorResult(
                name="tests",
                version="1",
                score=1.0 if passed else 0.0,
                passed=passed,
                detail={"exit_code": 0 if passed else 1, "counts": {"passed": 3}},
            ),
            EvaluatorResult(name="trajectory", version="2", score=0.75, passed=True),
        ),
        duration_s=1.25,
        cost_usd=0.002,
        input_tokens=120,
        output_tokens=45,
        trajectory_path="/tmp/trace.jsonl",
    )


@pytest.fixture
def store(tmp_path: Path):
    with Store.open(tmp_path / "runs.db") as opened:
        yield opened


# -- migrations ----------------------------------------------------------


def test_opening_a_fresh_database_applies_every_migration(tmp_path: Path) -> None:
    with Store.open(tmp_path / "runs.db") as store:
        assert store.schema_version == max(iter_migration_versions())


def test_reopening_applies_nothing_new(tmp_path: Path) -> None:
    path = tmp_path / "runs.db"
    with Store.open(path):
        pass

    with Store.open(path) as store:
        assert store.migrate() == ()


def test_a_database_from_a_newer_build_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "runs.db"
    with Store.open(path):
        pass

    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO schema_migrations (version, applied_at) VALUES (999, ?)",
        (datetime.now(UTC).isoformat(),),
    )
    connection.commit()
    connection.close()

    with pytest.raises(StoreError, match="newer version of EvalForge"):
        Store.open(path)


def test_the_parent_directory_is_created_on_demand(tmp_path: Path) -> None:
    with Store.open(tmp_path / "nested" / "deeper" / "runs.db") as store:
        assert store.schema_version > 0


def test_sql_splitting_respects_semicolons_inside_strings() -> None:
    script = """
    -- a comment
    CREATE TABLE t (a TEXT);
    INSERT INTO t (a) VALUES ('one; two');
    """

    statements = list(iter_sql_statements(script))

    assert len(statements) == 2
    assert statements[1].endswith("('one; two');")


def test_an_incomplete_trailing_statement_is_an_error() -> None:
    with pytest.raises(StoreError, match="incomplete statement"):
        list(iter_sql_statements("CREATE TABLE t (a TEXT)"))


def test_comment_only_scripts_yield_nothing() -> None:
    assert list(iter_sql_statements("-- nothing to see\n-- really\n")) == []


# -- round-tripping ------------------------------------------------------


def test_a_saved_run_reloads_with_every_field_intact(store: Store) -> None:
    run = make_run(case_results=(make_case_result("a"), make_case_result("b", passed=False)))

    store.save_run(run)
    loaded = store.load_run(run.run_id)

    assert loaded.agent_ref == run.agent_ref
    assert loaded.agent_hash == run.agent_hash
    assert loaded.dataset_hash == run.dataset_hash
    assert loaded.suite_hash == run.suite_hash
    assert loaded.samples_per_case == run.samples_per_case
    assert loaded.concurrency == run.concurrency
    assert loaded.notes == run.notes
    assert loaded.case_results == run.case_results


def test_evaluator_detail_survives_the_round_trip(store: Store) -> None:
    store.save_run(make_run(case_results=(make_case_result("a"),)))

    loaded = store.load_run("run-1")
    tests = loaded.case_results[0].evaluator("tests")

    assert tests is not None
    assert tests.detail == {"exit_code": 0, "counts": {"passed": 3}}
    assert tests.version == "1"


def test_evaluator_order_is_preserved(store: Store) -> None:
    store.save_run(make_run(case_results=(make_case_result("a"),)))

    names = [e.name for e in store.load_run("run-1").case_results[0].evaluators]

    assert names == ["tests", "trajectory"]


def test_created_at_survives_as_an_aware_timestamp(store: Store) -> None:
    moment = datetime(2026, 3, 1, 12, 30, tzinfo=UTC)
    store.save_run(make_run(created_at=moment))

    assert store.load_run("run-1").created_at == moment


def test_loading_an_unknown_run_is_a_key_error(store: Store) -> None:
    with pytest.raises(KeyError, match="no run 'nope'"):
        store.load_run("nope")


def test_run_existence_can_be_checked(store: Store) -> None:
    store.start_run(make_run())

    assert store.run_exists("run-1") is True
    assert store.run_exists("run-2") is False


# -- incremental writing -------------------------------------------------


def test_a_run_interrupted_halfway_keeps_the_attempts_that_finished(store: Store) -> None:
    store.start_run(make_run())
    store.record_case_result("run-1", make_case_result("a"))
    store.record_case_result("run-1", make_case_result("b", passed=False))
    # Simulating a kill here: nothing else is written.

    loaded = store.load_run("run-1")

    assert [result.case_id for result in loaded.case_results] == ["a", "b"]
    assert loaded.success_rate == 0.5


def test_a_header_with_no_results_is_still_readable(store: Store) -> None:
    store.start_run(make_run())

    loaded = store.load_run("run-1")

    assert loaded.case_results == ()
    assert loaded.success_rate == 0.0


def test_recording_the_same_attempt_twice_replaces_it(store: Store) -> None:
    store.start_run(make_run())
    store.record_case_result("run-1", make_case_result("a", passed=False))
    store.record_case_result("run-1", make_case_result("a", passed=True))

    loaded = store.load_run("run-1")

    assert len(loaded.case_results) == 1
    assert loaded.case_results[0].passed is True
    # The stale evaluator rows must go with it, not accumulate.
    assert len(loaded.case_results[0].evaluators) == 2


def test_repeated_attempts_of_one_case_are_kept_separately(store: Store) -> None:
    store.start_run(make_run(samples_per_case=2))
    store.record_case_result("run-1", make_case_result("a", attempt=0, passed=True))
    store.record_case_result("run-1", make_case_result("a", attempt=1, passed=False))

    tallies = store.load_run("run-1").tallies()

    assert tallies["a"].passed == 1
    assert tallies["a"].total == 2


# -- deletion and listing ------------------------------------------------


def test_deleting_a_run_removes_its_results(tmp_path: Path) -> None:
    path = tmp_path / "runs.db"
    with Store.open(path) as store:
        store.save_run(make_run(case_results=(make_case_result("a"),)))

        assert store.delete_run("run-1") is True
        assert store.run_exists("run-1") is False

    connection = sqlite3.connect(path)
    remaining = connection.execute("SELECT COUNT(*) FROM evaluator_results").fetchone()[0]
    connection.close()

    assert remaining == 0


def test_deleting_an_unknown_run_reports_nothing_removed(store: Store) -> None:
    assert store.delete_run("nope") is False


def test_runs_are_listed_newest_first_with_aggregates(store: Store) -> None:
    older = datetime(2026, 1, 1, tzinfo=UTC)
    store.save_run(
        make_run(
            "run-old",
            created_at=older,
            case_results=(make_case_result("a"), make_case_result("b", passed=False)),
        )
    )
    store.save_run(
        make_run(
            "run-new",
            created_at=older + timedelta(days=1),
            case_results=(make_case_result("a"),),
        )
    )

    summaries = store.list_runs()

    assert [summary.run_id for summary in summaries] == ["run-new", "run-old"]
    assert summaries[1].completed == 2
    assert summaries[1].passed == 1
    assert summaries[1].success_rate == 0.5
    assert summaries[1].dataset_ref == "synth@v1"
    assert summaries[0].total_cost_usd == pytest.approx(0.002)


def test_listing_respects_its_limit(store: Store) -> None:
    for index in range(5):
        created = datetime(2026, 1, index + 1, tzinfo=UTC)
        store.save_run(make_run(f"run-{index}", created_at=created))

    assert len(store.list_runs(limit=3)) == 3


def test_a_run_with_no_attempts_has_a_zero_success_rate_in_listings(store: Store) -> None:
    store.start_run(make_run())

    summary = store.list_runs()[0]

    assert summary.attempts == 0
    assert summary.success_rate == 0.0


def test_listing_an_empty_store_returns_nothing(store: Store) -> None:
    assert store.list_runs() == ()
