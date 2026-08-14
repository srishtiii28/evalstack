"""Schema invariants: content hashing, validation, and result aggregation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evalforge.hashing import canonical_json, content_hash, short, text_hash
from evalforge.schema.case import CaseMetadata, EvalCase, FileSpec
from evalforge.schema.dataset import Dataset, DatasetManifest, DatasetRef
from evalforge.schema.result import CaseResult, EvaluatorResult, RunResult
from evalforge.schema.trajectory import ModelCall, TaskStarted, ToolCall, Trajectory


def make_case(case_id: str = "case-1", **overrides: object) -> EvalCase:
    defaults: dict[str, object] = {
        "case_id": case_id,
        "prompt": "Fix the bug.",
        "files": (FileSpec(path="pkg/mod.py", contents="x = 1\n"),),
        "test_command": ("python", "-m", "pytest"),
    }
    return EvalCase.model_validate(defaults | overrides)


def make_case_result(case_id: str, *, passed: bool, status: str = "completed") -> CaseResult:
    return CaseResult(
        case_id=case_id,
        attempt=0,
        status=status,  # type: ignore[arg-type]
        passed=passed,
        evaluators=(
            EvaluatorResult(name="tests", score=1.0 if passed else 0.0, passed=passed),
            EvaluatorResult(name="trajectory", score=0.5, passed=True),
        ),
        duration_s=1.0,
        cost_usd=0.01,
        input_tokens=100,
        output_tokens=20,
    )


# -- hashing -------------------------------------------------------------


def test_canonical_json_is_key_order_independent() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_content_hash_is_stable_across_key_order() -> None:
    assert content_hash({"b": [1, 2], "a": "x"}) == content_hash({"a": "x", "b": [1, 2]})


def test_content_hash_changes_with_content() -> None:
    assert content_hash({"a": 1}) != content_hash({"a": 2})


def test_text_hash_is_prefixed_and_stable() -> None:
    assert text_hash("hello").startswith("sha256:")
    assert text_hash("hello") == text_hash("hello")
    assert text_hash("hello") != text_hash("world")


def test_short_strips_the_prefix() -> None:
    assert short(text_hash("hello"), 8) == text_hash("hello").removeprefix("sha256:")[:8]


def test_nan_is_refused_rather_than_silently_serialised() -> None:
    with pytest.raises(ValueError, match="Out of range float"):
        canonical_json({"value": float("nan")})


# -- cases ---------------------------------------------------------------


def test_case_hash_is_independent_of_construction() -> None:
    assert make_case().content_hash == make_case().content_hash


def test_absolute_file_paths_are_rejected() -> None:
    with pytest.raises(ValidationError, match="must be relative"):
        FileSpec(path="/etc/passwd", contents="")


def test_traversing_file_paths_are_rejected() -> None:
    with pytest.raises(ValidationError, match="must not traverse upwards"):
        FileSpec(path="../outside.py", contents="")


def test_unnormalised_file_paths_are_rejected() -> None:
    with pytest.raises(ValidationError, match="must be normalised"):
        FileSpec(path="pkg//mod.py", contents="")


def test_duplicate_file_paths_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate file paths"):
        make_case(
            files=(
                FileSpec(path="a.py", contents="1"),
                FileSpec(path="a.py", contents="2"),
            )
        )


def test_metadata_target_files_are_validated() -> None:
    with pytest.raises(ValidationError, match="must not traverse upwards"):
        CaseMetadata(target_files=("../escape.py",))


def test_case_file_map_exposes_the_workspace_seed() -> None:
    assert make_case().file_map() == {"pkg/mod.py": "x = 1\n"}


def test_cases_are_immutable() -> None:
    case = make_case()
    with pytest.raises(ValidationError):
        case.case_id = "changed"  # type: ignore[misc]


# -- datasets ------------------------------------------------------------


def test_dataset_ref_round_trips() -> None:
    assert str(DatasetRef.parse("synth@v3")) == "synth@v3"


@pytest.mark.parametrize(
    ("bad_ref", "message"),
    [
        ("synth", "must be 'name@version'"),
        ("Synth@v1", "invalid dataset name"),
        ("synth@1", "must look like 'v1'"),
    ],
)
def test_bad_dataset_refs_are_rejected(bad_ref: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DatasetRef.parse(bad_ref)


def test_duplicate_case_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate case ids"):
        Dataset(name="synth", version="v1", cases=(make_case("a"), make_case("a")))


def test_dataset_subset_preserves_order_and_hashes_differently() -> None:
    dataset = Dataset(
        name="synth", version="v1", cases=(make_case("a"), make_case("b"), make_case("c"))
    )

    subset = dataset.subset(["c", "a"])

    assert [case.case_id for case in subset.cases] == ["a", "c"]
    assert subset.content_hash != dataset.content_hash


def test_dataset_subset_rejects_unknown_ids() -> None:
    dataset = Dataset(name="synth", version="v1", cases=(make_case("a"),))

    with pytest.raises(KeyError, match="unknown case ids"):
        dataset.subset(["nope"])


def test_case_lookup_by_id() -> None:
    dataset = Dataset(name="synth", version="v1", cases=(make_case("a"), make_case("b")))

    assert dataset.case_by_id("b").case_id == "b"
    with pytest.raises(KeyError, match="no case 'zz'"):
        dataset.case_by_id("zz")


def test_manifest_describes_its_dataset() -> None:
    dataset = Dataset(name="synth", version="v1", cases=(make_case("a"),))

    manifest = DatasetManifest.for_dataset(dataset, generator="test", seed=3)

    assert manifest.case_count == 1
    assert manifest.content_hash == dataset.content_hash
    assert manifest.seed == 3


# -- trajectories --------------------------------------------------------


def test_trajectory_jsonl_round_trips() -> None:
    trajectory = Trajectory(
        run_id="run-1",
        case_id="case-1",
        attempt=0,
        events=(
            TaskStarted(seq=0, t_ms=0.0, case_id="case-1", prompt_hash="sha256:abc"),
            ToolCall(seq=1, t_ms=1.0, call_id="call-1", tool="read_file", args={"path": "a.py"}),
            ModelCall(
                seq=2,
                t_ms=2.0,
                model="claude-haiku-4-5",
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.001,
                latency_ms=120.0,
            ),
        ),
    )

    restored = Trajectory.from_jsonl(
        run_id="run-1", case_id="case-1", attempt=0, jsonl=trajectory.to_jsonl()
    )

    assert restored == trajectory
    assert restored.of_type(ToolCall)[0].args == {"path": "a.py"}


def test_trajectory_totals_come_from_model_calls() -> None:
    trajectory = Trajectory(
        run_id="r",
        case_id="c",
        attempt=0,
        events=(
            ModelCall(
                seq=0,
                t_ms=0.0,
                model="m",
                input_tokens=10,
                output_tokens=2,
                cost_usd=0.5,
                latency_ms=1.0,
            ),
            ModelCall(
                seq=1,
                t_ms=5.0,
                model="m",
                input_tokens=7,
                output_tokens=3,
                cost_usd=0.25,
                latency_ms=1.0,
            ),
        ),
    )

    assert trajectory.total_cost_usd == 0.75
    assert trajectory.total_input_tokens == 17
    assert trajectory.total_output_tokens == 5
    assert trajectory.duration_ms == 5.0
    assert trajectory.submitted is False


def test_empty_trajectory_has_zero_duration() -> None:
    assert Trajectory(run_id="r", case_id="c", attempt=0).duration_ms == 0.0


# -- run results ---------------------------------------------------------


def test_success_rate_ignores_attempts_that_never_completed() -> None:
    run = RunResult(
        run_id="run-1",
        agent_ref="scripted:baseline",
        agent_hash="sha256:a",
        dataset_name="synth",
        dataset_version="v1",
        dataset_hash="sha256:d",
        suite_name="default",
        suite_hash="sha256:s",
        case_results=(
            make_case_result("a", passed=True),
            make_case_result("b", passed=False),
            make_case_result("c", passed=False, status="infra_error"),
        ),
    )

    # Two completed attempts, one passed: infrastructure noise must not count
    # against the agent.
    assert run.success_rate == 0.5
    assert run.status_counts() == {"completed": 2, "infra_error": 1}
    assert run.dataset_ref == "synth@v1"


def test_tallies_group_attempts_by_case() -> None:
    run = RunResult(
        run_id="run-1",
        agent_ref="a",
        agent_hash="h",
        dataset_name="synth",
        dataset_version="v1",
        dataset_hash="h",
        suite_name="default",
        suite_hash="h",
        samples_per_case=2,
        case_results=(
            make_case_result("a", passed=True),
            make_case_result("a", passed=False),
            make_case_result("b", passed=True),
        ),
    )

    tallies = run.tallies()
    assert tallies["a"].passed == 1
    assert tallies["a"].total == 2
    assert tallies["a"].rate == 0.5
    assert tallies["b"].rate == 1.0


def test_evaluator_scores_average_across_completed_attempts() -> None:
    run = RunResult(
        run_id="run-1",
        agent_ref="a",
        agent_hash="h",
        dataset_name="synth",
        dataset_version="v1",
        dataset_hash="h",
        suite_name="default",
        suite_hash="h",
        case_results=(
            make_case_result("a", passed=True),
            make_case_result("b", passed=False),
        ),
    )

    assert run.evaluator_scores() == {"tests": 0.5, "trajectory": 0.5}
    assert run.total_cost_usd == pytest.approx(0.02)
    assert run.total_input_tokens == 200


def test_case_result_evaluator_lookup() -> None:
    result = make_case_result("a", passed=True)

    assert result.evaluator("tests") is not None
    assert result.evaluator("missing") is None


def test_scores_outside_the_unit_interval_are_rejected() -> None:
    with pytest.raises(ValidationError):
        EvaluatorResult(name="tests", score=1.5, passed=True)
