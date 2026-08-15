"""Evaluators, and the suite machinery that combines them into a verdict."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from evalforge.env.workspace import workspace_for
from evalforge.evaluators.base import (
    EvaluationContext,
    Evaluator,
    EvaluatorSuite,
    evaluate_with_timeout,
)
from evalforge.evaluators.outcome import SuiteOutcomeEvaluator, parse_pytest_counts
from evalforge.evaluators.patch import PatchLocalityEvaluator, PatchWeights
from evalforge.evaluators.registry import (
    default_suite,
    outcome_only_suite,
    resolve_suite,
    suite_names,
)
from evalforge.evaluators.trajectory import (
    TrajectoryEvaluator,
    TrajectorySignals,
    TrajectoryWeights,
    extract_signals,
)
from evalforge.schema.case import CaseMetadata, EvalCase, FileSpec
from evalforge.schema.result import EvaluatorResult
from evalforge.schema.trajectory import Trajectory
from evalforge.trace import FakeClock, TrajectoryRecorder

PASSING_COMMAND = ("python", "-c", "pass")
FAILING_COMMAND = ("python", "-c", "raise SystemExit(1)")


def make_case(*, test_command: tuple[str, ...] = PASSING_COMMAND, **overrides: object) -> EvalCase:
    defaults: dict[str, object] = {
        "case_id": "case-1",
        "prompt": "Fix the bug.",
        "files": (FileSpec(path="pkg/mod.py", contents="x = 1\n"),),
        "test_command": test_command,
        "metadata": CaseMetadata(bug_kind="off_by_one", target_files=("pkg/mod.py",)),
    }
    return EvalCase.model_validate(defaults | overrides)


def empty_trajectory() -> Trajectory:
    return Trajectory(run_id="run-1", case_id="case-1", attempt=0)


# -- pytest output parsing -----------------------------------------------


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("3 passed in 0.05s", {"passed": 3}),
        ("2 failed, 3 passed in 0.11s", {"failed": 2, "passed": 3}),
        ("1 error in 0.02s", {"error": 1}),
        ("2 passed, 1 skipped in 0.03s", {"passed": 2, "skipped": 1}),
    ],
)
def test_pytest_summary_counts_are_parsed(output: str, expected: dict[str, int]) -> None:
    assert parse_pytest_counts(output) == expected


def test_output_without_a_summary_reports_nothing_rather_than_zero() -> None:
    # A crashed or non-pytest command must not look like "zero tests failed".
    assert parse_pytest_counts("Traceback (most recent call last):") == {}


def test_the_last_summary_line_wins() -> None:
    output = "1 passed in 0.01s\n=== rerun ===\n4 passed in 0.02s\n"

    assert parse_pytest_counts(output) == {"passed": 4}


# -- outcome evaluator ---------------------------------------------------


async def test_outcome_passes_when_the_suite_is_green(tmp_path: Path) -> None:
    case = make_case()
    with workspace_for(case, base_dir=tmp_path) as workspace:
        result = await SuiteOutcomeEvaluator().evaluate(
            EvaluationContext(
                case=case,
                workspace=workspace,
                trajectory=empty_trajectory(),
                diff=workspace.diff(),
            )
        )

    assert result.passed is True
    assert result.score == 1.0
    assert result.detail["exit_code"] == 0


async def test_outcome_fails_when_the_suite_is_red(tmp_path: Path) -> None:
    case = make_case(test_command=FAILING_COMMAND)
    with workspace_for(case, base_dir=tmp_path) as workspace:
        result = await SuiteOutcomeEvaluator().evaluate(
            EvaluationContext(
                case=case,
                workspace=workspace,
                trajectory=empty_trajectory(),
                diff=workspace.diff(),
            )
        )

    assert result.passed is False
    assert result.score == 0.0
    assert result.detail["exit_code"] == 1


async def test_outcome_measures_the_final_state_not_what_the_agent_saw(tmp_path: Path) -> None:
    """An agent that goes green then keeps editing has not earned its pass."""
    check_flag = "import pathlib, sys; sys.exit(pathlib.Path('flag').read_text().strip() != 'ok')"
    case = make_case(test_command=("python", "-c", check_flag))
    with workspace_for(case, base_dir=tmp_path) as workspace:
        workspace.write_file("flag", "ok\n")
        green = await SuiteOutcomeEvaluator().evaluate(
            EvaluationContext(
                case=case,
                workspace=workspace,
                trajectory=empty_trajectory(),
                diff=workspace.diff(),
            )
        )
        workspace.write_file("flag", "broken\n")
        after = await SuiteOutcomeEvaluator().evaluate(
            EvaluationContext(
                case=case,
                workspace=workspace,
                trajectory=empty_trajectory(),
                diff=workspace.diff(),
            )
        )

    assert green.passed is True
    assert after.passed is False


# -- patch locality ------------------------------------------------------


async def evaluate_patch(workspace, case, weights: PatchWeights | None = None) -> EvaluatorResult:
    return await PatchLocalityEvaluator(weights).evaluate(
        EvaluationContext(
            case=case,
            workspace=workspace,
            trajectory=empty_trajectory(),
            diff=workspace.diff(),
        )
    )


async def test_no_edits_scores_zero_and_fails(tmp_path: Path) -> None:
    case = make_case()
    with workspace_for(case, base_dir=tmp_path) as workspace:
        result = await evaluate_patch(workspace, case)

    assert result.score == 0.0
    assert result.passed is False
    assert result.detail["reason"] == "no files were modified"


async def test_editing_only_the_target_scores_full_marks(tmp_path: Path) -> None:
    case = make_case()
    with workspace_for(case, base_dir=tmp_path) as workspace:
        workspace.write_file("pkg/mod.py", "x = 2\n")
        result = await evaluate_patch(workspace, case)

    assert result.score == 1.0
    assert result.passed is True
    assert result.detail["unrelated_files"] == []


async def test_unrelated_edits_are_penalised(tmp_path: Path) -> None:
    case = make_case()
    with workspace_for(case, base_dir=tmp_path) as workspace:
        workspace.write_file("pkg/mod.py", "x = 2\n")
        workspace.write_file("NOTES.md", "thinking out loud\n")
        result = await evaluate_patch(workspace, case, PatchWeights(unrelated_file_penalty=0.25))

    assert result.score == pytest.approx(0.75)
    assert result.passed is False
    assert result.detail["unrelated_files"] == ["NOTES.md"]


async def test_missing_the_target_entirely_is_penalised(tmp_path: Path) -> None:
    case = make_case()
    with workspace_for(case, base_dir=tmp_path) as workspace:
        workspace.write_file("NOTES.md", "did not touch the bug\n")
        result = await evaluate_patch(workspace, case)

    assert result.detail["untouched_targets"] == ["pkg/mod.py"]
    assert result.passed is False
    assert result.score == pytest.approx(0.25)


async def test_patch_score_never_goes_below_zero(tmp_path: Path) -> None:
    case = make_case()
    with workspace_for(case, base_dir=tmp_path) as workspace:
        for index in range(10):
            workspace.write_file(f"junk_{index}.txt", "noise\n")
        result = await evaluate_patch(workspace, case)

    assert result.score == 0.0


# -- trajectory signals --------------------------------------------------


def build_trajectory(script) -> Trajectory:
    recorder = TrajectoryRecorder(
        run_id="run-1", case_id="case-1", clock=FakeClock(auto_advance_ms=1.0)
    )
    script(recorder)
    return recorder.build()


def test_clean_trajectory_has_no_penalties() -> None:
    def script(recorder: TrajectoryRecorder) -> None:
        call = recorder.tool_call(tool="read_file", args={"path": "a.py"})
        recorder.tool_result(call_id=call, tool="read_file", ok=True, output="x")
        recorder.file_edit(
            path="a.py",
            before_hash="sha256:1",
            after_hash="sha256:2",
            lines_added=1,
            lines_removed=1,
        )

    signals = extract_signals(build_trajectory(script))

    assert signals.redundant_reads == 0
    assert signals.blind_edits == 0
    assert signals.failed_tool_calls == 0
    assert TrajectoryEvaluator().score_signals(signals) == 1.0


def test_repeated_reads_of_one_file_are_counted_once_each() -> None:
    def script(recorder: TrajectoryRecorder) -> None:
        for _ in range(3):
            call = recorder.tool_call(tool="read_file", args={"path": "a.py"})
            recorder.tool_result(call_id=call, tool="read_file", ok=True, output="x")

    signals = extract_signals(build_trajectory(script))

    assert signals.files_read == 1
    assert signals.redundant_reads == 2


def test_editing_a_file_that_was_never_read_is_a_blind_edit() -> None:
    def script(recorder: TrajectoryRecorder) -> None:
        recorder.file_edit(
            path="unseen.py",
            before_hash=None,
            after_hash="sha256:2",
            lines_added=3,
            lines_removed=0,
        )

    assert extract_signals(build_trajectory(script)).blind_edits == 1


def test_failed_tool_calls_are_counted() -> None:
    def script(recorder: TrajectoryRecorder) -> None:
        call = recorder.tool_call(tool="read_file", args={"path": "missing.py"})
        recorder.tool_result(call_id=call, tool="read_file", ok=False, error="no such file")

    assert extract_signals(build_trajectory(script)).failed_tool_calls == 1


def test_recovery_requires_an_edit_between_a_red_and_a_green_run() -> None:
    def script(recorder: TrajectoryRecorder) -> None:
        first = recorder.tool_call(tool="run_tests", args={})
        recorder.tool_result(call_id=first, tool="run_tests", ok=False, error="failed")
        recorder.file_edit(
            path="a.py",
            before_hash="sha256:1",
            after_hash="sha256:2",
            lines_added=1,
            lines_removed=1,
        )
        second = recorder.tool_call(tool="run_tests", args={})
        recorder.tool_result(call_id=second, tool="run_tests", ok=True, output="ok")

    assert extract_signals(build_trajectory(script)).recovered_after_failure is True


def test_rerunning_tests_without_editing_is_not_recovery() -> None:
    def script(recorder: TrajectoryRecorder) -> None:
        first = recorder.tool_call(tool="run_tests", args={})
        recorder.tool_result(call_id=first, tool="run_tests", ok=False, error="failed")
        second = recorder.tool_call(tool="run_tests", args={})
        recorder.tool_result(call_id=second, tool="run_tests", ok=True, output="ok")

    assert extract_signals(build_trajectory(script)).recovered_after_failure is False


# -- trajectory scoring --------------------------------------------------


def make_signals(**overrides: object) -> TrajectorySignals:
    defaults: dict[str, object] = {
        "tool_calls": 4,
        "redundant_reads": 0,
        "failed_tool_calls": 0,
        "blind_edits": 0,
        "files_read": 1,
        "files_edited": 1,
        "test_runs": 1,
        "recovered_after_failure": False,
    }
    return TrajectorySignals(**(defaults | overrides))  # type: ignore[arg-type]


def test_each_penalty_is_applied_at_its_configured_weight() -> None:
    evaluator = TrajectoryEvaluator(
        TrajectoryWeights(redundant_read=0.1, failed_tool_call=0.15, blind_edit=0.2)
    )

    assert evaluator.score_signals(make_signals(redundant_reads=2)) == pytest.approx(0.8)
    assert evaluator.score_signals(make_signals(failed_tool_calls=2)) == pytest.approx(0.7)
    assert evaluator.score_signals(make_signals(blind_edits=1)) == pytest.approx(0.8)


def test_tool_calls_beyond_the_budget_are_charged() -> None:
    evaluator = TrajectoryEvaluator(TrajectoryWeights(tool_call_budget=10, excess_tool_call=0.05))

    assert evaluator.score_signals(make_signals(tool_calls=10)) == pytest.approx(1.0)
    assert evaluator.score_signals(make_signals(tool_calls=14)) == pytest.approx(0.8)


def test_recovery_earns_credit_back() -> None:
    evaluator = TrajectoryEvaluator(
        TrajectoryWeights(failed_tool_call=0.15, recovery_credit=0.1)
    )

    penalised = evaluator.score_signals(make_signals(failed_tool_calls=2))
    recovered = evaluator.score_signals(
        make_signals(failed_tool_calls=2, recovered_after_failure=True)
    )

    assert recovered == pytest.approx(penalised + 0.1)


def test_score_is_clamped_to_the_unit_interval() -> None:
    evaluator = TrajectoryEvaluator(TrajectoryWeights(recovery_credit=5.0, failed_tool_call=10.0))

    assert evaluator.score_signals(make_signals(recovered_after_failure=True)) == 1.0
    assert evaluator.score_signals(make_signals(failed_tool_calls=3)) == 0.0


async def test_trajectory_evaluator_reports_signals_in_detail(tmp_path: Path) -> None:
    case = make_case()

    def script(recorder: TrajectoryRecorder) -> None:
        for _ in range(2):
            call = recorder.tool_call(tool="read_file", args={"path": "a.py"})
            recorder.tool_result(call_id=call, tool="read_file", ok=True, output="x")

    with workspace_for(case, base_dir=tmp_path) as workspace:
        result = await TrajectoryEvaluator().evaluate(
            EvaluationContext(
                case=case,
                workspace=workspace,
                trajectory=build_trajectory(script),
                diff=workspace.diff(),
            )
        )

    assert result.detail["redundant_reads"] == 1
    assert result.detail["files_read"] == 1
    assert result.passed is True


async def test_trajectory_fails_below_the_configured_threshold(tmp_path: Path) -> None:
    case = make_case()
    weights = TrajectoryWeights(failed_tool_call=0.5, pass_threshold=0.6)

    def script(recorder: TrajectoryRecorder) -> None:
        call = recorder.tool_call(tool="read_file", args={"path": "missing.py"})
        recorder.tool_result(call_id=call, tool="read_file", ok=False, error="gone")

    with workspace_for(case, base_dir=tmp_path) as workspace:
        result = await TrajectoryEvaluator(weights).evaluate(
            EvaluationContext(
                case=case,
                workspace=workspace,
                trajectory=build_trajectory(script),
                diff=workspace.diff(),
            )
        )

    assert result.score == pytest.approx(0.5)
    assert result.passed is False


# -- suites --------------------------------------------------------------


class StubEvaluator(Evaluator):
    def __init__(self, name: str, *, passed: bool, score: float = 1.0) -> None:
        self._name = name
        self._passed = passed
        self._score = score

    @property
    def name(self) -> str:
        return self._name

    async def evaluate(self, context: EvaluationContext) -> EvaluatorResult:
        return EvaluatorResult(name=self._name, score=self._score, passed=self._passed)


def test_suite_rejects_duplicate_evaluator_names() -> None:
    with pytest.raises(ValueError, match="duplicate evaluator names"):
        EvaluatorSuite(
            name="dupes",
            evaluators=(StubEvaluator("a", passed=True), StubEvaluator("a", passed=True)),
            gating=frozenset({"a"}),
        )


def test_suite_rejects_gating_on_an_absent_evaluator() -> None:
    with pytest.raises(ValueError, match="gating names not present"):
        EvaluatorSuite(
            name="bad-gate",
            evaluators=(StubEvaluator("a", passed=True),),
            gating=frozenset({"b"}),
        )


def test_suite_must_gate_on_something() -> None:
    with pytest.raises(ValueError, match="must gate on at least one"):
        EvaluatorSuite(
            name="ungated", evaluators=(StubEvaluator("a", passed=True),), gating=frozenset()
        )


def test_verdict_ignores_non_gating_evaluators() -> None:
    suite = EvaluatorSuite(
        name="mixed",
        evaluators=(StubEvaluator("tests", passed=True), StubEvaluator("style", passed=False)),
        gating=frozenset({"tests"}),
    )

    results = (
        EvaluatorResult(name="tests", score=1.0, passed=True),
        EvaluatorResult(name="style", score=0.0, passed=False),
    )

    # A wide diff or an inelegant trajectory must not fail a working fix.
    assert suite.verdict(results) is True


def test_verdict_requires_every_gating_result_to_be_present() -> None:
    suite = EvaluatorSuite(
        name="mixed",
        evaluators=(StubEvaluator("tests", passed=True),),
        gating=frozenset({"tests"}),
    )

    with pytest.raises(ValueError, match="missing results for gating evaluators"):
        suite.verdict(())


async def test_suite_runs_every_evaluator_in_order(tmp_path: Path) -> None:
    case = make_case()
    suite = EvaluatorSuite(
        name="two",
        evaluators=(StubEvaluator("first", passed=True), StubEvaluator("second", passed=False)),
        gating=frozenset({"first"}),
    )

    with workspace_for(case, base_dir=tmp_path) as workspace:
        results = await suite.evaluate(
            EvaluationContext(
                case=case,
                workspace=workspace,
                trajectory=empty_trajectory(),
                diff=workspace.diff(),
            )
        )

    assert [result.name for result in results] == ["first", "second"]


async def test_suite_evaluation_honours_a_deadline(tmp_path: Path) -> None:
    class SlowEvaluator(Evaluator):
        @property
        def name(self) -> str:
            return "slow"

        async def evaluate(self, context: EvaluationContext) -> EvaluatorResult:
            await asyncio.sleep(10)
            raise AssertionError("should have timed out")

    case = make_case()
    suite = EvaluatorSuite(name="slow", evaluators=(SlowEvaluator(),), gating=frozenset({"slow"}))

    with workspace_for(case, base_dir=tmp_path) as workspace:
        context = EvaluationContext(
                case=case,
                workspace=workspace,
                trajectory=empty_trajectory(),
                diff=workspace.diff(),
            )
        with pytest.raises(TimeoutError):
            await evaluate_with_timeout(suite, context, timeout_s=0.05)


def test_suite_hash_is_stable_and_sensitive_to_configuration() -> None:
    assert default_suite().content_hash == default_suite().content_hash

    tweaked = EvaluatorSuite(
        name="default",
        evaluators=(
            SuiteOutcomeEvaluator(),
            PatchLocalityEvaluator(PatchWeights(unrelated_file_penalty=0.5)),
            TrajectoryEvaluator(),
        ),
        gating=frozenset({"tests"}),
    )
    # Changing a weight changes the hash, so scores from before and after cannot
    # be silently compared.
    assert tweaked.content_hash != default_suite().content_hash


def test_default_suite_gates_only_on_tests() -> None:
    suite = default_suite()

    assert suite.gating == frozenset({"tests"})
    assert [evaluator.name for evaluator in suite.evaluators] == [
        "tests",
        "patch_locality",
        "trajectory",
    ]


def test_outcome_only_suite_is_just_the_tests() -> None:
    assert [evaluator.name for evaluator in outcome_only_suite().evaluators] == ["tests"]


def test_suites_are_resolvable_by_name() -> None:
    assert resolve_suite("default").name == "default"
    assert set(suite_names()) == {"default", "outcome-only"}


def test_unknown_suite_names_are_rejected() -> None:
    with pytest.raises(KeyError, match="unknown suite"):
        resolve_suite("nope")
