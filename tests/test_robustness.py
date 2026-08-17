"""Failure containment: one bad case must never take down a run.

Each test here corresponds to a bug found by auditing milestone 1 after the
suite was already green. They share a theme — the harness was correct on the
happy path and fatal off it, which is the worst failure mode for a batch runner
that may have spent an hour of model budget before the crash.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evalforge.datasets.builder import build_synthetic_dataset
from evalforge.env.workspace import workspace_for
from evalforge.evaluators.base import EvaluationContext, Evaluator, EvaluatorSuite
from evalforge.evaluators.outcome import SuiteOutcomeEvaluator
from evalforge.evaluators.patch import PatchLocalityEvaluator
from evalforge.evaluators.registry import SUITES
from evalforge.orchestrator.runner import LocalBackend, RunnerConfig
from evalforge.orchestrator.scheduler import InfrastructureError, Job, RetryPolicy, Scheduler
from evalforge.pipeline import RunRequest, execute_run
from evalforge.schema.case import CaseMetadata, EvalCase, FileSpec
from evalforge.schema.result import CaseResult, EvaluatorResult
from evalforge.schema.trajectory import Trajectory

BINARY_ARTEFACT_COMMAND = ("python", "-c", "open('artifact.bin','wb').write(bytes(range(256)))")


def make_case(case_id: str = "case-1", *, test_command=("python", "-c", "pass")) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        prompt="Fix the bug.",
        files=(FileSpec(path="pkg/mod.py", contents="x = 1\n"),),
        test_command=test_command,
        metadata=CaseMetadata(bug_kind="probe", target_files=("pkg/mod.py",)),
    )


def empty_trajectory() -> Trajectory:
    return Trajectory(run_id="run-1", case_id="case-1", attempt=0)


class ExplodingEvaluator(Evaluator):
    """Stands in for an evaluator with a bug in it."""

    @property
    def name(self) -> str:
        return "boom"

    async def evaluate(self, context: EvaluationContext) -> EvaluatorResult:
        if context.case.case_id.endswith("-001"):
            raise ValueError("evaluator hit an edge case")
        return EvaluatorResult(name="boom", score=1.0, passed=True)


# -- binary artefacts ----------------------------------------------------


async def test_a_binary_file_does_not_break_the_diff(tmp_path: Path) -> None:
    """A coverage database or compiled artefact must not crash change tracking."""
    case = make_case(test_command=BINARY_ARTEFACT_COMMAND)

    with workspace_for(case, base_dir=tmp_path) as workspace:
        await workspace.run(case.test_command, timeout_s=60)

        diff = workspace.diff()

    assert "artifact.bin" in diff.added


async def test_binary_content_is_hashed_by_bytes_not_decoded(tmp_path: Path) -> None:
    case = make_case()

    with workspace_for(case, base_dir=tmp_path) as workspace:
        (workspace.root / "blob.bin").write_bytes(b"\x80\x81\x82")
        first = workspace.snapshot()
        (workspace.root / "blob.bin").write_bytes(b"\x80\x81\x83")
        second = workspace.snapshot()

    assert first["blob.bin"] != second["blob.bin"]


# -- evaluator side effects ----------------------------------------------


async def test_files_created_by_an_evaluator_are_not_counted_as_agent_edits(
    tmp_path: Path,
) -> None:
    """The outcome evaluator runs the suite; whatever that leaves behind is not a patch."""
    case = make_case(test_command=BINARY_ARTEFACT_COMMAND)

    with workspace_for(case, base_dir=tmp_path) as workspace:
        # What the agent actually did, frozen at handoff.
        workspace.write_file("pkg/mod.py", "x = 2\n")
        agent_diff = workspace.diff()

        context = EvaluationContext(
            case=case, workspace=workspace, trajectory=empty_trajectory(), diff=agent_diff
        )
        await SuiteOutcomeEvaluator().evaluate(context)
        patch = await PatchLocalityEvaluator().evaluate(context)

    assert patch.detail["touched"] == ["pkg/mod.py"]
    assert patch.detail["unrelated_files"] == []
    assert patch.score == 1.0


async def test_suite_ordering_cannot_change_the_patch_verdict(tmp_path: Path) -> None:
    case = make_case(test_command=BINARY_ARTEFACT_COMMAND)

    async def score_with(evaluators: tuple[Evaluator, ...]) -> float:
        with workspace_for(case, base_dir=tmp_path) as workspace:
            workspace.write_file("pkg/mod.py", "x = 2\n")
            suite = EvaluatorSuite(
                name="ordered", evaluators=evaluators, gating=frozenset({"tests"})
            )
            results = await suite.evaluate(
                EvaluationContext(
                    case=case,
                    workspace=workspace,
                    trajectory=empty_trajectory(),
                    diff=workspace.diff(),
                )
            )
        return next(result.score for result in results if result.name == "patch_locality")

    tests_first = await score_with((SuiteOutcomeEvaluator(), PatchLocalityEvaluator()))
    patch_first = await score_with((PatchLocalityEvaluator(), SuiteOutcomeEvaluator()))

    assert tests_first == patch_first == 1.0


# -- exception containment -----------------------------------------------


async def test_a_broken_evaluator_is_an_infrastructure_fault(tmp_path: Path) -> None:
    suite = EvaluatorSuite(
        name="exploding",
        evaluators=(SuiteOutcomeEvaluator(), ExplodingEvaluator()),
        gating=frozenset({"tests"}),
    )
    backend = LocalBackend(
        run_id="run-1",
        agent_factory=lambda: _NoopAgent(),
        suite=suite,
        config=RunnerConfig(workspace_base_dir=tmp_path),
    )

    with pytest.raises(InfrastructureError, match="evaluator raised ValueError"):
        await backend.execute(Job(case=make_case("case-001")))


async def test_one_broken_case_does_not_lose_the_rest_of_the_run() -> None:
    SUITES["exploding-probe"] = lambda: EvaluatorSuite(
        name="exploding-probe",
        evaluators=(SuiteOutcomeEvaluator(), ExplodingEvaluator()),
        gating=frozenset({"tests"}),
    )
    try:
        dataset = build_synthetic_dataset(count=4, seed=7)

        run = await execute_run(
            RunRequest(
                dataset=dataset, agent_ref="scripted:idle", suite_name="exploding-probe"
            )
        )
    finally:
        del SUITES["exploding-probe"]

    # The bad case is recorded as an infrastructure fault; the others survive.
    assert len(run.case_results) == 4
    broken = [result for result in run.case_results if result.status == "infra_error"]
    assert len(broken) == 1
    assert "evaluator raised ValueError" in (broken[0].error or "")
    assert len(run.completed_results) == 3


async def test_an_unexpected_backend_error_is_recorded_not_propagated() -> None:
    class HostileBackend:
        async def execute(self, job: Job) -> CaseResult:
            raise MemoryError("something the scheduler never anticipated")

    results = await Scheduler(
        HostileBackend(), retry=RetryPolicy(max_attempts=1)
    ).run([Job(case=make_case("a")), Job(case=make_case("b"))])

    assert len(results) == 2
    assert all(result.status == "infra_error" for result in results)
    assert "unhandled MemoryError" in (results[0].error or "")


class _NoopAgent:
    """Minimal agent that satisfies the protocol without acting."""

    @property
    def name(self) -> str:
        return "test:noop"

    def config(self) -> dict[str, object]:
        return {}

    @property
    def config_hash(self) -> str:
        return "sha256:noop"

    async def run(self, context) -> None:
        return None


# -- audit findings ------------------------------------------------------


async def test_a_failing_result_sink_does_not_destroy_the_run() -> None:
    """Persisting one result must not cost the other twenty-nine.

    ``on_result`` is the store write. A transient disk or lock error used to
    escape the worker and take the whole gather down with it.
    """
    from evalforge.orchestrator.scheduler import Job, Scheduler

    seen: list[CaseResult] = []

    def flaky_sink(result: CaseResult) -> None:
        seen.append(result)
        if len(seen) == 2:
            raise OSError("disk full")

    class OkBackend:
        async def execute(self, job: Job) -> CaseResult:
            return CaseResult(
                case_id=job.case.case_id, attempt=job.attempt,
                status="completed", passed=True, duration_s=0.1,
            )

    jobs = [Job(case=make_case(f"c{index}")) for index in range(6)]
    scheduler = Scheduler(OkBackend(), on_result=flaky_sink)

    results = await scheduler.run(jobs)

    assert len(results) == 6
    # The casualty is recorded rather than swallowed.
    assert len(scheduler.unpersisted) == 1


async def test_unpersisted_results_are_reported_with_the_run_intact() -> None:
    """Raising without the run would discard what the containment protected."""
    from evalforge.datasets.builder import build_synthetic_dataset
    from evalforge.pipeline import RunRequest, UnpersistedResults, execute_run

    class BrokenStore:
        def start_run(self, run: object) -> None:
            return None

        def record_case_result(self, run_id: str, result: CaseResult) -> None:
            raise OSError("the database went away")

    dataset = build_synthetic_dataset(count=2, seed=7)

    with pytest.raises(UnpersistedResults) as caught:
        await execute_run(
            RunRequest(dataset=dataset, agent_ref="scripted:idle"),
            store=BrokenStore(),  # type: ignore[arg-type]
        )

    assert len(caught.value.results) == 2
    assert len(caught.value.run.case_results) == 2
    assert caught.value.run.run_id


def test_the_cache_refuses_a_key_that_is_not_a_content_hash(tmp_path: Path) -> None:
    """A path built from an unvalidated key is a file-write primitive."""
    from evalforge.model.cache import InvalidCacheKey, ResponseCache

    cache = ResponseCache(tmp_path)

    with pytest.raises(InvalidCacheKey):
        cache.path_for("../../../../tmp/escape")
    with pytest.raises(InvalidCacheKey):
        cache.path_for("sha256:not-hex-at-all")

    good = cache.path_for("sha256:" + "ab" * 32)
    assert tmp_path.resolve() in good.resolve().parents


def test_a_judge_can_be_attached_to_a_suite_without_gating_on_it() -> None:
    from evalforge.evaluators.llm_judge import LLMJudgeEvaluator
    from evalforge.evaluators.registry import default_suite, with_judge

    class Stub:
        model = "judge"

        async def complete(self, request: object) -> object:
            raise AssertionError("not called")

    base = default_suite()
    judged = with_judge(base, LLMJudgeEvaluator(Stub()))  # type: ignore[arg-type]

    assert "llm_judge" in [e.name for e in judged.evaluators]
    # An unvalidated judge informs a decision; it must not make one.
    assert judged.gating == base.gating
    # And the suite hash changes, so judged and unjudged runs are not confused.
    assert judged.content_hash != base.content_hash


def test_a_judge_cannot_be_attached_twice() -> None:
    from evalforge.evaluators.llm_judge import LLMJudgeEvaluator
    from evalforge.evaluators.registry import default_suite, with_judge

    class Stub:
        model = "judge"

        async def complete(self, request: object) -> object:
            raise AssertionError("not called")

    once = with_judge(default_suite(), LLMJudgeEvaluator(Stub()))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="already has"):
        with_judge(once, LLMJudgeEvaluator(Stub()))  # type: ignore[arg-type]
