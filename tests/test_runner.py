"""Executing a single attempt, and the failure/fault distinction it enforces."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from evalforge.agent.base import Agent, AgentContext
from evalforge.evaluators.base import EvaluationContext, Evaluator, EvaluatorSuite
from evalforge.evaluators.registry import outcome_only_suite
from evalforge.orchestrator.runner import LocalBackend, RunnerConfig
from evalforge.orchestrator.scheduler import InfrastructureError, Job
from evalforge.schema.case import CaseMetadata, EvalCase, FileSpec
from evalforge.schema.result import EvaluatorResult
from evalforge.schema.trajectory import AgentError, SafetyViolation, Trajectory
from evalforge.trace import FakeClock

PASSING_COMMAND = ("python", "-c", "pass")


def make_case(case_id: str = "case-1", *, test_command=PASSING_COMMAND) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        prompt="Fix the bug.",
        files=(FileSpec(path="pkg/mod.py", contents="x = 1\n"),),
        test_command=test_command,
        metadata=CaseMetadata(bug_kind="off_by_one", target_files=("pkg/mod.py",)),
    )


class CrashingAgent(Agent):
    @property
    def name(self) -> str:
        return "test:crashing"

    async def run(self, context: AgentContext) -> None:
        raise RuntimeError("the agent exploded")


class EscapingAgent(Agent):
    """An agent that tries to write outside its workspace."""

    @property
    def name(self) -> str:
        return "test:escaping"

    async def run(self, context: AgentContext) -> None:
        from evalforge.agent.tools import ToolBox

        tools = ToolBox(
            case=context.case, workspace=context.workspace, recorder=context.recorder
        )
        tools.write_file("../../escaped.txt", "pwned")


class QuietAgent(Agent):
    @property
    def name(self) -> str:
        return "test:quiet"

    async def run(self, context: AgentContext) -> None:
        context.recorder.task_started(prompt_hash=context.case.content_hash)


class SlowEvaluator(Evaluator):
    @property
    def name(self) -> str:
        return "slow"

    async def evaluate(self, context: EvaluationContext) -> EvaluatorResult:
        await asyncio.sleep(10)
        raise AssertionError("should have timed out")


def make_backend(agent_factory, *, config: RunnerConfig | None = None, suite=None) -> LocalBackend:
    return LocalBackend(
        run_id="run-1",
        agent_factory=agent_factory,
        suite=suite or outcome_only_suite(),
        config=config or RunnerConfig(),
        clock_factory=lambda: FakeClock(auto_advance_ms=1.0),
    )


async def test_a_crashing_agent_is_a_result_not_a_fault(tmp_path: Path) -> None:
    backend = make_backend(
        CrashingAgent, config=RunnerConfig(trajectory_dir=tmp_path / "traces")
    )

    result = await backend.execute(Job(case=make_case()))

    # The attempt completes and is measured; the crash is recorded in the trace.
    assert result.status == "completed"
    assert result.evaluator("tests") is not None

    assert result.trajectory_path is not None
    trajectory = Trajectory.from_jsonl(
        run_id="run-1",
        case_id="case-1",
        attempt=0,
        jsonl=Path(result.trajectory_path).read_text(encoding="utf-8"),
    )
    errors = trajectory.of_type(AgentError)
    assert errors[0].error_type == "RuntimeError"
    assert "exploded" in errors[0].message


async def test_a_crashing_agent_fails_a_case_it_did_not_fix(tmp_path: Path) -> None:
    failing = ("python", "-c", "raise SystemExit(1)")
    backend = make_backend(CrashingAgent)

    result = await backend.execute(Job(case=make_case(test_command=failing)))

    assert result.passed is False


async def test_an_escape_attempt_lands_in_the_trajectory(tmp_path: Path) -> None:
    backend = make_backend(
        EscapingAgent, config=RunnerConfig(trajectory_dir=tmp_path / "traces")
    )

    result = await backend.execute(Job(case=make_case()))

    assert result.trajectory_path is not None
    trajectory = Trajectory.from_jsonl(
        run_id="run-1",
        case_id="case-1",
        attempt=0,
        jsonl=Path(result.trajectory_path).read_text(encoding="utf-8"),
    )
    violations = trajectory.of_type(SafetyViolation)
    assert violations[0].rule == "path_escape"
    assert not (tmp_path / "escaped.txt").exists()


async def test_a_hung_evaluator_is_an_infrastructure_fault() -> None:
    suite = EvaluatorSuite(
        name="slow", evaluators=(SlowEvaluator(),), gating=frozenset({"slow"})
    )
    backend = make_backend(
        QuietAgent, config=RunnerConfig(evaluation_timeout_s=0.05), suite=suite
    )

    # Distinct from an agent failing: the harness broke, so the scheduler retries.
    with pytest.raises(InfrastructureError, match="evaluation exceeded"):
        await backend.execute(Job(case=make_case()))


async def test_trajectories_are_written_one_file_per_attempt(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    backend = make_backend(QuietAgent, config=RunnerConfig(trajectory_dir=traces))

    first = await backend.execute(Job(case=make_case("case-a"), attempt=0))
    second = await backend.execute(Job(case=make_case("case-a"), attempt=1))

    assert first.trajectory_path != second.trajectory_path
    assert Path(first.trajectory_path or "").name == "case-a--0.jsonl"
    assert Path(second.trajectory_path or "").name == "case-a--1.jsonl"


async def test_case_ids_with_awkward_characters_get_safe_filenames(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    backend = make_backend(QuietAgent, config=RunnerConfig(trajectory_dir=traces))

    result = await backend.execute(Job(case=make_case("weird/../id name")))

    assert result.trajectory_path is not None
    path = Path(result.trajectory_path)
    assert path.is_file()
    assert traces.resolve() in path.resolve().parents


async def test_attempt_duration_is_measured() -> None:
    backend = make_backend(QuietAgent)

    result = await backend.execute(Job(case=make_case()))

    assert result.duration_s > 0.0
