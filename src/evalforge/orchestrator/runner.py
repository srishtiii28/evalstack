"""Executing one attempt: workspace in, evaluated result out.

This is where the pieces meet. It also draws the line the whole harness depends
on: an exception from *the agent* is recorded as an ``AgentError`` event and the
attempt proceeds to evaluation (a crashed agent has failed the task, which is a
measurement), while an exception from *the harness* is raised as an
``InfrastructureError`` for the scheduler to retry.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from evalforge.agent.base import Agent, AgentContext
from evalforge.env.workspace import ResourceLimits, Violation, workspace_for
from evalforge.evaluators.base import EvaluationContext, EvaluatorSuite
from evalforge.hashing import short, text_hash
from evalforge.orchestrator.scheduler import InfrastructureError, Job
from evalforge.schema.result import CaseResult
from evalforge.schema.trajectory import Trajectory
from evalforge.trace import Clock, TrajectoryRecorder

DEFAULT_EVALUATION_TIMEOUT_S = 600.0

_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(value: str) -> str:
    """Make an identifier safe to use as a path component.

    Sanitising is lossy — ``a/b`` and ``a_b`` both reduce to ``a_b`` — so any
    name that had to be rewritten carries a hash of the original. Two cases
    would otherwise silently overwrite each other's trajectory.
    """
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", value).strip("._")
    if not cleaned:
        return f"unnamed-{short(text_hash(value), 8)}"
    if cleaned != value:
        return f"{cleaned}-{short(text_hash(value), 8)}"
    return cleaned


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    """Knobs for how an attempt is executed and where its trace lands."""

    trajectory_dir: Path | None = None
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    evaluation_timeout_s: float = DEFAULT_EVALUATION_TIMEOUT_S
    workspace_base_dir: Path | None = None


class LocalBackend:
    """Runs attempts in this process, one workspace per attempt.

    ``agent_factory`` rather than a shared agent instance: attempts run
    concurrently, and a model-driven agent carries per-attempt conversation
    state that must not be shared between them.
    """

    def __init__(
        self,
        *,
        run_id: str,
        agent_factory: Callable[[], Agent],
        suite: EvaluatorSuite,
        config: RunnerConfig | None = None,
        clock_factory: Callable[[], Clock] | None = None,
    ) -> None:
        self.run_id = run_id
        self._agent_factory = agent_factory
        self._suite = suite
        self._config = config or RunnerConfig()
        self._clock_factory = clock_factory

    async def execute(self, job: Job) -> CaseResult:
        case = job.case
        recorder = TrajectoryRecorder(
            run_id=self.run_id,
            case_id=case.case_id,
            attempt=job.attempt,
            clock=self._clock_factory() if self._clock_factory is not None else None,
        )

        def record_violation(violation: Violation) -> None:
            recorder.safety_violation(
                rule=violation.rule,
                detail=violation.detail,
                attempted=violation.attempted,
            )

        started = time.monotonic()
        try:
            with workspace_for(
                case,
                base_dir=self._config.workspace_base_dir,
                limits=self._config.limits,
                on_violation=record_violation,
            ) as workspace:
                agent = self._agent_factory()
                context = AgentContext(case=case, workspace=workspace, recorder=recorder)

                try:
                    await agent.run(context)
                except asyncio.CancelledError:
                    raise
                except InfrastructureError:
                    # The provider broke, not the agent. Recording this as an
                    # agent failure would turn an outage into a capability
                    # regression, so it propagates to the scheduler instead.
                    raise
                except Exception as exc:
                    # Deliberately broad: an agent that crashes has failed the
                    # task, which is a measurement. Letting it propagate would
                    # turn a bad agent into a broken harness.
                    recorder.agent_error(error_type=type(exc).__name__, message=str(exc))

                evaluation_context = EvaluationContext(
                    case=case,
                    workspace=workspace,
                    trajectory=recorder.build(),
                    # Frozen here, before any evaluator runs a command that
                    # could leave files behind and inflate the patch diff.
                    diff=workspace.diff(),
                )
                try:
                    evaluator_results = await asyncio.wait_for(
                        self._suite.evaluate(evaluation_context),
                        timeout=self._config.evaluation_timeout_s,
                    )
                except TimeoutError as exc:
                    raise InfrastructureError(
                        f"evaluation exceeded {self._config.evaluation_timeout_s:g}s"
                    ) from exc
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # A broken evaluator is a harness fault, not a verdict on the
                    # agent. Surfacing it as one keeps a single bad case from
                    # taking down every other result in the run.
                    raise InfrastructureError(
                        f"evaluator raised {type(exc).__name__}: {exc}"
                    ) from exc
        except OSError as exc:
            raise InfrastructureError(f"workspace failure: {exc}") from exc

        duration_s = time.monotonic() - started
        trajectory = recorder.build()
        trajectory_path = self._persist(trajectory)

        return CaseResult(
            case_id=case.case_id,
            attempt=job.attempt,
            status="completed",
            passed=self._suite.verdict(evaluator_results),
            evaluators=evaluator_results,
            duration_s=duration_s,
            cost_usd=trajectory.total_cost_usd,
            input_tokens=trajectory.total_input_tokens,
            output_tokens=trajectory.total_output_tokens,
            trajectory_path=trajectory_path,
        )

    def _persist(self, trajectory: Trajectory) -> str | None:
        directory = self._config.trajectory_dir
        if directory is None:
            return None

        target_dir = directory / _safe_filename(self.run_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{_safe_filename(trajectory.case_id)}--{trajectory.attempt}.jsonl"
        path = target_dir / filename
        path.write_text(trajectory.to_jsonl(), encoding="utf-8")
        return str(path)
