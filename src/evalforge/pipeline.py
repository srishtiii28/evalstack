"""One evaluation run, end to end.

Assembles the agent, dataset, suite, scheduler and store into a single call, and
records the three content hashes that make the run's claims checkable. Results
are handed to the store as each attempt finishes, so an interrupted run leaves a
readable partial record instead of nothing.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from evalforge.agent.base import Agent
from evalforge.agent.registry import resolve_agent
from evalforge.env.workspace import ResourceLimits
from evalforge.evaluators.registry import resolve_suite
from evalforge.orchestrator.runner import LocalBackend, RunnerConfig
from evalforge.orchestrator.scheduler import (
    DEFAULT_CONCURRENCY,
    DEFAULT_JOB_TIMEOUT_S,
    RetryPolicy,
    Scheduler,
    build_jobs,
)
from evalforge.schema.dataset import Dataset
from evalforge.schema.result import CaseResult, RunResult
from evalforge.store.db import Store

RUN_ID_ENTROPY_BYTES = 3


def new_run_id(*, now: datetime | None = None) -> str:
    """A sortable, human-readable run id with enough entropy to avoid collisions."""
    moment = now or datetime.now(UTC)
    return f"run-{moment.strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(RUN_ID_ENTROPY_BYTES)}"


@dataclass(frozen=True, slots=True)
class RunRequest:
    """Everything that defines a run, before it is given an identity."""

    dataset: Dataset
    agent_ref: str
    suite_name: str = "default"
    samples_per_case: int = 1
    concurrency: int = DEFAULT_CONCURRENCY
    job_timeout_s: float = DEFAULT_JOB_TIMEOUT_S
    case_ids: tuple[str, ...] = ()
    trajectory_dir: Path | None = None
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    notes: str = ""

    def selected_dataset(self) -> Dataset:
        """The dataset actually being run, honouring any case filter."""
        if not self.case_ids:
            return self.dataset
        return self.dataset.subset(self.case_ids)


async def execute_run(
    request: RunRequest,
    *,
    store: Store | None = None,
    run_id: str | None = None,
    agent_factory: Callable[[], Agent] | None = None,
) -> RunResult:
    """Run every case (times ``samples_per_case``) and return the assembled result."""
    dataset = request.selected_dataset()
    build_agent = agent_factory or (lambda: resolve_agent(request.agent_ref))
    suite = resolve_suite(request.suite_name)

    # Built once purely to record the configuration hash; each attempt gets its own.
    probe_agent = build_agent()

    header = RunResult(
        run_id=run_id or new_run_id(),
        agent_ref=request.agent_ref,
        agent_hash=probe_agent.config_hash,
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_hash=dataset.content_hash,
        suite_name=suite.name,
        suite_hash=suite.content_hash,
        samples_per_case=request.samples_per_case,
        concurrency=request.concurrency,
        notes=request.notes,
    )

    if store is not None:
        store.start_run(header)

    def persist(result: CaseResult) -> None:
        if store is not None:
            store.record_case_result(header.run_id, result)

    backend = LocalBackend(
        run_id=header.run_id,
        agent_factory=build_agent,
        suite=suite,
        config=RunnerConfig(
            trajectory_dir=request.trajectory_dir,
            limits=request.limits,
        ),
    )
    scheduler = Scheduler(
        backend,
        concurrency=request.concurrency,
        job_timeout_s=request.job_timeout_s,
        retry=RetryPolicy(),
        on_result=persist,
    )

    results = await scheduler.run(
        build_jobs(dataset.cases, samples_per_case=request.samples_per_case)
    )
    # Ordered by (case, attempt) rather than by completion or job order, so an
    # in-memory run and the same run read back from the store agree — `run` and
    # `show` should not disagree about the order of the same results.
    ordered = tuple(sorted(results, key=lambda result: (result.case_id, result.attempt)))
    return header.model_copy(update={"case_results": ordered})
