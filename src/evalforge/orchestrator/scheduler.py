"""Turning an evaluation request into executed jobs.

The interesting part is not parallelism, it is *what counts as a failure*. An
agent that cannot solve a task has produced a result and is never retried;
re-rolling it would quietly inflate the success rate. Only genuine
infrastructure faults — a workspace that could not be created, a transport that
returned a 503 — are retried, and a per-job timeout is recorded as its own
status rather than being retried or silently counted as a failure.

Results are emitted through ``on_result`` as they land, so a run killed halfway
leaves a store containing exactly the attempts that finished.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from evalforge.schema.case import EvalCase
from evalforge.schema.result import CaseResult, CaseStatus

DEFAULT_CONCURRENCY = 4
DEFAULT_JOB_TIMEOUT_S = 600.0


class InfrastructureError(Exception):
    """A harness-level fault worth retrying — never an agent's failure to solve."""


@dataclass(frozen=True, slots=True)
class Job:
    """One attempt at one case."""

    case: EvalCase
    attempt: int = 0

    @property
    def label(self) -> str:
        return f"{self.case.case_id}#{self.attempt}"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff for infrastructure faults."""

    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def delay_for(self, retry_index: int) -> float:
        """Delay before retry number ``retry_index`` (0-based)."""
        return min(self.max_delay_s, self.base_delay_s * (2.0**retry_index))


class ExecutorBackend(Protocol):
    """Where a job actually runs.

    The seam that keeps a distributed backend a swap rather than a rewrite: a
    Celery or Kubernetes executor implements this same method, and the scheduler
    above it is unchanged.
    """

    async def execute(self, job: Job) -> CaseResult: ...


def build_jobs(cases: Sequence[EvalCase], *, samples_per_case: int = 1) -> tuple[Job, ...]:
    """Expand cases into attempts, ordered so samples of a case are spread out.

    Interleaving matters under concurrency: running a case's k samples
    back-to-back on the same machine correlates their environment noise, which is
    exactly the variance the sampling is supposed to measure.
    """
    if samples_per_case < 1:
        raise ValueError("samples_per_case must be at least 1")
    return tuple(
        Job(case=case, attempt=attempt)
        for attempt in range(samples_per_case)
        for case in cases
    )


class Scheduler:
    """Runs jobs with bounded concurrency, timeouts, retries and cancellation."""

    def __init__(
        self,
        backend: ExecutorBackend,
        *,
        concurrency: int = DEFAULT_CONCURRENCY,
        job_timeout_s: float = DEFAULT_JOB_TIMEOUT_S,
        retry: RetryPolicy | None = None,
        on_result: Callable[[CaseResult], None] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self._backend = backend
        self._concurrency = concurrency
        self._job_timeout_s = job_timeout_s
        self._retry = retry or RetryPolicy()
        self._on_result = on_result
        self._sleep = sleep

    async def run(self, jobs: Iterable[Job]) -> tuple[CaseResult, ...]:
        """Execute every job, returning results in job order.

        On cancellation, in-flight work is cancelled and the results that already
        completed are returned rather than discarded.
        """
        job_list = list(jobs)
        if not job_list:
            return ()

        semaphore = asyncio.Semaphore(self._concurrency)
        results: list[CaseResult | None] = [None] * len(job_list)

        async def worker(index: int, job: Job) -> None:
            async with semaphore:
                result = await self._run_job(job)
            results[index] = result
            if self._on_result is not None:
                self._on_result(result)

        tasks = [
            asyncio.create_task(worker(index, job), name=f"evalforge-job-{job.label}")
            for index, job in enumerate(job_list)
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            return tuple(result for result in results if result is not None)

        return tuple(result for result in results if result is not None)

    async def _run_job(self, job: Job) -> CaseResult:
        last_error: Exception | None = None

        for attempt_index in range(self._retry.max_attempts):
            try:
                return await asyncio.wait_for(
                    self._backend.execute(job), timeout=self._job_timeout_s
                )
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                # A job that overruns is a fact about the attempt, not a fault to
                # re-roll: retrying would hide a systematically slow agent.
                return _status_result(
                    job,
                    status="timed_out",
                    error=f"job exceeded {self._job_timeout_s:g}s",
                )
            except InfrastructureError as exc:
                last_error = exc
                if attempt_index + 1 < self._retry.max_attempts:
                    await self._sleep(self._retry.delay_for(attempt_index))
                    continue

        return _status_result(
            job,
            status="infra_error",
            error=f"{type(last_error).__name__}: {last_error}"
            if last_error is not None
            else "infrastructure error",
        )


def _status_result(job: Job, *, status: CaseStatus, error: str) -> CaseResult:
    return CaseResult(
        case_id=job.case.case_id,
        attempt=job.attempt,
        status=status,
        passed=False,
        duration_s=0.0,
        error=error,
    )
