"""Scheduler semantics: concurrency, retries, timeouts and cancellation.

The distinction these tests exist to protect: an agent that fails a task must
never be retried (that would inflate success rates), while a genuine
infrastructure fault must be.
"""

from __future__ import annotations

import asyncio

import pytest

from evalforge.orchestrator.scheduler import (
    InfrastructureError,
    Job,
    RetryPolicy,
    Scheduler,
    build_jobs,
)
from evalforge.schema.case import EvalCase, FileSpec
from evalforge.schema.result import CaseResult


def make_case(case_id: str) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        prompt="Fix it.",
        files=(FileSpec(path="a.py", contents="x = 1\n"),),
        test_command=("python", "-c", "pass"),
    )


def completed(job: Job, *, passed: bool = True) -> CaseResult:
    return CaseResult(
        case_id=job.case.case_id,
        attempt=job.attempt,
        status="completed",
        passed=passed,
        duration_s=0.0,
    )


class RecordingBackend:
    """A backend whose behaviour each test dictates."""

    def __init__(self, behaviour=None) -> None:
        self.calls: list[str] = []
        self.live = 0
        self.peak = 0
        self._behaviour = behaviour

    async def execute(self, job: Job) -> CaseResult:
        self.calls.append(job.label)
        self.live += 1
        self.peak = max(self.peak, self.live)
        try:
            await asyncio.sleep(0)
            if self._behaviour is not None:
                return await self._behaviour(job, len(self.calls))
            return completed(job)
        finally:
            self.live -= 1


async def no_sleep(_delay: float) -> None:
    return None


async def test_runs_every_job_and_preserves_order() -> None:
    backend = RecordingBackend()
    jobs = [Job(case=make_case(f"case-{index}")) for index in range(5)]

    results = await Scheduler(backend, sleep=no_sleep).run(jobs)

    assert [result.case_id for result in results] == [job.case.case_id for job in jobs]
    assert all(result.status == "completed" for result in results)


async def test_empty_job_list_is_a_no_op() -> None:
    backend = RecordingBackend()
    assert await Scheduler(backend, sleep=no_sleep).run([]) == ()
    assert backend.calls == []


async def test_concurrency_cap_is_respected() -> None:
    async def slow(job: Job, _call_count: int) -> CaseResult:
        await asyncio.sleep(0.02)
        return completed(job)

    backend = RecordingBackend(slow)
    jobs = [Job(case=make_case(f"case-{index}")) for index in range(12)]

    await Scheduler(backend, concurrency=3, sleep=no_sleep).run(jobs)

    assert backend.peak <= 3
    assert len(backend.calls) == 12


async def test_concurrency_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="concurrency must be at least 1"):
        Scheduler(RecordingBackend(), concurrency=0)


async def test_infrastructure_error_is_retried_then_succeeds() -> None:
    async def flaky(job: Job, call_count: int) -> CaseResult:
        if call_count == 1:
            raise InfrastructureError("transient")
        return completed(job)

    backend = RecordingBackend(flaky)
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    results = await Scheduler(backend, sleep=record_sleep).run([Job(case=make_case("a"))])

    assert len(backend.calls) == 2
    assert results[0].status == "completed"
    assert delays == [0.5]


async def test_infrastructure_error_gives_up_after_max_attempts() -> None:
    async def always_broken(_job: Job, _call_count: int) -> CaseResult:
        raise InfrastructureError("disk on fire")

    backend = RecordingBackend(always_broken)
    retry = RetryPolicy(max_attempts=3)

    results = await Scheduler(backend, retry=retry, sleep=no_sleep).run([Job(case=make_case("a"))])

    assert len(backend.calls) == 3
    assert results[0].status == "infra_error"
    assert results[0].passed is False
    assert "disk on fire" in (results[0].error or "")


async def test_a_failing_agent_is_never_retried() -> None:
    async def agent_fails(job: Job, _call_count: int) -> CaseResult:
        return completed(job, passed=False)

    backend = RecordingBackend(agent_fails)

    results = await Scheduler(backend, sleep=no_sleep).run([Job(case=make_case("a"))])

    # Re-rolling a failed attempt would quietly inflate the success rate.
    assert len(backend.calls) == 1
    assert results[0].status == "completed"
    assert results[0].passed is False


async def test_timeout_is_recorded_and_not_retried() -> None:
    async def hangs(_job: Job, _call_count: int) -> CaseResult:
        await asyncio.sleep(10)
        raise AssertionError("should have been cancelled")

    backend = RecordingBackend(hangs)

    results = await Scheduler(backend, job_timeout_s=0.05, sleep=no_sleep).run(
        [Job(case=make_case("a"))]
    )

    assert len(backend.calls) == 1
    assert results[0].status == "timed_out"
    assert "0.05s" in (results[0].error or "")
    # A timeout that reported zero duration would drag the mean down and hide
    # exactly the slowness it is evidence of.
    assert results[0].duration_s > 0.0


async def test_results_are_reported_as_they_land() -> None:
    seen: list[str] = []
    backend = RecordingBackend()
    jobs = [Job(case=make_case(f"case-{index}")) for index in range(4)]

    await Scheduler(backend, on_result=lambda result: seen.append(result.case_id)).run(jobs)

    assert sorted(seen) == sorted(job.case.case_id for job in jobs)


async def test_cancellation_returns_the_results_already_finished() -> None:
    started = asyncio.Event()

    async def slow_after_first(job: Job, call_count: int) -> CaseResult:
        if call_count == 1:
            return completed(job)
        started.set()
        await asyncio.sleep(10)
        raise AssertionError("should have been cancelled")

    backend = RecordingBackend(slow_after_first)
    jobs = [Job(case=make_case(f"case-{index}")) for index in range(3)]
    scheduler = Scheduler(backend, concurrency=1, sleep=no_sleep)

    task = asyncio.create_task(scheduler.run(jobs))
    await started.wait()
    task.cancel()
    results = await task

    assert len(results) == 1
    assert results[0].case_id == "case-0"


def test_build_jobs_interleaves_samples() -> None:
    cases = [make_case("a"), make_case("b")]

    jobs = build_jobs(cases, samples_per_case=2)

    # Samples of the same case are spread apart so their environment noise is
    # not correlated by running back-to-back.
    assert [job.label for job in jobs] == ["a#0", "b#0", "a#1", "b#1"]


def test_build_jobs_rejects_zero_samples() -> None:
    with pytest.raises(ValueError, match="samples_per_case must be at least 1"):
        build_jobs([make_case("a")], samples_per_case=0)


def test_retry_policy_backs_off_exponentially_up_to_a_ceiling() -> None:
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=4.0)

    assert [policy.delay_for(index) for index in range(5)] == [1.0, 2.0, 4.0, 4.0, 4.0]


def test_retry_policy_requires_at_least_one_attempt() -> None:
    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        RetryPolicy(max_attempts=0)


def test_job_label_identifies_the_attempt() -> None:
    assert Job(case=make_case("alpha"), attempt=2).label == "alpha#2"
