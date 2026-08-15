"""Bugs found by auditing M2 after its test suite was already green.

All three share a shape: correct when called once, sequentially, from one
provider — and wrong under exactly the conditions a real evaluation run creates.
A harness that only holds up in the single-threaded happy path is not a harness.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from evalforge.model.base import ModelRequest, ModelResponse, Usage, user
from evalforge.model.budget import (
    BudgetedModelClient,
    BudgetExceeded,
    BudgetGuard,
    BudgetLimits,
)
from evalforge.model.cache import CachingModelClient, ResponseCache
from evalforge.model.chat_completions import ChatCompletionsClient
from evalforge.model.rate_limit import RateLimiter, RateLimits

MODEL = "shared-model-id"
DEFAULT_USAGE = Usage(1_000, 1_000)


def request_of(text: str = "same question", max_tokens: int = 1_000) -> ModelRequest:
    return ModelRequest(messages=(user(text),), max_tokens=max_tokens)


class SlowClient:
    """A client with latency between admission and completion."""

    model = MODEL

    def __init__(self, *, usage: Usage = DEFAULT_USAGE, cost_usd: float = 0.0) -> None:
        self.calls = 0
        self._usage = usage
        self._cost = cost_usd

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        await asyncio.sleep(0.01)
        return ModelResponse(model=MODEL, usage=self._usage, cost_usd=self._cost)


class ExplodingClient:
    model = MODEL

    async def complete(self, request: ModelRequest) -> ModelResponse:
        await asyncio.sleep(0.01)
        raise RuntimeError("the provider dropped the connection")


# -- the budget must hold under concurrency ------------------------------


async def test_a_shared_budget_refuses_concurrent_attempts_past_the_ceiling() -> None:
    """Checking then recording across an await lets every attempt pass at once."""
    inner = SlowClient()
    guard = BudgetGuard(limits=BudgetLimits(max_tokens=5_000))
    client = BudgetedModelClient(inner=inner, guard=guard)

    results = await asyncio.gather(
        *(client.complete(request_of()) for _ in range(10)), return_exceptions=True
    )

    refused = [r for r in results if isinstance(r, BudgetExceeded)]
    assert refused, "a ceiling that admits everything is not a ceiling"
    assert inner.calls < 10
    assert guard.refusals == len(refused)
    assert guard.in_flight == 0


async def test_once_the_ceiling_is_reached_everything_after_is_refused() -> None:
    """The guard bounds overshoot rather than eliminating it — so prove the bound.

    An input estimate can undershoot, so a wave of concurrent calls can settle
    above the ceiling. What must hold is that the *next* reservation sees the
    true settled total and refuses, instead of letting the overrun compound.
    """
    inner = SlowClient(usage=Usage(4_000, 1_000))
    guard = BudgetGuard(limits=BudgetLimits(max_tokens=5_000))
    client = BudgetedModelClient(inner=inner, guard=guard)

    await client.complete(request_of())
    assert guard.tokens == 5_000

    for _ in range(5):
        with pytest.raises(BudgetExceeded):
            await client.complete(request_of())

    assert inner.calls == 1


async def test_capacity_is_released_when_a_call_fails() -> None:
    guard = BudgetGuard(limits=BudgetLimits(max_tokens=10_000))
    client = BudgetedModelClient(inner=ExplodingClient(), guard=guard)

    with pytest.raises(RuntimeError):
        await client.complete(request_of())

    # A call that never returned must not hold its reservation forever.
    assert guard.in_flight == 0
    assert guard.committed_tokens == 0


async def test_reservations_are_visible_while_calls_are_in_flight() -> None:
    guard = BudgetGuard(limits=BudgetLimits(max_tokens=100_000))
    client = BudgetedModelClient(inner=SlowClient(), guard=guard)

    task = asyncio.create_task(client.complete(request_of()))
    await asyncio.sleep(0)  # let the reservation land, not the response

    assert guard.in_flight == 1
    assert guard.committed_tokens > guard.tokens

    await task
    assert guard.in_flight == 0
    assert guard.committed_tokens == guard.tokens


async def test_the_call_limit_also_counts_calls_in_flight() -> None:
    inner = SlowClient()
    guard = BudgetGuard(limits=BudgetLimits(max_calls=3))
    client = BudgetedModelClient(inner=inner, guard=guard)

    await asyncio.gather(
        *(client.complete(request_of()) for _ in range(8)), return_exceptions=True
    )

    assert inner.calls == 3


# -- the cache must distinguish providers --------------------------------


def _handler(label: str, seen: list[str]):
    def handler(_request: httpx.Request) -> httpx.Response:
        seen.append(label)
        return httpx.Response(
            200,
            json={
                "model": MODEL,
                "choices": [
                    {"message": {"role": "assistant", "content": f"answer from {label}"},
                     "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    return handler


async def _complete_via(base_url: str, label: str, seen: list[str], cache_dir: Path) -> Any:
    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler(label, seen))) as http:
        client = CachingModelClient(
            inner=ChatCompletionsClient(
                base_url=base_url, api_key="k", model=MODEL, http=http
            ),
            cache=ResponseCache(cache_dir),
            scope=base_url,
        )
        return await client.complete(request_of())


async def test_two_providers_serving_one_model_id_do_not_share_cache(tmp_path: Path) -> None:
    """Open-weights models are served by several providers under the same id."""
    seen: list[str] = []

    first = await _complete_via("https://api-a.test/v1", "provider-A", seen, tmp_path)
    second = await _complete_via("https://api-b.test/v1", "provider-B", seen, tmp_path)

    assert first.text == "answer from provider-A"
    assert second.text == "answer from provider-B"
    assert second.cached is False
    assert seen == ["provider-A", "provider-B"]


async def test_the_same_provider_still_hits_the_cache(tmp_path: Path) -> None:
    seen: list[str] = []

    await _complete_via("https://api-a.test/v1", "provider-A", seen, tmp_path)
    repeat = await _complete_via("https://api-a.test/v1", "provider-A", seen, tmp_path)

    assert repeat.cached is True
    assert seen == ["provider-A"]


def test_the_cache_key_covers_the_endpoint() -> None:
    request = request_of()

    assert request.cache_key(MODEL, "https://a.test") != request.cache_key(MODEL, "https://b.test")


# -- retries must be paced -----------------------------------------------


class Schedule:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.now += delay


async def test_retried_requests_count_against_the_rate_limit() -> None:
    """A retry costs the provider's allowance exactly as much as the original."""
    attempts: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 4:
            return httpx.Response(429, headers={"retry-after": "1"}, json={"error": "slow down"})
        return httpx.Response(
            200,
            json={
                "model": MODEL,
                "choices": [
                    {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    schedule = Schedule()
    limiter = RateLimiter(
        RateLimits(requests_per_minute=10), clock=schedule.time, sleep=schedule.sleep
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ChatCompletionsClient(
            base_url="https://x.test/v1",
            api_key="k",
            model=MODEL,
            http=http,
            rate_limiter=limiter,
            sleep=schedule.sleep,
        )
        await client.complete(request_of(max_tokens=32))

    assert len(attempts) == 4
    # Every attempt, not just the first, is inside the limiter's window.
    assert len(limiter._events) == 4
