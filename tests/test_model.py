"""The model layer: pricing, budgets, caching, rate limiting and the transport.

Everything here runs offline against a mock transport. A model layer you can
only test by spending money is a model layer nobody tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from evalforge.model.base import (
    ModelBehaviourError,
    ModelRequest,
    ModelResponse,
    PermanentModelError,
    QuotaExhaustedError,
    ToolSpec,
    TransientModelError,
    Usage,
    assistant,
    system,
    tool_result,
    user,
)
from evalforge.model.budget import (
    BudgetedModelClient,
    BudgetExceeded,
    BudgetGuard,
    BudgetLimits,
)
from evalforge.model.cache import CachingModelClient, ResponseCache
from evalforge.model.chat_completions import ChatCompletionsClient
from evalforge.model.pricing import FREE, UNKNOWN, ModelPricing, PricingTable
from evalforge.model.providers import (
    GROQ,
    ProviderNotConfiguredError,
    provider_names,
    resolve_provider,
)
from evalforge.model.rate_limit import RateLimiter, RateLimits

MODEL = "test-model"


def make_request(**overrides: Any) -> ModelRequest:
    defaults: dict[str, Any] = {
        "messages": (system("be helpful"), user("fix the bug")),
        "tools": (ToolSpec(name="noop", description="does nothing", parameters={}),),
        "temperature": 0.0,
        "max_tokens": 256,
    }
    return ModelRequest(**(defaults | overrides))


def chat_response(
    *, content: str = "", tool_calls: list[dict[str, Any]] | None = None, finish: str = "stop"
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "model": MODEL,
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


class StubClient:
    """A model client returning canned responses."""

    def __init__(self, *responses: ModelResponse) -> None:
        self._responses = list(responses)
        self.calls: list[ModelRequest] = []

    @property
    def model(self) -> str:
        return MODEL

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        if self._responses:
            return self._responses.pop(0)
        return ModelResponse(model=MODEL, text="ok", usage=Usage(10, 5))


async def no_sleep(_delay: float) -> None:
    return None


# -- pricing -------------------------------------------------------------


def test_cost_is_proportional_to_tokens() -> None:
    pricing = ModelPricing(input_per_mtok=1.0, output_per_mtok=5.0)

    assert pricing.cost_for(Usage(1_000_000, 0)) == pytest.approx(1.0)
    assert pricing.cost_for(Usage(0, 1_000_000)) == pytest.approx(5.0)
    assert pricing.cost_for(Usage(500_000, 200_000)) == pytest.approx(1.5)


def test_an_unknown_rate_reports_untracked_rather_than_zero() -> None:
    # A confident $0.00 would make a cost comparison silently meaningless.
    assert UNKNOWN.tracked is False
    assert PricingTable(default=UNKNOWN).tracks("mystery-model") is False


def test_a_free_tier_is_zero_and_tracked() -> None:
    assert FREE.tracked is True
    assert PricingTable().cost_for(MODEL, Usage(10_000, 10_000)) == 0.0


def test_rates_can_be_supplied_through_the_environment(monkeypatch) -> None:
    monkeypatch.setenv(
        "EVALFORGE_MODEL_PRICING", json.dumps({MODEL: {"input": 2.0, "output": 10.0}})
    )

    assert PricingTable().cost_for(MODEL, Usage(1_000_000, 0)) == pytest.approx(2.0)


def test_malformed_pricing_configuration_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("EVALFORGE_MODEL_PRICING", "not json")

    with pytest.raises(ValueError, match="not valid JSON"):
        PricingTable()


def test_pricing_entries_need_both_rates(monkeypatch) -> None:
    monkeypatch.setenv("EVALFORGE_MODEL_PRICING", json.dumps({MODEL: {"input": 1.0}}))

    with pytest.raises(ValueError, match="numeric 'input' and 'output'"):
        PricingTable()


# -- budgets -------------------------------------------------------------


async def test_a_call_over_the_ceiling_is_refused_before_it_is_made() -> None:
    inner = StubClient()
    guard = BudgetGuard(limits=BudgetLimits(max_tokens=100))
    client = BudgetedModelClient(inner=inner, guard=guard)

    with pytest.raises(BudgetExceeded, match="tokens"):
        await client.complete(make_request(max_tokens=1_000))

    # The point of a pre-flight guard: the request never left.
    assert inner.calls == []
    assert guard.refusals == 1


async def test_the_call_limit_stops_a_runaway_loop() -> None:
    guard = BudgetGuard(limits=BudgetLimits(max_calls=2))
    client = BudgetedModelClient(inner=StubClient(), guard=guard)

    await client.complete(make_request())
    await client.complete(make_request())
    with pytest.raises(BudgetExceeded, match="limit of 2 calls"):
        await client.complete(make_request())

    assert guard.calls == 2


async def test_spend_is_recorded_against_the_guard() -> None:
    inner = StubClient(ModelResponse(model=MODEL, usage=Usage(100, 50), cost_usd=0.25))
    guard = BudgetGuard(limits=BudgetLimits(max_usd=1.0))
    client = BudgetedModelClient(inner=inner, guard=guard)

    await client.complete(make_request())

    assert guard.spent_usd == pytest.approx(0.25)
    assert guard.tokens == 150
    assert guard.calls == 1


async def test_an_unset_budget_allows_everything() -> None:
    guard = BudgetGuard()
    client = BudgetedModelClient(inner=StubClient(), guard=guard)

    for _ in range(5):
        await client.complete(make_request())

    assert guard.limits.is_unlimited is True
    assert guard.calls == 5


def test_negative_limits_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        BudgetLimits(max_tokens=-1)


# -- caching -------------------------------------------------------------


async def test_a_repeated_request_is_served_from_cache(tmp_path: Path) -> None:
    inner = StubClient(
        ModelResponse(model=MODEL, text="first", usage=Usage(100, 20), cost_usd=0.5)
    )
    client = CachingModelClient(inner=inner, cache=ResponseCache(tmp_path))
    request = make_request()

    first = await client.complete(request)
    second = await client.complete(request)

    assert len(inner.calls) == 1
    assert second.text == first.text
    assert second.cached is True
    # The tokens were really spent once, but this call cost nothing further.
    assert second.usage.input_tokens == 100
    assert second.cost_usd == 0.0


async def test_a_different_request_is_not_a_cache_hit(tmp_path: Path) -> None:
    inner = StubClient()
    client = CachingModelClient(inner=inner, cache=ResponseCache(tmp_path))

    await client.complete(make_request())
    await client.complete(make_request(messages=(user("something else"),)))

    assert len(inner.calls) == 2


async def test_tool_calls_survive_the_cache_round_trip(tmp_path: Path) -> None:
    from evalforge.model.base import ToolInvocation

    inner = StubClient(
        ModelResponse(
            model=MODEL,
            tool_calls=(ToolInvocation(id="call-1", name="read_file", arguments={"path": "a.py"}),),
            stop_reason="tool_use",
            usage=Usage(10, 5),
        )
    )
    client = CachingModelClient(inner=inner, cache=ResponseCache(tmp_path))
    request = make_request()

    await client.complete(request)
    replayed = await client.complete(request)

    assert replayed.tool_calls[0].name == "read_file"
    assert replayed.tool_calls[0].arguments == {"path": "a.py"}
    assert replayed.stop_reason == "tool_use"


async def test_a_corrupt_cache_entry_is_a_miss_not_a_crash(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    inner = StubClient()
    client = CachingModelClient(inner=inner, cache=cache)
    request = make_request()

    key = request.cache_key(MODEL)
    path = cache.path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")

    response = await client.complete(request)

    assert response.text == "ok"
    assert len(inner.calls) == 1


def test_cache_writes_leave_no_partial_files(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    cache.put("sha256:abc123", ModelResponse(model=MODEL, text="hello"))

    assert list(tmp_path.rglob("*.tmp")) == []
    assert cache.get("sha256:abc123") is not None


# -- rate limiting -------------------------------------------------------


class FakeSchedule:
    """A clock that only moves when something sleeps on it."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.slept.append(delay)
        self.now += delay


async def test_requests_are_paced_to_the_allowance() -> None:
    schedule = FakeSchedule()
    limiter = RateLimiter(
        RateLimits(requests_per_minute=2), clock=schedule.time, sleep=schedule.sleep
    )

    await limiter.acquire()
    await limiter.acquire()
    await limiter.acquire()

    # The third request waits for the first to age out of the window.
    assert schedule.slept == [60.0]
    assert limiter.waits == 1


async def test_token_allowance_is_paced_too() -> None:
    schedule = FakeSchedule()
    limiter = RateLimiter(
        RateLimits(tokens_per_minute=1_000), clock=schedule.time, sleep=schedule.sleep
    )

    await limiter.acquire(600)
    await limiter.acquire(600)

    assert schedule.slept == [60.0]


async def test_an_unlimited_limiter_never_waits() -> None:
    schedule = FakeSchedule()
    limiter = RateLimiter(RateLimits(), clock=schedule.time, sleep=schedule.sleep)

    for _ in range(50):
        await limiter.acquire(10_000)

    assert schedule.slept == []


def test_rate_limits_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        RateLimits(requests_per_minute=0)


# -- transport -----------------------------------------------------------


def build_client(handler, **overrides: Any) -> tuple[ChatCompletionsClient, httpx.AsyncClient]:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    defaults: dict[str, Any] = {
        "base_url": "https://example.test/v1",
        "api_key": "test-key",
        "model": MODEL,
        "http": http,
        "sleep": no_sleep,
    }
    return ChatCompletionsClient(**(defaults | overrides)), http


async def test_a_plain_completion_is_parsed() -> None:
    client, http = build_client(lambda _r: httpx.Response(200, json=chat_response(content="hi")))
    async with http:
        response = await client.complete(make_request())

    assert response.text == "hi"
    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 20
    assert response.stop_reason == "end_turn"


async def test_tool_calls_are_parsed() -> None:
    payload = chat_response(
        tool_calls=[
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "solver/a.py"}'},
            }
        ],
        finish="tool_calls",
    )
    client, http = build_client(lambda _r: httpx.Response(200, json=payload))
    async with http:
        response = await client.complete(make_request())

    assert response.stop_reason == "tool_use"
    assert response.tool_calls[0].id == "call_abc"
    assert response.tool_calls[0].arguments == {"path": "solver/a.py"}
    assert response.tool_calls[0].malformed_arguments is None


async def test_malformed_tool_arguments_come_back_as_a_value() -> None:
    payload = chat_response(
        tool_calls=[
            {
                "id": "call_bad",
                "function": {"name": "write_file", "arguments": "{not valid json"},
            }
        ],
        finish="tool_calls",
    )
    client, http = build_client(lambda _r: httpx.Response(200, json=payload))
    async with http:
        response = await client.complete(make_request())

    # Small models emit this constantly; it is the agent's problem to recover
    # from, not the transport's to hide behind an exception.
    call = response.tool_calls[0]
    assert call.malformed_arguments == "{not valid json"
    assert call.arguments == {}


async def test_a_rate_limit_is_retried_honouring_retry_after() -> None:
    attempts: list[float] = []
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after": "7"}, json={"error": "slow down"})
        return httpx.Response(200, json=chat_response(content="recovered"))

    client, http = build_client(handler, sleep=record_sleep)
    async with http:
        response = await client.complete(make_request())

    assert response.text == "recovered"
    assert delays == [7.0]


async def test_server_errors_are_retried_then_surface_as_transient() -> None:
    calls: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, json={"error": {"message": "unavailable"}})

    client, http = build_client(handler, max_retries=2)
    async with http:
        with pytest.raises(TransientModelError, match="503"):
            await client.complete(make_request())

    assert len(calls) == 3


async def test_bad_credentials_fail_immediately() -> None:
    calls: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    client, http = build_client(handler, max_retries=3)
    async with http:
        with pytest.raises(PermanentModelError, match="invalid api key"):
            await client.complete(make_request())

    # Retrying a bad key just spends the quota more slowly.
    assert len(calls) == 1


async def test_the_request_carries_tools_and_auth() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=chat_response(content="ok"))

    client, http = build_client(handler)
    async with http:
        await client.complete(make_request())

    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == MODEL
    assert seen["body"]["tools"][0]["function"]["name"] == "noop"
    assert seen["body"]["tool_choice"] == "auto"


async def test_a_tool_conversation_serialises_correctly() -> None:
    from evalforge.model.base import ToolInvocation

    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=chat_response(content="done"))

    invocation = ToolInvocation(id="call-1", name="read_file", arguments={"path": "a.py"})
    client, http = build_client(handler)
    async with http:
        await client.complete(
            make_request(
                messages=(
                    user("fix it"),
                    assistant("", (invocation,)),
                    tool_result("call-1", "file contents"),
                )
            )
        )

    messages = seen["body"]["messages"]
    assert messages[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert json.loads(messages[1]["tool_calls"][0]["function"]["arguments"]) == {"path": "a.py"}
    assert messages[2] == {"role": "tool", "tool_call_id": "call-1", "content": "file contents"}


async def test_a_response_without_choices_is_transient() -> None:
    client, http = build_client(
        lambda _r: httpx.Response(200, json={"model": MODEL, "choices": []}), max_retries=0
    )
    async with http:
        with pytest.raises(TransientModelError, match="no choices"):
            await client.complete(make_request())


# -- providers -----------------------------------------------------------


def test_groq_is_the_known_provider() -> None:
    assert "groq" in provider_names()
    assert resolve_provider("groq").base_url == GROQ.base_url


def test_an_unknown_provider_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown provider"):
        resolve_provider("nope")


def test_a_missing_api_key_says_which_variable_to_set(monkeypatch) -> None:
    monkeypatch.delenv(GROQ.api_key_env, raising=False)

    with pytest.raises(ProviderNotConfiguredError, match=GROQ.api_key_env):
        resolve_provider("groq").api_key()


def test_rate_limits_can_be_tuned_by_environment(monkeypatch) -> None:
    monkeypatch.setenv("EVALFORGE_REQUESTS_PER_MINUTE", "7")

    assert resolve_provider("groq").rate_limits.requests_per_minute == 7


def test_a_nonsense_rate_limit_override_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("EVALFORGE_REQUESTS_PER_MINUTE", "many")

    with pytest.raises(ValueError, match="must be an integer"):
        resolve_provider("groq")


# -- request hashing -----------------------------------------------------


def test_the_cache_key_covers_everything_that_changes_the_answer() -> None:
    base = make_request()

    assert base.cache_key(MODEL) == make_request().cache_key(MODEL)
    assert base.cache_key(MODEL) != base.cache_key("other-model")
    assert base.cache_key(MODEL) != make_request(temperature=0.7).cache_key(MODEL)
    assert base.cache_key(MODEL) != make_request(max_tokens=512).cache_key(MODEL)
    assert base.cache_key(MODEL) != make_request(tools=()).cache_key(MODEL)


def test_input_estimation_grows_with_the_conversation() -> None:
    short_request = make_request(messages=(user("hi"),))
    long_request = make_request(messages=(user("hi" * 1000),))

    assert long_request.estimated_input_tokens() > short_request.estimated_input_tokens()
    assert short_request.estimated_input_tokens() >= 1


# -- failed generations vs genuine bad requests ---------------------------


async def test_a_failed_generation_is_reported_as_model_behaviour() -> None:
    body = {
        "error": {
            "message": "Failed to call a function. Please adjust your prompt.",
            "failed_generation": '<function=write_file>{"path": "a.py"',
        }
    }
    client, http = build_client(lambda _r: httpx.Response(400, json=body), max_retries=2)
    async with http:
        with pytest.raises(ModelBehaviourError, match="unusable tool call"):
            await client.complete(make_request())


async def test_a_failed_generation_without_the_payload_is_still_detected() -> None:
    body = {"error": {"message": "Failed to call a function. Please adjust your prompt."}}
    client, http = build_client(lambda _r: httpx.Response(400, json=body))
    async with http:
        with pytest.raises(ModelBehaviourError):
            await client.complete(make_request())


async def test_a_genuine_bad_request_is_still_permanent() -> None:
    body = {"error": {"message": "model `nonexistent` does not exist"}}
    client, http = build_client(lambda _r: httpx.Response(404, json=body))
    async with http:
        # Misreading this as the model misbehaving would hide a config error.
        with pytest.raises(PermanentModelError, match="does not exist"):
            await client.complete(make_request())


# -- learning the allowance from the provider ----------------------------


def test_the_limiter_tightens_to_a_reported_allowance() -> None:
    limiter = RateLimiter(RateLimits(tokens_per_minute=90_000))

    assert limiter.observe(tokens_per_minute=12_000) is True
    assert limiter.limits.tokens_per_minute == 12_000


def test_the_limiter_ignores_a_looser_reported_allowance() -> None:
    """A header is evidence about a ceiling, not permission to raise ours."""
    limiter = RateLimiter(RateLimits(tokens_per_minute=12_000))

    assert limiter.observe(tokens_per_minute=90_000) is False
    assert limiter.limits.tokens_per_minute == 12_000


def test_nonsense_allowances_are_ignored() -> None:
    limiter = RateLimiter(RateLimits(tokens_per_minute=12_000))

    assert limiter.observe(tokens_per_minute=0) is False
    assert limiter.observe(tokens_per_minute=None) is False


async def test_rate_limit_headers_are_adopted_from_a_live_response() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"x-ratelimit-limit-tokens": "12000"},
            json=chat_response(content="ok"),
        )

    limiter = RateLimiter(RateLimits(tokens_per_minute=90_000))
    client, http = build_client(handler, rate_limiter=limiter)
    async with http:
        await client.complete(make_request())

    # Guessing this number wrong is what made a healthy API look hung.
    assert limiter.limits.tokens_per_minute == 12_000


async def test_a_malformed_rate_limit_header_is_ignored() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"x-ratelimit-limit-tokens": "lots"}, json=chat_response(content="ok")
        )

    limiter = RateLimiter(RateLimits(tokens_per_minute=90_000))
    client, http = build_client(handler, rate_limiter=limiter)
    async with http:
        await client.complete(make_request())

    assert limiter.limits.tokens_per_minute == 90_000


# -- a spent quota is not a moment's pacing ------------------------------


async def test_a_long_retry_after_is_a_spent_quota_not_a_wait() -> None:
    """Sleeping 20 minutes inside a request is indistinguishable from a hang."""
    slept: list[float] = []

    async def record_sleep(delay: float) -> None:
        slept.append(delay)

    body = {
        "error": {
            "message": (
                "Rate limit reached on tokens per day (TPD): Limit 100000, Used 99566, "
                "Requested 1804. Please try again in 19m43.68s."
            )
        }
    }
    client, http = build_client(
        lambda _r: httpx.Response(429, headers={"retry-after": "1183"}, json=body),
        sleep=record_sleep,
    )
    async with http:
        with pytest.raises(QuotaExhaustedError, match="spent quota"):
            await client.complete(make_request())

    assert slept == [], "the client waited instead of surfacing an exhausted quota"


async def test_a_short_retry_after_is_still_honoured() -> None:
    """Per-minute pacing clears in seconds and is worth waiting out."""
    delays: list[float] = []
    attempts: list[int] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after": "3"}, json={"error": "slow down"})
        return httpx.Response(200, json=chat_response(content="ok"))

    client, http = build_client(handler, sleep=record_sleep)
    async with http:
        response = await client.complete(make_request())

    assert response.text == "ok"
    assert delays == [3.0]


async def test_an_exhausted_quota_is_permanent_so_the_scheduler_stops_retrying() -> None:
    calls: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, headers={"retry-after": "900"}, json={"error": "TPD reached"})

    client, http = build_client(handler, max_retries=4, sleep=no_sleep)
    async with http:
        with pytest.raises(PermanentModelError):
            await client.complete(make_request())

    # Rediscovering an empty quota thirty times over is not information.
    assert len(calls) == 1
