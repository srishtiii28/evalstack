"""Transport for the chat-completions HTTP API.

This is not a compatibility shim layered over a vendor SDK — it *is* the
Groq client. Groq serves this format directly, at ``.../openai/v1``, as do
OpenRouter, GitHub Models, Together and a local Ollama. Speaking the wire
protocol rather than importing a vendor package keeps ``Retry-After`` handling
and rate-limit pacing under our control, which is the part that matters on a
free tier, and costs one dependency instead of one per provider.

Two behaviours here are load-bearing rather than incidental:

* **Malformed tool arguments are a value, not an exception.** Models — smaller
  ones especially — emit arguments that are not valid JSON. That is the agent's
  problem to recover from and the evaluator's to measure, so it travels back as
  a field on the invocation instead of blowing up the transport.
* **Retries distinguish transient from permanent.** A 429 or 5xx is retried with
  backoff, honouring ``Retry-After``. A 401 or 400 is raised immediately;
  retrying bad credentials just spends the quota more slowly.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from pydantic import JsonValue

from evalforge.model.base import (
    Message,
    ModelBehaviourError,
    ModelRequest,
    ModelResponse,
    PermanentModelError,
    QuotaExhaustedError,
    StopReason,
    ToolInvocation,
    ToolSpec,
    TransientModelError,
    Usage,
)
from evalforge.model.pricing import PricingTable
from evalforge.model.rate_limit import RateLimiter

CHAT_COMPLETIONS_PATH = "/chat/completions"
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_RETRIES = 4
DEFAULT_BACKOFF_S = 1.0
MAX_BACKOFF_S = 30.0

#: Longest ``Retry-After`` worth honouring. Beyond this the provider is telling
#: you a quota is gone, not that you were briefly too quick, and waiting it out
#: inside a request looks exactly like a hung process.
MAX_RETRY_AFTER_S = 120.0

#: Provider finish reasons, normalised. Anything unrecognised becomes "other"
#: rather than being guessed at.
_STOP_REASONS: dict[str, StopReason] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
    "content_filter": "other",
}

_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: How much of a rejected generation to keep. Generous on purpose: this is the
#: only evidence of why an attempt failed, and 400 characters proved too few to
#: see where the model's JSON escaping went wrong.
FAILED_GENERATION_CAPTURE_CHARS = 1200

#: Markers that a 4xx describes the model's generation rather than the request.
_FAILED_GENERATION_MARKERS = ("failed_generation", "failed to call a function")


def _failed_generation(response: httpx.Response) -> str | None:
    """Return the offending generation when a 4xx blames the model's output."""
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None

    error = body.get("error")
    error_map = error if isinstance(error, dict) else {}
    generation = error_map.get("failed_generation") or body.get("failed_generation")
    if generation:
        return str(generation)[:FAILED_GENERATION_CAPTURE_CHARS]

    message = str(error_map.get("message") or error or "").lower()
    if any(marker in message for marker in _FAILED_GENERATION_MARKERS):
        return "(provider reported a failed generation without returning it)"
    return None


def _tool_payload(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _message_payload(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }

    if message.role == "assistant" and message.tool_calls:
        return {
            "role": "assistant",
            # Providers reject an empty string here but accept null.
            "content": message.content or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, sort_keys=True),
                    },
                }
                for call in message.tool_calls
            ],
        }

    return {"role": message.role, "content": message.content}


def _parse_tool_calls(raw_calls: list[dict[str, Any]]) -> tuple[ToolInvocation, ...]:
    invocations: list[ToolInvocation] = []
    for index, raw in enumerate(raw_calls):
        function = raw.get("function") or {}
        name = function.get("name") or ""
        raw_arguments = function.get("arguments")

        arguments: dict[str, JsonValue] = {}
        malformed: str | None = None
        if isinstance(raw_arguments, dict):
            arguments = raw_arguments
        elif isinstance(raw_arguments, str) and raw_arguments.strip():
            try:
                decoded = json.loads(raw_arguments)
            except json.JSONDecodeError:
                malformed = raw_arguments
            else:
                if isinstance(decoded, dict):
                    arguments = decoded
                else:
                    malformed = raw_arguments

        invocations.append(
            ToolInvocation(
                id=str(raw.get("id") or f"call-{index + 1}"),
                name=name,
                arguments=arguments,
                malformed_arguments=malformed,
            )
        )
    return tuple(invocations)


def _retry_after_seconds(response: httpx.Response) -> float | None:
    header = response.headers.get("retry-after")
    if not header:
        return None
    try:
        return max(0.0, float(header))
    except ValueError:
        # The HTTP-date form is legal but providers rarely use it; treating it
        # as absent falls back to exponential backoff rather than guessing.
        return None


class ChatCompletionsClient:
    """A chat-completions client. Groq by default; any provider serving this API."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        http: httpx.AsyncClient,
        pricing: PricingTable | None = None,
        rate_limiter: RateLimiter | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._http = http
        self._pricing = pricing or PricingTable()
        self._rate_limiter = rate_limiter
        self._max_retries = max_retries
        self._timeout_s = timeout_s
        self._sleep = sleep

    @property
    def model(self) -> str:
        return self._model

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [_message_payload(message) for message in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.tools:
            payload["tools"] = [_tool_payload(tool) for tool in request.tools]
            payload["tool_choice"] = "auto"
        return payload

    async def complete(self, request: ModelRequest) -> ModelResponse:
        estimated_tokens = request.estimated_input_tokens() + request.max_tokens
        started = time.monotonic()
        data = await self._post_with_retries(self._payload(request), estimated_tokens)
        latency_ms = (time.monotonic() - started) * 1000.0
        return self._to_response(data, latency_ms)

    async def _post_with_retries(
        self, payload: dict[str, Any], estimated_tokens: int
    ) -> dict[str, Any]:
        url = f"{self._base_url}{CHAT_COMPLETIONS_PATH}"
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            # Every attempt is paced, not just the first. A retried request costs
            # the provider's allowance exactly as much as the original, so pacing
            # only the first one lets a burst of 429s spend quota unaccounted for.
            if self._rate_limiter is not None:
                await self._rate_limiter.acquire(estimated_tokens)
            try:
                response = await self._http.post(
                    url, json=payload, headers=self._headers(), timeout=self._timeout_s
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = TransientModelError(f"{type(exc).__name__}: {exc}")
            else:
                self._learn_limits(response)
                if response.status_code < 400:
                    return self._decode(response)

                detail = _error_detail(response)
                if response.status_code in _RETRYABLE_STATUS:
                    retry_after = _retry_after_seconds(response)
                    if retry_after is not None and retry_after > MAX_RETRY_AFTER_S:
                        raise QuotaExhaustedError(
                            f"{self._base_url} asks for a {retry_after:.0f}s wait, which is a "
                            f"spent quota rather than a moment's pacing: {detail}"
                        )
                    last_error = TransientModelError(
                        f"HTTP {response.status_code} from {self._base_url}: {detail}"
                    )
                    if attempt < self._max_retries:
                        await self._sleep(retry_after or _backoff_for(attempt))
                        continue
                else:
                    generation = _failed_generation(response)
                    if generation is not None:
                        raise ModelBehaviourError(
                            f"the model produced an unusable tool call "
                            f"(HTTP {response.status_code}): {generation}"
                        )
                    raise PermanentModelError(
                        f"HTTP {response.status_code} from {self._base_url}: {detail}"
                    )

            if attempt < self._max_retries:
                await self._sleep(_backoff_for(attempt))

        raise last_error if last_error is not None else TransientModelError("request failed")

    def _learn_limits(self, response: httpx.Response) -> None:
        if self._rate_limiter is None:
            return
        raw = response.headers.get("x-ratelimit-limit-tokens")
        if not raw:
            return
        try:
            self._rate_limiter.observe(tokens_per_minute=int(raw))
        except ValueError:
            return

    @staticmethod
    def _decode(response: httpx.Response) -> dict[str, Any]:
        try:
            decoded = response.json()
        except ValueError as exc:
            raise TransientModelError(f"response was not JSON: {exc}") from exc
        if not isinstance(decoded, dict):
            raise TransientModelError("response JSON was not an object")
        return decoded

    def _to_response(self, data: dict[str, Any], latency_ms: float) -> ModelResponse:
        choices = data.get("choices") or []
        if not choices:
            raise TransientModelError("response contained no choices")

        choice = choices[0]
        message = choice.get("message") or {}
        raw_calls = message.get("tool_calls") or []
        tool_calls = _parse_tool_calls(raw_calls if isinstance(raw_calls, list) else [])

        raw_usage = data.get("usage") or {}
        usage = Usage(
            input_tokens=int(raw_usage.get("prompt_tokens") or 0),
            output_tokens=int(raw_usage.get("completion_tokens") or 0),
        )

        finish_reason = str(choice.get("finish_reason") or "")
        stop_reason: StopReason = _STOP_REASONS.get(finish_reason, "other")
        if stop_reason == "other" and tool_calls:
            # Some providers omit or mislabel the reason when returning tools.
            stop_reason = "tool_use"

        model = str(data.get("model") or self._model)
        return ModelResponse(
            model=model,
            text=str(message.get("content") or ""),
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=stop_reason,
            cost_usd=self._pricing.cost_for(model, usage),
            latency_ms=latency_ms,
        )


def _backoff_for(attempt: int) -> float:
    return min(MAX_BACKOFF_S, DEFAULT_BACKOFF_S * (2.0**attempt))


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
    return str(body)[:200]
