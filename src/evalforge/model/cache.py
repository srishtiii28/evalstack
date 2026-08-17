"""An on-disk cache for model responses.

Two things this buys, and the second is the one that matters here:

* **Cost and quota.** A repeated run spends nothing and consumes no rate-limit
  allowance, which on a free tier is the difference between iterating freely and
  waiting out a daily token cap.
* **Reproducibility.** Sampling is not deterministic even at temperature zero,
  so a model-driven run is not repeatable by configuration alone. Replaying
  identical requests from cache is what makes one repeatable in practice.

A cached response reports its original token counts — they were really spent,
once — but zero incremental cost, and carries ``cached=True`` so the efficiency
evaluator can tell the two apart rather than reporting a suspiciously free run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalforge.hashing import HASH_PREFIX
from evalforge.model.base import (
    ModelClient,
    ModelRequest,
    ModelResponse,
    StopReason,
    ToolInvocation,
    Usage,
)

SHARD_WIDTH = 2
_HEX = frozenset("0123456789abcdef")


class InvalidCacheKey(ValueError):
    """A key that is not a content hash, and so cannot be turned into a path."""



def _coerce_stop_reason(value: str) -> StopReason:
    """Narrow a persisted string back to the literal type, or fall back."""
    match value:
        case "end_turn" | "tool_use" | "max_tokens" | "stop_sequence" | "other":
            return value
        case _:
            return "other"


def _encode(response: ModelResponse) -> dict[str, Any]:
    return {
        "model": response.model,
        "text": response.text,
        "tool_calls": [call.describe() for call in response.tool_calls],
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "stop_reason": response.stop_reason,
        "cost_usd": response.cost_usd,
        "latency_ms": response.latency_ms,
    }


def _decode(payload: dict[str, Any]) -> ModelResponse:
    stop_reason = _coerce_stop_reason(str(payload.get("stop_reason") or "end_turn"))

    raw_usage = payload.get("usage") or {}
    tool_calls = tuple(
        ToolInvocation(
            id=str(call.get("id") or ""),
            name=str(call.get("name") or ""),
            arguments=call.get("arguments") or {},
            malformed_arguments=call.get("malformed_arguments"),
        )
        for call in payload.get("tool_calls") or []
    )

    return ModelResponse(
        model=str(payload.get("model") or ""),
        text=str(payload.get("text") or ""),
        tool_calls=tool_calls,
        usage=Usage(
            input_tokens=int(raw_usage.get("input_tokens") or 0),
            output_tokens=int(raw_usage.get("output_tokens") or 0),
        ),
        stop_reason=stop_reason,
        # A replay costs nothing further; the original spend already happened.
        cost_usd=0.0,
        latency_ms=float(payload.get("latency_ms") or 0.0),
        cached=True,
    )


class ResponseCache:
    """Content-addressed storage for responses, one JSON file per request."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.hits = 0
        self.misses = 0

    def path_for(self, key: str) -> Path:
        """Map a content hash to a file, refusing anything that is not one.

        Keys come from :meth:`ModelRequest.cache_key` and are always hex
        digests, so this never fires in practice. It exists because a path built
        by concatenating an unvalidated string is a file-write primitive waiting
        for its first careless caller, and the check costs nothing.
        """
        digest = key.removeprefix(HASH_PREFIX)
        if len(digest) < SHARD_WIDTH or not set(digest) <= _HEX:
            raise InvalidCacheKey(f"not a content hash: {key!r}")
        return self.directory / digest[:SHARD_WIDTH] / f"{digest}.json"

    def get(self, key: str) -> ModelResponse | None:
        path = self.path_for(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A corrupt entry is a cache miss, never a run-ending error.
            self.misses += 1
            return None
        if not isinstance(payload, dict):
            self.misses += 1
            return None
        self.hits += 1
        return _decode(payload)

    def put(self, key: str, response: ModelResponse) -> None:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a crash mid-write cannot leave a torn entry that
        # every later run would have to treat as corrupt.
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(_encode(response), sort_keys=True), encoding="utf-8")
        temporary.replace(path)


@dataclass(slots=True)
class CachingModelClient:
    """Serves repeated requests from disk.

    Sits outermost by design: a cache hit should consume no budget and no
    rate-limit allowance, which only holds if it short-circuits before them.
    """

    inner: ModelClient
    cache: ResponseCache
    #: Identifies the endpoint, so two providers serving the same model id do
    #: not share cache entries.
    scope: str = ""

    @property
    def model(self) -> str:
        return self.inner.model

    async def complete(self, request: ModelRequest) -> ModelResponse:
        key = request.cache_key(self.model, self.scope)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        response = await self.inner.complete(request)
        self.cache.put(key, response)
        return response
