"""Provider configuration and client assembly.

A provider is data — a base URL, the environment variable holding its key, a
default model, and its published per-minute allowances. The transport is the
same for all of them, so adding one is a table entry rather than a code path.

Assembly order is deliberate and is the part worth reading:

    cache → budget → rate limiter → HTTP

A cache hit must cost nothing, so the cache sits outermost and short-circuits
before the budget is charged or the rate-limit allowance is spent. Putting it
underneath would make replayed runs consume quota they do not use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import httpx

from evalforge.model.base import ModelClient
from evalforge.model.budget import BudgetedModelClient, BudgetGuard, BudgetLimits
from evalforge.model.cache import CachingModelClient, ResponseCache
from evalforge.model.chat_completions import ChatCompletionsClient
from evalforge.model.pricing import PricingTable
from evalforge.model.rate_limit import RateLimiter, RateLimits
from evalforge.paths import DEFAULT_CACHE_DIR


class ProviderNotConfiguredError(Exception):
    """The provider needs an API key that is not present in the environment."""


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str
    default_model: str
    #: Conservative defaults. Free-tier allowances change and differ per model,
    #: so treat these as a starting point and tune with EVALFORGE_RPM/TPM.
    rate_limits: RateLimits = field(default_factory=RateLimits)

    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise ProviderNotConfiguredError(
                f"{self.name} needs an API key in ${self.api_key_env}; "
                f"export {self.api_key_env}=... and retry"
            )
        return key


GROQ = ProviderConfig(
    name="groq",
    base_url="https://api.groq.com/openai/v1",
    api_key_env="GROQ_API_KEY",
    default_model="llama-3.3-70b-versatile",
    rate_limits=RateLimits(requests_per_minute=25, tokens_per_minute=90_000),
)

PROVIDERS: dict[str, ProviderConfig] = {GROQ.name: GROQ}

RPM_ENV_VAR = "EVALFORGE_REQUESTS_PER_MINUTE"
TPM_ENV_VAR = "EVALFORGE_TOKENS_PER_MINUTE"


def provider_names() -> tuple[str, ...]:
    return tuple(sorted(PROVIDERS))


def resolve_provider(name: str) -> ProviderConfig:
    try:
        config = PROVIDERS[name]
    except KeyError:
        known = ", ".join(provider_names())
        raise KeyError(f"unknown provider {name!r}; known providers: {known}") from None
    return replace(config, rate_limits=_rate_limits_from_environment(config.rate_limits))


def _positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


def _rate_limits_from_environment(defaults: RateLimits) -> RateLimits:
    return RateLimits(
        requests_per_minute=_positive_int_env(RPM_ENV_VAR) or defaults.requests_per_minute,
        tokens_per_minute=_positive_int_env(TPM_ENV_VAR) or defaults.tokens_per_minute,
    )


def build_model_client(
    *,
    provider: str = GROQ.name,
    model: str | None = None,
    http: httpx.AsyncClient,
    budget: BudgetLimits | None = None,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    pricing: PricingTable | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[ModelClient, BudgetGuard]:
    """Assemble a client for ``provider``, returning it with its budget guard.

    The guard is handed back rather than hidden so the caller can report what a
    run actually consumed, which is the point of tracking it at all.
    """
    config = resolve_provider(provider)
    table = pricing or PricingTable()

    transport = ChatCompletionsClient(
        base_url=config.base_url,
        api_key=config.api_key(),
        model=model or config.default_model,
        http=http,
        pricing=table,
        rate_limiter=rate_limiter or RateLimiter(config.rate_limits),
    )

    guard = BudgetGuard(limits=budget or BudgetLimits())
    client: ModelClient = BudgetedModelClient(inner=transport, guard=guard, pricing=table)

    if cache_dir is not None:
        client = CachingModelClient(
            inner=client, cache=ResponseCache(cache_dir), scope=config.base_url
        )

    return client, guard
