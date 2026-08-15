"""Constructing a model-driven agent, with the right things shared.

What is shared across concurrent attempts, and what is not, is the whole content
of this module:

* **Shared**: the HTTP connection pool, the rate limiter, and the budget guard.
  A per-attempt rate limiter would let twenty concurrent cases each believe they
  were within the allowance while collectively being far outside it — the exact
  mistake that turns a free tier into a wall of 429s.
* **Not shared**: the agent object and its conversation. Each attempt gets its
  own, because a shared message history would leak one case's work into another.

That is why the factory is a context manager handing back a *callable*: the
scheduler calls it once per attempt, and the expensive shared machinery is built
once and torn down cleanly at the end.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from evalforge.agent.base import Agent
from evalforge.agent.model_agent import ModelAgent, ModelAgentConfig
from evalforge.model.budget import BudgetGuard, BudgetLimits
from evalforge.model.providers import GROQ, build_model_client, resolve_provider
from evalforge.model.rate_limit import RateLimiter
from evalforge.paths import DEFAULT_CACHE_DIR

MODEL_PREFIX = "model"
DEFAULT_HTTP_POOL = 16


@dataclass(frozen=True, slots=True)
class ModelAgentSpec:
    """Everything needed to build a model agent, resolved from CLI flags."""

    provider: str = GROQ.name
    model: str | None = None
    settings: ModelAgentConfig = field(default_factory=ModelAgentConfig)
    budget: BudgetLimits = field(default_factory=BudgetLimits)
    cache_dir: Path | None = DEFAULT_CACHE_DIR

    @classmethod
    def from_reference(
        cls,
        reference: str,
        *,
        provider: str = GROQ.name,
        settings: ModelAgentConfig | None = None,
        budget: BudgetLimits | None = None,
        cache_dir: Path | None = DEFAULT_CACHE_DIR,
    ) -> ModelAgentSpec:
        """Parse ``model`` or ``model:<model-id>``."""
        scheme, separator, remainder = reference.partition(":")
        if scheme != MODEL_PREFIX:
            raise ValueError(f"not a model agent reference: {reference!r}")
        return cls(
            provider=provider,
            model=remainder.strip() if separator and remainder.strip() else None,
            settings=settings or ModelAgentConfig(),
            budget=budget or BudgetLimits(),
            cache_dir=cache_dir,
        )


@contextlib.asynccontextmanager
async def model_agent_factory(
    spec: ModelAgentSpec,
) -> AsyncIterator[tuple[Callable[[], Agent], BudgetGuard]]:
    """Yield a per-attempt agent factory and the shared budget guard."""
    config = resolve_provider(spec.provider)
    limiter = RateLimiter(config.rate_limits)

    limits = httpx.Limits(
        max_connections=DEFAULT_HTTP_POOL, max_keepalive_connections=DEFAULT_HTTP_POOL
    )
    async with httpx.AsyncClient(limits=limits) as http:
        client, guard = build_model_client(
            provider=spec.provider,
            model=spec.model,
            http=http,
            budget=spec.budget,
            cache_dir=spec.cache_dir,
            rate_limiter=limiter,
        )

        def make_agent() -> Agent:
            # A fresh agent per attempt; the client beneath it is stateless and
            # deliberately shared.
            return ModelAgent(client=client, settings=spec.settings)

        yield make_agent, guard
