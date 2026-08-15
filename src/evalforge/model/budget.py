"""Resource limits that refuse before the resource is gone.

The property worth keeping is *pre-flight refusal*: reject the call that would
cross a ceiling rather than reporting the overrun afterwards. Post-hoc
accounting on a runaway agent loop tells you what you lost, which is not a
control.

What is scarce depends on the provider. On a free tier nothing is billed, so a
dollar ceiling guards nothing — the binding limits are the daily token quota and
an agent that loops without ever submitting. So the budget is denominated in
whichever axes the caller sets: model calls, tokens, and optionally dollars.
Unset axes are unlimited.

Because output length is unknown before a call, estimates assume the worst case:
the full ``max_tokens`` of output. The guard is therefore conservative — it may
refuse a call that would have fitted, but never admits one that would not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evalforge.model.base import (
    ModelClient,
    ModelError,
    ModelRequest,
    ModelResponse,
    Usage,
)
from evalforge.model.pricing import PricingTable


class BudgetExceeded(ModelError):
    """The next call would cross a ceiling, so it was not made."""


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    """Ceilings for one budgeted scope. ``None`` means unlimited."""

    max_calls: int | None = None
    max_tokens: int | None = None
    max_usd: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_calls", self.max_calls),
            ("max_tokens", self.max_tokens),
            ("max_usd", self.max_usd),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative")

    @property
    def is_unlimited(self) -> bool:
        return self.max_calls is None and self.max_tokens is None and self.max_usd is None


@dataclass(slots=True)
class BudgetGuard:
    """Tracks consumption against :class:`BudgetLimits` for one scope.

    Scope is the caller's choice. One guard per case attempt bounds a runaway
    loop; a single guard shared across a run bounds a daily quota.
    """

    limits: BudgetLimits = field(default_factory=BudgetLimits)
    calls: int = 0
    tokens: int = 0
    spent_usd: float = 0.0
    refusals: int = 0

    def check(self, *, estimated_tokens: int = 0, estimated_usd: float = 0.0) -> None:
        """Raise :class:`BudgetExceeded` if the next call would cross a ceiling."""
        reason = self._breach(estimated_tokens=estimated_tokens, estimated_usd=estimated_usd)
        if reason is not None:
            self.refusals += 1
            raise BudgetExceeded(reason)

    def _breach(self, *, estimated_tokens: int, estimated_usd: float) -> str | None:
        limits = self.limits
        if limits.max_calls is not None and self.calls + 1 > limits.max_calls:
            return f"call {self.calls + 1} would exceed the limit of {limits.max_calls} calls"
        if limits.max_tokens is not None and self.tokens + estimated_tokens > limits.max_tokens:
            return (
                f"call would use about {estimated_tokens} tokens, but only "
                f"{max(0, limits.max_tokens - self.tokens)} of {limits.max_tokens} remain"
            )
        if limits.max_usd is not None and self.spent_usd + estimated_usd > limits.max_usd:
            return (
                f"call would cost about ${estimated_usd:.4f}, but only "
                f"${max(0.0, limits.max_usd - self.spent_usd):.4f} of "
                f"${limits.max_usd:.4f} remains"
            )
        return None

    def record(self, usage: Usage, cost_usd: float) -> None:
        self.calls += 1
        self.tokens += usage.total
        self.spent_usd += cost_usd

    def describe(self) -> str:
        parts = [f"{self.calls} calls", f"{self.tokens} tokens"]
        if self.spent_usd:
            parts.append(f"${self.spent_usd:.4f}")
        return ", ".join(parts)


@dataclass(slots=True)
class BudgetedModelClient:
    """Wraps a client so every call is checked against a :class:`BudgetGuard`."""

    inner: ModelClient
    guard: BudgetGuard
    pricing: PricingTable = field(default_factory=PricingTable)

    @property
    def model(self) -> str:
        return self.inner.model

    def _worst_case(self, request: ModelRequest) -> Usage:
        return Usage(
            input_tokens=request.estimated_input_tokens(),
            output_tokens=request.max_tokens,
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        worst_case = self._worst_case(request)
        self.guard.check(
            estimated_tokens=worst_case.total,
            estimated_usd=self.pricing.cost_for(self.model, worst_case),
        )
        response = await self.inner.complete(request)
        self.guard.record(response.usage, response.cost_usd)
        return response
