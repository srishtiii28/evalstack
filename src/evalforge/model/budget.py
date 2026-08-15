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
the full ``max_tokens`` of output. Input is estimated from character count, not
tokenised, so it can undershoot.

Be precise about what that buys, because the tempting claim is stronger than the
truth. The guard guarantees it never *admits* a call once committed usage —
settled plus reserved — has reached the ceiling. It cannot guarantee actual
consumption stays under it: if an input estimate is low, the settled figure can
overshoot by that error, multiplied by however many calls were in flight.
Overshoot is therefore bounded and self-correcting rather than eliminated, since
the next reservation sees the true settled total and refuses.

Exact enforcement would need a per-provider tokeniser on the request path. That
is a real cost for a bound that is already tight enough to stop a runaway loop,
which is what the ceiling is for.
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


@dataclass(frozen=True, slots=True)
class Reservation:
    """Capacity held for a call that is in flight."""

    tokens: int
    usd: float


@dataclass(slots=True)
class BudgetGuard:
    """Tracks consumption against :class:`BudgetLimits` for one scope.

    Scope is the caller's choice. One guard per case attempt bounds a runaway
    loop; a single guard shared across a run bounds a daily quota.

    Capacity is **reserved** at the moment a call is admitted and settled when it
    returns. Checking and then recording after the round trip would let every
    concurrent attempt pass the check before any of them recorded anything — with
    a shared guard and ten attempts in flight, that is a tenfold overrun of a
    ceiling that reported itself as enforced. :meth:`reserve` performs its check
    and its bookkeeping with no ``await`` between them, so on an event loop it is
    atomic without needing a lock.
    """

    limits: BudgetLimits = field(default_factory=BudgetLimits)
    calls: int = 0
    tokens: int = 0
    spent_usd: float = 0.0
    refusals: int = 0
    in_flight: int = 0
    _reserved_tokens: int = 0
    _reserved_usd: float = 0.0

    @property
    def committed_tokens(self) -> int:
        """Tokens spent plus tokens held for calls that have not returned."""
        return self.tokens + self._reserved_tokens

    @property
    def committed_usd(self) -> float:
        return self.spent_usd + self._reserved_usd

    def reserve(self, *, estimated_tokens: int = 0, estimated_usd: float = 0.0) -> Reservation:
        """Hold capacity for one call, or refuse it.

        Must not await: the check and the reservation have to land in the same
        event-loop step for the ceiling to hold under concurrency.
        """
        reason = self._breach(estimated_tokens=estimated_tokens, estimated_usd=estimated_usd)
        if reason is not None:
            self.refusals += 1
            raise BudgetExceeded(reason)

        self._reserved_tokens += estimated_tokens
        self._reserved_usd += estimated_usd
        self.in_flight += 1
        return Reservation(tokens=estimated_tokens, usd=estimated_usd)

    def _breach(self, *, estimated_tokens: int, estimated_usd: float) -> str | None:
        limits = self.limits
        attempted_call = self.calls + self.in_flight + 1
        if limits.max_calls is not None and attempted_call > limits.max_calls:
            return f"call {attempted_call} would exceed the limit of {limits.max_calls} calls"
        if (
            limits.max_tokens is not None
            and self.committed_tokens + estimated_tokens > limits.max_tokens
        ):
            return (
                f"call would use about {estimated_tokens} tokens, but only "
                f"{max(0, limits.max_tokens - self.committed_tokens)} of "
                f"{limits.max_tokens} remain"
            )
        if limits.max_usd is not None and self.committed_usd + estimated_usd > limits.max_usd:
            return (
                f"call would cost about ${estimated_usd:.4f}, but only "
                f"${max(0.0, limits.max_usd - self.committed_usd):.4f} of "
                f"${limits.max_usd:.4f} remains"
            )
        return None

    def _release(self, reservation: Reservation) -> None:
        self._reserved_tokens = max(0, self._reserved_tokens - reservation.tokens)
        self._reserved_usd = max(0.0, self._reserved_usd - reservation.usd)
        self.in_flight = max(0, self.in_flight - 1)

    def settle(self, reservation: Reservation, usage: Usage, cost_usd: float) -> None:
        """Replace a reservation with what the call actually consumed."""
        self._release(reservation)
        self.calls += 1
        self.tokens += usage.total
        self.spent_usd += cost_usd

    def abandon(self, reservation: Reservation) -> None:
        """Release capacity for a call that failed and consumed nothing."""
        self._release(reservation)

    def describe(self) -> str:
        parts = [f"{self.calls} calls", f"{self.tokens} tokens"]
        if self.spent_usd:
            parts.append(f"${self.spent_usd:.4f}")
        return ", ".join(parts)


@dataclass(slots=True)
class BudgetedModelClient:
    """Wraps a client so every call reserves capacity before it is made."""

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
        reservation = self.guard.reserve(
            estimated_tokens=worst_case.total,
            estimated_usd=self.pricing.cost_for(self.model, worst_case),
        )
        try:
            response = await self.inner.complete(request)
        except BaseException:
            # A call that never returned consumed no quota we can account for.
            self.guard.abandon(reservation)
            raise
        self.guard.settle(reservation, response.usage, response.cost_usd)
        return response
