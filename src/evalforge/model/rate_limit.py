"""Pacing requests so a free tier does not spend the run answering 429s.

A concurrent evaluation harness will happily fire twenty requests at once, which
on a free tier means nineteen rejections and a stalled run. Two mechanisms cover
the two ways you hit a limit:

* :class:`RateLimiter` paces *proactively* against a requests-per-minute and
  tokens-per-minute allowance, so most calls never get rejected at all.
* The transport retries *reactively* on a 429, honouring ``Retry-After`` when
  the provider sends it — see :mod:`evalforge.model.chat_completions`.

Proactive pacing is the important half. Backoff alone converges on a state where
every request is tried, rejected, and retried, which wastes the quota that the
rejections are supposed to protect.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

WINDOW_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class RateLimits:
    """Per-minute allowances. ``None`` means unlimited."""

    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("requests_per_minute", self.requests_per_minute),
            ("tokens_per_minute", self.tokens_per_minute),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1 when set")

    @property
    def is_unlimited(self) -> bool:
        return self.requests_per_minute is None and self.tokens_per_minute is None


class RateLimiter:
    """A sliding-window limiter shared by every caller of one provider.

    Sliding window rather than a token bucket because providers publish limits
    as "N per minute" and enforce them the same way; matching their accounting
    avoids tripping a limit the local model of it thinks is fine.
    """

    def __init__(
        self,
        limits: RateLimits,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._limits = limits
        self._clock = clock
        self._sleep = sleep
        self._events: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()
        self.waits = 0
        self.total_wait_s = 0.0

    @property
    def limits(self) -> RateLimits:
        return self._limits

    def observe(self, *, tokens_per_minute: int | None = None) -> bool:
        """Adopt an allowance the provider reported, when it is tighter than ours.

        Providers return their real limits on every response. Guessing instead
        is how this pacer ended up configured at 7.5x the true token rate, which
        turned every burst into a wall of 429s and made attempts look hung.

        Only tightening is applied: a header is evidence about a ceiling, and
        loosening on one risks overrunning a limit that other clients share.
        """
        if tokens_per_minute is None or tokens_per_minute < 1:
            return False
        current = self._limits.tokens_per_minute
        if current is not None and current <= tokens_per_minute:
            return False
        self._limits = RateLimits(
            requests_per_minute=self._limits.requests_per_minute,
            tokens_per_minute=tokens_per_minute,
        )
        return True

    def _prune(self, now: float) -> None:
        cutoff = now - WINDOW_SECONDS
        while self._events and self._events[0][0] <= cutoff:
            self._events.popleft()

    def _wait_needed(self, now: float, tokens: int) -> float:
        """Seconds to wait before a call of ``tokens`` would fit in the window."""
        limits = self._limits
        waits: list[float] = []

        request_cap = limits.requests_per_minute
        if request_cap is not None and len(self._events) >= request_cap:
            # Wait for the oldest request in excess of the allowance to age out.
            index = len(self._events) - request_cap
            waits.append(self._events[index][0] + WINDOW_SECONDS - now)

        if limits.tokens_per_minute is not None:
            used = sum(count for _, count in self._events)
            if used + tokens > limits.tokens_per_minute:
                # Drop oldest events until the request would fit.
                freed = 0
                for timestamp, count in self._events:
                    freed += count
                    if used - freed + tokens <= limits.tokens_per_minute:
                        waits.append(timestamp + WINDOW_SECONDS - now)
                        break
                else:
                    # Even an empty window cannot fit this request; let it
                    # through rather than deadlock, and let the provider judge.
                    waits.append(0.0)

        return max([0.0, *waits])

    async def acquire(self, estimated_tokens: int = 0) -> None:
        """Block until a call of this size fits inside the allowance."""
        if self._limits.is_unlimited:
            return

        async with self._lock:
            while True:
                now = self._clock()
                self._prune(now)
                wait = self._wait_needed(now, estimated_tokens)
                if wait <= 0:
                    self._events.append((now, estimated_tokens))
                    return
                self.waits += 1
                self.total_wait_s += wait
                await self._sleep(wait)
