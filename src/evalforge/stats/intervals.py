"""Confidence intervals: how much of a number is real.

A success rate quoted without an interval invites a decision the data cannot
support. Thirty tasks at 70% carries roughly ±16 points at 95% confidence, so a
four-point move between versions is indistinguishable from noise — and shipping
on it is a coin flip wearing a lab coat.

Two intervals, for two different questions:

* :func:`wilson_interval` — how precise is *this* rate?
* :func:`paired_bootstrap` — how big is the difference between two runs, given
  that both ran the same tasks?
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

DEFAULT_CONFIDENCE = 0.95
DEFAULT_RESAMPLES = 10_000


def _z_for(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    return NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)


@dataclass(frozen=True, slots=True)
class Interval:
    """An estimate with a confidence interval around it."""

    estimate: float
    low: float
    high: float
    confidence: float = DEFAULT_CONFIDENCE

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def excludes_zero(self) -> bool:
        """True when the whole interval sits on one side of zero."""
        return self.low > 0.0 or self.high < 0.0

    def format(self, *, as_percent: bool = True) -> str:
        scale = 100.0 if as_percent else 1.0
        unit = "%" if as_percent else ""
        return (
            f"{self.estimate * scale:.1f}{unit} "
            f"[{self.low * scale:.1f}, {self.high * scale:.1f}]"
        )


def wilson_interval(
    successes: int, trials: int, *, confidence: float = DEFAULT_CONFIDENCE
) -> Interval:
    """Wilson score interval for a success rate.

    Preferred over the textbook normal approximation because it stays inside
    [0, 1] and keeps its coverage near 0, near 1, and at small n — which is
    exactly where evaluation suites live. The normal approximation famously
    produces intervals extending below zero for a rate of 0/20.
    """
    if trials < 0 or successes < 0:
        raise ValueError("counts must not be negative")
    if successes > trials:
        raise ValueError("successes cannot exceed trials")
    if trials == 0:
        return Interval(estimate=0.0, low=0.0, high=1.0, confidence=confidence)

    z = _z_for(confidence)
    n = float(trials)
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    margin = (z / denominator) * ((p * (1.0 - p) / n + z * z / (4.0 * n * n)) ** 0.5)

    return Interval(
        estimate=p,
        low=max(0.0, centre - margin),
        high=min(1.0, centre + margin),
        confidence=confidence,
    )


def paired_bootstrap(
    before: Sequence[float],
    after: Sequence[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Interval:
    """Bootstrap a confidence interval for the mean difference ``after - before``.

    Resampling is **paired**: each draw takes a case and both of its outcomes,
    never one outcome from case 3 and the other from case 7. Both versions ran
    the same tasks, so pairing conditions on task difficulty and removes the
    variance that comes from some tasks simply being harder — which is most of
    the variance in any real suite.

    Non-parametric on purpose: per-case outcomes are usually 0/1, so assuming
    normality would be assuming away the actual distribution.
    """
    if len(before) != len(after):
        raise ValueError("paired samples must be the same length")
    if not before:
        raise ValueError("cannot bootstrap an empty sample")
    if resamples < 1:
        raise ValueError("resamples must be at least 1")

    differences = [b - a for a, b in zip(before, after, strict=True)]
    observed = sum(differences) / len(differences)

    rng = random.Random(seed)
    size = len(differences)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(size):
            total += differences[rng.randrange(size)]
        means.append(total / size)
    means.sort()

    tail = (1.0 - confidence) / 2.0
    low = means[_percentile_index(len(means), tail)]
    high = means[_percentile_index(len(means), 1.0 - tail)]
    return Interval(estimate=observed, low=low, high=high, confidence=confidence)


def _percentile_index(count: int, quantile: float) -> int:
    """Index into a sorted sample for ``quantile``, clamped to the sample."""
    index = round(quantile * (count - 1))
    return max(0, min(count - 1, index))
