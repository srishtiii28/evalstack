"""Capability versus reliability, from repeated samples of the same task.

An agent that solves a task one time in ten and an agent that solves it ten
times in ten have the same success rate at k=1 and are not the same product.
Two estimators separate them:

* ``pass@k`` — at least one of k attempts succeeds. Capability, and the right
  metric when a human reviews the output and can ask again.
* ``pass^k`` — all k attempts succeed. Reliability, and the right metric for
  anything running unattended.

Both are computed as unbiased estimators over the ``n`` samples actually run,
rather than by simulating draws. The naive alternative — "did any of my n
samples pass?" — is biased upward and gets worse as n grows, which makes it
useless for comparing runs that used different sample counts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import comb


def pass_at_k(samples: int, correct: int, k: int) -> float:
    """Unbiased estimate that at least one of ``k`` attempts succeeds.

    ``1 - C(n-c, k) / C(n, k)``: one minus the probability that a random
    k-subset of the n samples contains no successes.
    """
    _validate(samples, correct, k)
    failures = samples - correct
    if failures < k:
        # Every k-subset must contain at least one success.
        return 1.0
    return 1.0 - comb(failures, k) / comb(samples, k)


def pass_hat_k(samples: int, correct: int, k: int) -> float:
    """Unbiased estimate that *all* ``k`` attempts succeed.

    ``C(c, k) / C(n, k)``: the probability that a random k-subset of the n
    samples is drawn entirely from the successes.
    """
    _validate(samples, correct, k)
    if correct < k:
        return 0.0
    return comb(correct, k) / comb(samples, k)


def _validate(samples: int, correct: int, k: int) -> None:
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if not 0 <= correct <= samples:
        raise ValueError("correct must be between 0 and samples")
    if k < 1:
        raise ValueError("k must be at least 1")
    if k > samples:
        raise ValueError(f"cannot estimate k={k} from only {samples} samples")


@dataclass(frozen=True, slots=True)
class StabilityReport:
    """Capability and reliability across a dataset, at one value of k."""

    k: int
    pass_at_k: float
    pass_hat_k: float
    cases: int
    #: Cases that passed at least once but not every time.
    flaky_cases: int

    @property
    def reliability_gap(self) -> float:
        """How much capability does not survive the demand for consistency."""
        return self.pass_at_k - self.pass_hat_k


def stability_report(tallies: Mapping[str, tuple[int, int]], *, k: int) -> StabilityReport:
    """Aggregate per-case ``(passed, total)`` counts into a report at ``k``.

    Cases with fewer than ``k`` samples are skipped rather than extrapolated:
    an estimate of pass@5 from three samples is not an estimate, it is a guess.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    at_k: list[float] = []
    hat_k: list[float] = []
    flaky = 0
    for passed, total in tallies.values():
        if total < k:
            continue
        at_k.append(pass_at_k(total, passed, k))
        hat_k.append(pass_hat_k(total, passed, k))
        if 0 < passed < total:
            flaky += 1

    count = len(at_k)
    return StabilityReport(
        k=k,
        pass_at_k=sum(at_k) / count if count else 0.0,
        pass_hat_k=sum(hat_k) / count if count else 0.0,
        cases=count,
        flaky_cases=flaky,
    )


def max_usable_k(tallies: Iterable[tuple[int, int]]) -> int:
    """The largest ``k`` every case has enough samples to support."""
    totals = [total for _, total in tallies]
    return min(totals) if totals else 0
