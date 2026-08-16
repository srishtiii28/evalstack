"""Is a difference real, and how many tasks would it take to find out?

:func:`mcnemar` answers the first question for paired binary outcomes — the
shape every agent comparison has, because both versions run the same tasks.
:func:`required_sample_size` answers the second, which is the more useful one
in practice: it turns "is this significant?" into "how big does my dataset have
to be before it *could* be?"
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import comb, sqrt
from statistics import NormalDist

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80


@dataclass(frozen=True, slots=True)
class PairedCounts:
    """The two-by-two table of paired outcomes.

    Only the disagreements carry information. Cases both versions passed, or
    both failed, say nothing about which is better — which is precisely why a
    paired test is so much more sensitive than comparing two overall rates.
    """

    both_passed: int
    only_before_passed: int
    only_after_passed: int
    both_failed: int

    @property
    def discordant(self) -> int:
        return self.only_before_passed + self.only_after_passed

    @property
    def total(self) -> int:
        return (
            self.both_passed + self.only_before_passed + self.only_after_passed + self.both_failed
        )


@dataclass(frozen=True, slots=True)
class McNemarResult:
    counts: PairedCounts
    p_value: float
    alpha: float = DEFAULT_ALPHA

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    @property
    def direction(self) -> str:
        if self.counts.only_after_passed > self.counts.only_before_passed:
            return "improvement"
        if self.counts.only_after_passed < self.counts.only_before_passed:
            return "regression"
        return "no change"


def paired_counts(before: Sequence[bool], after: Sequence[bool]) -> PairedCounts:
    """Build the contingency table from two aligned sequences of per-case outcomes."""
    if len(before) != len(after):
        raise ValueError("paired outcomes must be the same length")

    both_passed = only_before = only_after = both_failed = 0
    for was, now in zip(before, after, strict=True):
        if was and now:
            both_passed += 1
        elif was and not now:
            only_before += 1
        elif not was and now:
            only_after += 1
        else:
            both_failed += 1

    return PairedCounts(
        both_passed=both_passed,
        only_before_passed=only_before,
        only_after_passed=only_after,
        both_failed=both_failed,
    )


def binomial_two_sided_p(successes: int, trials: int) -> float:
    """Exact two-sided binomial p-value against a fair coin."""
    if trials == 0:
        return 1.0
    smaller = min(successes, trials - successes)
    tail = sum(comb(trials, i) for i in range(smaller + 1)) / (2.0**trials)
    return min(1.0, 2.0 * tail)


def mcnemar(
    before: Sequence[bool], after: Sequence[bool], *, alpha: float = DEFAULT_ALPHA
) -> McNemarResult:
    """McNemar's exact test on paired pass/fail outcomes.

    The exact binomial form rather than the chi-squared approximation: eval
    suites routinely produce a handful of discordant pairs, and the
    approximation is unreliable below about 25 of them.

    With no disagreements at all the p-value is 1.0 — comparing a run against
    itself must never report a change, and that case matters more than
    detecting a real regression, because a gate that cries wolf gets disabled.
    """
    counts = paired_counts(before, after)
    p_value = binomial_two_sided_p(counts.only_after_passed, counts.discordant)
    return McNemarResult(counts=counts, p_value=p_value, alpha=alpha)


def required_sample_size(
    baseline_rate: float,
    detectable_difference: float,
    *,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> int:
    """Tasks needed per version to detect ``detectable_difference``.

    Inverts the usual question. Rather than asking whether a result reached
    significance, ask how large a dataset would have to be for the effect you
    care about to be detectable at all — which is what actually decides how many
    cases are worth building and running.

    Uses the unpaired two-proportion formula, so it is deliberately
    conservative: a paired test on the same tasks needs fewer.
    """
    if not 0.0 <= baseline_rate <= 1.0:
        raise ValueError("baseline_rate must be between 0 and 1")
    if detectable_difference <= 0.0:
        raise ValueError("detectable_difference must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be strictly between 0 and 1")

    target = baseline_rate + detectable_difference
    if not 0.0 <= target <= 1.0:
        raise ValueError(
            f"baseline_rate {baseline_rate} plus {detectable_difference} leaves the unit interval"
        )

    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - alpha / 2.0)
    z_power = normal.inv_cdf(power)

    pooled = (baseline_rate + target) / 2.0
    numerator = (
        z_alpha * sqrt(2.0 * pooled * (1.0 - pooled))
        + z_power
        * sqrt(baseline_rate * (1.0 - baseline_rate) + target * (1.0 - target))
    ) ** 2
    return _ceil_int(numerator / (detectable_difference**2))


def _ceil_int(value: float) -> int:
    whole = int(value)
    return whole if whole == value else whole + 1
