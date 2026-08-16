"""Comparing two runs, and saying honestly whether anything changed.

The output most eval tools produce is a delta: "68% versus 72%, down 4 points."
On thirty tasks that is about one task changing its mind, and acting on it is a
coin flip. So a comparison here reports three things instead:

* **Is it real?** McNemar's exact test on the paired per-case outcomes, plus a
  bootstrap interval on the size of the difference.
* **Where did it happen?** Per-case transitions — which tasks broke, which were
  fixed — and per-dimension deltas, so a regression points somewhere.
* **Could it have been real?** When nothing reaches significance, the report
  says how many tasks would be needed to detect the effect that was observed.
  "No significant change" and "this dataset is too small to tell" are different
  findings, and conflating them is how teams end up trusting noise.

Comparability is checked before any of that. Two runs against different dataset
versions or different evaluator suites are not comparable, and the content
hashes recorded on every run are what make that checkable rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from evalforge.schema.result import RunResult
from evalforge.stats.intervals import DEFAULT_CONFIDENCE, Interval, paired_bootstrap
from evalforge.stats.significance import (
    DEFAULT_ALPHA,
    McNemarResult,
    mcnemar,
    required_sample_size,
)

Verdict = Literal["regression", "improvement", "no significant change", "not comparable"]
TransitionKind = Literal["fixed", "broken", "stable pass", "stable fail", "mixed"]

#: Effect size used when asking "how many tasks would we have needed?" for a
#: comparison that observed no difference at all.
FALLBACK_EFFECT = 0.05


@dataclass(frozen=True, slots=True)
class CaseTransition:
    """What happened to one case between the two runs."""

    case_id: str
    before_rate: float
    after_rate: float

    @property
    def kind(self) -> TransitionKind:
        was, now = self.before_rate, self.after_rate
        if was == 0.0 and now == 1.0:
            return "fixed"
        if was == 1.0 and now == 0.0:
            return "broken"
        if was == 1.0 and now == 1.0:
            return "stable pass"
        if was == 0.0 and now == 0.0:
            return "stable fail"
        return "mixed"

    @property
    def delta(self) -> float:
        return self.after_rate - self.before_rate


@dataclass(frozen=True, slots=True)
class DimensionDelta:
    """How one evaluator's mean score moved."""

    name: str
    before: float
    after: float

    @property
    def delta(self) -> float:
        return self.after - self.before


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    before_run_id: str
    after_run_id: str
    before_rate: float
    after_rate: float
    shared_cases: int
    interval: Interval
    test: McNemarResult
    transitions: tuple[CaseTransition, ...]
    dimensions: tuple[DimensionDelta, ...]
    warnings: tuple[str, ...] = ()
    comparable: bool = True
    required_cases: int = 0

    @property
    def delta(self) -> float:
        return self.after_rate - self.before_rate

    @property
    def verdict(self) -> Verdict:
        if not self.comparable:
            return "not comparable"
        if not self.test.significant:
            return "no significant change"
        return "regression" if self.delta < 0 else "improvement"

    def transitions_of(self, kind: TransitionKind) -> tuple[CaseTransition, ...]:
        return tuple(t for t in self.transitions if t.kind == kind)

    @property
    def underpowered(self) -> bool:
        """True when the dataset is too small to have detected this effect."""
        return not self.test.significant and self.shared_cases < self.required_cases


def _case_rates(run: RunResult) -> dict[str, float]:
    """Per-case pass rate over completed attempts.

    A rate rather than a boolean so that k-sampled runs compare meaningfully:
    a case that went 3/3 and one that went 1/3 are different outcomes, and
    collapsing both to "passed" throws that away.
    """
    return {case_id: tally.rate for case_id, tally in run.tallies().items()}


def _comparability_warnings(before: RunResult, after: RunResult) -> tuple[str, ...]:
    warnings: list[str] = []
    if before.dataset_hash != after.dataset_hash:
        warnings.append(
            f"different datasets: {before.dataset_ref} and {after.dataset_ref} "
            "have different content hashes"
        )
    if before.suite_hash != after.suite_hash:
        warnings.append(
            f"different evaluator suites: {before.suite_name} was configured differently "
            "in each run, so the scores were not produced the same way"
        )
    if before.agent_hash == after.agent_hash and before.agent_ref == after.agent_ref:
        warnings.append("both runs used the same agent configuration")
    return tuple(warnings)


def compare(
    before: RunResult,
    after: RunResult,
    *,
    alpha: float = DEFAULT_ALPHA,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> ComparisonReport:
    """Compare two runs over the cases they have in common."""
    before_rates = _case_rates(before)
    after_rates = _case_rates(after)
    shared = sorted(set(before_rates) & set(after_rates))

    warnings = list(_comparability_warnings(before, after))
    dataset_mismatch = before.dataset_hash != after.dataset_hash
    suite_mismatch = before.suite_hash != after.suite_hash

    if not shared:
        warnings.append("the two runs share no cases")

    transitions = tuple(
        CaseTransition(
            case_id=case_id, before_rate=before_rates[case_id], after_rate=after_rates[case_id]
        )
        for case_id in shared
    )

    before_values = [before_rates[case_id] for case_id in shared]
    after_values = [after_rates[case_id] for case_id in shared]

    if shared:
        interval = paired_bootstrap(
            before_values, after_values, confidence=confidence, seed=seed
        )
        # The test needs booleans; a k-sampled case counts as passing when it
        # passed more often than not.
        test = mcnemar(
            [value > 0.5 for value in before_values],
            [value > 0.5 for value in after_values],
            alpha=alpha,
        )
    else:
        interval = Interval(estimate=0.0, low=0.0, high=0.0, confidence=confidence)
        test = mcnemar([], [], alpha=alpha)

    before_rate = sum(before_values) / len(before_values) if before_values else 0.0
    after_rate = sum(after_values) / len(after_values) if after_values else 0.0

    effect = abs(after_rate - before_rate) or FALLBACK_EFFECT
    baseline = min(max(before_rate, 0.0), 1.0 - effect) if effect < 1.0 else 0.0
    required = required_sample_size(baseline, effect, alpha=alpha)

    return ComparisonReport(
        before_run_id=before.run_id,
        after_run_id=after.run_id,
        before_rate=before_rate,
        after_rate=after_rate,
        shared_cases=len(shared),
        interval=interval,
        test=test,
        transitions=transitions,
        dimensions=_dimension_deltas(before, after),
        warnings=tuple(warnings),
        comparable=bool(shared) and not dataset_mismatch and not suite_mismatch,
        required_cases=required,
    )


def _dimension_deltas(before: RunResult, after: RunResult) -> tuple[DimensionDelta, ...]:
    """Mean score movement for every evaluator present in either run."""
    before_scores = before.evaluator_scores()
    after_scores = after.evaluator_scores()
    names = sorted(set(before_scores) | set(after_scores))
    return tuple(
        DimensionDelta(
            name=name, before=before_scores.get(name, 0.0), after=after_scores.get(name, 0.0)
        )
        for name in names
    )
