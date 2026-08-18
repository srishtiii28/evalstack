"""Choosing which cases are worth running.

A full suite on every commit is not affordable once each case costs model calls,
so the question stops being "how fast can we run everything" and becomes "which
cases would actually change our mind". Those are not the same set, and the gap
is large: a case every version passes and a case every version fails both cost
full price and carry no information at all.

Discriminative power here is the variance of a case's outcome across historical
runs. A case that has gone both ways separates agents; a case that has never
moved cannot. Dividing that by measured cost gives information per token, and a
greedy knapsack over that ratio fills the budget.

The greedy choice is deliberate. Optimal knapsack is achievable at this scale,
but the value estimates come from a handful of noisy historical runs — solving
exactly for an objective known to ±30% would be precision theatre.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from evalforge.schema.result import RunResult

#: Cost assumed for a case no historical run has ever executed. Non-zero so an
#: unmeasured case cannot look infinitely cheap and crowd out everything else.
DEFAULT_COST = 1.0


@dataclass(frozen=True, slots=True)
class CaseValue:
    """What one case is worth running, and what it costs to run."""

    case_id: str
    runs_seen: int
    pass_rate: float
    mean_cost: float

    @property
    def discrimination(self) -> float:
        """Bernoulli variance of the outcome across runs, scaled to [0, 1].

        ``4 * p * (1 - p)`` peaks at 1.0 for a case that half the versions pass and
        falls to 0 for one that is always passed or always failed.
        """
        return 4.0 * self.pass_rate * (1.0 - self.pass_rate)

    @property
    def value_per_cost(self) -> float:
        return self.discrimination / self.mean_cost if self.mean_cost > 0 else 0.0

    @property
    def informative(self) -> bool:
        """Whether this case has ever distinguished one run from another."""
        return self.discrimination > 0.0


@dataclass(frozen=True, slots=True)
class Selection:
    """A budgeted subset, and what it is expected to buy."""

    case_ids: tuple[str, ...]
    total_cost: float
    total_discrimination: float
    budget: float
    considered: int

    @property
    def within_budget(self) -> bool:
        return self.total_cost <= self.budget

    @property
    def coverage(self) -> float:
        return len(self.case_ids) / self.considered if self.considered else 0.0


def score_cases(runs: Sequence[RunResult]) -> tuple[CaseValue, ...]:
    """Summarise each case's historical outcome variance and cost.

    Only *completed* attempts count. A case that timed out tells you about the
    harness, not about whether the case separates agents, and letting those in
    would make flaky cases look maximally informative.
    """
    passes: dict[str, int] = {}
    totals: dict[str, int] = {}
    costs: dict[str, float] = {}

    for run in runs:
        # Per run, so a case sampled k times is not counted k times.
        seen: dict[str, tuple[int, int, float]] = {}
        for result in run.case_results:
            if result.status != "completed":
                continue
            hits, count, cost = seen.get(result.case_id, (0, 0, 0.0))
            tokens = result.input_tokens + result.output_tokens
            seen[result.case_id] = (
                hits + int(result.passed),
                count + 1,
                cost + (tokens or result.duration_s),
            )
        for case_id, (hits, count, cost) in seen.items():
            passes[case_id] = passes.get(case_id, 0) + (1 if hits * 2 >= count else 0)
            totals[case_id] = totals.get(case_id, 0) + 1
            costs[case_id] = costs.get(case_id, 0.0) + cost / count

    return tuple(
        sorted(
            (
                CaseValue(
                    case_id=case_id,
                    runs_seen=totals[case_id],
                    pass_rate=passes[case_id] / totals[case_id],
                    mean_cost=max(costs[case_id] / totals[case_id], DEFAULT_COST),
                )
                for case_id in totals
            ),
            key=lambda value: (-value.value_per_cost, value.case_id),
        )
    )


def select_within_budget(
    values: Iterable[CaseValue], *, budget: float, include_unmeasured: bool = True
) -> Selection:
    """Greedily take the most informative cases per unit cost, under ``budget``.

    Cases that have never discriminated are appended last and only if room
    remains: they are not evidence of nothing, they are an absence of evidence,
    and dropping them entirely would freeze the suite around what it already
    knows.
    """
    if budget < 0:
        raise ValueError("budget must not be negative")

    ranked = sorted(values, key=lambda value: (-value.value_per_cost, value.case_id))
    informative = [value for value in ranked if value.informative]
    rest = [value for value in ranked if not value.informative]

    chosen: list[CaseValue] = []
    spent = 0.0
    for value in informative + (rest if include_unmeasured else []):
        if spent + value.mean_cost > budget:
            continue
        chosen.append(value)
        spent += value.mean_cost

    return Selection(
        case_ids=tuple(value.case_id for value in chosen),
        total_cost=spent,
        total_discrimination=sum(value.discrimination for value in chosen),
        budget=budget,
        considered=len(ranked),
    )


def select_from_runs(runs: Sequence[RunResult], *, budget: float) -> Selection:
    """Score historical runs and pick a subset that fits the budget."""
    return select_within_budget(score_cases(runs), budget=budget)
