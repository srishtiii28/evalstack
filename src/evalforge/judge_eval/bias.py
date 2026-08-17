"""Probing a judge for the two failure modes agreement scores cannot see.

A judge can agree with humans often and still be unusable, because agreement
measures accuracy on the labelled set while these measure whether the verdict
depends on something it should not:

* **Position bias** — present the same pair in both orders. Any verdict that
  flips is a verdict about ordering, not quality. This is the reason a
  leaderboard built from pairwise judging can be almost entirely artefact.
* **Self-preference** — a judge that favours output from its own model family
  is a judge you cannot use to compare that family against others, which is
  usually exactly what it was bought for.

Both are reported as rates with the raw counts, because a single number invites
a threshold and these are diagnostic rather than pass/fail.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PositionBiasReport:
    """How often a verdict survives swapping the order of what it compared."""

    pairs: int
    consistent: int
    #: Times the judge preferred whichever candidate was shown first.
    favoured_first: int
    favoured_second: int

    @property
    def consistency(self) -> float:
        return self.consistent / self.pairs if self.pairs else 0.0

    @property
    def flip_rate(self) -> float:
        return 1.0 - self.consistency

    @property
    def positional_skew(self) -> float:
        """Signed -1 to 1: positive means a preference for whatever came first.

        Computed over the *inconsistent* pairs, since those are the ones whose
        verdict was decided by position rather than by content.
        """
        decided_by_position = self.favoured_first + self.favoured_second
        if not decided_by_position:
            return 0.0
        return (self.favoured_first - self.favoured_second) / decided_by_position


def position_bias(
    forward: Sequence[str], reversed_order: Sequence[str], *, first: str = "a", second: str = "b"
) -> PositionBiasReport:
    """Compare verdicts on the same pairs shown in both orders.

    ``forward`` holds verdicts with candidate A shown first; ``reversed_order``
    holds them with B shown first. A judge free of position bias should return
    the *same winner* both times — so the labels are compared after undoing the
    swap, not compared literally.
    """
    if len(forward) != len(reversed_order):
        raise ValueError("both orderings must cover the same pairs")

    consistent = favoured_first = favoured_second = 0
    for original, swapped in zip(forward, reversed_order, strict=True):
        # In the reversed presentation the positions are exchanged, so the same
        # underlying winner appears under the opposite label.
        undone = {first: second, second: first}.get(swapped, swapped)
        if original == undone:
            consistent += 1
        elif original == first and swapped == first:
            # Preferred the first-shown candidate in both presentations.
            favoured_first += 1
        elif original == second and swapped == second:
            favoured_second += 1

    return PositionBiasReport(
        pairs=len(forward),
        consistent=consistent,
        favoured_first=favoured_first,
        favoured_second=favoured_second,
    )


@dataclass(frozen=True, slots=True)
class SelfPreferenceReport:
    """Whether a judge scores its own family more generously than others."""

    own_family: int
    own_family_favourable: int
    other_family: int
    other_family_favourable: int

    @property
    def own_rate(self) -> float:
        return self.own_family_favourable / self.own_family if self.own_family else 0.0

    @property
    def other_rate(self) -> float:
        return self.other_family_favourable / self.other_family if self.other_family else 0.0

    @property
    def gap(self) -> float:
        """How much more favourably the judge treats its own family."""
        return self.own_rate - self.other_rate

    @property
    def comparable(self) -> bool:
        """Whether there is enough of both kinds for the gap to mean anything."""
        return self.own_family > 0 and self.other_family > 0


def self_preference(
    verdicts: Sequence[str], authored_by_judge_family: Sequence[bool], *, favourable: str = "pass"
) -> SelfPreferenceReport:
    """Split verdicts by whether the judge shares a family with the author."""
    if len(verdicts) != len(authored_by_judge_family):
        raise ValueError("verdicts and authorship flags must be the same length")

    own = own_ok = other = other_ok = 0
    for verdict, is_own in zip(verdicts, authored_by_judge_family, strict=True):
        if is_own:
            own += 1
            own_ok += verdict == favourable
        else:
            other += 1
            other_ok += verdict == favourable

    return SelfPreferenceReport(
        own_family=own,
        own_family_favourable=own_ok,
        other_family=other,
        other_family_favourable=other_ok,
    )
