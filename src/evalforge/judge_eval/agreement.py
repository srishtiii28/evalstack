"""Measuring whether a judge agrees with a human.

A model-based judge is itself a model, and can be wrong in ways that are
invisible from its own output — it is confidently articulate about verdicts it
got backwards. So a judge is not a measuring instrument until it has been
measured against something.

Raw accuracy is the obvious statistic and the misleading one. If 85% of attempts
fail, a judge that answers "fail" every single time scores 85% and has learned
nothing. Cohen's κ corrects for exactly that by subtracting the agreement you
would expect from the marginal rates alone, so the degenerate judge scores ~0.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: Conventional reading of κ. Coarse, and worth quoting as such rather than
#: pretending a threshold is a law of nature.
KAPPA_BANDS: tuple[tuple[float, str], ...] = (
    (0.81, "almost perfect"),
    (0.61, "substantial"),
    (0.41, "moderate"),
    (0.21, "fair"),
    (0.01, "slight"),
    (-1.0, "none or worse than chance"),
)


def interpret_kappa(kappa: float) -> str:
    for threshold, label in KAPPA_BANDS:
        if kappa >= threshold:
            return label
    return "none or worse than chance"


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """Precision and recall for one verdict class, judge scored against human."""

    label: str
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def support(self) -> int:
        """How many items the human actually assigned to this class."""
        return self.true_positives + self.false_negatives

    @property
    def precision(self) -> float:
        predicted = self.true_positives + self.false_positives
        return self.true_positives / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        return self.true_positives / self.support if self.support else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0


@dataclass(frozen=True, slots=True)
class AgreementReport:
    """What a judge's verdicts are worth, measured against human labels."""

    labels: tuple[str, ...]
    matrix: dict[tuple[str, str], int]
    accuracy: float
    kappa: float
    per_class: tuple[ClassMetrics, ...]
    count: int

    @property
    def interpretation(self) -> str:
        return interpret_kappa(self.kappa)

    def meets(self, threshold: float) -> bool:
        return self.kappa >= threshold

    def disagreements(self) -> int:
        return sum(
            n for (human, judge), n in self.matrix.items() if human != judge
        )


def confusion_matrix(
    human: Sequence[str], judge: Sequence[str]
) -> tuple[tuple[str, ...], dict[tuple[str, str], int]]:
    """Counts of ``(human label, judge label)`` pairs, over the union of labels."""
    if len(human) != len(judge):
        raise ValueError("human and judge verdicts must be the same length")

    labels = tuple(sorted(set(human) | set(judge)))
    counts = {(a, b): 0 for a in labels for b in labels}
    for actual, predicted in zip(human, judge, strict=True):
        counts[(actual, predicted)] += 1
    return labels, counts


def cohens_kappa(human: Sequence[str], judge: Sequence[str]) -> float:
    """Chance-corrected agreement between two sets of verdicts.

    ``(p_observed - p_expected) / (1 - p_expected)``, where ``p_expected`` comes
    from the two raters' marginal distributions. A judge that always answers the
    majority class agrees often and scores about zero, which is the point.
    """
    if not human:
        raise ValueError("cannot measure agreement over an empty sample")

    labels, counts = confusion_matrix(human, judge)
    total = len(human)

    observed = sum(counts[(label, label)] for label in labels) / total

    expected = 0.0
    for label in labels:
        human_share = sum(counts[(label, other)] for other in labels) / total
        judge_share = sum(counts[(other, label)] for other in labels) / total
        expected += human_share * judge_share

    if expected >= 1.0:
        # Both raters used a single label throughout; chance agreement is total,
        # so κ is undefined. Report perfect only if they actually agreed.
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def agreement_report(human: Sequence[str], judge: Sequence[str]) -> AgreementReport:
    """Full agreement report: accuracy, κ, and per-class precision and recall."""
    labels, counts = confusion_matrix(human, judge)
    total = len(human)
    if total == 0:
        raise ValueError("cannot measure agreement over an empty sample")

    correct = sum(counts[(label, label)] for label in labels)

    per_class = tuple(
        ClassMetrics(
            label=label,
            true_positives=counts[(label, label)],
            false_positives=sum(counts[(other, label)] for other in labels if other != label),
            false_negatives=sum(counts[(label, other)] for other in labels if other != label),
        )
        for label in labels
    )

    return AgreementReport(
        labels=labels,
        matrix=counts,
        accuracy=correct / total,
        kappa=cohens_kappa(human, judge),
        per_class=per_class,
        count=total,
    )
