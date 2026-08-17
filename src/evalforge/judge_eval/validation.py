"""The verdict on the judge.

Combines agreement against human labels with the bias probes, and — the part
that is easy to leave out and expensive to leave out — records *which* judge was
measured. A κ of 0.72 is a fact about one model with one prompt at one
temperature. Comparing it to a number produced by a different judge is the same
category of mistake as comparing success rates across different datasets, which
is why the dataset and suite hashes exist elsewhere in this project.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from evalforge.hashing import content_hash, text_hash
from evalforge.judge_eval.agreement import AgreementReport, agreement_report
from evalforge.judge_eval.bias import (
    PositionBiasReport,
    SelfPreferenceReport,
    position_bias,
    self_preference,
)
from evalforge.judge_eval.gold import GoldSet

#: Below this, a judge is not measuring what you think it is measuring.
DEFAULT_KAPPA_THRESHOLD = 0.60


@dataclass(frozen=True, slots=True)
class JudgeIdentity:
    """Everything that makes one judge a different instrument from another."""

    model: str
    prompt: str
    temperature: float = 0.0

    @property
    def prompt_hash(self) -> str:
        return text_hash(self.prompt)

    @property
    def fingerprint(self) -> str:
        return content_hash(
            {
                "model": self.model,
                "prompt_hash": self.prompt_hash,
                "temperature": self.temperature,
            }
        )

    def describe(self) -> str:
        return f"{self.model} @ T={self.temperature:g} (prompt {self.prompt_hash[7:19]})"


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Whether a judge is fit to be used, and on what evidence."""

    judge: JudgeIdentity
    gold_name: str
    gold_version: str
    gold_hash: str
    agreement: AgreementReport
    threshold: float = DEFAULT_KAPPA_THRESHOLD
    position: PositionBiasReport | None = None
    preference: SelfPreferenceReport | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return self.agreement.meets(self.threshold)

    @property
    def summary(self) -> str:
        verdict = "usable" if self.passed else "not usable"
        return (
            f"{verdict}: kappa {self.agreement.kappa:.3f} "
            f"({self.agreement.interpretation}) against {self.agreement.count} labels"
        )

    def comparable_to(self, other: ValidationReport) -> bool:
        """Whether two reports measure the same instrument on the same set."""
        return (
            self.judge.fingerprint == other.judge.fingerprint
            and self.gold_hash == other.gold_hash
        )

    def differences_from(self, other: ValidationReport) -> tuple[str, ...]:
        """Why two reports cannot be compared, in words."""
        reasons: list[str] = []
        if self.judge.model != other.judge.model:
            reasons.append(f"different judge model: {other.judge.model} then {self.judge.model}")
        if self.judge.prompt_hash != other.judge.prompt_hash:
            reasons.append("the judge prompt changed, so these are different instruments")
        if self.judge.temperature != other.judge.temperature:
            reasons.append(
                f"different temperature: {other.judge.temperature:g} "
                f"then {self.judge.temperature:g}"
            )
        if self.gold_hash != other.gold_hash:
            reasons.append("the gold set changed, so the labels are not the same labels")
        return tuple(reasons)


def _imbalance_warning(gold: GoldSet) -> str | None:
    balance = gold.label_balance()
    total = sum(balance.values())
    if total == 0:
        return None
    largest = max(balance.values())
    share = largest / total
    if share >= 0.8:
        dominant = max(balance, key=lambda label: balance[label])
        return (
            f"{share:.0%} of the gold set is labelled {dominant!r}; raw accuracy is "
            "close to meaningless here and kappa will be noisy"
        )
    return None


def validate_judge(
    gold: GoldSet,
    judge_verdicts: Sequence[str],
    *,
    judge: JudgeIdentity,
    threshold: float = DEFAULT_KAPPA_THRESHOLD,
    swapped_verdicts: Sequence[str] | None = None,
    forward_verdicts: Sequence[str] | None = None,
    own_family_flags: Sequence[bool] | None = None,
) -> ValidationReport:
    """Score a judge's verdicts against the gold set, with optional bias probes."""
    if len(judge_verdicts) != len(gold.examples):
        raise ValueError(
            f"judge produced {len(judge_verdicts)} verdicts for {len(gold.examples)} examples"
        )

    warnings: list[str] = []
    imbalance = _imbalance_warning(gold)
    if imbalance is not None:
        warnings.append(imbalance)
    if len(gold.examples) < 20:
        warnings.append(
            f"only {len(gold.examples)} labelled examples; kappa on a set this small "
            "carries a wide interval and should be read as indicative"
        )

    position = None
    if swapped_verdicts is not None:
        position = position_bias(
            forward_verdicts if forward_verdicts is not None else judge_verdicts,
            swapped_verdicts,
        )

    preference = None
    if own_family_flags is not None:
        preference = self_preference(judge_verdicts, own_family_flags)
        if preference.comparable and abs(preference.gap) >= 0.15:
            warnings.append(
                f"self-preference gap of {preference.gap:+.0%}: this judge treats its own "
                "family differently and should not arbitrate between families"
            )

    return ValidationReport(
        judge=judge,
        gold_name=gold.name,
        gold_version=gold.version,
        gold_hash=gold.content_hash,
        agreement=agreement_report(gold.verdicts, judge_verdicts),
        threshold=threshold,
        position=position,
        preference=preference,
        warnings=tuple(warnings),
    )
