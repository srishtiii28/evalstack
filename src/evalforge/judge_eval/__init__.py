"""Measuring the judge before trusting it."""

from evalforge.judge_eval.agreement import (
    AgreementReport,
    ClassMetrics,
    agreement_report,
    cohens_kappa,
    confusion_matrix,
    interpret_kappa,
)
from evalforge.judge_eval.bias import (
    PositionBiasReport,
    SelfPreferenceReport,
    position_bias,
    self_preference,
)
from evalforge.judge_eval.gold import GoldSet, JudgeExample
from evalforge.judge_eval.validation import (
    DEFAULT_KAPPA_THRESHOLD,
    JudgeIdentity,
    ValidationReport,
    validate_judge,
)

__all__ = [
    "DEFAULT_KAPPA_THRESHOLD",
    "AgreementReport",
    "ClassMetrics",
    "GoldSet",
    "JudgeExample",
    "JudgeIdentity",
    "PositionBiasReport",
    "SelfPreferenceReport",
    "ValidationReport",
    "agreement_report",
    "cohens_kappa",
    "confusion_matrix",
    "interpret_kappa",
    "position_bias",
    "self_preference",
    "validate_judge",
]
