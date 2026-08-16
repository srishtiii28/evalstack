"""Detecting and explaining differences between runs."""

from evalforge.regression.compare import (
    CaseTransition,
    ComparisonReport,
    DimensionDelta,
    TransitionKind,
    Verdict,
    compare,
)

__all__ = [
    "CaseTransition",
    "ComparisonReport",
    "DimensionDelta",
    "TransitionKind",
    "Verdict",
    "compare",
]
