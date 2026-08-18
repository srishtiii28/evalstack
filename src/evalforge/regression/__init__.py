"""Detecting and explaining differences between runs."""

from evalforge.regression.clustering import (
    FailureCluster,
    FailureSignature,
    cluster_failures,
    cluster_shift,
    overall_purity,
    signature_for,
)
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
    "FailureCluster",
    "FailureSignature",
    "TransitionKind",
    "Verdict",
    "cluster_failures",
    "cluster_shift",
    "compare",
    "overall_purity",
    "signature_for",
]
