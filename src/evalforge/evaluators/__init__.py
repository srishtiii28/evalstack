"""Evaluators: the plugin layer that turns a trajectory into scores."""

from evalforge.evaluators.base import (
    EvaluationContext,
    Evaluator,
    EvaluatorSuite,
    evaluate_with_timeout,
)
from evalforge.evaluators.outcome import SuiteOutcomeEvaluator, parse_pytest_counts
from evalforge.evaluators.patch import PatchLocalityEvaluator, PatchWeights
from evalforge.evaluators.registry import (
    SUITES,
    default_suite,
    outcome_only_suite,
    resolve_suite,
    suite_names,
)
from evalforge.evaluators.trajectory import (
    TrajectoryEvaluator,
    TrajectorySignals,
    TrajectoryWeights,
    extract_signals,
)

__all__ = [
    "SUITES",
    "EvaluationContext",
    "Evaluator",
    "EvaluatorSuite",
    "PatchLocalityEvaluator",
    "PatchWeights",
    "SuiteOutcomeEvaluator",
    "TrajectoryEvaluator",
    "TrajectorySignals",
    "TrajectoryWeights",
    "default_suite",
    "evaluate_with_timeout",
    "extract_signals",
    "outcome_only_suite",
    "parse_pytest_counts",
    "resolve_suite",
    "suite_names",
]
