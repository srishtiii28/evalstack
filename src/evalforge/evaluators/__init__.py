"""Evaluators: the plugin layer that turns a trajectory into scores."""

from evalforge.evaluators.base import (
    EvaluationContext,
    Evaluator,
    EvaluatorSuite,
    evaluate_with_timeout,
)
from evalforge.evaluators.efficiency import (
    EfficiencyBudgets,
    EfficiencyEvaluator,
    score_against_budget,
)
from evalforge.evaluators.outcome import SuiteOutcomeEvaluator, parse_pytest_counts
from evalforge.evaluators.patch import PatchLocalityEvaluator, PatchWeights
from evalforge.evaluators.registry import (
    SUITES,
    default_suite,
    outcome_only_suite,
    resolve_suite,
    strict_suite,
    suite_names,
    with_judge,
)
from evalforge.evaluators.safety import SafetyEvaluator, SafetyPolicy
from evalforge.evaluators.trajectory import (
    TrajectoryEvaluator,
    TrajectorySignals,
    TrajectoryWeights,
    extract_signals,
)

__all__ = [
    "SUITES",
    "EfficiencyBudgets",
    "EfficiencyEvaluator",
    "EvaluationContext",
    "Evaluator",
    "EvaluatorSuite",
    "PatchLocalityEvaluator",
    "PatchWeights",
    "SafetyEvaluator",
    "SafetyPolicy",
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
    "score_against_budget",
    "strict_suite",
    "suite_names",
    "with_judge",
]
