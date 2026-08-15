"""Named evaluator suites.

Suites are looked up by name and recorded on every run by content hash, so
"suite v2" is a checkable claim rather than a convention.
"""

from __future__ import annotations

from collections.abc import Callable

from evalforge.evaluators.base import EvaluatorSuite
from evalforge.evaluators.efficiency import EfficiencyEvaluator
from evalforge.evaluators.outcome import SuiteOutcomeEvaluator
from evalforge.evaluators.patch import PatchLocalityEvaluator
from evalforge.evaluators.safety import SafetyEvaluator
from evalforge.evaluators.trajectory import TrajectoryEvaluator


def default_suite() -> EvaluatorSuite:
    """Outcome decides pass/fail; everything else is measured alongside.

    Only ``tests`` gates. A wide diff or a wasteful trajectory should show up in
    the numbers and in comparisons without turning a working fix into a failure —
    that judgement belongs to whoever sets the thresholds, not to the harness.
    """
    return EvaluatorSuite(
        name="default",
        evaluators=(
            SuiteOutcomeEvaluator(),
            PatchLocalityEvaluator(),
            TrajectoryEvaluator(),
            EfficiencyEvaluator(),
            SafetyEvaluator(),
        ),
        gating=frozenset({"tests"}),
    )


def strict_suite() -> EvaluatorSuite:
    """Outcome *and* safety must pass.

    The suite to gate a deployment on: a fix that works but tried to write
    outside its workspace on the way is not a fix you want merged.
    """
    return EvaluatorSuite(
        name="strict",
        evaluators=(
            SuiteOutcomeEvaluator(),
            PatchLocalityEvaluator(),
            TrajectoryEvaluator(),
            EfficiencyEvaluator(),
            SafetyEvaluator(),
        ),
        gating=frozenset({"tests", "safety"}),
    )


def outcome_only_suite() -> EvaluatorSuite:
    """Just the tests — the cheapest suite, for smoke checks and CI gating."""
    return EvaluatorSuite(
        name="outcome-only",
        evaluators=(SuiteOutcomeEvaluator(),),
        gating=frozenset({"tests"}),
    )


SUITES: dict[str, Callable[[], EvaluatorSuite]] = {
    "default": default_suite,
    "strict": strict_suite,
    "outcome-only": outcome_only_suite,
}


def resolve_suite(name: str) -> EvaluatorSuite:
    try:
        factory = SUITES[name]
    except KeyError:
        known = ", ".join(sorted(SUITES))
        raise KeyError(f"unknown suite {name!r}; known suites: {known}") from None
    return factory()


def suite_names() -> tuple[str, ...]:
    return tuple(sorted(SUITES))
