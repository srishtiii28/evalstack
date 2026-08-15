"""What the attempt cost to produce.

Capability without cost is half a number. An agent that solves 5% more tasks
using triple the tokens is not obviously better, and the decision belongs to
whoever is paying — which means the harness has to report both.

Budgets are per-attempt allowances rather than hard limits: staying within one
scores full marks, and the score degrades past it rather than falling off a
cliff, because being slightly over budget is a different situation from
looping forever.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import JsonValue

from evalforge.evaluators.base import EvaluationContext, Evaluator
from evalforge.schema.result import EvaluatorResult
from evalforge.schema.trajectory import ModelCall, ToolCall


@dataclass(frozen=True, slots=True)
class EfficiencyBudgets:
    """Per-attempt allowances. Exceeding one by 100% scores zero on that axis."""

    tokens: int = 20_000
    model_calls: int = 8
    wall_seconds: float = 120.0
    pass_threshold: float = 0.5

    def __post_init__(self) -> None:
        for name, value in (
            ("tokens", self.tokens),
            ("model_calls", self.model_calls),
            ("wall_seconds", self.wall_seconds),
        ):
            if value <= 0:
                raise ValueError(f"{name} budget must be positive")

    def describe(self) -> dict[str, JsonValue]:
        return {
            "tokens": self.tokens,
            "model_calls": self.model_calls,
            "wall_seconds": self.wall_seconds,
            "pass_threshold": self.pass_threshold,
        }


def score_against_budget(used: float, budget: float) -> float:
    """Full marks up to the budget, linearly to zero at twice the budget."""
    if used <= budget:
        return 1.0
    return max(0.0, 1.0 - (used - budget) / budget)


class EfficiencyEvaluator(Evaluator):
    """Scores token, call and wall-clock consumption against a budget."""

    def __init__(self, budgets: EfficiencyBudgets | None = None) -> None:
        self._budgets = budgets or EfficiencyBudgets()

    @property
    def name(self) -> str:
        return "efficiency"

    def config(self) -> dict[str, JsonValue]:
        return self._budgets.describe()

    async def evaluate(self, context: EvaluationContext) -> EvaluatorResult:
        trajectory = context.trajectory
        model_calls = trajectory.of_type(ModelCall)
        cached_calls = sum(1 for call in model_calls if call.cached)

        tokens = trajectory.total_input_tokens + trajectory.total_output_tokens
        wall_seconds = trajectory.duration_ms / 1000.0

        budgets = self._budgets
        axes = {
            "tokens": score_against_budget(tokens, budgets.tokens),
            "model_calls": score_against_budget(len(model_calls), budgets.model_calls),
            "wall_seconds": score_against_budget(wall_seconds, budgets.wall_seconds),
        }
        score = sum(axes.values()) / len(axes)

        detail: dict[str, JsonValue] = {
            "input_tokens": trajectory.total_input_tokens,
            "output_tokens": trajectory.total_output_tokens,
            "total_tokens": tokens,
            "model_calls": len(model_calls),
            # A run served from cache is genuinely cheap; one that reports no
            # cost because rates are unknown is not. Surfacing the cache count
            # is what lets a reader tell those apart.
            "cached_model_calls": cached_calls,
            "cost_usd": round(trajectory.total_cost_usd, 6),
            "tool_calls": len(trajectory.of_type(ToolCall)),
            "wall_seconds": round(wall_seconds, 3),
            "axis_scores": {name: round(value, 4) for name, value in axes.items()},
        }

        return EvaluatorResult(
            name=self.name,
            version=self.version,
            score=score,
            passed=score >= budgets.pass_threshold,
            detail=detail,
        )
