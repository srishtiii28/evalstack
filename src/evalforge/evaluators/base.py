"""The evaluator contract and the suites that compose evaluators."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import JsonValue

from evalforge.env.workspace import Workspace
from evalforge.hashing import content_hash
from evalforge.schema.case import EvalCase
from evalforge.schema.result import EvaluatorResult
from evalforge.schema.trajectory import Trajectory


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    """What an evaluator gets to look at: the task, the trace, and the aftermath."""

    case: EvalCase
    workspace: Workspace
    trajectory: Trajectory


class Evaluator(ABC):
    """Scores one dimension of one attempt.

    Scores are normalised to ``[0, 1]`` so suites can be compared and aggregated
    without every consumer knowing each evaluator's native units; the raw units
    live in ``detail``, which is what regression analysis and clustering read.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def version(self) -> str:
        """Bump when scoring changes, so historical results are not silently mixed."""
        return "1"

    def config(self) -> dict[str, JsonValue]:
        return {}

    @abstractmethod
    async def evaluate(self, context: EvaluationContext) -> EvaluatorResult:
        """Score the attempt."""

    def describe(self) -> dict[str, JsonValue]:
        return {"name": self.name, "version": self.version, "config": self.config()}


@dataclass(frozen=True, slots=True)
class EvaluatorSuite:
    """An ordered set of evaluators plus the rule that turns them into a verdict.

    ``gating`` names the evaluators that decide pass/fail. The rest are measured
    but not decisive — trajectory quality should be *visible* without silently
    failing an agent that solved the task inelegantly.
    """

    name: str
    evaluators: tuple[Evaluator, ...]
    gating: frozenset[str]

    def __post_init__(self) -> None:
        names = [evaluator.name for evaluator in self.evaluators]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate evaluator names in suite {self.name!r}: {duplicates}")
        unknown = sorted(self.gating - set(names))
        if unknown:
            raise ValueError(f"gating names not present in suite {self.name!r}: {unknown}")
        if not self.gating:
            raise ValueError(f"suite {self.name!r} must gate on at least one evaluator")

    @property
    def content_hash(self) -> str:
        return content_hash(
            {
                "name": self.name,
                "gating": sorted(self.gating),
                "evaluators": [evaluator.describe() for evaluator in self.evaluators],
            }
        )

    async def evaluate(self, context: EvaluationContext) -> tuple[EvaluatorResult, ...]:
        """Run every evaluator in order.

        Sequential rather than concurrent on purpose: evaluators share one
        workspace, and running commands in it concurrently would let them
        observe each other's side effects.
        """
        results: list[EvaluatorResult] = []
        for evaluator in self.evaluators:
            results.append(await evaluator.evaluate(context))
        return tuple(results)

    def verdict(self, results: tuple[EvaluatorResult, ...]) -> bool:
        """Pass only when every gating evaluator passed."""
        by_name = {result.name: result for result in results}
        missing = sorted(self.gating - set(by_name))
        if missing:
            raise ValueError(f"missing results for gating evaluators: {missing}")
        return all(by_name[name].passed for name in self.gating)


async def evaluate_with_timeout(
    suite: EvaluatorSuite, context: EvaluationContext, *, timeout_s: float
) -> tuple[EvaluatorResult, ...]:
    """Run a suite under a deadline, so a hung evaluator cannot stall a run."""
    return await asyncio.wait_for(suite.evaluate(context), timeout=timeout_s)
