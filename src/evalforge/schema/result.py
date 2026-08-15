"""Results: what an evaluator said, what a case did, and what a run measured."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, JsonValue

# Why an attempt ended. Only ``completed`` means "the agent had its fair shot";
# the others describe harness-level outcomes and are excluded from success rates
# so infrastructure noise never masquerades as agent quality.
CaseStatus = Literal["completed", "timed_out", "infra_error", "cancelled"]


class EvaluatorResult(BaseModel):
    """One evaluator's verdict on one attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str = "1"
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    detail: dict[str, JsonValue] = Field(default_factory=dict)


class CaseResult(BaseModel):
    """The outcome of one attempt at one case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    attempt: int = Field(ge=0)
    status: CaseStatus
    passed: bool
    evaluators: tuple[EvaluatorResult, ...] = ()
    duration_s: float = Field(ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    trajectory_path: str | None = None
    error: str | None = None

    def evaluator(self, name: str) -> EvaluatorResult | None:
        for result in self.evaluators:
            if result.name == name:
                return result
        return None


class CaseTally(NamedTuple):
    """How many of a case's attempts passed."""

    passed: int
    total: int

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


class RunResult(BaseModel):
    """Everything needed to reproduce, compare and audit a single evaluation run.

    The three content hashes are the reproducibility contract: an identical
    ``(agent_hash, dataset_hash, suite_hash)`` triple should reproduce the same
    verdicts, and a differing one explains why it did not.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    agent_ref: str
    agent_hash: str
    dataset_name: str
    dataset_version: str
    dataset_hash: str
    suite_name: str
    suite_hash: str
    samples_per_case: int = Field(default=1, ge=1)
    concurrency: int = Field(default=1, ge=1)
    case_results: tuple[CaseResult, ...] = ()
    notes: str = ""

    @property
    def dataset_ref(self) -> str:
        return f"{self.dataset_name}@{self.dataset_version}"

    @property
    def completed_results(self) -> tuple[CaseResult, ...]:
        return tuple(result for result in self.case_results if result.status == "completed")

    @property
    def success_rate(self) -> float:
        """Fraction of *completed* attempts that passed the suite."""
        completed = self.completed_results
        if not completed:
            return 0.0
        return sum(1 for result in completed if result.passed) / len(completed)

    @property
    def total_cost_usd(self) -> float:
        return sum(result.cost_usd for result in self.case_results)

    @property
    def total_input_tokens(self) -> int:
        return sum(result.input_tokens for result in self.case_results)

    @property
    def total_output_tokens(self) -> int:
        return sum(result.output_tokens for result in self.case_results)

    @property
    def mean_duration_s(self) -> float:
        """Mean wall time over *completed* attempts.

        Restricted to completed attempts for the same reason as
        :attr:`success_rate`: a run that timed out or hit an infrastructure
        fault says nothing about how long the agent takes when it works.
        """
        completed = self.completed_results
        if not completed:
            return 0.0
        return sum(result.duration_s for result in completed) / len(completed)

    def tallies(self) -> dict[str, CaseTally]:
        """Per-case pass counts over completed attempts, keyed by case id."""
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for result in self.completed_results:
            counts[result.case_id][1] += 1
            if result.passed:
                counts[result.case_id][0] += 1
        return {
            case_id: CaseTally(passed, total) for case_id, (passed, total) in sorted(counts.items())
        }

    def status_counts(self) -> dict[CaseStatus, int]:
        counts: dict[CaseStatus, int] = {}
        for result in self.case_results:
            counts[result.status] = counts.get(result.status, 0) + 1
        return counts

    def evaluator_scores(self) -> dict[str, float]:
        """Mean score per evaluator across completed attempts."""
        totals: dict[str, list[float]] = defaultdict(list)
        for result in self.completed_results:
            for evaluation in result.evaluators:
                totals[evaluation.name].append(evaluation.score)
        return {
            name: sum(scores) / len(scores) for name, scores in sorted(totals.items()) if scores
        }
