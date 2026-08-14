"""Was the agent's behaviour acceptable?

Two agents can both finish green while behaving completely differently. This
evaluator reads the trace and charges for the things that make an agent
expensive or unpredictable rather than wrong: re-reading files it has already
read, calling tools that fail, editing files it never looked at, and grinding
through more steps than the task warrants.

Every penalty is a named, configurable weight. A scoring function baked into
code is a scoring function nobody can argue with, and this one *should* be
arguable — different teams care about different behaviours.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from pydantic import JsonValue

from evalforge.evaluators.base import EvaluationContext, Evaluator
from evalforge.schema.result import EvaluatorResult
from evalforge.schema.trajectory import FileEdit, ToolCall, ToolResult, Trajectory


@dataclass(frozen=True, slots=True)
class TrajectoryWeights:
    """What each undesirable behaviour costs, in score points."""

    redundant_read: float = 0.10
    failed_tool_call: float = 0.15
    blind_edit: float = 0.10
    excess_tool_call: float = 0.02
    #: Tool calls allowed before ``excess_tool_call`` starts being charged.
    tool_call_budget: int = 12
    #: Credit returned for recovering after a failed test run.
    recovery_credit: float = 0.10
    #: Score at or above which the trajectory is considered acceptable.
    pass_threshold: float = 0.60

    def describe(self) -> dict[str, JsonValue]:
        return {
            "redundant_read": self.redundant_read,
            "failed_tool_call": self.failed_tool_call,
            "blind_edit": self.blind_edit,
            "excess_tool_call": self.excess_tool_call,
            "tool_call_budget": self.tool_call_budget,
            "recovery_credit": self.recovery_credit,
            "pass_threshold": self.pass_threshold,
        }


@dataclass(frozen=True, slots=True)
class TrajectorySignals:
    """Raw behavioural counts, before any weighting.

    Kept separate from the score because these are what failure clustering and
    cross-version comparison actually consume; the weighted score is a summary
    for humans.
    """

    tool_calls: int
    redundant_reads: int
    failed_tool_calls: int
    blind_edits: int
    files_read: int
    files_edited: int
    test_runs: int
    recovered_after_failure: bool

    def describe(self) -> dict[str, JsonValue]:
        return {
            "tool_calls": self.tool_calls,
            "redundant_reads": self.redundant_reads,
            "failed_tool_calls": self.failed_tool_calls,
            "blind_edits": self.blind_edits,
            "files_read": self.files_read,
            "files_edited": self.files_edited,
            "test_runs": self.test_runs,
            "recovered_after_failure": self.recovered_after_failure,
        }


def extract_signals(trajectory: Trajectory) -> TrajectorySignals:
    """Reduce a trace to the behavioural counts the score is built from."""
    calls = trajectory.of_type(ToolCall)
    results = trajectory.of_type(ToolResult)

    read_counts: Counter[str] = Counter()
    for call in calls:
        if call.tool == "read_file":
            path = call.args.get("path")
            if isinstance(path, str):
                read_counts[path] += 1

    edited_paths = {edit.path for edit in trajectory.of_type(FileEdit)}
    read_paths = set(read_counts)

    failed_test_seq: int | None = None
    recovered = False
    for result in results:
        if result.tool != "run_tests":
            continue
        if not result.ok:
            failed_test_seq = result.seq
        elif failed_test_seq is not None:
            # A green run after a red one, with an edit in between, is recovery.
            recovered = any(
                failed_test_seq < edit.seq < result.seq for edit in trajectory.of_type(FileEdit)
            )

    return TrajectorySignals(
        tool_calls=len(calls),
        redundant_reads=sum(count - 1 for count in read_counts.values() if count > 1),
        failed_tool_calls=sum(1 for result in results if not result.ok),
        blind_edits=len(edited_paths - read_paths),
        files_read=len(read_paths),
        files_edited=len(edited_paths),
        test_runs=sum(1 for call in calls if call.tool == "run_tests"),
        recovered_after_failure=recovered,
    )


class TrajectoryEvaluator(Evaluator):
    """Turns behavioural signals into a configurable efficiency score."""

    def __init__(self, weights: TrajectoryWeights | None = None) -> None:
        self._weights = weights or TrajectoryWeights()

    @property
    def name(self) -> str:
        return "trajectory"

    def config(self) -> dict[str, JsonValue]:
        return self._weights.describe()

    def score_signals(self, signals: TrajectorySignals) -> float:
        weights = self._weights
        excess_calls = max(0, signals.tool_calls - weights.tool_call_budget)

        penalty = (
            weights.redundant_read * signals.redundant_reads
            + weights.failed_tool_call * signals.failed_tool_calls
            + weights.blind_edit * signals.blind_edits
            + weights.excess_tool_call * excess_calls
        )
        credit = weights.recovery_credit if signals.recovered_after_failure else 0.0
        return max(0.0, min(1.0, 1.0 - penalty + credit))

    async def evaluate(self, context: EvaluationContext) -> EvaluatorResult:
        signals = extract_signals(context.trajectory)
        score = self.score_signals(signals)

        detail: dict[str, JsonValue] = dict(signals.describe())
        detail["excess_tool_calls"] = max(
            0, signals.tool_calls - self._weights.tool_call_budget
        )

        return EvaluatorResult(
            name=self.name,
            version=self.version,
            score=score,
            passed=score >= self._weights.pass_threshold,
            detail=detail,
        )
