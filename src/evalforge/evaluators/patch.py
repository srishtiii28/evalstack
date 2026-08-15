"""Was the change well-scoped?

Passing tests say nothing about how much of the repository an agent disturbed on
the way. This evaluator measures edit locality: did the change land where the
task said the problem was, and did anything else get dragged along?

It is deliberately non-gating. A wide diff is a signal worth surfacing, not
grounds for calling a working fix a failure.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import JsonValue

from evalforge.evaluators.base import EvaluationContext, Evaluator
from evalforge.schema.result import EvaluatorResult


@dataclass(frozen=True, slots=True)
class PatchWeights:
    """Scoring knobs, exposed rather than buried so a team can disagree with them."""

    #: Deducted per file touched outside the case's declared target files.
    unrelated_file_penalty: float = 0.25
    #: Deducted when a declared target file was never touched at all.
    untouched_target_penalty: float = 0.5

    def describe(self) -> dict[str, JsonValue]:
        return {
            "unrelated_file_penalty": self.unrelated_file_penalty,
            "untouched_target_penalty": self.untouched_target_penalty,
        }


class PatchLocalityEvaluator(Evaluator):
    """Scores how tightly the diff matches the declared target files."""

    def __init__(self, weights: PatchWeights | None = None) -> None:
        self._weights = weights or PatchWeights()

    @property
    def name(self) -> str:
        return "patch_locality"

    def config(self) -> dict[str, JsonValue]:
        return self._weights.describe()

    async def evaluate(self, context: EvaluationContext) -> EvaluatorResult:
        # The diff frozen when the agent stopped — not recomputed here, which
        # would also capture whatever the outcome evaluator's test run created.
        diff = context.diff
        targets = set(context.case.metadata.target_files)
        touched = set(diff.touched)

        unrelated = tuple(sorted(touched - targets))
        untouched_targets = tuple(sorted(targets - touched))
        sorted_targets: list[JsonValue] = [*sorted(targets)]

        detail: dict[str, JsonValue] = {
            "added": list(diff.added),
            "modified": list(diff.modified),
            "deleted": list(diff.deleted),
            "touched": list(diff.touched),
            "unrelated_files": list(unrelated),
            "untouched_targets": list(untouched_targets),
            "target_files": sorted_targets,
        }

        if not touched:
            # Nothing changed: there is no patch to score, and calling that a
            # perfectly-scoped diff would be actively misleading.
            return EvaluatorResult(
                name=self.name,
                version=self.version,
                score=0.0,
                passed=False,
                detail=detail | {"reason": "no files were modified"},
            )

        penalty = self._weights.unrelated_file_penalty * len(unrelated)
        if targets:
            penalty += self._weights.untouched_target_penalty * bool(untouched_targets)

        score = max(0.0, min(1.0, 1.0 - penalty))
        return EvaluatorResult(
            name=self.name,
            version=self.version,
            score=score,
            passed=not unrelated and not untouched_targets,
            detail=detail,
        )
