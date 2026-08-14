"""Did the agent actually accomplish the task?

Outcome is measured by re-running the case's own suite against the final
workspace state, rather than by trusting whatever the agent last observed. An
agent that runs the tests, sees green, and then keeps editing has not earned the
pass its own transcript claims.
"""

from __future__ import annotations

import re

from pydantic import JsonValue

from evalforge.evaluators.base import EvaluationContext, Evaluator
from evalforge.schema.result import EvaluatorResult

#: pytest's terminal summary, e.g. "2 failed, 3 passed in 0.11s".
_SUMMARY_COUNT_RE = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed)\b")

_VERIFICATION_TIMEOUT_MARGIN_S = 30.0


def parse_pytest_counts(output: str) -> dict[str, int]:
    """Extract per-outcome test counts from pytest's summary line.

    Returns an empty mapping when no summary is present — a crashed or
    non-pytest command should report *nothing*, not a misleading zero.
    """
    counts: dict[str, int] = {}
    for line in reversed(output.splitlines()):
        matches = _SUMMARY_COUNT_RE.findall(line)
        if not matches:
            continue
        for value, label in matches:
            key = "error" if label.startswith("error") else label
            counts[key] = counts.get(key, 0) + int(value)
        break
    return counts


class SuiteOutcomeEvaluator(Evaluator):
    """Runs the case's test command and passes only on a clean exit."""

    @property
    def name(self) -> str:
        return "tests"

    def config(self) -> dict[str, JsonValue]:
        return {"timeout_margin_s": _VERIFICATION_TIMEOUT_MARGIN_S}

    async def evaluate(self, context: EvaluationContext) -> EvaluatorResult:
        case = context.case
        result = await context.workspace.run(
            case.test_command,
            timeout_s=case.timeout_s + _VERIFICATION_TIMEOUT_MARGIN_S,
        )

        counts = parse_pytest_counts(result.stdout or result.stderr)
        detail: dict[str, JsonValue] = {
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "command": " ".join(case.test_command),
            "duration_ms": round(result.duration_ms, 3),
            "counts": dict(sorted(counts.items())),
        }

        return EvaluatorResult(
            name=self.name,
            version=self.version,
            score=1.0 if result.ok else 0.0,
            passed=result.ok,
            detail=detail,
        )
