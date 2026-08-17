"""A model-based judge, for the properties deterministic checks cannot reach.

Some questions have no test: did the fix address what was actually asked, or
merely make the assertions pass? A judge can answer those, and is the only
evaluator here whose output cannot be verified from its own behaviour — which is
why :mod:`evalforge.judge_eval` exists and why every result carries the judge's
model and prompt hash. A score produced by a different judge is a different
measurement, and mixing them silently is the failure this guards against.

The verdict arrives through a tool call rather than as prose to be parsed. That
reuses the transport's existing structured-output path, including its handling
of arguments that are not valid JSON, instead of inventing a second and worse
parser for the same problem.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import JsonValue

from evalforge.evaluators.base import EvaluationContext, Evaluator
from evalforge.model.base import (
    ModelClient,
    ModelRequest,
    ToolInvocation,
    ToolSpec,
    system,
    user,
)
from evalforge.schema.result import EvaluatorResult
from evalforge.schema.trajectory import CommandRun, FileEdit, Submission, ToolCall

PASS = "pass"
FAIL = "fail"
DEFAULT_MAX_TOKENS = 512
MAX_OUTPUT_EXCERPT = 1_500

DEFAULT_JUDGE_PROMPT = """\
You are reviewing whether a software engineering agent actually solved the task
it was given.

Judge the substance, not the effort. A change that makes tests pass without
addressing the described problem is a failure. So is a fix that works but
rewrites unrelated code.

Answer only by calling record_verdict. Be decisive: reserve low confidence for
cases where the evidence genuinely does not settle it.\
"""

VERDICT_TOOL = ToolSpec(
    name="record_verdict",
    description="Record your judgement of whether the agent solved the task.",
    parameters={
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": [PASS, FAIL],
                "description": "pass if the agent genuinely solved the task.",
            },
            "reason": {
                "type": "string",
                "description": "One or two sentences justifying the verdict.",
            },
            "confidence": {
                "type": "number",
                "description": "0 to 1. How settled the evidence is.",
            },
            "failure_category": {
                "type": "string",
                "description": (
                    "When failing, a short slug for the kind of failure, e.g. "
                    "wrong-fix, incomplete, unrelated-changes, no-change."
                ),
            },
        },
        "required": ["verdict", "reason"],
    },
)


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """A parsed verdict, or a record of the judge failing to give one."""

    verdict: str
    reason: str = ""
    confidence: float = 0.0
    failure_category: str = ""
    answered: bool = True

    @property
    def passed(self) -> bool:
        return self.answered and self.verdict == PASS


def summarise_trajectory(context: EvaluationContext) -> str:
    """A compact account of what the agent did, for the judge to read.

    Deliberately not the raw trace: a judge given fifty events spends its
    attention on transcript mechanics rather than on whether the task was
    solved, and the trace is already available to deterministic evaluators.
    """
    trajectory = context.trajectory
    tools = [call.tool for call in trajectory.of_type(ToolCall)]
    edits = trajectory.of_type(FileEdit)
    commands = trajectory.of_type(CommandRun)
    submissions = trajectory.of_type(Submission)

    lines = [
        f"Tools used, in order: {', '.join(tools) if tools else '(none)'}",
        f"Files changed: {', '.join(sorted({edit.path for edit in edits})) or '(none)'}",
        f"Files touched vs the task's stated targets: "
        f"{', '.join(context.diff.touched) or '(none)'}",
    ]
    if commands:
        last = commands[-1]
        status = "passed" if last.exit_code == 0 else f"failed (exit {last.exit_code})"
        lines.append(f"Final test run: {status}")
        excerpt = (last.stdout_tail or last.stderr_tail).strip()
        if excerpt:
            lines.append(f"Test output:\n{excerpt[:MAX_OUTPUT_EXCERPT]}")
    else:
        lines.append("Final test run: the agent never ran the tests")
    if submissions:
        lines.append(f"Agent's own summary: {submissions[-1].summary or '(none given)'}")

    return "\n".join(lines)


def parse_verdict(call: ToolInvocation | None) -> JudgeVerdict:
    """Turn the judge's tool call into a verdict, or record that it did not answer."""
    if call is None or call.name != VERDICT_TOOL.name:
        return JudgeVerdict(
            verdict=FAIL, reason="the judge did not record a verdict", answered=False
        )
    if call.malformed_arguments is not None:
        return JudgeVerdict(
            verdict=FAIL,
            reason=f"the judge's arguments were not valid JSON: {call.malformed_arguments[:200]}",
            answered=False,
        )

    raw = call.arguments.get("verdict")
    verdict = raw.strip().lower() if isinstance(raw, str) else ""
    if verdict not in (PASS, FAIL):
        return JudgeVerdict(
            verdict=FAIL, reason=f"the judge returned an unusable verdict {raw!r}", answered=False
        )

    reason = call.arguments.get("reason")
    confidence = call.arguments.get("confidence")
    category = call.arguments.get("failure_category")
    return JudgeVerdict(
        verdict=verdict,
        reason=reason if isinstance(reason, str) else "",
        confidence=float(confidence) if isinstance(confidence, int | float) else 0.0,
        failure_category=category if isinstance(category, str) else "",
    )


class LLMJudgeEvaluator(Evaluator):
    """Asks a model whether the agent genuinely solved the task."""

    def __init__(
        self,
        client: ModelClient,
        *,
        prompt: str = DEFAULT_JUDGE_PROMPT,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._client = client
        self._prompt = prompt
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return "llm_judge"

    def config(self) -> dict[str, JsonValue]:
        # Recorded on every run, so a changed judge invalidates comparisons
        # loudly rather than silently shifting the numbers.
        return {
            "model": self._client.model,
            "prompt": self._prompt,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

    def build_request(self, context: EvaluationContext) -> ModelRequest:
        case = context.case
        expected = case.metadata.bug_kind or "not stated"
        content = (
            f"TASK GIVEN TO THE AGENT:\n{case.prompt}\n\n"
            f"KNOWN FAULT CATEGORY: {expected}\n\n"
            f"WHAT THE AGENT DID:\n{summarise_trajectory(context)}"
        )
        return ModelRequest(
            messages=(system(self._prompt), user(content)),
            tools=(VERDICT_TOOL,),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

    async def evaluate(self, context: EvaluationContext) -> EvaluatorResult:
        response = await self._client.complete(self.build_request(context))
        call = response.tool_calls[0] if response.tool_calls else None
        verdict = parse_verdict(call)

        detail: dict[str, JsonValue] = {
            "verdict": verdict.verdict,
            "reason": verdict.reason,
            "confidence": verdict.confidence,
            "failure_category": verdict.failure_category,
            # False means the judge failed to answer. Such a result is missing
            # data, not evidence against the agent, and should be filtered out
            # rather than averaged in.
            "answered": verdict.answered,
            "judge_model": response.model,
            "judge_cached": response.cached,
        }

        return EvaluatorResult(
            name=self.name,
            version=self.version,
            score=1.0 if verdict.passed else 0.0,
            passed=verdict.passed,
            detail=detail,
        )
