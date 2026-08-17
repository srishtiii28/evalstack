"""The model-based judge, driven offline by a scripted client.

A judge that can only be tested by spending money is a judge nobody tests — and
this is the one evaluator whose output cannot be checked against anything
observable, so its failure modes need covering most.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evalforge.env.workspace import workspace_for
from evalforge.evaluators.base import EvaluationContext
from evalforge.evaluators.llm_judge import (
    DEFAULT_JUDGE_PROMPT,
    VERDICT_TOOL,
    LLMJudgeEvaluator,
    parse_verdict,
    summarise_trajectory,
)
from evalforge.model.base import ModelRequest, ModelResponse, ToolInvocation, Usage
from evalforge.schema.case import CaseMetadata, EvalCase, FileSpec
from evalforge.schema.trajectory import Trajectory
from evalforge.trace import FakeClock, TrajectoryRecorder

MODEL = "judge-model"


def make_case() -> EvalCase:
    return EvalCase(
        case_id="case-1",
        prompt="last_n returns one item too few. Fix it.",
        files=(FileSpec(path="pkg/mod.py", contents="x = 1\n"),),
        test_command=("python", "-c", "pass"),
        metadata=CaseMetadata(bug_kind="off_by_one", target_files=("pkg/mod.py",)),
    )


class ScriptedJudge:
    """Returns one canned response, and records what it was asked."""

    def __init__(self, response: ModelResponse) -> None:
        self._response = response
        self.requests: list[ModelRequest] = []

    @property
    def model(self) -> str:
        return MODEL

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self._response


def verdict_response(**arguments: object) -> ModelResponse:
    return ModelResponse(
        model=MODEL,
        tool_calls=(
            ToolInvocation(id="call-1", name=VERDICT_TOOL.name, arguments=arguments),  # type: ignore[arg-type]
        ),
        stop_reason="tool_use",
        usage=Usage(300, 40),
    )


async def judge_case(response: ModelResponse, tmp_path: Path, *, script=None):
    case = make_case()
    recorder = TrajectoryRecorder(
        run_id="run-1", case_id=case.case_id, clock=FakeClock(auto_advance_ms=1.0)
    )
    if script is not None:
        script(recorder)
    client = ScriptedJudge(response)
    with workspace_for(case, base_dir=tmp_path) as workspace:
        context = EvaluationContext(
            case=case,
            workspace=workspace,
            trajectory=recorder.build(),
            diff=workspace.diff(),
        )
        result = await LLMJudgeEvaluator(client).evaluate(context)
    return result, client


# -- verdicts ------------------------------------------------------------


async def test_a_pass_verdict_is_recorded(tmp_path: Path) -> None:
    result, _ = await judge_case(
        verdict_response(verdict="pass", reason="Adjusted the slice bound correctly.",
                         confidence=0.9),
        tmp_path,
    )

    assert result.passed is True
    assert result.score == 1.0
    assert result.detail["answered"] is True
    assert result.detail["confidence"] == 0.9
    assert result.detail["judge_model"] == MODEL


async def test_a_fail_verdict_carries_its_category(tmp_path: Path) -> None:
    result, _ = await judge_case(
        verdict_response(
            verdict="fail", reason="Tests pass but the described bug is untouched.",
            failure_category="wrong-fix",
        ),
        tmp_path,
    )

    assert result.passed is False
    assert result.detail["failure_category"] == "wrong-fix"


async def test_verdicts_are_case_insensitive(tmp_path: Path) -> None:
    result, _ = await judge_case(verdict_response(verdict="PASS", reason="fine"), tmp_path)

    assert result.passed is True


# -- the judge failing to answer -----------------------------------------


async def test_a_judge_that_answers_in_prose_is_marked_unanswered(tmp_path: Path) -> None:
    result, _ = await judge_case(
        ModelResponse(model=MODEL, text="I think it looks fine, honestly."), tmp_path
    )

    assert result.detail["answered"] is False
    assert "did not record a verdict" in str(result.detail["reason"])


async def test_malformed_arguments_are_not_treated_as_a_verdict(tmp_path: Path) -> None:
    response = ModelResponse(
        model=MODEL,
        tool_calls=(
            ToolInvocation(
                id="call-1", name=VERDICT_TOOL.name, malformed_arguments='{"verdict": "pa'
            ),
        ),
        stop_reason="tool_use",
    )

    result, _ = await judge_case(response, tmp_path)

    assert result.detail["answered"] is False
    assert "not valid JSON" in str(result.detail["reason"])


async def test_an_unusable_verdict_string_is_rejected(tmp_path: Path) -> None:
    result, _ = await judge_case(verdict_response(verdict="maybe", reason="unsure"), tmp_path)

    assert result.detail["answered"] is False


def test_an_unanswered_judgement_is_missing_data_not_evidence() -> None:
    """The distinction that keeps a broken judge from looking like a bad agent."""
    verdict = parse_verdict(None)

    assert verdict.answered is False
    assert verdict.passed is False


def test_a_call_to_the_wrong_tool_is_not_a_verdict() -> None:
    verdict = parse_verdict(ToolInvocation(id="c", name="something_else", arguments={}))

    assert verdict.answered is False


# -- what the judge is shown ---------------------------------------------


async def test_the_judge_sees_the_task_and_a_summary_of_the_work(tmp_path: Path) -> None:
    def script(recorder: TrajectoryRecorder) -> None:
        call = recorder.tool_call(tool="read_file", args={"path": "pkg/mod.py"})
        recorder.tool_result(call_id=call, tool="read_file", ok=True, output="x = 1")
        recorder.command_run(
            argv=("python", "-m", "pytest"), exit_code=1, duration_ms=5.0, stdout="1 failed"
        )
        recorder.submission(summary="Adjusted the bound.")

    _, client = await judge_case(
        verdict_response(verdict="fail", reason="tests still red"), tmp_path, script=script
    )

    content = client.requests[0].messages[1].content
    assert "last_n returns one item too few" in content
    assert "off_by_one" in content
    assert "read_file" in content
    assert "failed (exit 1)" in content
    assert "Adjusted the bound." in content


async def test_the_judge_is_told_when_the_agent_never_ran_the_tests(tmp_path: Path) -> None:
    _, client = await judge_case(verdict_response(verdict="fail", reason="no evidence"), tmp_path)

    assert "never ran the tests" in client.requests[0].messages[1].content


def test_the_summary_reports_an_untouched_workspace(tmp_path: Path) -> None:
    case = make_case()
    with workspace_for(case, base_dir=tmp_path) as workspace:
        context = EvaluationContext(
            case=case,
            workspace=workspace,
            trajectory=Trajectory(run_id="r", case_id="c", attempt=0),
            diff=workspace.diff(),
        )
        summary = summarise_trajectory(context)

    assert "(none)" in summary


async def test_the_judge_is_asked_for_a_structured_verdict(tmp_path: Path) -> None:
    _, client = await judge_case(verdict_response(verdict="pass", reason="ok"), tmp_path)

    request = client.requests[0]
    assert [tool.name for tool in request.tools] == [VERDICT_TOOL.name]
    assert request.temperature == 0.0


# -- identity ------------------------------------------------------------


def test_the_prompt_and_model_are_part_of_the_evaluator_config() -> None:
    client = ScriptedJudge(verdict_response(verdict="pass", reason="ok"))

    default = LLMJudgeEvaluator(client)
    tweaked = LLMJudgeEvaluator(client, prompt="Be much stricter.")

    assert default.config()["prompt"] == DEFAULT_JUDGE_PROMPT
    # A changed judge prompt must change the suite hash, or two runs would claim
    # the same measurement while using different instruments.
    assert default.describe() != tweaked.describe()


def test_temperature_is_configuration_too() -> None:
    client = ScriptedJudge(verdict_response(verdict="pass", reason="ok"))

    assert LLMJudgeEvaluator(client).describe() != (
        LLMJudgeEvaluator(client, temperature=0.7).describe()
    )


@pytest.mark.parametrize("confidence", [0, 1, 0.5])
def test_numeric_confidence_is_accepted_in_any_form(confidence: float) -> None:
    verdict = parse_verdict(
        ToolInvocation(
            id="c",
            name=VERDICT_TOOL.name,
            arguments={"verdict": "pass", "reason": "ok", "confidence": confidence},
        )
    )

    assert verdict.confidence == pytest.approx(float(confidence))
