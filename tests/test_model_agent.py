"""The tool-use loop, driven by a scripted model.

No network and no key: a fake client returns whatever sequence of tool calls the
test needs. That is what makes the agent's error taxonomy testable — especially
the distinction between the agent behaving badly (recorded, attempt continues)
and the provider breaking (raised, scheduler decides).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evalforge.agent.base import AgentContext
from evalforge.agent.model_agent import ModelAgent, ModelAgentConfig
from evalforge.datasets.builder import blueprint_to_case
from evalforge.datasets.catalogue import TEMPLATES
from evalforge.env.workspace import workspace_for
from evalforge.model.base import (
    ModelBehaviourError,
    ModelRequest,
    ModelResponse,
    PermanentModelError,
    ToolInvocation,
    TransientModelError,
    Usage,
)
from evalforge.model.budget import BudgetExceeded
from evalforge.orchestrator.scheduler import FatalInfrastructureError, InfrastructureError
from evalforge.schema.case import EvalCase
from evalforge.schema.trajectory import AgentError, ModelCall, Submission, ToolCall, ToolResult
from evalforge.trace import FakeClock, TrajectoryRecorder

MODEL = "test-model"


def off_by_one_case() -> EvalCase:
    for template in TEMPLATES:
        blueprint = template("subject")
        if blueprint.kind == "off_by_one":
            return blueprint_to_case(blueprint, case_id="off_by_one-000")
    raise AssertionError("catalogue lost its off_by_one template")


def call(name: str, **arguments: object) -> ToolInvocation:
    return ToolInvocation(id=f"call-{name}", name=name, arguments=arguments)  # type: ignore[arg-type]


def turn(*calls: ToolInvocation, text: str = "") -> ModelResponse:
    return ModelResponse(
        model=MODEL,
        text=text,
        tool_calls=calls,
        usage=Usage(120, 40),
        stop_reason="tool_use" if calls else "end_turn",
        cost_usd=0.001,
        latency_ms=25.0,
    )


class ScriptedModel:
    """Returns a fixed sequence of turns, then raises if asked for more."""

    def __init__(self, *turns: ModelResponse | Exception) -> None:
        self._turns = list(turns)
        self.requests: list[ModelRequest] = []

    @property
    def model(self) -> str:
        return MODEL

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._turns:
            raise AssertionError("the agent asked for more turns than the script provides")
        nxt = self._turns.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


async def drive(client: ScriptedModel, *, case: EvalCase, base_dir: Path, max_steps: int = 12):
    recorder = TrajectoryRecorder(
        run_id="run-1", case_id=case.case_id, clock=FakeClock(auto_advance_ms=1.0)
    )
    agent = ModelAgent(client=client, settings=ModelAgentConfig(max_steps=max_steps))
    with workspace_for(case, base_dir=base_dir) as workspace:
        await agent.run(AgentContext(case=case, workspace=workspace, recorder=recorder))
        result = await workspace.run(case.test_command, timeout_s=120)
    return recorder.build(), result


# -- the happy path ------------------------------------------------------


async def test_a_full_loop_fixes_the_bug_and_submits(tmp_path: Path) -> None:
    case = off_by_one_case()
    fixed = case.reference_solution[0]

    client = ScriptedModel(
        turn(call("list_files")),
        turn(call("read_file", path=fixed.path)),
        turn(call("write_file", path=fixed.path, contents=fixed.contents)),
        turn(call("run_tests")),
        turn(call("submit", summary="Fixed the slice bound.")),
    )

    trajectory, result = await drive(client, case=case, base_dir=tmp_path)

    assert result.ok is True
    assert trajectory.submitted is True
    assert [c.tool for c in trajectory.of_type(ToolCall)] == [
        "list_files",
        "read_file",
        "write_file",
        "run_tests",
        "submit",
    ]
    assert len(trajectory.of_type(ModelCall)) == 5
    assert trajectory.total_cost_usd == pytest.approx(0.005)


async def test_the_loop_stops_as_soon_as_the_agent_submits(tmp_path: Path) -> None:
    case = off_by_one_case()
    client = ScriptedModel(turn(call("submit", summary="giving up")))

    trajectory, _ = await drive(client, case=case, base_dir=tmp_path)

    assert trajectory.of_type(Submission)[0].summary == "giving up"
    assert len(client.requests) == 1


async def test_several_tool_calls_in_one_turn_all_execute(tmp_path: Path) -> None:
    case = off_by_one_case()
    fixed = case.reference_solution[0]

    client = ScriptedModel(
        turn(call("list_files"), call("read_file", path=fixed.path)),
        turn(call("submit", summary="done")),
    )

    trajectory, _ = await drive(client, case=case, base_dir=tmp_path)

    assert [c.tool for c in trajectory.of_type(ToolCall)] == [
        "list_files",
        "read_file",
        "submit",
    ]


# -- misbehaviour is measured, not fatal ---------------------------------


async def test_a_failing_tool_returns_an_error_the_agent_can_act_on(tmp_path: Path) -> None:
    case = off_by_one_case()
    client = ScriptedModel(
        turn(call("read_file", path="solver/does_not_exist.py")),
        turn(call("submit", summary="gave up")),
    )

    trajectory, _ = await drive(client, case=case, base_dir=tmp_path)

    failed = [r for r in trajectory.of_type(ToolResult) if not r.ok]
    assert failed and "no such file" in (failed[0].error or "")
    assert trajectory.submitted is True


async def test_malformed_arguments_are_reported_back_to_the_model(tmp_path: Path) -> None:
    case = off_by_one_case()
    broken = ToolInvocation(
        id="call-1", name="write_file", malformed_arguments='{"path": "a.py", "contents"'
    )
    client = ScriptedModel(turn(broken), turn(call("submit", summary="done")))

    trajectory, _ = await drive(client, case=case, base_dir=tmp_path)

    tool_messages = [m for m in client.requests[-1].messages if m.role == "tool"]
    assert any("not valid JSON" in m.content for m in tool_messages)
    # A malformed call never reaches the workspace.
    assert not trajectory.of_type(ToolResult) or all(
        r.tool != "write_file" for r in trajectory.of_type(ToolResult)
    )


async def test_missing_arguments_are_rejected_without_touching_the_workspace(
    tmp_path: Path,
) -> None:
    case = off_by_one_case()
    client = ScriptedModel(
        turn(call("write_file", path="solver/subject.py")),
        turn(call("submit", summary="done")),
    )

    _, result = await drive(client, case=case, base_dir=tmp_path)

    tool_messages = [m for m in client.requests[-1].messages if m.role == "tool"]
    assert any("needs string 'path' and 'contents'" in m.content for m in tool_messages)
    assert result.ok is False


async def test_an_unknown_tool_lists_the_real_ones(tmp_path: Path) -> None:
    case = off_by_one_case()
    client = ScriptedModel(turn(call("delete_everything")), turn(call("submit", summary="done")))

    await drive(client, case=case, base_dir=tmp_path)

    tool_messages = [m for m in client.requests[-1].messages if m.role == "tool"]
    assert any("no tool named" in m.content and "run_tests" in m.content for m in tool_messages)


async def test_a_turn_with_no_tool_calls_is_nudged(tmp_path: Path) -> None:
    case = off_by_one_case()
    client = ScriptedModel(
        turn(text="I think the bug is in the slice."),
        turn(call("submit", summary="done")),
    )

    await drive(client, case=case, base_dir=tmp_path)

    prompts = [m.content for m in client.requests[-1].messages if m.role == "user"]
    assert any("did not call a tool" in p for p in prompts)


async def test_running_out_of_steps_is_recorded(tmp_path: Path) -> None:
    case = off_by_one_case()
    client = ScriptedModel(*[turn(call("list_files")) for _ in range(3)])

    trajectory, _ = await drive(client, case=case, base_dir=tmp_path, max_steps=3)

    errors = trajectory.of_type(AgentError)
    assert errors[0].error_type == "StepLimitReached"
    assert trajectory.submitted is False


async def test_exhausting_the_budget_ends_the_attempt_without_raising(tmp_path: Path) -> None:
    case = off_by_one_case()
    client = ScriptedModel(
        turn(call("list_files")),
        BudgetExceeded("call would use about 500 tokens, but only 10 remain"),
    )

    trajectory, _ = await drive(client, case=case, base_dir=tmp_path)

    errors = trajectory.of_type(AgentError)
    assert errors[0].error_type == "BudgetExceeded"
    # A ceiling is a fact about the attempt, so it is scored rather than raised.
    assert trajectory.of_type(ToolCall)[0].tool == "list_files"


# -- provider faults are infrastructure ----------------------------------


async def test_a_transient_provider_fault_propagates_for_retry(tmp_path: Path) -> None:
    case = off_by_one_case()
    client = ScriptedModel(TransientModelError("503 from provider"))

    with pytest.raises(InfrastructureError, match="model call failed"):
        await drive(client, case=case, base_dir=tmp_path)


async def test_a_permanent_provider_fault_is_not_retried(tmp_path: Path) -> None:
    case = off_by_one_case()
    client = ScriptedModel(PermanentModelError("invalid api key"))

    # Fatal so the scheduler records it once instead of burning the quota
    # rediscovering that the key is still wrong.
    with pytest.raises(FatalInfrastructureError, match="model rejected the request"):
        await drive(client, case=case, base_dir=tmp_path)


# -- configuration -------------------------------------------------------


def test_the_system_prompt_is_part_of_the_agent_hash() -> None:
    client = ScriptedModel()
    default = ModelAgent(client=client)
    tweaked = ModelAgent(client=client, settings=ModelAgentConfig(system_prompt="Be terse."))

    # Changing the prompt changes the agent, and the run hash must say so.
    assert default.config_hash != tweaked.config_hash


def test_the_agent_is_named_after_its_model() -> None:
    assert ModelAgent(client=ScriptedModel()).name == f"model:{MODEL}"


def test_nonsense_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="max_steps must be at least 1"):
        ModelAgentConfig(max_steps=0)


# -- a failed generation is the model's failure, not an outage ------------


async def test_a_failed_generation_is_scored_not_raised(tmp_path: Path) -> None:
    """Providers parse tool calls server-side and reject a bad one with a 4xx.

    Nothing about the request was wrong, so treating it as infrastructure would
    drop the case from the success rate and quietly flatter the agent.
    """
    case = off_by_one_case()
    client = ScriptedModel(
        turn(call("list_files")),
        ModelBehaviourError("the model produced an unusable tool call (HTTP 400)"),
    )

    trajectory, result = await drive(client, case=case, base_dir=tmp_path)

    errors = trajectory.of_type(AgentError)
    assert errors[0].error_type == "ModelBehaviourError"
    # The attempt completed and is scored on what it managed: nothing fixed.
    assert result.ok is False
    assert trajectory.of_type(ToolCall)[0].tool == "list_files"
