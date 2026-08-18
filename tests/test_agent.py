"""The tool surface and the deterministic agent built on it."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalforge.agent.base import AgentContext
from evalforge.agent.registry import agent_names, resolve_agent
from evalforge.agent.scripted import POLICIES, SCOPE_CREEP_PATH, scripted_agent
from evalforge.agent.tools import ToolBox
from evalforge.datasets.builder import blueprint_to_case
from evalforge.datasets.catalogue import TEMPLATES
from evalforge.env.workspace import Violation, workspace_for
from evalforge.schema.case import CaseMetadata, EvalCase, FileSpec
from evalforge.schema.trajectory import (
    CommandRun,
    FileEdit,
    SafetyViolation,
    Submission,
    ToolCall,
    ToolResult,
)
from evalforge.trace import FakeClock, TrajectoryRecorder

PASSING_COMMAND = ("python", "-c", "pass")
FAILING_COMMAND = ("python", "-c", "raise SystemExit(1)")


def make_case(*, test_command: tuple[str, ...] = PASSING_COMMAND, **overrides: object) -> EvalCase:
    defaults: dict[str, object] = {
        "case_id": "case-1",
        "prompt": "Fix the bug.",
        "files": (FileSpec(path="pkg/mod.py", contents="x = 1\n"),),
        "test_command": test_command,
        "metadata": CaseMetadata(bug_kind="off_by_one", target_files=("pkg/mod.py",)),
        "reference_solution": (FileSpec(path="pkg/mod.py", contents="x = 2\n"),),
    }
    return EvalCase.model_validate(defaults | overrides)


def make_recorder() -> TrajectoryRecorder:
    return TrajectoryRecorder(
        run_id="run-1", case_id="case-1", clock=FakeClock(auto_advance_ms=1.0)
    )


@pytest.fixture
def wired(tmp_path: Path):
    """A workspace, recorder and toolbox wired together the way the runner wires them."""
    case = make_case()
    recorder = make_recorder()

    def record(violation: Violation) -> None:
        recorder.safety_violation(
            rule=violation.rule, detail=violation.detail, attempted=violation.attempted
        )

    with workspace_for(case, base_dir=tmp_path, on_violation=record) as workspace:
        yield case, workspace, recorder, ToolBox(
            case=case, workspace=workspace, recorder=recorder
        )


# -- toolbox -------------------------------------------------------------


def test_list_files_records_a_call_and_result(wired) -> None:
    _case, _workspace, recorder, tools = wired

    outcome = tools.list_files()

    assert outcome.ok is True
    assert "pkg/mod.py" in outcome.output
    trajectory = recorder.build()
    assert trajectory.of_type(ToolCall)[0].tool == "list_files"
    assert trajectory.of_type(ToolResult)[0].ok is True


def test_read_file_returns_contents(wired) -> None:
    _case, _workspace, _recorder, tools = wired

    assert tools.read_file("pkg/mod.py").output == "x = 1\n"


def test_reading_a_missing_file_is_a_failed_outcome_not_an_exception(wired) -> None:
    _case, _workspace, recorder, tools = wired

    outcome = tools.read_file("pkg/nope.py")

    assert outcome.ok is False
    assert "no such file" in (outcome.error or "")
    assert recorder.build().of_type(ToolResult)[0].ok is False


def test_write_file_records_the_edit(wired) -> None:
    _case, workspace, recorder, tools = wired

    outcome = tools.write_file("pkg/mod.py", "x = 99\n")

    assert outcome.ok is True
    assert workspace.read_file("pkg/mod.py") == "x = 99\n"
    edit = recorder.build().of_type(FileEdit)[0]
    assert edit.path == "pkg/mod.py"
    assert edit.lines_added == 1
    # The tool layer must carry the workspace's diff through to the event; the
    # two are wired separately, so a passing hash check would not catch a drop.
    assert "+x = 99" in edit.diff


def test_escaping_writes_fail_the_tool_and_raise_a_safety_signal(wired) -> None:
    _case, _workspace, recorder, tools = wired

    outcome = tools.write_file("../escaped.txt", "nope")

    assert outcome.ok is False
    assert "outside the workspace" in (outcome.error or "")
    trajectory = recorder.build()
    # Contained, reported to the agent, and visible to the safety evaluator.
    assert trajectory.of_type(SafetyViolation)[0].rule == "path_escape"
    assert trajectory.of_type(ToolResult)[0].ok is False


async def test_run_tests_records_the_command_and_its_verdict(wired) -> None:
    _case, _workspace, recorder, tools = wired

    outcome = await tools.run_tests()

    assert outcome.ok is True
    assert tools.tests_passed is True
    assert recorder.build().of_type(CommandRun)[0].exit_code == 0


async def test_failing_tests_surface_the_exit_status(tmp_path: Path) -> None:
    case = make_case(test_command=FAILING_COMMAND)
    recorder = make_recorder()
    with workspace_for(case, base_dir=tmp_path) as workspace:
        tools = ToolBox(case=case, workspace=workspace, recorder=recorder)

        outcome = await tools.run_tests()

    assert outcome.ok is False
    assert "exited with status 1" in (outcome.error or "")
    assert tools.tests_passed is False


def test_tests_passed_is_unknown_before_any_run(wired) -> None:
    _case, _workspace, _recorder, tools = wired

    assert tools.tests_passed is None


def test_submit_records_a_submission(wired) -> None:
    _case, _workspace, recorder, tools = wired

    tools.submit(summary="all done")

    assert recorder.build().of_type(Submission)[0].summary == "all done"


# -- scripted agent ------------------------------------------------------


def blueprint_case(kind: str) -> EvalCase:
    for template in TEMPLATES:
        blueprint = template("subject")
        if blueprint.kind == kind:
            return blueprint_to_case(blueprint, case_id=f"{kind}-000")
    raise AssertionError(f"no template produces {kind!r}")


async def test_oracle_repairs_every_bug_kind(tmp_path: Path) -> None:
    for template in TEMPLATES:
        case = blueprint_to_case(template("subject"), case_id="probe")
        recorder = make_recorder()
        with workspace_for(case, base_dir=tmp_path) as workspace:
            await scripted_agent("oracle").run(
                AgentContext(case=case, workspace=workspace, recorder=recorder)
            )
            result = await workspace.run(case.test_command, timeout_s=120)

        assert result.ok is True, f"oracle failed to repair {case.metadata.bug_kind}"


async def test_idle_agent_touches_nothing(tmp_path: Path) -> None:
    case = blueprint_case("off_by_one")
    recorder = make_recorder()

    with workspace_for(case, base_dir=tmp_path) as workspace:
        await scripted_agent("idle").run(
            AgentContext(case=case, workspace=workspace, recorder=recorder)
        )
        diff = workspace.diff()

    assert diff.is_empty is True
    assert recorder.build().of_type(FileEdit) == ()


async def test_baseline_repairs_a_handled_kind(tmp_path: Path) -> None:
    case = blueprint_case("off_by_one")
    recorder = make_recorder()

    with workspace_for(case, base_dir=tmp_path) as workspace:
        await scripted_agent("baseline").run(
            AgentContext(case=case, workspace=workspace, recorder=recorder)
        )
        result = await workspace.run(case.test_command, timeout_s=120)

    assert result.ok is True


async def test_baseline_botches_an_unhandled_kind(tmp_path: Path) -> None:
    case = blueprint_case("missing_tiebreak")
    recorder = make_recorder()

    with workspace_for(case, base_dir=tmp_path) as workspace:
        await scripted_agent("baseline").run(
            AgentContext(case=case, workspace=workspace, recorder=recorder)
        )
        result = await workspace.run(case.test_command, timeout_s=120)
        diff = workspace.diff()

    # A botched attempt edits the right file without fixing anything: the common
    # real failure of doing work that does not help.
    assert result.ok is False
    assert diff.touched == case.metadata.target_files


async def test_regressed_agent_wastes_reads_and_creeps_in_scope(tmp_path: Path) -> None:
    case = blueprint_case("off_by_one")
    recorder = make_recorder()

    with workspace_for(case, base_dir=tmp_path) as workspace:
        await scripted_agent("regressed").run(
            AgentContext(case=case, workspace=workspace, recorder=recorder)
        )
        diff = workspace.diff()

    trajectory = recorder.build()
    reads = [call for call in trajectory.of_type(ToolCall) if call.tool == "read_file"]
    assert len(reads) == 3  # one genuine read plus two redundant ones
    assert SCOPE_CREEP_PATH in diff.touched


async def test_agent_summary_reflects_the_outcome(tmp_path: Path) -> None:
    case = blueprint_case("off_by_one")
    recorder = make_recorder()

    with workspace_for(case, base_dir=tmp_path) as workspace:
        await scripted_agent("oracle").run(
            AgentContext(case=case, workspace=workspace, recorder=recorder)
        )

    assert "suite passes" in recorder.build().of_type(Submission)[0].summary


# -- registry ------------------------------------------------------------


def test_policies_have_distinct_configuration_hashes() -> None:
    hashes = {name: scripted_agent(name).config_hash for name in POLICIES}

    assert len(set(hashes.values())) == len(POLICIES)


def test_agent_name_includes_the_policy() -> None:
    assert scripted_agent("baseline").name == "scripted:baseline"


def test_resolve_agent_builds_a_known_reference() -> None:
    assert resolve_agent("scripted:oracle").name == "scripted:oracle"


def test_resolve_agent_rejects_a_reference_without_a_family() -> None:
    with pytest.raises(ValueError, match="must look like 'family:variant'"):
        resolve_agent("baseline")


def test_resolve_agent_rejects_an_unknown_family() -> None:
    with pytest.raises(ValueError, match="unknown agent family"):
        resolve_agent("gpt:turbo")


def test_resolve_agent_rejects_an_unknown_policy() -> None:
    with pytest.raises(KeyError, match="unknown scripted policy"):
        resolve_agent("scripted:nonexistent")


def test_agent_names_lists_every_policy() -> None:
    assert set(agent_names()) == {f"scripted:{name}" for name in POLICIES}


# -- targeted editing ----------------------------------------------------


def test_replace_text_edits_one_occurrence(wired) -> None:
    _case, workspace, recorder, tools = wired

    outcome = tools.replace_text("pkg/mod.py", "x = 1", "x = 42")

    assert outcome.ok is True
    assert workspace.read_file("pkg/mod.py") == "x = 42\n"
    edit = recorder.build().of_type(FileEdit)[0]
    assert edit.path == "pkg/mod.py"
    assert "-x = 1" in edit.diff and "+x = 42" in edit.diff


def test_replace_text_refuses_an_ambiguous_match(wired) -> None:
    _case, workspace, _recorder, tools = wired
    workspace.write_file("pkg/mod.py", "y = 1\ny = 1\n")

    outcome = tools.replace_text("pkg/mod.py", "y = 1", "y = 2")

    # Guessing which occurrence was meant would be a silent wrong edit.
    assert outcome.ok is False
    assert "appears 2 times" in (outcome.error or "")
    assert workspace.read_file("pkg/mod.py") == "y = 1\ny = 1\n"


def test_replace_text_reports_a_missing_snippet(wired) -> None:
    _case, _workspace, _recorder, tools = wired

    outcome = tools.replace_text("pkg/mod.py", "not present", "anything")

    assert outcome.ok is False
    assert "does not appear" in (outcome.error or "")


def test_replace_text_is_contained_like_every_other_tool(wired) -> None:
    _case, _workspace, recorder, tools = wired

    outcome = tools.replace_text("../escaped.py", "a", "b")

    assert outcome.ok is False
    assert recorder.build().of_type(SafetyViolation)[0].rule == "path_escape"


def test_the_targeted_surface_adds_exactly_one_tool() -> None:
    from evalforge.agent.tools import TARGETED_SURFACE, WHOLE_FILE_SURFACE, tool_specs

    basic = {spec.name for spec in tool_specs(WHOLE_FILE_SURFACE)}
    targeted = {spec.name for spec in tool_specs(TARGETED_SURFACE)}

    assert targeted - basic == {"replace_text"}


def test_an_unknown_tool_surface_is_rejected() -> None:
    from evalforge.agent.tools import tool_specs

    with pytest.raises(ValueError, match="unknown tool surface"):
        tool_specs("imaginary")


def test_a_python_docstring_survives_a_targeted_edit(wired) -> None:
    """The defect that motivated the tool: triple quotes need no escaping here."""
    _case, workspace, _recorder, tools = wired
    workspace.write_file("pkg/mod.py", '"""Docs."""\n\ndef f():\n    return 1\n')

    outcome = tools.replace_text("pkg/mod.py", "return 1", "return 2")

    assert outcome.ok is True
    assert workspace.read_file("pkg/mod.py").startswith('"""Docs."""')
