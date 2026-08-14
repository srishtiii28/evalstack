"""The recorder, and the determinism guarantees it exists to provide."""

from __future__ import annotations

import pytest

from evalforge.schema.trajectory import (
    AgentError,
    CommandRun,
    FileEdit,
    ModelCall,
    SafetyViolation,
    Submission,
    TaskStarted,
    ToolCall,
    ToolResult,
)
from evalforge.trace import (
    FakeClock,
    MonotonicClock,
    TrajectoryRecorder,
    truncate,
)


def make_recorder(**overrides: object) -> TrajectoryRecorder:
    defaults: dict[str, object] = {
        "run_id": "run-1",
        "case_id": "case-1",
        "attempt": 0,
        "clock": FakeClock(auto_advance_ms=10.0),
    }
    return TrajectoryRecorder(**(defaults | overrides))  # type: ignore[arg-type]


def test_events_are_sequenced_and_timed_in_order() -> None:
    recorder = make_recorder()

    recorder.task_started(prompt_hash="sha256:abc")
    call_id = recorder.tool_call(tool="read_file", args={"path": "a.py"})
    recorder.tool_result(call_id=call_id, tool="read_file", ok=True, output="x = 1")

    trajectory = recorder.build()
    assert [event.seq for event in trajectory.events] == [0, 1, 2]
    assert [event.t_ms for event in trajectory.events] == [0.0, 10.0, 20.0]


def test_a_fake_clock_makes_trajectories_byte_identical() -> None:
    def record() -> str:
        recorder = make_recorder()
        recorder.task_started(prompt_hash="sha256:abc")
        recorder.tool_call(tool="list_files", args={})
        recorder.submission(summary="done")
        return recorder.build().to_jsonl()

    assert record() == record()


def test_call_ids_are_sequential_not_random() -> None:
    recorder = make_recorder()

    first = recorder.tool_call(tool="read_file", args={"path": "a.py"})
    second = recorder.tool_call(tool="read_file", args={"path": "b.py"})

    assert (first, second) == ("call-1", "call-2")


def test_every_event_type_is_recordable() -> None:
    recorder = make_recorder()

    recorder.task_started(prompt_hash="sha256:abc")
    recorder.model_call(
        model="claude-haiku-4-5",
        input_tokens=10,
        output_tokens=4,
        cost_usd=0.0001,
        latency_ms=42.0,
        cached=True,
        stop_reason="end_turn",
    )
    call_id = recorder.tool_call(tool="write_file", args={"path": "a.py"})
    recorder.tool_result(call_id=call_id, tool="write_file", ok=False, error="nope")
    recorder.file_edit(
        path="a.py", before_hash="sha256:1", after_hash="sha256:2", lines_added=1, lines_removed=1
    )
    recorder.command_run(
        argv=("python", "-m", "pytest"), exit_code=1, duration_ms=5.0, stdout="out", stderr="err"
    )
    recorder.safety_violation(rule="path_escape", detail="outside root", attempted="../x")
    recorder.agent_error(error_type="RuntimeError", message="boom")
    recorder.submission(summary="finished")

    trajectory = recorder.build()
    for event_type in (
        TaskStarted,
        ModelCall,
        ToolCall,
        ToolResult,
        FileEdit,
        CommandRun,
        SafetyViolation,
        AgentError,
        Submission,
    ):
        assert trajectory.of_type(event_type), f"{event_type.__name__} was not recorded"

    assert trajectory.total_cost_usd == pytest.approx(0.0001)
    assert trajectory.submitted is True


def test_long_output_is_truncated_in_the_middle() -> None:
    recorder = make_recorder(max_output_chars=100)
    call_id = recorder.tool_call(tool="read_file", args={"path": "big.py"})

    recorder.tool_result(call_id=call_id, tool="read_file", ok=True, output="A" * 500 + "TAIL")

    output = recorder.build().of_type(ToolResult)[0].output
    assert "characters omitted" in output
    assert output.startswith("A")
    assert output.endswith("TAIL")


def test_truncate_keeps_both_ends() -> None:
    result = truncate("HEAD" + "x" * 1000 + "TAIL", 50)

    assert result.startswith("HEAD")
    assert result.endswith("TAIL")
    assert "characters omitted" in result


def test_short_text_is_left_alone() -> None:
    assert truncate("short", 50) == "short"


def test_truncate_rejects_a_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        truncate("anything", 0)


def test_fake_clock_can_be_advanced_manually() -> None:
    clock = FakeClock()

    assert clock.elapsed_ms() == 0.0
    clock.advance(25.0)
    assert clock.elapsed_ms() == 25.0


def test_monotonic_clock_starts_at_zero_and_moves_forward() -> None:
    clock = MonotonicClock()

    first = clock.elapsed_ms()
    second = clock.elapsed_ms()

    assert first >= 0.0
    assert second >= first


def test_recorder_defaults_to_a_real_clock() -> None:
    recorder = TrajectoryRecorder(run_id="run-1", case_id="case-1")

    recorder.task_started(prompt_hash="sha256:abc")

    assert recorder.build().events[0].t_ms >= 0.0


def test_events_snapshot_does_not_alias_internal_state() -> None:
    recorder = make_recorder()
    recorder.task_started(prompt_hash="sha256:abc")

    snapshot = recorder.events
    recorder.submission(summary="later")

    assert len(snapshot) == 1
    assert len(recorder.events) == 2
