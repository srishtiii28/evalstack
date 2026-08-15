"""Replaying a recorded trajectory without calling a model.

The property that matters is not that the transcript matches — it is that
re-executing the recorded tool calls lands in the same final workspace state, so
outcome evaluators can be re-run against history for free.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evalforge.agent.base import AgentContext
from evalforge.agent.replay import ReplayAgent, ReplayError, load_trajectory
from evalforge.agent.scripted import scripted_agent
from evalforge.datasets.builder import blueprint_to_case
from evalforge.datasets.catalogue import TEMPLATES
from evalforge.env.workspace import workspace_for
from evalforge.schema.case import EvalCase
from evalforge.schema.trajectory import ModelCall, Trajectory
from evalforge.trace import FakeClock, TrajectoryRecorder


def a_case(kind: str = "off_by_one") -> EvalCase:
    for template in TEMPLATES:
        blueprint = template("subject")
        if blueprint.kind == kind:
            return blueprint_to_case(blueprint, case_id=f"{kind}-000")
    raise AssertionError(f"no template produces {kind!r}")


def recorder_for(case: EvalCase) -> TrajectoryRecorder:
    return TrajectoryRecorder(
        run_id="run-1", case_id=case.case_id, clock=FakeClock(auto_advance_ms=1.0)
    )


async def record_original(case: EvalCase, base_dir: Path, policy: str = "oracle") -> Trajectory:
    recorder = recorder_for(case)
    with workspace_for(case, base_dir=base_dir) as workspace:
        await scripted_agent(policy).run(
            AgentContext(case=case, workspace=workspace, recorder=recorder)
        )
    return recorder.build()


async def test_replay_reaches_the_same_final_state(tmp_path: Path) -> None:
    case = a_case()
    original = await record_original(case, tmp_path)

    recorder = recorder_for(case)
    with workspace_for(case, base_dir=tmp_path) as workspace:
        await ReplayAgent(trajectory=original).run(
            AgentContext(case=case, workspace=workspace, recorder=recorder)
        )
        result = await workspace.run(case.test_command, timeout_s=120)
        diff = workspace.diff()

    assert result.ok is True
    assert diff.touched == case.metadata.target_files


async def test_replay_reproduces_the_tool_call_sequence(tmp_path: Path) -> None:
    case = a_case()
    original = await record_original(case, tmp_path)

    recorder = recorder_for(case)
    with workspace_for(case, base_dir=tmp_path) as workspace:
        await ReplayAgent(trajectory=original).run(
            AgentContext(case=case, workspace=workspace, recorder=recorder)
        )
    replayed = recorder.build()

    def shape(trajectory: Trajectory) -> list[tuple[str, str]]:
        return [(event.kind, getattr(event, "tool", "")) for event in trajectory.events]

    assert shape(replayed) == shape(original)


async def test_replay_costs_nothing(tmp_path: Path) -> None:
    case = a_case()
    original = await record_original(case, tmp_path)
    # Splice in a model call so there is a cost to *not* pay twice.
    with_cost = original.model_copy(
        update={
            "events": (
                *original.events,
                ModelCall(
                    seq=len(original.events),
                    t_ms=99.0,
                    model="test-model",
                    input_tokens=1_000,
                    output_tokens=200,
                    cost_usd=0.42,
                    latency_ms=50.0,
                ),
            )
        }
    )

    recorder = recorder_for(case)
    with workspace_for(case, base_dir=tmp_path) as workspace:
        await ReplayAgent(trajectory=with_cost).run(
            AgentContext(case=case, workspace=workspace, recorder=recorder)
        )
    replayed = recorder.build()

    model_calls = replayed.of_type(ModelCall)
    assert model_calls[0].cached is True
    assert replayed.total_cost_usd == 0.0
    # Tokens are still reported: they were really spent, once.
    assert replayed.total_input_tokens == 1_000


async def test_unreplayable_calls_are_reported_rather_than_silently_dropped(
    tmp_path: Path,
) -> None:
    case = a_case()
    recorder = recorder_for(case)
    recorder.tool_call(tool="write_file", args={"path": "solver/subject.py"})  # no contents
    recorder.tool_call(tool="teleport", args={})

    agent = ReplayAgent(trajectory=recorder.build())
    with workspace_for(case, base_dir=tmp_path) as workspace:
        await agent.run(
            AgentContext(case=case, workspace=workspace, recorder=recorder_for(case))
        )

    assert agent.skipped_calls == ("write_file", "teleport")


async def test_a_trajectory_round_trips_through_a_file(tmp_path: Path) -> None:
    case = a_case()
    original = await record_original(case, tmp_path)
    path = tmp_path / "run-1" / f"{case.case_id}--0.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(original.to_jsonl(), encoding="utf-8")

    loaded = load_trajectory(path)

    assert loaded.events == original.events
    assert loaded.case_id == case.case_id
    assert loaded.run_id == "run-1"


def test_a_missing_trajectory_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="no trajectory file"):
        load_trajectory(tmp_path / "nowhere.jsonl")


async def test_replaying_a_different_trace_is_a_different_agent(tmp_path: Path) -> None:
    case = a_case()
    solved = await record_original(case, tmp_path, policy="oracle")
    idle = await record_original(case, tmp_path, policy="idle")

    assert ReplayAgent(trajectory=solved).config_hash != ReplayAgent(trajectory=idle).config_hash
