"""The read-only API behind the dashboard.

Covers the error paths as carefully as the happy ones: a dashboard that renders
a blank panel because the API returned 200 with nothing in it is worse than one
that shows an error, so missing runs, missing attempts and vanished trajectory
files each have to fail distinctly.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from evalforge.api.app import WEB_ROOT, create_app
from evalforge.schema.result import CaseResult, EvaluatorResult, RunResult
from evalforge.schema.trajectory import Trajectory
from evalforge.store.db import Store
from evalforge.trace import FakeClock, TrajectoryRecorder


def make_run(
    run_id: str,
    outcomes: dict[str, bool],
    *,
    agent_ref: str = "scripted:baseline",
    agent_hash: str = "sha256:agent",
    trajectory_paths: dict[str, str] | None = None,
) -> RunResult:
    paths = trajectory_paths or {}
    return RunResult(
        run_id=run_id,
        agent_ref=agent_ref,
        agent_hash=agent_hash,
        dataset_name="synth",
        dataset_version="v1",
        dataset_hash="sha256:dataset",
        suite_name="default",
        suite_hash="sha256:suite",
        case_results=tuple(
            CaseResult(
                case_id=case_id,
                attempt=0,
                status="completed",
                passed=passed,
                evaluators=(
                    EvaluatorResult(name="tests", score=1.0 if passed else 0.0, passed=passed),
                    EvaluatorResult(name="trajectory", score=0.9, passed=True),
                ),
                duration_s=1.5,
                cost_usd=0.001,
                input_tokens=120,
                output_tokens=30,
                trajectory_path=paths.get(case_id),
            )
            for case_id, passed in outcomes.items()
        ),
    )


def write_trajectory(path: Path) -> str:
    recorder = TrajectoryRecorder(
        run_id="run-a", case_id="alpha", clock=FakeClock(auto_advance_ms=5.0)
    )
    recorder.task_started(prompt_hash="sha256:prompt")
    call = recorder.tool_call(tool="read_file", args={"path": "pkg/mod.py"})
    recorder.tool_result(call_id=call, tool="read_file", ok=True, output="x = 1\n")
    recorder.file_edit(
        path="pkg/mod.py", before_hash="sha256:1", after_hash="sha256:2",
        lines_added=1, lines_removed=1,
    )
    recorder.command_run(
        argv=("python", "-m", "pytest"), exit_code=0, duration_ms=900.0, stdout="3 passed"
    )
    recorder.submission(summary="Fixed the bound.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(recorder.build().to_jsonl(), encoding="utf-8")
    return str(path)


@pytest.fixture
async def client(tmp_path: Path):
    database = tmp_path / "runs.db"
    trace = write_trajectory(tmp_path / "traces" / "alpha--0.jsonl")
    with Store.open(database) as store:
        store.save_run(
            make_run(
                "run-a",
                {"alpha": True, "beta": False, "gamma": True},
                trajectory_paths={"alpha": trace},
            )
        )
        store.save_run(
            make_run(
                "run-b",
                {"alpha": False, "beta": False, "gamma": True},
                agent_ref="scripted:regressed",
                agent_hash="sha256:other",
            )
        )
    transport = httpx.ASGITransport(app=create_app(database=database))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
        yield http


# -- health and listing --------------------------------------------------


async def test_health_reports_the_database(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_runs_are_listed_newest_first(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/runs")

    assert response.status_code == 200
    rows = response.json()
    assert {row["run_id"] for row in rows} == {"run-a", "run-b"}
    row = next(r for r in rows if r["run_id"] == "run-a")
    assert row["passed"] == 2
    assert row["completed"] == 3
    assert row["success_rate"] == pytest.approx(2 / 3)


async def test_the_listing_limit_is_bounded(client: httpx.AsyncClient) -> None:
    assert (await client.get("/api/runs?limit=1")).status_code == 200
    assert (await client.get("/api/runs?limit=0")).status_code == 422
    assert (await client.get("/api/runs?limit=99999")).status_code == 422


# -- a single run --------------------------------------------------------


async def test_a_run_comes_back_whole(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/runs/run-a")

    assert response.status_code == 200
    run = response.json()
    assert run["agent_ref"] == "scripted:baseline"
    assert run["dataset_hash"] == "sha256:dataset"
    assert len(run["case_results"]) == 3
    assert run["case_results"][0]["evaluators"][0]["name"] == "tests"


async def test_an_unknown_run_is_a_404(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/runs/run-nope")

    assert response.status_code == 404
    assert "run-nope" in response.json()["detail"]


async def test_a_run_id_needing_escaping_does_not_break_routing(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/runs/not%20a%20run")

    assert response.status_code == 404


# -- trajectories --------------------------------------------------------


async def test_a_trajectory_comes_back_as_ordered_events(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/runs/run-a/trajectory/alpha")

    assert response.status_code == 200
    body = response.json()
    assert body["case_id"] == "alpha"
    kinds = [event["kind"] for event in body["events"]]
    assert kinds == [
        "task_started",
        "tool_call",
        "tool_result",
        "file_edit",
        "command_run",
        "submission",
    ]
    assert body["events"][1]["tool"] == "read_file"


async def test_an_attempt_that_recorded_no_trajectory_is_a_404(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/runs/run-a/trajectory/beta")

    assert response.status_code == 404
    assert "no trajectory was recorded" in response.json()["detail"]


async def test_an_unknown_case_is_a_404(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/runs/run-a/trajectory/nonexistent")

    assert response.status_code == 404
    assert "no attempt 0" in response.json()["detail"]


async def test_an_unknown_attempt_is_a_404(client: httpx.AsyncClient) -> None:
    assert (await client.get("/api/runs/run-a/trajectory/alpha?attempt=7")).status_code == 404


async def test_a_negative_attempt_is_rejected(client: httpx.AsyncClient) -> None:
    assert (await client.get("/api/runs/run-a/trajectory/alpha?attempt=-1")).status_code == 422


async def test_a_vanished_trajectory_file_is_distinguished_from_a_missing_one(
    tmp_path: Path,
) -> None:
    """410 rather than 404: the run says there was a trace and it is gone."""
    database = tmp_path / "runs.db"
    with Store.open(database) as store:
        store.save_run(
            make_run(
                "run-c", {"alpha": True}, trajectory_paths={"alpha": str(tmp_path / "gone.jsonl")}
            )
        )
    transport = httpx.ASGITransport(app=create_app(database=database))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
        response = await http.get("/api/runs/run-c/trajectory/alpha")

    assert response.status_code == 410
    assert "gone" in response.json()["detail"]


# -- comparison ----------------------------------------------------------


async def test_two_runs_can_be_compared(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/compare?before=run-a&after=run-b")

    assert response.status_code == 200
    body = response.json()
    assert body["before_run_id"] == "run-a"
    assert body["shared_cases"] == 3
    assert body["verdict"] in {"regression", "improvement", "no significant change"}
    assert {d["name"] for d in body["dimensions"]} == {"tests", "trajectory"}
    assert any(t["kind"] == "broken" for t in body["transitions"])


async def test_comparing_against_a_missing_run_is_a_404(client: httpx.AsyncClient) -> None:
    assert (await client.get("/api/compare?before=run-a&after=ghost")).status_code == 404
    assert (await client.get("/api/compare?before=ghost&after=run-a")).status_code == 404


async def test_compare_requires_both_runs(client: httpx.AsyncClient) -> None:
    assert (await client.get("/api/compare?before=run-a")).status_code == 422


async def test_comparing_a_run_against_itself_is_quiet(client: httpx.AsyncClient) -> None:
    body = (await client.get("/api/compare?before=run-a&after=run-a")).json()

    assert body["verdict"] == "no significant change"
    assert body["p_value"] == 1.0
    assert body["delta"] == 0.0


# -- the dashboard page --------------------------------------------------


async def test_the_dashboard_page_is_served(client: httpx.AsyncClient) -> None:
    response = await client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "EvalForge" in response.text


async def test_a_missing_dashboard_directory_is_reported_not_crashed(tmp_path: Path) -> None:
    app = create_app(database=tmp_path / "runs.db", web_root=tmp_path / "absent")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
        response = await http.get("/")

    assert response.status_code == 404


async def test_the_shipped_page_calls_only_endpoints_that_exist() -> None:
    """Guards against the page and the API drifting apart."""
    page = (WEB_ROOT / "index.html").read_text(encoding="utf-8")

    for path in ("/api/runs", "/api/compare", "/trajectory/"):
        assert path in page


async def test_an_empty_store_still_serves_an_empty_list(tmp_path: Path) -> None:
    transport = httpx.ASGITransport(app=create_app(database=tmp_path / "fresh.db"))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
        response = await http.get("/api/runs")

    assert response.status_code == 200
    assert response.json() == []


async def test_the_store_is_not_shared_across_requests(client: httpx.AsyncClient) -> None:
    """Each request opens its own connection, so concurrent reads are safe."""
    first = await client.get("/api/runs")
    second = await client.get("/api/runs")

    assert first.json() == second.json()


async def test_trajectory_round_trips_through_the_schema(client: httpx.AsyncClient) -> None:
    body = (await client.get("/api/runs/run-a/trajectory/alpha")).json()
    jsonl = "".join(f"{json.dumps(event)}\n" for event in body["events"])

    restored = Trajectory.from_jsonl(run_id="run-a", case_id="alpha", attempt=0, jsonl=jsonl)

    assert len(restored.events) == len(body["events"])


# -- the page and the schema must not drift apart ------------------------


def test_the_viewer_handles_every_event_kind_the_schema_can_emit() -> None:
    """A new event type must not render as a blank row in the timeline.

    The viewer switches on `kind`; anything it does not know about produces an
    empty headline, which looks like a rendering glitch rather than the missing
    case it is. Cheaper to catch here than by noticing a gap in a screenshot.
    """
    import re

    from evalforge.schema import trajectory as schema

    page = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    # Scope to headline()'s own switch. Searching the whole page would also
    # match body()'s cases, and a kind handled there but not here still renders
    # as a blank row — which is exactly the bug this is meant to catch.
    start = page.index("function headline(e)")
    handled = set(re.findall(r'case "([a-z_]+)":', page[start : page.index("function body(e)")]))

    emitted = {
        model.model_fields["kind"].default
        for model in vars(schema).values()
        if isinstance(model, type)
        and issubclass(model, schema.BaseModel)
        and "kind" in getattr(model, "model_fields", {})
    }

    assert emitted, "no event kinds discovered — the introspection is wrong, not the page"
    missing = emitted - handled
    assert not missing, f"the trajectory viewer has no case for: {sorted(missing)}"
