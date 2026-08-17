"""HTTP access to recorded runs.

A thin read-only layer over the store. Runs are produced by the CLI and the
scheduler; the API exists so a human can look at them, so it exposes exactly
what the dashboard needs and nothing that would let a browser start work.

The store connection is opened per request rather than shared. SQLite
connections are not safe to use from several threads, FastAPI runs synchronous
handlers on a threadpool, and opening a connection is cheap — sharing one and
adding a lock would trade a real hazard for a hand-rolled bottleneck.
"""

# No `from __future__ import annotations` here, deliberately. FastAPI resolves
# handler annotations at import time via get_type_hints, which only sees module
# globals — so a locally-scoped alias like StoreDep would arrive as an
# unresolvable string and the dependency would be silently demoted to a query
# parameter, turning every endpoint into a 422.

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from evalforge.paths import DEFAULT_DATABASE
from evalforge.regression.compare import ComparisonReport, compare
from evalforge.schema.result import RunResult
from evalforge.schema.trajectory import Trajectory
from evalforge.store.db import Store

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"
MAX_LIST_LIMIT = 200


class RunSummaryOut(BaseModel):
    run_id: str
    created_at: datetime
    agent_ref: str
    dataset_ref: str
    suite_name: str
    attempts: int
    completed: int
    passed: int
    success_rate: float
    total_cost_usd: float


class TrajectoryOut(BaseModel):
    run_id: str
    case_id: str
    attempt: int
    events: list[dict[str, Any]]


class ComparisonOut(BaseModel):
    before_run_id: str
    after_run_id: str
    before_rate: float
    after_rate: float
    delta: float
    interval_low: float
    interval_high: float
    p_value: float
    verdict: str
    shared_cases: int
    required_cases: int
    underpowered: bool
    comparable: bool
    warnings: list[str]
    transitions: list[dict[str, Any]]
    dimensions: list[dict[str, Any]]


def _comparison_out(report: ComparisonReport) -> ComparisonOut:
    return ComparisonOut(
        before_run_id=report.before_run_id,
        after_run_id=report.after_run_id,
        before_rate=report.before_rate,
        after_rate=report.after_rate,
        delta=report.delta,
        interval_low=report.interval.low,
        interval_high=report.interval.high,
        p_value=report.test.p_value,
        verdict=report.verdict,
        shared_cases=report.shared_cases,
        required_cases=report.required_cases,
        underpowered=report.underpowered,
        comparable=report.comparable,
        warnings=list(report.warnings),
        transitions=[
            {"case_id": t.case_id, "kind": t.kind, "delta": t.delta}
            for t in report.transitions
        ],
        dimensions=[
            {"name": d.name, "before": d.before, "after": d.after, "delta": d.delta}
            for d in report.dimensions
        ],
    )


def _load_run(store: Store, run_id: str) -> RunResult:
    try:
        return store.load_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"no run {run_id!r}") from exc


def create_app(*, database: Path = DEFAULT_DATABASE, web_root: Path | None = None) -> FastAPI:
    """Build the application, reading from ``database``."""
    app = FastAPI(title="EvalForge", version="0.1.0", docs_url="/api/docs")
    static_root = web_root if web_root is not None else WEB_ROOT

    def get_store() -> Iterator[Store]:
        with Store.open(database) as store:
            yield store

    StoreDep = Annotated[Store, Depends(get_store)]

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "database": str(database)}

    @app.get("/api/runs", response_model=list[RunSummaryOut])
    def list_runs(
        store: StoreDep,
        limit: Annotated[int, Query(ge=1, le=MAX_LIST_LIMIT)] = 50,
    ) -> list[RunSummaryOut]:
        return [
            RunSummaryOut(
                run_id=s.run_id,
                created_at=s.created_at,
                agent_ref=s.agent_ref,
                dataset_ref=s.dataset_ref,
                suite_name=s.suite_name,
                attempts=s.attempts,
                completed=s.completed,
                passed=s.passed,
                success_rate=s.success_rate,
                total_cost_usd=s.total_cost_usd,
            )
            for s in store.list_runs(limit=limit)
        ]

    @app.get("/api/runs/{run_id}", response_model=RunResult)
    def get_run(run_id: str, store: StoreDep) -> RunResult:
        return _load_run(store, run_id)

    @app.get("/api/runs/{run_id}/trajectory/{case_id}", response_model=TrajectoryOut)
    def get_trajectory(
        run_id: str,
        case_id: str,
        store: StoreDep,
        attempt: Annotated[int, Query(ge=0)] = 0,
    ) -> TrajectoryOut:
        run = _load_run(store, run_id)
        for result in run.case_results:
            if result.case_id == case_id and result.attempt == attempt:
                if result.trajectory_path is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"no trajectory was recorded for {case_id}#{attempt}",
                    )
                path = Path(result.trajectory_path)
                if not path.is_file():
                    raise HTTPException(
                        status_code=410, detail=f"trajectory file is gone: {path}"
                    )
                trajectory = Trajectory.from_jsonl(
                    run_id=run_id,
                    case_id=case_id,
                    attempt=attempt,
                    jsonl=path.read_text(encoding="utf-8"),
                )
                return TrajectoryOut(
                    run_id=run_id,
                    case_id=case_id,
                    attempt=attempt,
                    events=[event.model_dump(mode="json") for event in trajectory.events],
                )
        raise HTTPException(
            status_code=404, detail=f"run {run_id} has no attempt {attempt} of {case_id!r}"
        )

    @app.get("/api/compare", response_model=ComparisonOut)
    def compare_runs(before: str, after: str, store: StoreDep) -> ComparisonOut:
        baseline = _load_run(store, before)
        candidate = _load_run(store, after)
        return _comparison_out(compare(baseline, candidate))

    # response_model=None: the union return type is a choice of response class,
    # not a schema for FastAPI to serialise against.
    @app.get("/", include_in_schema=False, response_model=None)
    def index() -> FileResponse | JSONResponse:
        page = static_root / "index.html"
        if not page.is_file():
            return JSONResponse(
                status_code=404, content={"detail": f"no dashboard page at {page}"}
            )
        return FileResponse(page)

    return app
