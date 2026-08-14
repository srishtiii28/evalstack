"""Persistence for runs and their results.

SQLite with hand-written SQL rather than an ORM: the schema is six columns of
scalars and two join tables, and an ORM would add a dependency, a mapping layer
and a migration tool to manage a model that fits on one screen.

Migrations are numbered ``.sql`` files applied in order and recorded in
``schema_migrations``, so opening an old database upgrades it and opening a
newer one than the code knows about fails loudly.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from pydantic import JsonValue

from evalforge.schema.result import CaseResult, EvaluatorResult, RunResult

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_MIGRATION_FILENAME_RE = re.compile(r"^(\d+)_[A-Za-z0-9_]+\.sql$")


class StoreError(Exception):
    """The database is unusable or inconsistent with this version of the code."""


@dataclass(frozen=True, slots=True)
class RunSummary:
    """A row for listings — cheap to compute, enough to choose a run to open."""

    run_id: str
    created_at: datetime
    agent_ref: str
    dataset_ref: str
    suite_name: str
    attempts: int
    completed: int
    passed: int
    total_cost_usd: float

    @property
    def success_rate(self) -> float:
        return self.passed / self.completed if self.completed else 0.0


def _is_only_comments(text: str) -> bool:
    return all(not line.strip() or line.strip().startswith("--") for line in text.splitlines())


def iter_sql_statements(script: str) -> Iterator[str]:
    """Split a SQL script into individual statements.

    ``executescript`` would be simpler, but it implicitly commits any open
    transaction — which would silently break the atomicity of "apply the schema
    *and* record the version". Splitting with :func:`sqlite3.complete_statement`
    keeps the whole migration inside one transaction, and unlike a naive split
    on semicolons it respects them inside string literals.
    """
    buffer = ""
    for line in script.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            statement = buffer.strip()
            if statement and not _is_only_comments(statement):
                yield statement
            buffer = ""

    remainder = buffer.strip()
    if remainder and not _is_only_comments(remainder):
        raise StoreError(f"migration ends with an incomplete statement: {remainder[:80]!r}")


def _migration_files() -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        match = _MIGRATION_FILENAME_RE.match(path.name)
        if match is None:
            raise StoreError(f"migration filename must be NNN_name.sql, got {path.name!r}")
        found.append((int(match.group(1)), path))
    if not found:
        raise StoreError(f"no migrations found in {MIGRATIONS_DIR}")
    return found


class Store:
    """A connection to the results database."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    # -- lifecycle -------------------------------------------------------

    @classmethod
    def open(cls, path: Path | str) -> Self:
        """Open (creating if needed) a database and bring its schema up to date."""
        target = Path(path)
        if target.parent and str(target) != ":memory:":
            target.parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(target, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        # Durability matters more than throughput here: a killed run must leave
        # the attempts it already reported.
        connection.execute("PRAGMA synchronous = FULL")

        store = cls(connection)
        store.migrate()
        return store

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # -- migrations ------------------------------------------------------

    def migrate(self) -> tuple[int, ...]:
        """Apply any pending migrations; return the versions applied now."""
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        rows = self._connection.execute("SELECT version FROM schema_migrations").fetchall()
        applied = {int(row["version"]) for row in rows}

        available = _migration_files()
        unknown = applied - {version for version, _ in available}
        if unknown:
            raise StoreError(
                "database was written by a newer version of EvalForge "
                f"(unknown migrations: {sorted(unknown)})"
            )

        newly_applied: list[int] = []
        for version, path in available:
            if version in applied:
                continue
            with self._transaction():
                for statement in iter_sql_statements(path.read_text(encoding="utf-8")):
                    self._connection.execute(statement)
                self._connection.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (version, datetime.now(UTC).isoformat()),
                )
            newly_applied.append(version)
        return tuple(newly_applied)

    @property
    def schema_version(self) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
        ).fetchone()
        return int(row["version"])

    # -- writing ---------------------------------------------------------

    def start_run(self, run: RunResult) -> None:
        """Write the run header. Case results are appended as they complete."""
        with self._transaction():
            self._connection.execute(
                "INSERT INTO runs ("
                " run_id, created_at, agent_ref, agent_hash, dataset_name, dataset_version,"
                " dataset_hash, suite_name, suite_hash, samples_per_case, concurrency, notes"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run.run_id,
                    run.created_at.isoformat(),
                    run.agent_ref,
                    run.agent_hash,
                    run.dataset_name,
                    run.dataset_version,
                    run.dataset_hash,
                    run.suite_name,
                    run.suite_hash,
                    run.samples_per_case,
                    run.concurrency,
                    run.notes,
                ),
            )

    def record_case_result(self, run_id: str, result: CaseResult) -> None:
        """Append one attempt's result, replacing any earlier row for that attempt."""
        with self._transaction():
            self._connection.execute(
                "INSERT OR REPLACE INTO case_results ("
                " run_id, case_id, attempt, status, passed, duration_s, cost_usd,"
                " input_tokens, output_tokens, trajectory_path, error"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    result.case_id,
                    result.attempt,
                    result.status,
                    int(result.passed),
                    result.duration_s,
                    result.cost_usd,
                    result.input_tokens,
                    result.output_tokens,
                    result.trajectory_path,
                    result.error,
                ),
            )
            self._connection.execute(
                "DELETE FROM evaluator_results WHERE run_id = ? AND case_id = ? AND attempt = ?",
                (run_id, result.case_id, result.attempt),
            )
            self._connection.executemany(
                "INSERT INTO evaluator_results ("
                " run_id, case_id, attempt, position, name, version, score, passed, detail_json"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        run_id,
                        result.case_id,
                        result.attempt,
                        position,
                        evaluation.name,
                        evaluation.version,
                        evaluation.score,
                        int(evaluation.passed),
                        json.dumps(evaluation.detail, sort_keys=True),
                    )
                    for position, evaluation in enumerate(result.evaluators)
                ],
            )

    def save_run(self, run: RunResult) -> None:
        """Write a complete run in one go."""
        self.start_run(run)
        for result in run.case_results:
            self.record_case_result(run.run_id, result)

    def delete_run(self, run_id: str) -> bool:
        with self._transaction():
            cursor = self._connection.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        return cursor.rowcount > 0

    # -- reading ---------------------------------------------------------

    def run_exists(self, run_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return row is not None

    def load_run(self, run_id: str) -> RunResult:
        row = self._connection.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no run {run_id!r} in the store")

        return RunResult(
            run_id=row["run_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            agent_ref=row["agent_ref"],
            agent_hash=row["agent_hash"],
            dataset_name=row["dataset_name"],
            dataset_version=row["dataset_version"],
            dataset_hash=row["dataset_hash"],
            suite_name=row["suite_name"],
            suite_hash=row["suite_hash"],
            samples_per_case=row["samples_per_case"],
            concurrency=row["concurrency"],
            notes=row["notes"],
            case_results=self._load_case_results(run_id),
        )

    def _load_case_results(self, run_id: str) -> tuple[CaseResult, ...]:
        evaluations: dict[tuple[str, int], list[EvaluatorResult]] = {}
        evaluator_rows = self._connection.execute(
            "SELECT case_id, attempt, name, version, score, passed, detail_json"
            " FROM evaluator_results WHERE run_id = ? ORDER BY case_id, attempt, position",
            (run_id,),
        ).fetchall()
        for row in evaluator_rows:
            detail: dict[str, JsonValue] = json.loads(row["detail_json"])
            evaluations.setdefault((row["case_id"], row["attempt"]), []).append(
                EvaluatorResult(
                    name=row["name"],
                    version=row["version"],
                    score=row["score"],
                    passed=bool(row["passed"]),
                    detail=detail,
                )
            )

        case_rows = self._connection.execute(
            "SELECT * FROM case_results WHERE run_id = ? ORDER BY case_id, attempt",
            (run_id,),
        ).fetchall()
        return tuple(
            CaseResult(
                case_id=row["case_id"],
                attempt=row["attempt"],
                status=row["status"],
                passed=bool(row["passed"]),
                evaluators=tuple(evaluations.get((row["case_id"], row["attempt"]), ())),
                duration_s=row["duration_s"],
                cost_usd=row["cost_usd"],
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                trajectory_path=row["trajectory_path"],
                error=row["error"],
            )
            for row in case_rows
        )

    def list_runs(self, *, limit: int = 50) -> tuple[RunSummary, ...]:
        """Most recent runs first, with aggregates computed in SQL."""
        rows = self._connection.execute(
            """
            SELECT
                r.run_id,
                r.created_at,
                r.agent_ref,
                r.dataset_name,
                r.dataset_version,
                r.suite_name,
                COUNT(c.case_id) AS attempts,
                COALESCE(SUM(c.status = 'completed'), 0) AS completed,
                COALESCE(SUM(c.status = 'completed' AND c.passed), 0) AS passed,
                COALESCE(SUM(c.cost_usd), 0.0) AS total_cost_usd
            FROM runs AS r
            LEFT JOIN case_results AS c ON c.run_id = r.run_id
            GROUP BY r.run_id
            ORDER BY r.created_at DESC, r.run_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return tuple(
            RunSummary(
                run_id=row["run_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                agent_ref=row["agent_ref"],
                dataset_ref=f"{row['dataset_name']}@{row['dataset_version']}",
                suite_name=row["suite_name"],
                attempts=row["attempts"],
                completed=row["completed"],
                passed=row["passed"],
                total_cost_usd=row["total_cost_usd"],
            )
            for row in rows
        )

    # -- internals -------------------------------------------------------

    def _transaction(self) -> _Transaction:
        return _Transaction(self._connection)


class _Transaction:
    """Explicit transaction, since the connection runs in autocommit mode."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def __enter__(self) -> sqlite3.Connection:
        self._connection.execute("BEGIN")
        return self._connection

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self._connection.execute("COMMIT")
        else:
            self._connection.execute("ROLLBACK")


def iter_migration_versions() -> Iterator[int]:
    """Every migration version this build knows about, in order."""
    for version, _ in _migration_files():
        yield version
