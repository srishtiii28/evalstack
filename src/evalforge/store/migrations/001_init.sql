-- Initial schema: runs, their per-attempt results, and per-evaluator scores.
--
-- Attempts are written as they finish rather than in one batch at the end, so a
-- run killed halfway leaves a consistent store containing exactly the attempts
-- that completed.

CREATE TABLE runs (
    run_id           TEXT    PRIMARY KEY,
    created_at       TEXT    NOT NULL,
    agent_ref        TEXT    NOT NULL,
    agent_hash       TEXT    NOT NULL,
    dataset_name     TEXT    NOT NULL,
    dataset_version  TEXT    NOT NULL,
    dataset_hash     TEXT    NOT NULL,
    suite_name       TEXT    NOT NULL,
    suite_hash       TEXT    NOT NULL,
    samples_per_case INTEGER NOT NULL,
    concurrency      INTEGER NOT NULL,
    notes            TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE case_results (
    run_id          TEXT    NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    case_id         TEXT    NOT NULL,
    attempt         INTEGER NOT NULL,
    status          TEXT    NOT NULL,
    passed          INTEGER NOT NULL,
    duration_s      REAL    NOT NULL,
    cost_usd        REAL    NOT NULL,
    input_tokens    INTEGER NOT NULL,
    output_tokens   INTEGER NOT NULL,
    trajectory_path TEXT,
    error           TEXT,
    PRIMARY KEY (run_id, case_id, attempt)
);

CREATE TABLE evaluator_results (
    run_id      TEXT    NOT NULL,
    case_id     TEXT    NOT NULL,
    attempt     INTEGER NOT NULL,
    position    INTEGER NOT NULL,
    name        TEXT    NOT NULL,
    version     TEXT    NOT NULL,
    score       REAL    NOT NULL,
    passed      INTEGER NOT NULL,
    detail_json TEXT    NOT NULL,
    PRIMARY KEY (run_id, case_id, attempt, name),
    FOREIGN KEY (run_id, case_id, attempt)
        REFERENCES case_results (run_id, case_id, attempt) ON DELETE CASCADE
);

CREATE INDEX idx_case_results_run ON case_results (run_id);
CREATE INDEX idx_runs_created_at ON runs (created_at DESC);
