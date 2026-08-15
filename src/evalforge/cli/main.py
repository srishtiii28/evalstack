"""The ``evalforge`` command line."""

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from evalforge import __version__
from evalforge.agent.build import MODEL_PREFIX, ModelAgentSpec, model_agent_factory
from evalforge.agent.model_agent import DEFAULT_MAX_STEPS, ModelAgentConfig
from evalforge.agent.registry import agent_names
from evalforge.cli.render import (
    render_case_results,
    render_run_header,
    render_run_list,
    render_run_metrics,
)
from evalforge.datasets.builder import (
    DEFAULT_COUNT,
    DEFAULT_NAME,
    DEFAULT_SEED,
    DEFAULT_VERSION,
    GENERATOR_ID,
    build_synthetic_dataset,
)
from evalforge.datasets.io import CASES_FILENAME, verify_dataset, write_dataset
from evalforge.datasets.registry import DatasetNotFoundError, resolve_dataset
from evalforge.env.limits import probe_limit_support
from evalforge.evaluators.registry import suite_names
from evalforge.hashing import short
from evalforge.model.budget import BudgetGuard, BudgetLimits
from evalforge.model.providers import GROQ, ProviderNotConfiguredError
from evalforge.paths import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATABASE,
    DEFAULT_DATASETS_ROOT,
    DEFAULT_TRAJECTORY_DIR,
)
from evalforge.pipeline import RunRequest, execute_run
from evalforge.schema.result import RunResult
from evalforge.store.db import Store

console = Console()
error_console = Console(stderr=True)

EXIT_USAGE_ERROR = 2
EXIT_CHECK_FAILED = 1

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Continuous evaluation infrastructure for coding agents.",
)
dataset_app = typer.Typer(no_args_is_help=True, help="Build and verify evaluation datasets.")
app.add_typer(dataset_app, name="dataset")


def _fail(message: str, *, code: int = EXIT_USAGE_ERROR) -> typer.Exit:
    error_console.print(f"[red]error:[/red] {message}")
    return typer.Exit(code)


@app.command()
def version() -> None:
    """Print the EvalForge version."""
    console.print(__version__)


@app.command()
def doctor() -> None:
    """Report what containment this machine can actually enforce."""
    support = probe_limit_support()
    console.print(f"python sandbox limits: {support.describe()}")
    console.print(f"known agents: {', '.join(agent_names())}")
    console.print(f"known suites: {', '.join(suite_names())}")


@dataset_app.command("build")
def dataset_build(
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Directory to write. Defaults to datasets/<name>."),
    ] = None,
    name: Annotated[str, typer.Option(help="Dataset name.")] = DEFAULT_NAME,
    dataset_version: Annotated[
        str, typer.Option("--dataset-version", help="Dataset version, e.g. v1.")
    ] = DEFAULT_VERSION,
    cases: Annotated[int, typer.Option(help="How many cases to generate.")] = DEFAULT_COUNT,
    seed: Annotated[int, typer.Option(help="Generation seed.")] = DEFAULT_SEED,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing dataset.")] = False,
) -> None:
    """Generate a synthetic dataset of seeded-bug repair tasks."""
    directory = out or (DEFAULT_DATASETS_ROOT / name)
    if (directory / CASES_FILENAME).exists() and not force:
        raise _fail(f"{directory} already holds a dataset; pass --force to overwrite")

    try:
        dataset = build_synthetic_dataset(
            name=name, version=dataset_version, count=cases, seed=seed
        )
    except ValueError as exc:
        raise _fail(str(exc)) from exc

    write_dataset(dataset, directory, generator=GENERATOR_ID, seed=seed)
    console.print(
        f"wrote [bold]{dataset.ref}[/bold] — {len(dataset.cases)} cases "
        f"[dim]({short(dataset.content_hash)})[/dim] → {directory}"
    )


@dataset_app.command("verify")
def dataset_verify(
    directory: Annotated[Path, typer.Argument(help="Dataset directory to check.")],
) -> None:
    """Recompute a dataset's content hash and compare it with its manifest."""
    try:
        report = verify_dataset(directory)
    except FileNotFoundError as exc:
        raise _fail(str(exc)) from exc

    if not report.ok:
        error_console.print(
            f"[red]drift:[/red] {report.name}@{report.version} at {directory}\n"
            f"  manifest says {report.expected_hash}\n"
            f"  contents give {report.actual_hash}"
        )
        raise typer.Exit(EXIT_CHECK_FAILED)

    console.print(
        f"[green]ok[/green] {report.name}@{report.version} — {report.case_count} cases "
        f"[dim]({short(report.actual_hash)})[/dim]"
    )


@app.command("run")
def run_command(
    dataset: Annotated[
        str, typer.Option("--dataset", help="Dataset ref (name@version) or directory path.")
    ] = f"{DEFAULT_NAME}@{DEFAULT_VERSION}",
    agent: Annotated[str, typer.Option("--agent", help="Agent reference.")] = "scripted:baseline",
    suite: Annotated[str, typer.Option("--suite", help="Evaluator suite.")] = "default",
    samples: Annotated[
        int, typer.Option("-k", "--samples", min=1, help="Attempts per case.")
    ] = 1,
    concurrency: Annotated[
        int, typer.Option("--concurrency", min=1, help="Attempts in flight at once.")
    ] = 4,
    case: Annotated[
        list[str] | None, typer.Option("--case", help="Restrict to these case ids (repeatable).")
    ] = None,
    datasets_root: Annotated[
        Path, typer.Option("--datasets-root", help="Where name@version refs are looked up.")
    ] = DEFAULT_DATASETS_ROOT,
    database: Annotated[
        Path, typer.Option("--db", help="Results database.")
    ] = DEFAULT_DATABASE,
    trajectories: Annotated[
        Path, typer.Option("--trajectories", help="Where trajectory traces are written.")
    ] = DEFAULT_TRAJECTORY_DIR,
    notes: Annotated[str, typer.Option("--notes", help="Free-text note stored with the run.")] = "",
    provider: Annotated[
        str, typer.Option("--provider", help="Model provider, for model agents.")
    ] = GROQ.name,
    max_steps: Annotated[
        int, typer.Option("--max-steps", min=1, help="Tool-use steps a model agent may take.")
    ] = DEFAULT_MAX_STEPS,
    max_model_calls: Annotated[
        int | None,
        typer.Option("--max-model-calls", min=1, help="Cap model calls per run."),
    ] = None,
    max_model_tokens: Annotated[
        int | None,
        typer.Option("--max-model-tokens", min=1, help="Cap total tokens per run."),
    ] = None,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Bypass the model response cache.")
    ] = False,
    show_all: Annotated[
        bool, typer.Option("--all", help="Show every case row, not just the first page.")
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the run as JSON instead of tables.")
    ] = False,
) -> None:
    """Evaluate an agent against a dataset."""
    try:
        loaded = resolve_dataset(dataset, root=datasets_root)
    except (DatasetNotFoundError, FileNotFoundError, ValueError) as exc:
        raise _fail(str(exc)) from exc

    request = RunRequest(
        dataset=loaded,
        agent_ref=agent,
        suite_name=suite,
        samples_per_case=samples,
        concurrency=concurrency,
        case_ids=tuple(case or ()),
        trajectory_dir=trajectories,
        notes=notes,
    )

    guard: BudgetGuard | None = None
    try:
        with Store.open(database) as store:
            if agent.split(":", 1)[0] == MODEL_PREFIX:
                spec = ModelAgentSpec.from_reference(
                    agent,
                    provider=provider,
                    settings=ModelAgentConfig(max_steps=max_steps),
                    budget=BudgetLimits(
                        max_calls=max_model_calls, max_tokens=max_model_tokens
                    ),
                    cache_dir=None if no_cache else DEFAULT_CACHE_DIR,
                )
                result, guard = asyncio.run(_run_with_model(request, spec, store))
            else:
                result = asyncio.run(execute_run(request, store=store))
    except ProviderNotConfiguredError as exc:
        raise _fail(str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise _fail(str(exc)) from exc

    if as_json:
        console.print_json(result.model_dump_json())
        return

    render_run_header(console, result)
    console.print()
    render_run_metrics(console, result)
    if guard is not None:
        console.print(f"[dim]model usage: {guard.describe()}[/dim]")
    console.print()
    render_case_results(console, result.case_results, limit=None if show_all else 40)


async def _run_with_model(
    request: RunRequest, spec: ModelAgentSpec, store: Store
) -> tuple[RunResult, BudgetGuard]:
    """Run with a shared HTTP pool, rate limiter and budget across all attempts."""
    async with model_agent_factory(spec) as (make_agent, guard):
        result = await execute_run(request, store=store, agent_factory=make_agent)
    return result, guard


@app.command("runs")
def runs_command(
    database: Annotated[Path, typer.Option("--db", help="Results database.")] = DEFAULT_DATABASE,
    limit: Annotated[int, typer.Option("--limit", min=1, help="How many runs to list.")] = 20,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the listing as JSON instead of a table.")
    ] = False,
) -> None:
    """List recorded runs, most recent first."""
    with Store.open(database) as store:
        summaries = store.list_runs(limit=limit)

    if as_json:
        console.print_json(
            json.dumps(
                [
                    {
                        "run_id": summary.run_id,
                        "created_at": summary.created_at.isoformat(),
                        "agent_ref": summary.agent_ref,
                        "dataset_ref": summary.dataset_ref,
                        "suite_name": summary.suite_name,
                        "attempts": summary.attempts,
                        "completed": summary.completed,
                        "passed": summary.passed,
                        "success_rate": summary.success_rate,
                        "total_cost_usd": summary.total_cost_usd,
                    }
                    for summary in summaries
                ]
            )
        )
        return

    render_run_list(console, summaries)


@app.command("show")
def show_command(
    run_id: Annotated[str, typer.Argument(help="Run id to display.")],
    database: Annotated[Path, typer.Option("--db", help="Results database.")] = DEFAULT_DATABASE,
    show_all: Annotated[bool, typer.Option("--all", help="Show every case row.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON instead of tables.")] = False,
) -> None:
    """Show a recorded run in detail."""
    with Store.open(database) as store:
        try:
            result = store.load_run(run_id)
        except KeyError as exc:
            raise _fail(str(exc)) from exc

    if as_json:
        console.print_json(result.model_dump_json())
        return

    render_run_header(console, result)
    console.print()
    render_run_metrics(console, result)
    console.print()
    render_case_results(console, result.case_results, limit=None if show_all else 40)


@app.command("trajectory")
def trajectory_command(
    run_id: Annotated[str, typer.Argument(help="Run id.")],
    case_id: Annotated[str, typer.Argument(help="Case id.")],
    attempt: Annotated[int, typer.Option("--attempt", min=0, help="Which attempt.")] = 0,
    database: Annotated[Path, typer.Option("--db", help="Results database.")] = DEFAULT_DATABASE,
) -> None:
    """Print a recorded trajectory as one JSON event per line."""
    with Store.open(database) as store:
        try:
            result = store.load_run(run_id)
        except KeyError as exc:
            raise _fail(str(exc)) from exc

    for case_result in result.case_results:
        if case_result.case_id == case_id and case_result.attempt == attempt:
            if case_result.trajectory_path is None:
                raise _fail(f"no trajectory was recorded for {case_id}#{attempt}")
            path = Path(case_result.trajectory_path)
            if not path.is_file():
                raise _fail(f"trajectory file is missing: {path}")
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    console.print_json(json.dumps(json.loads(line)))
            return

    raise _fail(f"run {run_id} has no attempt {attempt} of case {case_id!r}")


if __name__ == "__main__":  # pragma: no cover - console-script entry point
    app()
