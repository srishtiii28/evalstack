"""Turning results into tables a human can read at a glance."""

from rich.console import Console
from rich.table import Table

from evalforge.hashing import short
from evalforge.schema.result import CaseResult, RunResult
from evalforge.store.db import RunSummary

PASS_MARK = "[green]pass[/green]"
FAIL_MARK = "[red]fail[/red]"
DEFAULT_ROW_LIMIT = 40


def verdict_mark(passed: bool) -> str:
    return PASS_MARK if passed else FAIL_MARK


def render_run_header(console: Console, run: RunResult) -> None:
    """Identity and provenance: what ran, against what, measured how."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()

    table.add_row("run", run.run_id)
    table.add_row("agent", f"{run.agent_ref}  [dim]({short(run.agent_hash)})[/dim]")
    table.add_row("dataset", f"{run.dataset_ref}  [dim]({short(run.dataset_hash)})[/dim]")
    table.add_row("suite", f"{run.suite_name}  [dim]({short(run.suite_hash)})[/dim]")
    table.add_row("samples/case", str(run.samples_per_case))
    table.add_row("concurrency", str(run.concurrency))
    console.print(table)


def render_run_metrics(console: Console, run: RunResult) -> None:
    """Headline numbers, plus the status breakdown when anything did not complete."""
    completed = run.completed_results
    table = Table(title="metrics", title_justify="left", box=None, pad_edge=False)
    table.add_column("metric", style="dim")
    table.add_column("value", justify="right")

    table.add_row("attempts", str(len(run.case_results)))
    table.add_row("completed", str(len(completed)))
    table.add_row("passed", str(sum(1 for result in completed if result.passed)))
    table.add_row("success rate", f"{run.success_rate:.1%}")
    table.add_row("mean duration", f"{run.mean_duration_s:.2f}s")
    table.add_row("total cost", f"${run.total_cost_usd:.4f}")
    table.add_row("tokens in/out", f"{run.total_input_tokens:,} / {run.total_output_tokens:,}")

    for name, score in run.evaluator_scores().items():
        table.add_row(f"mean {name}", f"{score:.3f}")

    statuses = {
        status: count
        for status, count in run.status_counts().items()
        if status != "completed"
    }
    for status, count in sorted(statuses.items()):
        table.add_row(f"[yellow]{status}[/yellow]", str(count))

    console.print(table)


def render_case_results(
    console: Console, results: tuple[CaseResult, ...], *, limit: int | None = DEFAULT_ROW_LIMIT
) -> None:
    """Per-attempt detail, with each evaluator as its own column."""
    if not results:
        console.print("[dim]no case results[/dim]")
        return

    evaluator_names: list[str] = []
    for result in results:
        for evaluation in result.evaluators:
            if evaluation.name not in evaluator_names:
                evaluator_names.append(evaluation.name)

    table = Table(title="cases", title_justify="left")
    table.add_column("case", overflow="fold")
    table.add_column("try", justify="right")
    table.add_column("status")
    table.add_column("verdict")
    for name in evaluator_names:
        table.add_column(name, justify="right")
    table.add_column("seconds", justify="right")

    shown = results if limit is None else results[:limit]
    for result in shown:
        scores = {evaluation.name: evaluation.score for evaluation in result.evaluators}
        status = (
            "[dim]completed[/dim]"
            if result.status == "completed"
            else f"[yellow]{result.status}[/yellow]"
        )
        score_cells = [
            f"{scores[name]:.2f}" if name in scores else "[dim]—[/dim]"
            for name in evaluator_names
        ]
        table.add_row(
            result.case_id,
            str(result.attempt),
            status,
            verdict_mark(result.passed),
            *score_cells,
            f"{result.duration_s:.2f}",
        )

    console.print(table)
    if limit is not None and len(results) > limit:
        hidden = len(results) - limit
        console.print(f"[dim]… {hidden} more rows (use --all to show every case)[/dim]")


def render_run_list(console: Console, summaries: tuple[RunSummary, ...]) -> None:
    if not summaries:
        console.print("[dim]no runs recorded yet[/dim]")
        return

    table = Table(title="runs", title_justify="left")
    # Fold rather than truncate: a narrow terminal should wrap identifiers, not
    # silently cut the part that tells you which run this is.
    table.add_column("run", overflow="fold")
    table.add_column("created", overflow="fold")
    table.add_column("agent", overflow="fold")
    table.add_column("dataset", overflow="fold")
    table.add_column("suite", overflow="fold")
    table.add_column("passed", justify="right")
    table.add_column("rate", justify="right")
    table.add_column("cost", justify="right")

    for summary in summaries:
        table.add_row(
            summary.run_id,
            summary.created_at.strftime("%Y-%m-%d %H:%M"),
            summary.agent_ref,
            summary.dataset_ref,
            summary.suite_name,
            f"{summary.passed}/{summary.completed}",
            f"{summary.success_rate:.1%}",
            f"${summary.total_cost_usd:.4f}",
        )

    console.print(table)
