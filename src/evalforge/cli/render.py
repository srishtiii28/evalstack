"""Turning results into tables a human can read at a glance."""

from rich.console import Console
from rich.table import Table

from evalforge.hashing import short
from evalforge.judge_eval.validation import ValidationReport
from evalforge.regression.compare import ComparisonReport
from evalforge.schema.result import CaseResult, RunResult
from evalforge.stats.sampling import max_usable_k, stability_report
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

    # With repeated samples, capability and reliability come apart, and the gap
    # between them is the thing a single success rate cannot show.
    tallies = {case_id: (t.passed, t.total) for case_id, t in run.tallies().items()}
    k = max_usable_k(tallies.values())
    if k > 1:
        stability = stability_report(tallies, k=k)
        table.add_row(f"pass@{k} (capability)", f"{stability.pass_at_k:.1%}")
        table.add_row(f"pass^{k} (reliability)", f"{stability.pass_hat_k:.1%}")
        table.add_row("flaky cases", str(stability.flaky_cases))

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


def render_comparison(console: Console, report: ComparisonReport) -> None:
    """Show a comparison as a verdict with its evidence, never a bare delta."""
    for warning in report.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")
    if report.warnings:
        console.print()

    verdict_style = {
        "regression": "red",
        "improvement": "green",
        "no significant change": "dim",
        "not comparable": "yellow",
    }[report.verdict]
    console.print(f"verdict: [{verdict_style}]{report.verdict}[/{verdict_style}]")

    summary = Table(show_header=False, box=None, pad_edge=False)
    summary.add_column(style="dim")
    summary.add_column(justify="right")
    summary.add_row("before", f"{report.before_rate:.1%}  [dim]{report.before_run_id}[/dim]")
    summary.add_row("after", f"{report.after_rate:.1%}  [dim]{report.after_run_id}[/dim]")
    summary.add_row("difference", report.interval.format())
    summary.add_row("p-value", f"{report.test.p_value:.4f}")
    summary.add_row("shared cases", str(report.shared_cases))
    console.print(summary)

    counts = report.test.counts
    console.print(
        f"\n[dim]paired outcomes:[/dim] {counts.only_after_passed} fixed, "
        f"{counts.only_before_passed} broken, "
        f"{counts.both_passed} stable pass, {counts.both_failed} stable fail"
    )

    if report.underpowered:
        console.print(
            f"[yellow]underpowered:[/yellow] detecting a "
            f"{abs(report.delta):.1%} difference needs about "
            f"{report.required_cases} cases; this run compared {report.shared_cases}"
        )

    broken = report.transitions_of("broken")
    if broken:
        console.print("\n[red]broke:[/red] " + ", ".join(t.case_id for t in broken))
    fixed = report.transitions_of("fixed")
    if fixed:
        console.print("[green]fixed:[/green] " + ", ".join(t.case_id for t in fixed))

    movers = [d for d in report.dimensions if abs(d.delta) > 1e-9]
    if movers:
        table = Table(title="dimensions", title_justify="left", box=None, pad_edge=False)
        table.add_column("evaluator", style="dim", overflow="fold")
        table.add_column("before", justify="right")
        table.add_column("after", justify="right")
        table.add_column("delta", justify="right")
        for dimension in movers:
            colour = "green" if dimension.delta > 0 else "red"
            table.add_row(
                dimension.name,
                f"{dimension.before:.3f}",
                f"{dimension.after:.3f}",
                f"[{colour}]{dimension.delta:+.3f}[/{colour}]",
            )
        console.print()
        console.print(table)


def render_judge_validation(console: Console, report: ValidationReport) -> None:
    """Show what a judge is worth, leading with the number that matters."""
    for warning in report.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")
    if report.warnings:
        console.print()

    style = "green" if report.passed else "red"
    console.print(f"verdict: [{style}]{report.summary}[/{style}]")

    summary = Table(show_header=False, box=None, pad_edge=False)
    summary.add_column(style="dim")
    summary.add_column(justify="right")
    summary.add_row("judge", report.judge.describe())
    summary.add_row("gold set", f"{report.gold_name}@{report.gold_version}")
    summary.add_row("labels", str(report.agreement.count))
    # Accuracy is shown second and deliberately: on an imbalanced set a high
    # accuracy with a low kappa is the signature of a judge that learned the
    # majority class and nothing else.
    summary.add_row("accuracy", f"{report.agreement.accuracy:.1%}")
    summary.add_row("cohen's kappa", f"{report.agreement.kappa:.3f}")
    summary.add_row("threshold", f"{report.threshold:.2f}")
    console.print(summary)

    table = Table(title="per class", title_justify="left", box=None, pad_edge=False)
    table.add_column("label", style="dim", overflow="fold")
    table.add_column("precision", justify="right")
    table.add_column("recall", justify="right")
    table.add_column("f1", justify="right")
    table.add_column("support", justify="right")
    for metrics in report.agreement.per_class:
        table.add_row(
            metrics.label,
            f"{metrics.precision:.2f}",
            f"{metrics.recall:.2f}",
            f"{metrics.f1:.2f}",
            str(metrics.support),
        )
    console.print()
    console.print(table)

    if report.position is not None:
        console.print(
            f"\n[dim]position bias:[/dim] {report.position.consistency:.0%} consistent "
            f"across swapped orderings (skew {report.position.positional_skew:+.2f})"
        )
    if report.preference is not None and report.preference.comparable:
        console.print(
            f"[dim]self-preference:[/dim] {report.preference.own_rate:.0%} favourable to its "
            f"own family vs {report.preference.other_rate:.0%} to others "
            f"({report.preference.gap:+.0%})"
        )
