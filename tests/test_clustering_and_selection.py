"""Failure clustering and cost-aware selection.

Both are checked against ground truth rather than inspected: the synthetic
dataset injects known bug kinds, so cluster purity is a number; and selection is
compared against random choice under a fixed seed, so "better" is measured
rather than asserted.
"""

from __future__ import annotations

import random

import pytest

from evalforge.regression.clustering import (
    cluster_failures,
    cluster_shift,
    overall_purity,
    signature_for,
)
from evalforge.schema.result import CaseResult, EvaluatorResult, RunResult
from evalforge.selection.discriminative import (
    CaseValue,
    score_cases,
    select_from_runs,
    select_within_budget,
)


def evaluators(
    *,
    passed: bool,
    touched: list[str],
    unrelated: list[str] | None = None,
    untouched: list[str] | None = None,
    test_runs: int = 1,
    findings: int = 0,
) -> tuple[EvaluatorResult, ...]:
    return (
        EvaluatorResult(
            name="tests", score=1.0 if passed else 0.0, passed=passed,
            detail={"exit_code": 0 if passed else 1, "timed_out": False},
        ),
        EvaluatorResult(
            name="patch_locality", score=1.0, passed=True,
            detail={
                "touched": list(touched),
                "unrelated_files": list(unrelated or []),
                "untouched_targets": list(untouched or []),
            },
        ),
        EvaluatorResult(
            name="trajectory", score=1.0, passed=True, detail={"test_runs": test_runs}
        ),
        EvaluatorResult(
            name="safety", score=1.0 if not findings else 0.0, passed=not findings,
            detail={"finding_count": findings},
        ),
    )


def case_result(case_id: str, *, passed: bool, status: str = "completed", **kwargs) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        attempt=0,
        status=status,  # type: ignore[arg-type]
        passed=passed,
        evaluators=evaluators(passed=passed, **kwargs) if status == "completed" else (),
        duration_s=1.0,
        input_tokens=kwargs.pop("input_tokens", 100) if "input_tokens" in kwargs else 100,
    )


def make_run(run_id: str, results: tuple[CaseResult, ...]) -> RunResult:
    return RunResult(
        run_id=run_id,
        agent_ref="scripted:baseline",
        agent_hash="sha256:agent",
        dataset_name="synth",
        dataset_version="v1",
        dataset_hash="sha256:dataset",
        suite_name="default",
        suite_hash="sha256:suite",
        case_results=results,
    )


# -- signatures ----------------------------------------------------------


def test_an_agent_that_changed_nothing_is_its_own_shape() -> None:
    signature = signature_for(case_result("a", passed=False, touched=[], untouched=["pkg/m.py"]))

    assert signature.edit == "none"
    assert signature.label == "changed nothing"


def test_editing_the_right_file_without_fixing_it_is_a_distinct_shape() -> None:
    signature = signature_for(case_result("a", passed=False, touched=["pkg/m.py"]))

    assert signature.edit == "on_target"
    assert signature.label == "edited the right file, tests still failing"


def test_editing_unrelated_files_is_a_distinct_shape() -> None:
    signature = signature_for(
        case_result("a", passed=False, touched=["notes.md"], unrelated=["notes.md"],
                    untouched=["pkg/m.py"])
    )

    assert signature.label == "edited the wrong files"


def test_never_running_the_tests_is_a_distinct_shape() -> None:
    signature = signature_for(case_result("a", passed=False, touched=["pkg/m.py"], test_runs=0))

    assert signature.label == "edited without ever running the tests"


def test_unsafe_behaviour_dominates_the_label() -> None:
    signature = signature_for(case_result("a", passed=False, touched=["pkg/m.py"], findings=2))

    assert signature.unsafe is True
    assert signature.label == "unsafe behaviour recorded"


def test_a_harness_fault_is_not_blamed_on_the_agent() -> None:
    signature = signature_for(case_result("a", passed=False, status="infra_error"))

    assert signature.label == "infrastructure fault, not the agent"


def test_a_timeout_is_its_own_shape() -> None:
    assert signature_for(case_result("a", passed=False, status="timed_out")).label == (
        "ran out of time"
    )


# -- clustering ----------------------------------------------------------


def test_failures_of_the_same_shape_group_together() -> None:
    run = make_run(
        "run-a",
        (
            case_result("a", passed=False, touched=[], untouched=["m.py"]),
            case_result("b", passed=False, touched=[], untouched=["m.py"]),
            case_result("c", passed=False, touched=["m.py"]),
            case_result("d", passed=True, touched=["m.py"]),
        ),
    )

    clusters = cluster_failures(run)

    assert len(clusters) == 2
    # Passing cases are not failures and must not appear.
    assert sum(cluster.size for cluster in clusters) == 3
    assert clusters[0].size == 2
    assert clusters[0].label == "changed nothing"


def test_clustering_is_deterministic() -> None:
    run = make_run(
        "run-a",
        tuple(
            case_result(
                f"c{i}",
                passed=False,
                touched=[] if i % 2 else ["m.py"],
                untouched=["m.py"] if i % 2 else [],
            )
            for i in range(10)
        ),
    )

    first = cluster_failures(run)
    second = cluster_failures(run)

    assert [(c.label, c.case_ids) for c in first] == [(c.label, c.case_ids) for c in second]


def test_clustering_recovers_the_seeded_fault_when_the_shape_matches_it() -> None:
    """The measurable version of "does clustering find anything real?".

    A scripted agent fails one bug kind by doing nothing and another by editing
    without fixing, so the clusters should line up with the seeded faults.
    """
    run = make_run(
        "run-a",
        (
            case_result("mut-1", passed=False, touched=[], untouched=["m.py"]),
            case_result("mut-2", passed=False, touched=[], untouched=["m.py"]),
            case_result("tie-1", passed=False, touched=["m.py"]),
            case_result("tie-2", passed=False, touched=["m.py"]),
        ),
    )
    bug_kinds = {
        "mut-1": "shared_mutable_state", "mut-2": "shared_mutable_state",
        "tie-1": "missing_tiebreak", "tie-2": "missing_tiebreak",
    }

    clusters = cluster_failures(run, bug_kinds=bug_kinds)

    assert overall_purity(clusters) == 1.0
    assert {c.dominant_bug_kind for c in clusters} == {"shared_mutable_state", "missing_tiebreak"}


def test_a_shape_spanning_several_faults_reports_lower_purity() -> None:
    run = make_run(
        "run-a",
        (
            case_result("a", passed=False, touched=[], untouched=["m.py"]),
            case_result("b", passed=False, touched=[], untouched=["m.py"]),
        ),
    )

    clusters = cluster_failures(run, bug_kinds={"a": "off_by_one", "b": "integer_division"})

    # Honest signal: this failure shape does not correspond to one cause.
    assert clusters[0].purity == 0.5


def test_purity_is_zero_without_labels() -> None:
    run = make_run("run-a", (case_result("a", passed=False, touched=[]),))

    assert overall_purity(cluster_failures(run)) == 0.0


def test_a_run_with_no_failures_has_no_clusters() -> None:
    run = make_run("run-a", (case_result("a", passed=True, touched=["m.py"]),))

    assert cluster_failures(run) == ()


def test_cluster_shift_turns_a_regression_into_a_sentence() -> None:
    before = cluster_failures(make_run("a", (case_result("x", passed=False, touched=["m.py"]),)))
    after = cluster_failures(
        make_run(
            "b",
            (
                case_result("x", passed=False, touched=["m.py"]),
                case_result("y", passed=False, touched=[], untouched=["m.py"]),
                case_result("z", passed=False, touched=[], untouched=["m.py"]),
            ),
        )
    )

    shift = cluster_shift(before, after)

    growth = {label: (was, now) for label, was, now in shift}
    assert growth["changed nothing"] == (0, 2)
    assert growth["edited the right file, tests still failing"] == (1, 1)
    # Biggest growth first, so the headline cluster leads.
    assert shift[0][0] == "changed nothing"


# -- selection -----------------------------------------------------------


def test_a_case_everyone_passes_carries_no_information() -> None:
    runs = [
        make_run(f"r{i}", (case_result("always", passed=True, touched=["m.py"]),))
        for i in range(4)
    ]

    values = score_cases(runs)

    assert values[0].pass_rate == 1.0
    assert values[0].discrimination == 0.0
    assert values[0].informative is False


def test_a_case_that_splits_versions_is_maximally_informative() -> None:
    runs = [
        make_run("r1", (case_result("split", passed=True, touched=["m.py"]),)),
        make_run("r2", (case_result("split", passed=False, touched=["m.py"]),)),
    ]

    values = score_cases(runs)

    assert values[0].pass_rate == 0.5
    assert values[0].discrimination == 1.0


def test_selection_never_exceeds_its_budget() -> None:
    values = [
        CaseValue(case_id=f"c{i}", runs_seen=4, pass_rate=0.5, mean_cost=100.0)
        for i in range(20)
    ]

    selection = select_within_budget(values, budget=450.0)

    assert selection.within_budget is True
    assert selection.total_cost <= 450.0
    assert len(selection.case_ids) == 4


def test_a_zero_budget_selects_nothing() -> None:
    values = [CaseValue(case_id="a", runs_seen=2, pass_rate=0.5, mean_cost=10.0)]

    assert select_within_budget(values, budget=0.0).case_ids == ()


def test_a_negative_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        select_within_budget([], budget=-1.0)


def test_cheap_informative_cases_are_preferred_to_expensive_ones() -> None:
    values = [
        CaseValue(case_id="cheap", runs_seen=4, pass_rate=0.5, mean_cost=10.0),
        CaseValue(case_id="pricey", runs_seen=4, pass_rate=0.5, mean_cost=1000.0),
    ]

    selection = select_within_budget(values, budget=100.0)

    assert selection.case_ids == ("cheap",)


def test_uninformative_cases_are_taken_last_but_not_dropped() -> None:
    """Never having discriminated is an absence of evidence, not evidence."""
    values = [
        CaseValue(case_id="settled", runs_seen=4, pass_rate=1.0, mean_cost=10.0),
        CaseValue(case_id="splits", runs_seen=4, pass_rate=0.5, mean_cost=10.0),
    ]

    tight = select_within_budget(values, budget=10.0)
    roomy = select_within_budget(values, budget=100.0)

    assert tight.case_ids == ("splits",)
    assert set(roomy.case_ids) == {"splits", "settled"}


def test_selection_beats_random_choice_on_information_per_budget() -> None:
    """The claim that makes selection worth building, measured under a seed."""
    rng = random.Random(11)
    values = [
        CaseValue(
            case_id=f"c{i:03d}",
            runs_seen=6,
            # A realistic mix: most cases settled, a minority discriminating.
            pass_rate=rng.choice([0.0, 0.0, 1.0, 1.0, 0.5, 0.33, 0.67]),
            mean_cost=rng.uniform(50.0, 500.0),
        )
        for i in range(60)
    ]
    budget = 2_000.0

    chosen = select_within_budget(values, budget=budget)

    random_totals = []
    for seed in range(50):
        shuffled = list(values)
        random.Random(seed).shuffle(shuffled)
        spent, gained = 0.0, 0.0
        for value in shuffled:
            if spent + value.mean_cost <= budget:
                spent += value.mean_cost
                gained += value.discrimination
        random_totals.append(gained)

    average_random = sum(random_totals) / len(random_totals)
    assert chosen.total_discrimination > average_random
    assert chosen.total_discrimination >= max(random_totals)


def test_selecting_from_runs_reads_history_end_to_end() -> None:
    runs = [
        make_run("r1", (
            case_result("splits", passed=True, touched=["m.py"]),
            case_result("settled", passed=True, touched=["m.py"]),
        )),
        make_run("r2", (
            case_result("splits", passed=False, touched=["m.py"]),
            case_result("settled", passed=True, touched=["m.py"]),
        )),
    ]

    selection = select_from_runs(runs, budget=1_000.0)

    assert selection.case_ids[0] == "splits"


def test_incomplete_attempts_do_not_make_a_case_look_informative() -> None:
    """A case that times out tells you about the harness, not about the agent."""
    runs = [
        make_run("r1", (case_result("flaky", passed=True, touched=["m.py"]),)),
        make_run("r2", (case_result("flaky", passed=False, status="timed_out"),)),
    ]

    values = score_cases(runs)

    assert values[0].runs_seen == 1
    assert values[0].discrimination == 0.0


# -- end-to-end against a known control surface --------------------------


async def test_clustering_recovers_exactly_what_the_policy_does(tmp_path) -> None:
    """The right ground-truth check, and not the one I first wrote.

    Purity against *bug kind* is the wrong target: failure shape and bug kind
    are different axes, and with eight kinds and three shapes high purity is
    arithmetically impossible. What must hold is that cluster membership
    recovers what the agent is defined to do — the cases it abandons and the
    cases it edits uselessly land in different clusters, exactly.
    """
    from evalforge.agent.scripted import POLICIES
    from evalforge.datasets.builder import build_synthetic_dataset
    from evalforge.pipeline import RunRequest, execute_run

    dataset = build_synthetic_dataset(count=16, seed=7)
    policy = POLICIES["varied"]
    kinds = {case.case_id: case.metadata.bug_kind for case in dataset.cases}

    run = await execute_run(
        RunRequest(dataset=dataset, agent_ref="scripted:varied", concurrency=4)
    )
    clusters = cluster_failures(run, bug_kinds={k: v for k, v in kinds.items() if v})
    by_label = {c.label: {kinds[cid] for cid in c.case_ids} for c in clusters}

    abandoned = {
        kind for kind in kinds.values()
        if kind and kind not in policy.repairs and kind not in policy.botches
    }

    assert by_label["changed nothing"] == abandoned
    assert by_label["edited the right file, tests still failing"] == set(policy.botches)
    # Purity is below 1.0 and that is the honest answer: one shape, several causes.
    assert 0.0 < overall_purity(clusters) < 1.0
