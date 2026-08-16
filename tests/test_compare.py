"""Comparing runs: does it detect a real change, and stay quiet otherwise?

The load-bearing test is not that a planted regression is caught — it is that
comparing a run against itself reports nothing. A gate that fires on noise gets
switched off, and then it catches nothing at all.
"""

from __future__ import annotations

from evalforge.regression.compare import compare
from evalforge.schema.result import CaseResult, EvaluatorResult, RunResult


def make_run(
    run_id: str,
    outcomes: dict[str, bool],
    *,
    dataset_hash: str = "sha256:dataset",
    suite_hash: str = "sha256:suite",
    agent_hash: str = "sha256:agent",
    agent_ref: str = "scripted:baseline",
    trajectory_score: float = 1.0,
) -> RunResult:
    return RunResult(
        run_id=run_id,
        agent_ref=agent_ref,
        agent_hash=agent_hash,
        dataset_name="synth",
        dataset_version="v1",
        dataset_hash=dataset_hash,
        suite_name="default",
        suite_hash=suite_hash,
        case_results=tuple(
            CaseResult(
                case_id=case_id,
                attempt=0,
                status="completed",
                passed=passed,
                evaluators=(
                    EvaluatorResult(name="tests", score=1.0 if passed else 0.0, passed=passed),
                    EvaluatorResult(
                        name="trajectory", score=trajectory_score, passed=True
                    ),
                ),
                duration_s=1.0,
            )
            for case_id, passed in outcomes.items()
        ),
    )


def outcomes(passing: int, total: int) -> dict[str, bool]:
    return {f"case-{index:03d}": index < passing for index in range(total)}


# -- the quiet case ------------------------------------------------------


def test_a_run_compared_against_itself_reports_no_change() -> None:
    run = make_run("run-a", outcomes(21, 30))

    report = compare(run, run)

    assert report.verdict == "no significant change"
    assert report.delta == 0.0
    assert report.test.p_value == 1.0
    assert report.interval.low == report.interval.high == 0.0
    assert report.warnings  # same agent configuration is worth flagging


def test_two_runs_of_the_same_agent_with_identical_results_are_quiet() -> None:
    before = make_run("run-a", outcomes(21, 30))
    after = make_run("run-b", outcomes(21, 30))

    report = compare(before, after)

    assert report.verdict == "no significant change"
    assert report.test.counts.discordant == 0


def test_a_one_case_difference_on_thirty_is_not_significant() -> None:
    """The move real teams ship on, correctly reported as noise."""
    before = make_run("run-a", outcomes(21, 30))
    after = make_run("run-b", outcomes(20, 30))

    report = compare(before, after)

    assert report.delta < 0
    assert report.verdict == "no significant change"


# -- the loud case -------------------------------------------------------


def test_a_planted_regression_is_detected() -> None:
    # Nine cases that passed now fail: a twenty-point drop on thirty cases.
    before = make_run("run-a", outcomes(27, 30))
    after = make_run("run-b", outcomes(18, 30))

    report = compare(before, after)

    assert report.verdict == "regression"
    assert report.test.significant is True
    assert report.delta < -0.25
    assert report.interval.excludes_zero is True
    assert len(report.transitions_of("broken")) == 9


def test_a_planted_improvement_is_detected_and_named() -> None:
    before = make_run("run-a", outcomes(10, 30))
    after = make_run("run-b", outcomes(25, 30))

    report = compare(before, after)

    assert report.verdict == "improvement"
    assert len(report.transitions_of("fixed")) == 15


def test_the_report_names_which_cases_moved() -> None:
    before = make_run("run-a", {"a": True, "b": True, "c": False, "d": False})
    after = make_run("run-b", {"a": True, "b": False, "c": True, "d": False})

    report = compare(before, after)

    assert [t.case_id for t in report.transitions_of("broken")] == ["b"]
    assert [t.case_id for t in report.transitions_of("fixed")] == ["c"]
    assert [t.case_id for t in report.transitions_of("stable pass")] == ["a"]
    assert [t.case_id for t in report.transitions_of("stable fail")] == ["d"]


# -- comparability -------------------------------------------------------


def test_different_datasets_are_not_comparable() -> None:
    before = make_run("run-a", outcomes(20, 30))
    after = make_run("run-b", outcomes(28, 30), dataset_hash="sha256:different")

    report = compare(before, after)

    assert report.comparable is False
    assert report.verdict == "not comparable"
    assert any("different datasets" in w for w in report.warnings)


def test_a_reconfigured_suite_is_not_comparable() -> None:
    before = make_run("run-a", outcomes(20, 30))
    after = make_run("run-b", outcomes(28, 30), suite_hash="sha256:tweaked")

    report = compare(before, after)

    assert report.comparable is False
    assert any("different evaluator suites" in w for w in report.warnings)


def test_runs_sharing_no_cases_are_not_comparable() -> None:
    before = make_run("run-a", {"a": True, "b": False})
    after = make_run("run-b", {"c": True, "d": False})

    report = compare(before, after)

    assert report.shared_cases == 0
    assert report.comparable is False
    assert any("share no cases" in w for w in report.warnings)


def test_only_shared_cases_are_compared() -> None:
    before = make_run("run-a", {"a": True, "b": True, "extra": False})
    after = make_run("run-b", {"a": True, "b": False, "other": True})

    report = compare(before, after)

    assert report.shared_cases == 2
    assert {t.case_id for t in report.transitions} == {"a", "b"}


def test_a_genuine_agent_change_is_not_warned_about() -> None:
    before = make_run("run-a", outcomes(20, 30), agent_ref="scripted:baseline")
    after = make_run(
        "run-b", outcomes(18, 30), agent_ref="scripted:regressed", agent_hash="sha256:other"
    )

    report = compare(before, after)

    assert not any("same agent" in w for w in report.warnings)


# -- power ---------------------------------------------------------------


def test_a_small_dataset_is_reported_as_underpowered() -> None:
    """"No significant change" and "too small to tell" are different findings."""
    before = make_run("run-a", outcomes(7, 10))
    after = make_run("run-b", outcomes(6, 10))

    report = compare(before, after)

    assert report.verdict == "no significant change"
    assert report.underpowered is True
    assert report.required_cases > 10


def test_a_detected_regression_is_not_flagged_underpowered() -> None:
    before = make_run("run-a", outcomes(27, 30))
    after = make_run("run-b", outcomes(18, 30))

    report = compare(before, after)

    assert report.underpowered is False


# -- dimensions ----------------------------------------------------------


def test_behavioural_dimensions_are_reported_alongside_the_outcome() -> None:
    before = make_run("run-a", outcomes(20, 30), trajectory_score=1.0)
    after = make_run("run-b", outcomes(20, 30), trajectory_score=0.6)

    report = compare(before, after)

    by_name = {d.name: d for d in report.dimensions}
    # Outcome unchanged, behaviour clearly worse — the case for measuring both.
    assert report.delta == 0.0
    assert by_name["trajectory"].delta < -0.39
    assert by_name["tests"].delta == 0.0


# -- k-sampled runs ------------------------------------------------------


def test_partial_pass_rates_are_compared_not_flattened() -> None:
    """A case that went 3/3 and one that went 1/3 are different outcomes."""
    before = RunResult(
        run_id="run-a",
        agent_ref="a",
        agent_hash="h1",
        dataset_name="synth",
        dataset_version="v1",
        dataset_hash="sha256:dataset",
        suite_name="default",
        suite_hash="sha256:suite",
        samples_per_case=2,
        case_results=(
            CaseResult(case_id="x", attempt=0, status="completed", passed=True, duration_s=1.0),
            CaseResult(case_id="x", attempt=1, status="completed", passed=True, duration_s=1.0),
        ),
    )
    after = before.model_copy(
        update={
            "run_id": "run-b",
            "agent_hash": "h2",
            "case_results": (
                CaseResult(
                    case_id="x", attempt=0, status="completed", passed=True, duration_s=1.0
                ),
                CaseResult(
                    case_id="x", attempt=1, status="completed", passed=False, duration_s=1.0
                ),
            ),
        }
    )

    report = compare(before, after)

    assert report.before_rate == 1.0
    assert report.after_rate == 0.5
    assert report.transitions[0].kind == "mixed"
