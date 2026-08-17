"""Judge validation, verified against independently computed values.

The judge is the one evaluator that cannot be trusted on its own say-so, so the
statistics that measure it are the ones most worth checking by hand. Every
expected number below is derived on paper, not copied from a run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evalforge.judge_eval.agreement import (
    agreement_report,
    cohens_kappa,
    confusion_matrix,
    interpret_kappa,
)
from evalforge.judge_eval.bias import position_bias, self_preference
from evalforge.judge_eval.gold import GoldSet, JudgeExample
from evalforge.judge_eval.validation import JudgeIdentity, validate_judge


def make_gold(verdicts: list[str], *, name: str = "gold", version: str = "v1") -> GoldSet:
    return GoldSet(
        name=name,
        version=version,
        examples=tuple(
            JudgeExample(
                case_id=f"case-{index:03d}",
                task="Fix the off-by-one.",
                agent_summary="Adjusted the slice bound.",
                human_verdict=verdict,
                author_family="llama" if index % 2 else "other",
            )
            for index, verdict in enumerate(verdicts)
        ),
    )


JUDGE = JudgeIdentity(model="judge-model", prompt="Decide whether the fix is adequate.")


# -- Cohen's kappa -------------------------------------------------------


def test_kappa_matches_the_textbook_worked_example() -> None:
    """50 items: 20 both-yes, 5 only-human-yes, 10 only-judge-yes, 15 both-no.

    p_o = 35/50 = 0.70
    marginals: human yes 25/50, judge yes 30/50
    p_e = 0.5*0.6 + 0.5*0.4 = 0.50
    kappa = (0.70 - 0.50) / (1 - 0.50) = 0.40
    """
    human = ["yes"] * 20 + ["yes"] * 5 + ["no"] * 10 + ["no"] * 15
    judge = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15

    assert cohens_kappa(human, judge) == pytest.approx(0.40)


def test_perfect_agreement_is_one() -> None:
    human = ["pass", "fail", "pass", "fail", "pass"]

    assert cohens_kappa(human, list(human)) == 1.0


def test_total_disagreement_is_negative() -> None:
    human = ["pass", "fail", "pass", "fail"]
    judge = ["fail", "pass", "fail", "pass"]

    # Worse than chance, which κ can express and accuracy cannot.
    assert cohens_kappa(human, judge) == pytest.approx(-1.0)


def test_a_judge_that_always_says_the_majority_class_scores_about_zero() -> None:
    """The reason raw accuracy is not enough.

    Eighteen of twenty attempts genuinely failed. A judge that answers "fail"
    every time is right 90% of the time and has learned nothing.
    """
    human = ["fail"] * 18 + ["pass"] * 2
    lazy_judge = ["fail"] * 20

    report = agreement_report(human, lazy_judge)

    assert report.accuracy == pytest.approx(0.90)
    assert report.kappa == pytest.approx(0.0, abs=1e-9)
    assert report.interpretation == "none or worse than chance"


def test_kappa_over_an_empty_sample_is_an_error() -> None:
    with pytest.raises(ValueError, match="empty sample"):
        cohens_kappa([], [])


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        cohens_kappa(["pass"], ["pass", "fail"])


@pytest.mark.parametrize(
    ("kappa", "expected"),
    [(0.95, "almost perfect"), (0.70, "substantial"), (0.50, "moderate"), (0.30, "fair")],
)
def test_kappa_bands_read_conventionally(kappa: float, expected: str) -> None:
    assert interpret_kappa(kappa) == expected


# -- the full agreement report -------------------------------------------


def test_the_confusion_matrix_partitions_every_item() -> None:
    labels, counts = confusion_matrix(
        ["pass", "pass", "fail", "fail"], ["pass", "fail", "pass", "fail"]
    )

    assert labels == ("fail", "pass")
    assert counts[("pass", "pass")] == 1
    assert counts[("pass", "fail")] == 1
    assert sum(counts.values()) == 4


def test_per_class_precision_and_recall_are_computed_from_the_matrix() -> None:
    # Human: 3 pass, 3 fail. Judge calls one real fail a pass.
    human = ["pass", "pass", "pass", "fail", "fail", "fail"]
    judge = ["pass", "pass", "pass", "pass", "fail", "fail"]

    report = agreement_report(human, judge)
    by_label = {metrics.label: metrics for metrics in report.per_class}

    # 3 of 4 "pass" calls were right; all 3 real passes were found.
    assert by_label["pass"].precision == pytest.approx(0.75)
    assert by_label["pass"].recall == pytest.approx(1.0)
    # 2 of 3 real fails were caught.
    assert by_label["fail"].recall == pytest.approx(2 / 3)
    assert by_label["fail"].support == 3
    assert report.disagreements() == 1


def test_f1_is_the_harmonic_mean() -> None:
    human = ["pass", "pass", "fail", "fail"]
    judge = ["pass", "fail", "pass", "fail"]

    metrics = {m.label: m for m in agreement_report(human, judge).per_class}["pass"]

    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)


# -- position bias -------------------------------------------------------


def test_a_consistent_judge_shows_no_position_bias() -> None:
    """The same winner both times, which appears under the swapped label."""
    forward = ["a", "b", "a", "b"]
    swapped = ["b", "a", "b", "a"]

    report = position_bias(forward, swapped)

    assert report.consistency == 1.0
    assert report.flip_rate == 0.0
    assert report.positional_skew == 0.0


def test_a_judge_that_always_picks_the_first_option_is_caught() -> None:
    """Agreement scores cannot see this; only the swap can."""
    forward = ["a", "a", "a", "a"]
    swapped = ["a", "a", "a", "a"]

    report = position_bias(forward, swapped)

    assert report.consistency == 0.0
    assert report.favoured_first == 4
    assert report.positional_skew == pytest.approx(1.0)


def test_a_judge_that_always_picks_the_second_option_is_caught() -> None:
    report = position_bias(["b", "b", "b"], ["b", "b", "b"])

    assert report.favoured_second == 3
    assert report.positional_skew == pytest.approx(-1.0)


def test_partial_position_bias_is_quantified() -> None:
    forward = ["a", "b", "a", "a"]
    swapped = ["b", "a", "a", "a"]

    report = position_bias(forward, swapped)

    assert report.pairs == 4
    assert report.consistent == 2
    assert report.flip_rate == pytest.approx(0.5)


def test_mismatched_orderings_are_rejected() -> None:
    with pytest.raises(ValueError, match="same pairs"):
        position_bias(["a"], ["a", "b"])


# -- self-preference -----------------------------------------------------


def test_a_judge_favouring_its_own_family_is_quantified() -> None:
    verdicts = ["pass", "pass", "pass", "fail", "fail", "pass"]
    own = [True, True, True, False, False, False]

    report = self_preference(verdicts, own)

    assert report.own_rate == pytest.approx(1.0)
    assert report.other_rate == pytest.approx(1 / 3)
    assert report.gap == pytest.approx(2 / 3)
    assert report.comparable is True


def test_an_even_handed_judge_shows_no_gap() -> None:
    report = self_preference(["pass", "fail", "pass", "fail"], [True, True, False, False])

    assert report.gap == 0.0


def test_a_gap_needs_both_kinds_to_mean_anything() -> None:
    report = self_preference(["pass", "pass"], [True, True])

    assert report.comparable is False


# -- the combined validation report --------------------------------------


def test_a_good_judge_passes_and_a_lazy_one_does_not() -> None:
    gold = make_gold(["pass", "fail"] * 12)

    good = validate_judge(gold, list(gold.verdicts), judge=JUDGE)
    lazy = validate_judge(gold, ["fail"] * len(gold.examples), judge=JUDGE)

    assert good.passed is True
    assert good.agreement.kappa == 1.0
    assert lazy.passed is False
    assert lazy.agreement.kappa == pytest.approx(0.0, abs=1e-9)


def test_the_threshold_is_configurable() -> None:
    gold = make_gold(["pass", "fail"] * 12)
    # Judge gets 3 of 24 wrong.
    verdicts = list(gold.verdicts)
    verdicts[0] = verdicts[2] = verdicts[4] = "fail"

    lenient = validate_judge(gold, verdicts, judge=JUDGE, threshold=0.5)
    strict = validate_judge(gold, verdicts, judge=JUDGE, threshold=0.95)

    assert lenient.passed is True
    assert strict.passed is False


def test_a_verdict_count_mismatch_is_rejected() -> None:
    gold = make_gold(["pass", "fail"])

    with pytest.raises(ValueError, match="3 verdicts for 2 examples"):
        validate_judge(gold, ["pass", "fail", "pass"], judge=JUDGE)


def test_an_imbalanced_gold_set_is_flagged() -> None:
    gold = make_gold(["fail"] * 22 + ["pass"] * 2)

    report = validate_judge(gold, list(gold.verdicts), judge=JUDGE)

    assert any("close to meaningless" in w for w in report.warnings)


def test_a_small_gold_set_is_flagged() -> None:
    gold = make_gold(["pass", "fail"] * 4)

    report = validate_judge(gold, list(gold.verdicts), judge=JUDGE)

    assert any("only 8 labelled examples" in w for w in report.warnings)


def test_a_self_preferring_judge_earns_a_warning() -> None:
    gold = make_gold(["pass", "fail"] * 12)
    verdicts = ["pass" if example.author_family == "llama" else "fail" for example in gold.examples]

    report = validate_judge(
        gold,
        verdicts,
        judge=JUDGE,
        own_family_flags=[e.author_family == "llama" for e in gold.examples],
    )

    assert report.preference is not None
    assert any("self-preference gap" in w for w in report.warnings)


# -- judge identity ------------------------------------------------------


def test_changing_the_prompt_changes_the_instrument() -> None:
    first = JudgeIdentity(model="m", prompt="Be strict.")
    second = JudgeIdentity(model="m", prompt="Be lenient.")

    assert first.prompt_hash != second.prompt_hash
    assert first.fingerprint != second.fingerprint


def test_changing_the_model_or_temperature_changes_the_instrument() -> None:
    base = JudgeIdentity(model="m", prompt="p")

    assert base.fingerprint != JudgeIdentity(model="other", prompt="p").fingerprint
    assert base.fingerprint != JudgeIdentity(model="m", prompt="p", temperature=0.7).fingerprint


def test_two_reports_from_different_judges_are_not_comparable() -> None:
    gold = make_gold(["pass", "fail"] * 12)
    before = validate_judge(gold, list(gold.verdicts), judge=JUDGE)
    after = validate_judge(
        gold, list(gold.verdicts), judge=JudgeIdentity(model="judge-v2", prompt=JUDGE.prompt)
    )

    assert after.comparable_to(before) is False
    reasons = after.differences_from(before)
    # Silently comparing across judges is the mistake this exists to prevent.
    assert any("different judge model" in r for r in reasons)


def test_a_changed_gold_set_makes_reports_incomparable() -> None:
    before = validate_judge(
        make_gold(["pass", "fail"] * 12), ["pass", "fail"] * 12, judge=JUDGE
    )
    after = validate_judge(
        make_gold(["pass", "fail"] * 12 + ["pass", "fail"]),
        ["pass", "fail"] * 13,
        judge=JUDGE,
    )

    assert after.comparable_to(before) is False
    assert any("gold set changed" in r for r in after.differences_from(before))


def test_identical_judges_on_the_same_gold_set_are_comparable() -> None:
    gold = make_gold(["pass", "fail"] * 12)
    first = validate_judge(gold, list(gold.verdicts), judge=JUDGE)
    second = validate_judge(gold, list(gold.verdicts), judge=JUDGE)

    assert second.comparable_to(first) is True
    assert second.differences_from(first) == ()


# -- gold set round-tripping ---------------------------------------------


def test_a_gold_set_round_trips_through_a_file(tmp_path: Path) -> None:
    gold = make_gold(["pass", "fail", "pass"])
    path = gold.write(tmp_path / "judgments.jsonl")

    loaded = GoldSet.read(path, name=gold.name, version=gold.version)

    assert loaded.examples == gold.examples
    assert loaded.content_hash == gold.content_hash


def test_label_balance_is_reported() -> None:
    assert make_gold(["pass", "pass", "fail"]).label_balance() == {"fail": 1, "pass": 2}


def test_a_missing_gold_set_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no gold set"):
        GoldSet.read(tmp_path / "absent.jsonl")


def test_an_empty_gold_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no examples"):
        GoldSet.read(path)
