"""Statistics verified against independently computed values.

Every expected number here is derived by hand or from a published worked
example, never by running the implementation and pasting what it printed. A
test that records current behaviour cannot detect a wrong formula — and a
subtly wrong statistic is worse than none, because it launders a guess into a
decision.
"""

from __future__ import annotations

import random

import pytest

from evalforge.stats.intervals import Interval, paired_bootstrap, wilson_interval
from evalforge.stats.sampling import (
    max_usable_k,
    pass_at_k,
    pass_hat_k,
    stability_report,
)
from evalforge.stats.significance import (
    binomial_two_sided_p,
    mcnemar,
    paired_counts,
    required_sample_size,
)

# -- Wilson interval -----------------------------------------------------


def test_wilson_matches_the_published_value_for_zero_of_twenty() -> None:
    # Hand-computed with z = 1.959964:
    #   denominator = 1 + z^2/20                        = 1.192073
    #   centre      = (0 + z^2/40) / denominator        = 0.080565
    #   margin      = (z/denominator) * sqrt(z^2/1600)  = 0.080565
    result = wilson_interval(0, 20)

    assert result.estimate == 0.0
    assert result.low == pytest.approx(0.0, abs=1e-6)
    assert result.high == pytest.approx(0.161125, abs=1e-5)


def test_wilson_matches_the_published_value_for_five_of_twenty() -> None:
    # Hand-computed with z = 1.9599639845, z^2 = 3.841458821:
    #   denominator = 1 + z^2/20                     = 1.19207294
    #   centre      = (0.25 + z^2/40) / denominator  = 0.29028129
    #   margin      = (z/denominator)
    #                 * sqrt(0.25*0.75/20 + z^2/1600) = 0.17841963
    # Rounding z^2 to 3.8416 shifts the bound by ~1e-5, so carry the digits.
    result = wilson_interval(5, 20)

    assert result.estimate == 0.25
    assert result.low == pytest.approx(0.111862, abs=1e-5)
    assert result.high == pytest.approx(0.468701, abs=1e-5)


def test_wilson_stays_inside_the_unit_interval_at_the_extremes() -> None:
    # The normal approximation famously produces a negative lower bound here.
    for successes, trials in ((0, 5), (5, 5), (1, 3), (0, 1)):
        result = wilson_interval(successes, trials)
        assert 0.0 <= result.low <= result.high <= 1.0


def test_a_wider_confidence_level_gives_a_wider_interval() -> None:
    assert wilson_interval(7, 20, confidence=0.99).width > wilson_interval(7, 20).width


def test_the_interval_narrows_as_the_sample_grows() -> None:
    # The whole argument for bigger datasets, in one assertion.
    assert wilson_interval(70, 100).width < wilson_interval(7, 10).width


def test_thirty_tasks_at_seventy_percent_is_plus_or_minus_sixteen_points() -> None:
    """The number that motivates the entire milestone."""
    result = wilson_interval(21, 30)

    assert result.width == pytest.approx(0.30, abs=0.02)
    # So a four-point move between versions is well inside the noise.
    assert result.low < 0.70 - 0.04 < 0.70 + 0.04 < result.high


def test_no_trials_reports_total_ignorance() -> None:
    result = wilson_interval(0, 0)

    assert (result.low, result.high) == (0.0, 1.0)


@pytest.mark.parametrize(
    ("successes", "trials", "message"),
    [(-1, 5, "must not be negative"), (6, 5, "cannot exceed trials")],
)
def test_impossible_counts_are_rejected(successes: int, trials: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        wilson_interval(successes, trials)


# -- McNemar's exact test ------------------------------------------------


def test_binomial_p_matches_hand_computation() -> None:
    # 1 of 9 discordant pairs favouring "after":
    #   2 * (C(9,0) + C(9,1)) / 2^9 = 2 * 10 / 512 = 0.0390625
    assert binomial_two_sided_p(1, 9) == pytest.approx(0.0390625)
    # 2 of 9: 2 * (1 + 9 + 36) / 512 = 92 / 512
    assert binomial_two_sided_p(2, 9) == pytest.approx(0.1796875)
    # An even split cannot be evidence of anything.
    assert binomial_two_sided_p(5, 10) == 1.0


def test_comparing_a_run_against_itself_reports_no_change() -> None:
    """The false-positive case, which matters more than the true-positive one.

    A gate that fires when nothing changed gets switched off, and then it
    catches nothing at all.
    """
    outcomes = [True, False, True, True, False, True, False, False, True, True]

    result = mcnemar(outcomes, outcomes)

    assert result.counts.discordant == 0
    assert result.p_value == 1.0
    assert result.significant is False
    assert result.direction == "no change"


def test_a_clear_regression_is_detected() -> None:
    # Eight cases broke, one was fixed: p = 0.0390625.
    before = [True] * 8 + [False] + [True] * 11
    after = [False] * 8 + [True] + [True] * 11

    result = mcnemar(before, after)

    assert result.counts.only_before_passed == 8
    assert result.counts.only_after_passed == 1
    assert result.p_value == pytest.approx(0.0390625)
    assert result.significant is True
    assert result.direction == "regression"


def test_a_small_difference_is_not_called_significant() -> None:
    # Two broke, one fixed. Real teams ship on less than this.
    before = [True, True, False] + [True] * 20
    after = [False, False, True] + [True] * 20

    result = mcnemar(before, after)

    assert result.significant is False


def test_only_disagreements_count() -> None:
    """Adding cases both versions agree on cannot change the verdict."""
    before = [True, True, False]
    after = [False, False, True]

    small = mcnemar(before, after)
    padded = mcnemar(before + [True] * 100, after + [True] * 100)

    assert small.p_value == padded.p_value


def test_the_contingency_table_partitions_every_case() -> None:
    counts = paired_counts([True, True, False, False], [True, False, True, False])

    assert (counts.both_passed, counts.only_before_passed) == (1, 1)
    assert (counts.only_after_passed, counts.both_failed) == (1, 1)
    assert counts.total == 4
    assert counts.discordant == 2


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        mcnemar([True], [True, False])


# -- power ---------------------------------------------------------------


def test_required_sample_size_matches_the_textbook_case() -> None:
    # Detecting 0.50 -> 0.60 at alpha 0.05, power 0.80 needs ~388 per arm.
    assert required_sample_size(0.50, 0.10) == pytest.approx(388, abs=2)


def test_smaller_effects_need_more_tasks() -> None:
    assert required_sample_size(0.70, 0.02) > required_sample_size(0.70, 0.10)


def test_more_power_needs_more_tasks() -> None:
    assert required_sample_size(0.70, 0.05, power=0.95) > required_sample_size(0.70, 0.05)


def test_detecting_a_small_move_needs_far_more_than_thirty_tasks() -> None:
    """Why the synthetic dataset cannot settle a three-point difference."""
    assert required_sample_size(0.70, 0.03) > 30


def test_impossible_targets_are_rejected() -> None:
    with pytest.raises(ValueError, match="leaves the unit interval"):
        required_sample_size(0.95, 0.10)


# -- bootstrap -----------------------------------------------------------


def test_an_identical_pair_has_a_zero_width_interval_at_zero() -> None:
    values = [1.0, 0.0, 1.0, 1.0, 0.0]

    result = paired_bootstrap(values, values)

    assert result.estimate == 0.0
    assert (result.low, result.high) == (0.0, 0.0)
    assert result.excludes_zero is False


def test_a_constant_shift_is_recovered_exactly() -> None:
    before = [0.0] * 10
    after = [1.0] * 10

    result = paired_bootstrap(before, after)

    assert result.estimate == 1.0
    assert result.low == result.high == 1.0
    assert result.excludes_zero is True


def test_the_bootstrap_is_deterministic_for_a_seed() -> None:
    before = [1.0, 0.0] * 10
    after = [0.0, 1.0] * 10

    first = paired_bootstrap(before, after, seed=7)
    second = paired_bootstrap(before, after, seed=7)

    assert (first.low, first.high) == (second.low, second.high)


def test_the_interval_brackets_the_observed_difference() -> None:
    rng = random.Random(1)
    before = [float(rng.random() < 0.4) for _ in range(60)]
    after = [float(rng.random() < 0.7) for _ in range(60)]

    result = paired_bootstrap(before, after)

    assert result.low <= result.estimate <= result.high


@pytest.mark.slow
def test_the_interval_covers_the_truth_about_as_often_as_it_claims() -> None:
    """Coverage by simulation: a 95% interval should miss about 5% of the time.

    This is the check that distinguishes a correct bootstrap from one that
    merely returns plausible-looking numbers.
    """
    rng = random.Random(11)
    trials = 200
    covered = 0
    for _ in range(trials):
        # Same underlying rate for both, so the true difference is zero.
        before = [float(rng.random() < 0.5) for _ in range(40)]
        after = [float(rng.random() < 0.5) for _ in range(40)]
        interval = paired_bootstrap(before, after, resamples=400, seed=rng.randrange(10**6))
        if interval.low <= 0.0 <= interval.high:
            covered += 1

    coverage = covered / trials
    assert 0.88 <= coverage <= 0.99, f"95% interval covered {coverage:.0%} of the time"


def test_mismatched_or_empty_samples_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap([1.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="empty sample"):
        paired_bootstrap([], [])


def test_interval_formatting_is_readable() -> None:
    assert Interval(0.712, 0.65, 0.77).format() == "71.2% [65.0, 77.0]"


# -- pass@k and pass^k ---------------------------------------------------


def test_pass_at_one_is_just_the_success_rate() -> None:
    # The estimator must reduce to c/n at k=1, or it is not measuring capability.
    assert pass_at_k(5, 2, 1) == pytest.approx(0.4)
    assert pass_at_k(10, 7, 1) == pytest.approx(0.7)


def test_pass_at_k_matches_hand_computation() -> None:
    # 1 - C(3,3)/C(5,3) = 1 - 1/10
    assert pass_at_k(5, 2, 3) == pytest.approx(0.9)
    # 1 - C(4,2)/C(6,2) = 1 - 6/15
    assert pass_at_k(6, 2, 2) == pytest.approx(0.6)


def test_pass_at_k_saturates_when_failures_cannot_fill_a_subset() -> None:
    assert pass_at_k(5, 5, 2) == 1.0
    assert pass_at_k(5, 4, 2) == 1.0


def test_no_successes_gives_zero_capability() -> None:
    assert pass_at_k(5, 0, 2) == 0.0


def test_pass_hat_one_is_also_the_success_rate() -> None:
    assert pass_hat_k(5, 2, 1) == pytest.approx(0.4)


def test_pass_hat_k_matches_hand_computation() -> None:
    # C(2,2)/C(4,2) = 1/6
    assert pass_hat_k(4, 2, 2) == pytest.approx(1 / 6)
    # Not enough successes to fill the subset at all.
    assert pass_hat_k(5, 2, 3) == 0.0
    assert pass_hat_k(5, 5, 3) == 1.0


def test_capability_and_reliability_diverge_on_a_flaky_agent() -> None:
    """The distinction the two estimators exist to draw."""
    samples, correct, k = 10, 5, 5

    capability = pass_at_k(samples, correct, k)
    reliability = pass_hat_k(samples, correct, k)

    # Very likely to succeed at least once; almost never five times running.
    assert capability > 0.98
    assert reliability < 0.01


def test_a_perfectly_reliable_agent_scores_the_same_on_both() -> None:
    assert pass_at_k(6, 6, 3) == pass_hat_k(6, 6, 3) == 1.0


def test_asking_for_more_k_than_samples_is_an_error() -> None:
    with pytest.raises(ValueError, match="cannot estimate k=4 from only 3"):
        pass_at_k(3, 2, 4)


# -- aggregating across a dataset ----------------------------------------


def test_a_stability_report_separates_capability_from_reliability() -> None:
    tallies = {
        "always": (4, 4),
        "never": (0, 4),
        "flaky": (2, 4),
    }

    report = stability_report(tallies, k=2)

    assert report.cases == 3
    assert report.flaky_cases == 1
    assert report.pass_at_k > report.pass_hat_k
    assert report.reliability_gap > 0.0


def test_cases_with_too_few_samples_are_skipped_not_extrapolated() -> None:
    tallies = {"deep": (3, 5), "shallow": (1, 1)}

    report = stability_report(tallies, k=3)

    assert report.cases == 1


def test_usable_k_is_bounded_by_the_thinnest_case() -> None:
    assert max_usable_k([(3, 5), (1, 2), (4, 4)]) == 2
    assert max_usable_k([]) == 0
