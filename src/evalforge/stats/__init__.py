"""Statistics for deciding whether a measured difference is real."""

from evalforge.stats.intervals import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    Interval,
    paired_bootstrap,
    wilson_interval,
)
from evalforge.stats.sampling import (
    StabilityReport,
    max_usable_k,
    pass_at_k,
    pass_hat_k,
    stability_report,
)
from evalforge.stats.significance import (
    DEFAULT_ALPHA,
    DEFAULT_POWER,
    McNemarResult,
    PairedCounts,
    binomial_two_sided_p,
    mcnemar,
    paired_counts,
    required_sample_size,
)

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_POWER",
    "DEFAULT_RESAMPLES",
    "Interval",
    "McNemarResult",
    "PairedCounts",
    "StabilityReport",
    "binomial_two_sided_p",
    "max_usable_k",
    "mcnemar",
    "paired_bootstrap",
    "paired_counts",
    "pass_at_k",
    "pass_hat_k",
    "required_sample_size",
    "stability_report",
    "wilson_interval",
]
