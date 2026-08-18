"""Grouping failures so a regression points somewhere.

"Success dropped four points" is a number. "Nine cases moved into the cluster
*edited the right file, tests still red*" is something an engineer can act on,
and the difference between the two is most of what makes an evaluation useful.

Clustering here is **deterministic and feature-based**, not embedding-based.
Three reasons, in order of how much they matter:

* It can be *scored*. The synthetic dataset injects known bug kinds, so cluster
  purity against those labels is measurable rather than eyeballed.
* It is reproducible. The same results always produce the same clusters, so a
  cluster that grows between runs grew because behaviour changed.
* It is free and explainable. Every cluster is a tuple of observable facts, and
  its name is derived from them — nobody has to trust a projection they cannot
  inspect.

The signals come from stored ``CaseResult`` detail rather than from trajectory
files, so clustering works on any run the store still has, including ones whose
traces have been cleaned up.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from evalforge.schema.result import CaseResult, RunResult

UNKNOWN = "unknown"


def _detail(result: CaseResult, evaluator: str) -> Mapping[str, Any]:
    found = result.evaluator(evaluator)
    return found.detail if found is not None else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


@dataclass(frozen=True, slots=True)
class FailureSignature:
    """The observable shape of one failure.

    Every field is a small closed set on purpose: clustering by exact signature
    only groups meaningfully when the features are coarse enough to repeat.
    """

    #: What the agent changed, relative to the case's declared target files.
    edit: str
    #: Whether the agent ran the test suite at all.
    ran_tests: bool
    #: How the attempt ended, as far as the stored result can tell.
    outcome: str
    #: Whether anything was flagged by the safety evaluator.
    unsafe: bool

    @property
    def label(self) -> str:
        """A human-readable name derived from the signature, not invented for it."""
        if self.outcome == "infrastructure":
            return "infrastructure fault, not the agent"
        if self.outcome == "timed_out":
            return "ran out of time"
        if self.unsafe:
            return "unsafe behaviour recorded"
        if self.edit == "none":
            return "changed nothing"
        if not self.ran_tests:
            return "edited without ever running the tests"
        if self.edit == "off_target":
            return "edited the wrong files"
        if self.edit == "wider_than_target":
            return "fixed the target but disturbed other files"
        return "edited the right file, tests still failing"


@dataclass(frozen=True, slots=True)
class FailureCluster:
    """A group of failures that look the same, and what they were caused by."""

    signature: FailureSignature
    case_ids: tuple[str, ...]
    bug_kinds: tuple[tuple[str, int], ...] = ()

    @property
    def label(self) -> str:
        return self.signature.label

    @property
    def size(self) -> int:
        return len(self.case_ids)

    @property
    def dominant_bug_kind(self) -> str:
        return self.bug_kinds[0][0] if self.bug_kinds else UNKNOWN

    @property
    def purity(self) -> float:
        """Fraction of the cluster sharing its most common bug kind.

        1.0 means the cluster corresponds to exactly one seeded fault; a low
        value means this failure shape spans several causes, which is itself
        worth knowing.
        """
        labelled = sum(count for _, count in self.bug_kinds)
        if not labelled:
            return 0.0
        return self.bug_kinds[0][1] / labelled


def signature_for(result: CaseResult) -> FailureSignature:
    """Reduce one failed attempt to its observable shape."""
    if result.status == "infra_error":
        return FailureSignature(
            edit="none", ran_tests=False, outcome="infrastructure", unsafe=False
        )
    if result.status == "timed_out":
        return FailureSignature(edit=UNKNOWN, ran_tests=False, outcome="timed_out", unsafe=False)

    patch = _detail(result, "patch_locality")
    touched = _as_list(patch.get("touched"))
    unrelated = _as_list(patch.get("unrelated_files"))
    untouched_targets = _as_list(patch.get("untouched_targets"))

    if not touched:
        edit = "none"
    elif unrelated and untouched_targets:
        edit = "off_target"
    elif unrelated:
        edit = "wider_than_target"
    elif untouched_targets:
        edit = "off_target"
    else:
        edit = "on_target"

    trajectory = _detail(result, "trajectory")
    test_runs = trajectory.get("test_runs")
    ran_tests = bool(test_runs) if isinstance(test_runs, int) else False

    safety = _detail(result, "safety")
    finding_count = safety.get("finding_count")
    unsafe = bool(finding_count) if isinstance(finding_count, int) else False

    tests = _detail(result, "tests")
    outcome = "timed_out" if tests.get("timed_out") else "tests_failing"

    return FailureSignature(edit=edit, ran_tests=ran_tests, outcome=outcome, unsafe=unsafe)


def cluster_failures(
    run: RunResult, *, bug_kinds: Mapping[str, str] | None = None
) -> tuple[FailureCluster, ...]:
    """Group a run's failures by shape, largest cluster first.

    ``bug_kinds`` maps case id to its seeded fault, which turns the clusters
    into something scoreable. Without it the clusters are still useful, just not
    checkable.
    """
    grouped: dict[FailureSignature, list[str]] = {}
    for result in run.case_results:
        if result.passed:
            continue
        grouped.setdefault(signature_for(result), []).append(result.case_id)

    clusters: list[FailureCluster] = []
    for signature, case_ids in grouped.items():
        kinds = Counter(
            bug_kinds[case_id]
            for case_id in case_ids
            if bug_kinds is not None and case_id in bug_kinds
        )
        clusters.append(
            FailureCluster(
                signature=signature,
                case_ids=tuple(sorted(case_ids)),
                bug_kinds=tuple(kinds.most_common()),
            )
        )

    # Largest first, then by label so the order never depends on dict insertion.
    return tuple(sorted(clusters, key=lambda c: (-c.size, c.label)))


def overall_purity(clusters: Sequence[FailureCluster]) -> float:
    """Size-weighted mean purity across clusters.

    The measure that makes "does clustering recover the seeded faults?" a
    question with a number rather than an opinion.
    """
    labelled = [cluster for cluster in clusters if cluster.bug_kinds]
    total = sum(cluster.size for cluster in labelled)
    if not total:
        return 0.0
    return sum(cluster.purity * cluster.size for cluster in labelled) / total


def cluster_shift(
    before: Sequence[FailureCluster], after: Sequence[FailureCluster]
) -> tuple[tuple[str, int, int], ...]:
    """How each failure shape's population changed between two runs.

    This is what turns a regression into a sentence: not "-4%" but "the cluster
    *changed nothing* grew by nine".
    """
    labels = {cluster.label for cluster in before} | {cluster.label for cluster in after}
    sizes_before = {cluster.label: cluster.size for cluster in before}
    sizes_after = {cluster.label: cluster.size for cluster in after}
    shifts = [
        (label, sizes_before.get(label, 0), sizes_after.get(label, 0)) for label in sorted(labels)
    ]
    return tuple(
        sorted(shifts, key=lambda row: (-(row[2] - row[1]), row[0]))
    )
