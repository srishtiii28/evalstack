"""A catalogue of seeded bugs, each a distinct, realistic failure mode.

Every template produces a *working* module, a broken variant of it, and a test
suite that distinguishes the two. Because the injected failure mode is known,
this dataset doubles as ground truth: failure clustering can be scored against
the label it should have recovered, rather than eyeballed.

Templates vary the module name so cases are distinguishable; the genuine
diversity is across the eight bug kinds, not within them. Real-world variety is
the job of the SWE-bench-style loader that shares this same ``EvalCase`` schema.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from evalforge.schema.case import Difficulty

PACKAGE_NAME = "solver"


@dataclass(frozen=True, slots=True)
class TaskBlueprint:
    """One generated task: a broken module, its fix, and a test that tells them apart."""

    kind: str
    difficulty: Difficulty
    module_name: str
    fixed_source: str
    buggy_source: str
    test_source: str
    prompt: str

    @property
    def module_path(self) -> str:
        return f"{PACKAGE_NAME}/{self.module_name}.py"

    @property
    def test_path(self) -> str:
        return f"tests/test_{self.module_name}.py"


def _off_by_one(module: str) -> TaskBlueprint:
    fixed = '''"""Windowing helpers."""

from __future__ import annotations


def last_n(items: list[int], n: int) -> list[int]:
    """Return the final ``n`` items, in their original order."""
    if n <= 0:
        return []
    return items[-n:]
'''
    buggy = fixed.replace("    return items[-n:]\n", "    return items[-n + 1 :]\n")
    tests = f'''from {PACKAGE_NAME}.{module} import last_n


def test_returns_the_final_two_items():
    assert last_n([1, 2, 3, 4], 2) == [3, 4]


def test_returns_a_single_item():
    assert last_n([1, 2, 3], 1) == [3]


def test_non_positive_window_is_empty():
    assert last_n([1, 2, 3], 0) == []


def test_window_larger_than_input_returns_everything():
    assert last_n([1, 2], 5) == [1, 2]
'''
    prompt = (
        f"`last_n` in `{PACKAGE_NAME}/{module}.py` is meant to return the final `n` items of a "
        "list. It currently returns one item too few — `last_n([1, 2, 3, 4], 2)` gives `[4]` "
        "instead of `[3, 4]`. Fix it so the test suite passes."
    )
    return TaskBlueprint(
        kind="off_by_one",
        difficulty="easy",
        module_name=module,
        fixed_source=fixed,
        buggy_source=buggy,
        test_source=tests,
        prompt=prompt,
    )


def _inverted_comparison(module: str) -> TaskBlueprint:
    fixed = '''"""Expiry rules."""

from __future__ import annotations


def is_expired(age_days: int, max_age_days: int) -> bool:
    """An item is expired once its age reaches the maximum age."""
    return age_days >= max_age_days
'''
    buggy = fixed.replace(
        "    return age_days >= max_age_days\n", "    return age_days < max_age_days\n"
    )
    tests = f'''from {PACKAGE_NAME}.{module} import is_expired


def test_older_than_limit_is_expired():
    assert is_expired(10, 7) is True


def test_exactly_at_the_limit_is_expired():
    assert is_expired(7, 7) is True


def test_younger_than_limit_is_not_expired():
    assert is_expired(3, 7) is False
'''
    prompt = (
        f"`is_expired` in `{PACKAGE_NAME}/{module}.py` returns the opposite of what it should: "
        "items past their maximum age are reported as fresh, and fresh items as expired. An item "
        "counts as expired once its age reaches the limit. Fix it."
    )
    return TaskBlueprint(
        kind="inverted_comparison",
        difficulty="easy",
        module_name=module,
        fixed_source=fixed,
        buggy_source=buggy,
        test_source=tests,
        prompt=prompt,
    )


def _mutable_default(module: str) -> TaskBlueprint:
    fixed = '''"""Accumulation helpers."""

from __future__ import annotations


def collect(item: str, bucket: list[str] | None = None) -> list[str]:
    """Append ``item`` to ``bucket``, starting a fresh bucket when none is given."""
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket
'''
    buggy = '''"""Accumulation helpers."""

from __future__ import annotations


def collect(item: str, bucket: list[str] | None = None) -> list[str]:
    """Append ``item`` to ``bucket``, starting a fresh bucket when none is given."""
    if bucket is None:
        bucket = _DEFAULT_BUCKET
    bucket.append(item)
    return bucket


_DEFAULT_BUCKET: list[str] = []
'''
    tests = f'''from {PACKAGE_NAME}.{module} import collect


def test_appends_to_a_supplied_bucket():
    bucket = ["a"]
    assert collect("b", bucket) == ["a", "b"]


def test_each_call_starts_a_fresh_bucket():
    assert collect("first") == ["first"]
    assert collect("second") == ["second"]
'''
    prompt = (
        f"`collect` in `{PACKAGE_NAME}/{module}.py` is supposed to start a new list every time it "
        "is called without a bucket, but results leak between calls — the second call returns the "
        "first call's item too. Fix it."
    )
    return TaskBlueprint(
        kind="shared_mutable_state",
        difficulty="medium",
        module_name=module,
        fixed_source=fixed,
        buggy_source=buggy,
        test_source=tests,
        prompt=prompt,
    )


def _missing_empty_case(module: str) -> TaskBlueprint:
    fixed = '''"""Summary statistics."""

from __future__ import annotations


def mean(values: list[float]) -> float:
    """Arithmetic mean; an empty input is a caller error, not a crash."""
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)
'''
    buggy = '''"""Summary statistics."""

from __future__ import annotations


def mean(values: list[float]) -> float:
    """Arithmetic mean; an empty input is a caller error, not a crash."""
    return sum(values) / len(values)
'''
    tests = f'''import pytest

from {PACKAGE_NAME}.{module} import mean


def test_computes_the_mean():
    assert mean([1.0, 2.0, 3.0]) == 2.0


def test_empty_input_raises_value_error():
    with pytest.raises(ValueError):
        mean([])
'''
    prompt = (
        f"`mean` in `{PACKAGE_NAME}/{module}.py` raises `ZeroDivisionError` when given an empty "
        "list. It should raise `ValueError` instead, since an empty input is a caller mistake "
        "rather than an arithmetic accident. Fix it."
    )
    return TaskBlueprint(
        kind="missing_empty_case",
        difficulty="easy",
        module_name=module,
        fixed_source=fixed,
        buggy_source=buggy,
        test_source=tests,
        prompt=prompt,
    )


def _wrong_exception_type(module: str) -> TaskBlueprint:
    fixed = '''"""Registry lookups."""

from __future__ import annotations


def lookup(registry: dict[str, int], key: str) -> int:
    """Return the value for ``key``, or raise ``KeyError`` if absent."""
    if key not in registry:
        raise KeyError(key)
    return registry[key]
'''
    buggy = fixed.replace(
        '        raise KeyError(key)\n', '        raise RuntimeError(f"missing key: {key}")\n'
    )
    tests = f'''import pytest

from {PACKAGE_NAME}.{module} import lookup


def test_returns_a_present_value():
    assert lookup({{"a": 1}}, "a") == 1


def test_missing_key_raises_key_error():
    with pytest.raises(KeyError):
        lookup({{"a": 1}}, "b")
'''
    prompt = (
        f"`lookup` in `{PACKAGE_NAME}/{module}.py` raises `RuntimeError` for a missing key, so "
        "callers cannot catch it with the `KeyError` they expect from a mapping. Make it raise "
        "`KeyError` instead."
    )
    return TaskBlueprint(
        kind="wrong_exception_type",
        difficulty="easy",
        module_name=module,
        fixed_source=fixed,
        buggy_source=buggy,
        test_source=tests,
        prompt=prompt,
    )


def _integer_division(module: str) -> TaskBlueprint:
    fixed = '''"""Rate calculations."""

from __future__ import annotations


def success_rate(passed: int, total: int) -> float:
    """Fraction of attempts that passed, as a float in [0, 1]."""
    if total == 0:
        return 0.0
    return passed / total
'''
    buggy = fixed.replace("    return passed / total\n", "    return passed // total\n")
    tests = f'''from {PACKAGE_NAME}.{module} import success_rate


def test_partial_success_is_fractional():
    assert success_rate(1, 4) == 0.25


def test_full_success_is_one():
    assert success_rate(3, 3) == 1.0


def test_no_attempts_is_zero():
    assert success_rate(0, 0) == 0.0
'''
    prompt = (
        f"`success_rate` in `{PACKAGE_NAME}/{module}.py` should return a fraction between 0 and 1, "
        "but every partial result comes back as `0` — `success_rate(1, 4)` gives `0` instead of "
        "`0.25`. Fix it."
    )
    return TaskBlueprint(
        kind="integer_division",
        difficulty="easy",
        module_name=module,
        fixed_source=fixed,
        buggy_source=buggy,
        test_source=tests,
        prompt=prompt,
    )


def _exclusive_boundary(module: str) -> TaskBlueprint:
    fixed = '''"""Range helpers."""

from __future__ import annotations


def inclusive_range(start: int, stop: int) -> list[int]:
    """Every integer from ``start`` to ``stop``, including both endpoints."""
    if stop < start:
        return []
    return list(range(start, stop + 1))
'''
    buggy = fixed.replace(
        "    return list(range(start, stop + 1))\n", "    return list(range(start, stop))\n"
    )
    tests = f'''from {PACKAGE_NAME}.{module} import inclusive_range


def test_includes_both_endpoints():
    assert inclusive_range(1, 4) == [1, 2, 3, 4]


def test_single_point_range():
    assert inclusive_range(2, 2) == [2]


def test_reversed_bounds_are_empty():
    assert inclusive_range(5, 1) == []
'''
    prompt = (
        f"`inclusive_range` in `{PACKAGE_NAME}/{module}.py` drops the final value: "
        "`inclusive_range(1, 4)` returns `[1, 2, 3]` when it should include `4`. The range is "
        "meant to include both endpoints. Fix it."
    )
    return TaskBlueprint(
        kind="exclusive_boundary",
        difficulty="medium",
        module_name=module,
        fixed_source=fixed,
        buggy_source=buggy,
        test_source=tests,
        prompt=prompt,
    )


def _missing_tiebreak(module: str) -> TaskBlueprint:
    fixed = '''"""Leaderboard ordering."""

from __future__ import annotations


def rank(entries: list[tuple[str, int]]) -> list[str]:
    """Names ordered by descending score, ties broken alphabetically."""
    ordered = sorted(entries, key=lambda entry: (-entry[1], entry[0]))
    return [name for name, _ in ordered]
'''
    buggy = fixed.replace(
        "    ordered = sorted(entries, key=lambda entry: (-entry[1], entry[0]))\n",
        "    ordered = sorted(entries, key=lambda entry: -entry[1])\n",
    )
    tests = f'''from {PACKAGE_NAME}.{module} import rank


def test_orders_by_descending_score():
    assert rank([("ana", 1), ("bo", 5), ("cy", 3)]) == ["bo", "cy", "ana"]


def test_ties_are_broken_alphabetically():
    assert rank([("zoe", 4), ("ada", 4)]) == ["ada", "zoe"]
'''
    prompt = (
        f"`rank` in `{PACKAGE_NAME}/{module}.py` orders entries by descending score, but entries "
        "with equal scores come back in whatever order they were supplied. Equal scores should be "
        "broken alphabetically by name. Fix it."
    )
    return TaskBlueprint(
        kind="missing_tiebreak",
        difficulty="medium",
        module_name=module,
        fixed_source=fixed,
        buggy_source=buggy,
        test_source=tests,
        prompt=prompt,
    )


#: Every template, in a stable order so generation is reproducible.
TEMPLATES: tuple[Callable[[str], TaskBlueprint], ...] = (
    _off_by_one,
    _inverted_comparison,
    _mutable_default,
    _missing_empty_case,
    _wrong_exception_type,
    _integer_division,
    _exclusive_boundary,
    _missing_tiebreak,
)

#: Domain words used to give generated modules distinguishable names.
MODULE_WORDS: tuple[str, ...] = (
    "orders",
    "sessions",
    "metrics",
    "invoices",
    "playlists",
    "sensors",
    "tickets",
    "shipments",
    "batches",
    "routes",
    "ledgers",
    "signals",
)


def bug_kinds() -> tuple[str, ...]:
    """Every bug kind this catalogue can produce."""
    return tuple(sorted({template("probe").kind for template in TEMPLATES}))
