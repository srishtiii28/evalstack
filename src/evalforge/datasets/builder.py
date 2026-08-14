"""Assemble catalogue blueprints into a reproducible :class:`Dataset`."""

from __future__ import annotations

import random

from evalforge.datasets.catalogue import (
    MODULE_WORDS,
    PACKAGE_NAME,
    TEMPLATES,
    TaskBlueprint,
)
from evalforge.schema.case import CaseMetadata, EvalCase, FileSpec
from evalforge.schema.dataset import Dataset

DEFAULT_NAME = "synth"
DEFAULT_VERSION = "v1"
DEFAULT_COUNT = 30
DEFAULT_SEED = 7
GENERATOR_ID = "evalforge.datasets.builder"

#: Every generated case runs its suite the same way.
TEST_COMMAND: tuple[str, ...] = ("python", "-m", "pytest", "-q", "tests")
CASE_TIMEOUT_S = 120.0


def _package_init() -> FileSpec:
    return FileSpec(path=f"{PACKAGE_NAME}/__init__.py", contents='"""Generated task package."""\n')


def blueprint_to_case(blueprint: TaskBlueprint, case_id: str) -> EvalCase:
    """Turn a blueprint into a case whose workspace starts out broken."""
    return EvalCase(
        case_id=case_id,
        prompt=blueprint.prompt,
        files=(
            _package_init(),
            FileSpec(path=blueprint.module_path, contents=blueprint.buggy_source),
            FileSpec(path=blueprint.test_path, contents=blueprint.test_source),
        ),
        test_command=TEST_COMMAND,
        timeout_s=CASE_TIMEOUT_S,
        metadata=CaseMetadata(
            bug_kind=blueprint.kind,
            difficulty=blueprint.difficulty,
            target_files=(blueprint.module_path,),
            tags=("synthetic", "python"),
        ),
        reference_solution=(
            FileSpec(path=blueprint.module_path, contents=blueprint.fixed_source),
        ),
    )


def build_synthetic_dataset(
    *,
    name: str = DEFAULT_NAME,
    version: str = DEFAULT_VERSION,
    count: int = DEFAULT_COUNT,
    seed: int = DEFAULT_SEED,
) -> Dataset:
    """Generate ``count`` cases, cycling the catalogue so bug kinds stay balanced.

    Generation is a pure function of ``(count, seed)``: the same arguments give a
    byte-identical dataset and therefore an identical content hash.
    """
    if count < 1:
        raise ValueError("count must be at least 1")

    rng = random.Random(seed)
    words = list(MODULE_WORDS)
    rng.shuffle(words)

    cases: list[EvalCase] = []
    for index in range(count):
        template = TEMPLATES[index % len(TEMPLATES)]
        blueprint = template(words[index % len(words)])
        cases.append(blueprint_to_case(blueprint, case_id=f"{blueprint.kind}-{index:03d}"))

    return Dataset(
        name=name,
        version=version,
        description=(
            f"{count} synthetic Python repair tasks with seeded bugs drawn from "
            f"{len(TEMPLATES)} failure modes (seed {seed})."
        ),
        cases=tuple(cases),
    )
