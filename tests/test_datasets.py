"""Dataset generation, on-disk round-tripping, and drift detection.

The load-bearing test here is :func:`test_every_template_fails_broken_and_passes_fixed`:
a seeded bug that does not actually fail, or a reference fix that does not
actually pass, would make every downstream metric meaningless.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evalforge.datasets.builder import blueprint_to_case, build_synthetic_dataset
from evalforge.datasets.catalogue import TEMPLATES, bug_kinds
from evalforge.datasets.io import (
    CASES_FILENAME,
    MANIFEST_FILENAME,
    DatasetIntegrityError,
    read_dataset,
    verify_dataset,
    write_dataset,
)
from evalforge.env.workspace import workspace_for


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda t: t("probe").kind)
async def test_every_template_fails_broken_and_passes_fixed(template, tmp_path: Path) -> None:
    blueprint = template("subject")
    case = blueprint_to_case(blueprint, case_id="probe")

    with workspace_for(case, base_dir=tmp_path) as ws:
        broken = await ws.run(case.test_command, timeout_s=120)
        assert broken.ok is False, (
            f"{blueprint.kind}: seeded bug did not fail its own suite\n{broken.stdout}"
        )

        for spec in case.reference_solution:
            ws.write_file(spec.path, spec.contents)

        repaired = await ws.run(case.test_command, timeout_s=120)
        assert repaired.ok is True, (
            f"{blueprint.kind}: reference fix did not pass\n{repaired.stdout}\n{repaired.stderr}"
        )


def test_generation_is_deterministic_for_a_seed() -> None:
    first = build_synthetic_dataset(count=16, seed=7)
    second = build_synthetic_dataset(count=16, seed=7)

    assert first.content_hash == second.content_hash


def test_different_seeds_give_different_datasets() -> None:
    assert build_synthetic_dataset(count=16, seed=7).content_hash != (
        build_synthetic_dataset(count=16, seed=8).content_hash
    )


def test_case_ids_are_unique_and_cover_every_bug_kind() -> None:
    dataset = build_synthetic_dataset(count=len(TEMPLATES) * 2, seed=7)

    case_ids = [case.case_id for case in dataset.cases]
    assert len(set(case_ids)) == len(case_ids)

    produced = {case.metadata.bug_kind for case in dataset.cases}
    assert produced == set(bug_kinds())


def test_every_case_carries_ground_truth_metadata() -> None:
    dataset = build_synthetic_dataset(count=8, seed=7)

    for case in dataset.cases:
        assert case.metadata.bug_kind
        assert case.metadata.target_files
        assert case.reference_solution
        # The reference fix must target a file the case actually ships.
        shipped = set(case.file_map())
        assert {spec.path for spec in case.reference_solution} <= shipped


def test_count_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="count must be at least 1"):
        build_synthetic_dataset(count=0)


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    dataset = build_synthetic_dataset(count=8, seed=7)
    directory = write_dataset(dataset, tmp_path / "synth", generator="test", seed=7)

    assert (directory / CASES_FILENAME).is_file()
    assert (directory / MANIFEST_FILENAME).is_file()

    loaded = read_dataset(directory)
    assert loaded.content_hash == dataset.content_hash
    assert loaded.cases == dataset.cases


def test_verify_reports_a_clean_dataset(tmp_path: Path) -> None:
    dataset = build_synthetic_dataset(count=8, seed=7)
    directory = write_dataset(dataset, tmp_path / "synth")

    report = verify_dataset(directory)
    assert report.ok is True
    assert report.case_count == 8
    assert report.expected_hash == report.actual_hash


def test_edited_cases_are_detected_as_drift(tmp_path: Path) -> None:
    dataset = build_synthetic_dataset(count=4, seed=7)
    directory = write_dataset(dataset, tmp_path / "synth")

    cases_path = directory / CASES_FILENAME
    cases_path.write_text(cases_path.read_text(encoding="utf-8").replace("Fix it", "Fix it now"))

    report = verify_dataset(directory)
    assert report.ok is False

    with pytest.raises(DatasetIntegrityError, match="drifted from its manifest"):
        read_dataset(directory)


def test_reading_a_missing_dataset_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no dataset manifest"):
        read_dataset(tmp_path / "nowhere")
