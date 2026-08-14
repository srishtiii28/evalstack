"""Reading and writing datasets on disk, with drift detection.

Layout is deliberately boring — one JSON object per line plus a manifest — so a
dataset is greppable, diffable in review, and streamable if it grows. The
manifest carries the content hash, which is what turns "did this dataset change?"
into a question with an answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evalforge.schema.case import EvalCase
from evalforge.schema.dataset import Dataset, DatasetManifest

CASES_FILENAME = "cases.jsonl"
MANIFEST_FILENAME = "manifest.json"


class DatasetIntegrityError(Exception):
    """On-disk cases no longer match the manifest that describes them."""


@dataclass(frozen=True, slots=True)
class VerificationReport:
    directory: Path
    name: str
    version: str
    case_count: int
    expected_hash: str
    actual_hash: str

    @property
    def ok(self) -> bool:
        return self.expected_hash == self.actual_hash


def write_dataset(
    dataset: Dataset,
    directory: Path,
    *,
    generator: str = "",
    seed: int | None = None,
) -> Path:
    """Write ``dataset`` to ``directory``, returning the directory."""
    directory.mkdir(parents=True, exist_ok=True)

    lines = "".join(f"{case.model_dump_json()}\n" for case in dataset.cases)
    (directory / CASES_FILENAME).write_text(lines, encoding="utf-8")

    manifest = DatasetManifest.for_dataset(dataset, generator=generator, seed=seed)
    (directory / MANIFEST_FILENAME).write_text(
        f"{manifest.model_dump_json(indent=2)}\n", encoding="utf-8"
    )
    return directory


def read_manifest(directory: Path) -> DatasetManifest:
    path = directory / MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no dataset manifest at {path}")
    return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))


def read_dataset(directory: Path, *, verify: bool = True) -> Dataset:
    """Load a dataset, optionally refusing one that has drifted from its manifest."""
    manifest = read_manifest(directory)
    cases_path = directory / CASES_FILENAME
    if not cases_path.is_file():
        raise FileNotFoundError(f"no cases file at {cases_path}")

    cases = tuple(
        EvalCase.model_validate_json(line)
        for line in cases_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    dataset = Dataset(
        name=manifest.name,
        version=manifest.version,
        description=manifest.description,
        cases=cases,
    )

    if verify and dataset.content_hash != manifest.content_hash:
        raise DatasetIntegrityError(
            f"dataset at {directory} has drifted from its manifest: "
            f"expected {manifest.content_hash}, computed {dataset.content_hash}"
        )
    return dataset


def verify_dataset(directory: Path) -> VerificationReport:
    """Recompute the content hash and compare it with the manifest."""
    manifest = read_manifest(directory)
    dataset = read_dataset(directory, verify=False)
    return VerificationReport(
        directory=directory,
        name=dataset.name,
        version=dataset.version,
        case_count=len(dataset.cases),
        expected_hash=manifest.content_hash,
        actual_hash=dataset.content_hash,
    )
