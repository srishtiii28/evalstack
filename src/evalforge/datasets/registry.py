"""Resolving a dataset from a reference or a path.

``synth@v1`` reads ``<root>/synth`` and checks its manifest actually says ``v1``,
so a run that claims a dataset version cannot quietly have used another.
"""

from __future__ import annotations

from pathlib import Path

from evalforge.datasets.io import read_dataset, read_manifest
from evalforge.paths import DEFAULT_DATASETS_ROOT
from evalforge.schema.dataset import Dataset, DatasetRef


class DatasetNotFoundError(Exception):
    """The requested dataset is not on disk where it was expected."""


def dataset_directory(ref: DatasetRef, *, root: Path | None = None) -> Path:
    return (root or DEFAULT_DATASETS_ROOT) / ref.name


def resolve_dataset(spec: str, *, root: Path | None = None) -> Dataset:
    """Load a dataset from ``name@version`` or from a directory path."""
    if "@" in spec:
        ref = DatasetRef.parse(spec)
        directory = dataset_directory(ref, root=root)
        if not directory.is_dir():
            raise DatasetNotFoundError(
                f"no dataset directory for {ref} at {directory}; "
                f"generate one with: evalforge dataset build --name {ref.name}"
            )
        manifest = read_manifest(directory)
        if manifest.version != ref.version:
            raise DatasetNotFoundError(
                f"{directory} holds {manifest.name}@{manifest.version}, not {ref}"
            )
        return read_dataset(directory)

    directory = Path(spec)
    if not directory.is_dir():
        raise DatasetNotFoundError(f"no dataset directory at {directory}")
    return read_dataset(directory)
