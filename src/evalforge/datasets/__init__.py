"""Generating, storing and verifying evaluation datasets."""

from evalforge.datasets.builder import blueprint_to_case, build_synthetic_dataset
from evalforge.datasets.catalogue import TEMPLATES, TaskBlueprint, bug_kinds
from evalforge.datasets.io import (
    DatasetIntegrityError,
    VerificationReport,
    read_dataset,
    read_manifest,
    verify_dataset,
    write_dataset,
)

__all__ = [
    "TEMPLATES",
    "DatasetIntegrityError",
    "TaskBlueprint",
    "VerificationReport",
    "blueprint_to_case",
    "bug_kinds",
    "build_synthetic_dataset",
    "read_dataset",
    "read_manifest",
    "verify_dataset",
    "write_dataset",
]
