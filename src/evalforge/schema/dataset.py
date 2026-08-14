"""Versioned, content-addressed collections of evaluation cases."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import NamedTuple, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from evalforge.hashing import content_hash
from evalforge.schema.case import EvalCase

_VERSION_RE = re.compile(r"^v\d+$")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class DatasetRef(NamedTuple):
    """A parsed ``name@version`` reference, e.g. ``synth@v1``."""

    name: str
    version: str

    def __str__(self) -> str:
        return f"{self.name}@{self.version}"

    @classmethod
    def parse(cls, ref: str) -> Self:
        name, separator, version = ref.partition("@")
        if not separator:
            raise ValueError(f"dataset reference must be 'name@version', got {ref!r}")
        if not _NAME_RE.match(name):
            raise ValueError(f"invalid dataset name {name!r}")
        if not _VERSION_RE.match(version):
            raise ValueError(f"dataset version must look like 'v1', got {version!r}")
        return cls(name, version)


class Dataset(BaseModel):
    """An immutable, hashable set of cases.

    Mutating a dataset means publishing a new *version*: the content hash is
    recorded on every run, so a silently edited dataset shows up as a hash
    mismatch rather than as an unexplained metric shift.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str
    description: str = ""
    cases: tuple[EvalCase, ...] = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if not _NAME_RE.match(value):
            raise ValueError(f"invalid dataset name {value!r}")
        return value

    @field_validator("version")
    @classmethod
    def _check_version(cls, value: str) -> str:
        if not _VERSION_RE.match(value):
            raise ValueError(f"dataset version must look like 'v1', got {value!r}")
        return value

    @field_validator("cases")
    @classmethod
    def _check_unique_case_ids(cls, value: tuple[EvalCase, ...]) -> tuple[EvalCase, ...]:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for case in value:
            if case.case_id in seen:
                duplicates.add(case.case_id)
            seen.add(case.case_id)
        if duplicates:
            raise ValueError(f"duplicate case ids: {sorted(duplicates)}")
        return value

    @property
    def ref(self) -> DatasetRef:
        return DatasetRef(self.name, self.version)

    @property
    def content_hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))

    def case_by_id(self, case_id: str) -> EvalCase:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError(f"no case {case_id!r} in {self.ref}")

    def subset(self, case_ids: Iterable[str]) -> Dataset:
        """Return a new dataset containing only ``case_ids``, preserving order."""
        wanted = set(case_ids)
        missing = wanted - {case.case_id for case in self.cases}
        if missing:
            raise KeyError(f"unknown case ids: {sorted(missing)}")
        return self.model_copy(
            update={"cases": tuple(case for case in self.cases if case.case_id in wanted)}
        )


class DatasetManifest(BaseModel):
    """Sidecar describing an on-disk dataset, used to detect drift."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str
    description: str = ""
    case_count: int = Field(gt=0)
    content_hash: str
    generator: str = ""
    seed: int | None = None

    @classmethod
    def for_dataset(cls, dataset: Dataset, *, generator: str = "", seed: int | None = None) -> Self:
        return cls(
            name=dataset.name,
            version=dataset.version,
            description=dataset.description,
            case_count=len(dataset.cases),
            content_hash=dataset.content_hash,
            generator=generator,
            seed=seed,
        )
