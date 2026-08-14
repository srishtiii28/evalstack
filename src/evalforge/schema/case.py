"""The unit of evaluation: a self-contained, reproducible task for an agent."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from evalforge.hashing import content_hash

Difficulty = Literal["easy", "medium", "hard"]


def validate_relative_path(path: str) -> str:
    """Reject anything that is not a tame relative path inside the workspace.

    This is a *schema*-level guard against malformed datasets. It is not the
    security boundary — that lives in :mod:`evalforge.env.workspace`, which
    resolves paths at runtime against a real filesystem root.
    """
    if not path:
        raise ValueError("path must not be empty")
    pure = PurePosixPath(path)
    if pure.is_absolute():
        raise ValueError(f"path must be relative, got {path!r}")
    if any(part == ".." for part in pure.parts):
        raise ValueError(f"path must not traverse upwards, got {path!r}")
    if str(pure) != path:
        raise ValueError(f"path must be normalised, got {path!r} (expected {str(pure)!r})")
    return path


RelativePath = Annotated[str, Field(min_length=1)]


class FileSpec(BaseModel):
    """A single file materialised into an agent's workspace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: RelativePath
    contents: str

    _validate_path = field_validator("path")(validate_relative_path)


class CaseMetadata(BaseModel):
    """Ground truth *about* a case, used for slicing, clustering and reporting.

    ``bug_kind`` is deliberately a plain string rather than an enum: synthetic
    cases draw it from a typed catalogue, but imported datasets (SWE-bench and
    friends) carry labels we do not control.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bug_kind: str | None = None
    difficulty: Difficulty = "medium"
    target_files: tuple[RelativePath, ...] = ()
    tags: tuple[str, ...] = ()

    @field_validator("target_files")
    @classmethod
    def _check_target_files(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for path in value:
            validate_relative_path(path)
        return value


class EvalCase(BaseModel):
    """One evaluation task: a workspace, an instruction, and a way to check it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    files: tuple[FileSpec, ...] = Field(min_length=1)
    test_command: tuple[str, ...] = Field(min_length=1)
    setup_commands: tuple[tuple[str, ...], ...] = ()
    timeout_s: float = Field(default=120.0, gt=0)
    metadata: CaseMetadata = CaseMetadata()

    # The fixed version of the files the reference solution touches. Present so
    # deterministic agents and oracle checks can be built without an LLM; it is
    # never shown to an agent under evaluation.
    reference_solution: tuple[FileSpec, ...] = ()

    @field_validator("files")
    @classmethod
    def _check_unique_paths(cls, value: tuple[FileSpec, ...]) -> tuple[FileSpec, ...]:
        paths = [spec.path for spec in value]
        duplicates = sorted({path for path in paths if paths.count(path) > 1})
        if duplicates:
            raise ValueError(f"duplicate file paths in case: {duplicates}")
        return value

    @property
    def content_hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))

    def file_map(self) -> dict[str, str]:
        """Workspace seed as a path -> contents mapping."""
        return {spec.path: spec.contents for spec in self.files}
