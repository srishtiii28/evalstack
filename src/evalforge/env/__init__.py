"""Execution environments: where agent actions actually happen."""

from evalforge.env.workspace import (
    CommandResult,
    FileDiff,
    FileEditRecord,
    PathEscapeError,
    ResourceLimits,
    Violation,
    Workspace,
    WorkspaceError,
    workspace_for,
)

__all__ = [
    "CommandResult",
    "FileDiff",
    "FileEditRecord",
    "PathEscapeError",
    "ResourceLimits",
    "Violation",
    "Workspace",
    "WorkspaceError",
    "workspace_for",
]
