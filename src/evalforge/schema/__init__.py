"""Declarative data model for cases, datasets, trajectories and results."""

from evalforge.schema.case import CaseMetadata, Difficulty, EvalCase, FileSpec
from evalforge.schema.dataset import Dataset, DatasetManifest, DatasetRef
from evalforge.schema.result import (
    CaseResult,
    CaseStatus,
    CaseTally,
    EvaluatorResult,
    RunResult,
)
from evalforge.schema.trajectory import (
    AgentError,
    CommandRun,
    Event,
    FileEdit,
    ModelCall,
    SafetyViolation,
    Submission,
    TaskStarted,
    ToolCall,
    ToolResult,
    Trajectory,
)

__all__ = [
    "AgentError",
    "CaseMetadata",
    "CaseResult",
    "CaseStatus",
    "CaseTally",
    "CommandRun",
    "Dataset",
    "DatasetManifest",
    "DatasetRef",
    "Difficulty",
    "EvalCase",
    "EvaluatorResult",
    "Event",
    "FileEdit",
    "FileSpec",
    "ModelCall",
    "RunResult",
    "SafetyViolation",
    "Submission",
    "TaskStarted",
    "ToolCall",
    "ToolResult",
    "Trajectory",
]
