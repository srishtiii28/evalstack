"""The execution trace of a single agent attempt.

The trajectory is the spine of EvalForge: every evaluator reads it, and nothing
re-derives facts by scraping logs. Events form a discriminated union so adding a
new event type is a schema change that the type checker propagates to every
consumer, rather than an untyped dict that silently spreads.

Times are milliseconds since the start of the attempt (not wall-clock stamps) so
that a trajectory produced under a fake clock is byte-identical across runs —
which is what makes replay and golden-file tests meaningful.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter


class _EventBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int = Field(ge=0)
    t_ms: float = Field(ge=0)


class TaskStarted(_EventBase):
    kind: Literal["task_started"] = "task_started"
    case_id: str
    prompt_hash: str


class ModelCall(_EventBase):
    """One request/response pair against a language model."""

    kind: Literal["model_call"] = "model_call"
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    latency_ms: float = Field(ge=0)
    cached: bool = False
    stop_reason: str | None = None


class ToolCall(_EventBase):
    kind: Literal["tool_call"] = "tool_call"
    call_id: str
    tool: str
    args: dict[str, JsonValue] = Field(default_factory=dict)


class ToolResult(_EventBase):
    kind: Literal["tool_result"] = "tool_result"
    call_id: str
    tool: str
    ok: bool
    output: str = ""
    error: str | None = None


class FileEdit(_EventBase):
    kind: Literal["file_edit"] = "file_edit"
    path: str
    before_hash: str | None
    after_hash: str
    lines_added: int = Field(ge=0)
    lines_removed: int = Field(ge=0)


class CommandRun(_EventBase):
    kind: Literal["command_run"] = "command_run"
    argv: tuple[str, ...]
    exit_code: int | None
    duration_ms: float = Field(ge=0)
    timed_out: bool = False
    stdout_tail: str = ""
    stderr_tail: str = ""


class SafetyViolation(_EventBase):
    """A policy breach that was *contained*, recorded rather than raised.

    Recording instead of raising is deliberate: an agent that tries to escape the
    workspace should produce a measurable safety signal, not an infrastructure
    error that looks like a harness bug.
    """

    kind: Literal["safety_violation"] = "safety_violation"
    rule: str
    detail: str
    attempted: str


class Submission(_EventBase):
    kind: Literal["submission"] = "submission"
    summary: str = ""


class AgentError(_EventBase):
    kind: Literal["agent_error"] = "agent_error"
    error_type: str
    message: str


Event = Annotated[
    TaskStarted
    | ModelCall
    | ToolCall
    | ToolResult
    | FileEdit
    | CommandRun
    | SafetyViolation
    | Submission
    | AgentError,
    Field(discriminator="kind"),
]

EventT = TypeVar("EventT", bound=_EventBase)

_EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)


class Trajectory(BaseModel):
    """The complete, ordered record of one attempt at one case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    case_id: str
    attempt: int = Field(ge=0)
    events: tuple[Event, ...] = ()

    def of_type(self, event_type: type[EventT]) -> tuple[EventT, ...]:
        """All events of a given concrete type, in order."""
        return tuple(event for event in self.events if isinstance(event, event_type))

    @property
    def total_cost_usd(self) -> float:
        return sum(event.cost_usd for event in self.of_type(ModelCall))

    @property
    def total_input_tokens(self) -> int:
        return sum(event.input_tokens for event in self.of_type(ModelCall))

    @property
    def total_output_tokens(self) -> int:
        return sum(event.output_tokens for event in self.of_type(ModelCall))

    @property
    def duration_ms(self) -> float:
        return self.events[-1].t_ms if self.events else 0.0

    @property
    def submitted(self) -> bool:
        return bool(self.of_type(Submission))

    def to_jsonl(self) -> str:
        """One JSON object per line — greppable, streamable, diffable."""
        return "".join(f"{event.model_dump_json()}\n" for event in self.events)

    @classmethod
    def from_jsonl(cls, *, run_id: str, case_id: str, attempt: int, jsonl: str) -> Self:
        events = tuple(
            _EVENT_ADAPTER.validate_json(line) for line in jsonl.splitlines() if line.strip()
        )
        return cls(run_id=run_id, case_id=case_id, attempt=attempt, events=events)
