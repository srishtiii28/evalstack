"""Recording agent execution into a :class:`~evalforge.schema.trajectory.Trajectory`.

Two decisions here exist to keep trajectories reproducible:

* Time comes from an injectable :class:`Clock`. Under :class:`FakeClock` a
  trajectory is byte-identical across runs, which is what lets replay and
  golden-file tests assert on the whole trace rather than on a filtered subset.
* Tool call ids are a monotonic counter, not UUIDs, for the same reason.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Protocol

from pydantic import JsonValue

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

DEFAULT_MAX_OUTPUT_CHARS = 4_000
_ELISION = "\n… [{omitted} characters omitted] …\n"


class Clock(Protocol):
    """Milliseconds elapsed since the start of the attempt."""

    def elapsed_ms(self) -> float: ...


class MonotonicClock:
    """Real elapsed time, zeroed when the recorder is created."""

    def __init__(self) -> None:
        self._start = time.monotonic()

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start) * 1000.0


class FakeClock:
    """Deterministic clock for tests; time only moves when you move it."""

    def __init__(self, start_ms: float = 0.0, auto_advance_ms: float = 0.0) -> None:
        self._now = start_ms
        self._auto_advance = auto_advance_ms

    def elapsed_ms(self) -> float:
        now = self._now
        self._now += self._auto_advance
        return now

    def advance(self, ms: float) -> None:
        self._now += ms


def truncate(text: str, limit: int = DEFAULT_MAX_OUTPUT_CHARS) -> str:
    """Shorten ``text`` to at most ``limit`` characters, keeping both ends.

    Head and tail are both kept because the informative part differs by source:
    a traceback leads with the failing assertion, a pytest run ends with the
    summary line.

    The elision marker is counted against the limit, so the result never exceeds
    it — a truncation that overshoots its own budget defeats the point when the
    output is about to be written into a trace file.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    if len(text) <= limit:
        return text

    # The marker's own length depends on the omitted count, so budget against
    # its widest possible form rather than iterating to a fixed point.
    widest_marker = len(_ELISION.format(omitted=len(text)))
    budget = max(0, limit - widest_marker)
    if budget == 0:
        return text[:limit]

    head = budget // 2
    tail = budget - head
    marker = _ELISION.format(omitted=len(text) - budget)
    return f"{text[:head]}{marker}{text[-tail:]}" if tail else f"{text[:head]}{marker}"


class TrajectoryRecorder:
    """Append-only builder for a single attempt's trajectory."""

    def __init__(
        self,
        *,
        run_id: str,
        case_id: str,
        attempt: int = 0,
        clock: Clock | None = None,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> None:
        self.run_id = run_id
        self.case_id = case_id
        self.attempt = attempt
        self._clock = clock if clock is not None else MonotonicClock()
        self._max_output_chars = max_output_chars
        self._events: list[Event] = []
        self._call_counter = 0

    # -- internals -------------------------------------------------------

    def _next_seq(self) -> tuple[int, float]:
        return len(self._events), self._clock.elapsed_ms()

    def _cut(self, text: str) -> str:
        return truncate(text, self._max_output_chars)

    # -- recording -------------------------------------------------------

    def task_started(self, *, prompt_hash: str) -> None:
        seq, t_ms = self._next_seq()
        self._events.append(
            TaskStarted(seq=seq, t_ms=t_ms, case_id=self.case_id, prompt_hash=prompt_hash)
        )

    def model_call(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: float,
        cached: bool = False,
        stop_reason: str | None = None,
    ) -> None:
        seq, t_ms = self._next_seq()
        self._events.append(
            ModelCall(
                seq=seq,
                t_ms=t_ms,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                cached=cached,
                stop_reason=stop_reason,
            )
        )

    def tool_call(self, *, tool: str, args: Mapping[str, JsonValue]) -> str:
        """Record a tool invocation and return its generated call id."""
        self._call_counter += 1
        call_id = f"call-{self._call_counter}"
        seq, t_ms = self._next_seq()
        self._events.append(
            ToolCall(seq=seq, t_ms=t_ms, call_id=call_id, tool=tool, args=dict(args))
        )
        return call_id

    def tool_result(
        self,
        *,
        call_id: str,
        tool: str,
        ok: bool,
        output: str = "",
        error: str | None = None,
    ) -> None:
        seq, t_ms = self._next_seq()
        self._events.append(
            ToolResult(
                seq=seq,
                t_ms=t_ms,
                call_id=call_id,
                tool=tool,
                ok=ok,
                output=self._cut(output),
                error=error,
            )
        )

    def file_edit(
        self,
        *,
        path: str,
        before_hash: str | None,
        after_hash: str,
        lines_added: int,
        lines_removed: int,
    ) -> None:
        seq, t_ms = self._next_seq()
        self._events.append(
            FileEdit(
                seq=seq,
                t_ms=t_ms,
                path=path,
                before_hash=before_hash,
                after_hash=after_hash,
                lines_added=lines_added,
                lines_removed=lines_removed,
            )
        )

    def command_run(
        self,
        *,
        argv: tuple[str, ...],
        exit_code: int | None,
        duration_ms: float,
        timed_out: bool = False,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        seq, t_ms = self._next_seq()
        self._events.append(
            CommandRun(
                seq=seq,
                t_ms=t_ms,
                argv=argv,
                exit_code=exit_code,
                duration_ms=duration_ms,
                timed_out=timed_out,
                stdout_tail=self._cut(stdout),
                stderr_tail=self._cut(stderr),
            )
        )

    def safety_violation(self, *, rule: str, detail: str, attempted: str) -> None:
        seq, t_ms = self._next_seq()
        self._events.append(
            SafetyViolation(seq=seq, t_ms=t_ms, rule=rule, detail=detail, attempted=attempted)
        )

    def submission(self, *, summary: str = "") -> None:
        seq, t_ms = self._next_seq()
        self._events.append(Submission(seq=seq, t_ms=t_ms, summary=self._cut(summary)))

    def agent_error(self, *, error_type: str, message: str) -> None:
        seq, t_ms = self._next_seq()
        self._events.append(
            AgentError(seq=seq, t_ms=t_ms, error_type=error_type, message=self._cut(message))
        )

    # -- output ----------------------------------------------------------

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    def build(self) -> Trajectory:
        return Trajectory(
            run_id=self.run_id,
            case_id=self.case_id,
            attempt=self.attempt,
            events=tuple(self._events),
        )
