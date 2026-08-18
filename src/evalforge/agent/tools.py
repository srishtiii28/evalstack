"""The tool surface every agent acts through.

One implementation, shared by the deterministic agent and the model-driven one.
That sharing is the point: evaluators reason about `read_file` and `run_tests`
events without caring which kind of agent produced them, so a trajectory metric
means the same thing for both.

Every tool records a ``ToolCall`` and a matching ``ToolResult``. Failures —
including refused path escapes — come back as an unsuccessful outcome carrying a
message the agent can act on, never as an exception that would look like a
harness fault.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import JsonValue

from evalforge.env.workspace import PathEscapeError, Workspace
from evalforge.model.base import ToolSpec
from evalforge.schema.case import EvalCase
from evalforge.trace import TrajectoryRecorder

#: The tool vocabulary. Kept small on purpose: a wider surface makes trajectories
#: harder to compare across agents without making the task more solvable.
TOOL_NAMES: tuple[str, ...] = ("list_files", "read_file", "write_file", "run_tests", "submit")

#: Named tool surfaces. Which one an agent gets is part of its configuration and
#: its hash, because the tools available to an agent change what it can do at
#: least as much as its prompt does.
WHOLE_FILE_SURFACE = "whole-file"
TARGETED_SURFACE = "targeted"
TOOL_SURFACES: tuple[str, ...] = (WHOLE_FILE_SURFACE, TARGETED_SURFACE)

MAX_LISTED_FILES = 200

_NO_ARGUMENTS: dict[str, JsonValue] = {"type": "object", "properties": {}, "required": []}


def tool_specs(surface: str = WHOLE_FILE_SURFACE) -> tuple[ToolSpec, ...]:
    """The tool surface as described to a model.

    Lives beside the implementation so a tool cannot be renamed or given a new
    argument without its schema moving too. The descriptions carry the two
    instructions models most often get wrong: that a write replaces the whole
    file, and that submitting is an explicit act rather than something implied
    by falling silent.
    """
    if surface not in TOOL_SURFACES:
        raise ValueError(f"unknown tool surface {surface!r}; known: {', '.join(TOOL_SURFACES)}")

    specs: tuple[ToolSpec, ...] = (
        ToolSpec(
            name="list_files",
            description="List every file in the workspace, as relative paths.",
            parameters=_NO_ARGUMENTS,
        ),
        ToolSpec(
            name="read_file",
            description="Read one file from the workspace and return its full contents.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path, e.g. solver/orders.py",
                    }
                },
                "required": ["path"],
            },
        ),
        ToolSpec(
            name="write_file",
            description=(
                "Write a file, replacing its entire contents. There is no partial edit: "
                "send the complete corrected file, not a diff or a fragment."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative path."},
                    "contents": {
                        "type": "string",
                        "description": "The complete new contents of the file.",
                    },
                },
                "required": ["path", "contents"],
            },
        ),
        ToolSpec(
            name="run_tests",
            description="Run the repository's test suite and return its output.",
            parameters=_NO_ARGUMENTS,
        ),
        ToolSpec(
            name="submit",
            description=(
                "Finish the task. Call this once the tests pass; the attempt does not "
                "end until you do."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "One sentence on what you changed and why.",
                    }
                },
                "required": ["summary"],
            },
        ),
    )
    if surface == TARGETED_SURFACE:
        specs = (*specs, _replace_text_spec())
    return specs


def _replace_text_spec() -> ToolSpec:
    return ToolSpec(
        name="replace_text",
        description=(
            "Replace one exact snippet in a file. Prefer this over write_file for "
            "edits: it needs only the few lines that change, so there is far less "
            "to quote and escape. The snippet must appear exactly once."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative path."},
                "old": {
                    "type": "string",
                    "description": "Exact text to replace, unique within the file.",
                },
                "new": {"type": "string", "description": "Text to put in its place."},
            },
            "required": ["path", "old", "new"],
        },
    )


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """What a tool returned, in the form an agent sees it."""

    ok: bool
    output: str = ""
    error: str | None = None

    @classmethod
    def failure(cls, error: str) -> ToolOutcome:
        return cls(ok=False, output="", error=error)


class ToolBox:
    """Bound tools for one attempt at one case."""

    def __init__(
        self,
        *,
        case: EvalCase,
        workspace: Workspace,
        recorder: TrajectoryRecorder,
        test_timeout_s: float | None = None,
    ) -> None:
        self.case = case
        self.workspace = workspace
        self.recorder = recorder
        self.test_timeout_s = test_timeout_s if test_timeout_s is not None else case.timeout_s
        self._tests_passed: bool | None = None
        self._submitted = False

    @property
    def tests_passed(self) -> bool | None:
        """Result of the most recent ``run_tests``; ``None`` if never run."""
        return self._tests_passed

    @property
    def submitted(self) -> bool:
        """Whether the agent has declared itself finished."""
        return self._submitted

    # -- tools -----------------------------------------------------------

    def list_files(self) -> ToolOutcome:
        call_id = self.recorder.tool_call(tool="list_files", args={})
        names = self.workspace.list_files()
        shown = names[:MAX_LISTED_FILES]
        output = "\n".join(shown)
        if len(names) > len(shown):
            output += f"\n… {len(names) - len(shown)} more files not shown"
        outcome = ToolOutcome(ok=True, output=output)
        self.recorder.tool_result(call_id=call_id, tool="list_files", ok=True, output=output)
        return outcome

    def read_file(self, path: str) -> ToolOutcome:
        call_id = self.recorder.tool_call(tool="read_file", args={"path": path})
        try:
            contents = self.workspace.read_file(path)
        except PathEscapeError as exc:
            return self._fail(call_id, "read_file", str(exc))
        except FileNotFoundError as exc:
            return self._fail(call_id, "read_file", str(exc))
        except OSError as exc:
            return self._fail(call_id, "read_file", f"could not read {path}: {exc}")

        self.recorder.tool_result(call_id=call_id, tool="read_file", ok=True, output=contents)
        return ToolOutcome(ok=True, output=contents)

    def write_file(self, path: str, contents: str) -> ToolOutcome:
        # The full contents go into the trace, not just their length. A write is
        # the one action whose payload *is* the action, and recording only its
        # size would make a trajectory impossible to replay faithfully.
        call_id = self.recorder.tool_call(
            tool="write_file", args={"path": path, "contents": contents}
        )
        try:
            record = self.workspace.write_file(path, contents)
        except PathEscapeError as exc:
            return self._fail(call_id, "write_file", str(exc))
        except OSError as exc:
            return self._fail(call_id, "write_file", f"could not write {path}: {exc}")

        self.recorder.file_edit(
            path=record.path,
            before_hash=record.before_hash,
            after_hash=record.after_hash,
            lines_added=record.lines_added,
            lines_removed=record.lines_removed,
            diff=record.diff,
        )
        summary = (
            f"wrote {path} (+{record.lines_added}/-{record.lines_removed}"
            f"{', created' if record.created else ''})"
        )
        self.recorder.tool_result(call_id=call_id, tool="write_file", ok=True, output=summary)
        return ToolOutcome(ok=True, output=summary)

    async def run_tests(self) -> ToolOutcome:
        call_id = self.recorder.tool_call(
            tool="run_tests", args={"command": " ".join(self.case.test_command)}
        )
        result = await self.workspace.run(self.case.test_command, timeout_s=self.test_timeout_s)
        self.recorder.command_run(
            argv=result.argv,
            exit_code=result.exit_code,
            duration_ms=result.duration_ms,
            timed_out=result.timed_out,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        self._tests_passed = result.ok

        output = result.stdout if result.stdout.strip() else result.stderr
        if result.timed_out:
            error = f"test command timed out after {self.test_timeout_s:g}s"
        elif not result.ok:
            error = f"test command exited with status {result.exit_code}"
        else:
            error = None

        self.recorder.tool_result(
            call_id=call_id, tool="run_tests", ok=result.ok, output=output, error=error
        )
        return ToolOutcome(ok=result.ok, output=output, error=error)

    def replace_text(self, path: str, old: str, new: str) -> ToolOutcome:
        """Replace one unique occurrence of ``old`` with ``new``.

        Exists because ``write_file`` asks a model to JSON-escape an entire
        source file, and Python files open with a triple-quoted docstring —
        the exact sequence models most reliably fail to escape. A targeted
        replacement keeps arguments short and usually free of quote runs.

        Refuses a non-unique match rather than guessing which one was meant: a
        silent wrong edit is far worse than an error the agent can recover from.
        """
        call_id = self.recorder.tool_call(
            tool="replace_text", args={"path": path, "old": old, "new": new}
        )
        try:
            contents = self.workspace.read_file(path)
        except PathEscapeError as exc:
            return self._fail(call_id, "replace_text", str(exc))
        except FileNotFoundError as exc:
            return self._fail(call_id, "replace_text", str(exc))
        except OSError as exc:
            return self._fail(call_id, "replace_text", f"could not read {path}: {exc}")

        if not old:
            return self._fail(call_id, "replace_text", "'old' must not be empty")

        occurrences = contents.count(old)
        if occurrences == 0:
            return self._fail(
                call_id, "replace_text", f"'old' does not appear in {path}; read the file again"
            )
        if occurrences > 1:
            return self._fail(
                call_id,
                "replace_text",
                f"'old' appears {occurrences} times in {path}; include more surrounding "
                "text so it matches exactly once",
            )

        try:
            record = self.workspace.write_file(path, contents.replace(old, new, 1))
        except (PathEscapeError, OSError) as exc:
            return self._fail(call_id, "replace_text", str(exc))

        self.recorder.file_edit(
            path=record.path,
            before_hash=record.before_hash,
            after_hash=record.after_hash,
            lines_added=record.lines_added,
            lines_removed=record.lines_removed,
            diff=record.diff,
        )
        summary = f"replaced 1 occurrence in {path} (+{record.lines_added}/-{record.lines_removed})"
        self.recorder.tool_result(call_id=call_id, tool="replace_text", ok=True, output=summary)
        return ToolOutcome(ok=True, output=summary)

    def submit(self, summary: str = "") -> ToolOutcome:
        call_id = self.recorder.tool_call(tool="submit", args={"summary": summary})
        self.recorder.tool_result(call_id=call_id, tool="submit", ok=True, output=summary)
        self.recorder.submission(summary=summary)
        self._submitted = True
        return ToolOutcome(ok=True, output=summary)

    # -- internals -------------------------------------------------------

    def _fail(self, call_id: str, tool: str, error: str) -> ToolOutcome:
        self.recorder.tool_result(call_id=call_id, tool=tool, ok=False, error=error)
        return ToolOutcome.failure(error)
