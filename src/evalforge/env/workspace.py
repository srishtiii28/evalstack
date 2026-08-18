"""The sandbox an agent acts inside: a temp directory, a shell, and a boundary.

Containment is enforced here and nowhere else. Every path an agent supplies is
resolved through :meth:`Workspace.resolve`, which follows symlinks and rejects
anything landing outside the workspace root. Rejections are *recorded* as
violations and raised as :class:`PathEscapeError` — so the tool layer can hand
the agent an ordinary error message while the safety evaluator still sees a
hard signal.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import os
import shutil
import signal
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from evalforge.hashing import bytes_hash, text_hash
from evalforge.schema.case import EvalCase

#: Generated artefacts that must never count as agent edits.
IGNORED_DIRECTORY_NAMES = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".git"})
IGNORED_SUFFIXES = (".pyc", ".pyo")

#: Interpreter names in dataset commands that map to the running interpreter, so
#: datasets stay portable across machines and virtualenvs.
_INTERPRETER_ALIASES = frozenset({"python", "python3"})


class WorkspaceError(Exception):
    """Base class for workspace failures."""


class PathEscapeError(WorkspaceError):
    """An operation tried to touch a path outside the workspace root."""


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Limits applied to every child process.

    ``memory_mb`` is deliberately generous: an interpreter reserves a large
    address space at startup, and a limit tight enough to be interesting is also
    tight enough to make Python fail to boot, which would look like a harness bug.
    """

    cpu_seconds: int | None = None
    memory_mb: int | None = 4096
    file_size_mb: int | None = 64


@dataclass(frozen=True, slots=True)
class Violation:
    """A contained policy breach."""

    rule: str
    detail: str
    attempted: str


@dataclass(frozen=True, slots=True)
class CommandResult:
    argv: tuple[str, ...]
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@dataclass(frozen=True, slots=True)
class FileEditRecord:
    """What changed when an agent wrote a file."""

    path: str
    before_hash: str | None
    after_hash: str
    lines_added: int
    lines_removed: int
    #: Unified diff of the change. Hashes prove *that* something changed; only
    #: this shows *what*, and it is captured here because this is the one moment
    #: both versions of the file exist.
    diff: str = ""

    @property
    def created(self) -> bool:
        return self.before_hash is None

    @property
    def unchanged(self) -> bool:
        return self.before_hash == self.after_hash


@dataclass(frozen=True, slots=True)
class FileDiff:
    added: tuple[str, ...] = ()
    modified: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()

    @property
    def touched(self) -> tuple[str, ...]:
        return tuple(sorted({*self.added, *self.modified, *self.deleted}))

    @property
    def is_empty(self) -> bool:
        return not self.touched


def _unified_diff(before: str, after: str, path: str) -> str:
    """Render a change as a unified diff, for a human reading the trajectory.

    Kept separate from :func:`_count_line_changes` rather than sharing one
    difflib pass: the counts are computed with no context lines, and quietly
    changing that to suit a display concern would move a number other people's
    thresholds are set against.
    """
    lines = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(lines)


def _count_line_changes(before: str, after: str) -> tuple[int, int]:
    """Return ``(added, removed)`` line counts between two file versions."""
    added = removed = 0
    diff = difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), n=0
    )
    for line in diff:
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


def _is_ignored(relative: Path) -> bool:
    if any(part in IGNORED_DIRECTORY_NAMES for part in relative.parts):
        return True
    return relative.suffix in IGNORED_SUFFIXES


@dataclass(slots=True)
class Workspace:
    """A contained directory an agent may read, write and run commands in."""

    root: Path
    home: Path
    tmp: Path
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    on_violation: Callable[[Violation], None] | None = None
    _violations: list[Violation] = field(default_factory=list, init=False)
    _baseline: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.root = Path(os.path.realpath(self.root))
        self.home = Path(os.path.realpath(self.home))
        self.tmp = Path(os.path.realpath(self.tmp))

    # -- violations ------------------------------------------------------

    @property
    def violations(self) -> tuple[Violation, ...]:
        return tuple(self._violations)

    def _record_violation(self, rule: str, detail: str, attempted: str) -> Violation:
        violation = Violation(rule=rule, detail=detail, attempted=attempted)
        self._violations.append(violation)
        if self.on_violation is not None:
            self.on_violation(violation)
        return violation

    # -- paths -----------------------------------------------------------

    def resolve(self, relative: str) -> Path:
        """Resolve an agent-supplied path, or refuse it.

        Symlinks are followed before the containment check, so a link planted
        inside the workspace cannot be used as an exit.
        """
        candidate = self.root / relative
        real = Path(os.path.realpath(candidate))
        if real != self.root and self.root not in real.parents:
            self._record_violation(
                rule="path_escape",
                detail=f"resolved outside workspace root to {real}",
                attempted=relative,
            )
            raise PathEscapeError(f"path {relative!r} resolves outside the workspace")
        return real

    def exists(self, relative: str) -> bool:
        return self.resolve(relative).is_file()

    def list_files(self) -> tuple[str, ...]:
        """Every tracked file, as workspace-relative posix paths."""
        found: list[str] = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.root)
            if _is_ignored(relative):
                continue
            found.append(relative.as_posix())
        return tuple(sorted(found))

    def read_file(self, relative: str) -> str:
        path = self.resolve(relative)
        if not path.is_file():
            raise FileNotFoundError(f"no such file in workspace: {relative}")
        return path.read_text(encoding="utf-8")

    def write_file(self, relative: str, contents: str) -> FileEditRecord:
        path = self.resolve(relative)
        before: str | None = None
        if path.is_file():
            before = path.read_text(encoding="utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        added, removed = _count_line_changes(before or "", contents)
        return FileEditRecord(
            path=relative,
            before_hash=None if before is None else text_hash(before),
            after_hash=text_hash(contents),
            lines_added=added,
            lines_removed=removed,
            diff=_unified_diff(before or "", contents, relative),
        )

    def delete_file(self, relative: str) -> bool:
        path = self.resolve(relative)
        if not path.is_file():
            return False
        path.unlink()
        return True

    # -- change tracking -------------------------------------------------

    def snapshot(self) -> dict[str, str]:
        """Hash every tracked file, for diffing against a later state.

        Hashes raw bytes rather than decoded text: a test run can leave a
        coverage database or another binary artefact in the tree, and a diff
        that crashes on undecodable bytes would take the whole run down with it.
        """
        return {name: bytes_hash(self.resolve(name).read_bytes()) for name in self.list_files()}

    def set_baseline(self) -> None:
        """Freeze the current contents as the reference for :meth:`diff`."""
        self._baseline = self.snapshot()

    def diff(self) -> FileDiff:
        """Changes made since :meth:`set_baseline`."""
        current = self.snapshot()
        baseline = self._baseline
        added = tuple(sorted(set(current) - set(baseline)))
        deleted = tuple(sorted(set(baseline) - set(current)))
        modified = tuple(
            sorted(name for name in set(current) & set(baseline) if current[name] != baseline[name])
        )
        return FileDiff(added=added, modified=modified, deleted=deleted)

    # -- execution -------------------------------------------------------

    def _child_env(self) -> dict[str, str]:
        """A deliberately small environment: nothing inherited that need not be."""
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.home),
            "TMPDIR": str(self.tmp),
            "PYTHONPATH": str(self.root),
            # Keep the tree free of build artefacts so a diff reflects agent edits only.
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
        }

    def _launch_argv(self, argv: tuple[str, ...], timeout_s: float) -> list[str]:
        command = list(argv)
        if command[0] in _INTERPRETER_ALIASES:
            command[0] = sys.executable

        cpu_seconds = self.limits.cpu_seconds
        if cpu_seconds is None:
            # CPU time is a backstop behind the wall-clock timeout: a process that
            # burns CPU continuously should die at roughly the same moment.
            cpu_seconds = int(timeout_s) + 5

        launcher = [
            sys.executable,
            "-m",
            "evalforge.env._launcher",
            "--cpu-seconds",
            str(cpu_seconds),
        ]
        if self.limits.memory_mb is not None:
            launcher += ["--memory-mb", str(self.limits.memory_mb)]
        if self.limits.file_size_mb is not None:
            launcher += ["--file-size-mb", str(self.limits.file_size_mb)]
        return [*launcher, "--", *command]

    async def run(self, argv: tuple[str, ...], *, timeout_s: float) -> CommandResult:
        """Run a command inside the workspace, killing the whole group on timeout."""
        if not argv:
            raise ValueError("argv must not be empty")

        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *self._launch_argv(argv, timeout_s),
            cwd=self.root,
            env=self._child_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # Its own session, so a timeout can reap grandchildren too.
            start_new_session=True,
        )

        timed_out = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout_s
            )
        except TimeoutError:
            timed_out = True
            _kill_process_group(process.pid)
            stdout_bytes, stderr_bytes = await process.communicate()

        duration_ms = (time.monotonic() - started) * 1000.0
        return CommandResult(
            argv=argv,
            exit_code=None if timed_out else process.returncode,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            duration_ms=duration_ms,
            timed_out=timed_out,
        )


def _kill_process_group(pid: int) -> None:
    """Best-effort SIGKILL of a child's whole process group."""
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(os.getpgid(pid), signal.SIGKILL)


@contextlib.contextmanager
def workspace_for(
    case: EvalCase,
    *,
    base_dir: Path | None = None,
    limits: ResourceLimits | None = None,
    on_violation: Callable[[Violation], None] | None = None,
) -> Iterator[Workspace]:
    """Materialise a case into a fresh workspace, and clean it up afterwards."""
    base = Path(tempfile.mkdtemp(prefix="evalforge-", dir=base_dir))
    try:
        root = base / "repo"
        home = base / "home"
        tmp = base / "tmp"
        for directory in (root, home, tmp):
            directory.mkdir(parents=True)

        workspace = Workspace(
            root=root,
            home=home,
            tmp=tmp,
            limits=limits or ResourceLimits(),
            on_violation=on_violation,
        )
        for spec in case.files:
            workspace.write_file(spec.path, spec.contents)
        workspace.set_baseline()
        yield workspace
    finally:
        shutil.rmtree(base, ignore_errors=True)
