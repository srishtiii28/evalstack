"""Apply POSIX resource limits to a child process, then exec it.

Run as ``python -m evalforge.env._launcher --cpu-seconds N -- <argv...>``.

This exists instead of ``subprocess(preexec_fn=...)`` because ``preexec_fn`` is
documented as unsafe in the presence of threads, and the scheduler is
thread-and-async heavy. Setting the limits inside a short-lived launcher that
then ``execvp``s the real command achieves the same containment with no such
hazard, at the cost of one interpreter startup per command.

Limits are best-effort *by platform, not by accident*: Darwin rejects lowering
``RLIMIT_AS``/``RLIMIT_DATA`` outright, so a memory cap silently cannot be
applied there. Unsupported limits are skipped rather than fatal, and
:mod:`evalforge.env.limits` probes which ones actually work so nothing in the
system claims enforcement it does not have.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

BYTES_PER_MB = 1024 * 1024
EXIT_LAUNCH_FAILED = 127


def _try_setrlimit(resource_id: int, soft: int, hard: int) -> bool:
    """Set one limit, returning whether the platform accepted it."""
    import resource

    try:
        resource.setrlimit(resource_id, (soft, hard))
    except (ValueError, OSError):
        return False
    return True


def apply_limits(
    *,
    cpu_seconds: int | None = None,
    memory_mb: int | None = None,
    file_size_mb: int | None = None,
) -> tuple[str, ...]:
    """Apply the requested limits; return the names that were actually applied."""
    try:
        import resource
    except ImportError:  # pragma: no cover - POSIX-only in practice
        return ()

    applied: list[str] = []

    # Soft limit raises SIGXCPU; the hard limit a second later escalates to
    # SIGKILL, giving a runaway process one chance to exit cleanly first.
    if cpu_seconds is not None and _try_setrlimit(
        resource.RLIMIT_CPU, cpu_seconds, cpu_seconds + 1
    ):
        applied.append("cpu")

    if memory_mb is not None:
        limit = memory_mb * BYTES_PER_MB
        # Darwin rejects both; Linux honours RLIMIT_AS. Try the stronger one first.
        for name in ("RLIMIT_AS", "RLIMIT_DATA"):
            resource_id = getattr(resource, name, None)
            if resource_id is not None and _try_setrlimit(resource_id, limit, limit):
                applied.append("memory")
                break

    if file_size_mb is not None:
        limit = file_size_mb * BYTES_PER_MB
        if _try_setrlimit(resource.RLIMIT_FSIZE, limit, limit):
            applied.append("file_size")

    return tuple(applied)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evalforge-launcher", add_help=False)
    parser.add_argument("--cpu-seconds", type=int, default=None)
    parser.add_argument("--memory-mb", type=int, default=None)
    parser.add_argument("--file-size-mb", type=int, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("evalforge-launcher: no command given", file=sys.stderr)
        return EXIT_LAUNCH_FAILED

    apply_limits(
        cpu_seconds=args.cpu_seconds,
        memory_mb=args.memory_mb,
        file_size_mb=args.file_size_mb,
    )

    try:
        os.execvp(command[0], command)
    except OSError as exc:
        print(f"evalforge-launcher: cannot execute {command[0]!r}: {exc}", file=sys.stderr)
        return EXIT_LAUNCH_FAILED
    raise AssertionError("unreachable: execvp replaces the process on success")


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main())
