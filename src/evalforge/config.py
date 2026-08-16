"""Loading local configuration from a ``.env`` file.

Secrets belong in a git-ignored file rather than a shell history or a flag, and
a project that needs one key should not require the person running it to
remember an ``export`` incantation.

Two rules make the behaviour predictable:

* **The real environment wins.** A variable already set in the process is never
  overwritten by the file. Otherwise ``GROQ_API_KEY=... evalforge run`` would be
  silently ignored in favour of a stale file, which is a genuinely nasty
  half-hour of debugging.
* **A missing file is not an error.** CI sets real environment variables and has
  no ``.env``; that is the normal case, not a misconfiguration.

Written against the stdlib rather than taking a dependency: the format this
project needs is one key per line, and the parser is short enough to test
exhaustively.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path

from evalforge.paths import DEFAULT_ENV_FILE

_EXPORT_PREFIX = "export "
_QUOTES = ("'", '"')


def _strip_inline_comment(value: str) -> str:
    """Drop a trailing ``# comment`` from an unquoted value."""
    marker = value.find(" #")
    return value if marker == -1 else value[:marker]


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in _QUOTES:
        return value[1:-1]
    return _strip_inline_comment(value).strip()


def parse_env_file(text: str) -> dict[str, str]:
    """Parse ``.env`` contents into a mapping.

    Supports comments, blank lines, an optional ``export`` prefix, and single or
    double quoted values. A quoted value is taken literally — including any
    ``#`` inside it — while an unquoted one has trailing comments removed.
    """
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(_EXPORT_PREFIX):
            line = line[len(_EXPORT_PREFIX) :].lstrip()

        key, separator, raw_value = line.partition("=")
        if not separator:
            # A line without '=' is a typo, not a directive. Skipping it beats
            # guessing at what was meant.
            continue

        name = key.strip()
        if not name:
            continue
        values[name] = _unquote(raw_value.strip())
    return values


def load_env_file(
    path: Path | None = None,
    *,
    environ: MutableMapping[str, str] | None = None,
    override: bool = False,
) -> tuple[str, ...]:
    """Load ``path`` into the environment; return the names actually applied."""
    target = path if path is not None else DEFAULT_ENV_FILE
    destination = environ if environ is not None else os.environ

    if not target.is_file():
        return ()

    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # An unreadable or non-UTF-8 .env must not stop a run that may not need
        # it. This loader runs before *every* command, so letting it raise would
        # take down `evalforge --help` too.
        return ()

    applied: list[str] = []
    for name, value in parse_env_file(text).items():
        if not override and destination.get(name):
            continue
        destination[name] = value
        applied.append(name)
    return tuple(applied)
