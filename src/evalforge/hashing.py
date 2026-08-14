"""Content hashing used to make datasets, suites and agent configs verifiable.

Every hash in EvalForge is a SHA-256 over *canonical* JSON: keys sorted, no
insignificant whitespace, UTF-8 without escaping. Two structurally equal objects
therefore hash identically regardless of construction order or process, which is
what lets a run record "dataset v3, suite v2" and have that claim be checkable.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

HASH_PREFIX = "sha256:"


def canonical_json(value: Any) -> str:
    """Serialise ``value`` to the canonical form used for all content hashes."""
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    """Return the prefixed SHA-256 content hash of a JSON-serialisable value."""
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{HASH_PREFIX}{digest}"


def text_hash(text: str) -> str:
    """Return the prefixed SHA-256 hash of raw text (file contents, prompts)."""
    return f"{HASH_PREFIX}{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def short(hash_value: str, length: int = 12) -> str:
    """Shorten a prefixed hash for display, keeping it recognisable."""
    return hash_value.removeprefix(HASH_PREFIX)[:length]
