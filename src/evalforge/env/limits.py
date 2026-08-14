"""Which resource limits this platform will actually enforce.

Sandbox guarantees should be reported, not assumed. Darwin accepts a CPU-time
and file-size cap but rejects lowering the address-space limit, so a memory cap
requested on macOS is silently ignored by the kernel. Probing once — in a
throwaway subprocess, so the probe cannot lower *our* limits — lets the CLI say
plainly what containment is in force and lets tests assert only what the
platform can deliver.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache

PROBE_TIMEOUT_S = 30.0

_PROBE_SOURCE = """
import json
from evalforge.env._launcher import apply_limits

applied = apply_limits(cpu_seconds=3600, memory_mb=4096, file_size_mb=1024)
print(json.dumps(list(applied)))
"""


@dataclass(frozen=True, slots=True)
class LimitSupport:
    """Which of the requested limits the running platform enforces."""

    cpu: bool
    memory: bool
    file_size: bool

    @property
    def all_supported(self) -> bool:
        return self.cpu and self.memory and self.file_size

    def unsupported(self) -> tuple[str, ...]:
        missing = []
        if not self.cpu:
            missing.append("cpu")
        if not self.memory:
            missing.append("memory")
        if not self.file_size:
            missing.append("file_size")
        return tuple(missing)

    def describe(self) -> str:
        if self.all_supported:
            return "cpu, memory and file-size limits enforced"
        missing = ", ".join(self.unsupported())
        return f"not enforced on this platform: {missing}"


@lru_cache(maxsize=1)
def probe_limit_support() -> LimitSupport:
    """Run the limit probe once and cache the answer for the process."""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE_SOURCE],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return LimitSupport(cpu=False, memory=False, file_size=False)

    if completed.returncode != 0:
        return LimitSupport(cpu=False, memory=False, file_size=False)

    try:
        applied = set(json.loads(completed.stdout))
    except (json.JSONDecodeError, TypeError):
        return LimitSupport(cpu=False, memory=False, file_size=False)

    return LimitSupport(
        cpu="cpu" in applied,
        memory="memory" in applied,
        file_size="file_size" in applied,
    )
