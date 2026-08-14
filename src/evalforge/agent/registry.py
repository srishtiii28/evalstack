"""Turning an agent reference string into an agent.

References look like ``scripted:baseline``. The scheme prefix names a family and
the remainder configures it, so the CLI, the store and a future config file all
speak the same identifier — and a run's ``agent_ref`` stays meaningful when read
back months later.
"""

from __future__ import annotations

from evalforge.agent.base import Agent
from evalforge.agent.scripted import POLICIES, SCRIPTED_PREFIX, scripted_agent


def resolve_agent(reference: str) -> Agent:
    """Build the agent named by ``reference``, e.g. ``scripted:baseline``."""
    scheme, separator, remainder = reference.partition(":")
    if not separator:
        raise ValueError(
            f"agent reference must look like 'family:variant', got {reference!r}; "
            f"known references: {', '.join(agent_names())}"
        )

    if scheme == SCRIPTED_PREFIX:
        return scripted_agent(remainder)

    raise ValueError(
        f"unknown agent family {scheme!r}; known references: {', '.join(agent_names())}"
    )


def agent_names() -> tuple[str, ...]:
    """Every agent reference that :func:`resolve_agent` accepts today."""
    return tuple(f"{SCRIPTED_PREFIX}:{name}" for name in sorted(POLICIES))
