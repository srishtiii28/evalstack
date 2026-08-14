"""The agent contract.

EvalForge does not care how an agent thinks. It cares that an agent acts inside
a workspace and that every action lands in the trajectory — which is why the
context handed to :meth:`Agent.run` bundles the three together.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import JsonValue

from evalforge.env.workspace import Workspace
from evalforge.hashing import content_hash
from evalforge.schema.case import EvalCase
from evalforge.trace import TrajectoryRecorder


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Everything an agent is given for one attempt."""

    case: EvalCase
    workspace: Workspace
    recorder: TrajectoryRecorder


class Agent(ABC):
    """A tool-using agent under evaluation."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier recorded on every run, e.g. ``scripted:baseline``."""

    def config(self) -> dict[str, JsonValue]:
        """Configuration that materially affects behaviour.

        This feeds the content hash recorded on every run, so two runs claiming
        the same agent can be shown to have used the same configuration.
        """
        return {}

    @property
    def config_hash(self) -> str:
        return content_hash({"name": self.name, "config": self.config()})

    @abstractmethod
    async def run(self, context: AgentContext) -> None:
        """Attempt the task, recording every action into the trajectory.

        Implementations should not raise for ordinary failure — an agent that
        cannot solve a task has produced a *result*, not an error. Raising is
        reserved for genuine infrastructure faults.
        """
