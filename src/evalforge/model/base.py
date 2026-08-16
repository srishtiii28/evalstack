"""The provider-agnostic model interface.

EvalForge never talks to a vendor SDK directly. Everything above this layer sees
:class:`ModelClient`, which means swapping providers is a configuration change
and the cost, caching, budgeting and rate-limiting wrappers are written once
rather than once per vendor.

The message shape is the widely-implemented chat-completions one — roles, tool
calls on the assistant turn, a tool turn carrying each result. That is a
deliberate choice of lowest common denominator: it maps onto Groq, OpenRouter,
GitHub Models, Gemini's compatibility endpoint and a local Ollama without
per-provider special casing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import JsonValue

from evalforge.hashing import content_hash

Role = Literal["system", "user", "assistant", "tool"]

#: Why the model stopped. Normalised across providers, which disagree on names.
StopReason = Literal["end_turn", "tool_use", "max_tokens", "stop_sequence", "other"]


class ModelError(Exception):
    """A model call failed in a way the caller must handle."""


class TransientModelError(ModelError):
    """A fault worth retrying: rate limiting, a 5xx, a dropped connection."""


class PermanentModelError(ModelError):
    """A fault retrying cannot fix: bad credentials, an unknown model, a bad request."""


class ModelBehaviourError(ModelError):
    """The request was valid; the model's own generation was unusable.

    Providers that parse tool calls server-side reject a malformed one with a
    4xx, which looks like a client error but is not: nothing about the request
    was wrong. This is the model failing at the task, so it belongs in the
    trajectory as a measurement rather than being counted as an outage — the
    opposite mistake to letting a provider fault masquerade as a capability
    regression, and just as damaging, because excluding the case inflates the
    success rate.
    """


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool as described *to* the model."""

    name: str
    description: str
    parameters: dict[str, JsonValue]

    def describe(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """A tool call the model asked for."""

    id: str
    name: str
    arguments: dict[str, JsonValue] = field(default_factory=dict)
    #: Set when the model emitted arguments that were not valid JSON. Kept as a
    #: value rather than an exception because it is the agent's problem to
    #: recover from, not the transport's to hide.
    malformed_arguments: str | None = None

    def describe(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": dict(self.arguments),
            "malformed_arguments": self.malformed_arguments,
        }


@dataclass(frozen=True, slots=True)
class Message:
    """One turn of the conversation."""

    role: Role
    content: str = ""
    tool_calls: tuple[ToolInvocation, ...] = ()
    tool_call_id: str | None = None

    def describe(self) -> dict[str, JsonValue]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [call.describe() for call in self.tool_calls],
            "tool_call_id": self.tool_call_id,
        }


@dataclass(frozen=True, slots=True)
class Usage:
    """Tokens consumed by one call."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """What a model returned, normalised."""

    model: str
    text: str = ""
    tool_calls: tuple[ToolInvocation, ...] = ()
    usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason = "end_turn"
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    #: True when served from the on-disk cache: the tokens were spent once,
    #: earlier, so this call's incremental cost is zero.
    cached: bool = False

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """A fully-specified call, and the thing the cache is keyed on."""

    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    temperature: float = 0.0
    max_tokens: int = 2048

    def cache_key(self, model: str, scope: str = "") -> str:
        """Content hash covering everything that can change the response.

        ``scope`` identifies the endpoint. Two providers commonly serve the same
        model id — an open-weights model on Groq and on OpenRouter, say — and
        without it one provider's cached answers would be served for the other.
        """
        return content_hash(
            {
                "scope": scope,
                "model": model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": [message.describe() for message in self.messages],
                "tools": [tool.describe() for tool in self.tools],
            }
        )

    def estimated_input_tokens(self) -> int:
        """A cheap character-based estimate, used only for pre-flight budgeting.

        Deliberately not a real tokeniser: pulling one in per provider would add
        a dependency and still be approximate. The budget guard compensates by
        assuming the worst case on output, so the estimate erring low cannot
        cause an overspend.
        """
        characters = sum(
            len(message.content) + sum(len(call.name) + 32 for call in message.tool_calls)
            for message in self.messages
        )
        characters += sum(len(tool.description) + len(str(tool.parameters)) for tool in self.tools)
        return max(1, characters // 4)


class ModelClient(Protocol):
    """Anything that can answer a :class:`ModelRequest`.

    Wrappers in this package — caching, budgeting, rate limiting — implement it
    too, so they compose in any order the caller finds useful.
    """

    @property
    def model(self) -> str: ...

    async def complete(self, request: ModelRequest) -> ModelResponse: ...


def user(content: str) -> Message:
    return Message(role="user", content=content)


def system(content: str) -> Message:
    return Message(role="system", content=content)


def assistant(content: str = "", tool_calls: Sequence[ToolInvocation] = ()) -> Message:
    return Message(role="assistant", content=content, tool_calls=tuple(tool_calls))


def tool_result(call_id: str, content: str) -> Message:
    return Message(role="tool", content=content, tool_call_id=call_id)
