"""A model-driven agent: the tool-use loop that produces real trajectories.

The loop is deliberately plain — call the model, execute whatever tools it asked
for, feed the results back, repeat until it submits or runs out of steps. What
takes care is the error taxonomy, because the harness above needs to tell three
different things apart:

* The agent behaved badly (bad arguments, wrong tool, never submitted). That is
  a *measurement*: it goes into the trajectory and the attempt continues.
* The run hit a ceiling (steps, tokens, budget). Also a measurement — the
  attempt ends and is scored on what it managed.
* The provider broke (5xx, timeout, bad credentials). That is *infrastructure*:
  it propagates so the scheduler can retry or fail the case, and never gets
  recorded as though the agent failed the task.

Collapsing the third into the first is the easy mistake, and it silently turns
provider downtime into a capability regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import JsonValue

from evalforge.agent.base import Agent, AgentContext
from evalforge.agent.tools import WHOLE_FILE_SURFACE, ToolBox, ToolOutcome, tool_specs
from evalforge.model.base import (
    Message,
    ModelBehaviourError,
    ModelClient,
    ModelRequest,
    PermanentModelError,
    ToolInvocation,
    TransientModelError,
    assistant,
    system,
    tool_result,
    user,
)
from evalforge.model.budget import BudgetExceeded
from evalforge.orchestrator.scheduler import FatalInfrastructureError, InfrastructureError

DEFAULT_MAX_STEPS = 12
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0

DEFAULT_SYSTEM_PROMPT = """\
You are a software engineer fixing a bug in a small Python repository.

Work through the tools provided. A typical approach:
  1. list_files to see what is there
  2. read_file on the file the task points at
  3. write_file with the corrected contents
  4. run_tests to check your work
  5. submit once the tests pass

Rules:
- write_file replaces the whole file. Read a file before writing it, and send
  the complete corrected contents — never a diff, a fragment, or a placeholder.
- Make the smallest change that fixes the reported problem.
- Do not modify the tests. They define what correct means.
- Always run_tests before you submit.
- The attempt does not end until you call submit.\
"""

_NUDGE = (
    "You did not call a tool. Use the tools to inspect and fix the repository, "
    "then call submit when the tests pass."
)


@dataclass(frozen=True, slots=True)
class ModelAgentConfig:
    """Everything that changes how the agent behaves, and therefore its hash."""

    max_steps: int = DEFAULT_MAX_STEPS
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    tool_surface: str = WHOLE_FILE_SURFACE

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

    def describe(self) -> dict[str, JsonValue]:
        return {
            "max_steps": self.max_steps,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            # The prompt is part of the configuration: changing it changes the
            # agent, and the run hash should say so.
            "system_prompt": self.system_prompt,
            # So are the tools. Which tools an agent has changes what it can do
            # at least as much as its prompt does, and two runs claiming the same
            # agent while holding different tools would be a false comparison.
            "tool_surface": self.tool_surface,
            "tools": [spec.name for spec in tool_specs(self.tool_surface)],
        }


@dataclass(slots=True)
class ModelAgent(Agent):
    """Drives a model through the shared tool surface."""

    client: ModelClient
    settings: ModelAgentConfig = field(default_factory=ModelAgentConfig)

    @property
    def name(self) -> str:
        return f"model:{self.client.model}"

    def config(self) -> dict[str, JsonValue]:
        return {"model": self.client.model, **self.settings.describe()}

    async def run(self, context: AgentContext) -> None:
        case = context.case
        recorder = context.recorder
        tools = ToolBox(case=case, workspace=context.workspace, recorder=recorder)
        specs = tool_specs(self.settings.tool_surface)

        recorder.task_started(prompt_hash=case.content_hash)

        messages: list[Message] = [
            system(self.settings.system_prompt),
            user(case.prompt),
        ]

        for _step in range(self.settings.max_steps):
            request = ModelRequest(
                messages=tuple(messages),
                tools=specs,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            )

            try:
                response = await self.client.complete(request)
            except BudgetExceeded as exc:
                # A ceiling is a fact about the attempt, so it is scored, not raised.
                recorder.agent_error(error_type="BudgetExceeded", message=str(exc))
                return
            except ModelBehaviourError as exc:
                # The model could not produce a usable tool call. That is the
                # agent failing the task, so the attempt is scored on what it
                # managed rather than discarded as an outage.
                recorder.agent_error(error_type="ModelBehaviourError", message=str(exc))
                return
            except PermanentModelError as exc:
                raise FatalInfrastructureError(f"model rejected the request: {exc}") from exc
            except TransientModelError as exc:
                raise InfrastructureError(f"model call failed: {exc}") from exc

            recorder.model_call(
                model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cost_usd=response.cost_usd,
                latency_ms=response.latency_ms,
                cached=response.cached,
                stop_reason=response.stop_reason,
            )

            messages.append(assistant(response.text, response.tool_calls))

            if not response.tool_calls:
                # Answering in prose is a behaviour worth recording, not a crash.
                messages.append(user(_NUDGE))
                continue

            for call in response.tool_calls:
                messages.append(tool_result(call.id, await self._invoke(tools, call)))

            if tools.submitted:
                return

        recorder.agent_error(
            error_type="StepLimitReached",
            message=f"stopped after {self.settings.max_steps} steps without submitting",
        )

    async def _invoke(self, tools: ToolBox, call: ToolInvocation) -> str:
        """Execute one tool call and return what the model should see."""
        if call.malformed_arguments is not None:
            return (
                "error: arguments were not valid JSON. Send a JSON object matching "
                f"the tool's schema. Received: {call.malformed_arguments[:200]}"
            )

        match call.name:
            case "list_files":
                return _render(tools.list_files())
            case "read_file":
                path = _string_argument(call, "path")
                if path is None:
                    return "error: read_file needs a string 'path' argument"
                return _render(tools.read_file(path))
            case "write_file":
                path = _string_argument(call, "path")
                contents = _string_argument(call, "contents")
                if path is None or contents is None:
                    return "error: write_file needs string 'path' and 'contents' arguments"
                return _render(tools.write_file(path, contents))
            case "replace_text":
                path = _string_argument(call, "path")
                old = _string_argument(call, "old")
                new = _string_argument(call, "new")
                if path is None or old is None or new is None:
                    return "error: replace_text needs string 'path', 'old' and 'new' arguments"
                return _render(tools.replace_text(path, old, new))
            case "run_tests":
                return _render(await tools.run_tests())
            case "submit":
                summary = _string_argument(call, "summary") or ""
                return _render(tools.submit(summary))
            case _:
                known = ", ".join(spec.name for spec in tool_specs(self.settings.tool_surface))
                return f"error: no tool named {call.name!r}. Available tools: {known}"


def _string_argument(call: ToolInvocation, name: str) -> str | None:
    value = call.arguments.get(name)
    return value if isinstance(value, str) else None


def _render(outcome: ToolOutcome) -> str:
    if outcome.ok:
        return outcome.output or "ok"
    return f"error: {outcome.error}"
