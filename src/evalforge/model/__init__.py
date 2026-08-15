"""Provider-agnostic model access, with caching, budgeting and rate limiting."""

from evalforge.model.base import (
    Message,
    ModelClient,
    ModelError,
    ModelRequest,
    ModelResponse,
    PermanentModelError,
    StopReason,
    ToolInvocation,
    ToolSpec,
    TransientModelError,
    Usage,
    assistant,
    system,
    tool_result,
    user,
)
from evalforge.model.budget import (
    BudgetedModelClient,
    BudgetExceeded,
    BudgetGuard,
    BudgetLimits,
)
from evalforge.model.cache import CachingModelClient, ResponseCache
from evalforge.model.chat_completions import ChatCompletionsClient
from evalforge.model.pricing import FREE, UNKNOWN, ModelPricing, PricingTable
from evalforge.model.providers import (
    GROQ,
    PROVIDERS,
    ProviderConfig,
    ProviderNotConfiguredError,
    build_model_client,
    provider_names,
    resolve_provider,
)
from evalforge.model.rate_limit import RateLimiter, RateLimits

__all__ = [
    "FREE",
    "GROQ",
    "PROVIDERS",
    "UNKNOWN",
    "BudgetExceeded",
    "BudgetGuard",
    "BudgetLimits",
    "BudgetedModelClient",
    "CachingModelClient",
    "ChatCompletionsClient",
    "Message",
    "ModelClient",
    "ModelError",
    "ModelPricing",
    "ModelRequest",
    "ModelResponse",
    "PermanentModelError",
    "PricingTable",
    "ProviderConfig",
    "ProviderNotConfiguredError",
    "RateLimiter",
    "RateLimits",
    "ResponseCache",
    "StopReason",
    "ToolInvocation",
    "ToolSpec",
    "TransientModelError",
    "Usage",
    "assistant",
    "build_model_client",
    "provider_names",
    "resolve_provider",
    "system",
    "tool_result",
    "user",
]
