"""What a model call costs.

Rates change and differ per provider, so they are configuration rather than
constants compiled into the code. One rule keeps the numbers honest: a model
with no known rate reports ``tracked=False`` rather than $0.00. A confident zero
is worse than an admitted unknown — it would quietly make a cost comparison
between two agent versions meaningless.

Groq's free tier really is zero, so that is the default here and it is a fact
rather than a placeholder. If you move to a billed account, supply real rates
with ``EVALFORGE_MODEL_PRICING``, a JSON object of
``{"model-id": {"input": <usd per million>, "output": <usd per million>}}``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from evalforge.model.base import Usage

PRICING_ENV_VAR = "EVALFORGE_MODEL_PRICING"
TOKENS_PER_MILLION = 1_000_000


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Cost per million tokens, and whether the rate is actually known."""

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    tracked: bool = True

    def cost_for(self, usage: Usage) -> float:
        if not self.tracked:
            return 0.0
        return (
            usage.input_tokens * self.input_per_mtok + usage.output_tokens * self.output_per_mtok
        ) / TOKENS_PER_MILLION


#: No charge at the API boundary — a free tier, or a locally-served model.
FREE = ModelPricing(0.0, 0.0, tracked=True)

#: A billed model whose rate has not been configured.
UNKNOWN = ModelPricing(0.0, 0.0, tracked=False)


def _rates_from_environment() -> dict[str, ModelPricing]:
    raw = os.environ.get(PRICING_ENV_VAR)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{PRICING_ENV_VAR} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{PRICING_ENV_VAR} must be a JSON object mapping model ids to rates")

    table: dict[str, ModelPricing] = {}
    for model, rates in parsed.items():
        if not isinstance(rates, dict):
            raise ValueError(f"{PRICING_ENV_VAR}: rates for {model!r} must be an object")
        try:
            table[model] = ModelPricing(float(rates["input"]), float(rates["output"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{PRICING_ENV_VAR}: rates for {model!r} need numeric 'input' and 'output'"
            ) from exc
    return table


class PricingTable:
    """Resolves a model id to its rate."""

    def __init__(
        self, rates: dict[str, ModelPricing] | None = None, *, default: ModelPricing = FREE
    ) -> None:
        self._rates: dict[str, ModelPricing] = {}
        self._rates.update(_rates_from_environment())
        if rates:
            self._rates.update(rates)
        self._default = default

    def for_model(self, model: str) -> ModelPricing:
        return self._rates.get(model, self._default)

    def cost_for(self, model: str, usage: Usage) -> float:
        return self.for_model(model).cost_for(usage)

    def tracks(self, model: str) -> bool:
        return self.for_model(model).tracked
