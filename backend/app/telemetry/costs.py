from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from collections.abc import Iterable

from backend.app.models import InternalChatCompletion, ModelDefinition

_MILLION = Decimal("1000000")
_MONEY_PRECISION = Decimal("0.00000001")


@dataclass(frozen=True)
class CostAccounting:
    actual_cost: Decimal
    estimated_baseline_cost: Decimal
    estimated_cost_saved: Decimal
    savings_percentage: float


class CostCalculator:
    """Catalog-price accounting; baseline figures are estimates, never incurred spend."""

    def __init__(self, baseline_model: ModelDefinition) -> None:
        self._baseline_model = baseline_model

    @staticmethod
    def calculate_cost(model: ModelDefinition, input_tokens: int, output_tokens: int) -> Decimal:
        cost = (
            Decimal(input_tokens) * Decimal(str(model.input_cost_per_million_tokens))
            + Decimal(output_tokens) * Decimal(str(model.output_cost_per_million_tokens))
        ) / _MILLION
        return cost.quantize(_MONEY_PRECISION, rounding=ROUND_HALF_UP)

    def account_for(
        self, executions: Iterable[tuple[ModelDefinition, InternalChatCompletion]]
    ) -> CostAccounting:
        actual = Decimal("0")
        baseline = Decimal("0")
        for model, completion in executions:
            input_tokens = completion.usage.input_tokens
            output_tokens = completion.usage.output_tokens
            actual += self.calculate_cost(model, input_tokens, output_tokens)
            baseline += self.calculate_cost(self._baseline_model, input_tokens, output_tokens)
        actual = actual.quantize(_MONEY_PRECISION, rounding=ROUND_HALF_UP)
        baseline = baseline.quantize(_MONEY_PRECISION, rounding=ROUND_HALF_UP)
        saved = max(Decimal("0"), baseline - actual).quantize(_MONEY_PRECISION)
        savings_percentage = float((saved / baseline * 100).quantize(Decimal("0.01"))) if baseline else 0.0
        return CostAccounting(actual, baseline, saved, savings_percentage)
