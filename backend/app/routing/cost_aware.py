import time
from collections.abc import Sequence

from backend.app.models import CandidateModel, ModelDefinition, RejectedModel, RequestProfile, RoutingDecision
from backend.app.routing.base import RoutingEngine


class CostAwareRoutingEngine(RoutingEngine):
    """Select the least expensive model meeting deterministic capability constraints."""

    async def route(self, profile: RequestProfile, candidates: Sequence[ModelDefinition]) -> RoutingDecision:
        started = time.perf_counter()
        accepted: list[CandidateModel] = []
        rejected: list[RejectedModel] = []
        for model in candidates:
            rejection = self._hard_rejection(model, profile)
            if rejection:
                rejected.append(RejectedModel(model_id=model.id, tier=model.tier, reason=rejection))
                continue
            quality = self._predicted_quality(model, profile)
            if quality < profile.quality_target:
                rejected.append(RejectedModel(model_id=model.id, tier=model.tier, reason=(f"predicted quality {quality:.2f} is below target {profile.quality_target:.2f}")))
                continue
            cost = self._estimated_cost(model, profile)
            accepted.append(CandidateModel(model_id=model.id, tier=model.tier, provider=model.provider, model_name=model.model_name, predicted_quality=quality, estimated_cost=cost, estimated_latency=model.expected_latency_ms, fitness_score=self._fitness(quality, cost, model.expected_latency_ms, profile)))

        if not accepted:
            enabled = [model for model in candidates if model.enabled]
            if not enabled:
                raise ValueError("No enabled models are available for routing")
            fallback = max(enabled, key=lambda model: self._predicted_quality(model, profile))
            quality = self._predicted_quality(fallback, profile)
            cost = self._estimated_cost(fallback, profile)
            accepted.append(CandidateModel(model_id=fallback.id, tier=fallback.tier, provider=fallback.provider, model_name=fallback.model_name, predicted_quality=quality, estimated_cost=cost, estimated_latency=fallback.expected_latency_ms, fitness_score=self._fitness(quality, cost, fallback.expected_latency_ms, profile)))
            reason_prefix = "No model met every target; selected the highest-quality enabled fallback. "
        else:
            reason_prefix = ""

        accepted.sort(key=lambda item: (item.estimated_cost, item.estimated_latency, -item.predicted_quality))
        selected_candidate = accepted[0]
        selected_model = next(model for model in candidates if model.id == selected_candidate.model_id)
        routing_reason = (f"{reason_prefix}Selected {selected_model.tier} because it is the lowest-cost enabled model predicted to satisfy reasoning={profile.reasoning_requirement:.2f}, general={profile.general_requirement:.2f} and quality_target={profile.quality_target:.2f}.")
        return RoutingDecision(request_id=profile.request_id, selected_model=selected_model, candidate_models=accepted, rejected_models=rejected, predicted_quality=selected_candidate.predicted_quality, estimated_cost=selected_candidate.estimated_cost, estimated_latency=selected_candidate.estimated_latency, routing_reason=routing_reason, request_profile=profile, router_latency_ms=int((time.perf_counter() - started) * 1000))

    @staticmethod
    def _hard_rejection(model: ModelDefinition, profile: RequestProfile) -> str | None:
        if not model.enabled:
            return "model is disabled"
        if model.context_window < profile.required_context_tokens:
            return f"context window {model.context_window} is below required {profile.required_context_tokens} tokens"
        for capability, score, requirement in (("reasoning", model.reasoning_score, profile.reasoning_requirement), ("coding", model.coding_score, profile.coding_requirement), ("general", model.general_score, profile.general_requirement)):
            if requirement >= 0.55 and score + 0.08 < requirement:
                return f"{capability} capability {score:.2f} cannot meet requirement {requirement:.2f}"
        latency_budget = int(200 + (1 - profile.latency_priority) * 3_300)
        if profile.latency_priority >= 0.70 and model.expected_latency_ms > latency_budget:
            return f"expected latency {model.expected_latency_ms}ms exceeds {latency_budget}ms priority budget"
        return None

    @staticmethod
    def _predicted_quality(model: ModelDefinition, profile: RequestProfile) -> float:
        dimensions = ((profile.reasoning_requirement, model.reasoning_score), (profile.coding_requirement, model.coding_score), (profile.general_requirement, model.general_score), (profile.context_requirement, min(1.0, model.context_window / max(profile.required_context_tokens, 1))))
        total_weight = sum(weight for weight, _ in dimensions)
        return model.general_score if total_weight == 0 else round(sum(weight * score for weight, score in dimensions) / total_weight, 3)

    @staticmethod
    def _estimated_cost(model: ModelDefinition, profile: RequestProfile) -> float:
        return round((profile.estimated_input_tokens * model.input_cost_per_million_tokens + profile.estimated_output_tokens * model.output_cost_per_million_tokens) / 1_000_000, 8)

    @staticmethod
    def _fitness(quality: float, cost: float, latency_ms: int, profile: RequestProfile) -> float:
        return round(quality - 0.18 * min(1.0, cost / 0.05) - 0.20 * profile.latency_priority * min(1.0, latency_ms / 4_000), 3)
