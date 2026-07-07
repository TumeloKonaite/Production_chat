from __future__ import annotations

from dataclasses import dataclass

from app.infrastructure.llm.base import TokenUsage
from app.infrastructure.llm.model_config import ModelConfig, load_model_configs


@dataclass(frozen=True, slots=True)
class CostEstimate:
    prompt_cost_usd: float | None
    completion_cost_usd: float | None
    total_cost_usd: float | None


class UnknownModelError(ValueError):
    def __init__(self, model_config_id: str, available_models: list[str]) -> None:
        available = ", ".join(available_models)
        super().__init__(
            f"Unknown model config ID: {model_config_id}. Available models: {available}"
        )


class ModelRegistry:
    def __init__(
        self,
        *,
        model_configs: dict[str, ModelConfig] | None = None,
        model_configs_json: str | None = None,
        default_model_config_id: str,
        default_model_config: ModelConfig | None = None,
    ) -> None:
        self._model_configs = (
            dict(model_configs)
            if model_configs is not None
            else load_model_configs(model_configs_json)
        )
        if default_model_config is not None:
            existing_model_config = self._model_configs.get(default_model_config.config_id)
            if existing_model_config is None:
                self._model_configs[default_model_config.config_id] = default_model_config
            else:
                self._model_configs[default_model_config.config_id] = ModelConfig(
                    config_id=default_model_config.config_id,
                    provider=default_model_config.provider,
                    model=default_model_config.model,
                    input_cost_per_1m_tokens=(
                        default_model_config.input_cost_per_1m_tokens
                        if default_model_config.input_cost_per_1m_tokens is not None
                        else existing_model_config.input_cost_per_1m_tokens
                    ),
                    output_cost_per_1m_tokens=(
                        default_model_config.output_cost_per_1m_tokens
                        if default_model_config.output_cost_per_1m_tokens is not None
                        else existing_model_config.output_cost_per_1m_tokens
                    ),
                )
        self._default_model_config_id = default_model_config_id
        self.get_model(default_model_config_id)

    @property
    def default_model_config_id(self) -> str:
        return self._default_model_config_id

    def available_model_ids(self) -> list[str]:
        return sorted(self._model_configs)

    def get_default_model(self) -> ModelConfig:
        return self.get_model(self.default_model_config_id)

    def resolve(self, model_config_id: str | None = None) -> ModelConfig:
        if model_config_id is None:
            return self.get_default_model()
        return self.get_model(model_config_id)

    def get_model(self, model_config_id: str) -> ModelConfig:
        normalized_model_config_id = self._normalize_model_config_id(model_config_id)
        model_config = self._model_configs.get(normalized_model_config_id)
        if model_config is None:
            raise UnknownModelError(model_config_id, self.available_model_ids())
        return model_config

    def estimate_cost(
        self,
        model_config_id: str,
        token_usage: TokenUsage,
    ) -> float | None:
        return self.estimate_costs(model_config_id, token_usage).total_cost_usd

    def estimate_costs(
        self,
        model_config_id: str,
        token_usage: TokenUsage,
    ) -> CostEstimate:
        model_config = self.get_model(model_config_id)
        if token_usage.input_tokens is None or token_usage.output_tokens is None:
            return CostEstimate(None, None, None)
        if (
            model_config.input_cost_per_1m_tokens is None
            or model_config.output_cost_per_1m_tokens is None
        ):
            return CostEstimate(None, None, None)

        input_cost = round(
            token_usage.input_tokens / 1_000_000
            * model_config.input_cost_per_1m_tokens,
            6,
        )
        output_cost = round(
            token_usage.output_tokens / 1_000_000
            * model_config.output_cost_per_1m_tokens,
            6,
        )
        return CostEstimate(
            prompt_cost_usd=input_cost,
            completion_cost_usd=output_cost,
            total_cost_usd=round(input_cost + output_cost, 6),
        )

    def _normalize_model_config_id(self, model_config_id: str) -> str:
        if ":" in model_config_id:
            return model_config_id

        legacy_openai_id = f"openai:{model_config_id}"
        if legacy_openai_id in self._model_configs:
            return legacy_openai_id

        return model_config_id
