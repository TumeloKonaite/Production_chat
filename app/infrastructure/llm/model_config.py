from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    config_id: str
    provider: str
    model: str
    input_cost_per_1m_tokens: float
    output_cost_per_1m_tokens: float


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "openai:gpt-4.1-mini": ModelConfig(
        config_id="openai:gpt-4.1-mini",
        provider="openai",
        model="gpt-4.1-mini",
        input_cost_per_1m_tokens=0.40,
        output_cost_per_1m_tokens=1.60,
    ),
    "openai:gpt-4.1": ModelConfig(
        config_id="openai:gpt-4.1",
        provider="openai",
        model="gpt-4.1",
        input_cost_per_1m_tokens=2.00,
        output_cost_per_1m_tokens=8.00,
    ),
}
