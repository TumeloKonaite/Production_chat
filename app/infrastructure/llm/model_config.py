from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from app.services.llm.errors import LLMConfigurationError

SUPPORTED_MODEL_PROVIDERS = {"openai", "openrouter"}


@dataclass(frozen=True, slots=True)
class ModelConfig:
    config_id: str
    provider: str
    model: str
    input_cost_per_1m_tokens: float
    output_cost_per_1m_tokens: float


class ModelConfigValidationError(LLMConfigurationError):
    """Raised when model config settings are malformed."""


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


def load_model_configs(model_configs_json: str | None = None) -> dict[str, ModelConfig]:
    model_configs = dict(MODEL_CONFIGS)
    model_configs.update(parse_model_configs_json(model_configs_json))
    return model_configs


def parse_model_configs_json(model_configs_json: str | None) -> dict[str, ModelConfig]:
    if model_configs_json is None:
        return {}

    try:
        payload = json.loads(model_configs_json)
    except json.JSONDecodeError as exc:
        raise ModelConfigValidationError("MODEL_CONFIGS_JSON must contain valid JSON.") from exc

    if not isinstance(payload, list):
        raise ModelConfigValidationError(
            "MODEL_CONFIGS_JSON must be a JSON array of model config objects."
        )

    parsed_configs: dict[str, ModelConfig] = {}
    for index, entry in enumerate(payload, start=1):
        model_config = _parse_model_config(index, entry)
        parsed_configs[model_config.config_id] = model_config
    return parsed_configs


def _parse_model_config(index: int, entry: object) -> ModelConfig:
    if not isinstance(entry, dict):
        raise ModelConfigValidationError(
            f"MODEL_CONFIGS_JSON entry {index} must be a JSON object."
        )

    config_id = _require_string(entry, "config_id", index)
    provider = _require_provider(entry, index)
    model = _require_string(entry, "model", index)
    input_cost = _require_number(entry, "input_cost_per_1m_tokens", index)
    output_cost = _require_number(entry, "output_cost_per_1m_tokens", index)

    return ModelConfig(
        config_id=config_id,
        provider=provider,
        model=model,
        input_cost_per_1m_tokens=input_cost,
        output_cost_per_1m_tokens=output_cost,
    )


def _require_string(entry: dict[str, Any], field_name: str, index: int) -> str:
    value = entry.get(field_name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ModelConfigValidationError(
        f"MODEL_CONFIGS_JSON entry {index} field '{field_name}' must be a non-empty string."
    )


def _require_provider(entry: dict[str, Any], index: int) -> str:
    provider = _require_string(entry, "provider", index)
    if provider not in SUPPORTED_MODEL_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_MODEL_PROVIDERS))
        raise ModelConfigValidationError(
            f"MODEL_CONFIGS_JSON entry {index} field 'provider' must be one of: {supported}."
        )
    return provider


def _require_number(entry: dict[str, Any], field_name: str, index: int) -> float:
    value = entry.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ModelConfigValidationError(
            f"MODEL_CONFIGS_JSON entry {index} field '{field_name}' must be a number."
        )
    return float(value)
