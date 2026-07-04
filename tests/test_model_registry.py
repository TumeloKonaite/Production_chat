from __future__ import annotations

import json

import pytest

from app.infrastructure.llm import CostEstimate, ModelConfig, ModelRegistry, TokenUsage, UnknownModelError
from app.infrastructure.llm.model_config import ModelConfigValidationError, load_model_configs


def test_model_registry_returns_default_model() -> None:
    registry = ModelRegistry(default_model_config_id="openai:gpt-4.1-mini")

    model = registry.get_default_model()

    assert model.config_id == "openai:gpt-4.1-mini"
    assert model.provider == "openai"
    assert model.model == "gpt-4.1-mini"


def test_model_registry_normalizes_legacy_openai_ids() -> None:
    registry = ModelRegistry(default_model_config_id="openai:gpt-4.1-mini")

    model = registry.get_model("gpt-4.1-mini")

    assert model.config_id == "openai:gpt-4.1-mini"


def test_model_registry_estimates_cost_from_token_usage() -> None:
    registry = ModelRegistry(default_model_config_id="openai:gpt-4.1-mini")

    cost = registry.estimate_cost(
        "openai:gpt-4.1-mini",
        TokenUsage(input_tokens=1200, output_tokens=180, total_tokens=1380),
    )

    assert cost == pytest.approx(0.000768)


def test_model_registry_returns_cost_breakdown() -> None:
    registry = ModelRegistry(default_model_config_id="openai:gpt-4.1-mini")

    estimate = registry.estimate_costs(
        "openai:gpt-4.1-mini",
        TokenUsage(input_tokens=1200, output_tokens=180, total_tokens=1380),
    )

    assert estimate == CostEstimate(
        prompt_cost_usd=pytest.approx(0.00048),
        completion_cost_usd=pytest.approx(0.000288),
        total_cost_usd=pytest.approx(0.000768),
    )


def test_model_registry_preserves_existing_costs_when_default_override_has_none() -> None:
    registry = ModelRegistry(
        default_model_config_id="openai:gpt-4.1-mini",
        default_model_config=ModelConfig(
            config_id="openai:gpt-4.1-mini",
            provider="openai",
            model="gpt-4.1-mini",
        ),
    )

    estimate = registry.estimate_costs(
        "openai:gpt-4.1-mini",
        TokenUsage(input_tokens=1200, output_tokens=180, total_tokens=1380),
    )

    assert estimate.total_cost_usd == pytest.approx(0.000768)


def test_model_registry_loads_custom_model_configs_from_json() -> None:
    registry = ModelRegistry(
        default_model_config_id="openai:gpt-4.1-mini",
        model_configs_json=json.dumps(
            [
                {
                    "config_id": "openrouter:anthropic/claude-3.5-sonnet",
                    "provider": "openrouter",
                    "model": "anthropic/claude-3.5-sonnet",
                    "input_cost_per_1m_tokens": 3.0,
                    "output_cost_per_1m_tokens": 15.0,
                }
            ]
        ),
    )

    model = registry.get_model("openrouter:anthropic/claude-3.5-sonnet")

    assert model.config_id == "openrouter:anthropic/claude-3.5-sonnet"
    assert model.provider == "openrouter"
    assert model.model == "anthropic/claude-3.5-sonnet"


def test_model_registry_allows_costless_custom_model_configs() -> None:
    registry = ModelRegistry(
        default_model_config_id="openrouter:google/gemini-2.5-flash",
        model_configs_json=json.dumps(
            [
                {
                    "config_id": "openrouter:google/gemini-2.5-flash",
                    "provider": "openrouter",
                    "model": "google/gemini-2.5-flash",
                }
            ]
        ),
    )

    model = registry.get_model("openrouter:google/gemini-2.5-flash")
    estimate = registry.estimate_costs(
        "openrouter:google/gemini-2.5-flash",
        TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120),
    )

    assert model.input_cost_per_1m_tokens is None
    assert model.output_cost_per_1m_tokens is None
    assert estimate == CostEstimate(None, None, None)


def test_model_registry_resolves_custom_default_model_config_id() -> None:
    registry = ModelRegistry(
        default_model_config_id="openrouter:openai/gpt-4.1-mini",
        model_configs_json=json.dumps(
            [
                {
                    "config_id": "openrouter:openai/gpt-4.1-mini",
                    "provider": "openrouter",
                    "model": "openai/gpt-4.1-mini",
                    "input_cost_per_1m_tokens": 0.0,
                    "output_cost_per_1m_tokens": 0.0,
                }
            ]
        ),
    )

    model = registry.get_default_model()

    assert model.config_id == "openrouter:openai/gpt-4.1-mini"
    assert model.model == "openai/gpt-4.1-mini"


def test_model_registry_rejects_unknown_model() -> None:
    registry = ModelRegistry(default_model_config_id="openai:gpt-4.1-mini")

    with pytest.raises(UnknownModelError) as exc_info:
        registry.get_model("openai:missing")

    assert str(exc_info.value) == (
        "Unknown model config ID: openai:missing. Available models: "
        "openai:gpt-4.1, openai:gpt-4.1-mini"
    )


def test_load_model_configs_rejects_malformed_json() -> None:
    with pytest.raises(ModelConfigValidationError) as exc_info:
        load_model_configs("{not-json}")

    assert str(exc_info.value) == "MODEL_CONFIGS_JSON must contain valid JSON."


def test_load_model_configs_rejects_unsupported_provider() -> None:
    with pytest.raises(ModelConfigValidationError) as exc_info:
        load_model_configs(
            json.dumps(
                [
                    {
                        "config_id": "custom:model",
                        "provider": "anthropic",
                        "model": "claude-3.5-sonnet",
                        "input_cost_per_1m_tokens": 3.0,
                        "output_cost_per_1m_tokens": 15.0,
                    }
                ]
            )
        )

    assert str(exc_info.value) == (
        "MODEL_CONFIGS_JSON entry 1 field 'provider' must be one of: openai, openrouter."
    )
