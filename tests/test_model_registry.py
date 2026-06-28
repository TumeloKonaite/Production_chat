from __future__ import annotations

import pytest

from app.infrastructure.llm import ModelRegistry, TokenUsage, UnknownModelError


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


def test_model_registry_rejects_unknown_model() -> None:
    registry = ModelRegistry(default_model_config_id="openai:gpt-4.1-mini")

    with pytest.raises(UnknownModelError) as exc_info:
        registry.get_model("openai:missing")

    assert str(exc_info.value) == (
        "Unknown model config ID: openai:missing. Available models: "
        "openai:gpt-4.1, openai:gpt-4.1-mini"
    )
