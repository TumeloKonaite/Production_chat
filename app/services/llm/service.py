from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.config import Settings
from app.infrastructure.llm import (
    LLMChatMessage,
    ModelConfig,
    ModelRegistry,
    OpenAIClient,
    TokenUsage,
)
from app.services.llm.errors import LLMConfigurationError


@dataclass(frozen=True, slots=True)
class LLMGeneratedResponse:
    message: str
    model: str
    model_provider: str
    model_name: str
    model_config_id: str
    prompt_version: str
    retrieval_config: str
    latency_ms: int | None
    token_usage: TokenUsage
    estimated_cost_usd: float | None


class LLMService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model_registry = ModelRegistry(
            default_model_config_id=settings.default_model_config_id,
            model_configs_json=settings.model_configs_json,
        )
        self._clients = {
            "openai": OpenAIClient.from_settings(settings, provider="openai"),
            "openrouter": OpenAIClient.from_settings(settings, provider="openrouter"),
        }

    @property
    def model(self) -> str:
        return self._model_registry.get_default_model().model

    @property
    def default_model_config_id(self) -> str:
        return self._model_registry.default_model_config_id

    def get_model_config(self, model_config_id: str | None = None) -> ModelConfig:
        return self._model_registry.resolve(model_config_id)

    async def generate_response(
        self,
        messages: Sequence[LLMChatMessage],
        *,
        system_prompt: str,
        prompt_version: str,
        retrieval_config: str = "default",
        temperature: float | None = None,
        model_config_id: str | None = None,
    ) -> LLMGeneratedResponse:
        model_config = self.get_model_config(model_config_id)
        client = self._clients.get(model_config.provider)
        if client is None:
            raise LLMConfigurationError(
                f"No LLM client configured for provider: {model_config.provider}"
            )

        response = await client.generate(
            [
                LLMChatMessage(role="developer", content=system_prompt),
                *list(messages),
            ],
            model=model_config.model,
            temperature=temperature,
        )

        return LLMGeneratedResponse(
            message=response.content,
            model=response.model,
            model_provider=model_config.provider,
            model_name=response.model,
            model_config_id=model_config.config_id,
            prompt_version=prompt_version,
            retrieval_config=retrieval_config,
            latency_ms=response.latency_ms,
            token_usage=TokenUsage(
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=response.total_tokens,
            ),
            estimated_cost_usd=self._model_registry.estimate_cost(
                model_config.config_id,
                TokenUsage(
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    total_tokens=response.total_tokens,
                ),
            ),
        )
