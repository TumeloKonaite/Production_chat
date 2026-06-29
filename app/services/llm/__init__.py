__all__ = [
    "LLMChatMessage",
    "LLMConfigurationError",
    "LLMGeneratedResponse",
    "LLMService",
    "LLMServiceError",
    "ModelConfig",
    "TokenUsage",
]


def __getattr__(name: str):
    if name in {"LLMConfigurationError", "LLMServiceError"}:
        from app.services.llm.errors import LLMConfigurationError, LLMServiceError

        exports = {
            "LLMConfigurationError": LLMConfigurationError,
            "LLMServiceError": LLMServiceError,
        }
        return exports[name]

    if name in {"LLMChatMessage", "LLMGeneratedResponse", "LLMService", "ModelConfig", "TokenUsage"}:
        from app.services.llm.service import (
            LLMChatMessage,
            LLMGeneratedResponse,
            LLMService,
            ModelConfig,
            TokenUsage,
        )

        exports = {
            "LLMChatMessage": LLMChatMessage,
            "LLMGeneratedResponse": LLMGeneratedResponse,
            "LLMService": LLMService,
            "ModelConfig": ModelConfig,
            "TokenUsage": TokenUsage,
        }
        return exports[name]

    raise AttributeError(f"module 'app.services.llm' has no attribute {name!r}")
