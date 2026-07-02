__all__ = [
    "TavusConfigurationError",
    "TavusConversationSession",
    "TavusService",
    "TavusServiceError",
]


def __getattr__(name: str):
    if name in {"TavusConfigurationError", "TavusServiceError"}:
        from app.services.tavus.errors import TavusConfigurationError, TavusServiceError

        exports = {
            "TavusConfigurationError": TavusConfigurationError,
            "TavusServiceError": TavusServiceError,
        }
        return exports[name]

    if name in {"TavusConversationSession", "TavusService"}:
        from app.services.tavus.service import TavusConversationSession, TavusService

        exports = {
            "TavusConversationSession": TavusConversationSession,
            "TavusService": TavusService,
        }
        return exports[name]

    raise AttributeError(f"module 'app.services.tavus' has no attribute {name!r}")
