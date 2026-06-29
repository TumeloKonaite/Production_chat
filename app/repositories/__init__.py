__all__ = [
    "ConversationRepository",
    "ConversationRepositoryError",
    "EvalRepository",
    "EvalRepositoryError",
    "KnowledgeRepository",
    "KnowledgeRepositoryError",
]


def __getattr__(name: str):
    if name in {"ConversationRepository", "ConversationRepositoryError"}:
        from app.repositories.chat_repository import (
            ConversationRepository,
            ConversationRepositoryError,
        )

        exports = {
            "ConversationRepository": ConversationRepository,
            "ConversationRepositoryError": ConversationRepositoryError,
        }
        return exports[name]

    if name in {"KnowledgeRepository", "KnowledgeRepositoryError"}:
        from app.repositories.knowledge_repository import (
            KnowledgeRepository,
            KnowledgeRepositoryError,
        )

        exports = {
            "KnowledgeRepository": KnowledgeRepository,
            "KnowledgeRepositoryError": KnowledgeRepositoryError,
        }
        return exports[name]

    if name in {"EvalRepository", "EvalRepositoryError"}:
        from app.repositories.eval_repository import EvalRepository, EvalRepositoryError

        exports = {
            "EvalRepository": EvalRepository,
            "EvalRepositoryError": EvalRepositoryError,
        }
        return exports[name]

    raise AttributeError(f"module 'app.repositories' has no attribute {name!r}")
