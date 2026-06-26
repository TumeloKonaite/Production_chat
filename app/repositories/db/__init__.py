__all__ = ["Base", "Conversation", "KnowledgeChunk", "Message", "utcnow"]


def __getattr__(name: str):
    if name == "Base":
        from app.repositories.db.base import Base

        return Base

    if name in {"Conversation", "KnowledgeChunk", "Message", "utcnow"}:
        from app.repositories.models import Conversation, KnowledgeChunk, Message, utcnow

        exports = {
            "Conversation": Conversation,
            "KnowledgeChunk": KnowledgeChunk,
            "Message": Message,
            "utcnow": utcnow,
        }
        return exports[name]

    raise AttributeError(f"module 'app.repositories.db' has no attribute {name!r}")
