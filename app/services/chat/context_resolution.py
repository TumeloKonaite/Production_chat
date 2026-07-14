from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Protocol


class ChatMessage(Protocol):
    role: str
    content: str


_DEPENDENT_PATTERNS = (
    re.compile(
        r"^(?:tell me more|what about (?:that|this|it)|and (?:that|it))\b", re.I
    ),
    re.compile(r"\b(?:it|that project|this project|this one|that one)\b", re.I),
    re.compile(r"\bwhich one\b", re.I),
    re.compile(
        r"\b(?:the first|the second|the third|first one|second one|third one)\b", re.I
    ),
)
_PORTFOLIO_CONTEXT_MARKERS = (
    "tumelo",
    "project",
    "portfolio",
    "experience",
    "education",
    "degree",
    "skill",
    "worked",
    "chatbot",
)


class ConversationContextResolver:
    """Resolve only obvious follow-ups; ambiguous turns are deliberately left unresolved."""

    def requires_resolution(self, message: str) -> bool:
        normalized = " ".join(message.split())
        if any(pattern.search(normalized) for pattern in _DEPENDENT_PATTERNS):
            return True
        tokens = re.findall(r"[a-z0-9']+", normalized.casefold())
        return (
            len(tokens) <= 4
            and normalized.endswith(("?", "."))
            and any(token in {"he", "his", "it", "that", "more"} for token in tokens)
        )

    def resolve(
        self,
        *,
        message: str,
        recent_messages: Sequence[ChatMessage],
    ) -> str | None:
        prior_messages = self._prior_messages(message, recent_messages)
        if not prior_messages:
            return None

        prior_user = next(
            (
                item.content.strip()
                for item in reversed(prior_messages)
                if item.role == "user"
            ),
            None,
        )
        prior_assistant = next(
            (
                item.content.strip()
                for item in reversed(prior_messages)
                if item.role == "assistant"
            ),
            None,
        )
        context = prior_user or prior_assistant
        if context is None or not self._is_portfolio_context(context, prior_assistant):
            return None

        normalized = " ".join(message.split())
        if re.search(r"\b(?:second|first|third) (?:one|project)\b", normalized, re.I):
            if prior_assistant is None:
                return None
            return f"{normalized} from Tumelo's project list in the previous answer"

        # Keeping the original words plus the prior subject makes the resolution auditable and
        # avoids pretending that a deterministic resolver understood more than it did.
        return f"{normalized} Regarding Tumelo and the previous topic: {context}"

    def _prior_messages(
        self,
        current_message: str,
        recent_messages: Sequence[ChatMessage],
    ) -> list[ChatMessage]:
        messages = list(recent_messages)
        if (
            messages
            and messages[-1].role == "user"
            and messages[-1].content.strip() == current_message.strip()
        ):
            messages.pop()
        return messages

    def _is_portfolio_context(
        self,
        prior_user: str,
        prior_assistant: str | None,
    ) -> bool:
        combined = f"{prior_user} {prior_assistant or ''}".casefold()
        return any(marker in combined for marker in _PORTFOLIO_CONTEXT_MARKERS)
