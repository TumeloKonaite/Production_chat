from __future__ import annotations

import re

from app.services.chat.context_resolution import (
    ChatMessage,
    ConversationContextResolver,
)
from app.services.chat.models import (
    DirectResponseDecision,
    DirectResponseKind,
    PortfolioKnowledgeDecision,
    QueryRoute,
    RetrievalMode,
    RoutingDecision,
)
from app.services.chat.prompting import is_broad_project_query


_GREETING_RE = re.compile(
    r"^(?:hi|hello|hey|good (?:morning|afternoon|evening)|how are you)[!.? ]*$",
    re.I,
)
_ACKNOWLEDGEMENT_RE = re.compile(
    r"^(?:thanks|thank you|thanks a lot|okay|ok|great|that helps|got it)[!. ]*$",
    re.I,
)
_PORTFOLIO_MARKERS = (
    "tumelo",
    "portfolio",
    "project",
    "projects",
    "background",
    "experience",
    "education",
    "degree",
    "qualification",
    "skill",
    "skills",
    "career",
    "resume",
    " cv ",
    "employer",
    "employment",
    "worked",
    "role at",
    "certification",
)
_TECHNICAL_PORTFOLIO_MARKERS = (
    "chatbot",
    "database",
    "technology",
    "technologies",
    "implementation",
    "pgvector",
    "redis",
    "fastapi",
    "python",
    "rag",
)
DIRECT_RESPONSE_TEMPLATES = {
    DirectResponseKind.GREETING: (
        "Hi! Ask me anything about Tumelo's background, experience, skills, or projects."
    ),
    DirectResponseKind.ACKNOWLEDGEMENT: "You're welcome.",
    DirectResponseKind.CLARIFICATION: (
        "Which part of Tumelo's background or which project would you like to know more about?"
    ),
    DirectResponseKind.OUT_OF_SCOPE: (
        "I'm here to answer questions about Tumelo's background and projects."
    ),
}
_REDIRECT_TOPICS = (
    "pgvector",
    "redis",
    "fastapi",
    "python",
    "postgresql",
    "postgres",
    "docker",
    "rag",
    "retrieval",
    "embedding",
    "llm",
)


def build_direct_response_text(kind: DirectResponseKind, message: str) -> str:
    response = DIRECT_RESPONSE_TEMPLATES[kind]
    if kind != DirectResponseKind.OUT_OF_SCOPE:
        return response
    normalized = message.casefold()
    topic = next((item for item in _REDIRECT_TOPICS if item in normalized), None)
    if topic is None:
        return response
    return f"{response} You could ask how Tumelo has used {topic} in his work."


class PortfolioScopeRouter:
    def __init__(
        self, context_resolver: ConversationContextResolver | None = None
    ) -> None:
        self.context_resolver = context_resolver or ConversationContextResolver()

    def route(
        self,
        *,
        message: str,
        resolved_query: str | None = None,
        recent_messages: list[ChatMessage] | None = None,
    ) -> RoutingDecision:
        normalized = " ".join(message.split())
        if _GREETING_RE.fullmatch(normalized):
            return DirectResponseDecision(
                route=QueryRoute.DIRECT_RESPONSE,
                reason_code="greeting_detected",
                direct_response_kind=DirectResponseKind.GREETING,
            )
        if _ACKNOWLEDGEMENT_RE.fullmatch(normalized):
            return DirectResponseDecision(
                route=QueryRoute.DIRECT_RESPONSE,
                reason_code="acknowledgement_detected",
                direct_response_kind=DirectResponseKind.ACKNOWLEDGEMENT,
            )

        if (
            self.context_resolver.requires_resolution(normalized)
            and resolved_query is None
        ):
            return DirectResponseDecision(
                route=QueryRoute.DIRECT_RESPONSE,
                reason_code="context_unresolved",
                direct_response_kind=DirectResponseKind.CLARIFICATION,
            )

        query = resolved_query or normalized
        if self._is_portfolio_query(query):
            retrieval_mode = (
                RetrievalMode.PROJECT_OVERVIEW
                if is_broad_project_query(query)
                else RetrievalMode.HYBRID
            )
            return PortfolioKnowledgeDecision(
                route=QueryRoute.PORTFOLIO_KNOWLEDGE,
                resolved_query=query,
                reason_code=(
                    "portfolio_project_overview"
                    if retrieval_mode == RetrievalMode.PROJECT_OVERVIEW
                    else "portfolio_scope_detected"
                ),
                retrieval_mode=retrieval_mode,
            )

        return DirectResponseDecision(
            route=QueryRoute.DIRECT_RESPONSE,
            reason_code="outside_portfolio_scope",
            direct_response_kind=DirectResponseKind.OUT_OF_SCOPE,
        )

    def _is_portfolio_query(self, query: str) -> bool:
        padded = f" {query.casefold()} "
        if any(marker in padded for marker in _PORTFOLIO_MARKERS):
            return True
        has_person_reference = bool(re.search(r"\b(?:he|his|you|your)\b", padded))
        return has_person_reference and any(
            marker in padded for marker in _TECHNICAL_PORTFOLIO_MARKERS
        )
