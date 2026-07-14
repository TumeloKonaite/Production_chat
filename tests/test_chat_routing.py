from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import TypeAdapter, ValidationError

from app.services.chat.context_resolution import ConversationContextResolver
from app.services.chat.models import (
    DirectResponseKind,
    QueryRoute,
    RetrievalMode,
    RoutingDecision,
)
from app.services.chat.routing import PortfolioScopeRouter


@dataclass
class Message:
    role: str
    content: str


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        ("Hello", DirectResponseKind.GREETING),
        ("Thank you", DirectResponseKind.ACKNOWLEDGEMENT),
    ],
)
def test_router_selects_non_retrieval_responses(message, kind) -> None:
    decision = PortfolioScopeRouter().route(message=message)

    assert decision.route == QueryRoute.DIRECT_RESPONSE
    assert decision.direct_response_kind == kind


@pytest.mark.parametrize(
    ("message", "mode"),
    [
        ("What projects has Tumelo built?", RetrievalMode.PROJECT_OVERVIEW),
        (
            "What are his main machine-learning projects?",
            RetrievalMode.PROJECT_OVERVIEW,
        ),
        ("Tell me about Tumelo's portfolio.", RetrievalMode.PROJECT_OVERVIEW),
        ("What degree does Tumelo have?", RetrievalMode.HYBRID),
        ("How has Tumelo used pgvector?", RetrievalMode.HYBRID),
    ],
)
def test_router_selects_portfolio_retrieval_mode(message, mode) -> None:
    decision = PortfolioScopeRouter().route(message=message)

    assert decision.route == QueryRoute.PORTFOLIO_KNOWLEDGE
    assert decision.retrieval_mode == mode


def test_router_rejects_general_technical_question() -> None:
    decision = PortfolioScopeRouter().route(message="What is pgvector?")

    assert decision.route == QueryRoute.DIRECT_RESPONSE
    assert decision.direct_response_kind == DirectResponseKind.OUT_OF_SCOPE


def test_contextual_follow_up_resolves_with_portfolio_history() -> None:
    resolver = ConversationContextResolver()
    history = [
        Message("user", "Tell me about Tumelo's production chatbot."),
        Message("assistant", "Tumelo built a production chatbot with FastAPI."),
        Message("user", "What database does it use?"),
    ]

    resolved = resolver.resolve(
        message="What database does it use?", recent_messages=history
    )
    decision = PortfolioScopeRouter(resolver).route(
        message="What database does it use?",
        resolved_query=resolved,
        recent_messages=history,
    )

    assert resolved is not None
    assert "production chatbot" in resolved
    assert decision.route == QueryRoute.PORTFOLIO_KNOWLEDGE
    assert decision.retrieval_mode == RetrievalMode.HYBRID


def test_contextual_follow_up_without_history_requests_clarification() -> None:
    decision = PortfolioScopeRouter().route(message="Tell me more.")

    assert decision.route == QueryRoute.DIRECT_RESPONSE
    assert decision.direct_response_kind == DirectResponseKind.CLARIFICATION


def test_routing_union_rejects_contradictory_fields() -> None:
    adapter = TypeAdapter(RoutingDecision)

    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "route": "direct_response",
                "resolved_query": None,
                "reason_code": "invalid",
                "retrieval_mode": "hybrid",
            }
        )
