from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class QueryRoute(StrEnum):
    PORTFOLIO_KNOWLEDGE = "portfolio_knowledge"
    DIRECT_RESPONSE = "direct_response"


class RetrievalMode(StrEnum):
    HYBRID = "hybrid"
    PROJECT_OVERVIEW = "project_overview"


class DirectResponseKind(StrEnum):
    GREETING = "greeting"
    ACKNOWLEDGEMENT = "acknowledgement"
    CLARIFICATION = "clarification"
    OUT_OF_SCOPE = "out_of_scope"


class PortfolioKnowledgeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: Literal[QueryRoute.PORTFOLIO_KNOWLEDGE]
    resolved_query: str
    reason_code: str
    retrieval_mode: RetrievalMode


class DirectResponseDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: Literal[QueryRoute.DIRECT_RESPONSE]
    resolved_query: str | None = None
    reason_code: str
    direct_response_kind: DirectResponseKind


RoutingDecision = Annotated[
    PortfolioKnowledgeDecision | DirectResponseDecision,
    Field(discriminator="route"),
]
