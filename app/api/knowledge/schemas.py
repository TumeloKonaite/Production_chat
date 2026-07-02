from typing import Literal

from pydantic import BaseModel


class KnowledgeIngestionDocumentResponse(BaseModel):
    source: str
    chunk_count: int


class KnowledgeIngestionResponse(BaseModel):
    status: Literal["ok"]
    documents_loaded: int
    results: list[KnowledgeIngestionDocumentResponse]
