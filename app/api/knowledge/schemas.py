from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class KnowledgeIngestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: Literal["local_directory", "uploaded_file"] = "local_directory"
    file_id: UUID | None = None
    experiment_name: str | None = Field(default=None, min_length=1, max_length=200)
    embedding_provider: Literal["hf", "openai", "openrouter"] | None = None
    embedding_model: str | None = Field(default=None, min_length=1, max_length=300)
    embedding_dimension: int | None = Field(default=None, gt=0)
    reset_existing_vectors: bool | None = None

    @property
    def has_embedding_override(self) -> bool:
        return (
            self.embedding_provider is not None
            or self.embedding_model is not None
            or self.embedding_dimension is not None
        )

    @model_validator(mode="after")
    def validate_embedding_override(self) -> KnowledgeIngestionRequest:
        if self.source_type == "uploaded_file" and self.file_id is None:
            raise ValueError("file_id is required when source_type=uploaded_file.")
        if self.source_type == "local_directory" and self.file_id is not None:
            raise ValueError("file_id is only supported when source_type=uploaded_file.")

        override_values = (
            self.embedding_provider,
            self.embedding_model,
            self.embedding_dimension,
        )
        provided_values = [value is not None for value in override_values]
        if any(provided_values) and not all(provided_values):
            raise ValueError(
                "embedding_provider, embedding_model, and embedding_dimension must all be "
                "provided together."
            )

        if self.has_embedding_override and self.reset_existing_vectors is not True:
            raise ValueError(
                "reset_existing_vectors=true is required when overriding the embedding "
                "configuration."
            )

        return self


class KnowledgeIngestionDocumentResponse(BaseModel):
    source: str
    chunk_count: int


class KnowledgeIngestionResponse(BaseModel):
    job_id: UUID
    status: Literal["queued", "skipped"]
    source_type: Literal["local_directory", "uploaded_file"]
    file_id: UUID | None = None


class KnowledgeFileUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    original_filename: str
    content_type: str | None
    file_size_bytes: int
    storage_provider: str
    storage_bucket: str
    storage_path: str
    status: Literal["uploaded", "ingesting", "ingested", "failed", "deleted"]
    created_at: datetime
