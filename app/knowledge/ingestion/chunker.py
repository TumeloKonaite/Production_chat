from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.config import DEFAULT_KNOWLEDGE_CHUNK_OVERLAP, DEFAULT_KNOWLEDGE_CHUNK_SIZE
from app.knowledge.ingestion.loader import SourceDocument


@dataclass(frozen=True, slots=True)
class ChunkedDocument:
    source: str
    source_type: str
    section: str
    content: str
    metadata: dict[str, object]
    updated_at: datetime


def chunk_markdown_document(
    document: SourceDocument,
    *,
    chunk_size: int = DEFAULT_KNOWLEDGE_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_KNOWLEDGE_CHUNK_OVERLAP,
) -> list[ChunkedDocument]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be zero or positive.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    # Split on markdown headings first so the resulting chunks keep useful
    # section context such as "Projects" or "Portfolio Chatbot".
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    section_documents = header_splitter.split_text(document.text)

    if not section_documents and document.text.strip():
        section_documents = [Document(page_content=document.text, metadata={})]

    # Apply character-based chunking after section splitting to keep the code
    # short while still enforcing a predictable chunk size and overlap.
    split_documents = recursive_splitter.split_documents(section_documents)
    chunked_documents: list[ChunkedDocument] = []
    section_indexes: dict[str, int] = {}
    global_chunk_index = 0

    for split_document in split_documents:
        content = split_document.page_content.strip()
        if not content:
            continue

        section = _resolve_section_name(split_document.metadata)
        section_chunk_index = section_indexes.get(section, 0)
        section_indexes[section] = section_chunk_index + 1

        metadata = {
            "chunk_index": global_chunk_index,
            "section_chunk_index": section_chunk_index,
            "source": document.source,
            "section": section,
            "content_type": document.source.removesuffix(".md"),
            "source_updated_at": document.updated_at.isoformat(),
        }
        chunked_documents.append(
            ChunkedDocument(
                source=document.source,
                source_type="markdown",
                section=section,
                content=content,
                metadata=metadata,
                updated_at=document.updated_at,
            )
        )
        global_chunk_index += 1

    return chunked_documents


def _resolve_section_name(metadata: dict[str, str]) -> str:
    for key in ("h3", "h2", "h1"):
        value = metadata.get(key)
        if value:
            return value
    return "Document"
