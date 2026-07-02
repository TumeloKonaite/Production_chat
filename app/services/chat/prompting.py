from __future__ import annotations

from collections.abc import Sequence
import re

from app.services.retrieval import RetrievedChunk

PERSONAL_QUERY_MARKERS = {
    "tumelo",
    "you",
    "your",
    "yours",
    "experience",
    "background",
    "education",
    "contact",
    "email",
    "linkedin",
    "github",
    "portfolio",
    "project",
    "projects",
    "skill",
    "skills",
    "worked",
    "career",
    "resume",
    "cv",
    "employer",
    "employment",
    "degree",
    "certification",
    "location",
}
PROJECT_QUERY_MARKERS = {
    "project",
    "projects",
}
_PROJECT_QUERY_TOKEN_RE = re.compile(r"[a-z0-9]+")
_BROAD_PROJECT_STOPWORDS = {
    "about",
    "and",
    "any",
    "are",
    "built",
    "details",
    "has",
    "list",
    "me",
    "of",
    "on",
    "project",
    "projects",
    "tell",
    "the",
    "tumelo",
    "what",
    "which",
}
GENERAL_TECH_MARKERS = {
    "api",
    "backend",
    "chatbot",
    "database",
    "docker",
    "embedding",
    "fastapi",
    "llm",
    "postgres",
    "python",
    "rag",
    "retrieval",
    "sql",
    "sqlalchemy",
    "vector",
}


def build_chat_system_prompt(
    *,
    base_prompt: str,
    message: str,
    retrieved_chunks: Sequence[RetrievedChunk],
) -> str:
    context_block = format_retrieved_context(retrieved_chunks)
    guidance = [
        "Use the retrieved context to answer the visitor's question when it is relevant to Tumelo.",
        "Do not invent experience, projects, employers, dates, tools, certifications, or achievements.",
        "If the approved context does not contain enough Tumelo-specific information, say that you do not have that information available.",
        "If the user is asking a general technical question, you may answer generally, but do not present general knowledge as Tumelo's personal experience.",
    ]
    if is_broad_project_query(message):
        guidance.append(
            "If the user is asking broadly about Tumelo's projects, summarize the most relevant projects from the approved project context with project names and concise descriptions before offering more detail."
        )
    if not retrieved_chunks:
        guidance.append(
            "No relevant approved Tumelo context was retrieved for this turn, so avoid personal claims unless they are already established in the conversation."
        )

    return "\n\n".join(
        [
            base_prompt,
            "Approved Tumelo knowledge base context:\n" + context_block,
            "Additional rules:\n" + "\n".join(f"- {rule}" for rule in guidance),
            f"Current user question:\n{message}",
        ]
    )


def format_retrieved_context(retrieved_chunks: Sequence[RetrievedChunk]) -> str:
    if not retrieved_chunks:
        return "No approved context retrieved."

    formatted_chunks = []
    for item in retrieved_chunks:
        formatted_chunks.append(
            "\n".join(
                [
                    f"Source: {item.source}",
                    f"Section: {item.section}",
                    f"Similarity: {item.similarity:.3f}",
                    item.content,
                ]
            )
        )
    return "\n\n---\n\n".join(formatted_chunks)


def should_use_direct_fallback(
    message: str,
    retrieved_chunks: Sequence[RetrievedChunk],
) -> bool:
    return not retrieved_chunks and _message_prefers_direct_fallback(message)


def is_personal_query(message: str) -> bool:
    normalized_message = message.casefold()
    return any(marker in normalized_message for marker in PERSONAL_QUERY_MARKERS)


def build_direct_fallback_text(message: str) -> str:
    if is_personal_query(message):
        return "I do not have enough approved information about that in Tumelo's knowledge base yet."

    return "Could you clarify whether you're asking about Tumelo's background or a general technical topic?"


def is_broad_project_query(message: str) -> bool:
    normalized_message = message.casefold()
    if not any(marker in normalized_message for marker in PROJECT_QUERY_MARKERS):
        return False

    query_terms = [
        token
        for token in _PROJECT_QUERY_TOKEN_RE.findall(normalized_message)
        if len(token) >= 3 and token not in _BROAD_PROJECT_STOPWORDS
    ]
    return len(query_terms) == 0


def _message_prefers_direct_fallback(message: str) -> bool:
    if is_personal_query(message):
        return True
    if _is_general_technical_query(message):
        return False
    return True


def _is_general_technical_query(message: str) -> bool:
    normalized_message = message.casefold()
    if is_personal_query(message):
        return False
    return any(marker in normalized_message for marker in GENERAL_TECH_MARKERS)
