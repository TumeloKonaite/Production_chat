from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.infrastructure.tavus import TavusClient
from app.repositories import ConversationRepository, ConversationRepositoryError
from app.services.chat.errors import ConversationNotFoundError, InvalidConversationIdError
from app.services.tavus.errors import TavusConfigurationError, TavusServiceError


@dataclass(frozen=True, slots=True)
class TavusConversationSession:
    conversation_id: str
    conversation_url: str


class TavusService:
    def __init__(
        self,
        *,
        settings: Settings,
        client: TavusClient,
        repository: ConversationRepository,
    ) -> None:
        self._settings = settings
        self._client = client
        self._repository = repository

    async def create_conversation(
        self,
        *,
        visitor_name: str,
        backend_conversation_id: str | None = None,
    ) -> TavusConversationSession:
        face_id = self._require_setting(self._settings.tavus_face_id, "Tavus face ID")
        pal_id = self._require_setting(self._settings.tavus_pal_id, "Tavus PAL ID")

        payload = await self._client.create_conversation(
            face_id=face_id,
            pal_id=pal_id,
            conversational_context=self._build_conversational_context(visitor_name=visitor_name),
        )
        conversation_id = self._extract_string(
            payload,
            "conversation_id",
            fallback_keys=("conversationId", "id"),
        )
        conversation_url = self._extract_string(
            payload,
            "conversation_url",
            fallback_keys=("conversationUrl", "conversation_url_override", "join_url", "url"),
        )

        if backend_conversation_id is not None:
            self._link_backend_conversation(
                backend_conversation_id=backend_conversation_id,
                tavus_conversation_id=conversation_id,
                visitor_name=visitor_name,
            )

        return TavusConversationSession(
            conversation_id=conversation_id,
            conversation_url=conversation_url,
        )

    async def end_conversation(self, *, conversation_id: str) -> dict[str, object]:
        return await self._client.end_conversation(conversation_id)

    def _build_conversational_context(self, *, visitor_name: str) -> str:
        normalized_visitor_name = visitor_name.strip() or "Website visitor"
        lines = [
            f"You are speaking with {normalized_visitor_name}.",
            "Use the ask_tumelo_backend tool for factual questions about Tumelo's experience, projects, skills, education, certifications, and contact details.",
            "Do not invent facts about Tumelo.",
            "Speak the backend tool response as the final answer.",
        ]
        if self._settings.public_backend_url:
            lines.append(
                f"The backend tool is served from {self._settings.public_backend_url.rstrip('/')}/api/tavus/tools/ask-tumelo."
            )
        return " ".join(lines)

    def _link_backend_conversation(
        self,
        *,
        backend_conversation_id: str,
        tavus_conversation_id: str,
        visitor_name: str,
    ) -> None:
        if not self._is_valid_uuid(backend_conversation_id):
            raise InvalidConversationIdError("conversation_id must be a valid UUID.")

        try:
            conversation = self._repository.get_conversation(backend_conversation_id)
        except ConversationRepositoryError as exc:
            raise TavusServiceError() from exc

        if conversation is None:
            raise ConversationNotFoundError("Conversation not found.")

        try:
            self._repository.update_conversation(
                conversation,
                visitor_id=tavus_conversation_id,
                title=visitor_name.strip() or conversation.title,
            )
        except ConversationRepositoryError as exc:
            raise TavusServiceError() from exc

    def _require_setting(self, value: str | None, label: str) -> str:
        if not value:
            raise TavusConfigurationError(f"{label} is not configured.")
        return value

    def _extract_string(
        self,
        payload: dict[str, object],
        key: str,
        *,
        fallback_keys: tuple[str, ...] = (),
    ) -> str:
        for candidate in (key, *fallback_keys):
            value = payload.get(candidate)
            if isinstance(value, str) and value.strip():
                return value
        raise TavusServiceError(f"Missing Tavus response field: {key}")

    def _is_valid_uuid(self, value: str) -> bool:
        try:
            import uuid

            uuid.UUID(value)
        except ValueError:
            return False
        return True
