from fastapi import Depends

from app.api.dependencies.chat_dependencies import get_chat_repository
from app.api.dependencies.common_dependencies import get_app_settings
from app.config import Settings
from app.infrastructure.tavus import TavusClient
from app.repositories import ConversationRepository
from app.services.tavus import TavusService


def get_tavus_client(settings: Settings = Depends(get_app_settings)) -> TavusClient:
    return TavusClient(settings=settings)


def get_tavus_service(
    settings: Settings = Depends(get_app_settings),
    client: TavusClient = Depends(get_tavus_client),
    repository: ConversationRepository = Depends(get_chat_repository),
) -> TavusService:
    return TavusService(
        settings=settings,
        client=client,
        repository=repository,
    )
