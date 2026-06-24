from fastapi.testclient import TestClient

from app.api.chat import get_chat_service
from app.main import app
from app.services.llm_service import LLMServiceError


class FakeChatService:
    def __init__(self, reply: str = "Mocked assistant response.") -> None:
        self.reply = reply

    async def generate_reply(self, message: str) -> str:
        return self.reply


class FailingChatService:
    async def generate_reply(self, message: str) -> str:
        raise LLMServiceError("sk-test-should-not-leak")


client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_accepts_valid_message() -> None:
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService(
        "Tumelo has worked on AI systems."
    )

    response = client.post("/chat", json={"message": "Tell me about Tumelo's AI projects"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"response": "Tumelo has worked on AI systems."}


def test_chat_rejects_empty_message() -> None:
    response = client.post("/chat", json={"message": ""})

    assert response.status_code == 422


def test_chat_rejects_whitespace_only_message() -> None:
    response = client.post("/chat", json={"message": "   "})

    assert response.status_code == 400
    assert response.json() == {"detail": "Chat message cannot be empty."}


def test_llm_failures_return_safe_error_response() -> None:
    app.dependency_overrides[get_chat_service] = lambda: FailingChatService()

    response = client.post("/chat", json={"message": "Hello"})

    app.dependency_overrides.clear()

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Unable to generate assistant response. Please try again."
    }
    assert "sk-test-should-not-leak" not in response.text
