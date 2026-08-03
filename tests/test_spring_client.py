import asyncio
import os

os.environ.setdefault("API_KEY", "test-key")

from app.schemas.enums import DetectionStatus
from app.schemas.response import DetectResponse
from app.services import spring_client


def test_notify_forwards_client_id_unchanged(monkeypatch):
    captured: dict = {}

    class FakeResponse:
        status_code = 200
        text = ""

    class FakeClient:
        def __init__(self, timeout: float):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, json: dict):
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(spring_client.settings, "LOG_RESULTS", False)
    monkeypatch.setattr(
        spring_client.settings,
        "SPRING_CALLBACK_URL",
        "https://oneexpo.kro.kr/api/v1/feedback-detail/result",
    )
    monkeypatch.setattr(spring_client.httpx, "AsyncClient", FakeClient)

    result = DetectResponse(
        client_id="hardware-user-001",
        status=DetectionStatus.NOT_DETECTED,
    )
    asyncio.run(spring_client.notify(result))

    assert captured["json"]["client_id"] == "hardware-user-001"
    assert captured["url"] == "https://oneexpo.kro.kr/api/v1/feedback-detail/result"


def test_notify_retries_transient_server_errors(monkeypatch):
    attempts: list[int] = []
    sleeps: list[float] = []
    statuses = [503, 502, 200]

    class FakeResponse:
        text = "temporary"

        def __init__(self, status_code: int):
            self.status_code = status_code

    class FakeClient:
        def __init__(self, timeout: float):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, json: dict):
            attempts.append(len(attempts) + 1)
            return FakeResponse(statuses.pop(0))

    async def fake_sleep(seconds: float):
        sleeps.append(seconds)

    monkeypatch.setattr(spring_client.settings, "LOG_RESULTS", False)
    monkeypatch.setattr(
        spring_client.settings,
        "SPRING_CALLBACK_URL",
        "https://oneexpo.kro.kr/api/v1/feedback-detail/result",
    )
    monkeypatch.setattr(spring_client.settings, "SPRING_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(spring_client.settings, "SPRING_RETRY_BACKOFF_SEC", 0.5)
    monkeypatch.setattr(spring_client.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(spring_client.asyncio, "sleep", fake_sleep)

    result = DetectResponse(client_id="hardware-user-002", status=DetectionStatus.NOT_DETECTED)
    asyncio.run(spring_client.notify(result))

    assert attempts == [1, 2, 3]
    assert sleeps == [0.5, 1.0]


def test_notify_does_not_retry_contract_errors(monkeypatch):
    attempts = 0

    class FakeResponse:
        status_code = 400
        text = "bad request"

    class FakeClient:
        def __init__(self, timeout: float):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, json: dict):
            nonlocal attempts
            attempts += 1
            return FakeResponse()

    monkeypatch.setattr(spring_client.settings, "LOG_RESULTS", False)
    monkeypatch.setattr(
        spring_client.settings,
        "SPRING_CALLBACK_URL",
        "https://oneexpo.kro.kr/api/v1/feedback-detail/result",
    )
    monkeypatch.setattr(spring_client.settings, "SPRING_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(spring_client.httpx, "AsyncClient", FakeClient)

    result = DetectResponse(client_id="hardware-user-003", status=DetectionStatus.NOT_DETECTED)
    asyncio.run(spring_client.notify(result))

    assert attempts == 1
