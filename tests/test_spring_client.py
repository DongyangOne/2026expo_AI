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
        "https://oneexpo.kro.kr/api/v1/feedbackDetail/results",
    )
    monkeypatch.setattr(spring_client.httpx, "AsyncClient", FakeClient)

    result = DetectResponse(
        client_id="hardware-user-001",
        status=DetectionStatus.NOT_DETECTED,
    )
    asyncio.run(spring_client.notify(result))

    assert captured["json"]["client_id"] == "hardware-user-001"
    assert captured["url"] == "https://oneexpo.kro.kr/api/v1/feedbackDetail/results"
