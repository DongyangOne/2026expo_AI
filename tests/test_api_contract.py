import os

os.environ.setdefault("API_KEY", "test-key")

from fastapi.testclient import TestClient

from app.api.v1 import detect
from app.main import app
from app.schemas.enums import DetectionStatus
from app.schemas.response import DetectResponse
from app.services import pipeline, request_capture, spring_client


def test_detect_contract_requires_and_returns_client_id():
    schema = app.openapi()
    operation = schema["paths"]["/api/v1/detect"]["post"]
    request_ref = operation["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
    request_schema = schema["components"]["schemas"][request_ref.rsplit("/", 1)[-1]]
    response_schema = schema["components"]["schemas"]["DetectResponse"]

    assert "client_id" in request_schema["required"]
    assert request_schema["properties"]["client_id"]["minLength"] == 1
    assert "client_id" in response_schema["required"]


def test_ai_response_contract_matches_spring_guidance_and_conditions():
    schemas = app.openapi()["components"]["schemas"]

    assert schemas["GuidanceCode"]["enum"] == [
        "EMPTY_CONTENTS",
        "REMOVE_LABEL",
        "COMPRESS",
        "REMOVE_FOREIGN_MATERIAL",
    ]
    assert set(schemas["Conditions"]["properties"]) == {"has_label", "is_dented"}
    assert "anyOf" not in schemas["Conditions"]["properties"]["has_label"]

    description = app.openapi()["paths"]["/api/v1/detect"]["post"]["description"]
    assert "해당 판별 모델이 탑재된 경우에만 검사" in description
    assert "has_foreign_material`은 반환하지 않" in description
    assert "REMOVE_FOREIGN_MATERIAL" in description


def test_detect_response_omits_spring_optional_null_fields(monkeypatch):
    app.dependency_overrides[detect._get_registry] = lambda: object()

    async def fake_run(image, weight_g, client_id, registry):
        return DetectResponse(client_id=client_id, status=DetectionStatus.NOT_DETECTED)

    async def fake_notify(result):
        return None

    monkeypatch.setattr(pipeline, "run", fake_run)
    monkeypatch.setattr(spring_client, "notify", fake_notify)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/v1/detect",
            headers={"X-API-Key": "test-key"},
            files={"image": ("sample.jpg", b"image", "image/jpeg")},
            data={"client_id": "spring-null-contract"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "client_id": "spring-null-contract",
        "status": "NOT_DETECTED",
        "conditions": {},
        "weight": {"anomaly": False},
        "guidance": [],
    }


def test_detect_echoes_client_id_to_response_and_callback(monkeypatch):
    captured: dict = {}
    background_order: list[str] = []
    image_bytes = b"\xff\xd8\xffcaptured-image"

    async def fake_run(image, weight_g, client_id, registry):
        captured["pipeline_client_id"] = client_id
        return DetectResponse(client_id=client_id, status=DetectionStatus.NOT_DETECTED)

    async def fake_notify(result):
        background_order.append("callback")
        captured["callback_client_id"] = result.client_id

    def fake_save_capture(**kwargs):
        background_order.append("capture")
        captured["capture_client_id"] = kwargs["client_id"]
        captured["capture_image"] = kwargs["image_bytes"]
        captured["capture_result_client_id"] = kwargs["result"].client_id

    monkeypatch.setattr(pipeline, "run", fake_run)
    monkeypatch.setattr(spring_client, "notify", fake_notify)
    monkeypatch.setattr(request_capture, "save_capture", fake_save_capture)
    monkeypatch.setattr(detect.settings, "CAPTURE_REQUESTS", True)
    app.dependency_overrides[detect._get_registry] = lambda: object()

    try:
        client = TestClient(app)
        response = client.post(
            "/api/v1/detect",
            headers={"X-API-Key": "test-key"},
            files={"image": ("sample.jpg", image_bytes, "image/jpeg")},
            data={"client_id": "hardware-user-001", "weight_g": "28.0"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["client_id"] == "hardware-user-001"
    assert captured == {
        "pipeline_client_id": "hardware-user-001",
        "callback_client_id": "hardware-user-001",
        "capture_client_id": "hardware-user-001",
        "capture_image": image_bytes,
        "capture_result_client_id": "hardware-user-001",
    }
    assert background_order == ["capture", "callback"]


def test_detect_rejects_missing_client_id():
    app.dependency_overrides[detect._get_registry] = lambda: object()
    try:
        client = TestClient(app)
        response = client.post(
            "/api/v1/detect",
            headers={"X-API-Key": "test-key"},
            files={"image": ("sample.jpg", b"not-read", "image/jpeg")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_detect_rejects_negative_sensor_weight():
    app.dependency_overrides[detect._get_registry] = lambda: object()
    try:
        client = TestClient(app)
        response = client.post(
            "/api/v1/detect",
            headers={"X-API-Key": "test-key"},
            files={"image": ("sample.jpg", b"not-read", "image/jpeg")},
            data={"client_id": "hardware-user-001", "weight_g": "-1"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
