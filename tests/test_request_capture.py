import json
import os

os.environ.setdefault("API_KEY", "test-key")

from app.schemas.enums import DetectionStatus
from app.schemas.response import DetectResponse
from app.services import request_capture


def test_save_capture_writes_paired_image_and_result_json(tmp_path, monkeypatch):
    capture_dir = tmp_path / "captures"
    image_bytes = b"\xff\xd8\xffsample-jpeg"
    result = DetectResponse(
        client_id="hardware/user-001",
        status=DetectionStatus.NOT_DETECTED,
    )

    monkeypatch.setattr(request_capture.settings, "CAPTURE_DIR", str(capture_dir))
    monkeypatch.setattr(request_capture.settings, "CAPTURE_MAX_IMAGE_BYTES", 1024)
    monkeypatch.setattr(request_capture.settings, "CAPTURE_RETENTION_DAYS", 90)
    monkeypatch.setattr(request_capture.settings, "CAPTURE_MAX_STORAGE_MB", 10)
    monkeypatch.setattr(request_capture, "_last_prune_at", 0.0)

    request_capture.save_capture(
        image_bytes=image_bytes,
        original_filename="../camera/sample.jpg",
        content_type="image/jpeg",
        client_id="hardware/user-001",
        weight_g=28.0,
        result=result,
    )

    images = list(capture_dir.rglob("*.jpg"))
    metadata_files = list(capture_dir.rglob("*.json"))
    assert len(images) == 1
    assert len(metadata_files) == 1
    assert images[0].stem == metadata_files[0].stem
    assert images[0].read_bytes() == image_bytes

    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    assert metadata["request"] == {
        "client_id": "hardware/user-001",
        "weight_g": 28.0,
    }
    assert metadata["result"]["client_id"] == "hardware/user-001"
    assert metadata["result"]["status"] == "NOT_DETECTED"
    assert metadata["image"]["original_filename"] == "sample.jpg"
    assert metadata["review"] == {
        "is_correct": None,
        "expected_class": None,
        "is_dented": None,
        "has_label": None,
        "has_foreign_material": None,
        "notes": None,
    }
    assert "api_key" not in json.dumps(metadata).lower()


def test_save_capture_skips_oversized_image(tmp_path, monkeypatch):
    capture_dir = tmp_path / "captures"
    monkeypatch.setattr(request_capture.settings, "CAPTURE_DIR", str(capture_dir))
    monkeypatch.setattr(request_capture.settings, "CAPTURE_MAX_IMAGE_BYTES", 3)

    request_capture.save_capture(
        image_bytes=b"1234",
        original_filename="large.jpg",
        content_type="image/jpeg",
        client_id="client-1",
        weight_g=None,
        result=DetectResponse(
            client_id="client-1",
            status=DetectionStatus.NOT_DETECTED,
        ),
    )

    assert not capture_dir.exists()
