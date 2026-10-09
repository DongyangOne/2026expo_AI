import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

os.environ.setdefault("API_KEY", "test-key")

from app.api import monitor
from app.services.monitoring import build_monitor_snapshot


def _write_capture(
    root: Path,
    *,
    capture_id: str,
    timestamp: str,
    status: str,
    bbox: list[float] | None,
    class_name: str | None,
    process_ms: float | None,
) -> None:
    day = root / timestamp[:10]
    day.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), "white").save(day / f"{capture_id}.jpg")
    classification = None
    if class_name is not None:
        classification = {"class_id": 3, "class_name": class_name, "confidence": 0.95}
    metadata = {
        "capture_id": capture_id,
        "timestamp": timestamp,
        "image": {"path": f"{timestamp[:10]}/{capture_id}.jpg"},
        "request": {"client_id": capture_id, "weight_g": 10.0},
        "result": {
            "client_id": capture_id,
            "status": status,
            "classification": classification,
            "conditions": {},
            "weight": {"value_g": 10.0, "anomaly": False},
            "guidance": [],
            "bbox": bbox,
        },
        "metrics": {"process_ms": process_ms} if process_ms is not None else {},
    }
    (day / f"{capture_id}.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def test_snapshot_reports_detection_latency_and_suspicious_bbox(tmp_path):
    _write_capture(
        tmp_path,
        capture_id="wide-box",
        timestamp="2026-10-09T01:00:00+00:00",
        status="ALLOWED",
        bbox=[0, 0, 100, 100],
        class_name="plastic",
        process_ms=3200.0,
    )
    _write_capture(
        tmp_path,
        capture_id="missed",
        timestamp="2026-10-09T01:01:00+00:00",
        status="NOT_DETECTED",
        bbox=None,
        class_name=None,
        process_ms=2800.0,
    )

    snapshot = build_monitor_snapshot(tmp_path, limit=20)

    assert snapshot["summary"]["total_requests"] == 2
    assert snapshot["summary"]["bbox_detection_rate"] == 50.0
    assert snapshot["summary"]["not_detected_count"] == 1
    assert snapshot["summary"]["suspicious_bbox_count"] == 1
    assert snapshot["summary"]["latency_ms"]["median"] == 3000.0
    assert snapshot["summary"]["latency_ms"]["p95"] == 3200.0
    assert snapshot["captures"][1]["bbox_quality"]["area_ratio"] == 1.0
    assert snapshot["captures"][1]["bbox_quality"]["suspicious"] is True


def test_monitor_page_and_data_are_public(tmp_path, monkeypatch):
    _write_capture(
        tmp_path,
        capture_id="sample",
        timestamp="2026-10-09T01:00:00+00:00",
        status="ALLOWED",
        bbox=[20, 20, 80, 80],
        class_name="plastic",
        process_ms=3000.0,
    )
    monkeypatch.setattr(monitor.settings, "CAPTURE_DIR", str(tmp_path))
    test_app = FastAPI()
    test_app.include_router(monitor.router)
    client = TestClient(test_app)

    page = client.get("/monitor")
    assert page.status_code == 200
    assert "EXPO AI 관제" in page.text
    assert "카메라가 무엇을 보고" not in page.text
    assert "X-API-Key" not in page.text
    assert "blobs: new Map()" in page.text

    allowed = client.get("/api/v1/monitor/summary")
    assert allowed.status_code == 200
    assert allowed.json()["summary"]["total_requests"] == 1


def test_monitor_image_is_public_and_resolved_from_capture_index(
    tmp_path, monkeypatch
):
    _write_capture(
        tmp_path,
        capture_id="sample",
        timestamp="2026-10-09T01:00:00+00:00",
        status="ALLOWED",
        bbox=[20, 20, 80, 80],
        class_name="plastic",
        process_ms=3000.0,
    )
    monkeypatch.setattr(monitor.settings, "CAPTURE_DIR", str(tmp_path))
    test_app = FastAPI()
    test_app.include_router(monitor.router)
    client = TestClient(test_app)

    response = client.get("/api/v1/monitor/captures/sample/image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

    missing = client.get("/api/v1/monitor/captures/not-present/image")
    assert missing.status_code == 404
