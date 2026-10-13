import json
import os
from datetime import datetime
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


def test_snapshot_exposes_selected_llm_runtime_and_recent_cloud_usage(tmp_path):
    _write_capture(
        tmp_path,
        capture_id="runtime-sample",
        timestamp="2026-10-13T01:00:05+00:00",
        status="ALLOWED",
        bbox=[20, 20, 80, 80],
        class_name="paper",
        process_ms=1450.0,
    )
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text(json.dumps({
        "timestamp": "2026-10-13T01:00:04+00:00",
        "client_id": "runtime-sample",
        "mode": "primary",
        "selected": True,
        "reason": "llm_primary",
        "yolo": None,
        "local_llm": {
            "class_id": 2,
            "class_name": "paper",
            "runtime": {
                "backend": "openrouter",
                "model": "qwen/qwen3.8-27b",
                "provider_requested": "deepinfra",
                "provider_returned": "DeepInfra",
                "duration_ms": 910.5,
                "input_tokens": 120,
                "output_tokens": 20,
                "total_tokens": 140,
                "cost_usd": 0.0004,
            },
        },
    }) + "\n", encoding="utf-8")
    usage_path = tmp_path / "usage.jsonl"
    usage_path.write_text("\n".join([
        json.dumps({
            "timestamp": "2026-10-13T01:00:04+00:00", "kind": "classification",
            "model": "qwen/qwen3.8-27b", "provider_requested": "deepinfra",
            "provider_returned": "DeepInfra", "status": "ok", "duration_ms": 910.5,
            "input_tokens": 120, "output_tokens": 20, "total_tokens": 140,
            "cost_usd": 0.0004,
        }),
        json.dumps({
            "timestamp": "2026-10-13T01:00:05+00:00", "kind": "classification",
            "model": "qwen/qwen3.8-27b", "provider_requested": "dekallm",
            "provider_returned": None, "status": "fallback", "duration_ms": 3200,
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "cost_usd": 0,
        }),
    ]) + "\n", encoding="utf-8")

    snapshot = build_monitor_snapshot(
        tmp_path, limit=20, audit_log_path=audit_path, usage_log_path=usage_path,
        now=datetime.fromisoformat("2026-10-13T01:30:00+00:00"),
    )

    runtime = snapshot["captures"][0]["audit"]["local_llm"]["runtime"]
    assert runtime["provider_returned"] == "DeepInfra"
    usage = snapshot["cloud_usage_24h"]
    assert usage["attempts"] == 2
    assert usage["successful_attempts"] == 1
    assert usage["fallback_attempts"] == 1
    assert usage["total_tokens"] == 140
    assert usage["cost_usd"] == 0.0004
    assert usage["providers"] == {"deepinfra": 1, "dekallm": 1}


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
    assert "24시간 LLM 비용" in page.text
    assert "판정 backend" in page.text
    assert "선택 호출 비용" in page.text

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
