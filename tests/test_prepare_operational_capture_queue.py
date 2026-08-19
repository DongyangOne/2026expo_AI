import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.prepare_operational_capture_queue import prepare_queue


def _capture(root: Path, capture_id: str, timestamp: str, sha256: str, client_id: str):
    day = root / timestamp[:10]
    day.mkdir(parents=True, exist_ok=True)
    image = day / f"{capture_id}.jpg"
    image.write_bytes(b"jpeg")
    metadata = {
        "timestamp": timestamp,
        "image": {"path": f"{timestamp[:10]}/{capture_id}.jpg", "sha256": sha256},
        "request": {"client_id": client_id},
        "result": {
            "status": "ALLOWED",
            "classification": {"class_name": "plastic", "confidence": 0.9},
            "bbox": [1, 2, 3, 4],
        },
    }
    (day / f"{capture_id}.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_queue_keeps_known_split_and_redacts_client_ids(tmp_path):
    captures = tmp_path / "captures"
    _capture(captures, "known", "2026-07-31T15:01:00+00:00", "known-sha", "private-a")
    _capture(captures, "new", "2026-08-01T01:00:00+00:00", "new-sha", "private-b")
    shadow = tmp_path / "shadow.jsonl"
    shadow.write_text(
        json.dumps(
            {
                "timestamp": "2026-08-01T01:00:01+00:00",
                "client_id": "private-b",
                "material_agreement": False,
                "verifier": {"material": {"class_name": "vinyl", "confidence": 0.8}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    known = tmp_path / "known.json"
    known.write_text(
        json.dumps({"known-sha": {"split": "val", "label": "plastic"}}),
        encoding="utf-8",
    )

    output = tmp_path / "output"
    summary = prepare_queue(
        captures_dir=captures,
        shadow_log=shadow,
        known_audit=known,
        output_dir=output,
        start_kst=datetime(2026, 8, 1, tzinfo=timezone(timedelta(hours=9))),
    )

    assert summary["decisions"] == {"protected_validation": 1, "teacher_required": 1}
    queue = (output / "teacher_queue.jsonl").read_text(encoding="utf-8")
    assert "new-sha" in queue
    assert "private-a" not in queue
    assert "private-b" not in queue
