import json
import csv
from pathlib import Path

import cv2
import numpy as np

from scripts.prepare_hardware_capture_dataset import prepare_dataset


def _write_capture(root: Path, capture_id: str, timestamp: str, sha256: str, value: int):
    day = root / "2026-08-01"
    day.mkdir(parents=True, exist_ok=True)
    image_name = f"{capture_id}.jpg"
    image = np.full((64, 80, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    (day / image_name).write_bytes(encoded.tobytes())
    metadata = {
        "capture_id": capture_id,
        "timestamp": timestamp,
        "image": {
            "path": f"2026-08-01/{image_name}",
            "sha256": sha256,
        },
        "result": {"bbox": None, "classification": None},
    }
    (day / f"{capture_id}.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_prepare_dataset_deduplicates_sha_and_writes_negative_label(tmp_path):
    captures = tmp_path / "captures"
    _write_capture(captures, "old", "2026-08-01T00:00:00+00:00", "sha-positive", 10)
    _write_capture(captures, "latest", "2026-08-01T00:00:01+00:00", "sha-positive", 20)
    _write_capture(captures, "negative", "2026-08-01T00:00:02+00:00", "sha-negative", 30)

    spec = {
        "snapshot": {"unique_count": 2, "anchors": {"1": "sha-pos", "2": "sha-neg"}},
        "labels": {"plastic": [1], "negative": [2]},
        "sources": {},
        "detector_sources": ["hardware"],
        "object_groups": {"positive": [1], "negative": [2]},
        "validation_groups": ["negative"],
        "bbox_overrides": {"1": [10, 10, 50, 50]},
    }
    spec_path = tmp_path / "audit.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    output = tmp_path / "dataset"
    report = prepare_dataset(captures, spec_path, output)

    assert report["snapshot_unique_images"] == 2
    assert report["detector_total"] == 2
    train_labels = list((output / "yolo" / "labels" / "train").glob("*.txt"))
    val_labels = list((output / "yolo" / "labels" / "val").glob("*.txt"))
    assert len(train_labels) == 1
    assert train_labels[0].read_text(encoding="utf-8").startswith("3 ")
    assert len(val_labels) == 1
    assert val_labels[0].read_text(encoding="utf-8") == ""

    resolved = json.loads((output / "resolved_audit_by_sha.json").read_text(encoding="utf-8"))
    assert set(resolved) == {"sha-positive", "sha-negative"}
    assert resolved["sha-positive"]["source_image"].endswith("latest.jpg")

    with (output / "verifier" / "hardware_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        verifier_rows = list(csv.DictReader(file))
    assert verifier_rows[0]["filepath"].startswith("train/plastic/")
    assert (output / "verifier" / verifier_rows[0]["filepath"]).is_file()
