import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.prepare_operational_capture_queue import prepare_queue


def _capture(
    root: Path,
    capture_id: str,
    timestamp: str,
    client_id: str,
    *,
    value: int = 100,
    size: tuple[int, int] = (240, 320),
    readable: bool = True,
    declared_sha256: str | None = None,
) -> str:
    day = root / timestamp[:10]
    day.mkdir(parents=True, exist_ok=True)
    image = day / f"{capture_id}.jpg"
    if readable:
        pixels = np.full((*size, 3), value, dtype=np.uint8)
        assert cv2.imwrite(str(image), pixels)
    else:
        image.write_bytes(b"not-an-image")
    actual_sha256 = hashlib.sha256(image.read_bytes()).hexdigest()
    metadata = {
        "timestamp": timestamp,
        "image": {
            "path": f"{timestamp[:10]}/{capture_id}.jpg",
            "sha256": declared_sha256 or actual_sha256,
        },
        "request": {"client_id": client_id},
        "result": {
            "status": "ALLOWED",
            "classification": {"class_name": "plastic", "confidence": 0.9},
            "bbox": [1, 2, 3, 4],
        },
    }
    (day / f"{capture_id}.json").write_text(json.dumps(metadata), encoding="utf-8")
    return actual_sha256


def test_queue_keeps_known_split_and_redacts_client_ids(tmp_path):
    captures = tmp_path / "captures"
    known_sha = _capture(
        captures, "known", "2026-07-31T15:01:00+00:00", "private-a", value=90
    )
    new_sha = _capture(
        captures, "new", "2026-08-01T01:00:00+00:00", "private-b", value=110
    )
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
        json.dumps({known_sha: {"split": "validation", "label": "plastic"}}),
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
    assert new_sha in queue
    assert known_sha not in queue
    assert "private-a" not in queue
    assert "private-b" not in queue
    queue_row = json.loads(queue)
    assert queue_row == {
        "sha256": new_sha,
        "timestamp": "2026-08-01T01:00:00+00:00",
        "image_ref": "2026-08-01/new.jpg",
        "decision": "teacher_required",
    }
    assert not Path(queue_row["image_ref"]).is_absolute()
    assert "deployed" not in queue_row
    assert "verifier" not in queue_row
    assert summary["operational_capture_cutoff_kst"] == "2026-08-01T00:00:00+09:00"
    assert summary["quality_policy"]["blur_filter_enabled"] is False
    assert summary["quality_policy"]["deployed_prediction_filter_enabled"] is False


def test_start_kst_cannot_override_the_fixed_floor(tmp_path):
    with pytest.raises(ValueError, match="cannot be earlier"):
        prepare_queue(
            captures_dir=tmp_path / "captures",
            shadow_log=tmp_path / "shadow.jsonl",
            known_audit=tmp_path / "known.json",
            output_dir=tmp_path / "output",
            start_kst=datetime(2026, 7, 31, 23, 59, tzinfo=timezone(timedelta(hours=9))),
        )

    with pytest.raises(ValueError, match="explicit UTC offset"):
        prepare_queue(
            captures_dir=tmp_path / "captures",
            shadow_log=tmp_path / "shadow.jsonl",
            known_audit=tmp_path / "known.json",
            output_dir=tmp_path / "output",
            start_kst=datetime(2026, 8, 1),
        )


def test_cutoff_naive_timestamp_and_objective_bad_captures_are_excluded(tmp_path):
    captures = tmp_path / "captures"
    good_sha = _capture(
        captures, "good-uniform", "2026-08-01T00:00:00+09:00", "good-client"
    )
    good_metadata_path = captures / "2026-08-01" / "good-uniform.json"
    good_metadata = json.loads(good_metadata_path.read_text(encoding="utf-8"))
    good_metadata["result"] = {
        "status": "NOT_DETECTED",
        "classification": {"class_name": "glass", "confidence": 0.0},
        "bbox": None,
    }
    good_metadata_path.write_text(json.dumps(good_metadata), encoding="utf-8")
    _capture(
        captures, "before", "2026-07-31T14:59:59Z", "old-client", value=101
    )
    _capture(captures, "naive", "2026-08-01T00:00:01", "naive-client", value=102)
    missing_sha = _capture(
        captures, "missing", "2026-08-01T00:00:02+09:00", "missing-client", value=103
    )
    (captures / "2026-08-01" / "missing.jpg").unlink()
    _capture(
        captures,
        "hash-mismatch",
        "2026-08-01T00:00:03+09:00",
        "hash-client",
        value=104,
        declared_sha256="0" * 64,
    )
    _capture(
        captures,
        "unreadable",
        "2026-08-01T00:00:04+09:00",
        "unreadable-client",
        readable=False,
    )
    _capture(
        captures,
        "tiny",
        "2026-08-01T00:00:05+09:00",
        "tiny-client",
        size=(60, 80),
    )
    _capture(
        captures, "black", "2026-08-01T00:00:06+09:00", "black-client", value=0
    )
    _capture(
        captures, "white", "2026-08-01T00:00:07+09:00", "white-client", value=255
    )
    shadow = tmp_path / "shadow.jsonl"
    shadow.write_text("", encoding="utf-8")
    known = tmp_path / "known.json"
    known.write_text("{}", encoding="utf-8")

    output = tmp_path / "output"
    summary = prepare_queue(
        captures_dir=captures,
        shadow_log=shadow,
        known_audit=known,
        output_dir=output,
        start_kst=datetime(2026, 8, 1, tzinfo=timezone(timedelta(hours=9))),
    )

    queue_rows = [
        json.loads(line)
        for line in (output / "teacher_queue.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["sha256"] for row in queue_rows] == [good_sha]
    assert missing_sha not in {row["sha256"] for row in queue_rows}
    assert summary["capture_rejection_counts"] == {
        "capture_timestamp_missing_invalid_or_naive": 1,
        "image_extreme_overexposure": 1,
        "image_extreme_underexposure": 1,
        "image_missing": 1,
        "image_resolution_below_minimum": 1,
        "image_sha256_mismatch": 1,
        "image_unreadable": 1,
    }
    # A uniformly mid-tone image is intentionally retained: there is no
    # camera-specific blur threshold and deployed predictions are not a gate.
    assert summary["teacher_queue"] == 1


@pytest.mark.parametrize("bad_path", ["../outside.jpg", "C:/outside.jpg", None])
def test_queue_rejects_escaping_or_invalid_image_refs(tmp_path, bad_path):
    captures = tmp_path / "captures"
    _capture(captures, "bad", "2026-08-01T00:00:00+09:00", "private")
    metadata_path = captures / "2026-08-01" / "bad.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["image"]["path"] = bad_path
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    shadow = tmp_path / "shadow.jsonl"
    shadow.write_text("", encoding="utf-8")
    known = tmp_path / "known.json"
    known.write_text("{}", encoding="utf-8")

    summary = prepare_queue(
        captures_dir=captures,
        shadow_log=shadow,
        known_audit=known,
        output_dir=tmp_path / "output",
        start_kst=datetime(2026, 8, 1, tzinfo=timezone(timedelta(hours=9))),
    )

    assert summary["teacher_queue"] == 0
    assert summary["capture_rejection_counts"] == {
        "image_path_invalid_or_outside_capture_root": 1
    }


def test_queue_rejects_symlink_that_resolves_outside_capture_root(tmp_path):
    captures = tmp_path / "captures"
    captures.mkdir()
    outside = tmp_path / "outside.jpg"
    pixels = np.full((240, 320, 3), 100, dtype=np.uint8)
    assert cv2.imwrite(str(outside), pixels)
    link = captures / "linked.jpg"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is not available")
    sha = hashlib.sha256(outside.read_bytes()).hexdigest()
    (captures / "capture.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-08-01T00:00:00+09:00",
                "image": {"path": "linked.jpg", "sha256": sha},
                "request": {"client_id": "private"},
            }
        ),
        encoding="utf-8",
    )
    shadow = tmp_path / "shadow.jsonl"
    shadow.write_text("", encoding="utf-8")
    known = tmp_path / "known.json"
    known.write_text("{}", encoding="utf-8")

    summary = prepare_queue(
        captures_dir=captures,
        shadow_log=shadow,
        known_audit=known,
        output_dir=tmp_path / "output",
        start_kst=datetime(2026, 8, 1, tzinfo=timezone(timedelta(hours=9))),
    )

    assert summary["teacher_queue"] == 0
    assert summary["capture_rejection_counts"] == {
        "image_path_invalid_or_outside_capture_root": 1
    }
