import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

import scripts.prepare_operational_capture_queue as capture_queue
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
    hash_mismatch_sha = _capture(
        captures,
        "hash-mismatch",
        "2026-08-01T00:00:03+09:00",
        "hash-client",
        value=104,
        declared_sha256="0" * 64,
    )
    invalid_sha = _capture(
        captures,
        "invalid-sha",
        "2026-08-01T00:00:03.500000+09:00",
        "invalid-client",
        value=105,
        declared_sha256="G" * 64,
    )
    unreadable_sha = _capture(
        captures,
        "unreadable",
        "2026-08-01T00:00:04+09:00",
        "unreadable-client",
        readable=False,
    )
    unreadable_mismatch_sha = _capture(
        captures,
        "unreadable-mismatch",
        "2026-08-01T00:00:04.500000+09:00",
        "unreadable-mismatch-client",
        readable=False,
        declared_sha256="0" * 64,
    )
    unreadable_mismatch_path = (
        captures / "2026-08-01" / "unreadable-mismatch.jpg"
    )
    unreadable_mismatch_path.write_bytes(b"different-not-an-image")
    unreadable_mismatch_sha = hashlib.sha256(
        unreadable_mismatch_path.read_bytes()
    ).hexdigest()
    tiny_sha = _capture(
        captures,
        "tiny",
        "2026-08-01T00:00:05+09:00",
        "tiny-client",
        size=(60, 80),
    )
    black_sha = _capture(
        captures, "black", "2026-08-01T00:00:06+09:00", "black-client", value=0
    )
    white_sha = _capture(
        captures, "white", "2026-08-01T00:00:07+09:00", "white-client", value=255
    )
    tiny_black_sha = _capture(
        captures,
        "tiny-black",
        "2026-08-01T00:00:08+09:00",
        "tiny-black-client",
        value=0,
        size=(60, 80),
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
        "image_extreme_underexposure": 2,
        "invalid_image_sha256": 1,
        "image_missing": 1,
        "image_resolution_below_minimum": 2,
        "image_sha256_mismatch": 2,
        "image_unreadable": 2,
    }
    objective_rows = [
        json.loads(line)
        for line in (output / capture_queue.OBJECTIVE_REJECTIONS_FILE)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [row["source_sha256"] for row in objective_rows] == sorted(
        row["source_sha256"] for row in objective_rows
    )
    assert all(
        set(row)
        == {
            "schema_version",
            "source_sha256",
            "capture_timestamp_utc",
            "metadata_ref",
            "metadata_sha256",
            "image_ref",
            "raw_reasons",
            "quality_reason",
        }
        for row in objective_rows
    )
    by_sha = {row["source_sha256"]: row for row in objective_rows}
    assert {
        sha: by_sha[sha]["quality_reason"]
        for sha in (
            unreadable_sha,
            tiny_sha,
            black_sha,
            white_sha,
            tiny_black_sha,
        )
    } == {
        unreadable_sha: "objective_unreadable",
        tiny_sha: "resolution_too_low",
        black_sha: "extreme_exposure",
        white_sha: "extreme_exposure",
        # Resolution has higher fixed priority than exposure when both fire.
        tiny_black_sha: "resolution_too_low",
    }
    assert by_sha[unreadable_sha]["raw_reasons"] == ["image_unreadable"]
    assert by_sha[tiny_sha]["raw_reasons"] == [
        "image_resolution_below_minimum"
    ]
    assert by_sha[black_sha]["raw_reasons"] == [
        "image_extreme_underexposure"
    ]
    assert by_sha[white_sha]["raw_reasons"] == [
        "image_extreme_overexposure"
    ]
    assert by_sha[tiny_black_sha]["raw_reasons"] == [
        "image_extreme_underexposure",
        "image_resolution_below_minimum",
    ]
    # Integrity failures are counted but are never promoted to trusted quality
    # exclusions, even when an objective detector also fires.
    assert {
        missing_sha,
        hash_mismatch_sha,
        invalid_sha,
        unreadable_mismatch_sha,
    }.isdisjoint(by_sha)
    assert summary["objective_quality_rejections"] == 5
    assert summary["objective_quality_reason_counts"] == {
        "extreme_exposure": 2,
        "objective_unreadable": 1,
        "resolution_too_low": 2,
    }
    receipt = json.loads(
        (output / capture_queue.OBJECTIVE_RECEIPT_FILE).read_text(encoding="utf-8")
    )
    assert {path.name for path in output.iterdir()} == set(
        capture_queue.OUTPUT_FILES.values()
    )
    capture_index = [
        json.loads(line)
        for line in (output / capture_queue.OBJECTIVE_CAPTURE_INDEX_FILE)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(capture_index) == 12
    assert [row["metadata_ref"] for row in capture_index] == sorted(
        row["metadata_ref"] for row in capture_index
    )
    assert all(
        set(row) == {"schema_version", "metadata_ref", "metadata_sha256"}
        and row["schema_version"] == 1
        and not Path(row["metadata_ref"]).is_absolute()
        for row in capture_index
    )
    assert summary["capture_metadata_index_rows"] == len(capture_index)
    assert receipt["counts"]["capture_metadata_index_rows"] == len(capture_index)
    assert receipt["counts"]["objective_quality_rejections"] == 5
    assert receipt["counts"]["objective_quality_reason_counts"] == summary[
        "objective_quality_reason_counts"
    ]
    assert receipt["quality_policy"]["objective_reason_priority"] == [
        {"raw_reason": raw, "quality_reason": reason}
        for raw, reason in capture_queue.OBJECTIVE_REASON_PRIORITY
    ]
    assert receipt["privacy"] == {
        "objective_evidence_structured_client_id_fields_exported": False,
        "objective_evidence_structured_device_id_fields_exported": False,
        "objective_evidence_prediction_outputs_exported": False,
        "objective_evidence_absolute_paths_exported": False,
        "objective_evidence_untrusted_relative_local_refs_present": True,
        "objective_evidence_relative_refs_may_contain_identifiers": True,
    }
    evidence_text = (output / capture_queue.OBJECTIVE_REJECTIONS_FILE).read_text(
        encoding="utf-8"
    )
    for private_value in (
        "good-client",
        "unreadable-client",
        "tiny-client",
        "black-client",
        "white-client",
        "tiny-black-client",
    ):
        assert private_value not in evidence_text
    assert str(captures.resolve()) not in evidence_text
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


def test_prepare_rejects_duplicate_objective_source_sha(tmp_path):
    captures = tmp_path / "captures"
    first_sha = _capture(
        captures,
        "tiny-a",
        "2026-08-01T00:00:00+09:00",
        "private-a",
        size=(60, 80),
    )
    second_sha = _capture(
        captures,
        "tiny-b",
        "2026-08-01T00:00:01+09:00",
        "private-b",
        size=(60, 80),
    )
    assert first_sha == second_sha
    shadow = tmp_path / "shadow.jsonl"
    shadow.write_text("", encoding="utf-8")
    known = tmp_path / "known.json"
    known.write_text("{}", encoding="utf-8")
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="duplicate objective-quality source SHA"):
        prepare_queue(
            captures_dir=captures,
            shadow_log=shadow,
            known_audit=known,
            output_dir=output,
            start_kst=capture_queue.OPERATIONAL_CAPTURE_CUTOFF_KST,
        )
    assert not output.exists()


def test_prepare_rehashes_capture_source_before_publish(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    captures = tmp_path / "captures"
    _capture(
        captures,
        "good",
        "2026-08-01T00:00:00+09:00",
        "private",
    )
    source = (captures / "2026-08-01" / "good.jpg").resolve()
    shadow = tmp_path / "shadow.jsonl"
    shadow.write_text("", encoding="utf-8")
    known = tmp_path / "known.json"
    known.write_text("{}", encoding="utf-8")
    output = tmp_path / "output"
    real_stable = capture_queue._stable_regular_bytes
    source_reads = 0

    def mutate_on_final_source_rehash(path: Path, *, description: str):
        nonlocal source_reads
        if path.resolve() == source:
            source_reads += 1
            if source_reads == 2:
                source.write_bytes(b"changed-before-publication")
        return real_stable(path, description=description)

    monkeypatch.setattr(
        capture_queue, "_stable_regular_bytes", mutate_on_final_source_rehash
    )

    with pytest.raises(RuntimeError, match="capture image changed before queue publication"):
        prepare_queue(
            captures_dir=captures,
            shadow_log=shadow,
            known_audit=known,
            output_dir=output,
            start_kst=capture_queue.OPERATIONAL_CAPTURE_CUTOFF_KST,
        )
    assert source_reads == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".output.*"))


def test_prepare_never_replaces_racing_destination(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    captures = tmp_path / "captures"
    _capture(
        captures,
        "good",
        "2026-08-01T00:00:00+09:00",
        "private",
    )
    shadow = tmp_path / "shadow.jsonl"
    shadow.write_text("", encoding="utf-8")
    known = tmp_path / "known.json"
    known.write_text("{}", encoding="utf-8")
    output = tmp_path / "output"
    real_publish = capture_queue._publish_directory_no_replace

    def collide(staging: Path, destination: Path) -> None:
        assert destination == output
        destination.mkdir()
        (destination / "sentinel.txt").write_text(
            "do-not-replace", encoding="utf-8"
        )
        real_publish(staging, destination)

    monkeypatch.setattr(capture_queue, "_publish_directory_no_replace", collide)

    with pytest.raises(FileExistsError, match="overwrite immutable output"):
        prepare_queue(
            captures_dir=captures,
            shadow_log=shadow,
            known_audit=known,
            output_dir=output,
            start_kst=capture_queue.OPERATIONAL_CAPTURE_CUTOFF_KST,
        )
    assert {
        path.name: path.read_text(encoding="utf-8") for path in output.iterdir()
    } == {"sentinel.txt": "do-not-replace"}
    assert not list(tmp_path.glob(".output.*"))
