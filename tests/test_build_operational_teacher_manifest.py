import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.build_operational_teacher_manifest import (
    ARTIFACT_NAMES,
    build_operational_teacher_manifest,
    main,
)
from scripts.audit_verifier_dataset import audit_manifest


def _image(root: Path, name: str, value: int = 80) -> tuple[Path, str]:
    path = root / f"{name}.png"
    pixels = np.full((100, 120, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), pixels)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _queue_row(path: Path, sha: str, timestamp: str, bbox=None) -> dict:
    return {
        "sha256": sha,
        "image_path": str(path),
        "timestamp": timestamp,
        "deployed": {"bbox": bbox},
    }


def _teacher_row(
    path: Path,
    sha: str,
    *,
    material: str = "paper",
    confidence: float = 0.91,
    single_object: bool = True,
    foreign_material: bool = False,
    errors: list[str] | None = None,
) -> dict:
    decision = {
        "material": material,
        "single_object": single_object,
        "foreign_material": foreign_material,
        "votes": 2,
        "pass_count": 2,
    }
    passes = [
        {
            "material": material,
            "confidence": confidence,
            "single_object": single_object,
            "foreign_material": foreign_material,
        },
        {
            "material": material,
            "confidence": confidence + 0.01 if confidence < 0.99 else confidence,
            "single_object": single_object,
            "foreign_material": foreign_material,
        },
    ]
    return {
        "sha256": sha,
        "image_path": str(path),
        "model": "qwen3-vl:8b",
        "passes": passes,
        "errors": [] if errors is None else errors,
        "consensus": True,
        "consensus_decision": decision,
        "minimum_confidence": confidence,
    }


def _build_inputs(tmp_path: Path, queue: list[dict], labels: list[dict]):
    queue_path = tmp_path / "teacher_queue.jsonl"
    labels_path = tmp_path / "teacher_labels.jsonl"
    _jsonl(queue_path, queue)
    _jsonl(labels_path, labels)
    return queue_path, labels_path


def test_accepts_positive_crop_and_preserves_negative_as_train_only_inventory(
    tmp_path,
):
    paper_path, paper_sha = _image(tmp_path, "paper", 70)
    negative_path, negative_sha = _image(tmp_path, "negative", 170)
    queue, labels = _build_inputs(
        tmp_path,
        [
            _queue_row(
                paper_path,
                paper_sha,
                "2026-08-01T01:00:00Z",
                [10, 10, 100, 90],
            ),
            _queue_row(
                negative_path,
                negative_sha,
                "2026-08-01T01:00:20Z",
                [5, 5, 110, 95],
            ),
        ],
        [
            _teacher_row(
                paper_path,
                paper_sha,
                material="paper",
                foreign_material=True,
            ),
            _teacher_row(
                negative_path,
                negative_sha,
                material="negative",
                single_object=False,
            ),
        ],
    )
    inventory = tmp_path / "capture_inventory.json"
    inventory.write_text(
        json.dumps(
            [
                {
                    "sha256": paper_sha,
                    "timestamp": "2026-08-01T01:00:00Z",
                    "request": {"client_id": "private-feedback-id"},
                },
                {
                    "sha256": negative_sha,
                    "timestamp": "2026-08-01T01:00:20Z",
                    "request": {"client_id": "private-feedback-id"},
                },
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    result = build_operational_teacher_manifest(
        teacher_queue=queue,
        teacher_labels=labels,
        capture_inventory=inventory,
        output_dir=output,
    )

    assert result["accepted"] == 2
    assert result["crop_ready_manifest_rows"] == 1
    assert result["empty_scene_inventory_rows"] == 1
    assert result["rejected_records"] == 0
    rows = list(
        csv.DictReader((output / ARTIFACT_NAMES["csv"]).open(encoding="utf-8"))
    )
    assert len(rows) == 1
    assert rows[0]["teacher_material"] == "paper"
    assert rows[0]["material"] == "2"
    assert rows[0]["category"] == "paper"
    assert rows[0]["foreign_material"] == "1"

    empty_rows = list(
        csv.DictReader(
            (output / ARTIFACT_NAMES["empty_scene_csv"]).open(encoding="utf-8")
        )
    )
    assert len(empty_rows) == 1
    negative = empty_rows[0]
    assert negative["teacher_material"] == "negative"
    assert negative["material"] == "9"
    assert negative["category"] == "background"
    assert negative["source_object_count"] == "0"
    assert negative["role"] == "train"
    assert negative["split"] == "training"
    assert negative["training_crop_ready"] == "false"
    assert negative["bbox_source"] == ""
    assert all(
        negative[field] == ""
        for field in (
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "source_bbox_x",
            "source_bbox_y",
            "source_bbox_w",
            "source_bbox_h",
        )
    )
    assert "negative_full_frame" not in json.dumps(negative)
    assert {rows[0]["object_group"], negative["object_group"]} == {
        rows[0]["object_group"]
    }
    assert {rows[0]["capture_session"], negative["capture_session"]} == {
        rows[0]["capture_session"]
    }
    rendered = "".join(
        (output / ARTIFACT_NAMES[name]).read_text(encoding="utf-8")
        for name in ("jsonl", "empty_scene_jsonl")
    )
    assert "private-feedback-id" not in rendered
    assert all(row["role"] == "train" for row in rows)
    assert all(row["blind_test_eligible"] == "false" for row in rows)
    strict_audit = audit_manifest(
        output / ARTIFACT_NAMES["csv"], require_source_references=True
    )
    assert strict_audit["missing_images"] == 0
    assert strict_audit["missing_source_images"] == 0
    assert strict_audit["invalid_source_references"] == 0
    assert strict_audit["roles"] == ["train"]
    # This intentionally tiny fixture is not a complete nine-class dataset;
    # the strict auditor's only dataset-level complaint is missing classes.
    assert all("missing material ids" in item for item in strict_audit["problems"])


def test_negative_inventory_is_train_only_but_positive_calibration_is_allowed(
    tmp_path,
):
    negative_path, negative_sha = _image(tmp_path, "negative", 170)
    queue, labels = _build_inputs(
        tmp_path,
        [
            _queue_row(
                negative_path,
                negative_sha,
                "2026-08-01T01:00:20Z",
                None,
            )
        ],
        [
            _teacher_row(
                negative_path,
                negative_sha,
                material="negative",
                single_object=False,
            )
        ],
    )

    result = build_operational_teacher_manifest(
        teacher_queue=queue,
        teacher_labels=labels,
        output_dir=tmp_path / "calibration-output",
        role="calibration",
    )

    assert result["accepted"] == 0
    assert result["empty_scene_inventory_rows"] == 0
    assert result["reason_counts"] == {"negative_source_must_be_train": 1}

    positive_path, positive_sha = _image(tmp_path, "positive", 70)
    positive_inputs = tmp_path / "positive-inputs"
    positive_inputs.mkdir()
    positive_queue, positive_labels = _build_inputs(
        positive_inputs,
        [
            _queue_row(
                positive_path,
                positive_sha,
                "2026-08-01T01:01:00Z",
                [10, 10, 100, 90],
            )
        ],
        [_teacher_row(positive_path, positive_sha, material="paper")],
    )
    positive_result = build_operational_teacher_manifest(
        teacher_queue=positive_queue,
        teacher_labels=positive_labels,
        output_dir=tmp_path / "positive-calibration-output",
        role="calibration",
    )
    assert positive_result["accepted"] == 1
    assert positive_result["crop_ready_manifest_rows"] == 1


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("errors", "teacher_errors"),
        ("low_confidence", "minimum_confidence_below_threshold"),
        ("not_single", "not_single_object"),
        ("exclude", "unsupported_teacher_material"),
        ("missing_bbox", "bbox_missing"),
        ("out_of_bounds", "bbox_out_of_bounds"),
        ("hash_mismatch", "image_sha256_mismatch"),
    ],
)
def test_reason_coded_rejections(tmp_path, mutation, reason):
    image_path, sha = _image(tmp_path, mutation)
    declared_sha = sha if mutation != "hash_mismatch" else "0" * 64
    bbox = [10, 10, 100, 90]
    label = _teacher_row(image_path, declared_sha)
    if mutation == "errors":
        label["errors"] = ["timeout"]
    elif mutation == "low_confidence":
        label = _teacher_row(image_path, declared_sha, confidence=0.79)
    elif mutation == "not_single":
        label = _teacher_row(image_path, declared_sha, single_object=False)
    elif mutation == "exclude":
        label = _teacher_row(image_path, declared_sha, material="exclude")
    elif mutation == "missing_bbox":
        bbox = None
    elif mutation == "out_of_bounds":
        bbox = [-1, 10, 100, 90]
    queue, labels = _build_inputs(
        tmp_path,
        [_queue_row(image_path, declared_sha, "2026-08-01T01:00:00Z", bbox)],
        [label],
    )

    result = build_operational_teacher_manifest(
        teacher_queue=queue,
        teacher_labels=labels,
        output_dir=tmp_path / "output",
    )

    assert result["accepted"] == 0
    assert result["reason_counts"][reason] == 1
    report = json.loads(
        ((tmp_path / "output") / ARTIFACT_NAMES["rejections"]).read_text(
            encoding="utf-8"
        )
    )
    assert reason in report["rejections"][0]["reasons"]


def test_duplicate_label_is_rejected_instead_of_selecting_a_stale_row(tmp_path):
    image_path, sha = _image(tmp_path, "duplicate")
    queue, labels = _build_inputs(
        tmp_path,
        [_queue_row(image_path, sha, "2026-08-01T01:00:00Z", [10, 10, 100, 90])],
        [_teacher_row(image_path, sha), _teacher_row(image_path, sha)],
    )
    result = build_operational_teacher_manifest(
        teacher_queue=queue,
        teacher_labels=labels,
        output_dir=tmp_path / "output",
    )
    assert result["accepted"] == 0
    assert result["reason_counts"] == {"duplicate_teacher_label_sha256": 1}


def test_privacy_redacted_adjacent_stream_frames_share_a_conservative_group(tmp_path):
    first_path, first_sha = _image(tmp_path, "first", 60)
    second_path, second_sha = _image(tmp_path, "second", 120)
    queue, labels = _build_inputs(
        tmp_path,
        [
            _queue_row(first_path, first_sha, "2026-08-01T01:00:00Z", [10, 10, 100, 90]),
            _queue_row(second_path, second_sha, "2026-08-01T01:00:30Z", [10, 10, 100, 90]),
        ],
        [_teacher_row(first_path, first_sha), _teacher_row(second_path, second_sha)],
    )
    output = tmp_path / "output"
    build_operational_teacher_manifest(
        teacher_queue=queue,
        teacher_labels=labels,
        output_dir=output,
        burst_gap_seconds=45,
    )
    rows = list(csv.DictReader((output / ARTIFACT_NAMES["csv"]).open(encoding="utf-8")))
    assert len({row["object_group"] for row in rows}) == 1
    assert {row["lineage_key_source"] for row in rows} == {"timestamp_burst"}


def test_dry_run_overwrite_guard_role_boundary_and_deterministic_outputs(tmp_path):
    image_path, sha = _image(tmp_path, "paper")
    queue, labels = _build_inputs(
        tmp_path,
        [_queue_row(image_path, sha, "2026-08-01T01:00:00Z", [10, 10, 100, 90])],
        [_teacher_row(image_path, sha)],
    )
    output = tmp_path / "output"
    preview = build_operational_teacher_manifest(
        teacher_queue=queue,
        teacher_labels=labels,
        output_dir=output,
        role="calibration",
        dry_run=True,
    )
    assert preview["accepted"] == 1
    assert not output.exists()

    first = build_operational_teacher_manifest(
        teacher_queue=queue,
        teacher_labels=labels,
        output_dir=output,
        role="calibration",
    )
    assert first["output_digests"] == preview["output_digests"]
    rows = list(csv.DictReader((output / ARTIFACT_NAMES["csv"]).open(encoding="utf-8")))
    assert rows[0]["role"] == "calibration"
    assert rows[0]["split"] == "calibration"

    with pytest.raises(FileExistsError, match="--overwrite"):
        build_operational_teacher_manifest(
            teacher_queue=queue,
            teacher_labels=labels,
            output_dir=output,
            role="calibration",
        )
    second = build_operational_teacher_manifest(
        teacher_queue=queue,
        teacher_labels=labels,
        output_dir=output,
        role="calibration",
        overwrite=True,
    )
    assert second["output_digests"] == first["output_digests"]

    with pytest.raises(ValueError, match="never be blind_test"):
        build_operational_teacher_manifest(
            teacher_queue=queue,
            teacher_labels=labels,
            output_dir=tmp_path / "blind",
            role="blind_test",
        )
    with pytest.raises(SystemExit):
        main(
            [
                "--teacher-queue",
                str(queue),
                "--teacher-labels",
                str(labels),
                "--output-dir",
                str(tmp_path / "cli-blind"),
                "--role",
                "blind_test",
            ]
        )
