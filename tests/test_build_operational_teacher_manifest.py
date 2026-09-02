import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.build_operational_teacher_manifest import (
    ARTIFACT_NAMES,
    TEACHER_LABEL_SCHEMA_VERSION,
    build_operational_teacher_manifest as _real_build_operational_teacher_manifest,
    main,
)
from scripts.audit_verifier_dataset import audit_manifest
import scripts.label_operational_captures_ollama as teacher
from scripts.build_independent_localization_consensus import (
    AGGREGATE_METHOD,
    AGGREGATE_TOLERANCE,
    CONTRACT_SCHEMA_VERSION,
    IOU_THRESHOLD,
    LOCALIZATION_SCHEMA_VERSION,
    bbox_iou,
    canonical_json as localization_canonical_json,
    provider_output_core,
    sha256_bytes as localization_sha256_bytes,
)


def _image(
    root: Path,
    name: str,
    value: int = 80,
    *,
    size: tuple[int, int] = (240, 320),
    readable: bool = True,
) -> tuple[Path, str]:
    path = root / f"{name}.png"
    if readable:
        pixels = np.full((*size, 3), value, dtype=np.uint8)
        assert cv2.imwrite(str(path), pixels)
    else:
        path.write_bytes(b"not-an-image")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _queue_row(path: Path, sha: str, timestamp: str, bbox=None) -> dict:
    return {
        "sha256": sha,
        "image_ref": path.name,
        "timestamp": timestamp,
        "decision": "teacher_required",
    }


def _independent_localization(sha: str, bbox=None) -> dict:
    first_box = [10.0, 10.0, 100.0, 90.0]
    aggregate_box = first_box if bbox is None else [float(x) for x in bbox]
    second_box = list(first_box)
    first_core = provider_output_core(
        provider="detector_a", source_sha=sha, box=first_box,
        model_sha="1" * 64, spec_sha="2" * 64,
    )
    second_core = provider_output_core(
        provider="segmenter_b", source_sha=sha, box=second_box,
        model_sha="3" * 64, spec_sha="4" * 64,
    )
    providers = [
        dict(first_core, provider_output_sha256=localization_sha256_bytes(
            localization_canonical_json(first_core).encode("utf-8")
        )),
        dict(second_core, provider_output_sha256=localization_sha256_bytes(
            localization_canonical_json(second_core).encode("utf-8")
        )),
    ]
    evidence = [
        {
            "provider": "detector_a", "manifest_sha256": "5" * 64,
            "model_sha256": "1" * 64, "inference_spec_sha256": "2" * 64,
        },
        {
            "provider": "segmenter_b", "manifest_sha256": "6" * 64,
            "model_sha256": "3" * 64, "inference_spec_sha256": "4" * 64,
        },
    ]
    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "method": AGGREGATE_METHOD,
        "iou_threshold": IOU_THRESHOLD,
        "aggregate_tolerance": AGGREGATE_TOLERANCE,
        "source_image_sha256": sha,
        "providers": evidence,
    }
    return {
        "schema_version": LOCALIZATION_SCHEMA_VERSION,
        "source_image_sha256": sha,
        "bbox_xyxy": aggregate_box,
        "providers": providers,
        "provider_iou": bbox_iou(first_box, second_box),
        "contract": contract,
        "contract_sha256": localization_sha256_bytes(
            localization_canonical_json(contract).encode("utf-8")
        ),
        "deployed_prediction_used": False,
        "consensus": True,
    }


def _teacher_row(
    path: Path,
    sha: str,
    *,
    material: str = "paper",
    confidence: float = 0.91,
    single_object: bool = True,
    foreign_material: bool = False,
    training_usable: bool = True,
    quality_reason: str = "usable",
    errors: list[str] | None = None,
    bbox=None,
    include_localization: bool = True,
) -> dict:
    decision = {
        "material": material,
        "single_object": single_object,
        "foreign_material": foreign_material,
        "training_usable": training_usable,
        "quality_reason": quality_reason,
        "votes": 2,
        "pass_count": 2,
    }
    passes = [
        {
            "material": material,
            "confidence": confidence,
            "single_object": single_object,
            "foreign_material": foreign_material,
            "training_usable": training_usable,
            "quality_reason": quality_reason,
        },
        {
            "material": material,
            "confidence": confidence + 0.01 if confidence < 0.99 else confidence,
            "single_object": single_object,
            "foreign_material": foreign_material,
            "training_usable": training_usable,
            "quality_reason": quality_reason,
        },
    ]
    contract, contract_sha = teacher.build_teacher_contract("qwen3-vl:8b", "a" * 64)
    row = {
        "schema_version": TEACHER_LABEL_SCHEMA_VERSION,
        "sha256": sha,
        "image_ref": path.name,
        "input_image_sha256": sha,
        "teacher_contract": contract,
        "teacher_contract_sha256": contract_sha,
        "model": "qwen3-vl:8b",
        "model_digest": "a" * 64,
        "passes": passes,
        "errors": [] if errors is None else errors,
        "consensus": True,
        "consensus_decision": decision,
        "minimum_confidence": confidence,
    }
    if include_localization and material != "negative":
        row["independent_localization"] = _independent_localization(sha, bbox)
    return row


def _build_inputs(tmp_path: Path, queue: list[dict], labels: list[dict]):
    queue_path = tmp_path / "teacher_queue.jsonl"
    labels_path = tmp_path / "teacher_labels.jsonl"
    a_model, a_spec = tmp_path / "provider-a.model", tmp_path / "provider-a.spec"
    b_model, b_spec = tmp_path / "provider-b.model", tmp_path / "provider-b.spec"
    a_model.write_bytes(b"provider-a-model")
    a_spec.write_bytes(b"provider-a-spec")
    b_model.write_bytes(b"provider-b-model")
    b_spec.write_bytes(b"provider-b-spec")
    a_model_sha = hashlib.sha256(a_model.read_bytes()).hexdigest()
    a_spec_sha = hashlib.sha256(a_spec.read_bytes()).hexdigest()
    b_model_sha = hashlib.sha256(b_model.read_bytes()).hexdigest()
    b_spec_sha = hashlib.sha256(b_spec.read_bytes()).hexdigest()
    a_rows, b_rows = [], []
    for label in labels:
        localization = label.get("independent_localization")
        if not isinstance(localization, dict):
            continue
        sha = label["sha256"]
        provider_boxes = [
            list(item.get("bbox_xyxy") or localization.get("bbox_xyxy"))
            for item in localization.get("providers", [{}, {}])
        ]
        a_core = provider_output_core(
            provider="detector_a", source_sha=sha, box=provider_boxes[0],
            model_sha=a_model_sha, spec_sha=a_spec_sha,
        )
        b_core = provider_output_core(
            provider="segmenter_b", source_sha=sha, box=provider_boxes[1],
            model_sha=b_model_sha, spec_sha=b_spec_sha,
        )
        a_rows.append(dict(a_core, provider_output_sha256=localization_sha256_bytes(
            localization_canonical_json(a_core).encode("utf-8")
        )))
        b_rows.append(dict(b_core, provider_output_sha256=localization_sha256_bytes(
            localization_canonical_json(b_core).encode("utf-8")
        )))
    a_rows = list({row["source_image_sha256"]: row for row in a_rows}.values())
    b_rows = list({row["source_image_sha256"]: row for row in b_rows}.values())
    _jsonl(tmp_path / "provider-a.jsonl", a_rows)
    _jsonl(tmp_path / "provider-b.jsonl", b_rows)
    a_manifest_sha = hashlib.sha256((tmp_path / "provider-a.jsonl").read_bytes()).hexdigest()
    b_manifest_sha = hashlib.sha256((tmp_path / "provider-b.jsonl").read_bytes()).hexdigest()
    by_a = {row["source_image_sha256"]: row for row in a_rows}
    by_b = {row["source_image_sha256"]: row for row in b_rows}
    for label in labels:
        localization = label.get("independent_localization")
        if not isinstance(localization, dict):
            continue
        sha = label["sha256"]
        original = localization["providers"]
        trusted = [by_a[sha], by_b[sha]]
        for index in range(2):
            if original[index].get("provider") == ("detector_a", "segmenter_b")[index]:
                original[index]["provider"] = trusted[index]["provider"]
            if original[index].get("model_sha256") in {"1" * 64, "3" * 64}:
                original[index]["model_sha256"] = trusted[index]["model_sha256"]
            if original[index].get("inference_spec_sha256") in {"2" * 64, "4" * 64}:
                original[index]["inference_spec_sha256"] = trusted[index]["inference_spec_sha256"]
            if original[index].get("provider_output_sha256") != "0" * 64:
                core = {key: original[index][key] for key in trusted[index] if key != "provider_output_sha256"}
                original[index]["provider_output_sha256"] = localization_sha256_bytes(
                    localization_canonical_json(core).encode("utf-8")
                )
        evidence = [
            {"provider": "detector_a", "manifest_sha256": a_manifest_sha,
             "model_sha256": a_model_sha, "inference_spec_sha256": a_spec_sha},
            {"provider": "segmenter_b", "manifest_sha256": b_manifest_sha,
             "model_sha256": b_model_sha, "inference_spec_sha256": b_spec_sha},
        ]
        if localization["contract"]["providers"][0].get("manifest_sha256") != "7" * 64:
            localization["contract"]["providers"] = evidence
        localization["contract_sha256"] = localization_sha256_bytes(
            localization_canonical_json(localization["contract"]).encode("utf-8")
        )
    _jsonl(queue_path, queue)
    _jsonl(labels_path, labels)
    (tmp_path / "known_audit.json").write_text("{}", encoding="utf-8")
    (tmp_path / "capture_inventory.json").write_text(
        json.dumps(queue), encoding="utf-8"
    )
    return queue_path, labels_path


def _evidence_args(root: Path) -> dict:
    return {
        "known_audit": root / "known_audit.json",
        "capture_inventory": root / "capture_inventory.json",
        "provider_a_manifest": root / "provider-a.jsonl",
        "provider_a_name": "detector_a",
        "provider_a_model": root / "provider-a.model",
        "provider_a_spec": root / "provider-a.spec",
        "provider_b_manifest": root / "provider-b.jsonl",
        "provider_b_name": "segmenter_b",
        "provider_b_model": root / "provider-b.model",
        "provider_b_spec": root / "provider-b.spec",
    }


def build_operational_teacher_manifest(**kwargs):
    root = kwargs["teacher_queue"].parent
    defaults = _evidence_args(root)
    defaults.update(kwargs)
    return _real_build_operational_teacher_manifest(**defaults)


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
                        "image_ref": paper_path.name,
                        "decision": "teacher_required",
                    "request": {"client_id": "private-feedback-id"},
                },
                {
                        "sha256": negative_sha,
                        "timestamp": "2026-08-01T01:00:20Z",
                        "image_ref": negative_path.name,
                        "decision": "teacher_required",
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
        image_root=tmp_path,
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
    assert rows[0]["bbox_source"] == "independent_localization_consensus"

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
    lineage = json.loads(
        (output / ARTIFACT_NAMES["lineage"]).read_text(encoding="utf-8")
    )
    assert lineage["policy"]["operational_capture_cutoff_kst"] == (
        "2026-08-01T00:00:00+09:00"
    )
    assert lineage["policy"]["blur_filter_enabled"] is False
    assert lineage["policy"]["deployed_prediction_filter_enabled"] is False
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
        image_root=tmp_path,
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
        image_root=tmp_path,
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
        label = _teacher_row(
            image_path, declared_sha, include_localization=False
        )
    elif mutation == "out_of_bounds":
        label = _teacher_row(
            image_path, declared_sha, bbox=[-1, 10, 100, 90]
        )
    queue, labels = _build_inputs(
        tmp_path,
        [_queue_row(image_path, declared_sha, "2026-08-01T01:00:00Z", bbox)],
        [label],
    )

    result = build_operational_teacher_manifest(
        teacher_queue=queue,
        teacher_labels=labels,
        image_root=tmp_path,
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


@pytest.mark.parametrize(
    ("timestamp_source", "timestamp", "reason"),
    [
        (
            "queue",
            "2026-07-31T14:59:59Z",
            "queue_capture_before_operational_cutoff",
        ),
        (
            "queue",
            "2026-08-01T01:00:00",
            "queue_capture_timestamp_missing_invalid_or_naive",
        ),
        (
            "inventory",
            "2026-07-31T14:59:59Z",
            "inventory_capture_before_operational_cutoff",
        ),
        (
            "inventory",
            "2026-08-01T01:00:00",
            "inventory_capture_timestamp_missing_invalid_or_naive",
        ),
    ],
)
def test_manifest_rechecks_fixed_cutoff_for_queue_and_inventory(
    tmp_path, timestamp_source, timestamp, reason
):
    image_path, sha = _image(tmp_path, f"cutoff-{timestamp_source}")
    queue_timestamp = timestamp if timestamp_source == "queue" else "2026-08-01T01:00:00Z"
    inventory_timestamp = (
        timestamp if timestamp_source == "inventory" else "2026-08-01T01:00:00Z"
    )
    queue, labels = _build_inputs(
        tmp_path,
        [_queue_row(image_path, sha, queue_timestamp, [10, 10, 100, 90])],
        [_teacher_row(image_path, sha)],
    )
    inventory = tmp_path / "capture_inventory.json"
    inventory.write_text(
        json.dumps([{"sha256": sha, "timestamp": inventory_timestamp}]),
        encoding="utf-8",
    )

    result = build_operational_teacher_manifest(
        teacher_queue=queue,
        teacher_labels=labels,
        image_root=tmp_path,
        capture_inventory=inventory,
        output_dir=tmp_path / "output",
    )

    assert result["accepted"] == 0
    assert result["reason_counts"][reason] == 1
    assert result["operational_capture_cutoff_kst"] == "2026-08-01T00:00:00+09:00"


def test_manifest_rejects_missing_inventory_lineage_when_inventory_is_supplied(tmp_path):
    image_path, sha = _image(tmp_path, "missing-inventory")
    queue, labels = _build_inputs(
        tmp_path,
        [_queue_row(image_path, sha, "2026-08-01T01:00:00Z", [10, 10, 100, 90])],
        [_teacher_row(image_path, sha)],
    )
    inventory = tmp_path / "capture_inventory.json"
    inventory.write_text("[]", encoding="utf-8")

    result = build_operational_teacher_manifest(
        teacher_queue=queue,
        teacher_labels=labels,
        image_root=tmp_path,
        capture_inventory=inventory,
        output_dir=tmp_path / "output",
    )

    assert result["accepted"] == 0
    assert result["reason_counts"] == {
        "capture_inventory_row_missing_or_duplicate": 1
    }


@pytest.mark.parametrize(
    ("kind", "expected_reason"),
    [
        ("tiny", "image_resolution_below_minimum"),
        ("black", "image_extreme_underexposure"),
        ("white", "image_extreme_overexposure"),
        ("unreadable", "image_unreadable"),
    ],
)
def test_manifest_rechecks_objective_capture_quality(tmp_path, kind, expected_reason):
    options = {
        "tiny": {"size": (60, 80)},
        "black": {"value": 0},
        "white": {"value": 255},
        "unreadable": {"readable": False},
    }[kind]
    image_path, sha = _image(tmp_path, kind, **options)
    queue, labels = _build_inputs(
        tmp_path,
        [_queue_row(image_path, sha, "2026-08-01T01:00:00Z", [10, 10, 50, 50])],
        [_teacher_row(image_path, sha)],
    )

    result = build_operational_teacher_manifest(
        teacher_queue=queue,
        teacher_labels=labels,
        image_root=tmp_path,
        output_dir=tmp_path / "output",
    )

    assert result["accepted"] == 0
    assert result["reason_counts"][expected_reason] == 1


@pytest.mark.parametrize(
    "quality_reason",
    [
        "severe_frame_crop",
        "person_occlusion_or_dominance",
        "clutter_or_multiple_objects",
        "boundary_unreadable",
    ],
)
def test_manifest_rejects_high_confidence_training_unusable_consensus(
    tmp_path, quality_reason
):
    image_path, sha = _image(tmp_path, quality_reason)
    queue, labels = _build_inputs(
        tmp_path,
        [_queue_row(image_path, sha, "2026-08-01T01:00:00Z", [10, 10, 100, 90])],
        [
            _teacher_row(
                image_path,
                sha,
                confidence=0.99,
                training_usable=False,
                quality_reason=quality_reason,
            )
        ],
    )

    result = build_operational_teacher_manifest(
        teacher_queue=queue,
        teacher_labels=labels,
        image_root=tmp_path,
        output_dir=tmp_path / "output",
    )

    assert result["accepted"] == 0
    assert result["reason_counts"] == {
        f"training_unusable_{quality_reason}": 1
    }
    report = json.loads(
        ((tmp_path / "output") / ARTIFACT_NAMES["rejections"]).read_text(
            encoding="utf-8"
        )
    )
    assert report["rejections"][0]["teacher_training_usable"] is False
    assert report["rejections"][0]["teacher_quality_reason"] == quality_reason


def test_manifest_rejects_legacy_teacher_schema_fail_closed(tmp_path):
    image_path, sha = _image(tmp_path, "legacy-schema")
    queue_row = _queue_row(
        image_path, sha, "2026-08-01T01:00:00Z", [10, 10, 100, 90]
    )
    teacher_row = _teacher_row(image_path, sha)
    teacher_row.pop("schema_version")
    for item in teacher_row["passes"]:
        item.pop("training_usable")
        item.pop("quality_reason")
    teacher_row["consensus_decision"].pop("training_usable")
    teacher_row["consensus_decision"].pop("quality_reason")
    queue, labels = _build_inputs(tmp_path, [queue_row], [teacher_row])

    result = build_operational_teacher_manifest(
        teacher_queue=queue,
        teacher_labels=labels,
        image_root=tmp_path,
        output_dir=tmp_path / "output",
    )

    assert result["accepted"] == 0
    assert result["reason_counts"]["teacher_label_schema_version_mismatch"] == 1
    assert result["reason_counts"]["invalid_consensus_payload"] == 1


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
        image_root=tmp_path,
        output_dir=tmp_path / "output",
    )
    assert result["accepted"] == 0
    assert result["reason_counts"] == {"duplicate_teacher_label_sha256": 1}


def test_deployed_only_bbox_is_never_accepted(tmp_path):
    image_path, sha = _image(tmp_path, "deployed-only")
    queue_row = _queue_row(
        image_path, sha, "2026-08-01T01:00:00Z", [10, 10, 100, 90]
    )
    queue_row["deployed"] = {"bbox": [10, 10, 100, 90]}
    label = _teacher_row(image_path, sha, include_localization=False)
    label["deployed"] = {"bbox": [10, 10, 100, 90]}
    queue, labels = _build_inputs(tmp_path, [queue_row], [label])

    result = build_operational_teacher_manifest(
        teacher_queue=queue,
        teacher_labels=labels,
        image_root=tmp_path,
        output_dir=tmp_path / "output",
    )

    assert result["accepted"] == 0
    assert result["reason_counts"]["independent_localization_missing"] == 1
    assert result["reason_counts"]["bbox_missing"] == 1


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("schema", "independent_localization_schema_version_mismatch"),
        ("source_sha", "independent_localization_source_sha256_mismatch"),
        ("deployed", "independent_localization_used_deployed_prediction"),
        ("no_consensus", "independent_localization_no_consensus"),
        ("same_provider", "independent_localization_providers_not_distinct"),
        ("bad_model_hash", "independent_localization_invalid_provider_model_sha256"),
        ("bad_bbox", "bbox_out_of_bounds"),
        ("aggregate", "independent_localization_aggregate_mismatch"),
        ("provider_output", "independent_localization_provider_output_sha256_mismatch"),
        ("manifest_hash", "independent_localization_contract_sha256_mismatch"),
        ("model_tamper", "independent_localization_provider_files_mismatch"),
        ("spec_tamper", "independent_localization_provider_files_mismatch"),
    ],
)
def test_independent_localization_tamper_rejects_fail_closed(
    tmp_path, mutation, reason
):
    image_path, sha = _image(tmp_path, f"localization-{mutation}")
    label = _teacher_row(image_path, sha)
    localization = label["independent_localization"]
    if mutation == "schema":
        localization["schema_version"] = "independent_localization.v0"
    elif mutation == "source_sha":
        localization["source_image_sha256"] = "0" * 64
    elif mutation == "deployed":
        localization["deployed_prediction_used"] = True
    elif mutation == "no_consensus":
        localization["consensus"] = False
    elif mutation == "same_provider":
        localization["providers"][1]["provider"] = "detector_a"
    elif mutation == "bad_model_hash":
        localization["providers"][1]["model_sha256"] = "bad"
    elif mutation == "bad_bbox":
        localization["bbox_xyxy"] = [-1, 10, 100, 90]
    elif mutation == "aggregate":
        localization["bbox_xyxy"][0] += 1
    elif mutation == "provider_output":
        localization["providers"][0]["provider_output_sha256"] = "0" * 64
    elif mutation == "manifest_hash":
        localization["contract"]["providers"][0]["manifest_sha256"] = "7" * 64
    elif mutation == "model_tamper":
        localization["providers"][0]["model_sha256"] = "7" * 64
    elif mutation == "spec_tamper":
        localization["providers"][0]["inference_spec_sha256"] = "7" * 64
    queue, labels = _build_inputs(
        tmp_path,
        [_queue_row(image_path, sha, "2026-08-01T01:00:00Z")],
        [label],
    )

    result = build_operational_teacher_manifest(
        teacher_queue=queue,
        teacher_labels=labels,
        image_root=tmp_path,
        output_dir=tmp_path / "output",
    )

    assert result["accepted"] == 0
    assert result["reason_counts"][reason] == 1


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("contract_hash", "teacher_contract_not_exact_trusted_contract"),
        ("contract_prompt", "teacher_contract_not_exact_trusted_contract"),
        ("contract_options", "teacher_contract_not_exact_trusted_contract"),
        ("model_digest", "teacher_contract_not_exact_trusted_contract"),
        ("input_sha", "teacher_input_image_sha256_mismatch"),
        ("bool_confidence", "invalid_teacher_pass"),
        ("nan_confidence", "invalid_teacher_pass"),
        ("minimum", "minimum_confidence_mismatch"),
        ("votes", "consensus_vote_mismatch"),
    ],
)
def test_teacher_contract_sha_and_consensus_are_recomputed(
    tmp_path, mutation, reason
):
    image_path, sha = _image(tmp_path, f"teacher-{mutation}")
    label = _teacher_row(image_path, sha)
    if mutation == "contract_hash":
        label["teacher_contract"]["model_identifier"] = "tampered-model"
    elif mutation == "contract_prompt":
        label["teacher_contract"]["rendered_prompts"]["initial"][0] += " tampered"
    elif mutation == "contract_options":
        label["teacher_contract"]["request"]["options"]["num_predict"] = 1
    elif mutation == "model_digest":
        label["model_digest"] = "b" * 64
    elif mutation == "input_sha":
        label["input_image_sha256"] = "0" * 64
    elif mutation == "bool_confidence":
        label["passes"][0]["confidence"] = True
    elif mutation == "nan_confidence":
        label["passes"][0]["confidence"] = float("nan")
    elif mutation == "minimum":
        label["minimum_confidence"] = 0.1
    elif mutation == "votes":
        label["consensus_decision"]["votes"] = True
    queue, labels = _build_inputs(
        tmp_path,
        [_queue_row(image_path, sha, "2026-08-01T01:00:00Z")],
        [label],
    )

    result = build_operational_teacher_manifest(
        teacher_queue=queue,
        teacher_labels=labels,
        image_root=tmp_path,
        output_dir=tmp_path / "output",
    )

    assert result["accepted"] == 0
    assert result["reason_counts"][reason] == 1


@pytest.mark.parametrize(
    ("single_object", "foreign_material", "reason"),
    [
        (True, False, "negative_single_object_must_be_false"),
        (False, True, "negative_foreign_material_must_be_false"),
    ],
)
def test_negative_requires_empty_scene_semantics(
    tmp_path, single_object, foreign_material, reason
):
    image_path, sha = _image(tmp_path, f"negative-{reason}")
    queue, labels = _build_inputs(
        tmp_path,
        [_queue_row(image_path, sha, "2026-08-01T01:00:00Z")],
        [
            _teacher_row(
                image_path,
                sha,
                material="negative",
                single_object=single_object,
                foreign_material=foreign_material,
            )
        ],
    )

    result = build_operational_teacher_manifest(
        teacher_queue=queue,
        teacher_labels=labels,
        image_root=tmp_path,
        output_dir=tmp_path / "output",
    )

    assert result["accepted"] == 0
    assert result["reason_counts"][reason] == 1


def test_manifest_rejects_absolute_or_symlink_escape_image_ref(tmp_path):
    image_path, sha = _image(tmp_path, "inside")
    label = _teacher_row(image_path, sha)
    absolute_queue = _queue_row(image_path, sha, "2026-08-01T01:00:00Z")
    absolute_queue["image_ref"] = str(image_path.resolve())
    absolute_dir = tmp_path / "absolute"
    absolute_dir.mkdir()
    queue, labels = _build_inputs(absolute_dir, [absolute_queue], [label])
    result = build_operational_teacher_manifest(
        teacher_queue=queue,
        teacher_labels=labels,
        image_root=tmp_path,
        output_dir=tmp_path / "absolute-output",
    )
    assert result["reason_counts"]["image_ref_invalid_or_outside_root"] == 1

    outside = tmp_path.parent / "outside-builder.png"
    outside.write_bytes(image_path.read_bytes())
    link = tmp_path / "outside-link.png"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is not available")
    link_queue = _queue_row(image_path, sha, "2026-08-01T01:00:00Z")
    link_queue["image_ref"] = link.name
    link_dir = tmp_path / "link-input"
    link_dir.mkdir()
    queue, labels = _build_inputs(link_dir, [link_queue], [label])
    result = build_operational_teacher_manifest(
        teacher_queue=queue,
        teacher_labels=labels,
        image_root=tmp_path,
        output_dir=tmp_path / "link-output",
    )
    assert result["reason_counts"]["image_ref_invalid_or_outside_root"] == 1


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
        image_root=tmp_path,
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
        image_root=tmp_path,
        output_dir=output,
        role="calibration",
        dry_run=True,
    )
    assert preview["accepted"] == 1
    assert not output.exists()

    first = build_operational_teacher_manifest(
        teacher_queue=queue,
        teacher_labels=labels,
        image_root=tmp_path,
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
            image_root=tmp_path,
            output_dir=output,
            role="calibration",
        )
    second = build_operational_teacher_manifest(
        teacher_queue=queue,
        teacher_labels=labels,
        image_root=tmp_path,
        output_dir=output,
        role="calibration",
        overwrite=True,
    )
    assert second["output_digests"] == first["output_digests"]

    with pytest.raises(ValueError, match="never be blind_test"):
        build_operational_teacher_manifest(
            teacher_queue=queue,
            teacher_labels=labels,
            image_root=tmp_path,
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
                "--image-root",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "cli-blind"),
                "--role",
                "blind_test",
            ]
        )
