import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import scripts.assemble_operational_quality_exclusions as quality_assembler
import scripts.prepare_operational_capture_queue as capture_queue
from scripts.build_operational_teacher_manifest import (
    ARTIFACT_NAMES,
    TEACHER_LABEL_SCHEMA_VERSION,
    build_operational_teacher_manifest as _real_build_operational_teacher_manifest,
    main,
)
from scripts.assemble_operational_quality_exclusions import (
    ASSEMBLY_FILES,
    assemble_operational_quality_exclusions,
)
from scripts.build_v4_candidate_training_authority import _validate_quality_manifest
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


def _quality_assembly_fixture(
    tmp_path: Path,
    reasons: tuple[str, ...] = (
        "severe_frame_crop",
        "person_occlusion_or_dominance",
        "clutter_or_multiple_objects",
        "boundary_unreadable",
    ),
) -> tuple[dict, list[Path]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    queue_rows: list[dict] = []
    label_rows: list[dict] = []
    for index, reason in enumerate(reasons):
        image_path, sha = _image(tmp_path, f"quality-{index}", 70 + index * 20)
        image_paths.append(image_path)
        queue_rows.append(
            _queue_row(
                image_path,
                sha,
                f"2026-08-01T01:0{index}:00Z",
                [10, 10, 100, 90],
            )
        )
        label_rows.append(
            _teacher_row(
                image_path,
                sha,
                training_usable=False,
                quality_reason=reason,
            )
        )
    usable_path, usable_sha = _image(tmp_path, "usable-control", 170)
    queue_rows.append(
        _queue_row(
            usable_path,
            usable_sha,
            "2026-08-01T02:00:00Z",
            [10, 10, 100, 90],
        )
    )
    label_rows.append(_teacher_row(usable_path, usable_sha))
    queue_path, labels_path = _build_inputs(tmp_path, queue_rows, label_rows)
    inventory_path = tmp_path / "capture_inventory.json"
    inventory_rows = json.loads(inventory_path.read_text(encoding="utf-8"))
    for index, row in enumerate(inventory_rows):
        row["request"] = {"client_id": f"private-client-{index}"}
    inventory_path.write_text(json.dumps(inventory_rows), encoding="utf-8")
    teacher_output = tmp_path / "teacher-output"
    build_operational_teacher_manifest(
        teacher_queue=queue_path,
        teacher_labels=labels_path,
        image_root=tmp_path,
        output_dir=teacher_output,
    )
    evidence = _evidence_args(tmp_path)
    args = {
        "teacher_output_dir": teacher_output,
        "teacher_queue": queue_path,
        "teacher_labels": labels_path,
        "capture_inventory": inventory_path,
        "known_audit": evidence["known_audit"],
        "provider_a_manifest": evidence["provider_a_manifest"],
        "provider_a_model": evidence["provider_a_model"],
        "provider_a_spec": evidence["provider_a_spec"],
        "provider_b_manifest": evidence["provider_b_manifest"],
        "provider_b_model": evidence["provider_b_model"],
        "provider_b_spec": evidence["provider_b_spec"],
        "image_root": tmp_path,
        "output_dir": tmp_path / "quality-assembly",
    }
    return args, image_paths


def _write_operational_capture_metadata(
    captures: Path,
    image_path: Path,
    sha: str,
    timestamp: str,
    *,
    client_id: str,
) -> Path:
    metadata_path = captures / f"{image_path.stem}.json"
    metadata_path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "image": {"path": image_path.name, "sha256": sha},
                "request": {"client_id": client_id, "device_id": "private-device"},
                "result": {
                    "status": "ALLOWED",
                    "classification": {
                        "class_name": "plastic",
                        "confidence": 0.99,
                    },
                    "bbox": [1, 2, 3, 4],
                },
            }
        ),
        encoding="utf-8",
    )
    return metadata_path


def _objective_quality_assembly_fixture(
    tmp_path: Path, *, no_quality_exclusions: bool = False,
) -> tuple[dict, Path, Path]:
    """Build the real prepare -> teacher -> assembler input chain."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    captures = tmp_path / "captures"
    captures.mkdir()
    objective_path, objective_sha = _image(
        captures, "objective-tiny", 100,
        size=(240, 320) if no_quality_exclusions else (60, 80),
    )
    subjective_path, subjective_sha = _image(
        captures, "subjective-boundary", 120
    )
    usable_path, usable_sha = _image(captures, "usable-control", 170)
    _write_operational_capture_metadata(
        captures,
        objective_path,
        objective_sha,
        "2026-08-01T00:00:00+09:00",
        client_id="private-objective-client",
    )
    _write_operational_capture_metadata(
        captures,
        subjective_path,
        subjective_sha,
        "2026-08-01T00:01:00+09:00",
        client_id="private-subjective-client",
    )
    _write_operational_capture_metadata(
        captures,
        usable_path,
        usable_sha,
        "2026-08-01T00:02:00+09:00",
        client_id="private-usable-client",
    )
    shadow = tmp_path / "shadow.jsonl"
    shadow.write_text("", encoding="utf-8")
    known = tmp_path / "known_audit.json"
    known.write_text("{}", encoding="utf-8")
    prepare_output = tmp_path / "prepare-output"
    capture_queue.prepare_queue(
        captures_dir=captures,
        shadow_log=shadow,
        known_audit=known,
        output_dir=prepare_output,
        start_kst=capture_queue.OPERATIONAL_CAPTURE_CUTOFF_KST,
    )
    queue_path = prepare_output / "teacher_queue.jsonl"
    queue_rows = [
        json.loads(line)
        for line in queue_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_queue = {subjective_sha, usable_sha}
    if no_quality_exclusions:
        expected_queue.add(objective_sha)
    assert {row["sha256"] for row in queue_rows} == expected_queue
    labels = []
    for row in queue_rows:
        path = captures / row["image_ref"]
        if row["sha256"] == subjective_sha and not no_quality_exclusions:
            labels.append(
                _teacher_row(
                    path,
                    row["sha256"],
                    training_usable=False,
                    quality_reason="boundary_unreadable",
                )
            )
        else:
            labels.append(_teacher_row(path, row["sha256"]))
    _, labels_path = _build_inputs(tmp_path, queue_rows, labels)
    evidence = _evidence_args(tmp_path)
    teacher_output = tmp_path / "teacher-output"
    _real_build_operational_teacher_manifest(
        teacher_queue=queue_path,
        teacher_labels=labels_path,
        image_root=captures,
        known_audit=known,
        provider_a_manifest=evidence["provider_a_manifest"],
        provider_a_name="detector_a",
        provider_a_model=evidence["provider_a_model"],
        provider_a_spec=evidence["provider_a_spec"],
        provider_b_manifest=evidence["provider_b_manifest"],
        provider_b_name="segmenter_b",
        provider_b_model=evidence["provider_b_model"],
        provider_b_spec=evidence["provider_b_spec"],
        output_dir=teacher_output,
        capture_inventory=prepare_output / "capture_inventory.json",
    )
    args = {
        "teacher_output_dir": teacher_output,
        "teacher_queue": queue_path,
        "teacher_labels": labels_path,
        "capture_inventory": prepare_output / "capture_inventory.json",
        "known_audit": known,
        "provider_a_manifest": evidence["provider_a_manifest"],
        "provider_a_model": evidence["provider_a_model"],
        "provider_a_spec": evidence["provider_a_spec"],
        "provider_b_manifest": evidence["provider_b_manifest"],
        "provider_b_model": evidence["provider_b_model"],
        "provider_b_spec": evidence["provider_b_spec"],
        "image_root": captures,
        "output_dir": tmp_path / "quality-assembly",
        "objective_prepare_output_dir": prepare_output,
    }
    return args, objective_path, subjective_path


def _rewrite_bound_lineage_input(args: dict, name: str, path: Path) -> None:
    lineage_path = args["teacher_output_dir"] / ARTIFACT_NAMES["lineage"]
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["inputs"][f"{name}_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    lineage_path.write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _candidate_authority_test_helpers():
    helper_path = Path(__file__).with_name(
        "test_build_v4_candidate_training_authority.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_operational_quality_candidate_test_helpers", helper_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_operational_quality_assembler_emits_only_sha_and_canonical_reason(
    tmp_path: Path,
) -> None:
    args, image_paths = _quality_assembly_fixture(tmp_path)

    receipt = assemble_operational_quality_exclusions(**args)

    output = args["output_dir"]
    assert {path.name for path in output.iterdir()} == set(ASSEMBLY_FILES.values())
    manifest_path = output / ASSEMBLY_FILES["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        hashlib.sha256(image_paths[0].read_bytes()).hexdigest(): "severe_frame_crop",
        hashlib.sha256(image_paths[1].read_bytes()).hexdigest(): (
            "person_occlusion_or_dominance"
        ),
        hashlib.sha256(image_paths[2].read_bytes()).hexdigest(): (
            "excessive_background_or_multi_object"
        ),
        hashlib.sha256(image_paths[3].read_bytes()).hexdigest(): "unreadable_boundary",
    }
    assert _validate_quality_manifest(manifest) == expected
    persisted_receipt = json.loads(
        (output / ASSEMBLY_FILES["receipt"]).read_text(encoding="utf-8")
    )
    assert persisted_receipt == receipt
    assert receipt["assembly_mode"] == "legacy_subjective_only"
    assert receipt["selected_source_count"] == 4
    assert set(receipt["authority"].values()) == {False}
    assert receipt["scope"] == {
        "teacher_subjective_quality_included": True,
        "objective_queue_quality_included": False,
        "objective_prepare_bundle_validated": False,
        "subjective_quality_source_count": 4,
        "objective_quality_source_count": 0,
        "paths_or_private_ids_exported": False,
        "trusted_policy_pinned": False,
        "executed_code_cryptographically_attested": False,
    }
    rendered = "".join(
        path.read_text(encoding="utf-8") for path in output.iterdir()
    )
    assert "private-client" not in rendered
    assert not any(path.name in rendered for path in image_paths)
    input_paths = {
        name: args[name]
        for name in (
            "teacher_queue",
            "teacher_labels",
            "capture_inventory",
            "known_audit",
            "provider_a_manifest",
            "provider_a_model",
            "provider_a_spec",
            "provider_b_manifest",
            "provider_b_model",
            "provider_b_spec",
        )
    }
    expected_input_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in input_paths.items()
    }
    expected_input_hashes.update(
        {
            f"teacher_output_{name}": hashlib.sha256(
                (args["teacher_output_dir"] / filename).read_bytes()
            ).hexdigest()
            for name, filename in ARTIFACT_NAMES.items()
        }
    )
    assert receipt["input_sha256"] == dict(sorted(expected_input_hashes.items()))
    marker_sources = {
        name: (output / name).read_bytes()
        for name in (ASSEMBLY_FILES["manifest"], ASSEMBLY_FILES["receipt"])
    }
    expected_marker = "".join(
        f"{hashlib.sha256(content).hexdigest()}  {name}\n"
        for name, content in sorted(marker_sources.items())
    )
    assert (output / ASSEMBLY_FILES["marker"]).read_text(
        encoding="ascii"
    ) == expected_marker
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError, match="overwrite immutable output"):
        assemble_operational_quality_exclusions(**args)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_legacy_operational_quality_assembly_is_rejected_by_candidate_authority(
    tmp_path: Path,
) -> None:
    args, image_paths = _quality_assembly_fixture(
        tmp_path / "assembly", ("clutter_or_multiple_objects",)
    )
    assemble_operational_quality_exclusions(**args)
    quality_manifest = args["output_dir"] / ASSEMBLY_FILES["manifest"]
    selected_bytes = image_paths[0].read_bytes()
    selected_sha = hashlib.sha256(selected_bytes).hexdigest()

    candidate = _candidate_authority_test_helpers()
    fixture = candidate._fixture(tmp_path / "candidate")
    bad_row = fixture["bad_row"]
    bad_source = Path(bad_row["source_filepath"])
    bad_source.write_bytes(selected_bytes)
    bad_row["source_sha256"] = selected_sha
    bad_row["origin"] = "ops"
    bad_row["captured_at"] = "2026-08-01T00:00:00+09:00"
    bad_row["auditor_sha256"] = candidate._fake_sha("ops-auditor")
    bad_row["teacher_output_sha256"] = candidate._fake_sha("ops-teacher")
    bad_row["localizer_output_sha256"] = candidate._fake_sha("ops-localizer")
    fixture["quality"] = quality_manifest
    fixture["quality_assembly_receipt"] = (
        args["output_dir"] / ASSEMBLY_FILES["receipt"]
    )
    candidate._refresh(fixture)

    with pytest.raises(ValueError, match="assembly_mode mismatch"):
        candidate._run(fixture)


def test_prepare_objective_quality_flows_through_assembler_and_candidate_authority(
    tmp_path: Path,
) -> None:
    args, objective_path, subjective_path = _objective_quality_assembly_fixture(
        tmp_path / "assembly"
    )

    receipt = assemble_operational_quality_exclusions(**args)

    manifest_path = args["output_dir"] / ASSEMBLY_FILES["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    objective_sha = hashlib.sha256(objective_path.read_bytes()).hexdigest()
    subjective_sha = hashlib.sha256(subjective_path.read_bytes()).hexdigest()
    assert _validate_quality_manifest(manifest) == {
        objective_sha: "too_low_resolution",
        subjective_sha: "unreadable_boundary",
    }
    assert receipt["assembly_mode"] == "objective_and_subjective_quality"
    assert receipt["scope"] == {
        "teacher_subjective_quality_included": True,
        "objective_queue_quality_included": True,
        "objective_prepare_bundle_validated": True,
        "subjective_quality_source_count": 1,
        "objective_quality_source_count": 1,
        "paths_or_private_ids_exported": False,
        "trusted_policy_pinned": False,
        "executed_code_cryptographically_attested": False,
    }
    published_text = "".join(
        path.read_text(encoding="utf-8") for path in args["output_dir"].iterdir()
    )
    for forbidden in (
        objective_path.name,
        subjective_path.name,
        "private-objective-client",
        "private-subjective-client",
        "capture_timestamp_utc",
        "metadata_ref",
        "image_ref",
    ):
        assert forbidden not in published_text

    candidate = _candidate_authority_test_helpers()
    fixture = candidate._fixture(tmp_path / "candidate")
    bad_row = fixture["bad_row"]
    bad_source = Path(bad_row["source_filepath"])
    bad_source.write_bytes(objective_path.read_bytes())
    bad_row["source_sha256"] = objective_sha
    bad_row["origin"] = "ops"
    bad_row["captured_at"] = "2026-08-01T00:00:00+09:00"
    bad_row["auditor_sha256"] = candidate._fake_sha("ops-auditor-objective")
    bad_row["teacher_output_sha256"] = candidate._fake_sha(
        "ops-teacher-objective"
    )
    bad_row["localizer_output_sha256"] = candidate._fake_sha(
        "ops-localizer-objective"
    )
    subjective_source = (
        fixture["global_root"] / "data" / "source-subjective-quality.jpg"
    )
    subjective_crop = (
        fixture["global_root"] / "data" / "crop-subjective-quality.jpg"
    )
    subjective_source.write_bytes(subjective_path.read_bytes())
    subjective_crop.write_bytes(b"subjective-quality-crop")
    subjective_row = dict(bad_row)
    subjective_row.update(
        {
            "filepath": str(subjective_crop.resolve()),
            "source_id": "source-id-subjective-quality",
            "sample_id": "sample-subjective-quality",
            "source_sha256": subjective_sha,
            "image_sha256": hashlib.sha256(
                subjective_crop.read_bytes()
            ).hexdigest(),
            "object_group": "group-subjective-quality",
            "capture_session": "session-subjective-quality",
            "source_filepath": str(subjective_source.resolve()),
            "auditor_sha256": candidate._fake_sha("ops-auditor-subjective"),
            "teacher_output_sha256": candidate._fake_sha(
                "ops-teacher-subjective"
            ),
            "localizer_output_sha256": candidate._fake_sha(
                "ops-localizer-subjective"
            ),
        }
    )
    fixture["rows"].append(subjective_row)
    fixture["quality"] = manifest_path
    fixture["quality_assembly_receipt"] = (
        args["output_dir"] / ASSEMBLY_FILES["receipt"]
    )
    candidate._refresh(fixture)

    authority = candidate._run(fixture)

    assert authority["counts"]["excluded"] == {
        "operational/before_2026_08_01_kst": 1,
        "quality/too_low_resolution": 1,
        "quality/unreadable_boundary": 1,
    }
    train_rows = list(
        csv.DictReader(
            (
                fixture["global_root"] / "authority" / "train_manifest.csv"
            ).open(encoding="utf-8")
        )
    )
    selected_train_shas = {row["source_sha256"] for row in train_rows}
    assert {objective_sha, subjective_sha}.isdisjoint(selected_train_shas)


def test_objective_quality_assembler_rejects_resealed_fuzzy_reason(
    tmp_path: Path,
) -> None:
    args, _, _ = _objective_quality_assembly_fixture(tmp_path)
    prepare_output = args["objective_prepare_output_dir"]
    evidence_path = prepare_output / capture_queue.OBJECTIVE_REJECTIONS_FILE
    rows = [
        json.loads(line)
        for line in evidence_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    rows[0]["quality_reason"] = "resolution_low"
    _jsonl(evidence_path, rows)
    receipt_path = prepare_output / capture_queue.OBJECTIVE_RECEIPT_FILE
    objective_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    objective_receipt["output_digests"]["objective_rejections_sha256"] = (
        hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )
    receipt_path.write_text(
        json.dumps(objective_receipt, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="does not exactly cover|selected reason does not revalidate"
    ):
        assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()


def test_objective_quality_assembler_rejects_resealed_later_start_kst(
    tmp_path: Path,
) -> None:
    args, _, _ = _objective_quality_assembly_fixture(tmp_path)
    prepare_output = args["objective_prepare_output_dir"]
    later_start = "2026-08-02T00:00:00+09:00"
    summary_path = prepare_output / "queue_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["start_kst"] = later_start
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt_path = prepare_output / capture_queue.OBJECTIVE_RECEIPT_FILE
    objective_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    objective_receipt["start_kst"] = later_start
    objective_receipt["output_digests"]["summary_sha256"] = hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    receipt_path.write_text(
        json.dumps(objective_receipt, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="objective quality evidence must cover the exact operational cutoff",
    ):
        assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()


@pytest.mark.parametrize(
    ("field_path", "forged_value", "expected_error"),
    (
        (("schema_version",), True, "receipt identity mismatch"),
        (
            ("quality_policy", "minimum_width"),
            True,
            "evidence policy mismatch",
        ),
        (
            ("privacy", "objective_evidence_absolute_paths_exported"),
            0,
            "privacy declaration mismatch",
        ),
        (
            ("authority", "ground_truth"),
            0,
            "evidence authority mismatch",
        ),
    ),
)
def test_objective_quality_assembler_rejects_bool_int_schema_masquerade(
    tmp_path: Path,
    field_path: tuple[str, ...],
    forged_value: object,
    expected_error: str,
) -> None:
    args, _, _ = _objective_quality_assembly_fixture(tmp_path)
    receipt_path = (
        args["objective_prepare_output_dir"]
        / capture_queue.OBJECTIVE_RECEIPT_FILE
    )
    objective_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    target = objective_receipt
    for field in field_path[:-1]:
        target = target[field]
    assert target[field_path[-1]] != forged_value or type(
        target[field_path[-1]]
    ) is not type(forged_value)
    target[field_path[-1]] = forged_value
    receipt_path.write_text(
        json.dumps(objective_receipt, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=expected_error):
        assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()


@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        ("capture_rows_after_cutoff", 999),
        ("capture_rows_rejected", 999),
        ("capture_rejection_counts", {"forged_reason": 999}),
        ("unique_images", 999),
        ("decisions", {"teacher_required": 999}),
    ),
)
def test_objective_quality_assembler_rejects_resealed_summary_diagnostics(
    tmp_path: Path,
    field: str,
    forged_value: object,
) -> None:
    args, _, _ = _objective_quality_assembly_fixture(tmp_path)
    prepare_output = args["objective_prepare_output_dir"]
    summary_path = prepare_output / "queue_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary[field] != forged_value
    summary[field] = forged_value
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt_path = prepare_output / capture_queue.OBJECTIVE_RECEIPT_FILE
    objective_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    objective_receipt["output_digests"]["summary_sha256"] = hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    receipt_path.write_text(
        json.dumps(objective_receipt, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=f"objective queue summary diagnostic mismatch: {field}",
    ):
        assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()


def test_objective_quality_assembler_reconstructs_subjective_queue_from_index(
    tmp_path: Path,
) -> None:
    args, _, subjective_path = _objective_quality_assembly_fixture(tmp_path)
    prepare_output = args["objective_prepare_output_dir"]
    metadata_path = args["image_root"] / f"{subjective_path.stem}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["timestamp"] = "2026-07-31T23:59:59+09:00"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    capture_index_path = (
        prepare_output / capture_queue.OBJECTIVE_CAPTURE_INDEX_FILE
    )
    capture_index = [
        json.loads(line)
        for line in capture_index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metadata_ref = metadata_path.relative_to(args["image_root"]).as_posix()
    indexed_row = next(
        row for row in capture_index if row["metadata_ref"] == metadata_ref
    )
    indexed_row["metadata_sha256"] = hashlib.sha256(
        metadata_path.read_bytes()
    ).hexdigest()
    _jsonl(capture_index_path, capture_index)

    receipt_path = prepare_output / capture_queue.OBJECTIVE_RECEIPT_FILE
    objective_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    objective_receipt["output_digests"]["capture_index_sha256"] = hashlib.sha256(
        capture_index_path.read_bytes()
    ).hexdigest()
    receipt_path.write_text(
        json.dumps(objective_receipt, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="teacher queue does not exactly match the indexed capture snapshot",
    ):
        assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()


def test_objective_quality_assembler_rejects_resealed_duplicate_sha(
    tmp_path: Path,
) -> None:
    args, _, _ = _objective_quality_assembly_fixture(tmp_path)
    prepare_output = args["objective_prepare_output_dir"]
    evidence_path = prepare_output / capture_queue.OBJECTIVE_REJECTIONS_FILE
    row = json.loads(evidence_path.read_text(encoding="utf-8").strip())
    _jsonl(evidence_path, [row, dict(row)])
    receipt_path = prepare_output / capture_queue.OBJECTIVE_RECEIPT_FILE
    objective_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    objective_receipt["output_digests"]["objective_rejections_sha256"] = (
        hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )
    receipt_path.write_text(
        json.dumps(objective_receipt, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not exactly cover|duplicate source SHA"):
        assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()


def test_objective_quality_assembler_rejects_fully_resealed_omission(
    tmp_path: Path,
) -> None:
    args, _, _ = _objective_quality_assembly_fixture(tmp_path)
    prepare_output = args["objective_prepare_output_dir"]
    evidence_path = prepare_output / capture_queue.OBJECTIVE_REJECTIONS_FILE
    assert evidence_path.read_text(encoding="utf-8").strip()
    evidence_path.write_bytes(b"")

    summary_path = prepare_output / "queue_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["capture_rows_rejected"] = 0
    summary["capture_rejection_counts"] = {}
    summary["objective_quality_rejections"] = 0
    summary["objective_quality_reason_counts"] = {}
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    receipt_path = prepare_output / capture_queue.OBJECTIVE_RECEIPT_FILE
    objective_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    objective_receipt["counts"]["objective_quality_rejections"] = 0
    objective_receipt["counts"]["objective_quality_reason_counts"] = {}
    objective_receipt["input_digests"]["objective_source_bindings_sha256"] = (
        hashlib.sha256(capture_queue._json_bytes([], pretty=False)).hexdigest()
    )
    for name, filename in capture_queue.OUTPUT_FILES.items():
        if name == "objective_receipt":
            continue
        objective_receipt["output_digests"][f"{name}_sha256"] = hashlib.sha256(
            (prepare_output / filename).read_bytes()
        ).hexdigest()
    receipt_path.write_text(
        json.dumps(objective_receipt, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not exactly cover"):
        assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()


def test_objective_and_subjective_same_sha_collision_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, image_paths = _quality_assembly_fixture(
        tmp_path, ("boundary_unreadable",)
    )
    source = image_paths[0].resolve()
    content = source.read_bytes()
    sha = hashlib.sha256(content).hexdigest()

    monkeypatch.setattr(
        quality_assembler,
        "_objective_quality_rows",
        lambda **_kwargs: (
            [{"path": source.name, "reason": "resolution_too_low"}],
            [(source, content, sha, "too_low_resolution")],
            {},
            {},
            {},
            (),
        ),
    )
    objective_placeholder = tmp_path.parent / f"{tmp_path.name}-objective-placeholder"
    objective_placeholder.mkdir()
    args["objective_prepare_output_dir"] = objective_placeholder
    args["output_dir"] = tmp_path.parent / f"{tmp_path.name}-quality-output"

    with pytest.raises(ValueError, match="overlap by SHA"):
        assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()


def test_objective_quality_assembler_rehashes_capture_metadata_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _, _ = _objective_quality_assembly_fixture(tmp_path)
    prepare_output = args["objective_prepare_output_dir"]
    evidence_row = json.loads(
        (prepare_output / capture_queue.OBJECTIVE_REJECTIONS_FILE)
        .read_text(encoding="utf-8")
        .strip()
    )
    metadata_path = args["image_root"] / evidence_row["metadata_ref"]
    original_metadata = metadata_path.read_bytes()
    real_builder = quality_assembler.build_quality_exclusion_manifest

    def mutate_metadata_after_manifest(**kwargs):
        result = real_builder(**kwargs)
        metadata_path.write_bytes(original_metadata + b" ")
        return result

    monkeypatch.setattr(
        quality_assembler,
        "build_quality_exclusion_manifest",
        mutate_metadata_after_manifest,
    )

    with pytest.raises(
        RuntimeError, match="objective evidence metadata or source changed before publish"
    ):
        assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()


def test_objective_quality_assembler_rechecks_exact_metadata_set_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _, _ = _objective_quality_assembly_fixture(tmp_path)
    injected_metadata = args["image_root"] / "late-capture.json"
    real_builder = quality_assembler.build_quality_exclusion_manifest

    def add_metadata_after_manifest(**kwargs):
        result = real_builder(**kwargs)
        injected_metadata.write_text("{}", encoding="utf-8")
        return result

    monkeypatch.setattr(
        quality_assembler,
        "build_quality_exclusion_manifest",
        add_metadata_after_manifest,
    )

    with pytest.raises(
        RuntimeError, match="objective capture metadata set changed before publish"
    ):
        assemble_operational_quality_exclusions(**args)
    assert injected_metadata.is_file()
    assert not args["output_dir"].exists()


def test_objective_quality_assembler_rejects_output_inside_image_root(
    tmp_path: Path,
) -> None:
    args, _, _ = _objective_quality_assembly_fixture(tmp_path)
    args["output_dir"] = args["image_root"] / "quality-assembly"

    with pytest.raises(ValueError, match="output directory must not be inside image root"):
        assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()


def test_objective_quality_assembler_rejects_queue_inventory_sha_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, subjective_paths = _quality_assembly_fixture(
        tmp_path / "assembly", ("boundary_unreadable",)
    )
    queue_rows = [
        json.loads(line)
        for line in args["teacher_queue"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    inventory_rows = json.loads(
        args["capture_inventory"].read_text(encoding="utf-8")
    )
    subjective_sha = hashlib.sha256(subjective_paths[0].read_bytes()).hexdigest()
    overlap_row = next(row for row in queue_rows if row["sha256"] != subjective_sha)
    overlap_sha = overlap_row["sha256"]
    assert overlap_sha in {row["sha256"] for row in inventory_rows}
    overlap_source = (args["image_root"] / overlap_row["image_ref"]).resolve()
    overlap_content = overlap_source.read_bytes()
    assert hashlib.sha256(overlap_content).hexdigest() == overlap_sha

    monkeypatch.setattr(
        quality_assembler,
        "_objective_quality_rows",
        lambda **_kwargs: (
            [{"path": overlap_source.name, "reason": "resolution_too_low"}],
            [(overlap_source, overlap_content, overlap_sha, "too_low_resolution")],
            {},
            {},
            {},
            (),
        ),
    )
    objective_placeholder = tmp_path / "objective-placeholder"
    objective_placeholder.mkdir()
    args["objective_prepare_output_dir"] = objective_placeholder
    args["output_dir"] = tmp_path / "quality-output"

    with pytest.raises(
        ValueError,
        match="objective quality evidence overlaps the teacher queue or inventory",
    ):
        assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()


def test_operational_quality_assembler_rejects_extra_decision_reason(
    tmp_path: Path,
) -> None:
    args, _ = _quality_assembly_fixture(tmp_path, ("severe_frame_crop",))
    rejection_path = args["teacher_output_dir"] / ARTIFACT_NAMES["rejections"]
    rejection_report = json.loads(rejection_path.read_text(encoding="utf-8"))
    rejection_report["rejections"][0]["reasons"].append(
        "minimum_confidence_below_threshold"
    )
    rejection_report["rejections"][0]["reasons"].sort()
    rejection_report["reason_counts"]["minimum_confidence_below_threshold"] = 1
    rejection_path.write_text(
        json.dumps(rejection_report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    lineage_path = args["teacher_output_dir"] / ARTIFACT_NAMES["lineage"]
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["output_digests"]["rejections_sha256"] = hashlib.sha256(
        rejection_path.read_bytes()
    ).hexdigest()
    lineage_path.write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="additional rejection reasons"):
        assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()


def test_operational_quality_assembler_rechecks_cutoff_after_bound_input_change(
    tmp_path: Path,
) -> None:
    args, _ = _quality_assembly_fixture(tmp_path, ("boundary_unreadable",))
    queue_rows = [
        json.loads(line)
        for line in args["teacher_queue"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    queue_rows[0]["timestamp"] = "2026-07-31T14:59:59Z"
    _jsonl(args["teacher_queue"], queue_rows)
    inventory_rows = json.loads(args["capture_inventory"].read_text(encoding="utf-8"))
    inventory_rows[0]["timestamp"] = "2026-07-31T14:59:59Z"
    args["capture_inventory"].write_text(
        json.dumps(inventory_rows), encoding="utf-8"
    )
    _rewrite_bound_lineage_input(args, "teacher_queue", args["teacher_queue"])
    _rewrite_bound_lineage_input(
        args, "capture_inventory", args["capture_inventory"]
    )

    with pytest.raises(ValueError, match="before the operational cutoff"):
        assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()


def test_operational_quality_assembler_rejects_forged_path_and_changed_bytes(
    tmp_path: Path,
) -> None:
    args, image_paths = _quality_assembly_fixture(tmp_path, ("boundary_unreadable",))
    image_paths[0].write_bytes(b"changed-after-teacher-output")
    with pytest.raises(ValueError, match="bytes do not match teacher SHA"):
        assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()

    # Rebuild a clean fixture in a separate directory for a self-consistently
    # rebound path traversal attempt.
    forged_root = tmp_path / "forged"
    forged_root.mkdir()
    forged_args, _ = _quality_assembly_fixture(
        forged_root, ("boundary_unreadable",)
    )
    queue_rows = [
        json.loads(line)
        for line in forged_args["teacher_queue"].read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    label_rows = [
        json.loads(line)
        for line in forged_args["teacher_labels"].read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    inventory_rows = json.loads(
        forged_args["capture_inventory"].read_text(encoding="utf-8")
    )
    for row in (*queue_rows, *label_rows, *inventory_rows):
        row["image_ref"] = "../outside.png"
    _jsonl(forged_args["teacher_queue"], queue_rows)
    _jsonl(forged_args["teacher_labels"], label_rows)
    forged_args["capture_inventory"].write_text(
        json.dumps(inventory_rows), encoding="utf-8"
    )
    for name in ("teacher_queue", "teacher_labels", "capture_inventory"):
        _rewrite_bound_lineage_input(forged_args, name, forged_args[name])

    with pytest.raises(ValueError, match="normalized and relative"):
        assemble_operational_quality_exclusions(**forged_args)
    assert not forged_args["output_dir"].exists()


def test_operational_quality_assembler_refuses_empty_quality_selection(
    tmp_path: Path,
) -> None:
    args, _ = _quality_assembly_fixture(tmp_path, ())

    with pytest.raises(ValueError, match="zero exclusions require full objective and subjective evidence"):
        assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()


def test_operational_quality_assembler_rejects_known_audit_overlap(
    tmp_path: Path,
) -> None:
    args, image_paths = _quality_assembly_fixture(tmp_path, ("severe_frame_crop",))
    selected_sha = hashlib.sha256(image_paths[0].read_bytes()).hexdigest()
    args["known_audit"].write_text(
        json.dumps({selected_sha: {"split": "train"}}), encoding="utf-8"
    )
    _rewrite_bound_lineage_input(args, "known_audit", args["known_audit"])

    with pytest.raises(ValueError, match="already in known audit"):
        assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()


def test_operational_quality_assembler_rehashes_inputs_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _ = _quality_assembly_fixture(tmp_path, ("severe_frame_crop",))
    real_builder = quality_assembler.build_quality_exclusion_manifest

    def mutate_bound_input_after_manifest(**kwargs):
        result = real_builder(**kwargs)
        args["teacher_queue"].write_bytes(
            args["teacher_queue"].read_bytes() + b"\n"
        )
        return result

    monkeypatch.setattr(
        quality_assembler,
        "build_quality_exclusion_manifest",
        mutate_bound_input_after_manifest,
    )
    with pytest.raises(RuntimeError, match="input changed before publish"):
        assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()
    assert not list(tmp_path.glob(f".{args['output_dir'].name}.*"))


def test_operational_quality_assembler_never_replaces_racing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _ = _quality_assembly_fixture(tmp_path, ("severe_frame_crop",))
    real_stable_directory = quality_assembler._stable_directory

    def create_destination_after_initial_check(path: Path, *, description: str):
        resolved = real_stable_directory(path, description=description)
        if description == "output parent":
            args["output_dir"].mkdir()
            (args["output_dir"] / "sentinel.txt").write_text(
                "do-not-replace", encoding="utf-8"
            )
        return resolved

    monkeypatch.setattr(
        quality_assembler,
        "_stable_directory",
        create_destination_after_initial_check,
    )
    with pytest.raises(FileExistsError, match="overwrite immutable output"):
        assemble_operational_quality_exclusions(**args)
    assert {
        path.name: path.read_text(encoding="utf-8")
        for path in args["output_dir"].iterdir()
    } == {"sentinel.txt": "do-not-replace"}
    assert not list(tmp_path.glob(f".{args['output_dir'].name}.*"))


def test_operational_quality_assembler_recomputes_forged_usable_decision(
    tmp_path: Path,
) -> None:
    args, _ = _quality_assembly_fixture(tmp_path, ("severe_frame_crop",))
    rejection_path = args["teacher_output_dir"] / ARTIFACT_NAMES["rejections"]
    rejection_report = json.loads(rejection_path.read_text(encoding="utf-8"))
    rejection = rejection_report["rejections"][0]
    old_reason = rejection["reasons"][0]
    rejection["teacher_training_usable"] = True
    rejection["teacher_quality_reason"] = "usable"
    rejection["reasons"] = ["minimum_confidence_below_threshold"]
    del rejection_report["reason_counts"][old_reason]
    rejection_report["reason_counts"] = {"minimum_confidence_below_threshold": 1}
    rejection_path.write_text(
        json.dumps(rejection_report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    lineage_path = args["teacher_output_dir"] / ARTIFACT_NAMES["lineage"]
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["output_digests"]["rejections_sha256"] = hashlib.sha256(
        rejection_path.read_bytes()
    ).hexdigest()
    lineage_path.write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="decision does not match rejection"):
        assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()


def test_operational_quality_assembler_rejects_output_inside_teacher_authority(
    tmp_path: Path,
) -> None:
    args, _ = _quality_assembly_fixture(tmp_path, ("severe_frame_crop",))
    teacher_files = {
        path.name: path.read_bytes() for path in args["teacher_output_dir"].iterdir()
    }
    args["output_dir"] = args["teacher_output_dir"] / "nested-output"

    with pytest.raises(ValueError, match="inside teacher output authority"):
        assemble_operational_quality_exclusions(**args)
    assert {
        path.name: path.read_bytes() for path in args["teacher_output_dir"].iterdir()
    } == teacher_files


def test_operational_quality_assembler_requires_complete_queue_partition(
    tmp_path: Path,
) -> None:
    args, _ = _quality_assembly_fixture(
        tmp_path, ("severe_frame_crop", "boundary_unreadable")
    )
    rejection_path = args["teacher_output_dir"] / ARTIFACT_NAMES["rejections"]
    rejection_report = json.loads(rejection_path.read_text(encoding="utf-8"))
    removed = rejection_report["rejections"].pop(0)
    removed_reason = removed["reasons"][0]
    rejection_report["rejected_records"] -= 1
    rejection_report["reason_counts"][removed_reason] -= 1
    if rejection_report["reason_counts"][removed_reason] == 0:
        del rejection_report["reason_counts"][removed_reason]
    rejection_path.write_text(
        json.dumps(rejection_report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    lineage_path = args["teacher_output_dir"] / ARTIFACT_NAMES["lineage"]
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["counts"]["rejected_records"] -= 1
    lineage["output_digests"]["rejections_sha256"] = hashlib.sha256(
        rejection_path.read_bytes()
    ).hexdigest()
    lineage_path.write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="do not partition the queue"):
        assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()


def test_operational_quality_assembler_requires_decision_rejection_for_unusable_label(
    tmp_path: Path,
) -> None:
    args, _ = _quality_assembly_fixture(
        tmp_path, ("severe_frame_crop", "boundary_unreadable")
    )
    rejection_path = args["teacher_output_dir"] / ARTIFACT_NAMES["rejections"]
    rejection_report = json.loads(rejection_path.read_text(encoding="utf-8"))
    replaced = rejection_report["rejections"][0]
    old_reason = replaced["reasons"][0]
    rejection_report["rejections"][0] = {
        "sha256": replaced["sha256"],
        "reasons": ["missing_teacher_label"],
    }
    rejection_report["reason_counts"][old_reason] -= 1
    if rejection_report["reason_counts"][old_reason] == 0:
        del rejection_report["reason_counts"][old_reason]
    rejection_report["reason_counts"]["missing_teacher_label"] = 1
    rejection_report["reason_counts"] = dict(
        sorted(rejection_report["reason_counts"].items())
    )
    rejection_path.write_text(
        json.dumps(rejection_report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    lineage_path = args["teacher_output_dir"] / ARTIFACT_NAMES["lineage"]
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["output_digests"]["rejections_sha256"] = hashlib.sha256(
        rejection_path.read_bytes()
    ).hexdigest()
    lineage_path.write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match its exact consensus"):
        assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()
