import base64
import csv
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

import scripts.audit_v4_detector_replay_drift as audit_module
from scripts.audit_v4_detector_replay_drift import audit_detector_replay_drift
from scripts.prepare_proposal_verifier_dataset import (
    MANIFEST_FIELDS,
    PredictedFrame,
    Proposal,
)
from scripts.verifier_preprocessing_contract import crop_and_letterbox_bgr


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "configs" / "detector_inference_v3.json"


def _fixture(tmp_path: Path) -> dict[str, object]:
    source = tmp_path / "dataset" / "images" / "train" / "source.jpg"
    label = tmp_path / "dataset" / "labels" / "train" / "source.txt"
    source.parent.mkdir(parents=True)
    pixels = np.zeros((100, 100, 3), dtype=np.uint8)
    pixels[:, :] = (17, 31, 47)
    assert cv2.imwrite(str(source), pixels)
    label.parent.mkdir(parents=True)
    label.write_text("0 0.2 0.2 0.2 0.2\n", encoding="utf-8")

    output_dir = tmp_path / "proposal"
    crop = output_dir / "training" / "background" / "crop.jpg"
    declared_bbox = (50.0, 50.0, 70.0, 70.0)
    expected, crop_bounds = crop_and_letterbox_bgr(
        pixels,
        declared_bbox,
        padding=0.08,
        size=320,
        fill=114,
    )
    ok, encoded = cv2.imencode(
        ".jpg", expected, [cv2.IMWRITE_JPEG_QUALITY, 92]
    )
    assert ok
    crop.parent.mkdir(parents=True)
    crop.write_bytes(encoded.tobytes())

    detector = tmp_path / "best.pt"
    detector.write_bytes(b"detector-weights")
    manifest = output_dir / "manifest.csv"
    row = {
        "filepath": "training/background/crop.jpg",
        "split": "training",
        "source_id": hashlib.sha256(source.read_bytes()).hexdigest(),
        "material": 9,
        "category": "background",
        "dent": -1,
        "label": -1,
        "foreign_material": -1,
        "source_object_count": 1,
        "source_path_b64": base64.urlsafe_b64encode(os.fsencode(source)).decode(
            "ascii"
        ),
        "proposal_index": 0,
        "assignment": "low_iou",
        "matched_iou": "0.00000000",
        "gt_class_id": 0,
        "gt_class_name": "can",
        "gt_bbox_x1": 10,
        "gt_bbox_y1": 10,
        "gt_bbox_x2": 30,
        "gt_bbox_y2": 30,
        "predicted_class_id": 2,
        "predicted_class_name": "paper",
        "predicted_confidence": "0.42000000",
        "predicted_bbox_x1": declared_bbox[0],
        "predicted_bbox_y1": declared_bbox[1],
        "predicted_bbox_x2": declared_bbox[2],
        "predicted_bbox_y2": declared_bbox[3],
        "crop_x1": crop_bounds[0],
        "crop_y1": crop_bounds[1],
        "crop_x2": crop_bounds[2],
        "crop_y2": crop_bounds[3],
        "source_width": 100,
        "source_height": 100,
        "crop_bytes": crop.stat().st_size,
    }
    output_dir.mkdir(exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    info = output_dir / "dataset_info.json"
    info.write_text(
        json.dumps(
            {
                "model": str(detector.resolve()),
                "manifest": str(manifest.resolve()),
                "proposal_policy": {
                    "selection_mode": "runtime-top1",
                    "background_policy": "strict-zero-intersection",
                    "background_gt_margin": 0.10,
                },
                "inference": {
                    "device": "cpu",
                    "batch": 1,
                    "imgsz": 640,
                    "conf": 0.10,
                    "nms_iou": 0.70,
                },
                "assignment": {
                    "positive_iou_inclusive": 0.50,
                    "negative_iou_inclusive": 0.10,
                    "ambiguous_iou_skipped": True,
                },
                "crop": {"size": 320, "padding": 0.08, "jpeg_quality": 92},
            }
        ),
        encoding="utf-8",
    )
    return {
        "source": source,
        "label": label,
        "crop": crop,
        "detector": detector,
        "manifest": manifest,
        "info": info,
        "output": tmp_path / "audit" / "detector_replay_drift.json",
        "declared_proposal": Proposal(2, 0.42, declared_bbox, "paper"),
    }


def _audit(fixture: dict[str, object], provider) -> dict[str, object]:
    return audit_detector_replay_drift(
        input_manifest=fixture["manifest"],
        dataset_info=fixture["info"],
        detector_model=fixture["detector"],
        inference_spec=SPEC_PATH,
        output_report=fixture["output"],
        prediction_provider=provider,
    )


def test_custom_drift_audit_records_semantics_without_authorizing_use(tmp_path):
    fixture = _fixture(tmp_path)
    before = {
        name: hashlib.sha256(fixture[name].read_bytes()).hexdigest()
        for name in ("source", "label", "crop", "detector", "manifest")
    }

    def provider(records):
        for record in records:
            proposals = (
                Proposal(2, 0.20, (50.0, 50.0, 70.0, 70.0), "paper"),
                Proposal(3, 0.80, (10.0, 10.0, 30.0, 30.0), "plastic"),
            )
            yield PredictedFrame(record, 100, 100, proposals)

    report = _audit(fixture, provider)

    assert report["artifact_role"].startswith(
        "v4_historical_manifest_replay_drift_diagnostic"
    )
    assert report["diagnostic_only"] is True
    assert report["ready_for_lineage_upgrade"] is False
    assert report["training_eligible"] is False
    assert report["blind_test_eligible"] is False
    assert report["production_deployment_authorized"] is False
    assert report["authority"] == {
        "provider_kind": "custom_test_provider_non_authoritative",
        "runtime_detector_executed": False,
        "cuda_runtime_verified": False,
        "input_validator_report_bound": False,
        "input_authority": (
            "source_manifest_contract_checked_but_not_validator_approved"
        ),
        "lineage_authority": False,
        "training_authority": False,
        "blind_test_authority": False,
        "deployment_authority": False,
    }
    assert report["interpretation"]["numeric_tolerance_pass_fail_applied"] is False
    counts = report["replay"]["hard_semantic_mismatch_counts"]
    assert counts["top1_index_changed"] == 1
    assert counts["class_changed"] == 1
    assert counts["crop_bounds_changed"] == 1
    assert counts["assignment_material_changed"] == 1
    assert counts["strict_zero_intersection_decision_changed"] == 1
    assert report["replay"]["assignment_transition_counts"]["9/low_iou->0/positive_iou"] == 1
    assert report["replay"]["strict_zero_intersection_transition_counts"][
        "accepted_zero_intersection->not_background"
    ] == 1
    assert report["replay"]["replayed_proposal_count"]["distribution"]["max"] == 2.0
    assert report["replay"]["replayed_proposal_count"]["histogram"] == {"2": 1}
    assert report["replay"]["confidence_abs_drift"]["distribution"]["max"] == pytest.approx(0.38)
    assert report["replay"]["confidence_signed_drift"]["distribution"]["max"] == pytest.approx(0.38)
    assert report["replay"]["strata"]["training/background"][
        "hard_semantic_mismatch_sources"
    ] == 1
    assert len(report["bindings"]["source_label_crop_binding_sha256"]) == 64
    assert json.loads(fixture["output"].read_text(encoding="utf-8")) == report
    after = {
        name: hashlib.sha256(fixture[name].read_bytes()).hexdigest()
        for name in before
    }
    assert after == before
    assert not list(fixture["manifest"].parent.glob(".v4-detector-drift-audit-*"))


def test_no_proposal_is_a_diagnostic_count_not_a_failed_report(tmp_path):
    fixture = _fixture(tmp_path)

    def provider(records):
        for record in records:
            yield PredictedFrame(record, 100, 100, ())

    report = _audit(fixture, provider)

    assert report["completion"]["source_replay_complete"] is True
    assert report["completion"]["full_metric_coverage"] is False
    assert report["completion"]["metric_skip_reasons"] == {"no_proposal": 1}
    assert report["replay"]["hard_semantic_mismatch_counts"]["no_proposal"] == 1
    assert report["replay"]["hard_semantic_mismatch_sources"] == 1


def test_existing_output_is_rejected_before_provider_or_partial_write(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["output"].parent.mkdir(parents=True)
    fixture["output"].write_text("existing", encoding="utf-8")

    def provider(_records):
        raise AssertionError("provider must not run")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _audit(fixture, provider)

    assert fixture["output"].read_text(encoding="utf-8") == "existing"


def test_provider_failure_leaves_no_partial_report(tmp_path):
    fixture = _fixture(tmp_path)

    def provider(_records):
        raise RuntimeError("injected replay failure")

    with pytest.raises(RuntimeError, match="injected replay failure"):
        _audit(fixture, provider)

    assert not fixture["output"].exists()
    assert not list(fixture["manifest"].parent.glob(".v4-detector-drift-audit-*"))


def test_report_must_use_separate_audit_directory(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["output"] = fixture["manifest"].parent / "drift.json"

    with pytest.raises(ValueError, match="separate audit directory"):
        _audit(fixture, lambda _records: ())

    assert not fixture["output"].exists()


def test_cli_rejects_cpu_before_claiming_cuda_runtime_evidence(tmp_path):
    fixture = _fixture(tmp_path)

    with pytest.raises(ValueError, match="requires logical CUDA device 0"):
        audit_detector_replay_drift(
            input_manifest=fixture["manifest"],
            dataset_info=fixture["info"],
            detector_model=fixture["detector"],
            inference_spec=SPEC_PATH,
            output_report=fixture["output"],
        )

    assert not fixture["output"].exists()


def test_cli_authoritative_path_keeps_guard_and_uses_snapshots(tmp_path):
    fixture = _fixture(tmp_path)
    info = json.loads(fixture["info"].read_text(encoding="utf-8"))
    info["inference"]["device"] = "0"
    fixture["info"].write_text(json.dumps(info), encoding="utf-8")
    events: list[str] = []
    guard = object()

    def initialize(device):
        assert device == "0"
        events.append("init")
        return guard

    def predictions(records, *, model_path, device, batch, imgsz, conf, nms_iou):
        events.append("replay")
        assert model_path.name == fixture["detector"].name
        assert model_path.resolve() != fixture["detector"].resolve()
        assert model_path.read_bytes() == fixture["detector"].read_bytes()
        assert (device, batch, imgsz, conf, nms_iou) == ("0", 1, 640, 0.10, 0.70)
        for record in records:
            assert record.path.read_bytes() == fixture["source"].read_bytes()
            yield PredictedFrame(record, 100, 100, (fixture["declared_proposal"],))

    with (
        patch.object(audit_module, "eager_initialize_cuda_context", initialize),
        patch.object(
            audit_module,
            "_runtime_metadata",
            return_value={
                "cuda_observed": True,
                "runtime_identity_authoritative": True,
            },
        ),
        patch.object(audit_module, "iter_yolo_predictions", predictions),
    ):
        report = audit_detector_replay_drift(
            input_manifest=fixture["manifest"],
            dataset_info=fixture["info"],
            detector_model=fixture["detector"],
            inference_spec=SPEC_PATH,
            output_report=fixture["output"],
        )

    assert events == ["init", "replay"]
    assert report["authority"]["runtime_detector_executed"] is True
    assert report["authority"]["cuda_runtime_verified"] is True
    assert report["authority"]["provider_kind"] == "frozen_yolo_runtime"
    assert report["static_contract"]["cuda_client_initialized_before_source_crop_scan"] is True
    assert report["static_contract"]["detector_replay_used_unique_snapshot"] is True
