import base64
import csv
import hashlib
import json
import os
import weakref
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

import scripts.validate_v4_background_candidates as validator_module
from scripts.prepare_proposal_verifier_dataset import (
    MANIFEST_FIELDS,
    PredictedFrame,
    Proposal,
)
from scripts.validate_v4_background_candidates import (
    SCHEMA_VERSION,
    validate_manifest,
)
from scripts.verifier_preprocessing_contract import crop_and_letterbox_bgr


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "configs" / "detector_inference_v3.json"
_AUTHORITATIVE_RUNTIME = object()


def _write_jpeg(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, :] = (17, 31, 47)
    assert cv2.imwrite(str(path), pixels)


def _write_expected_crop(
    source: Path,
    crop: Path,
    bbox: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    source_image = cv2.imdecode(
        np.frombuffer(source.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    expected, bounds = crop_and_letterbox_bgr(
        source_image,
        bbox,
        padding=0.08,
        size=320,
        fill=114,
    )
    ok, encoded = cv2.imencode(
        ".jpg", expected, [cv2.IMWRITE_JPEG_QUALITY, 92]
    )
    assert ok
    crop.parent.mkdir(parents=True, exist_ok=True)
    crop.write_bytes(encoded.tobytes())
    return bounds


def _write_manifest(path: Path, row: dict[str, object], extra_fields=()) -> None:
    fields = [*MANIFEST_FIELDS, *extra_fields]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def _fixture(tmp_path: Path) -> dict[str, object]:
    source = tmp_path / "dataset" / "images" / "train" / "source.jpg"
    label = tmp_path / "dataset" / "labels" / "train" / "source.txt"
    _write_jpeg(source, (100, 100))
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("0 0.2 0.2 0.2 0.2\n", encoding="utf-8")

    output_dir = tmp_path / "proposal"
    crop = output_dir / "training" / "background" / "crop.jpg"
    detector_bbox = (70.0, 70.0, 90.0, 90.0)
    crop_bounds = _write_expected_crop(source, crop, detector_bbox)
    detector = tmp_path / "best.pt"
    detector.write_bytes(b"detector-v4-fixture")
    manifest = output_dir / "manifest.csv"
    row: dict[str, object] = {field: "" for field in MANIFEST_FIELDS}
    row.update(
        {
            "filepath": "training/background/crop.jpg",
            "split": "training",
            "source_id": hashlib.sha256(source.read_bytes()).hexdigest(),
            "material": 9,
            "category": "background",
            "dent": -1,
            "label": -1,
            "foreign_material": -1,
            "source_object_count": 1,
            "source_path_b64": base64.urlsafe_b64encode(os.fsencode(source)).decode("ascii"),
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
            "predicted_confidence": "0.42",
            "predicted_bbox_x1": detector_bbox[0],
            "predicted_bbox_y1": detector_bbox[1],
            "predicted_bbox_x2": detector_bbox[2],
            "predicted_bbox_y2": detector_bbox[3],
            "crop_x1": crop_bounds[0],
            "crop_y1": crop_bounds[1],
            "crop_x2": crop_bounds[2],
            "crop_y2": crop_bounds[3],
            "source_width": 100,
            "source_height": 100,
            "crop_bytes": crop.stat().st_size,
        }
    )
    _write_manifest(manifest, row)
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
        "row": row,
        "detector_proposal": Proposal(
            2, 0.42, detector_bbox, "paper"
        ),
        "output": output_dir / "manifest.v4.validated.csv",
        "report": output_dir / "manifest.v4.validation.json",
    }


def _validate(
    fixture: dict[str, object],
    *,
    prediction_provider=_AUTHORITATIVE_RUNTIME,
    diagnostic_only: bool = False,
) -> dict:
    if prediction_provider is _AUTHORITATIVE_RUNTIME:
        proposal = fixture["detector_proposal"]
        replay_roots: list[Path] = []

        def frozen_predictions(
            records,
            *,
            model_path,
            device,
            batch,
            imgsz,
            conf,
            nms_iou,
        ):
            detector = fixture["detector"]
            source = fixture["source"]
            label = fixture["label"]
            events = fixture.get("_events")
            if isinstance(events, list):
                events.append("replay")
            guard_ref = fixture.get("_guard_ref")
            if callable(guard_ref):
                assert guard_ref() is not None
            replay_root = model_path.parents[1]
            replay_roots.append(replay_root)
            assert model_path.name == detector.name
            assert model_path.resolve() != detector.resolve()
            assert model_path.read_bytes() == detector.read_bytes()
            expected_info = json.loads(fixture["info"].read_text(encoding="utf-8"))
            assert device == expected_info["inference"]["device"]
            assert batch == 1
            assert imgsz == 640
            assert conf == 0.10
            assert nms_iou == 0.70
            for record in records:
                assert record.path.read_bytes() == source.read_bytes()
                snapshot_label = validator_module._label_path(record.path)
                assert snapshot_label.read_bytes() == label.read_bytes()
                assert record.path.parents[3] == replay_root
                if fixture.get("_mutate_replay_source"):
                    record.path.write_bytes(b"mutated replay snapshot")
                if fixture.get("_mutate_original_detector"):
                    detector.write_bytes(b"changed original detector")
                yield PredictedFrame(record, 100, 100, (proposal,))

        with patch.object(
            validator_module, "iter_yolo_predictions", frozen_predictions
        ):
            report = validate_manifest(
                input_manifest=fixture["manifest"],
                dataset_info=fixture["info"],
                detector_model=fixture["detector"],
                inference_spec=SPEC_PATH,
                output_manifest=fixture["output"],
                output_report=fixture["report"],
                diagnostic_only=diagnostic_only,
            )
        assert replay_roots
        assert all(not root.exists() for root in replay_roots)
        return report

    return validate_manifest(
        input_manifest=fixture["manifest"],
        dataset_info=fixture["info"],
        detector_model=fixture["detector"],
        inference_spec=SPEC_PATH,
        output_manifest=fixture["output"],
        output_report=fixture["report"],
        prediction_provider=prediction_provider,
        diagnostic_only=diagnostic_only,
    )


def test_rescues_current_manifest_and_preserves_two_object_count_semantics(tmp_path):
    fixture = _fixture(tmp_path)

    report = _validate(fixture)

    assert report["ready_for_lineage_upgrade"] is True
    assert report["blind_test_eligible"] is False
    assert report["production_deployment_authorized"] is False
    provenance = report["contract"]["proposal_provenance"]
    assert provenance["provider_kind"] == "frozen_yolo_runtime"
    assert provenance["runtime_detector_executed"] is True
    assert provenance["runtime_top1_replayed"] is True
    assert provenance["detector_replay_used_unique_snapshot"] is True
    assert provenance["source_and_label_replay_used_unique_snapshots"] is True
    assert provenance["replay_snapshots_verified_after_inference"] is True
    assert provenance["original_generation_event_cryptographically_attested"] is False
    assert provenance["production_or_blind_authority"] is False
    with fixture["output"].open(encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file))
    assert row["manifest_schema_version"] == SCHEMA_VERSION
    assert row["source_object_count"] == "1"
    assert row["crop_object_count"] == "0"
    assert row["background_exclusion_policy"] == "strict-zero-intersection"
    assert row["blind_test_eligible"] == "false"
    stored = json.loads(fixture["report"].read_text(encoding="utf-8"))
    assert stored == report


def test_runtime_diagnostic_replay_never_grants_lineage_authority(tmp_path):
    fixture = _fixture(tmp_path)

    report = _validate(fixture, diagnostic_only=True)

    assert report["artifact_role"] == (
        "v4_runtime_replay_diagnostic_not_lineage_blind_or_deployment_authority"
    )
    assert report["ready_for_lineage_upgrade"] is False
    assert report["lineage_execution_authorized"] is False
    assert report["blind_test_eligible"] is False
    assert report["production_deployment_authorized"] is False
    provenance = report["contract"]["proposal_provenance"]
    assert provenance["provider_kind"] == "frozen_yolo_runtime"
    assert provenance["runtime_detector_executed"] is True
    assert provenance["runtime_top1_replayed"] is True
    assert provenance["proposal_class_confidence_bbox_matched"] is True


def test_cuda_guard_precedes_scan_and_survives_runtime_replay(tmp_path):
    fixture = _fixture(tmp_path)
    info = json.loads(fixture["info"].read_text(encoding="utf-8"))
    info["inference"]["device"] = "0"
    fixture["info"].write_text(json.dumps(info), encoding="utf-8")
    events: list[str] = []
    fixture["_events"] = events

    class Guard:
        pass

    def initialize(device):
        assert device == "0"
        guard = Guard()
        fixture["_guard_ref"] = weakref.ref(guard)
        events.append("init")
        return guard

    original_validate_rows = validator_module.validate_rows

    def validate_rows_after_guard(*args, **kwargs):
        assert fixture["_guard_ref"]() is not None
        events.append("scan")
        return original_validate_rows(*args, **kwargs)

    with (
        patch.object(
            validator_module,
            "eager_initialize_cuda_context",
            initialize,
        ),
        patch.object(validator_module, "validate_rows", validate_rows_after_guard),
    ):
        report = _validate(fixture)

    assert events == ["init", "scan", "replay"]
    assert fixture["_guard_ref"]() is None
    provenance = report["contract"]["proposal_provenance"]
    assert provenance["cuda_client_initialized_before_source_crop_scan"] is True


def test_custom_prediction_provider_is_non_authoritative_and_not_lineage_ready(tmp_path):
    fixture = _fixture(tmp_path)
    proposal = fixture["detector_proposal"]
    info = json.loads(fixture["info"].read_text(encoding="utf-8"))
    info["inference"]["batch"] = "unused-by-custom-provider"
    fixture["info"].write_text(json.dumps(info), encoding="utf-8")

    def custom_provider(records):
        for record in records:
            yield PredictedFrame(record, 100, 100, (proposal,))

    with patch.object(
        validator_module,
        "eager_initialize_cuda_context",
        side_effect=AssertionError("custom provider must not initialize CUDA"),
    ):
        report = _validate(fixture, prediction_provider=custom_provider)

    assert report["artifact_role"].startswith("v4_custom_provider_diagnostics")
    assert report["ready_for_lineage_upgrade"] is False
    provenance = report["contract"]["proposal_provenance"]
    assert provenance["provider_kind"] == "custom_non_authoritative"
    assert provenance["runtime_detector_executed"] is False
    assert provenance["runtime_top1_replayed"] is False
    assert provenance["provided_top1_predictions_matched"] is True
    assert provenance["detector_artifact_bytes_bound"] is False
    assert provenance["detector_replay_used_unique_snapshot"] is False
    assert provenance["source_and_label_replay_used_unique_snapshots"] is False


def test_original_detector_is_rehashed_after_runtime_replay(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["_mutate_original_detector"] = True

    with pytest.raises(ValueError, match="original detector model changed"):
        _validate(fixture)

    assert not fixture["output"].exists()
    assert not fixture["report"].exists()
    assert not list(fixture["manifest"].parent.glob(".v4-detector-replay-*"))


def test_mutated_private_source_snapshot_cannot_publish_report(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["_mutate_replay_source"] = True

    with pytest.raises(ValueError, match="detector replay source snapshot changed"):
        _validate(fixture)

    assert not fixture["output"].exists()
    assert not fixture["report"].exists()
    assert not list(fixture["manifest"].parent.glob(".v4-detector-replay-*"))


def test_missing_label_is_not_treated_as_empty_scene_and_publishes_nothing(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["label"].unlink()

    with pytest.raises(ValueError, match="explicit YOLO label file is required"):
        _validate(fixture)

    assert not fixture["output"].exists()
    assert not fixture["report"].exists()


def test_random_320_crop_cannot_be_injected(tmp_path):
    fixture = _fixture(tmp_path)
    random_pixels = np.random.default_rng(7).integers(
        0, 256, size=(320, 320, 3), dtype=np.uint8
    )
    ok, encoded = cv2.imencode(
        ".jpg", random_pixels, [cv2.IMWRITE_JPEG_QUALITY, 92]
    )
    assert ok
    fixture["crop"].write_bytes(encoded.tobytes())

    with pytest.raises(ValueError, match="frozen source/bbox transform"):
        _validate(fixture)

    assert not fixture["output"].exists()
    assert not fixture["report"].exists()


def test_positive_assignment_requires_actual_frozen_iou(tmp_path):
    fixture = _fixture(tmp_path)
    row = dict(fixture["row"])
    row.update(
        {
            "material": 0,
            "category": "can",
            "assignment": "positive_iou",
            "matched_iou": "1.00000000",
        }
    )
    _write_manifest(fixture["manifest"], row)

    with pytest.raises(ValueError, match="matched_iou disagrees|frozen IoU policy"):
        _validate(fixture)

    assert not fixture["output"].exists()
    assert not fixture["report"].exists()


def test_appended_crop_bytes_are_rejected_even_when_jpeg_still_decodes(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["crop"].write_bytes(fixture["crop"].read_bytes() + b"tampered")
    assert cv2.imread(str(fixture["crop"]), cv2.IMREAD_COLOR) is not None

    with pytest.raises(ValueError, match="frozen source/bbox transform"):
        _validate(fixture)

    assert not fixture["output"].exists()
    assert not fixture["report"].exists()


def test_one_pixel_expanded_gt_intersection_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    row = dict(fixture["row"])
    row.update(
        {
            "predicted_bbox_x1": 32,
            "predicted_bbox_y1": 32,
            "predicted_bbox_x2": 42,
            "predicted_bbox_y2": 42,
            "crop_x1": 31,
            "crop_y1": 31,
            "crop_x2": 43,
            "crop_y2": 43,
        }
    )
    _write_manifest(fixture["manifest"], row)

    with pytest.raises(ValueError, match="intersects expanded ground truth"):
        _validate(fixture)

    assert not fixture["output"].exists()
    assert not fixture["report"].exists()


def test_declared_tampered_detector_binding_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    row = dict(fixture["row"])
    row["detector_model_sha256"] = "0" * 64
    _write_manifest(fixture["manifest"], row, extra_fields=("detector_model_sha256",))

    with pytest.raises(ValueError, match="detector_model_sha256 conflicts"):
        _validate(fixture)

    assert not fixture["output"].exists()
    assert not fixture["report"].exists()


def test_detector_replay_rejects_tampered_manifest_proposal(tmp_path):
    fixture = _fixture(tmp_path)
    row = dict(fixture["row"])
    row.update(
        predicted_class_id=3,
        predicted_class_name="plastic",
    )
    _write_manifest(fixture["manifest"], row)

    with pytest.raises(ValueError, match="detector replay class mismatch"):
        _validate(fixture)

    assert not fixture["output"].exists()
    assert not fixture["report"].exists()


def test_never_overwrites_validated_artifacts(tmp_path):
    fixture = _fixture(tmp_path)
    _validate(fixture)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _validate(fixture)
