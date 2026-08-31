import base64
import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

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
    prediction_provider=None,
) -> dict:
    if prediction_provider is None:
        proposal = fixture["detector_proposal"]

        def prediction_provider(records):
            for record in records:
                yield PredictedFrame(record, 100, 100, (proposal,))

    return validate_manifest(
        input_manifest=fixture["manifest"],
        dataset_info=fixture["info"],
        detector_model=fixture["detector"],
        inference_spec=SPEC_PATH,
        output_manifest=fixture["output"],
        output_report=fixture["report"],
        prediction_provider=prediction_provider,
    )


def test_rescues_current_manifest_and_preserves_two_object_count_semantics(tmp_path):
    fixture = _fixture(tmp_path)

    report = _validate(fixture)

    assert report["ready_for_lineage_upgrade"] is True
    assert report["blind_test_eligible"] is False
    assert report["production_deployment_authorized"] is False
    provenance = report["contract"]["proposal_provenance"]
    assert provenance["runtime_top1_replayed"] is True
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
