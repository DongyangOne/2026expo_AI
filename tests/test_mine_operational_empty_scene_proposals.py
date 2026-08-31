import builtins
import csv
import hashlib
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.audit_verifier_dataset import audit_manifest
from scripts.mine_operational_empty_scene_proposals import (
    ARTIFACT_NAMES,
    CLASS_NAMES,
    DETECTOR_CONFIDENCE,
    DETECTOR_NMS_IOU,
    PredictedFrame,
    Proposal,
    SourceRecord,
    iter_detector_predictions,
    mine_operational_empty_scene_proposals,
)
from scripts.verifier_preprocessing_contract import (
    BBOX_ROUNDING,
    COLOR_CONVERSION,
    CONTRACT_VERSION,
    LETTERBOX_ALIGNMENT,
    RESIZE_INTERPOLATION,
    RESIZE_ROUNDING,
)


def _write_image(path: Path, *, marker: int, width: int = 100, height: int = 60) -> str:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(width, dtype=np.uint8)[None, :]
    image[:, :, 1] = np.arange(height, dtype=np.uint8)[:, None]
    image[:, :, 2] = marker
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(encoded.tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spec() -> dict:
    return {
        "format_version": 1,
        "artifact_role": "offline_candidate_spec_not_production_authorization",
        "detector": {
            "task": "detect",
            "model_reference": "best.pt",
            "input_size": 640,
            "candidate_confidence": 0.10,
            "nms_iou": 0.70,
            "proposal_selection": "highest_confidence_then_original_order",
        },
        "crop": {
            "source": "selected_detector_bbox",
            "padding_ratio": 0.08,
            "clip_to_source": True,
            "preprocessing_contract_version": CONTRACT_VERSION,
            "bbox_rounding": BBOX_ROUNDING,
            "resize": "aspect_preserving_letterbox",
            "resize_rounding": RESIZE_ROUNDING,
            "resize_interpolation": RESIZE_INTERPOLATION,
            "letterbox_size": 320,
            "letterbox_fill": 114,
            "letterbox_alignment": LETTERBOX_ALIGNMENT,
            "color_conversion": COLOR_CONVERSION,
            "normalization": {
                "input_scale": 255.0,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "layout": "NCHW",
                "dtype": "float32",
            },
            "jpeg_quality": 92,
        },
        "detector_classes": list(CLASS_NAMES),
    }


def _write_spec(path: Path, *, candidate_confidence: float = 0.10) -> None:
    value = _spec()
    value["detector"]["candidate_confidence"] = candidate_confidence
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _inventory_row(
    image: Path,
    sha256: str,
    *,
    role: str = "train",
    fold: str = "operational_teacher_v1",
    group: str = "object-group-1",
    session: str = "capture-session-1",
) -> dict[str, object]:
    return {
        "sample_id": f"source-{sha256[:12]}",
        "role": role,
        "split_role": role,
        "fold": fold,
        "filepath": image.name,
        "split": "training" if role == "train" else role,
        "source_id": sha256,
        "source_sha256": sha256,
        "image_sha256": sha256,
        "content_identity": f"sha256:{sha256}",
        "object_group": group,
        "capture_session": session,
        "origin": "operational_empty_scene_vlm_teacher_source",
        "selection_reason": "exact_tuple_high_confidence_negative_source_inventory",
        "material": 9,
        "category": "background",
        "teacher_material": "negative",
        "dent": -1,
        "label": -1,
        "foreign_material": 0,
        "source_object_count": 0,
        "source_width": 100,
        "source_height": 60,
        "source_bbox_x": "",
        "source_bbox_y": "",
        "source_bbox_w": "",
        "source_bbox_h": "",
        "bbox_x1": "",
        "bbox_y1": "",
        "bbox_x2": "",
        "bbox_y2": "",
        "bbox_area_ratio": "",
        "bbox_source": "",
        "teacher_minimum_confidence": "0.91",
        "teacher_consensus": "true",
        "teacher_consensus_votes": 2,
        "teacher_pass_count": 2,
        "pseudo_label": "true",
        "ground_truth_authority": "vlm_teacher_pseudo_label_train_only",
        "blind_test_eligible": "false",
        "training_crop_ready": "false",
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _fixture_files(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "detector.pt"
    model.write_bytes(b"frozen detector bytes")
    spec = tmp_path / "inference.json"
    _write_spec(spec)
    return model, spec


def test_runtime_top1_crop_preserves_lineage_and_reports_no_proposal(tmp_path, monkeypatch):
    selected_image = tmp_path / "selected.png"
    empty_image = tmp_path / "empty.png"
    selected_sha = _write_image(selected_image, marker=180)
    empty_sha = _write_image(empty_image, marker=220)
    inventory = tmp_path / "empty-scenes.csv"
    # Reverse source order intentionally; validated inference order is stable by SHA.
    _write_csv(
        inventory,
        [
            _inventory_row(empty_image, empty_sha, group="empty-group", session="empty-session"),
            _inventory_row(selected_image, selected_sha, group="selected-group", session="selected-session"),
        ],
    )
    model, spec = _fixture_files(tmp_path)

    real_import = builtins.__import__

    def no_ultralytics(name, *args, **kwargs):
        if name == "ultralytics" or name.startswith("ultralytics."):
            raise AssertionError("injected predictions must not import Ultralytics")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ultralytics)

    def predictions(sources):
        assert [item.source_sha256 for item in sources] == sorted(
            [selected_sha, empty_sha]
        )
        for source in sources:
            if source.path.name == "selected.png":
                # Equal confidence proves original proposal order is the tie-breaker.
                proposals = (
                    Proposal(2, 0.80, (10.0, 10.0, 30.0, 50.0), "paper"),
                    Proposal(3, 0.80, (40.0, 5.0, 80.0, 45.0), "plastic"),
                    Proposal(5, 0.09, (1.0, 1.0, 20.0, 20.0), "vinyl"),
                )
            else:
                proposals = ()
            yield PredictedFrame(source, 100, 60, proposals)

    output = tmp_path / "mined"
    result = mine_operational_empty_scene_proposals(
        input_inventory=inventory,
        detector_model=model,
        inference_spec=spec,
        output_dir=output,
        prediction_provider=predictions,
    )

    assert result["counts"] == {
        "validated_sources": 2,
        "written_background_crops": 1,
        "no_eligible_proposal_frames": 1,
        "written_crop_bytes": result["counts"]["written_crop_bytes"],
    }
    assert result["proposal_stats"] == {
        "below_candidate_confidence": 1,
        "discarded_by_runtime_top1": 1,
        "frames_seen": 2,
        "frames_without_eligible_proposal": 1,
        "proposals_seen": 3,
        "proposals_selected": 1,
    }
    rows = _read_csv(output / ARTIFACT_NAMES["csv"])
    assert len(rows) == 1
    row = rows[0]
    assert row["source_sha256"] == selected_sha
    assert row["object_group"] == "selected-group"
    assert row["capture_session"] == "selected-session"
    assert row["proposal_index"] == "0"
    assert row["predicted_class_name"] == "paper"
    assert row["predicted_confidence"] == "0.80000000"
    assert row["material"] == "9"
    assert row["category"] == "background"
    assert row["source_object_count"] == "0"
    assert row["training_crop_ready"] == "true"
    assert row["detector_model_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert row["inference_spec_sha256"] == hashlib.sha256(spec.read_bytes()).hexdigest()
    assert row["source_inventory_sha256"] == hashlib.sha256(inventory.read_bytes()).hexdigest()
    # width 20 with 8% padding: floor(8.4), ceil(31.6).
    assert [row[field] for field in (
        "crop_bbox_x1", "crop_bbox_y1", "crop_bbox_x2", "crop_bbox_y2"
    )] == ["8", "6", "32", "54"]
    crop_path = output / row["filepath"]
    assert hashlib.sha256(crop_path.read_bytes()).hexdigest() == row["image_sha256"]
    crop = cv2.imdecode(np.frombuffer(crop_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert crop.shape == (320, 320, 3)

    no_proposal = json.loads(
        (output / ARTIFACT_NAMES["no_proposal"]).read_text(encoding="utf-8")
    )
    assert no_proposal["policy"] == "report_only_never_full_frame_crop"
    assert no_proposal["frames"] == [
        {
            "capture_session": "empty-session",
            "object_group": "empty-group",
            "reasons": ["no_detector_proposals"],
            "source_sha256": empty_sha,
        }
    ]
    assert json.loads(
        (output / ARTIFACT_NAMES["jsonl"]).read_text(encoding="utf-8").strip()
    ) == row
    audit = audit_manifest(
        output / ARTIFACT_NAMES["csv"],
        require_source_references=True,
        allow_partial_class_coverage=True,
    )
    assert audit["problems"] == []


def test_outputs_are_byte_deterministic_across_fresh_directories_and_jsonl_input(tmp_path):
    image = tmp_path / "scene.png"
    sha = _write_image(image, marker=190)
    row = _inventory_row(image, sha)
    inventory = tmp_path / "scenes.jsonl"
    inventory.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    model, spec = _fixture_files(tmp_path)

    def predictions(sources):
        for source in sources:
            yield PredictedFrame(
                source,
                100,
                60,
                (Proposal(4, 0.91, (10.25, 4.5, 85.75, 55.5), "styrofoam"),),
            )

    first = tmp_path / "first"
    second = tmp_path / "second"
    for output in (first, second):
        mine_operational_empty_scene_proposals(
            input_inventory=inventory,
            detector_model=model,
            inference_spec=spec,
            output_dir=output,
            prediction_provider=predictions,
        )
    assert _tree_bytes(first) == _tree_bytes(second)


def test_invalid_and_below_threshold_detector_proposals_are_report_only(tmp_path):
    image = tmp_path / "scene.png"
    sha = _write_image(image, marker=125)
    inventory = tmp_path / "inventory.csv"
    _write_csv(inventory, [_inventory_row(image, sha)])
    model, spec = _fixture_files(tmp_path)

    def predictions(sources):
        source = sources[0]
        yield PredictedFrame(
            source,
            100,
            60,
            (
                Proposal(2, 0.99, (20.0, 20.0, 20.0, 40.0), "paper"),
                Proposal(3, 0.09, (1.0, 1.0, 20.0, 20.0), "plastic"),
            ),
        )

    output = tmp_path / "output"
    result = mine_operational_empty_scene_proposals(
        input_inventory=inventory,
        detector_model=model,
        inference_spec=spec,
        output_dir=output,
        prediction_provider=predictions,
    )
    assert result["counts"]["written_background_crops"] == 0
    assert result["proposal_stats"] == {
        "below_candidate_confidence": 1,
        "frames_seen": 1,
        "frames_without_eligible_proposal": 1,
        "invalid_proposals": 1,
        "proposals_seen": 2,
    }
    report = json.loads(
        (output / ARTIFACT_NAMES["no_proposal"]).read_text(encoding="utf-8")
    )
    assert report["frames"][0]["reasons"] == [
        "all_valid_proposals_below_confidence",
        "invalid_proposals_excluded",
    ]
    assert not list((output / "crops").rglob("*.jpg"))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda row: row.update(role="calibration", split_role="calibration", split="calibration"), "role must be train"),
        (lambda row: row.update(role="blind_test", split_role="blind_test", split="blind_test"), "role must be train"),
        (lambda row: row.update(fold="hardware41_calibration"), "forbidden calibration/blind/hardware41"),
        (lambda row: row.update(origin="operational_capture_vlm_teacher"), "origin must be"),
    ],
)
def test_forbidden_partition_or_origin_is_rejected_before_detector_import(
    tmp_path, monkeypatch, mutator, message
):
    image = tmp_path / "scene.png"
    sha = _write_image(image, marker=120)
    row = _inventory_row(image, sha)
    mutator(row)
    inventory = tmp_path / "inventory.csv"
    _write_csv(inventory, [row])
    model, spec = _fixture_files(tmp_path)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "ultralytics" or name.startswith("ultralytics."):
            raise AssertionError("Ultralytics imported before input preflight")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match=message):
        mine_operational_empty_scene_proposals(
            input_inventory=inventory,
            detector_model=model,
            inference_spec=spec,
            output_dir=output,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda row: row.update(source_sha256="0" * 64, source_id="0" * 64, image_sha256="0" * 64, content_identity="sha256:" + "0" * 64), "source_sha256 mismatch"),
        (lambda row: row.update(bbox_x1="0"), "bbox_x1 must be blank"),
        (lambda row: row.update(training_crop_ready="true"), "must not already be crop-ready"),
        (lambda row: row.update(teacher_minimum_confidence="0.79"), "teacher confidence"),
    ],
)
def test_hash_geometry_and_teacher_authority_fail_preflight(tmp_path, change, message):
    image = tmp_path / "scene.png"
    sha = _write_image(image, marker=130)
    row = _inventory_row(image, sha)
    change(row)
    inventory = tmp_path / "inventory.csv"
    _write_csv(inventory, [row])
    model, spec = _fixture_files(tmp_path)
    with pytest.raises(ValueError, match=message):
        mine_operational_empty_scene_proposals(
            input_inventory=inventory,
            detector_model=model,
            inference_spec=spec,
            output_dir=tmp_path / "output",
            prediction_provider=lambda _sources: pytest.fail("prediction must not run"),
        )


def test_existing_output_guard_precedes_all_input_and_detector_work(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "ultralytics" or name.startswith("ultralytics."):
            raise AssertionError("Ultralytics imported before overwrite guard")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        mine_operational_empty_scene_proposals(
            input_inventory=tmp_path / "missing.csv",
            detector_model=tmp_path / "missing.pt",
            inference_spec=tmp_path / "missing.json",
            output_dir=output,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_bad_frozen_spec_and_provider_failure_leave_no_published_or_partial_output(tmp_path):
    image = tmp_path / "scene.png"
    sha = _write_image(image, marker=140)
    inventory = tmp_path / "inventory.csv"
    _write_csv(inventory, [_inventory_row(image, sha)])
    model, spec = _fixture_files(tmp_path)
    _write_spec(spec, candidate_confidence=0.11)
    bad_output = tmp_path / "bad-spec-output"
    with pytest.raises(ValueError, match="confidence=0.10"):
        mine_operational_empty_scene_proposals(
            input_inventory=inventory,
            detector_model=model,
            inference_spec=spec,
            output_dir=bad_output,
            prediction_provider=lambda _sources: pytest.fail("prediction must not run"),
        )
    assert not bad_output.exists()

    _write_spec(spec)

    def failing_provider(_sources):
        raise RuntimeError("detector failed")
        yield  # pragma: no cover

    failed_output = tmp_path / "failed-output"
    with pytest.raises(RuntimeError, match="detector failed"):
        mine_operational_empty_scene_proposals(
            input_inventory=inventory,
            detector_model=model,
            inference_spec=spec,
            output_dir=failed_output,
            prediction_provider=failing_provider,
        )
    assert not failed_output.exists()
    assert not list(tmp_path.glob(".failed-output.tmp-*"))


def test_dry_run_validates_every_source_without_loading_or_writing(tmp_path):
    image = tmp_path / "scene.png"
    sha = _write_image(image, marker=150)
    inventory = tmp_path / "inventory.csv"
    _write_csv(inventory, [_inventory_row(image, sha)])
    model, spec = _fixture_files(tmp_path)
    output = tmp_path / "dry-run-output"
    result = mine_operational_empty_scene_proposals(
        input_inventory=inventory,
        detector_model=model,
        inference_spec=spec,
        output_dir=output,
        prediction_provider=lambda _sources: pytest.fail("dry-run must not predict"),
        dry_run=True,
    )
    assert result["dry_run"] is True
    assert result["validated_sources"] == 1
    assert not output.exists()


def test_real_prediction_adapter_passes_exact_runtime_thresholds(monkeypatch, tmp_path):
    image = tmp_path / "scene.png"
    sha = _write_image(image, marker=160)
    source = SourceRecord(
        image,
        sha,
        image.as_posix(),
        "group",
        "session",
        "fold",
        100,
        60,
        0.9,
        2,
        2,
        2,
    )
    calls = []

    class Values:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.value

    class Boxes:
        xyxy = Values([[1.0, 2.0, 20.0, 30.0]])
        cls = Values([2.0])
        conf = Values([0.75])

    class Result:
        orig_shape = (60, 100)
        names = {index: name for index, name in enumerate(CLASS_NAMES)}
        boxes = Boxes()

    class FakeModel:
        names = {index: name for index, name in enumerate(CLASS_NAMES)}

        def predict(self, **kwargs):
            calls.append(kwargs)
            return iter([Result()])

    module = types.ModuleType("ultralytics")
    module.YOLO = lambda *_args, **_kwargs: FakeModel()
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    frames = list(
        iter_detector_predictions(
            [source],
            model_path=tmp_path / "model.pt",
            device="cpu",
            batch=4,
            imgsz=640,
        )
    )
    assert len(frames) == 1
    assert calls[0]["conf"] == DETECTOR_CONFIDENCE
    assert calls[0]["iou"] == DETECTOR_NMS_IOU
    assert calls[0]["imgsz"] == 640
    assert calls[0]["source"] == [str(image)]


def test_prediction_count_mismatch_is_atomic(tmp_path):
    image = tmp_path / "scene.png"
    sha = _write_image(image, marker=170)
    inventory = tmp_path / "inventory.csv"
    _write_csv(inventory, [_inventory_row(image, sha)])
    model, spec = _fixture_files(tmp_path)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="fewer frames"):
        mine_operational_empty_scene_proposals(
            input_inventory=inventory,
            detector_model=model,
            inference_spec=spec,
            output_dir=output,
            prediction_provider=lambda _sources: (),
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".output.tmp-*"))


def test_no_proposal_source_replacement_is_detected_and_atomic(tmp_path):
    image = tmp_path / "scene.png"
    sha = _write_image(image, marker=175)
    inventory = tmp_path / "inventory.csv"
    _write_csv(inventory, [_inventory_row(image, sha)])
    model, spec = _fixture_files(tmp_path)

    def replacing_provider(sources):
        source = sources[0]
        _write_image(source.path, marker=176)
        yield PredictedFrame(source, 100, 60, ())

    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="source image changed after preflight"):
        mine_operational_empty_scene_proposals(
            input_inventory=inventory,
            detector_model=model,
            inference_spec=spec,
            output_dir=output,
            prediction_provider=replacing_provider,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".output.tmp-*"))
