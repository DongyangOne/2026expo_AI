"""Real source-evidence/geometry contracts with CPU-only detector replay doubles."""

import base64
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

from scripts import build_operational_source_evidence as adapter
from scripts import validate_v4_background_candidates as validator
from scripts.prepare_proposal_verifier_dataset import CLASS_NAMES, PredictedFrame, Proposal


def _module(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("_operational_validator_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_row(case):
    row = case["row"]
    with case["manifest"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


@pytest.fixture
def case(tmp_path):
    evidence_helpers = _module("test_build_operational_source_evidence")
    inputs = evidence_helpers.case.__wrapped__(tmp_path)
    bundle, _ = evidence_helpers._bundle(inputs)
    record = adapter.validate_source_evidence_bundle(bundle)[0]
    helpers = _module("test_validate_v4_background_candidates")
    fixture = helpers._fixture(tmp_path / "validation")
    source = Path(record["source_filepath"])
    bbox = tuple(record["source_bbox_xyxy"])
    predicted_class = (record["material"] + 1) % len(CLASS_NAMES)
    crop = fixture["manifest"].parent / "training" / record["category"] / "operational.jpg"
    bounds = helpers._write_expected_crop(source, crop, bbox)
    row = fixture["row"]
    row.update({
        "filepath": crop.as_posix(), "source_path_b64": base64.urlsafe_b64encode(os.fsencode(source)).decode("ascii"),
        "material": record["material"], "category": record["category"],
        "source_object_count": 1, "crop_object_count": 1,
        "source_width": record["source_width"], "source_height": record["source_height"],
        "gt_class_id": record["material"], "gt_class_name": record["category"],
        "matched_iou": "1.00000000", "assignment": "positive_iou",
        "predicted_class_id": predicted_class, "predicted_class_name": CLASS_NAMES[predicted_class],
        "predicted_confidence": "0.72", "crop_bytes": crop.stat().st_size,
        "dent": "-1", "label": "-1", "foreign_material": "-1",
        "source_foreign_material": record["foreign_material"], "role": "train", "fold": "train",
        "annotation_authority": validator.OPERATIONAL_ANNOTATION_AUTHORITY,
    })
    for field in ("source_id", "source_sha256", "source_filepath", "origin", "captured_at", "object_group", "capture_session",
                  "teacher_output_sha256", "localizer_output_sha256", "auditor_sha256", "source_evidence_ref"):
        row[field] = record[field]
    for axis, value, bound in zip(("x1", "y1", "x2", "y2"), bbox, bounds):
        row["predicted_bbox_" + axis] = value
        row["gt_bbox_" + axis] = value
        row["crop_" + axis] = bound
    fixture.update(source=source, crop=crop, bundle=bundle, record=record, helpers=helpers,
                   proposal=Proposal(predicted_class, 0.72, bbox, CLASS_NAMES[predicted_class]))
    info = json.loads(fixture["info"].read_bytes())
    info["operational_source_evidence"] = {
        "bundle_dir": bundle.resolve().as_posix(),
        "receipt_sha256": _sha(bundle / adapter.FILES["receipt"]),
        "index_sha256": _sha(bundle / adapter.FILES["index"]),
        "marker_sha256": _sha(bundle / adapter.FILES["marker"]),
    }
    fixture["info"].write_text(json.dumps(info), encoding="utf-8")
    _write_row(fixture)
    return fixture


def _run(case, monkeypatch, *, mutate=None, custom=False, include_bundle=True):
    def predictions(records, **kwargs):
        if not custom:
            assert kwargs["conf"] == 0.10 and kwargs["nms_iou"] == 0.70
            assert kwargs["imgsz"] == 640
        for record in records:
            assert record.path.read_bytes() == case["source"].read_bytes()
            if not custom:
                annotation = validator._label_path(record.path).read_bytes()
                assert annotation == (case["bundle"] / case["record"]["source_evidence_ref"]).read_bytes()
                assert json.loads(annotation)["record"]["annotation_authority"] == validator.OPERATIONAL_ANNOTATION_AUTHORITY
                assert record.path != case["source"]
            if mutate:
                mutate(case, record)
            yield PredictedFrame(record, case["record"]["source_width"], case["record"]["source_height"], (case["proposal"],))
    monkeypatch.setattr(validator, "iter_yolo_predictions", predictions)
    return validator.validate_manifest(
        input_manifest=case["manifest"], dataset_info=case["info"], detector_model=case["detector"],
        inference_spec=case["helpers"].SPEC_PATH, output_manifest=case["output"], output_report=case["report"],
        operational_source_evidence_dir=case["bundle"] if include_bundle else None,
        prediction_provider=predictions if custom else None,
    )


def _not_published(case):
    assert not case["output"].exists()
    assert not case["report"].exists()


def test_real_adapter_evidence_labels_actual_crop_without_aihub_claim(case, monkeypatch):
    report = _run(case, monkeypatch)
    with case["output"].open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["material"] == str(case["record"]["material"])
    assert row["material"] != row["predicted_class_id"]
    assert row["ground_truth_authority"] == validator.OPERATIONAL_ANNOTATION_AUTHORITY
    assert row["source_annotation_sha256"] == row["auditor_sha256"]
    assert row["source_foreign_material"] == str(case["record"]["foreign_material"])
    assert row["dent"] == row["label"] == row["foreign_material"] == "-1"
    assert report["production_deployment_authorized"] is False
    assert report["blind_test_eligible"] is False
    assert report["bindings"]["dataset_info_sha256"] == _sha(case["info"])
    assert report["contract"]["proposal_provenance"]["confidence_abs_tolerance"] == 1e-6
    assert report["contract"]["proposal_provenance"]["bbox_abs_tolerance"] == 1e-4


def test_operational_custom_provider_remains_non_authoritative(case, monkeypatch):
    report = _run(case, monkeypatch, custom=True)
    assert report["ready_for_lineage_upgrade"] is False
    assert report["contract"]["proposal_provenance"]["provider_kind"] == "custom_non_authoritative"


def test_material_semantics_hold_rejects_before_replay(case, monkeypatch):
    marker = case["bundle"].parent / "material_semantics_hold.json"
    marker.mkdir()  # Any filesystem entry at the marker path is quarantine.
    def forbidden_replay(*args, **kwargs):
        pytest.fail("semantically quarantined input must not reach replay")
    monkeypatch.setattr(validator, "eager_initialize_cuda_context", forbidden_replay)
    with pytest.raises(ValueError, match="material semantics hold"):
        _run(case, monkeypatch)
    assert marker.is_dir()
    _not_published(case)


def test_material_semantics_hold_after_publication_removes_own_outputs(case, monkeypatch):
    marker = case["bundle"].parent / "material_semantics_hold.json"
    original = validator._publish_pair
    def publish_then_hold(*args):
        original(*args)
        marker.write_bytes(b"invalid-json-still-quarantined")
    monkeypatch.setattr(validator, "_publish_pair", publish_then_hold)
    with pytest.raises(ValueError, match="material semantics hold"):
        _run(case, monkeypatch)
    assert marker.read_bytes() == b"invalid-json-still-quarantined"
    _not_published(case)


@pytest.mark.parametrize("field,value", [
    ("annotation_authority", "aihub_annotation_geometry_development_only"),
    ("origin", "aihub_commercial"), ("role", "model_validation"), ("split", "validation"),
    ("fold", "model_validation"), ("source_sha256", "f" * 64),
    ("teacher_output_sha256", "f" * 64), ("localizer_output_sha256", "f" * 64),
    ("auditor_sha256", "f" * 64), ("source_evidence_ref", "source_evidence/other.json"),
    ("captured_at", "2026-07-31T00:00:00Z"), ("foreign_material", "1"),
    ("dent", "0"), ("label", "0"), ("source_foreign_material", "-1"),
    ("gt_bbox_x1", "0"), ("material", "9"),
])
def test_operational_row_cannot_change_authority_role_or_evidence(case, monkeypatch, field, value):
    case["row"][field] = value
    _write_row(case)
    with pytest.raises(ValueError):
        _run(case, monkeypatch)
    _not_published(case)


def test_verified_bundle_is_not_optional(case, monkeypatch):
    with pytest.raises(ValueError, match="evidence directory"):
        _run(case, monkeypatch, include_bundle=False)
    _not_published(case)


@pytest.mark.parametrize("field", ["receipt_sha256", "index_sha256", "marker_sha256", "bundle_dir"])
def test_generation_bundle_bindings_must_match(case, monkeypatch, field):
    info = json.loads(case["info"].read_bytes())
    info["operational_source_evidence"][field] = "f" * 64
    case["info"].write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="binding mismatch"):
        _run(case, monkeypatch)
    _not_published(case)


@pytest.mark.parametrize("target", ["receipt", "index", "marker", "source", "annotation", "snapshot_annotation", "info"])
def test_replay_tamper_never_publishes(case, monkeypatch, target):
    def mutate(case, record):
        if target in adapter.FILES:
            path = case["bundle"] / adapter.FILES[target]
        elif target == "annotation":
            path = case["bundle"] / case["record"]["source_evidence_ref"]
        elif target == "snapshot_annotation":
            path = validator._label_path(record.path)
        else:
            path = case[target]
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises((ValueError, RuntimeError)):
        _run(case, monkeypatch, mutate=mutate)
    _not_published(case)


def test_evidence_mutation_during_publication_rolls_back_both_outputs(case, monkeypatch):
    publish = validator._publish_pair
    def publish_then_mutate(*args):
        publish(*args)
        path = case["bundle"] / adapter.FILES["receipt"]
        path.write_bytes(path.read_bytes() + b"\n")
    monkeypatch.setattr(validator, "_publish_pair", publish_then_mutate)
    with pytest.raises(ValueError, match="operational evidence changed"):
        _run(case, monkeypatch)
    _not_published(case)


@pytest.mark.parametrize("when", ["before_replay", "during_replay"])
def test_generation_failure_marker_is_never_accepted(case, monkeypatch, when):
    marker = case["info"].parent / "failed.json"
    def fail_generation(*_):
        marker.write_text('{"generation_ready":false}\n', encoding="utf-8")
    if when == "before_replay":
        fail_generation()
    with pytest.raises(ValueError, match="failure marker"):
        _run(case, monkeypatch, mutate=fail_generation if when == "during_replay" else None)
    _not_published(case)


def test_empty_directory_mutation_during_publication_is_rejected(case, monkeypatch):
    publish = validator._publish_pair
    def publish_then_mutate(*args):
        publish(*args)
        (case["bundle"] / "unexpected-empty-directory").mkdir()
    monkeypatch.setattr(validator, "_publish_pair", publish_then_mutate)
    with pytest.raises(ValueError, match="operational evidence changed"):
        _run(case, monkeypatch)
    _not_published(case)


def test_operational_reader_runs_after_cuda_context_reservation(case, monkeypatch):
    events = []
    initialize = validator.eager_initialize_cuda_context
    read = adapter.validate_source_evidence_bundle
    def initialize_first(device):
        events.append("cuda_guard")
        return initialize(device)
    def read_after_guard(path):
        assert events and events[0] == "cuda_guard"
        events.append("adapter_reader")
        return read(path)
    monkeypatch.setattr(validator, "eager_initialize_cuda_context", initialize_first)
    monkeypatch.setattr(adapter, "validate_source_evidence_bundle", read_after_guard)
    _run(case, monkeypatch)
    assert events.count("adapter_reader") >= 3


def test_output_must_not_modify_immutable_evidence(case):
    info = json.loads(case["info"].read_bytes())
    before = {path: path.read_bytes() for path in case["bundle"].rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="inside operational evidence"):
        validator._load_operational_evidence(case["bundle"], info, case["bundle"] / "nested")
    assert before == {path: path.read_bytes() for path in case["bundle"].rglob("*") if path.is_file()}


@pytest.mark.parametrize("target", ["bundle", "quality", "teacher"])
def test_actual_report_output_cannot_poison_bound_evidence(case, monkeypatch, target):
    if target == "bundle":
        protected = case["bundle"]
    else:
        receipt = json.loads((case["bundle"] / adapter.FILES["receipt"]).read_bytes())
        prefix = "quality_" if target == "quality" else "teacher_output_"
        protected = Path(next(item["path"] for name, item in receipt["inputs"].items() if name.startswith(prefix))).parent
    before = {path.relative_to(protected).as_posix(): path.read_bytes()
              for path in protected.rglob("*") if path.is_file()}
    forbidden = protected / "new-validator-output"
    case["report"] = forbidden / "report.json"
    with pytest.raises(ValueError, match="inside operational evidence"):
        _run(case, monkeypatch)
    assert not forbidden.exists()
    assert not case["output"].exists()
    assert before == {path.relative_to(protected).as_posix(): path.read_bytes()
                      for path in protected.rglob("*") if path.is_file()}


def test_operational_source_reference_cannot_depend_on_working_directory(case, monkeypatch):
    monkeypatch.chdir(case["source"].parent)
    case["row"]["source_path_b64"] = base64.urlsafe_b64encode(os.fsencode(case["source"].name)).decode("ascii")
    _write_row(case)
    with pytest.raises(ValueError, match="must be absolute"):
        _run(case, monkeypatch)
    _not_published(case)
