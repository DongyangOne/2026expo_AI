"""Original JSON -> JPEG -> actual crop geometry; detectors are CPU test doubles."""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import materialize_audited_aihub_sources as materializer
from scripts import prepare_proposal_verifier_dataset as prepare
from scripts import validate_v4_background_candidates as validate
from test_materialize_audited_aihub_sources import fixture as materializer_inputs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predictions(records):
    for source in records:
        image = cv2.imdecode(np.frombuffer(source.path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        yield prepare.PredictedFrame(source, width, height, (
            prepare.Proposal(0, .9, source.ground_truth.xyxy(width, height), "can"),
        ))


@pytest.fixture
def inputs(materializer_inputs, tmp_path):
    data, save, root, originals, annotations, _ = materializer_inputs

    def create(*, full=True):
        data["full_cohort"] = full
        values = save()
        materializer.materialize_sources(**values)
        model = tmp_path / "best.pt"
        model.write_bytes(b"CPU test double only")
        return dict(model_path=model, data_path=root / "dataset.yaml", dataset_dir=root,
                    output_dir=tmp_path / "proposals", device="cpu", batch=1, imgsz=640,
                    conf=.1, nms_iou=.7, positive_iou=.5, negative_iou=.1,
                    crop_size=320, padding=.08, max_per_class=10, val_max_per_class=10,
                    max_background=10, val_max_background=10, seed=42, min_free_gb=0,
                    max_output_gb=1, jpeg_quality=92, proposal_selection="runtime-top1",
                    background_policy="strict-zero-intersection", background_gt_margin=.1,
                    aihub_origin="aihub_original_direct_capture_development",
                    audited_aihub_report=root / "report.json",
                    audited_aihub_report_sha256=sha(root / "report.json"),
                    audited_aihub_cohort=values["cohort"])

    return create, originals, annotations


def run_validate(args, **kwargs):
    root = args["output_dir"]
    return validate.validate_manifest(
        input_manifest=root / "manifest.csv", dataset_info=root / "dataset_info.json",
        detector_model=args["model_path"],
        inference_spec=Path(__file__).resolve().parents[1] / "configs/detector_inference_v3.json",
        output_manifest=root / "validated.csv", output_report=root / "validation.json",
        prediction_provider=kwargs.pop("prediction_provider", predictions), **kwargs,
    )


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_real_materializer_to_canonical_crop_and_replay_preserves_original_identity(inputs):
    create, originals, annotations = inputs
    args = create()
    summary = prepare.build_proposal_verifier_dataset(**args, prediction_provider=predictions)
    raw = rows(args["output_dir"] / "manifest.csv")
    assert len(raw) == 2
    assert summary["audited_aihub_snapshot"]["require_full_cohort"] is True
    assert "do not prove physical" in summary["original_identity_semantics"]
    for row in raw:
        index = 0 if row["split"] == "training" else 1
        assert row["source_id"] == row["source_sha256"] == sha(Path(row["source_filepath"]))
        assert row["original_source_sha256"] == sha(originals[index]) != row["source_sha256"]
        assert row["original_annotation_sha256"] == sha(annotations[index])
        assert row["object_group"] == row["capture_session"] == "source_sha256:" + row["source_sha256"]
        assert all(row[key] == "-1" for key in ("dent", "label", "foreign_material"))
    initial = (args["output_dir"] / "manifest.csv").read_bytes()
    report = run_validate(args)
    assert (args["output_dir"] / "manifest.csv").read_bytes() == initial
    checked = rows(args["output_dir"] / "validated.csv")
    assert all({key: row[key] for key in prepare.AUDITED_AIHUB_FIELDS} ==
               {key: original[key] for key in prepare.AUDITED_AIHUB_FIELDS}
               for row, original in zip(checked, raw))
    assert report["ready_for_lineage_upgrade"] is False
    assert report["production_deployment_authorized"] is False
    assert report["bindings"]["dataset_info_sha256"] == sha(args["output_dir"] / "dataset_info.json")


@pytest.mark.parametrize("missing", ["audited_aihub_report", "audited_aihub_report_sha256", "audited_aihub_cohort"])
def test_partial_arguments_reject_before_prediction_or_output(inputs, missing):
    args = inputs[0]()
    args.pop(missing)
    with pytest.raises(ValueError, match="supplied together"):
        prepare.build_proposal_verifier_dataset(**args, prediction_provider=lambda _: pytest.fail("prediction"))
    assert not args["output_dir"].exists()


def test_partial_snapshot_requires_both_generation_opt_in_and_diagnostic_replay(inputs):
    args = inputs[0](full=False)
    with pytest.raises(ValueError, match="diagnostic"):
        prepare.build_proposal_verifier_dataset(**args, prediction_provider=predictions)
    assert not args["output_dir"].exists()
    prepare.build_proposal_verifier_dataset(**args, audited_aihub_diagnostic=True, prediction_provider=predictions)
    with pytest.raises(ValueError, match="diagnostic_only"):
        run_validate(args)
    report = run_validate(args, diagnostic_only=True)
    assert not report["ready_for_lineage_upgrade"]
    assert not report["lineage_execution_authorized"]


@pytest.mark.parametrize("fault", ["missing_binding", "wrong_original_sha", "wrong_annotation_sha", "false_full_claim"])
def test_unverified_original_provenance_does_not_pass(inputs, fault):
    args = inputs[0](full=fault != "false_full_claim")
    prepare.build_proposal_verifier_dataset(**args, audited_aihub_diagnostic=fault == "false_full_claim", prediction_provider=predictions)
    root = args["output_dir"]
    if fault in {"missing_binding", "false_full_claim"}:
        path = root / "dataset_info.json"
        info = json.loads(path.read_text())
        if fault == "missing_binding":
            del info["audited_aihub_snapshot"]
        else:
            info["audited_aihub_snapshot"]["require_full_cohort"] = True
        path.write_text(json.dumps(info), encoding="utf-8")
    else:
        path = root / "manifest.csv"
        data = rows(path)
        data[0]["original_source_sha256" if fault == "wrong_original_sha" else "original_annotation_sha256"] = "0" * 64
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    with pytest.raises(ValueError):
        run_validate(args)
    assert not (root / "validated.csv").exists()


@pytest.mark.parametrize("kind", ["original", "annotation", "report"])
def test_upstream_change_during_generation_is_rejected(inputs, kind):
    create, originals, annotations = inputs
    args = create()
    target = {"original": originals[0], "annotation": annotations[0], "report": args["audited_aihub_report"]}[kind]

    def changing(records):
        yield from predictions(records)
        target.write_bytes(target.read_bytes() + b"changed")

    with pytest.raises(ValueError):
        prepare.build_proposal_verifier_dataset(**args, prediction_provider=changing)
    assert not args["output_dir"].exists()


def test_publication_boundary_source_change_removes_only_new_validator_outputs(inputs, monkeypatch):
    create, originals, _ = inputs
    args = create()
    prepare.build_proposal_verifier_dataset(**args, prediction_provider=predictions)
    original = validate._publish_pair

    def changing(*values):
        original(*values)
        originals[0].write_bytes(originals[0].read_bytes() + b"changed")

    monkeypatch.setattr(validate, "_publish_pair", changing)
    with pytest.raises(ValueError):
        run_validate(args)
    assert not (args["output_dir"] / "validated.csv").exists()
    assert not (args["output_dir"] / "validation.json").exists()
    assert (args["output_dir"] / "manifest.csv").exists()


def test_generation_publication_boundary_preserves_failure_marker(inputs, monkeypatch):
    create, originals, _ = inputs
    args = create()
    original = prepare._publish_mixed_metadata

    def changing(*values, **kwargs):
        check = kwargs["validate"]

        def injected(full):
            if full:
                originals[0].write_bytes(originals[0].read_bytes() + b"changed")
            check(full)

        kwargs["validate"] = injected
        return original(*values, **kwargs)

    monkeypatch.setattr(prepare, "_publish_mixed_metadata", changing)
    with pytest.raises(ValueError):
        prepare.build_proposal_verifier_dataset(**args, prediction_provider=predictions)
    assert (args["output_dir"] / "failed.json").exists()
    with pytest.raises(ValueError, match="failure marker"):
        run_validate(args)


def test_exact_snapshot_split_membership_checked_before_prediction(inputs):
    args = inputs[0]()
    wrong_data = args["dataset_dir"].parent / "swapped.yaml"
    wrong_data.write_text(args["data_path"].read_text(encoding="utf-8")
                          .replace("train: images/train", "train: images/val")
                          .replace("val: images/val", "val: images/train"), encoding="utf-8")
    args["data_path"] = wrong_data
    with pytest.raises(ValueError, match="official split"):
        prepare.build_proposal_verifier_dataset(**args, prediction_provider=lambda _: pytest.fail("prediction"))
    assert not args["output_dir"].exists()


@pytest.mark.parametrize("root_name", ["materialized", "cohort", "original"])
def test_generation_output_cannot_modify_audited_input_trees(inputs, root_name):
    create, originals, _ = inputs
    args = create()
    root = {"materialized": args["dataset_dir"], "cohort": args["audited_aihub_cohort"].parent,
            "original": originals[0].parent}[root_name]
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    args["output_dir"] = root / "new-proposals"
    with pytest.raises(ValueError, match="nested"):
        prepare.build_proposal_verifier_dataset(**args, prediction_provider=predictions)
    assert not args["output_dir"].exists()
    assert before == {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_real_provider_model_change_is_not_published(inputs, monkeypatch):
    args = inputs[0]()

    def changed(records, **kwargs):
        yield from predictions(records)
        args["model_path"].write_bytes(b"changed model")

    monkeypatch.setattr(prepare, "iter_yolo_predictions", changed)
    with pytest.raises(RuntimeError, match="input changed"):
        prepare.build_proposal_verifier_dataset(**args)
    assert not args["output_dir"].exists()


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_non_boolean_full_requirement_is_rejected(inputs, value):
    args = inputs[0]()
    prepare.build_proposal_verifier_dataset(**args, prediction_provider=predictions)
    path = args["output_dir"] / "dataset_info.json"
    info = json.loads(path.read_text())
    info["audited_aihub_snapshot"]["require_full_cohort"] = value
    path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid audited"):
        run_validate(args)


@pytest.mark.parametrize("field,value", [
    ("split", "validation"), ("role", "model_validation"),
    ("fold", "model_validation"), ("dent", "1"), ("label", "0"),
    ("foreign_material", "1"), ("annotation_authority", ""),
    ("source_filepath", "/unverified/source.jpg"),
])
def test_audited_source_cannot_change_official_split_or_invent_condition_labels(inputs, field, value):
    args = inputs[0]()
    prepare.build_proposal_verifier_dataset(**args, prediction_provider=predictions)
    path = args["output_dir"] / "manifest.csv"
    data = rows(path)
    target = next(row for row in data if row["split"] == "training")
    target[field] = value
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    with pytest.raises(ValueError):
        run_validate(args)
    assert not (args["output_dir"] / "validated.csv").exists()


def test_runtime_path_detector_change_at_publication_does_not_leave_ready_outputs(inputs, monkeypatch):
    args = inputs[0]()
    prepare.build_proposal_verifier_dataset(**args, prediction_provider=predictions)
    original = validate._publish_pair

    def changed(*values):
        original(*values)
        args["model_path"].write_bytes(b"changed model after replay")

    # Exercise the runtime snapshot branch without claiming a real model test.
    monkeypatch.setattr(validate, "iter_yolo_predictions", lambda records, **kwargs: predictions(records))
    monkeypatch.setattr(validate, "_publish_pair", changed)
    with pytest.raises(ValueError, match="model changed"):
        run_validate(args, prediction_provider=None)
    assert not (args["output_dir"] / "validated.csv").exists()
    assert not (args["output_dir"] / "validation.json").exists()


def test_background_crop_keeps_original_positive_annotation_without_becoming_source_gt(inputs):
    args = inputs[0]()

    def outside(records):
        for frame in predictions(records):
            yield prepare.PredictedFrame(frame.source, frame.width, frame.height, (
                prepare.Proposal(0, .9, (0., 0., 50., 50.), "can"),
            ))

    prepare.build_proposal_verifier_dataset(**args, prediction_provider=outside)
    report = run_validate(args, prediction_provider=outside)
    data = rows(args["output_dir"] / "validated.csv")
    assert len(data) == 2 and not report["ready_for_lineage_upgrade"]
    assert all(row["material"] == "9" and row["gt_class_id"] == "0" for row in data)
    assert all(row["source_object_count"] == "1" and row["crop_object_count"] == "0" for row in data)
    assert all(row["original_annotation_sha256"] and row["source_annotation_sha256"] for row in data)
