"""Operational connection tests; injected predictions do not claim real YOLO replay."""
import csv
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import build_operational_source_evidence as adapter
from scripts import prepare_proposal_verifier_dataset as prep


@pytest.fixture
def mixed(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "_operational_teacher_fixture", Path(__file__).with_name("test_build_operational_teacher_manifest.py")
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    args, _, _ = helper._objective_quality_assembly_fixture(tmp_path / "input", no_quality_exclusions=True)
    helper.assemble_operational_quality_exclusions(**args)
    bundle = tmp_path / "source-evidence"
    adapter.build_source_evidence(
        **{name: args[name] for name in adapter.INPUT_NAMES},
        teacher_output_dir=args["teacher_output_dir"], image_root=args["image_root"],
        quality_assembly_receipt=args["output_dir"] / helper.ASSEMBLY_FILES["receipt"],
        output_dir=bundle,
    )
    base = tmp_path / "base"
    for index, split in enumerate(("train", "val")):
        images, labels = base / "images" / split, base / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        assert cv2.imwrite(str(images / "base.png"), np.full((100, 120, 3), 20 + 40 * index, dtype=np.uint8))
        (labels / "base.txt").write_text("2 0.5 0.5 0.5 0.4\n", encoding="utf-8")
    data = tmp_path / "data.yaml"
    data.write_text(
        "path: .\ntrain: images/train\nval: images/val\n"
        "names: [can, pet, paper, plastic, styrofoam, vinyl, glass, battery, fluorescent]\n",
        encoding="utf-8",
    )
    return dict(
        model_path=tmp_path / "not-loaded.pt", data_path=data, dataset_dir=base,
        output_dir=tmp_path / "output", device="cpu", batch=1, imgsz=640, conf=0.1,
        nms_iou=0.7, positive_iou=0.5, negative_iou=0.1, crop_size=320, padding=0.08,
        max_per_class=100, val_max_per_class=100, max_background=100, val_max_background=100,
        seed=3, min_free_gb=0, max_output_gb=0, jpeg_quality=92,
        proposal_selection="runtime-top1", background_policy="strict-zero-intersection",
        operational_source_evidence_dir=bundle, aihub_origin="test_aihub_annotation",
    )


def _predictions(sources):
    for source in sources:
        image = prep._read_image(source.path)
        height, width = image.shape[:2]
        bbox = source.ground_truth.xyxy(width, height)
        # Deliberately different bbox and wrong detector class. Targets must
        # use the reference, crops must use the actual prediction provider.
        bbox = (bbox[0] + 1, bbox[1], bbox[2], bbox[3])
        yield prep.PredictedFrame(source, width, height, (prep.Proposal(8, 0.93, bbox, "fluorescent"),))


def test_real_adapter_to_predicted_crop_preserves_provenance_and_masks_states(mixed):
    summary = prep.build_proposal_verifier_dataset(**mixed, prediction_provider=_predictions)
    with (mixed["output_dir"] / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    operational = [row for row in rows if row["annotation_authority"] == adapter.SOURCE_ROLE]
    assert len(rows) == 5 and len(operational) == 3
    assert summary["operational_sources"] == 3
    assert set(summary["operational_source_evidence"]) == {"bundle_dir", "receipt_sha256", "index_sha256", "marker_sha256"}
    for row in rows:
        assert Path(row["filepath"]).is_absolute()
        assert hashlib.sha256(Path(row["filepath"]).read_bytes()).hexdigest() == row["image_sha256"]
        assert row["role"] == row["fold"]
        assert row["predicted_class_id"] == "8" and row["material"] != "8"
        assert float(row["predicted_bbox_x1"]) == float(row["gt_bbox_x1"]) + 1
        assert row["dent"] == row["label"] == row["foreign_material"] == "-1"
        assert cv2.imread(row["filepath"]).shape == (320, 320, 3)
        image = prep._read_image(Path(row["source_filepath"]))
        height, width = image.shape[:2]
        bbox = tuple(float(row[f"predicted_bbox_{axis}"]) for axis in ("x1", "y1", "x2", "y2"))
        left, top, right, bottom = prep._crop_bounds(bbox, width, height, mixed["padding"])
        expected = prep.letterbox(image[top:bottom, left:right], mixed["crop_size"])
        ok, encoded = cv2.imencode(".jpg", expected, [cv2.IMWRITE_JPEG_QUALITY, mixed["jpeg_quality"]])
        assert ok and Path(row["filepath"]).read_bytes() == encoded.tobytes()
    for row in operational:
        assert row["role"] == "train" and row["split"] == "training"
        assert row["source_foreign_material"] in {"0", "1"}
        assert row["captured_at"] and row["origin"] != mixed["aihub_origin"]
        evidence = mixed["operational_source_evidence_dir"] / row["source_evidence_ref"]
        assert hashlib.sha256(evidence.read_bytes()).hexdigest() == row["auditor_sha256"]


def test_operational_background_proposals_are_not_pseudo_negatives(mixed):
    rows = adapter.validate_source_evidence_bundle(mixed["operational_source_evidence_dir"])
    sources = prep.append_operational_sources([], rows, all_split_images={}, dataset_dir=mixed["dataset_dir"])
    source = sources[0]
    # No overlap with a reference object. Even if strict geometry could pass,
    # the VLM reference is not exhaustive enough to authorize a negative.
    frame = prep.PredictedFrame(source, 1000, 1000, (prep.Proposal(2, 0.9, (990, 990, 999, 999)),))
    from collections import Counter
    stats = Counter()
    result = list(prep.candidates_from_frames([frame], positive_iou=0.5, negative_iou=0.1, policy_stats=stats))
    assert result == []
    assert stats["operational_background_not_authorized"] == 1


def test_validation_duplicate_rejected_even_when_missing_sidecar(mixed):
    records = adapter.validate_source_evidence_bundle(mixed["operational_source_evidence_dir"])
    destination = mixed["dataset_dir"] / "images" / "val" / "duplicate.png"
    destination.write_bytes(Path(records[0]["source_filepath"]).read_bytes())
    with pytest.raises(ValueError, match="duplicates a base"):
        prep.build_proposal_verifier_dataset(**mixed, prediction_provider=_predictions)
    assert not (mixed["output_dir"] / "manifest.csv").exists()


def test_changed_adapter_after_inference_cannot_publish_manifest(mixed):
    def mutate(sources):
        yield from _predictions(sources)
        marker = mixed["operational_source_evidence_dir"] / "source_evidence.sha256"
        marker.write_bytes(marker.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="source evidence changed"):
        prep.build_proposal_verifier_dataset(**mixed, prediction_provider=mutate)
    assert not (mixed["output_dir"] / "manifest.csv").exists()
    assert (mixed["output_dir"] / "failed.json").is_file()


def test_explicit_base_origin_required_before_loading(mixed):
    mixed["aihub_origin"] = None
    with pytest.raises(ValueError, match="explicit aihub_origin"):
        prep.build_proposal_verifier_dataset(**mixed, prediction_provider=_predictions)


@pytest.mark.parametrize("field,value", [
    ("annotation_authority", "aihub_annotation_geometry_development_only"),
    ("role", "model_validation"), ("source_object_count", True),
    ("material", True), ("source_width", True), ("source_height", 0),
    ("source_bbox_xyxy", [0, 0, float("nan"), 100]),
    ("source_bbox_xyxy", [-1, 0, 20, 100]),
    ("source_bbox_xyxy", [False, 0, 20, 100]),
])
def test_invalid_operational_authority_or_geometry_is_not_promoted(mixed, field, value):
    rows = adapter.validate_source_evidence_bundle(mixed["operational_source_evidence_dir"])
    rows[0][field] = value
    with pytest.raises(ValueError):
        prep.append_operational_sources([], rows, all_split_images={}, dataset_dir=mixed["dataset_dir"])


@pytest.mark.parametrize("target", ["source", "sidecar", "yaml", "missing_sidecar"])
def test_changed_base_inputs_during_inference_do_not_publish(mixed, target):
    base = mixed["dataset_dir"]
    missing = base / "images" / "train" / "quarantined.png"
    assert cv2.imwrite(str(missing), np.full((100, 120, 3), 120, dtype=np.uint8))
    targets = {
        "source": base / "images" / "train" / "base.png",
        "sidecar": base / "labels" / "train" / "base.txt",
        "yaml": mixed["data_path"],
        "missing_sidecar": base / "labels" / "train" / "quarantined.txt",
    }
    def mutate(sources):
        yield from _predictions(sources)
        path = targets[target]
        path.write_bytes((path.read_bytes() if path.exists() else b"") + b"changed")
    with pytest.raises(RuntimeError, match="generation input changed"):
        prep.build_proposal_verifier_dataset(**mixed, prediction_provider=mutate)
    assert not (mixed["output_dir"] / "manifest.csv").exists()


@pytest.mark.parametrize("directory_model", [False, True])
def test_real_provider_model_artifact_change_is_rejected(mixed, monkeypatch, directory_model):
    model = mixed["model_path"]
    if directory_model:
        model.mkdir()
        model = model / "model.bin"
    model.write_bytes(b"test detector bytes, not an executed model")
    monkeypatch.setattr(prep, "eager_initialize_cuda_context", lambda _: None)
    def mutate(sources, **kwargs):
        yield from _predictions(sources)
        model.write_bytes(b"changed detector")
    monkeypatch.setattr(prep, "iter_yolo_predictions", mutate)
    with pytest.raises(RuntimeError, match="generation input changed"):
        prep.build_proposal_verifier_dataset(**mixed)
    assert not (mixed["output_dir"] / "manifest.csv").exists()


def _at_info_publication(monkeypatch, action):
    original = Path.open
    @contextmanager
    def intercepted(path, *args, **kwargs):
        with original(path, *args, **kwargs) as stream:
            yield stream
        if path.name == "dataset_info.json" and args and args[0] == "x":
            action()
    monkeypatch.setattr(Path, "open", intercepted)


@pytest.mark.parametrize("target", ["sidecar", "source", "yaml", "evidence", "new_base_source", "crop", "manifest", "info"])
def test_mutation_at_metadata_publication_is_failed_not_success(mixed, monkeypatch, target):
    def mutate():
        if target == "evidence":
            path = next((mixed["operational_source_evidence_dir"] / "source_evidence").glob("*.json"))
        elif target == "crop":
            path = next(mixed["output_dir"].rglob("*.jpg"))
        elif target in ("manifest", "info"):
            path = mixed["output_dir"] / ("manifest.csv" if target == "manifest" else "dataset_info.json")
        elif target == "new_base_source":
            path = mixed["dataset_dir"] / "images" / "train" / "new.png"
            path.write_bytes((path.parent / "base.png").read_bytes())
            return
        else:
            path = {"sidecar": mixed["dataset_dir"] / "labels" / "train" / "base.txt",
                    "source": mixed["dataset_dir"] / "images" / "train" / "base.png",
                    "yaml": mixed["data_path"]}[target]
        path.write_bytes(path.read_bytes() + b"changed")
    _at_info_publication(monkeypatch, mutate)
    with pytest.raises((RuntimeError, ValueError)):
        prep.build_proposal_verifier_dataset(**mixed, prediction_provider=_predictions)
    assert (mixed["output_dir"] / "failed.json").is_file()


def test_metadata_write_failure_leaves_explicit_failed_marker(mixed, monkeypatch):
    original = Path.open
    def fail_info(path, *args, **kwargs):
        if path.name == "dataset_info.json" and args and args[0] == "x":
            raise OSError("injected full filesystem")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", fail_info)
    with pytest.raises(OSError, match="injected"):
        prep.build_proposal_verifier_dataset(**mixed, prediction_provider=_predictions)
    assert (mixed["output_dir"] / "failed.json").is_file()


def test_forged_manifest_is_not_overwritten(mixed, monkeypatch):
    original = prep.write_selected_crops
    def forged(*args, **kwargs):
        result = original(*args, **kwargs)
        (mixed["output_dir"] / "manifest.csv").write_bytes(b"foreign metadata\n")
        return result
    monkeypatch.setattr(prep, "write_selected_crops", forged)
    with pytest.raises(FileExistsError):
        prep.build_proposal_verifier_dataset(**mixed, prediction_provider=_predictions)
    assert (mixed["output_dir"] / "manifest.csv").read_bytes() == b"foreign metadata\n"
    assert (mixed["output_dir"] / "failed.json").is_file()


def test_output_cannot_be_nested_in_source_bundle(mixed):
    bundle = mixed["operational_source_evidence_dir"]
    before = {path.relative_to(bundle): path.read_bytes() for path in bundle.rglob("*") if path.is_file()}
    mixed["output_dir"] = bundle / "nested-output"
    with pytest.raises(ValueError, match="nested"):
        prep.build_proposal_verifier_dataset(**mixed, prediction_provider=_predictions)
    assert before == {path.relative_to(bundle): path.read_bytes() for path in bundle.rglob("*") if path.is_file()}


@pytest.mark.parametrize("target", ["image_root", "teacher", "quality"])
def test_output_cannot_be_nested_in_bound_operational_inputs(mixed, target):
    bundle = mixed["operational_source_evidence_dir"]
    receipt = json.loads((bundle / adapter.FILES["receipt"]).read_bytes())
    if target == "image_root":
        protected = Path(receipt["image_root"])
    else:
        prefix = "teacher_output_" if target == "teacher" else "quality_"
        protected = Path(next(item["path"] for name, item in receipt["inputs"].items()
                              if name.startswith(prefix))).parent
    def snapshot():
        return {path.relative_to(protected).as_posix(): path.read_bytes() if path.is_file() else None
                for path in protected.rglob("*")}
    before = snapshot()
    mixed["output_dir"] = protected / "new-proposal-output"
    with pytest.raises(ValueError, match="nested in generation inputs"):
        prep.build_proposal_verifier_dataset(**mixed, prediction_provider=_predictions)
    assert not mixed["output_dir"].exists()
    assert snapshot() == before


def test_changed_bundle_between_reader_and_binding_never_claims_output(mixed, monkeypatch):
    original = prep._operational_bundle_reader
    def read_then_mutate(path):
        records = original(path)
        marker = path / adapter.FILES["marker"]
        marker.write_bytes(marker.read_bytes() + b"changed\n")
        return records
    monkeypatch.setattr(prep, "_operational_bundle_reader", read_then_mutate)
    with pytest.raises(RuntimeError, match="changed during initial validation"):
        prep.build_proposal_verifier_dataset(**mixed, prediction_provider=_predictions)
    assert not mixed["output_dir"].exists()


def test_material_semantics_hold_rejects_before_prediction(mixed):
    bundle = mixed["operational_source_evidence_dir"]
    before = {path.relative_to(bundle): path.read_bytes() for path in bundle.rglob("*") if path.is_file()}
    marker = bundle.parent / "material_semantics_hold.json"
    marker.write_bytes(b"")  # Presence alone is a hold, not a JSON clearance.
    def forbidden_prediction(_):
        pytest.fail("semantically quarantined input must not reach inference")
    with pytest.raises(ValueError, match="material semantics hold"):
        prep.build_proposal_verifier_dataset(**mixed, prediction_provider=forbidden_prediction)
    assert not mixed["output_dir"].exists()
    assert marker.read_bytes() == b""
    assert before == {path.relative_to(bundle): path.read_bytes() for path in bundle.rglob("*") if path.is_file()}


def test_material_semantics_hold_at_publication_marks_preparation_failed(mixed, monkeypatch):
    marker = mixed["operational_source_evidence_dir"].parent / "material_semantics_hold.json"
    original = prep._publish_mixed_metadata
    def publish_with_hold(output_dir, rows, summary, *, validate, identity):
        def validate_with_hold(full_rehash):
            if full_rehash:
                marker.write_bytes(b"invalid-json-still-quarantined")
            validate(full_rehash)
        return original(output_dir, rows, summary, validate=validate_with_hold, identity=identity)
    monkeypatch.setattr(prep, "_publish_mixed_metadata", publish_with_hold)
    with pytest.raises(ValueError, match="material semantics hold"):
        prep.build_proposal_verifier_dataset(**mixed, prediction_provider=_predictions)
    assert (mixed["output_dir"] / "manifest.csv").is_file()
    assert (mixed["output_dir"] / "failed.json").is_file()
    assert marker.read_bytes() == b"invalid-json-still-quarantined"


def test_mixed_output_must_be_fresh_even_when_existing_directory_is_empty(mixed):
    mixed["output_dir"].mkdir()
    with pytest.raises(FileExistsError):
        prep.build_proposal_verifier_dataset(**mixed, prediction_provider=_predictions)
    assert list(mixed["output_dir"].iterdir()) == []


def test_symlink_source_is_rejected_before_prediction(mixed):
    path = mixed["dataset_dir"] / "images" / "train" / "base.png"
    target = path.with_name("external.bin")
    path.rename(target)
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="symlink"):
        prep.build_proposal_verifier_dataset(**mixed, prediction_provider=_predictions)
    assert not mixed["output_dir"].exists()


def test_output_symlink_ancestor_is_not_followed(mixed):
    actual = mixed["output_dir"].with_name("foreign-output")
    actual.mkdir()
    link = mixed["output_dir"].with_name("linked-output")
    try:
        link.symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    mixed["output_dir"] = link / "run"
    with pytest.raises(ValueError, match="symlink"):
        prep.build_proposal_verifier_dataset(**mixed, prediction_provider=_predictions)
    assert list(actual.iterdir()) == []
