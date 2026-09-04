"""Real small JPEG fixtures + fake predictions; not GPU or semantic evidence."""
import base64
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts import observe_protected_proposals as observe


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def example(tmp_path):
    metadata, sources, weights = (tmp_path / name for name in ("metadata", "sources", "weights"))
    for directory in (metadata, sources, weights): directory.mkdir()
    inventory, fingerprint, model, spec = metadata / "inventory.json", metadata / "fingerprints.json", weights / "model.pt", metadata / "spec.json"
    model.write_bytes(b"test-weight-snapshot-not-real-YOLO")
    spec.write_bytes((Path(__file__).parents[1] / "configs/detector_inference_v3.json").read_bytes())
    rows, refs = [], []
    for index in range(3):
        image = np.zeros((80, 120, 3), np.uint8)
        image[:, :50] = (21 + index, 66, 99)
        image[10:50, 20:100] = (111, 33 + index, 70)
        path = sources / f"{index}.jpg"
        assert cv2.imwrite(str(path), image)
        digest = sha(path)
        source_roles = ["capture", "known_audit"] if index == 0 else ["qx3"]
        rows.append({"sha256": digest, "path": str(path), "roles": source_roles})
        refs.append({"source_sha256": digest, "source_path_b64": base64.urlsafe_b64encode(os.fsencode(path)).decode(),
                     "roles": source_roles, "source_bytes": path.stat().st_size, "image_width": 120,
                     "image_height": 80, "source_phash64": "0" * 16})
    inv = {"records": rows, "metadata_bindings": []}
    fp = {"schema": "protected_image_fingerprint_snapshot.v1", "status": "snapshot_complete",
          "snapshot_only": True, "consumer_must_rehash_sources": True, "records": refs,
          "expected_sources": 3, "verified_sources": 3, "missing_sources": 0,
          "training_authorized": False, "deployment_authorized": False,
          "blind_test_authorized": False, "selection_authorized": False}
    def save():
        inventory.write_text(json.dumps(inv), encoding="utf-8")
        fingerprint.write_text(json.dumps(fp), encoding="utf-8")
        return dict(inventory=inventory, inventory_sha256=sha(inventory), protected_report=fingerprint,
                    protected_report_sha256=sha(fingerprint), model=model, model_sha256=sha(model),
                    inference_spec=spec, inference_spec_sha256=sha(spec),
                    code_pins={name: sha(path) for name, path in observe.code_paths().items()}, output=tmp_path / "new")
    return save, inv, fp


def proposal(confidence=0.8, bbox=(10.0, 20.0, 100.0, 60.0), class_id=0, class_name="can"):
    return observe.prepare.Proposal(class_id, confidence, bbox, class_name)


def provider(records):
    for index, record in enumerate(records):
        candidates = (proposal(), proposal(bbox=(20., 25., 40., 45.))) if index == 0 else ((proposal(.09),) if index == 1 else ())
        yield observe.prepare.PredictedFrame(record, 120, 80, candidates)


def test_actual_canonical_crop_bytes_and_distinct_absence_counts(example):
    save, inv, _ = example
    args = save()
    report = observe.observe(**args, prediction_provider=provider)
    assert report == json.loads((args["output"] / "report.json").read_bytes())
    assert report["requested_sources"] == report["observed_sources"] == 3
    assert report["crop_generated"] == 1 and report["no_eligible_proposal"] == 2
    assert report["runtime"]["runtime_detector_executed"] is False
    assert all(report[key] is False for key in observe.AUTHORITY)
    assert "after model confidence filtering" in report["runtime"]["returned_count_stage"]
    first, below, none = report["records"]
    assert first["roles"] == inv["records"][0]["roles"]
    assert first["selected_proposal"]["index"] == 0  # Equal confidence keeps original order.
    assert below["returned_proposals_after_model_confidence_nms"] == below["below_confidence_floor"] == 1
    assert none["returned_proposals_after_model_confidence_nms"] == 0
    for row in (below, none):
        assert row["crop"] is row["selected_proposal"] is None
        assert row["object_absence_established"] is False
        assert not any(key in row for key in ("objectness", "label", "dent", "foreign_material", "ground_truth"))
    original = cv2.imread(inv["records"][0]["path"])
    pixels, bounds = observe.preprocessing.crop_and_letterbox_bgr(original, proposal().bbox, padding=.08, size=320, fill=114)
    ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    crop = args["output"] / first["crop"]["path"]
    assert crop.read_bytes() == encoded.tobytes()
    assert first["crop"]["sha256"] == sha(crop)
    assert first["crop"]["bounds_xyxy"] == list(bounds)
    assert not list(args["output"].glob(".frozen-observation-*"))
    assert not list(args["output"].rglob("*.csv"))


@pytest.mark.parametrize("bad", [proposal(float("nan")), proposal(float("inf")), proposal(-.1), proposal(1.1), proposal(True),
    proposal(bbox=(1., 1., float("nan"), 5.)), proposal(bbox=(10., 20., 10., 30.)),
    proposal(bbox=(200., 200., 250., 250.)), proposal(.09, bbox=(0., 0., float("inf"), 8.)),
    proposal(class_id=9, class_name="background"), proposal(class_name="plastic")])
def test_invalid_returned_proposal_is_error_never_absent(example, bad):
    save, _, _ = example
    args = save()
    def invalid(records):
        yield observe.prepare.PredictedFrame(records[0], 120, 80, (bad,))
    with pytest.raises((ValueError, TypeError)):
        observe.observe(**args, prediction_provider=invalid)
    assert not (args["output"] / "report.json").exists()
    assert json.loads((args["output"] / "failed.json").read_bytes())["partial_outputs_not_authoritative"] is True


@pytest.mark.parametrize("fault", ["error", "omitted", "duplicate", "order", "shape"])
def test_failed_or_incomplete_detector_run_cannot_publish_absence(example, fault):
    save, _, _ = example
    args = save()
    def broken(records):
        if fault == "error": raise RuntimeError("synthetic inference failure")
        if fault == "omitted": return
        first = next(provider(records))
        if fault == "order": first = replace(first, source=records[1])
        if fault == "shape": first = replace(first, width=119)
        yield first
        if fault == "duplicate": yield first
    with pytest.raises((ValueError, RuntimeError)):
        observe.observe(**args, prediction_provider=broken)
    assert not (args["output"] / "report.json").exists()
    assert (args["output"] / "failed.json").exists()


@pytest.mark.parametrize("fault", ["membership", "roles", "path", "duplicate", "too_many", "incomplete"])
def test_strict_fingerprint_membership_and_source_bound(example, fault):
    save, inv, fp = example
    if fault == "membership": inv["records"][0]["sha256"] = "f" * 64
    elif fault == "roles": inv["records"][0]["roles"] = ["qx3"]
    elif fault == "path": inv["records"][0]["path"] = inv["records"][1]["path"]
    elif fault == "duplicate": inv["records"].append(inv["records"][0].copy())
    elif fault == "too_many": inv["records"] *= 11
    else: fp["missing_sources"] = 1
    args = save()
    with pytest.raises(ValueError): observe.observe(**args, prediction_provider=provider)
    assert not args["output"].exists()


@pytest.mark.parametrize("target", ["inventory", "protected_report", "model", "inference_spec", "code"])
def test_pinned_input_sha_tamper_is_rejected(example, target):
    save, _, _ = example
    args = save()
    if target == "code": args["code_pins"]["observe_protected_proposals.py"] = "a" * 64
    else: args[target + "_sha256"] = "a" * 64
    with pytest.raises(ValueError): observe.observe(**args, prediction_provider=provider)
    assert not (args["output"] / "report.json").exists()


@pytest.mark.parametrize("fault", ["conf", "crop", "jpeg"])
def test_hash_bound_but_noncanonical_spec_is_rejected(example, fault):
    save, _, _ = example
    args = save()
    spec = json.loads(args["inference_spec"].read_bytes())
    if fault == "conf": spec["detector"]["candidate_confidence"] = .01
    elif fault == "crop": spec["crop"]["padding_ratio"] = .1
    else: spec["crop"]["jpeg_quality"] = 90
    args["inference_spec"].write_text(json.dumps(spec))
    args["inference_spec_sha256"] = sha(args["inference_spec"])
    with pytest.raises(ValueError): observe.observe(**args, prediction_provider=provider)
    assert not args["output"].exists()


@pytest.mark.parametrize("target", ["source", "model_snapshot", "source_snapshot", "crop"])
def test_mid_inference_mutation_leaves_failure_not_success(example, target):
    save, inv, _ = example
    args = save()
    def changed(records):
        for index, frame in enumerate(provider(records)):
            yield frame
            if index == 0:
                if target == "source": path = Path(inv["records"][0]["path"])
                elif target == "model_snapshot": path = records[0].path.parent / "detector.pt"
                elif target == "source_snapshot": path = records[1].path
                else: path = next((args["output"] / "crops").glob("*.jpg"))
                path.write_bytes(b"changed")
    with pytest.raises(ValueError): observe.observe(**args, prediction_provider=changed)
    assert not (args["output"] / "report.json").exists()
    assert (args["output"] / "failed.json").exists()


def test_postpublication_input_mutation_revokes_success_report(example, monkeypatch):
    save, _, _ = example
    args = save()
    read = observe.files.read_file
    def changed(path, *a, **kw):
        if path == args["inventory"] and (args["output"] / "report.json").exists(): path.write_bytes(b"changed")
        return read(path, *a, **kw)
    monkeypatch.setattr(observe.files, "read_file", changed)
    with pytest.raises(ValueError): observe.observe(**args, prediction_provider=provider)
    assert not (args["output"] / "report.json").exists()
    assert (args["output"] / "failed.json").exists()


def test_late_crop_directory_symlink_is_rejected_before_external_write(example, tmp_path):
    save, _, _ = example
    args = save()
    outside = tmp_path / "outside"
    outside.mkdir()
    probe = tmp_path / "probe-link"
    try: probe.symlink_to(outside, target_is_directory=True)
    except OSError: pytest.skip("symlink creation unavailable")
    probe.unlink()
    def swapped(records):
        (args["output"] / "crops").symlink_to(outside, target_is_directory=True)
        yield next(provider(records))
    with pytest.raises(ValueError, match="symlink"):
        observe.observe(**args, prediction_provider=swapped)
    assert list(outside.iterdir()) == []
    assert not (args["output"] / "report.json").exists()


def test_code_snapshot_change_is_detected_without_mutating_repository_source(example, tmp_path, monkeypatch):
    code_dir = tmp_path / "copied-code"
    code_dir.mkdir()
    copied = {}
    for name, source in observe.code_paths().items():
        copied[name] = code_dir / name
        copied[name].write_bytes(source.read_bytes())
    monkeypatch.setattr(observe, "code_paths", lambda: copied)
    save, _, _ = example
    args = save()
    def changed(records):
        copied["observe_protected_proposals.py"].write_bytes(b"modified source snapshot")
        yield from provider(records)
    with pytest.raises(ValueError): observe.observe(**args, prediction_provider=changed)
    assert not (args["output"] / "report.json").exists()
    assert (args["output"] / "failed.json").exists()


@pytest.mark.parametrize("where", ["existing", "metadata", "source"])
def test_output_must_be_fresh_and_disjoint(example, where):
    save, inv, _ = example
    args = save()
    if where == "existing": args["output"].mkdir()
    elif where == "metadata": args["output"] = args["inventory"].parent / "new"
    else: args["output"] = Path(inv["records"][0]["path"]).parent / "new"
    with pytest.raises(ValueError): observe.observe(**args, prediction_provider=provider)
    assert not (args["output"] / "report.json").exists()


def test_real_runtime_path_initializes_cuda_before_bulk_reads_and_freezes_arguments(example, monkeypatch):
    save, _, _ = example
    args = save()
    events, calls = [], []
    read = observe.files.read_file
    def observed_read(path, *a, **kw):
        if path in (args["inventory"], args["model"]): events.append("bulk")
        return read(path, *a, **kw)
    monkeypatch.setattr(observe.files, "read_file", observed_read)
    monkeypatch.setattr(observe.prepare, "eager_initialize_cuda_context", lambda device: events.append("cuda:" + device) or object())
    def canonical(records, **kwargs):
        calls.append(kwargs)
        return provider(records)
    monkeypatch.setattr(observe, "strict_yolo_predictions", canonical)
    monkeypatch.setattr(observe, "runtime_info", lambda real: {"runtime_detector_executed": False, "synthetic_canonical_call_fixture": real})
    report = observe.observe(**args)
    assert events[0] == "cuda:0"
    assert set(calls[0]) == {"model_path"}
    assert report["runtime"]["runtime_detector_executed"] is False


def test_cuda_failure_has_no_cpu_fallback_or_bulk_reads(example, monkeypatch):
    save, _, _ = example
    args = save()
    def fail(device): raise RuntimeError("synthetic CUDA failure")
    monkeypatch.setattr(observe.prepare, "eager_initialize_cuda_context", fail)
    with pytest.raises(RuntimeError, match="CUDA"): observe.observe(**args)
    assert not args["output"].exists()


def test_cli_has_no_custom_or_cpu_runtime_switch(example):
    save, _, _ = example
    args = save()
    cli = [part for key, value in args.items() if key != "code_pins" for part in ("--" + key.replace("_", "-"), str(value))]
    for name, pin in args["code_pins"].items(): cli.extend(["--code-pin", name + "=" + pin])
    with pytest.raises(SystemExit): observe.main(cli + ["--prediction-provider", "fake"])
    with pytest.raises(SystemExit): observe.main(cli + ["--device", "cpu"])


@pytest.mark.parametrize("fault", [None, "none", "length", "bbox_shape", "class_fraction", "nan", "pixels", "cpu"])
def test_strict_actual_results_boundary_never_turns_malformed_arrays_into_absence(example, monkeypatch, fault):
    save, inv, _ = example
    args = save()
    path = Path(inv["records"][0]["path"])
    record = observe.prepare.SourceRecord(path, "protected", sha(path), None)
    class Tensor:
        def __init__(self, value): self.value = np.asarray(value, dtype=np.float32)
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.value
    boxes = SimpleNamespace(xyxy=Tensor(np.empty((0, 4))), cls=Tensor([]), conf=Tensor([]))
    if fault == "none": boxes = None
    elif fault == "length": boxes.xyxy = Tensor([[1, 1, 20, 20]])
    elif fault == "bbox_shape": boxes.xyxy = Tensor([])
    elif fault in ("class_fraction", "nan"):
        boxes.xyxy, boxes.cls, boxes.conf = Tensor([[1, 1, 20, 20]]), Tensor([.5 if fault == "class_fraction" else 0]), Tensor([float("nan") if fault == "nan" else .8])
    original = cv2.imread(str(path))
    if fault == "pixels": original[:] = 255
    result = SimpleNamespace(orig_img=original, orig_shape=(80, 120), boxes=boxes,
                             names=dict(enumerate(observe.prepare.CLASS_NAMES)))
    calls = []
    class Detector:
        def __init__(self, model, task):
            assert model == str(args["model"]) and task == "detect"
            self.predictor = SimpleNamespace(model=SimpleNamespace(device=SimpleNamespace(type="cpu" if fault == "cpu" else "cuda", index=0)))
        def predict(self, **kwargs):
            calls.append(kwargs)
            return iter([result])
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Detector))
    if fault is None:
        frames = list(observe.strict_yolo_predictions([record], model_path=args["model"]))
        assert frames[0].proposals == ()
        assert calls == [dict(source=[str(path)], device="0", batch=1, imgsz=640, conf=.1, iou=.7, stream=True, save=False, verbose=False)]
    else:
        with pytest.raises(ValueError): list(observe.strict_yolo_predictions([record], model_path=args["model"]))
