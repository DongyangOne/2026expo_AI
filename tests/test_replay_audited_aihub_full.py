"""CPU orchestration tests: real original reader/validator, mocked CUDA/YOLO only."""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import materialize_audited_aihub_sources as materializer
from scripts import prepare_proposal_verifier_dataset as prepare
from scripts import validate_v4_background_candidates as validator
from scripts.nas import replay_audited_aihub_full as runner
from test_materialize_audited_aihub_sources import fixture as materializer_inputs


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_bytes(runner.json_bytes(value))


def predictions(records, **_):
    for source in records:
        image = cv2.imdecode(np.frombuffer(source.path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        yield prepare.PredictedFrame(source, width, height, (
            prepare.Proposal(0, .9, source.ground_truth.xyxy(width, height), "can"),
        ))


@pytest.fixture
def setup(materializer_inputs, tmp_path, monkeypatch, request):
    cohort, save, data, originals, annotations, metadata = materializer_inputs
    cohort["full_cohort"] = True
    if getattr(request, "param", False):
        source, label = originals[0].with_name("blank.png"), annotations[0].with_name("blank.json")
        source.write_bytes(cv2.imencode(".png", np.zeros((480, 800, 3), np.uint8))[1].tobytes())
        annotation = json.loads(annotations[0].read_text(encoding="utf-8")); annotation["IMAGE_INFO"]["FILE_NAME"] = source.name
        dump(label, annotation)
        row = copy.deepcopy(cohort["records"][0])
        row.update(source_id=hashlib.sha1(f"Training/{label.parent.name}/{label.name}".encode()).hexdigest()[:20],
                   source_path_b64=materializer.audit.encode_path(source), label_path_b64=materializer.audit.encode_path(label),
                   source_sha256=sha(source), label_sha256=sha(label))
        cohort["records"].append(row); originals.append(source); annotations.append(label)
    materializer_args = save()
    materializer.materialize_sources(**materializer_args)
    generation = tmp_path / "generation"
    generation.mkdir()
    raw, control = generation / "raw", generation / "control"
    control.mkdir()
    model = tmp_path / "models" / "best.pt"
    model.parent.mkdir()
    model.write_bytes(b"CPU test fixture; never passed to a model loader")
    prepare.build_proposal_verifier_dataset(model_path=model, data_path=data / "dataset.yaml", dataset_dir=data,
        output_dir=raw, device="0", batch=1, imgsz=640, conf=.1, nms_iou=.7, positive_iou=.5,
        negative_iou=.1, crop_size=320, padding=.08, max_per_class=10, val_max_per_class=10,
        max_background=10, val_max_background=10, seed=20260901, min_free_gb=0, max_output_gb=1,
        jpeg_quality=92, proposal_selection="runtime-top1", background_policy="strict-zero-intersection",
        background_gt_margin=.1, aihub_origin="aihub_original_annotation_v1", prediction_provider=predictions,
        audited_aihub_report=data / "report.json", audited_aihub_report_sha256=sha(data / "report.json"),
        audited_aihub_cohort=materializer_args["cohort"])
    source_inventory, output_inventory = control / "dataset_input_inventory.json", control / "raw_output_inventory.json"
    entries = []
    for row in map(json.loads, (data / "lineage.jsonl").read_text().splitlines()):
        for kind, field in (("source", "image_ref"), ("label", "label_ref")):
            p = data / row[field]
            entries.append(dict(kind=kind, path=p.as_posix(), split=row["split"], exists=True,
                                size=p.stat().st_size, sha256=sha(p)))
    dump(source_inventory, dict(contract="resolved_yolo_train_val_sources_and_label_sidecars_sha256.v1",
        data_path=(data / "dataset.yaml").as_posix(), dataset_dir=data.as_posix(), artifact_count=len(entries), artifacts=entries))
    state = {"calls": [], "reader_calls": [], "cuda": []}
    args = dict(generation_dir=generation, code_root=ROOT, output_dir=tmp_path / "replay",
        original_dataset_root=tmp_path / "dataset", detector_model=model, detector_model_sha256=sha(model),
        inference_spec=ROOT / "configs/detector_inference_v3.json", inference_spec_sha256=sha(ROOT / "configs/detector_inference_v3.json"),
        audited_aihub_report=data / "report.json", audited_aihub_report_sha256=sha(data / "report.json"),
        audited_aihub_cohort=materializer_args["cohort"], audited_aihub_cohort_sha256=sha(materializer_args["cohort"]),
        code_pins={name: sha(ROOT / name) for name in runner.CODE_FILES})

    def refresh():
        tree = runner.raw_tree(raw)
        dump(output_inventory, dict(root=raw.as_posix(), file_count=len(tree), files=[dict(path=p, **v) for p, v in tree.items()]))
        inputs = [model, data / "dataset.yaml", source_inventory, data / "report.json", materializer_args["cohort"]]
        inputs += [ROOT / "scripts" / name for name in ("prepare_proposal_verifier_dataset.py",
            "verifier_preprocessing_contract.py", "nas/run_v4_reproducible_generation.sh", "audited_aihub_snapshot.py",
            "audit_aihub_original_annotations.py", "materialize_audited_aihub_sources.py")]
        input_marker, output_marker = control / "inputs.sha256", control / "outputs.sha256"
        for p, files in ((input_marker, inputs), (output_marker, [raw / "manifest.csv", raw / "dataset_info.json", output_inventory])):
            p.write_text("".join(f"{sha(f)}  {f.as_posix()}\n" for f in files), encoding="utf-8")
        ready = control / "raw_generation_ready.json"
        dump(ready, dict(schema_version=1, status="raw_generation_ready", artifact_role=runner.RAW_ROLE,
            batch=1, seed=20260901, **{k: False for k in ("validator_authority", "judge_authority", "training_authority",
            "blind_test_authority", "production_deployment_authorized")}, bindings=dict(input_marker_sha256=sha(input_marker),
            output_marker_sha256=sha(output_marker), manifest_sha256=sha(raw / "manifest.csv"), dataset_info_sha256=sha(raw / "dataset_info.json"))))
        args.update(generation_ready_sha256=sha(ready), manifest_sha256=sha(raw / "manifest.csv"), dataset_info_sha256=sha(raw / "dataset_info.json"))
    refresh()
    original_validate = validator.validate_manifest
    actual_reader = validator._audited_aihub_reader()
    def read(*a, **kw):
        assert state["cuda"]
        state["reader_calls"].append(kw)
        return actual_reader(*a, **kw)
    def validate(**values):
        assert values["prediction_provider"] is None and values["diagnostic_only"] is False
        state["calls"].append(values)
        return original_validate(**values)
    monkeypatch.setattr(validator, "_audited_aihub_reader", lambda: read)
    monkeypatch.setattr(validator, "validate_manifest", validate)
    monkeypatch.setattr(validator, "eager_initialize_cuda_context", lambda device: state["cuda"].append(device) or object())
    monkeypatch.setattr(validator, "iter_yolo_predictions", predictions)
    state.update(args=args, originals=originals, annotations=annotations, metadata=metadata, raw=raw, data=data,
                 control=control, refresh=refresh, original_validate=validate, source_inventory=source_inventory)
    return state


def failed(state):
    out = state["args"]["output_dir"]
    assert not (out / "replay_ready.json").exists()
    report = json.loads((out / "failed.json").read_text())
    assert report["ready_for_lineage_upgrade"] is False
    assert all(report[key] is False for key in runner.AUTHORITY)


def test_full_real_reader_and_strict_validator_once_with_originals_unchanged(setup):
    args = setup["args"]
    initial_raw = runner.raw_tree(setup["raw"])
    originals = {p: sha(p) for p in setup["originals"] + setup["annotations"]}
    result = runner.run(**args)
    assert len(setup["calls"]) == len(setup["reader_calls"]) == 1
    assert setup["reader_calls"][0]["require_full_cohort"] is True
    assert setup["cuda"] == ["0", "0"]  # Wrapper and validator, same process; no duplicate reader.
    assert result["ready_for_lineage_upgrade"] is True and result["runtime_replay_count"] == 1
    assert result["raw_rows"] == result["materialized_sources"] == 2
    assert all(result[key] is False for key in runner.AUTHORITY)
    assert runner.raw_tree(setup["raw"]) == initial_raw and {p: sha(p) for p in originals} == originals
    workspace = args["output_dir"] / "replay"
    for name in runner.WORKSPACE_NAMES:
        assert (workspace / name).is_symlink() and not os.path.isabs(os.readlink(workspace / name))
    rows = list(csv.DictReader((workspace / "validated_manifest.csv").open()))
    assert all(row["original_source_sha256"] != row["source_sha256"] for row in rows)
    assert not (args["output_dir"] / "failed.json").exists()


@pytest.mark.parametrize("key", ["manifest_sha256", "dataset_info_sha256", "generation_ready_sha256",
    "detector_model_sha256", "inference_spec_sha256", "audited_aihub_cohort_sha256"])
def test_bad_fixed_pin_blocks_before_validator(setup, key):
    setup["args"][key] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.run(**setup["args"])
    assert setup["calls"] == setup["reader_calls"] == []
    failed(setup)


@pytest.mark.parametrize("field,value", [("require_full_cohort", False), ("require_full_cohort", 1), ("report_sha256", "0" * 64)])
def test_partial_or_forged_snapshot_info_refused(setup, field, value):
    path = setup["raw"] / "dataset_info.json"
    info = json.loads(path.read_text()); info["audited_aihub_snapshot"][field] = value; dump(path, info)
    setup["refresh"]()
    with pytest.raises(ValueError, match="full audited"):
        runner.run(**setup["args"])
    assert setup["reader_calls"] == []; failed(setup)


@pytest.mark.parametrize("kind", ["source_sha", "split", "missing", "traversal"])
def test_bound_generation_inventory_must_equal_materialized_lineage(setup, kind):
    p = setup["source_inventory"]; value = json.loads(p.read_text())
    if kind == "source_sha": value["artifacts"][0]["sha256"] = "0" * 64
    if kind == "split": value["artifacts"][0]["split"] = "validation"
    if kind == "missing": value["artifacts"].pop(); value["artifact_count"] -= 1
    if kind == "traversal": value["artifacts"][0]["path"] = str(setup["data"] / ".." / "bad")
    dump(p, value); setup["refresh"]()
    with pytest.raises(ValueError): runner.run(**setup["args"])
    assert setup["reader_calls"] == []; failed(setup)


@pytest.mark.parametrize("target", ["raw", "materialized", "original", "cohort", "metadata", "existing"])
def test_output_cannot_touch_input_parents_or_existing_directory(setup, target):
    args = setup["args"]
    roots = dict(raw=setup["raw"], materialized=setup["data"], original=args["original_dataset_root"],
                 cohort=args["audited_aihub_cohort"].parent, metadata=setup["metadata"].parent)
    if target == "existing": args["output_dir"].mkdir()
    else: args["output_dir"] = roots[target] / "do_not_create"
    with pytest.raises(ValueError, match="fresh/disjoint"): runner.run(**args)
    assert setup["cuda"] == []
    if target != "existing": assert not args["output_dir"].exists()


@pytest.mark.parametrize("fault", ["original", "annotation", "derived", "code_pin", "extra_raw", "raw_symlink", "failed"])
def test_invalid_actual_inputs_never_produce_ready(setup, fault):
    if fault in ("original", "annotation"):
        p = setup["originals" if fault == "original" else "annotations"][0]
        p.write_bytes(p.read_bytes() + b" ")
    if fault == "derived":
        p = next((setup["data"] / "images").rglob("*.jpg")); p.write_bytes(p.read_bytes() + b" ")
    if fault == "code_pin": setup["args"]["code_pins"]["scripts/audited_aihub_snapshot.py"] = "0" * 64
    if fault == "extra_raw": (setup["raw"] / "unexpected.txt").write_text("extra")
    if fault == "raw_symlink":
        p = next((setup["raw"] / "training").rglob("*.jpg")); destination = p.with_suffix(".copy"); p.rename(destination); p.symlink_to(destination)
    if fault == "failed": (setup["control"] / "failed.txt").write_text("failed")
    with pytest.raises((ValueError, RuntimeError)): runner.run(**setup["args"])
    if fault == "code_pin": assert not setup["args"]["output_dir"].exists()
    else: failed(setup)


@pytest.mark.parametrize("fault", ["custom", "diagnostic", "tolerance", "rows", "binding"])
def test_runtime_attestation_cannot_be_downgraded(setup, monkeypatch, fault):
    original = setup["original_validate"]
    def forged(**args):
        report = original(**args)
        if fault == "custom": report["contract"]["proposal_provenance"]["provider_kind"] = "custom_non_authoritative"
        if fault == "diagnostic": report["ready_for_lineage_upgrade"] = False
        if fault == "tolerance": report["contract"]["proposal_provenance"]["confidence_abs_tolerance"] = 1e-3
        if fault == "rows": report["rows"] = True
        if fault == "binding": report["bindings"]["input_manifest_sha256"] = "0" * 64
        dump(args["output_report"], report)
        return report
    monkeypatch.setattr(validator, "validate_manifest", forged)
    with pytest.raises(ValueError): runner.run(**setup["args"])
    failed(setup)
    assert (setup["args"]["output_dir"] / "replay" / "validated_manifest.csv").exists()


@pytest.mark.parametrize("when", ["validator", "ready"])
def test_raw_change_during_replay_or_ready_preserves_failure(setup, monkeypatch, when):
    crop = next((setup["raw"] / "training").rglob("*.jpg"))
    if when == "validator":
        original = setup["original_validate"]
        def mutate(**args):
            result = original(**args); crop.write_bytes(crop.read_bytes() + b"changed"); return result
        monkeypatch.setattr(validator, "validate_manifest", mutate)
    else:
        original = runner.publish
        def mutate(path, value, publications):
            original(path, value, publications)
            if path.name == "replay_ready.json": crop.write_bytes(crop.read_bytes() + b"changed")
        monkeypatch.setattr(runner, "publish", mutate)
    with pytest.raises(ValueError): runner.run(**setup["args"])
    failed(setup)
    assert (setup["args"]["output_dir"] / "replay" / "validation_report.json").exists()


def test_ready_link_error_removes_only_own_marker_and_preserves_report(setup, monkeypatch):
    original = os.link
    def fail_after_link(source, destination, **kwargs):
        original(source, destination, **kwargs)
        if Path(destination).name == "replay_ready.json": raise OSError("publication interrupted")
    monkeypatch.setattr(runner.os, "link", fail_after_link)
    with pytest.raises(OSError): runner.run(**setup["args"])
    failed(setup)


def test_original_change_at_ready_boundary_is_not_hidden_by_completed_validator(setup, monkeypatch):
    original = runner.publish
    def mutate(path, value, publications):
        original(path, value, publications)
        if path.name == "replay_ready.json":
            source = setup["originals"][0]
            source.write_bytes(source.read_bytes() + b"changed")
    monkeypatch.setattr(runner, "publish", mutate)
    with pytest.raises(ValueError, match="source or label changed"): runner.run(**setup["args"])
    failed(setup)


@pytest.mark.parametrize("setup", [True], indirect=True)
def test_quality_excluded_original_remains_bound_at_terminal_boundary(setup, monkeypatch):
    materialized = json.loads(setup["args"]["audited_aihub_report"].read_text())
    assert materialized["quality_excluded_sources"] == 1
    original = runner.publish
    def mutate(path, value, publications):
        original(path, value, publications)
        if path.name == "replay_ready.json":
            source = setup["originals"][-1]
            source.write_bytes(source.read_bytes() + b"changed")
    monkeypatch.setattr(runner, "publish", mutate)
    with pytest.raises(ValueError, match="source or label changed"): runner.run(**setup["args"])
    assert len(setup["reader_calls"]) == 1
    failed(setup)


def test_foreign_ready_is_never_removed(setup, monkeypatch):
    original = runner.publish
    foreign = b'{"foreign":true}\n'
    def publish(path, value, publications):
        if path.name == "replay_ready.json": path.write_bytes(foreign)
        original(path, value, publications)
    monkeypatch.setattr(runner, "publish", publish)
    with pytest.raises(FileExistsError): runner.run(**setup["args"])
    out = setup["args"]["output_dir"]
    assert (out / "replay_ready.json").read_bytes() == foreign
    assert json.loads((out / "failed.json").read_text())["ready_for_lineage_upgrade"] is False


def test_cuda_failure_occurs_before_full_reader(setup, monkeypatch):
    def unavailable(_): raise RuntimeError("CUDA unavailable")
    monkeypatch.setattr(validator, "eager_initialize_cuda_context", unavailable)
    with pytest.raises(RuntimeError): runner.run(**setup["args"])
    assert setup["reader_calls"] == setup["calls"] == []; failed(setup)


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_metadata_parser_rejects_ambiguous_json(raw):
    with pytest.raises(ValueError): runner.parse_json(raw)


def test_metadata_size_limit_and_code_set_validation(setup, monkeypatch):
    monkeypatch.setattr(runner, "MAX_METADATA_BYTES", 16)
    with pytest.raises(ValueError, match="byte limit"): runner.run(**setup["args"])
    assert not setup["args"]["output_dir"].exists()


@pytest.mark.parametrize("fault", ["missing", "extra"])
def test_exact_code_pin_catalog_required(setup, fault):
    pins = setup["args"]["code_pins"]
    if fault == "missing": pins.pop("scripts/audited_aihub_snapshot.py")
    else: pins["scripts/unrelated.py"] = "0" * 64
    with pytest.raises(ValueError, match="code pin set"): runner.run(**setup["args"])
    assert not setup["args"]["output_dir"].exists() and setup["cuda"] == []


def test_cli_exposes_no_diagnostic_or_custom_escape():
    with pytest.raises(SystemExit) as error: runner.main(["--help"])
    assert error.value.code == 0
