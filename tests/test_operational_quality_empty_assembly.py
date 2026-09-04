"""Real CPU prepare/teacher/assembly evidence with zero quality exclusions."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from scripts import assemble_operational_quality_exclusions as assembler
from scripts import operational_quality_assembly_contract as contract


def _helpers():
    path = Path(__file__).with_name("test_build_operational_teacher_manifest.py")
    spec = importlib.util.spec_from_file_location("_empty_quality_test_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def valid_args(tmp_path):
    helpers = _helpers()
    args, _, _ = helpers._objective_quality_assembly_fixture(
        tmp_path / "source", no_quality_exclusions=True,
    )
    return helpers, args


def _validate(args, tmp_path):
    manifest_path = args["output_dir"] / assembler.ASSEMBLY_FILES["manifest"]
    value, content = contract._load_json(manifest_path, "manifest")
    bundle = contract._validate_operational_quality_assembly(
        receipt_path=args["output_dir"] / assembler.ASSEMBLY_FILES["receipt"],
        quality_path=manifest_path, quality_value=value, quality_content=content,
        output_dir=tmp_path / "consumer",
    )
    return value, bundle


def _rewrite_receipt(args, mutation):
    root = args["output_dir"]
    receipt_path = root / assembler.ASSEMBLY_FILES["receipt"]
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutation(value)
    receipt_path.write_bytes(contract._canonical_json(value))
    (root / assembler.ASSEMBLY_FILES["marker"]).write_bytes(
        contract._quality_assembly_marker_bytes(
            manifest_content=(root / assembler.ASSEMBLY_FILES["manifest"]).read_bytes(),
            receipt_content=receipt_path.read_bytes(),
        )
    )


def _rebuild_teacher(helpers, args):
    output = args["teacher_output_dir"].with_name("teacher-output-updated")
    kwargs = {name: value for name, value in args.items() if name in {
        "teacher_queue", "teacher_labels", "capture_inventory", "known_audit",
        "provider_a_manifest", "provider_a_model", "provider_a_spec",
        "provider_b_manifest", "provider_b_model", "provider_b_spec", "image_root",
    }}
    helpers._real_build_operational_teacher_manifest(
        **kwargs, provider_a_name="detector_a", provider_b_name="segmenter_b",
        output_dir=output,
    )
    args["teacher_output_dir"] = output


def test_full_zero_exclusion_bundle_is_valid_and_standalone_empty_is_not(valid_args, tmp_path):
    _, args = valid_args
    receipt = assembler.assemble_operational_quality_exclusions(**args)
    manifest, bundle = _validate(args, tmp_path)
    assert manifest["entries"] == []
    assert manifest["excluded_source_count"] == 0
    assert manifest["reason_counts"] == {}
    assert manifest["source_list_sha256"] == hashlib.sha256(b"[]\n").hexdigest()
    assert receipt["selected_source_count"] == 0
    assert receipt["scope"]["objective_prepare_bundle_validated"] is True
    assert receipt["scope"]["subjective_quality_source_count"] == 0
    assert receipt["scope"]["objective_quality_source_count"] == 0
    assert receipt["scope"]["teacher_subjective_quality_included"] is False
    assert receipt["scope"]["objective_queue_quality_included"] is False
    assert contract._validate_quality_manifest(manifest, assembly_bundle=bundle) == {}
    with pytest.raises(ValueError, match="validated full assembly bundle"):
        contract._validate_quality_manifest(manifest)
    assert not any(manifest["authority"].values())
    assert not any(receipt["authority"].values())
    original = {path.name: path.read_bytes() for path in args["output_dir"].iterdir()}
    with pytest.raises(FileExistsError):
        assembler.assemble_operational_quality_exclusions(**args)
    assert original == {path.name: path.read_bytes() for path in args["output_dir"].iterdir()}


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(selected_source_count=False),
    lambda value: value.update(selected_source_count=0.0),
    lambda value: value.update(selected_source_count=-1),
    lambda value: value.update(selected_source_count=None),
    lambda value: value.pop("selected_source_count"),
    lambda value: value.update(assembly_mode="legacy_subjective_only"),
    lambda value: value.update(status="dry_run"),
    lambda value: value["scope"].update(objective_prepare_bundle_validated=False),
    lambda value: value["scope"].update(subjective_quality_source_count=False),
    lambda value: value["input_sha256"].pop("teacher_labels"),
    lambda value: value["input_sha256"].pop("objective_prepare_objective_receipt"),
])
def test_empty_bundle_needs_exact_full_receipt(valid_args, tmp_path, mutation):
    _, args = valid_args
    assembler.assemble_operational_quality_exclusions(**args)
    _rewrite_receipt(args, mutation)
    with pytest.raises(ValueError):
        _validate(args, tmp_path)


def test_empty_bundle_rechecks_proof_after_validation(valid_args, tmp_path):
    _, args = valid_args
    assembler.assemble_operational_quality_exclusions(**args)
    manifest, bundle = _validate(args, tmp_path)
    bundle.receipt_path.write_bytes(bundle.receipt_content + b"\n")
    with pytest.raises(RuntimeError, match="receipt changed"):
        contract._validate_quality_manifest(manifest, assembly_bundle=bundle)


@pytest.mark.parametrize("mutation", ["missing_label", "invalid_label", "no_objective"])
def test_incomplete_subjective_or_objective_work_is_not_zero_success(valid_args, mutation):
    helpers, args = valid_args
    if mutation == "no_objective":
        args["objective_prepare_output_dir"] = None
    else:
        labels = [json.loads(line) for line in args["teacher_labels"].read_text(encoding="utf-8").splitlines()]
        if mutation == "missing_label":
            labels.pop()
        elif mutation == "invalid_label":
            labels[0]["errors"] = ["incomplete provider attempt"]
        else:
            labels[0]["minimum_confidence"] = 0.2
            for decision in labels[0]["passes"]:
                decision["confidence"] = 0.2
        helpers._jsonl(args["teacher_labels"], labels)
        _rebuild_teacher(helpers, args)
    with pytest.raises(ValueError, match="zero exclusions require|quality teacher consensus"):
        assembler.assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()


def test_empty_batch_without_subjective_evidence_is_not_success():
    with pytest.raises(ValueError, match="nonempty fully labeled"):
        assembler._validate_zero_exclusion_coverage(
            queue_rows=[], label_rows=[], rejections=[],
            objective_prepare_output_dir=Path("unused"),
        )


def _set_nonconsensus(helpers, args):
    labels = [json.loads(line) for line in args["teacher_labels"].read_text(encoding="utf-8").splitlines()]
    row = labels[0]
    base = row["passes"][0]
    row["passes"] = [dict(base, material=material) for material in ("pet", "glass", "plastic")]
    row.update(consensus=False, consensus_decision=None, minimum_confidence=0.0)
    return labels


def test_completed_nonconsensus_is_retained_as_rejection_not_positive(valid_args, tmp_path):
    helpers, args = valid_args
    labels = _set_nonconsensus(helpers, args)
    nonconsensus_sha = labels[0]["sha256"]
    helpers._jsonl(args["teacher_labels"], labels)
    _rebuild_teacher(helpers, args)
    receipt = assembler.assemble_operational_quality_exclusions(**args)
    manifest, bundle = _validate(args, tmp_path)
    assert receipt["selected_source_count"] == 0
    assert contract._validate_quality_manifest(manifest, assembly_bundle=bundle) == {}
    rejected = json.loads((args["teacher_output_dir"] / helpers.ARTIFACT_NAMES["rejections"]).read_text())
    assert any(row["sha256"] == nonconsensus_sha and "no_exact_tuple_consensus" in row["reasons"]
               for row in rejected["rejections"])
    accepted = (args["teacher_output_dir"] / helpers.ARTIFACT_NAMES["jsonl"]).read_text()
    assert nonconsensus_sha not in accepted


@pytest.mark.parametrize("mutation", [
    lambda row: row["passes"].pop(),
    lambda row: row.update(errors=["HTTP 400"]),
    lambda row: row.update(minimum_confidence=False),
    lambda row: row["passes"][0].update(confidence=True),
    lambda row: row["passes"][0].update(material="unknown_material"),
    lambda row: row["passes"][0].update(training_usable=1),
    lambda row: row["passes"][0].update(confidence=float("nan")),
    lambda row: row["passes"][0].update(extra="forbidden"),
    lambda row: row["passes"][0].update(material="glass"),
    lambda row: row["teacher_contract"].update(model_identifier="changed"),
])
def test_incomplete_or_invalid_nonconsensus_is_not_empty_success(valid_args, mutation):
    helpers, args = valid_args
    labels = _set_nonconsensus(helpers, args)
    mutation(labels[0])
    helpers._jsonl(args["teacher_labels"], labels)
    _rebuild_teacher(helpers, args)
    with pytest.raises(ValueError):
        assembler.assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()


def test_completed_low_confidence_rejection_is_not_promoted_by_empty_assembly(valid_args):
    helpers, args = valid_args
    labels = [json.loads(line) for line in args["teacher_labels"].read_text(encoding="utf-8").splitlines()]
    low_sha = labels[0]["sha256"]
    labels[0]["minimum_confidence"] = 0.2
    for item in labels[0]["passes"]:
        item["confidence"] = 0.2
    helpers._jsonl(args["teacher_labels"], labels)
    _rebuild_teacher(helpers, args)
    assert assembler.assemble_operational_quality_exclusions(**args)["selected_source_count"] == 0
    accepted = (args["teacher_output_dir"] / helpers.ARTIFACT_NAMES["jsonl"]).read_text()
    assert low_sha not in accepted


def test_zero_bundle_missing_objective_artifact_is_not_success(valid_args):
    _, args = valid_args
    artifact = args["objective_prepare_output_dir"] / assembler.capture_queue.OUTPUT_FILES["objective_receipt"]
    artifact.unlink()
    with pytest.raises(ValueError, match="file set"):
        assembler.assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()


@pytest.mark.parametrize("target", ["provider_a_model", "teacher_labels", "objective_metadata"])
def test_zero_bundle_rehashes_sources_before_publication(valid_args, monkeypatch, target):
    _, args = valid_args
    real_serializer = assembler._manifest_value
    def mutate_after_serializing(entries):
        result = real_serializer(entries)
        if target == "objective_metadata":
            path = next(args["image_root"].glob("*.json"))
        else:
            path = args[target]
        path.write_bytes(path.read_bytes() + b"\n")
        return result
    monkeypatch.setattr(assembler, "_manifest_value", mutate_after_serializing)
    with pytest.raises(RuntimeError, match="changed before publish"):
        assembler.assemble_operational_quality_exclusions(**args)
    assert not args["output_dir"].exists()


def test_model_weights_are_hashed_in_bounded_chunks(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    content = b"weights" * 400000
    model.write_bytes(content)
    original = Path.open
    sizes = []
    class BoundedReader:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.handle.close()
        def fileno(self):
            return self.handle.fileno()
        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024
            sizes.append(size)
            return self.handle.read(size)
    def open_model(path, *args, **kwargs):
        handle = original(path, *args, **kwargs)
        return BoundedReader(handle) if path == model else handle
    monkeypatch.setattr(Path, "open", open_model)
    resolved, digest = assembler._stable_file_sha256(model, description="provider weights")
    assert resolved == model.resolve()
    assert digest == hashlib.sha256(content).hexdigest()
    assert len(sizes) >= 3


def test_assembly_never_loads_provider_weight_bytes(valid_args, monkeypatch):
    _, args = valid_args
    original = assembler._stable_regular_file
    def no_weight_bytes(path, *, description):
        assert path not in {args["provider_a_model"], args["provider_b_model"]}
        return original(path, description=description)
    monkeypatch.setattr(assembler, "_stable_regular_file", no_weight_bytes)
    receipt = assembler.assemble_operational_quality_exclusions(**args)
    for field in ("provider_a_model", "provider_b_model"):
        assert receipt["input_sha256"][field] == hashlib.sha256(args[field].read_bytes()).hexdigest()
