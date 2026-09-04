import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from scripts import build_operational_source_evidence as adapter


def _helpers():
    path = Path(__file__).with_name("test_build_operational_teacher_manifest.py")
    spec = importlib.util.spec_from_file_location("_source_evidence_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def case(tmp_path):
    helpers = _helpers()
    args, _, _ = helpers._objective_quality_assembly_fixture(tmp_path / "input", no_quality_exclusions=True)
    helpers.assemble_operational_quality_exclusions(**args)
    arguments = {name: args[name] for name in adapter.INPUT_NAMES}
    arguments.update(
        teacher_output_dir=args["teacher_output_dir"], image_root=args["image_root"],
        quality_assembly_receipt=args["output_dir"] / helpers.ASSEMBLY_FILES["receipt"],
        output_dir=tmp_path / "source-evidence",
    )
    return helpers, arguments


def _bundle(case):
    _, arguments = case
    receipt = adapter.build_source_evidence(**arguments)
    return arguments["output_dir"], receipt


def test_real_teacher_dryrun_and_quality_bundle_produce_bound_train_only_sources(case):
    _, arguments = case
    originals = {path: path.read_bytes() for path in arguments["teacher_output_dir"].iterdir()}
    output, receipt = _bundle(case)
    rows = adapter.validate_source_evidence_bundle(output)
    assert len(rows) == receipt["source_count"] == 3
    assert receipt["artifact_role"] == "source_evidence_only_not_training_authority"
    assert not any(receipt["authority"].values())
    assert originals == {path: path.read_bytes() for path in arguments["teacher_output_dir"].iterdir()}
    labels = {row["sha256"]: row for row in (json.loads(line) for line in arguments["teacher_labels"].read_text(encoding="utf-8").splitlines())}
    for row in rows:
        evidence_path = output / row["source_evidence_ref"]
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        assert row["auditor_sha256"] == hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        assert "auditor_sha256" not in evidence["record"]
        assert row["teacher_output_sha256"] == adapter._sha(adapter._json_bytes(labels[row["source_sha256"]]))
        assert row["localizer_output_sha256"] == adapter._sha(adapter._json_bytes(evidence["independent_localization"]))
        assert row["source_sha256"] == hashlib.sha256(Path(row["source_filepath"]).read_bytes()).hexdigest()
        assert row["role"] == "train"
        assert row["dent"] == row["label"] == -1
        assert type(row["foreign_material"]) is int
        assert row["bbox_source"] == "independent_localization_consensus"
        assert row["training_crop_ready"] is False
        assert row["runtime_detector_executed"] is False
        assert row["captured_at"].endswith("Z")
        assert set(evidence["input_sha256"]) == set(receipt["inputs"])
    rendered = "".join(path.read_text(encoding="utf-8") for path in output.rglob("*") if path.is_file())
    assert "private-objective-client" not in rendered
    assert "private-device" not in rendered
    assert '"client_id"' not in rendered
    assert '"device_id"' not in rendered


@pytest.mark.parametrize("target", ["teacher_labels", "known_audit", "provider_a_model", "provider_b_spec"])
def test_changed_frozen_input_does_not_publish(case, target):
    _, arguments = case
    path = arguments[target]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        adapter.build_source_evidence(**arguments)
    assert not arguments["output_dir"].exists()


def test_missing_quality_receipt_is_not_optional(case):
    _, arguments = case
    arguments["quality_assembly_receipt"].unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        adapter.build_source_evidence(**arguments)
    assert not arguments["output_dir"].exists()


def test_dry_run_digest_mismatch_rejects_even_hash_bound_originals(case, monkeypatch):
    _, arguments = case
    original = adapter.teacher.build_operational_teacher_manifest
    def changed_result(**kwargs):
        assert kwargs["dry_run"] is True
        result = original(**kwargs)
        result["output_digests"]["jsonl_sha256"] = "a" * 64
        return result
    monkeypatch.setattr(adapter.teacher, "build_operational_teacher_manifest", changed_result)
    with pytest.raises(ValueError, match="dry-run output digests"):
        adapter.build_source_evidence(**arguments)
    assert not arguments["output_dir"].exists()


@pytest.mark.parametrize("when", ["during_dryrun", "at_publication"])
def test_source_mutation_never_returns_success(case, monkeypatch, when):
    _, arguments = case
    source = next(arguments["image_root"].glob("*.png"))
    if when == "during_dryrun":
        original = adapter.teacher.build_operational_teacher_manifest
        def mutate(**kwargs):
            result = original(**kwargs)
            source.write_bytes(source.read_bytes() + b"changed")
            return result
        monkeypatch.setattr(adapter.teacher, "build_operational_teacher_manifest", mutate)
    else:
        original = adapter.assembler._publish_directory_no_replace
        def mutate(staging, destination):
            original(staging, destination)
            source.write_bytes(source.read_bytes() + b"changed")
        monkeypatch.setattr(adapter.assembler, "_publish_directory_no_replace", mutate)
    with pytest.raises(RuntimeError, match="image changed"):
        adapter.build_source_evidence(**arguments)
    if arguments["output_dir"].exists():
        assert (arguments["output_dir"] / "failed.json").exists()
        with pytest.raises(ValueError, match="failure marker"):
            adapter.validate_source_evidence_bundle(arguments["output_dir"])


@pytest.mark.parametrize("target", ["index", "evidence", "receipt", "extra", "source", "model", "quality"])
def test_reader_revalidates_every_material_binding(case, target):
    _, arguments = case
    output, _ = _bundle(case)
    if target == "index":
        path = output / adapter.FILES["index"]
    elif target == "evidence":
        path = next((output / "source_evidence").glob("*.json"))
    elif target == "receipt":
        path = output / adapter.FILES["receipt"]
        value = json.loads(path.read_text(encoding="utf-8"))
        value["source_count"] = True
        path.write_bytes(adapter._json_bytes(value))
    elif target == "extra":
        path = output / "extra.txt"
        path.write_bytes(b"extra")
    elif target == "source":
        path = next(arguments["image_root"].glob("*.png"))
    elif target == "model":
        path = arguments["provider_b_model"]
    else:
        path = arguments["quality_assembly_receipt"]
    if target not in ("receipt", "extra"):
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises((ValueError, RuntimeError)):
        adapter.validate_source_evidence_bundle(output)


def test_immutable_output_and_nested_input_root_are_rejected(case):
    _, arguments = case
    output, _ = _bundle(case)
    old = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        adapter.build_source_evidence(**arguments)
    assert old == {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    arguments["output_dir"] = arguments["image_root"] / "nested"
    with pytest.raises(ValueError, match="nested"):
        adapter.build_source_evidence(**arguments)


def test_source_ancestor_symlink_is_rejected(case):
    _, arguments = case
    source = next(arguments["image_root"].glob("*.png"))
    other = source.with_name("replacement.png")
    source.rename(other)
    try:
        source.symlink_to(other)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="symlink"):
        adapter.build_source_evidence(**arguments)
    assert not arguments["output_dir"].exists()


def test_provider_model_bytes_are_not_retained_in_adapter(case, monkeypatch):
    _, arguments = case
    original = adapter.assembler._stable_regular_file
    def no_large_read(path, *, description):
        assert path not in (arguments["provider_a_model"], arguments["provider_b_model"])
        return original(path, description=description)
    monkeypatch.setattr(adapter.assembler, "_stable_regular_file", no_large_read)
    assert adapter.build_source_evidence(**arguments)["source_count"] == 3


def _rebind_quality(arguments):
    paths = adapter._input_paths(
        teacher_output_dir=arguments["teacher_output_dir"],
        quality_assembly_receipt=arguments["quality_assembly_receipt"],
        inputs={name: arguments[name] for name in adapter.INPUT_NAMES},
    )
    receipt_path = arguments["quality_assembly_receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for name in (*adapter.INPUT_NAMES, *(f"teacher_output_{key}" for key in adapter.teacher.ARTIFACT_NAMES)):
        receipt["input_sha256"][name] = adapter._sha(paths[name].read_bytes())
    receipt_path.write_bytes(adapter.quality._canonical_json(receipt))
    (receipt_path.parent / adapter.quality.QUALITY_ASSEMBLY_FILES["marker"]).write_bytes(
        adapter.quality._quality_assembly_marker_bytes(
            manifest_content=(receipt_path.parent / adapter.quality.QUALITY_ASSEMBLY_FILES["manifest"]).read_bytes(),
            receipt_content=receipt_path.read_bytes(),
        )
    )


@pytest.mark.parametrize("mutation", [
    lambda row: row.update(capture_timestamp="2026-07-31T00:00:00Z"),
    lambda row: row.update(material=True),
    lambda row: row.update(dent=True),
    lambda row: row.update(foreign_material=False),
    lambda row: row.update(bbox_x1="200.000000"),
    lambda row: row.update(role="validation"),
])
def test_resealed_false_output_does_not_replace_actual_dryrun(case, mutation):
    _, arguments = case
    root = arguments["teacher_output_dir"]
    manifest = root / adapter.teacher.ARTIFACT_NAMES["jsonl"]
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    mutation(rows[0])
    manifest.write_bytes(b"".join(adapter._json_bytes(row) for row in rows))
    lineage_path = root / adapter.teacher.ARTIFACT_NAMES["lineage"]
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["output_digests"]["jsonl_sha256"] = adapter._sha(manifest.read_bytes())
    lineage_path.write_bytes(adapter.quality._canonical_json(lineage))
    _rebind_quality(arguments)
    with pytest.raises(ValueError, match="dry-run output digests"):
        adapter.build_source_evidence(**arguments)
    assert not arguments["output_dir"].exists()


def test_known_protected_source_is_excluded_even_with_rebound_quality_receipt(case):
    _, arguments = case
    queue = json.loads(arguments["teacher_queue"].read_text(encoding="utf-8").splitlines()[0])
    known = arguments["known_audit"]
    known.write_bytes(adapter._json_bytes({queue["sha256"]: {"split": "protected_validation"}}))
    lineage_path = arguments["teacher_output_dir"] / adapter.teacher.ARTIFACT_NAMES["lineage"]
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["inputs"]["known_audit_sha256"] = adapter._sha(known.read_bytes())
    lineage_path.write_bytes(adapter.quality._canonical_json(lineage))
    _rebind_quality(arguments)
    with pytest.raises(ValueError, match="dry-run output digests"):
        adapter.build_source_evidence(**arguments)
    assert not arguments["output_dir"].exists()


def test_publisher_raising_after_exposing_bundle_marks_failure(case, monkeypatch):
    _, arguments = case
    original = adapter.assembler._publish_directory_no_replace
    def raised_after_publish(staging, destination):
        original(staging, destination)
        raise RuntimeError("publication acknowledgement lost")
    monkeypatch.setattr(adapter.assembler, "_publish_directory_no_replace", raised_after_publish)
    with pytest.raises(RuntimeError, match="acknowledgement"):
        adapter.build_source_evidence(**arguments)
    assert (arguments["output_dir"] / "failed.json").exists()
    with pytest.raises(ValueError, match="failure marker"):
        adapter.validate_source_evidence_bundle(arguments["output_dir"])


def test_concurrent_foreign_output_is_untouched(case, monkeypatch):
    _, arguments = case
    def foreign_winner(staging, destination):
        destination.mkdir()
        (destination / "foreign").write_bytes(b"keep")
        raise FileExistsError("foreign publisher won")
    monkeypatch.setattr(adapter.assembler, "_publish_directory_no_replace", foreign_winner)
    with pytest.raises(FileExistsError):
        adapter.build_source_evidence(**arguments)
    assert {path.name: path.read_bytes() for path in arguments["output_dir"].iterdir()} == {"foreign": b"keep"}
