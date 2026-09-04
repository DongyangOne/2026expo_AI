"""CPU contract fixtures only: no GPU inference, real training or accuracy claim.

Real JPEGs, the actual protected-reference assembler/reader, lineage upgrader and
research leakage auditor are exercised. Original snapshot and trainer execution
are explicit doubles; the pretrained digest sentinel is not actual weight proof.
"""
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts import assemble_research_protected_references as assembly
from scripts import audit_research_reference_leakage as research
from scripts import audit_v4_near_duplicate_leakage as near
from scripts import audited_aihub_snapshot as original_reader
from scripts import upgrade_proposal_manifest_lineage as lineage
import test_assemble_research_protected_references as protected_fixture
import test_upgrade_proposal_manifest_lineage as legacy_fixture

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/nas/run_aihub_material_research.sh"
PRETRAINED_SHA = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(near._report_bytes(value))


def write_csv(p, values):
    with p.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(values)


def worker_namespace():
    raw = WRAPPER.read_bytes()
    body = raw.split(b"\n: <<'PY_WORKER'\n", 1)[1].rsplit(b"\nPY_WORKER", 1)[0]
    namespace = {"__name__": "cpu_fixture_worker"}
    exec(compile(body, str(WRAPPER), "exec"), namespace)
    return namespace


@pytest.fixture
def case(tmp_path, monkeypatch):
    worker = worker_namespace()
    protected_args, _, proof_paths = protected_fixture.make_fixture(tmp_path / "protected")
    assembly.assemble(**protected_args)
    inventory = protected_args["output"] / "reference_inventory.json"
    data = tmp_path / "candidate"; data.mkdir()
    materializer = data / "materializer.json"
    write_json(materializer, {"explicit_cpu_original_snapshot_double": True, "training_authorized": False})
    metadata, official, raw_rows = {}, {}, []
    for i, split in enumerate(("training", "validation")):
        source, crop = data / f"source{i}.png", data / f"crop{i}.png"
        for p, seed, shape in ((source, 819+i*100, (80, 96, 3)), (crop, 837+i*100, (320, 320, 3))):
            pixels = np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)
            assert cv2.imwrite(str(p), pixels)
        row = legacy_fixture._row(filepath=crop.name, source=source, split=split, source_id=f"fixture-{i}")
        row.update(source_sha256=sha(source), image_sha256=sha(crop), source_filepath=source.as_posix())
        meta = dict(zip(original_reader.METADATA_FIELDS, (
            f"{i+1:020x}", sha(source), str(i+1)*64,
            legacy_fixture._b64(source), legacy_fixture._b64(data / f"original{i}.json"), sha(materializer))))
        row.update(meta); raw_rows.append(row)
        metadata[source] = meta; official[source] = split
    validated = data / "validated.csv"; write_csv(validated, raw_rows)
    original = data / "original.json"
    # A complete audit legitimately contains quarantined entries, not only valid rows.
    write_json(original, {"schema": "aihub_original_annotation_audit_v1", "selected": 3,
        "records": [{"status": "verified"}, {"status": "verified"}, {"status": "quarantined"}],
        "manifest_counts": {"training": 2, "validation": 1}})
    cohort = data / "cohort.json"
    write_json(cohort, {"schema": "aihub_original_cohort_v1", "full_cohort": True,
        "training_authorized": False, "pending_checks": ["independent_hardware_blind"],
        "legacy_exclusion_evidence": {"unresolved_legacy_references": 0},
        "metadata_bindings": [{"path": original.as_posix(), "sha256": sha(original)}]})
    snapshot_binding = dict(report_path=materializer.as_posix(), report_sha256=sha(materializer),
        cohort_path=cohort.as_posix(), cohort_sha256=sha(cohort), require_full_cohort=True)
    info = data / "dataset_info.json"; write_json(info, {"audited_aihub_snapshot": snapshot_binding})
    replay = data / "replay.json"; legacy_fixture._validator_report(replay, validated)
    replay_value = json.loads(replay.read_bytes())
    model, spec = proof_paths["metadata"]["model.pt"], proof_paths["metadata"]["spec.json"]
    replay_value["bindings"].update(detector_model_sha256=sha(model), inference_spec_sha256=sha(spec), dataset_info_sha256=sha(info))
    replay_value["contract"]["proposal_provenance"].update(provider_kind="frozen_yolo_runtime", runtime_detector_executed=True,
        runtime_top1_replayed=True, provided_top1_predictions_matched=True, proposal_class_confidence_bbox_matched=True,
        confidence_abs_tolerance=1e-6, bbox_abs_tolerance=1e-4)
    write_json(replay, replay_value)
    outputs = legacy_fixture._outputs(data)
    lineage.upgrade_proposal_manifests(inputs=[validated], validator_report_paths=[replay], validator_report_sha256s=[sha(replay)],
        quarantine_validation_near_phash_distance=4, origin="aihub_original_cpu_fixture", **outputs)
    combined = legacy_fixture._read_csv(outputs["output_csv"])
    manifests = {}
    for role in sorted(near.CANDIDATE_ROLES):
        p = data / f"{role}.csv"; write_csv(p, [r for r in combined if r["role"] == role]); manifests[role] = p
    audit_output = tmp_path / "separation"
    report = research.audit_research(candidate_manifests=manifests, candidate_manifest_sha256={k: sha(p) for k, p in manifests.items()},
        lineage_manifest=outputs["output_csv"], lineage_manifest_sha256=sha(outputs["output_csv"]),
        lineage_report=outputs["lineage_path"], lineage_report_sha256=sha(outputs["lineage_path"]),
        replay_report=replay, replay_report_sha256=sha(replay), reference_inventory=inventory, reference_inventory_sha256=sha(inventory),
        detector_model=model, detector_model_sha256=sha(model), inference_spec=spec, inference_spec_sha256=sha(spec),
        code_sha256={name: sha(ROOT / "scripts" / name) for name in research.CODE_FILES}, output_dir=audit_output)
    assert report["status"] == "passed"
    cache = tmp_path / "cache"
    pretrained = cache / "hub/checkpoints/mobilenet_v3_small-047dcff4.pth"
    pretrained.parent.mkdir(parents=True); pretrained.write_bytes(b"CPU fixture pretrained sentinel, not model weights")
    license_evidence = data / "license.json"; write_json(license_evidence, {"fixture_only": True, "legal_approval": False})
    files = dict(TRAIN_MANIFEST=manifests["train"], VALIDATION_MANIFEST=manifests["model_validation"],
        LINEAGE_MANIFEST=outputs["output_csv"], ORIGINAL_REPORT=original, COHORT=cohort, MATERIALIZER_REPORT=materializer,
        GENERATION_INFO=info, REPLAY_REPORT=replay, LINEAGE_REPORT=outputs["lineage_path"],
        PROTECTED_AUDIT=audit_output / "report.json", PROTECTED_INVENTORY=inventory,
        DETECTOR_MODEL=model, INFERENCE_SPEC=spec, PRETRAINED=pretrained, LICENSE_EVIDENCE=license_evidence)
    run = tmp_path / "research_run"
    for key, value in dict(CODE_ROOT=ROOT, RUN=run, RAW_SOURCE_ROOT=data, TORCH_HOME=cache,
                          AIHUB_ORIGIN="aihub_original_cpu_fixture", WRAPPER_SHA256=sha(WRAPPER)).items():
        monkeypatch.setenv(key, str(value))
    for key, p in files.items():
        monkeypatch.setenv(key, str(p)); monkeypatch.setenv(key+"_SHA256", PRETRAINED_SHA if key == "PRETRAINED" else sha(p))
    for key, name in worker["CODE"].items(): monkeypatch.setenv(key+"_SHA256", sha(ROOT / "scripts" / name))
    monkeypatch.setattr(worker["sys"], "argv", ["CPU fixture", str(WRAPPER)])
    monkeypatch.setattr(worker["sys"], "path", list(worker["sys"].path))
    events, callbacks = [], {}
    def cuda():
        events.append(("cuda_double", os.getpid())); return object()
    def train(args):
        phase = "dry_run" if "--dry-run" in args else "train"
        events.append((phase, os.getpid(), list(args)))
        if phase == "dry_run": print(json.dumps({"condition_heads": [], "synthetic_fixture_only": True}))
        else:
            output = Path(args[args.index("--output-dir")+1]); output.mkdir()
            (output / "fixture.pt").write_bytes(b"synthetic checkpoint, not trained")
            (output / "fixture.onnx").write_bytes(b"synthetic export, not trained")
            write_json(output / "fixture.json", {"model_config": {"condition_heads": []},
                "candidate_only": True, "production_runtime_modified": False, "synthetic_fixture_only": True})
        if phase in callbacks: callbacks[phase]()
        return 0
    class Snapshot:
        def binding(self): return snapshot_binding
        def metadata_for(self, p): return metadata[p]
        def split_for(self, p): return official[p]
        def recheck(self): events.append(("original_snapshot_recheck_double", os.getpid()))
    def load_snapshot(p, expected, **kwargs):
        assert p == materializer and expected == sha(materializer)
        assert kwargs == dict(cohort_path=cohort, require_full_cohort=True)
        return Snapshot()
    def dataset_audit(paths, **kwargs):
        assert paths == [manifests["train"], manifests["model_validation"]]
        assert kwargs == dict(require_masked_status=True, require_single_object=True, require_source_references=True,
            phash_distance=4, fail_on_near_phash=True)
        return {"ok": True, "explicit_cpu_dataset_audit_double": True}
    doubles = {
        "scripts.train_multitask_verifier": SimpleNamespace(eager_initialize_cuda_context=cuda, main=train,
            METADATA_NAME="fixture.json", CHECKPOINT_NAME="fixture.pt", ONNX_NAME="fixture.onnx"),
        "scripts.audited_aihub_snapshot": SimpleNamespace(load_audited_aihub_snapshot=load_snapshot),
        "scripts.audit_verifier_dataset": SimpleNamespace(audit_manifests=dataset_audit),
    }
    for name, module in doubles.items(): module.__file__ = str(ROOT / (name.replace(".", "/") + ".py"))
    worker["importlib"] = SimpleNamespace(import_module=lambda name: doubles[name] if name in doubles else importlib.import_module(name))
    actual_digest = worker["digest"]
    worker["digest"] = lambda p: PRETRAINED_SHA if Path(p) == pretrained else actual_digest(p)
    def save_audit():
        write_json(files["PROTECTED_AUDIT"], report)
        monkeypatch.setenv("PROTECTED_AUDIT_SHA256", sha(files["PROTECTED_AUDIT"]))
        write_json(audit_output / "research_audit_ready.json", dict(schema=research.SCHEMA, status="passed",
            report_sha256=sha(files["PROTECTED_AUDIT"]), **research.AUTHORITY))
    def repin(key): monkeypatch.setenv(key+"_SHA256", sha(files[key]))
    return SimpleNamespace(worker=worker, run=run, files=files, report=report, save_audit=save_audit,
        repin=repin, events=events, callbacks=callbacks, proof_paths=proof_paths)


def test_draft_cpu_contract_happy_path_preserves_fixed_training_and_no_authority(case):
    case.worker["run_research"]()
    assert not (case.run / "failed.json").exists()
    complete = json.loads((case.run / "research_complete.json").read_bytes())
    assert all(complete[k] is v for k, v in case.worker["FALSE_AUTHORITY"].items())
    assert complete["scope"] == "training completed, not accuracy improvement or deployment approval"
    calls = [e for e in case.events if e[0] in {"cuda_double", "dry_run", "train"}]
    assert [e[0] for e in calls] == ["cuda_double", "dry_run", "train"]
    assert {e[1] for e in calls} == {os.getpid()}
    args = calls[-1][2]
    for name, value in {"--epochs": "100", "--patience": "15", "--batch": "64", "--workers": "2",
                        "--seed": "20260827", "--device": "cuda", "--backbone": "mobilenet_v3_small"}.items():
        assert args[args.index(name)+1] == value
    assert "--no-condition-heads" in args
    running = json.loads((case.run / "running.json").read_bytes())
    assert running["license_evidence"]["sha256"] == sha(case.files["LICENSE_EVIDENCE"])
    assert running["license_evidence"]["license_rights_independently_determined"] is False
    assert running["protected_reference_binding"] == case.report["bindings"]["reference_inventory"]
    assert json.loads(case.files["ORIGINAL_REPORT"].read_bytes())["selected"] == 3


@pytest.mark.parametrize("fault", ["v1", "authority", "bool_authority", "threshold", "code", "manifest",
    "model", "spec", "lineage", "reference", "candidate_payload", "protected_payload", "count_bool", "membership"])
def test_v2_exact_bindings_and_false_authority_required_before_training(case, fault):
    report = case.report
    if fault == "v1": report["schema"] = "v4_near_duplicate_audit.v1"
    elif fault == "authority": report["training_authorized"] = True
    elif fault == "bool_authority": report["formal_protected_coverage"] = 0
    elif fault == "threshold": report["algorithm"]["threshold"] = 5
    elif fault == "code": report["bindings"]["code_sha256"][research.CODE_FILES[0]] = "0"*64
    elif fault == "manifest": report["bindings"]["candidate_manifest_sha256"]["train"] = "0"*64
    elif fault in {"model", "spec", "lineage"}:
        key = {"model": "detector_model", "spec": "inference_spec", "lineage": "lineage_manifest"}[fault]
        report["bindings"]["files"][key]["sha256"] = "0"*64
    elif fault == "reference": report["bindings"]["reference_inventory"]["inventory_sha256"] = "0"*64
    elif fault.endswith("payload"): report["bindings"][fault+"_set_sha256"] = "0"*64
    elif fault == "count_bool": report["coverage"]["formal_protected_coverage"] = 0
    else: report["reference_memberships"][next(iter(report["reference_memberships"]))] = ["capture"]
    case.save_audit()
    with pytest.raises(ValueError): case.worker["run_research"]()
    assert not case.run.exists()
    assert not any(e[0] == "train" for e in case.events)


@pytest.mark.parametrize("marker", ["failed.json", "blocked.json", "wrong_ready"])
def test_failed_blocked_or_unbound_ready_never_train(case, marker):
    parent = case.files["PROTECTED_AUDIT"].parent
    if marker == "wrong_ready":
        p = parent / "research_audit_ready.json"; value = json.loads(p.read_bytes()); value["report_sha256"] = "0"*64; write_json(p, value)
    else: write_json(parent / marker, {})
    with pytest.raises(ValueError): case.worker["run_research"]()
    assert not case.run.exists()


@pytest.mark.parametrize("phase,target", [("dry_run", "candidate"), ("train", "candidate"), ("train", "protected")])
def test_real_asset_drift_during_mock_training_is_failure_not_completion(case, phase, target):
    p = case.files["TRAIN_MANIFEST"].parent / "source0.png" if target == "candidate" else case.proof_paths["sources"][1]
    case.callbacks[phase] = lambda: p.write_bytes(p.read_bytes()+b"late fixture mutation")
    with pytest.raises(ValueError): case.worker["run_research"]()
    assert (case.run / "failed.json").exists()
    assert not (case.run / "research_complete.json").exists()


def test_terminal_publication_drift_is_explicitly_failed(case):
    original = case.worker["mark"]
    def mark(name, values):
        original(name, values)
        if name == "research_complete.json":
            p = case.proof_paths["sources"][1]; p.write_bytes(p.read_bytes()+b"terminal drift")
    case.worker["mark"] = mark
    with pytest.raises(ValueError): case.worker["run_research"]()
    assert (case.run / "failed.json").exists()  # Dominates retained diagnostic completion.


def test_protected_proof_parent_output_rejected_before_mkdir(case, monkeypatch):
    output = case.proof_paths["reports"]["roi"].parent / "never-created"
    monkeypatch.setenv("RUN", str(output))
    with pytest.raises(ValueError, match="overlaps"): case.worker["run_research"]()
    assert not output.exists()


@pytest.mark.parametrize("fault", ["partial", "unresolved", "state", "partition"])
def test_original_scope_and_partition_never_silently_change(case, fault):
    if fault in {"partial", "unresolved"}:
        p = case.files["COHORT"]; value = json.loads(p.read_bytes())
        if fault == "partial": value["full_cohort"] = False
        else: value["legacy_exclusion_evidence"]["unresolved_legacy_references"] = 1
        write_json(p, value); case.repin("COHORT")
    else:
        p = case.files["TRAIN_MANIFEST"]; values = legacy_fixture._read_csv(p)
        values[0]["dent" if fault == "state" else "predicted_confidence"] = "1" if fault == "state" else "0.125"
        write_csv(p, values); case.repin("TRAIN_MANIFEST")
    with pytest.raises(ValueError): case.worker["run_research"]()
    assert not case.run.exists()


def test_shell_bootstrap_and_worker_compile_without_execution():
    raw = WRAPPER.read_bytes()
    assert b"\r" not in raw  # Bootstrap intentionally extracts exact LF delimiters.
    compile(raw.split(b"<<'BOOT'\n", 1)[1].split(b"\nBOOT", 1)[0], str(WRAPPER), "exec")
    worker = worker_namespace()
    assert "PROTECTED_SOURCES" not in worker["INPUTS"]
    assert "DETECTOR_MODEL" in worker["INPUTS"]
    assert {"REFERENCE_READER", "RESEARCH_AUDITOR"} <= set(worker["CODE"])
