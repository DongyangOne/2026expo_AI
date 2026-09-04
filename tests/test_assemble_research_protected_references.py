"""Real JPEG/rot4 checks, synthetic pinned proof reports; NOT runtime/GT evidence."""
import base64
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import assemble_research_protected_references as assembly


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_fixture(tmp_path):
    """Return kwargs, mutable proof documents and paths/save helper for CPU tests."""
    inputs = tmp_path / "inputs"
    sources = inputs / "sources"
    sources.mkdir(parents=True)
    report_paths = {name: inputs / name / "report.json" for name in assembly.PROOF_NAMES}
    for p in report_paths.values(): p.parent.mkdir()
    support = inputs / "support"
    support.mkdir()
    producer_names = {"audit_proposal_crop_reuse.py", "verifier_preprocessing_contract.py",
        "materialize_protected_reference_crops.py", "observe_protected_proposals.py", "prepare_proposal_verifier_dataset.py"}
    code = {}
    for name in producer_names:
        code[name] = support / name
        code[name].write_bytes(("synthetic proof implementation " + name).encode())
    metadata = {}
    for name in ("model.pt", "spec.json", "known.json", "capture.json", "selection.json", "manifest.csv"):
        metadata[name] = support / name
        metadata[name].write_bytes(("synthetic pinned metadata " + name).encode())
    refs, source_paths = [], []
    for index in range(5):
        rng = np.random.default_rng(400 + index)
        pixels = rng.integers(0, 256, (80, 120, 3), dtype=np.uint8)
        p = sources / f"{index}.jpg"
        assert cv2.imwrite(str(p), pixels)
        source_paths.append(p)
        refs.append({"source_sha256": digest(p), "source_path_b64": assembly.encode(p),
                     "source_bytes": p.stat().st_size, "image_width": 120, "image_height": 80,
                     "roles": ["qx3"] if index < 2 else ["capture", "known_audit"], "source_phash64": "0" * 16})
    def generated_crop(name, index, provenance):
        folder = report_paths[name].parent / "crops"
        folder.mkdir(exist_ok=True)
        p = folder / (refs[index]["source_sha256"] + ".jpg")
        pixels = cv2.resize(cv2.imread(str(source_paths[index])), (320, 320))
        assert cv2.imwrite(str(p), pixels)
        return {"path": "crops/" + p.name, "sha256": digest(p), "bytes": p.stat().st_size,
                "width": 320, "height": 320, "bounds_xyxy": [0, 0, 120, 80], "provenance": provenance}
    reused = generated_crop("reuse", 0, "existing_qx3_replay_crop")
    roi_crop = generated_crop("roi", 2, "known_audit_reference")
    observed_crop = generated_crop("observation", 3, "actual_yolo_runtime_top1")
    fp = {"schema": "protected_image_fingerprint_snapshot.v1", "status": "snapshot_complete",
          "snapshot_only": True, "consumer_must_rehash_sources": True, "metadata_bindings": [],
          "expected_sources": 5, "verified_sources": 5, "missing_sources": 0, "records": refs,
          **{k: False for k in ("training_authorized", "deployment_authorized", "blind_test_authorized", "selection_authorized")}}
    common_false = {k: False for k in ("training_authorized", "deployment_authorized", "blind_test_authorized", "formal_protected_coverage")}
    reuse = {"schema": "proposal_crop_reuse_audit.v1", "status": "reuse_candidates_verified", **common_false,
        "original_sources_rehashed": False, "crop_transform_recomputed": False, "detector_inference_executed": False,
        "selection_sources": 2, "verified_crop_rows": 1, "missing_sources": 1,
        "records": [{"source_sha256": refs[0]["source_sha256"], "source_path": str(source_paths[0]),
            "crop_path": str(report_paths["reuse"].parent / reused["path"]), "crop_sha256": reused["sha256"],
            "crop_bytes": reused["bytes"], "declared_crop_xyxy": reused["bounds_xyxy"], "declared_source_size_wh": [120, 80]}],
        "missing_selection_sources": [{"source_sha256": refs[1]["source_sha256"], "source_path": str(source_paths[1]), "reason": "no_manifest_crop"}],
        "bindings": {"declared_model_sha256": digest(metadata["model.pt"]), "declared_spec_sha256": digest(metadata["spec.json"]),
            "audit_code_sha256": digest(code["audit_proposal_crop_reuse.py"]),
            "crop_root": str(report_paths["reuse"].parent / "crops"),
            "manifest_path": str(metadata["manifest.csv"]), "manifest_sha256": digest(metadata["manifest.csv"]),
            "selection_path": str(metadata["selection.json"]), "selection_sha256": digest(metadata["selection.json"])}}
    roi_rows = []
    for index in (2, 3, 4):
        roi_rows.append({**{k: v for k, v in refs[index].items() if k != "source_phash64"},
            "status": "reference_roi_generated" if index == 2 else "missing_reference", "object_absence_established": False,
            "crop": roi_crop if index == 2 else None,
            "reference": {"kind": "known_audit_reference", "bbox_source": "synthetic_test_reference",
                "bbox_xyxy": [0, 0, 120, 80], "metadata_sha256": digest(metadata["known.json"]), "field": "bbox"} if index == 2 else None})
    roi = {"schema": "protected_reference_roi.v1", "status": "reference_materialization_complete", **common_false,
        **{k: False for k in ("label_authority", "selection_authorized", "semantic_truth_established", "runtime_detector_executed", "state_targets_emitted")},
        "raw_source_count": 3, "reference_roi_count": 1, "missing_reference_count": 2,
        "crop_configuration": assembly.ROI_CONFIG, "records": roi_rows, "bindings": {}}
    observed = []
    for index in (1, 3, 4):
        generated = index == 3
        observed.append({**{k: v for k, v in refs[index].items() if k != "source_phash64"},
            "observation_status": "crop_generated" if generated else "no_eligible_proposal",
            "object_absence_established": False, "returned_proposals_after_model_confidence_nms": int(generated),
            "eligible_proposals": int(generated), "below_confidence_floor": 0,
            "selected_proposal": {"index": 0, "confidence": .7, "bbox_xyxy": [0., 0., 120., 80.],
                                   "predicted_class_id": 0, "predicted_class_name": "can"} if generated else None,
            "crop": observed_crop if generated else None})
    observation = {"schema": "protected_proposal_observation.v1", "status": "observation_complete", **common_false,
        **{k: False for k in ("label_authority", "selection_authorized", "semantic_truth_established")},
        "requested_sources": 3, "observed_sources": 3, "crop_generated": 1, "no_eligible_proposal": 2,
        "runtime": {"runtime_detector_executed": True, "provider_kind": "frozen_yolo_runtime", "requested_configuration": assembly.OBS_CONFIG},
        "bindings": {}, "records": observed}
    documents = dict(fingerprint=fp, reuse=reuse, roi=roi, observation=observation)
    args = {"model_sha256": digest(metadata["model.pt"]), "inference_spec_sha256": digest(metadata["spec.json"]),
            "code_pins": {name: digest(p) for name, p in assembly.code_paths().items()}, "output": tmp_path / "result"}
    def save():
        report_paths["fingerprint"].write_bytes(assembly.rendered(fp))
        for name in ("roi", "observation"):
            names = producer_names - ({"observe_protected_proposals.py", "prepare_proposal_verifier_dataset.py"} if name == "roi" else {"materialize_protected_reference_crops.py"})
            b = documents[name]["bindings"]
            b.update(protected_report_sha256=digest(report_paths["fingerprint"]),
                code_sha256={k: digest(code[k]) for k in names})
            paths = [report_paths["fingerprint"], *(code[k] for k in sorted(names))]
            if name == "roi":
                b.update(known_audit_sha256=digest(metadata["known.json"]), capture_inventory_sha256=digest(metadata["capture.json"]))
                paths += [metadata["known.json"], metadata["capture.json"]]
            else:
                b.update(model_sha256=args["model_sha256"], inference_spec_sha256=args["inference_spec_sha256"])
                paths += [metadata["model.pt"], metadata["spec.json"]]
            b["input_files"] = [{"path_b64": assembly.encode(p), "sha256": digest(p)} for p in paths]
        for name, document in documents.items():
            report_paths[name].write_bytes(assembly.rendered(document))
            args[name + "_report"] = report_paths[name]
            args[name + "_report_sha256"] = digest(report_paths[name])
        return args
    save()
    return args, documents, {"save": save, "sources": source_paths, "reports": report_paths,
                            "metadata": metadata, "code": code, "refs": refs}


@pytest.fixture
def example(tmp_path):
    return make_fixture(tmp_path)


def load(args):
    p = args["output"] / "reference_inventory.json"
    return assembly.load_reference_inventory(p, digest(p), expected_model_sha256=args["model_sha256"], expected_spec_sha256=args["inference_spec_sha256"])


def test_real_bytes_rot4_memberships_and_reader_no_authority(example):
    args, documents, paths = example
    result = assembly.assemble(**args)
    assert result["coverage"] == {"sources": 5, "crops": 3, "observed_crop_absences": 2,
        "all_sources_present": True, "all_available_crops_present": True, "full_source_crop_coverage": False}
    bundle = load(args)
    assert len(bundle.records) == 8
    assert all(r.role == r.cohort == "protected_reference" for r in bundle.records)
    assert {r.source_sha256 for r in bundle.records if r.view_kind == "source"} == {r["source_sha256"] for r in paths["refs"]}
    for source in bundle.inventory["sources"]:
        original = next(r for r in paths["refs"] if r["source_sha256"] == source["source_sha256"])
        assert source["roles"] == original["roles"]
        signature, _, _ = assembly.near._phash_signature(assembly.decode(source["source"]["path_b64"]).read_bytes())
        assert source["source"]["phash_rot4"] == [f"{v:016x}" for v in signature]
        assert (source["crop"] is None) != (source["absence"] is None)
    assert all(result[k] is value for k, value in assembly.AUTHORITY.items())
    assert set(paths["sources"]) <= set(bundle.input_paths)
    assert bundle.binding()["inventory_sha256"] == digest(args["output"] / "reference_inventory.json")
    bundle.recheck()


@pytest.mark.parametrize("fault", ["source_missing", "source_duplicate", "missing_observation", "duplicate_observation",
    "fake_provider", "failed_observation", "fake_absence", "absence_truth", "roi_gt", "roi_bad_bbox", "missing_reference_as_absence",
    "reuse_model", "roi_count_bool", "observation_count_float", "source_roles", "producer_code", "crop_path_traversal"])
def test_proof_boundaries_reject_without_output(example, fault):
    args, docs, paths = example
    if fault == "source_missing": docs["fingerprint"]["records"].pop()
    elif fault == "source_duplicate": docs["fingerprint"]["records"].append(docs["fingerprint"]["records"][0])
    elif fault == "missing_observation": docs["observation"]["records"].pop()
    elif fault == "duplicate_observation": docs["observation"]["records"].append(docs["observation"]["records"][0])
    elif fault == "fake_provider": docs["observation"]["runtime"]["provider_kind"] = "custom_test_provider"
    elif fault == "failed_observation": docs["observation"]["status"] = "failed"
    elif fault == "fake_absence": docs["observation"]["records"][0]["eligible_proposals"] = 1
    elif fault == "absence_truth": docs["observation"]["records"][0]["object_absence_established"] = True
    elif fault == "roi_gt": docs["roi"]["label_authority"] = True
    elif fault == "roi_bad_bbox": docs["roi"]["records"][0]["reference"]["bbox_xyxy"] = [20, 10, 0, 5]
    elif fault == "missing_reference_as_absence": docs["roi"]["records"][1]["status"] = "no_eligible_proposal"
    elif fault == "reuse_model": docs["reuse"]["bindings"]["declared_model_sha256"] = "a" * 64
    elif fault == "roi_count_bool": docs["roi"]["reference_roi_count"] = True
    elif fault == "observation_count_float": docs["observation"]["observed_sources"] = 3.0
    elif fault == "source_roles": docs["observation"]["records"][0]["roles"] = ["capture"]
    elif fault == "crop_path_traversal": docs["roi"]["records"][0]["crop"]["path"] = "../crop.jpg"
    paths["save"]()
    if fault == "producer_code":
        docs["roi"]["bindings"]["code_sha256"]["materialize_protected_reference_crops.py"] = "b" * 64
        paths["reports"]["roi"].write_bytes(assembly.rendered(docs["roi"]))
        args["roi_report_sha256"] = digest(paths["reports"]["roi"])
    with pytest.raises((ValueError, KeyError)):
        assembly.assemble(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("fault", ["source", "crop", "input", "hardlink", "symlink", "failure_marker"])
def test_actual_input_bytes_paths_and_failure_marker(example, tmp_path, fault):
    args, docs, paths = example
    p = paths["sources"][1]
    if fault == "source": p.write_bytes(p.read_bytes() + b"mutated")
    elif fault == "crop": Path(docs["reuse"]["records"][0]["crop_path"]).write_bytes(b"bad crop")
    elif fault == "input": paths["metadata"]["model.pt"].write_bytes(b"changed")
    elif fault == "hardlink": os.link(p, p.with_name("hardlink.jpg"))
    elif fault == "symlink":
        target = p.with_name("moved.jpg"); p.rename(target)
        try: p.symlink_to(target)
        except OSError: pytest.skip("platform lacks symlink permission")
    else: (paths["reports"]["observation"].parent / "failed.json").write_bytes(b"{}")
    with pytest.raises(ValueError): assembly.assemble(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("where", ["metadata", "source", "crop"])
def test_output_cannot_pollute_evidence(example, where):
    args, docs, paths = example
    root = paths["reports"]["roi"].parent if where == "metadata" else paths["sources"][0].parent if where == "source" else Path(docs["reuse"]["records"][0]["crop_path"]).parent
    args["output"] = root / "new"
    with pytest.raises(ValueError): assembly.assemble(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("boundary", ["reference_inventory.json", "report.json"])
def test_late_input_change_does_not_leave_success(example, monkeypatch, boundary):
    args, _, paths = example
    original = assembly._publish
    def mutate(p, content, recheck):
        original(p, content, recheck)
        if p.name == boundary: paths["sources"][1].write_bytes(b"late mutation")
    monkeypatch.setattr(assembly, "_publish", mutate)
    with pytest.raises(ValueError): assembly.assemble(**args)
    assert not (args["output"] / "report.json").exists()
    assert (args["output"] / "failed.json").exists()


@pytest.mark.parametrize("fault", ["inventory_absence", "authority", "report_pin", "post_load_source"])
def test_reader_rebuilds_inventory_and_rechecks(example, fault):
    args, _, paths = example
    assembly.assemble(**args)
    p = args["output"] / "reference_inventory.json"
    if fault == "post_load_source":
        bundle = load(args)
        paths["sources"][1].write_bytes(b"late")
        with pytest.raises(ValueError): bundle.recheck()
        return
    value = json.loads(p.read_bytes())
    if fault == "inventory_absence": value["sources"][0]["absence"] = {"reason": "no_eligible_proposal"}
    elif fault == "authority": value["training_authorized"] = True
    else:
        report_path = args["output"] / "report.json"
        report = json.loads(report_path.read_bytes()); report["inventory_sha256"] = "a" * 64
        report_path.write_bytes(assembly.rendered(report))
    p.write_bytes(assembly.rendered(value))
    with pytest.raises(ValueError): load(args)


def test_v1_formal_loader_still_rejects_research_inventory(example):
    args, _, _ = example
    assembly.assemble(**args)
    with pytest.raises(ValueError, match="fields mismatch|schema mismatch"):
        assembly.near._load_protected_inventory(args["output"] / "reference_inventory.json", cohort_by_sha={}, expected_union=set())


def test_absent_source_still_participates_in_existing_exact_source_phash_graph(example):
    args, _, _ = example
    assembly.assemble(**args)
    bundle = load(args)
    absent = next(row["source_sha256"] for row in bundle.inventory["sources"] if row["absence"] is not None)
    protected = next(record for record in bundle.records if record.source_sha256 == absent)
    assert protected.view_kind == "source"
    candidate = replace(protected, role="train", cohort="candidate", sample_id="synthetic-overlap-probe")
    edges, clusters = assembly.near._graph_evidence([*bundle.records, candidate])
    assert any(edge["blocking"] and {"exact_image_sha256", "source_sha256", "perceptual_hash"} <= set(edge["evidence"]) for edge in edges)
    assert any(cluster["blocking"] and "train" in cluster["roles"] for cluster in clusters)


def test_missing_reuse_path_cannot_be_silently_ignored(example):
    args, docs, paths = example
    docs["reuse"]["missing_selection_sources"][0]["source_path"] = str(paths["sources"][0])
    paths["save"]()
    with pytest.raises(ValueError, match="missing reuse"):
        assembly.assemble(**args)


def test_cli_explicit_proof_and_code_pins(example):
    args, _, _ = example
    cli = [part for key, value in args.items() if key != "code_pins" for part in ("--" + key.replace("_", "-"), str(value))]
    for name, pin in args["code_pins"].items(): cli.extend(["--code-pin", name + "=" + pin])
    assert assembly.main(cli) == 0
    assert load(args).inventory["research_only"] is True


def test_reader_detects_proof_change_during_terminal_image_recheck(example):
    args, _, paths = example
    assembly.assemble(**args)
    bundle = load(args)
    class ChangeAfterImages(tuple):
        def __iter__(self):
            yield from super().__iter__()
            paths["reports"]["observation"].write_bytes(b"changed after images")
    bundle.records = ChangeAfterImages(bundle.records)
    with pytest.raises(ValueError): bundle.recheck()
    assert paths["reports"]["observation"].read_bytes() == b"changed after images"
