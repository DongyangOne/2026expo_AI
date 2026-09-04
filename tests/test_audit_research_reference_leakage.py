"""Research adapter tests; no changes to the formal v1 auditor contract."""
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import audit_research_reference_leakage as audit
from scripts import audit_v4_near_duplicate_leakage as near
from scripts import upgrade_proposal_manifest_lineage as lineage
import test_upgrade_proposal_manifest_lineage as legacy_fixture

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(p, value):
    p.write_bytes(near._report_bytes(value))


def image(p, seed, shape=(80, 96, 3)):
    p.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)
    p.write_bytes(cv2.imencode(".png", pixels)[1].tobytes())
    return p


@pytest.fixture
def candidate(tmp_path, request):
    data = tmp_path / "data"; data.mkdir()
    model, spec = data / "model.pt", data / "spec.json"
    model.write_bytes(b"never loaded: CPU fixture")
    dump(spec, {"fixture": "pinned inference specification"})
    raw_rows = []
    for i, split in enumerate(("training", "validation")):
        source = image(data / "sources" / f"source{i}.png", i * 100 + 17)
        crop = image(data / "crops" / f"crop{i}.png", i * 100 + 29, (320, 320, 3))
        row = legacy_fixture._row(filepath=crop.relative_to(data).as_posix(), source=source,
            split=split, source_id=f"source-{i}")
        row.update(source_sha256=sha(source), image_sha256=sha(crop))
        if getattr(request, "param", "") == "bbox":
            row.update(crop_x1="1.0", crop_y1="2.0", crop_x2="41.0", crop_y2="42.0")
        raw_rows.append(row)
    validated = data / "validated.csv"
    legacy_fixture._write_manifest(validated, raw_rows)
    replay = data / "replay.json"
    legacy_fixture._validator_report(replay, validated)
    value = json.loads(replay.read_bytes())
    value["bindings"].update(detector_model_sha256=sha(model), inference_spec_sha256=sha(spec))
    value["contract"]["proposal_provenance"].update(provider_kind="frozen_yolo_runtime", runtime_detector_executed=True,
        runtime_top1_replayed=True, provided_top1_predictions_matched=True, proposal_class_confidence_bbox_matched=True,
        confidence_abs_tolerance=1e-6, bbox_abs_tolerance=1e-4)
    dump(replay, value)
    outputs = legacy_fixture._outputs(data)
    lineage.upgrade_proposal_manifests(inputs=[validated], validator_report_paths=[replay],
        validator_report_sha256s=[sha(replay)], quarantine_validation_near_phash_distance=4, **outputs)
    combined = legacy_fixture._read_csv(outputs["output_csv"])
    manifests = {}
    for role in sorted(near.CANDIDATE_ROLES):
        p = data / f"{role}.csv"
        values = [row for row in combined if row["role"] == role]
        p.write_bytes(lineage._render_csv(values, list(combined[0])))
        manifests[role] = p
    return dict(candidate_manifests=manifests, candidate_manifest_sha256={k: sha(p) for k, p in manifests.items()},
        lineage_manifest=outputs["output_csv"], lineage_manifest_sha256=sha(outputs["output_csv"]),
        lineage_report=outputs["lineage_path"], lineage_report_sha256=sha(outputs["lineage_path"]),
        replay_report=replay, replay_report_sha256=sha(replay), detector_model=model,
        detector_model_sha256=sha(model), inference_spec=spec, inference_spec_sha256=sha(spec),
        output_dir=tmp_path / "out")


@pytest.fixture
def setup(candidate, tmp_path, monkeypatch):
    """Wire isolation fixture; the separate integration test exercises actual v2 reader."""
    from scripts import assemble_research_protected_references as assembler
    reference_dir = tmp_path / "references"; reference_dir.mkdir()
    source = image(reference_dir / "source.png", 999)
    crop = image(reference_dir / "crop.png", 1000)
    absent = image(reference_dir / "no-eligible-proposal.png", 1001)
    assets = []
    for p, kind, source_p in ((source, "source", source), (crop, "crop", source), (absent, "source", absent)):
        candidate_asset = near.AuditAsset(p, "train", "candidate", kind, f"reference-{p.stem}", sha(source_p), sha(p))
        assets.append(replace(near._verify_audit_asset(candidate_asset, protected=False), role="protected_reference", cohort="protected_reference"))
    inventory = reference_dir / "reference_inventory.json"
    value = dict(schema="research_protected_reference_inventory.v2", research_only=True,
        sources=[dict(source_sha256=sha(source), roles=["qx3", "known_audit"], crop={"path": crop.name}, absence=None),
                 dict(source_sha256=sha(absent), roles=["capture"], crop=None,
                      absence=dict(reason="no_eligible_proposal", source_sha256=sha(absent)))])
    dump(inventory, value)
    class References:
        records = tuple(assets)
        input_paths = (inventory, source, crop, absent)
        def binding(self):
            return dict(inventory_sha256=sha(inventory), model_sha256=candidate["detector_model_sha256"],
                        inference_spec_sha256=candidate["inference_spec_sha256"])
        def recheck(self):
            near.reverify_assets(self.records)
    references = References()
    def load(p, expected_sha256, *, expected_model_sha256, expected_spec_sha256):
        assert p == inventory and expected_sha256 == sha(inventory)
        assert expected_model_sha256 == candidate["detector_model_sha256"] and expected_spec_sha256 == candidate["inference_spec_sha256"]
        return references
    monkeypatch.setattr(assembler, "load_reference_inventory", load)
    args = candidate | dict(reference_inventory=inventory, reference_inventory_sha256=sha(inventory),
        code_sha256={name: sha(ROOT / "scripts" / name) for name in audit.CODE_FILES})
    return dict(args=args, references=references, source=source, crop=crop, absent=absent, inventory=value)


def test_clean_research_only_preserves_memberships_and_missing_crop(setup):
    report = audit.audit_research(**setup["args"])
    assert report["status"] == "passed" and report["ok"] is True
    assert all(report[k] is v for k, v in audit.AUTHORITY.items())
    assert report["coverage"]["protected_sources"] == 2 and report["coverage"]["protected_crops"] == 1
    assert report["coverage"]["candidate_assets"] == 4
    assert report["reference_memberships"][sha(setup["source"])] == ["qx3", "known_audit"]
    assert report["summary"]["protected_internal_edges_nonblocking"] == 1
    assert (setup["args"]["output_dir"] / "research_audit_ready.json").exists()
    assert near.PROTECTED_INVENTORY_SCHEMA == "v4_near_duplicate_protected_inventory.v1"
    assert near.PHASH_DISTANCE == 4


@pytest.mark.parametrize("target", ["absent", "crop"])
def test_missing_proposal_source_and_historical_crop_each_block_candidates(setup, monkeypatch, target):
    args = setup["args"]
    candidate_path = args["candidate_manifests"]["train"].parent / "sources/source0.png"
    # Copying a protected payload into an actual candidate requires rebuilding
    # lineage pins; changing candidate records alone may not bypass that chain.
    records = list(setup["references"].records)
    candidate_sha = sha(candidate_path)
    record = next(r for r in records if r.path == setup[target])
    replacement = near._verify_audit_asset(near.AuditAsset(candidate_path, "train", "candidate", record.view_kind,
        record.sample_id, candidate_sha if target == "absent" else record.source_sha256, candidate_sha), protected=False)
    records[records.index(record)] = replace(replacement, role="protected_reference", cohort="protected_reference")
    setup["references"].records = tuple(records)
    if target == "absent":
        inventory = setup["inventory"]
        inventory["sources"][1]["source_sha256"] = candidate_sha
        dump(args["reference_inventory"], inventory)
        args["reference_inventory_sha256"] = sha(args["reference_inventory"])
    report = audit.audit_research(**args)
    assert report["status"] == "blocked" and report["summary"]["blocking_multi_role_clusters"] >= 1
    assert any("exact_image_sha256" in e["evidence"] and e["blocking"] for e in report["edges"])
    assert (args["output_dir"] / "blocked.json").exists()
    assert not (args["output_dir"] / "research_audit_ready.json").exists()


def test_protected_membership_edges_nonblocking_but_transitive_candidate_path_blocks(setup):
    actual = setup["references"].records[0]
    records = [replace(actual, sample_id="a", source_sha256="a"*64, image_sha256="a"*64, signature=(0,)*4),
               replace(actual, sample_id="b", source_sha256="b"*64, image_sha256="b"*64, signature=(15,)*4),
               replace(actual, sample_id="c", role="train", cohort="candidate", source_sha256="c"*64,
                       image_sha256="c"*64, signature=(255,)*4)]
    edges, clusters = near._graph_evidence(records)
    assert len(edges) == 2 and sum(e["blocking"] for e in edges) == 1
    assert len(clusters) == 1 and clusters[0]["blocking"] is True
    protected_edges, protected_clusters = near._graph_evidence(records[:2])
    assert protected_edges[0]["blocking"] is False and protected_clusters[0]["blocking"] is False


@pytest.mark.parametrize("name", ["lineage_manifest", "lineage_report", "replay_report", "reference_inventory", "detector_model", "inference_spec"])
def test_exact_input_pins_required(setup, name):
    setup["args"][name + "_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA"):
        audit.audit_research(**setup["args"])
    assert not setup["args"]["output_dir"].exists()


@pytest.mark.parametrize("change", ["row", "bool", "diagnostic", "tolerance"])
def test_lineage_and_replay_shape_mutations_fail_closed(setup, change):
    args = setup["args"]
    if change == "row":
        p = args["candidate_manifests"]["train"]
        p.write_bytes(p.read_bytes().replace(b"preserve-me", b"changed-data"))
        args["candidate_manifest_sha256"]["train"] = sha(p)
    else:
        p = args["replay_report"]; value = json.loads(p.read_bytes())
        if change == "bool": value["schema_version"] = True
        if change == "diagnostic": value["contract"]["proposal_provenance"]["provider_kind"] = "custom_non_authoritative"
        if change == "tolerance": value["contract"]["proposal_provenance"]["confidence_abs_tolerance"] = .1
        dump(p, value); args["replay_report_sha256"] = sha(p)
        receipt = json.loads(args["lineage_report"].read_bytes())
        receipt["validator_reports"][0]["sha256"] = sha(p)
        dump(args["lineage_report"], receipt); args["lineage_report_sha256"] = sha(args["lineage_report"])
    with pytest.raises(ValueError): audit.audit_research(**args)


@pytest.mark.parametrize("field,value", [("predicted_confidence", "0.125"), ("material", "1"),
    ("crop_x1", "9"), ("source_object_count", "0"), ("dent", "1"),
    ("image_sha256", "f" * 64), ("custom_note", "fabricated-evidence")])
def test_lineage_and_partition_cannot_rewrite_pinned_replay_values(setup, field, value):
    args = setup["args"]
    before = sha(args["replay_report"])
    for p in (args["lineage_manifest"], args["candidate_manifests"]["train"]):
        values = legacy_fixture._read_csv(p)
        target = next(row for row in values if row["role"] == "train")
        target[field] = value
        p.write_bytes(lineage._render_csv(values, list(values[0])))
    args["lineage_manifest_sha256"] = sha(args["lineage_manifest"])
    args["candidate_manifest_sha256"]["train"] = sha(args["candidate_manifests"]["train"])
    receipt = json.loads(args["lineage_report"].read_bytes())
    receipt["outputs"]["csv"]["sha256"] = args["lineage_manifest_sha256"]
    dump(args["lineage_report"], receipt); args["lineage_report_sha256"] = sha(args["lineage_report"])
    with pytest.raises(ValueError, match="replay|bbox"):
        audit.audit_research(**args)
    assert sha(args["replay_report"]) == before
    assert not args["output_dir"].exists()


@pytest.mark.parametrize("candidate", ["bbox"], indirect=True)
def test_existing_lineage_bbox_derivation_and_path_normalization_remain_valid(setup):
    row = legacy_fixture._read_csv(setup["args"]["lineage_manifest"])[0]
    assert row["source_bbox_x"] == "1" and row["source_bbox_w"] == "40"
    assert audit.audit_research(**setup["args"])["ok"] is True


def test_frozen_replay_source_crop_hashes_must_not_be_empty(setup):
    args = setup["args"]
    receipt = json.loads(args["lineage_report"].read_bytes())
    p = Path(receipt["inputs"][0]["path"])
    values = legacy_fixture._read_csv(p); values[0]["source_sha256"] = ""
    p.write_bytes(lineage._render_csv(values, list(values[0])))
    receipt["inputs"][0]["sha256"] = sha(p)
    replay = json.loads(args["replay_report"].read_bytes())
    replay["bindings"]["validated_manifest_sha256"] = sha(p)
    dump(args["replay_report"], replay); args["replay_report_sha256"] = sha(args["replay_report"])
    receipt["validator_reports"][0]["sha256"] = args["replay_report_sha256"]
    dump(args["lineage_report"], receipt); args["lineage_report_sha256"] = sha(args["lineage_report"])
    with pytest.raises(ValueError, match="replay source SHA"):
        audit.audit_research(**args)


@pytest.mark.parametrize("target", ["reference_inventory", "detector_model", "source", "crop", "absent"])
def test_ready_boundary_input_drift_removes_only_own_ready_and_keeps_failure(setup, monkeypatch, target):
    original = audit._publish
    def publish(p, value, publications):
        original(p, value, publications)
        if p.name == "research_audit_ready.json":
            mutation = setup["args"].get(target, setup.get(target))
            with mutation.open("ab") as stream: stream.write(b"changed")
    monkeypatch.setattr(audit, "_publish", publish)
    with pytest.raises(ValueError): audit.audit_research(**setup["args"])
    out = setup["args"]["output_dir"]
    assert not (out / "research_audit_ready.json").exists()
    assert json.loads((out / "failed.json").read_bytes())["status"] == "failed"
    assert (out / "report.json").exists()


def test_foreign_ready_is_preserved_and_failed_dominates(setup, monkeypatch):
    original = audit._publish
    def publish(p, value, publications):
        if p.name == "research_audit_ready.json":
            p.write_bytes(b"foreign marker")
        original(p, value, publications)
    monkeypatch.setattr(audit, "_publish", publish)
    with pytest.raises(FileExistsError): audit.audit_research(**setup["args"])
    out = setup["args"]["output_dir"]
    assert (out / "research_audit_ready.json").read_bytes() == b"foreign marker"
    assert (out / "failed.json").exists()


@pytest.mark.parametrize("name", ["lineage_report", "reference_inventory", "detector_model"])
def test_output_must_not_enter_input_tree(setup, name):
    setup["args"]["output_dir"] = setup["args"][name].parent / "new-output"
    with pytest.raises(ValueError, match="overlaps"):
        audit.audit_research(**setup["args"])
    assert not setup["args"]["output_dir"].exists()


def test_v1_loader_rejects_v2_reference_inventory(setup):
    with pytest.raises(ValueError, match="schema|top-level fields"):
        near._load_protected_inventory(setup["args"]["reference_inventory"], cohort_by_sha={}, expected_union=set())


def test_exact_code_catalog_and_pins_required(setup):
    setup["args"]["code_sha256"].pop(audit.CODE_FILES[0])
    with pytest.raises(ValueError, match="code pins"):
        audit.audit_research(**setup["args"])


def test_rejects_symlink_output_ancestor(setup, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(setup["args"]["lineage_report"].parent, target_is_directory=True)
    setup["args"]["output_dir"] = link / "new-output"
    with pytest.raises(ValueError, match="symlink"):
        audit.audit_research(**setup["args"])


def test_formal_default_cap_and_absolute_path_policy_unchanged(setup):
    assert near.MAX_ENCODED_BYTES == 64 * 1024**2
    assert inspect.signature(near._load_candidate_manifest).parameters["max_bytes"].default == near.MAX_ENCODED_BYTES
    assert inspect.signature(near._load_candidate_manifest).parameters["allow_absolute_crop_paths"].default is False
    p = setup["args"]["candidate_manifests"]["train"]
    assert Path(legacy_fixture._read_csv(p)[0]["filepath"]).is_absolute()
    # Frozen v1 parses POSIX inventory paths. On Windows a drive-qualified path
    # inside the manifest root had already been accepted; do not change that
    # historical behavior or pretend this new option retroactively rejects it.
    posix = p.with_name("posix-absolute.csv")
    values = legacy_fixture._read_csv(p)
    values[0]["filepath"] = "/app/absolute/crop.png"
    posix.write_bytes(lineage._render_csv(values, list(values[0])))
    with pytest.raises(ValueError, match="relative"):
        near._load_candidate_manifest("train", posix)
    with pytest.raises(ValueError, match="cap|large|limit|exceed"):
        near._load_candidate_manifest("train", p, max_bytes=p.stat().st_size - 1)
    assets, _ = near._load_candidate_manifest("train", p, allow_absolute_crop_paths=True)
    assert len(assets) == 2
    result = audit.audit_research(**setup["args"])
    assert result["ok"] is True


def test_large_valid_csv_requires_explicit_larger_read_limit(setup):
    p = setup["args"]["candidate_manifests"]["train"]
    with p.open("ab") as stream:
        stream.write(b"\n" * (near.MAX_ENCODED_BYTES + 1))
    with pytest.raises(ValueError, match="cap|large|limit|exceed"):
        near._load_candidate_manifest("train", p)
    assets, actual = near._load_candidate_manifest("train", p, max_bytes=audit.MAX_METADATA_BYTES, allow_absolute_crop_paths=True)
    assert len(assets) == 2 and actual == sha(p)
    setup["args"]["candidate_manifest_sha256"]["train"] = actual
    assert audit.audit_research(**setup["args"])["ok"] is True


def test_research_metadata_cap_is_fail_closed(setup, monkeypatch):
    # A smaller test-only local limit exercises the exact oversize branch;
    # the unchanged v1 image/parser cap is not modified.
    monkeypatch.setattr(audit, "MAX_METADATA_BYTES", 1)
    with pytest.raises(ValueError, match="cap|large|limit|exceed"):
        audit.audit_research(**setup["args"])


@pytest.fixture
def integrated(candidate, tmp_path):
    from scripts import assemble_research_protected_references as assembler
    from test_assemble_research_protected_references import make_fixture
    reference_args, documents, paths = make_fixture(tmp_path / "protected-fixture")
    model, spec = paths["metadata"]["model.pt"], paths["metadata"]["spec.json"]
    candidate.update(detector_model=model, detector_model_sha256=sha(model), inference_spec=spec,
                     inference_spec_sha256=sha(spec))
    replay = json.loads(candidate["replay_report"].read_bytes())
    replay["bindings"].update(detector_model_sha256=sha(model), inference_spec_sha256=sha(spec))
    dump(candidate["replay_report"], replay); candidate["replay_report_sha256"] = sha(candidate["replay_report"])
    receipt = json.loads(candidate["lineage_report"].read_bytes())
    receipt["validator_reports"][0]["sha256"] = candidate["replay_report_sha256"]
    dump(candidate["lineage_report"], receipt); candidate["lineage_report_sha256"] = sha(candidate["lineage_report"])
    def assemble():
        paths["save"]()
        assembler.assemble(**reference_args)
        inventory = reference_args["output"] / "reference_inventory.json"
        return candidate | dict(reference_inventory=inventory, reference_inventory_sha256=sha(inventory),
            code_sha256={name: sha(ROOT / "scripts" / name) for name in audit.CODE_FILES})
    return candidate, reference_args, documents, paths, assemble


def test_real_v2_loader_actual_assets_and_lineage_integration(integrated):
    *_, assemble = integrated
    args = assemble()
    report = audit.audit_research(**args)
    assert report["status"] == "passed" and report["coverage"]["protected_sources"] == 5
    assert report["coverage"]["protected_crops"] == 3
    assert report["bindings"]["reference_inventory"]["inventory_sha256"] == args["reference_inventory_sha256"]
    assert all(report[k] is v for k, v in audit.AUTHORITY.items())
    with pytest.raises(ValueError):
        near._load_protected_inventory(args["reference_inventory"], cohort_by_sha={}, expected_union=set())


@pytest.mark.parametrize("target", ["absent_source", "reference_crop"])
def test_real_reference_loader_conflicts_block_without_fake_crops(integrated, target):
    candidate, _, docs, paths, assemble = integrated
    if target == "reference_crop":
        row = docs["reuse"]["records"][0]
        crop = Path(row["crop_path"])
        crop.write_bytes((candidate["candidate_manifests"]["train"].parent / "crops/crop0.png").read_bytes())
        row.update(crop_sha256=sha(crop), crop_bytes=crop.stat().st_size)
    else:
        source = paths["sources"][1]
        old_sha = sha(source)
        source.write_bytes((candidate["candidate_manifests"]["train"].parent / "sources/source0.png").read_bytes())
        for document in docs.values():
            for row in document.get("records", []) + document.get("missing_selection_sources", []):
                if row["source_sha256"] == old_sha:
                    row["source_sha256"] = sha(source)
                    if "source_bytes" in row:
                        row.update(source_bytes=source.stat().st_size, image_width=96, image_height=80)
    report = audit.audit_research(**assemble())
    assert report["status"] == "blocked" and report["summary"]["blocking_multi_role_clusters"] > 0
    assert report["coverage"]["protected_crops"] == 3


@pytest.mark.parametrize("fault", ["omitted_source", "count_bool", "formal_schema", "assembly_bool"])
def test_real_reference_shape_or_inventory_tampering_is_not_accepted(integrated, fault):
    *_, assemble = integrated
    args = assemble()
    inventory = json.loads(args["reference_inventory"].read_bytes())
    if fault == "omitted_source": inventory["sources"].pop()
    if fault == "count_bool": inventory["coverage"]["sources"] = True
    if fault == "formal_schema": inventory["schema"] = near.PROTECTED_INVENTORY_SCHEMA
    if fault != "assembly_bool":
        dump(args["reference_inventory"], inventory)
        args["reference_inventory_sha256"] = sha(args["reference_inventory"])
    else:
        p = args["reference_inventory"].parent / "report.json"
        value = json.loads(p.read_bytes()); value["research_only"] = 1; dump(p, value)
    with pytest.raises(ValueError): audit.audit_research(**args)
    assert not args["output_dir"].exists()
