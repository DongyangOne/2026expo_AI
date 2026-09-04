"""Research-only separation evidence; never a formal v1 holdout audit or approval.

Uses the unchanged v1 auditor's image verification and exact/rot4-pHash graph.
All historical reference memberships share one protected boundary; connections
within that boundary are retained, not mistaken for candidate leakage.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_v4_near_duplicate_leakage as near
from scripts import upgrade_proposal_manifest_lineage as lineage

SCHEMA = "research_reference_leakage_audit.v2"
AUTHORITY = dict(research_only=True, formal_protected_coverage=False,
                 training_authorized=False, blind_test_authorized=False,
                 promotion_authorized=False, deployment_authorized=False,
                 automatic_delete_or_relabel=False)
CODE_FILES = ("audit_research_reference_leakage.py", "assemble_research_protected_references.py",
              "audit_v4_near_duplicate_leakage.py", "upgrade_proposal_manifest_lineage.py")
MAX_METADATA_BYTES = 512 * 1024**2


def require(condition, message):
    if not condition:
        raise near.AuditError(message)


def path(value, *, missing=False):
    value = Path(value)
    require(value.is_absolute() and ".." not in value.parts, "absolute traversal-free path required")
    current = value
    if missing:
        while not current.exists() and not current.is_symlink():
            current = current.parent
    near._reject_symlink_chain(current)
    return value


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def nonfinite(_):
        raise near.AuditError("non-finite JSON number")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def rows(raw):
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    fields = reader.fieldnames or []
    require(fields and len(fields) == len(set(fields)), "invalid CSV header")
    result = list(reader)
    require(result and all(None not in row and all(v is not None for v in row.values()) for row in result), "invalid or empty CSV")
    return result


def _row_key(row, parent):
    value = dict(row)
    value["filepath"] = path(parent / value["filepath"]).as_posix()
    return near._canonical_bytes(value)


def _identity(p):
    state = path(p).stat()
    return state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns


def _verify_retained_replay_rows(replay_rows, combined, replay_path, combined_path, receipt):
    """Only lineage's documented representation changes may alter replay rows."""
    rewritten = (set(lineage.CANONICAL_FIELDS) - {"source_sha256", "image_sha256"}) | {"filepath", "source_path_b64", "split"}
    bbox = {"source_bbox_x", "source_bbox_y", "source_bbox_w", "source_bbox_h"}
    fields = (set(replay_rows[0]) - rewritten) | bbox
    allowed_new = rewritten | bbox | {"legacy_" + name for name in rewritten}
    require(not set(combined[0]) - set(replay_rows[0]) - allowed_new, "lineage adds an unsupported replay field")
    remaps = [(item["from"], item["to"]) for item in receipt.get("path_remaps", [])]
    lineage._validate_path_remaps(remaps)
    def fingerprint(row, manifest, *, replay):
        row = {k: v.strip() for k, v in row.items()}
        split, role = lineage._normalize_role(row, location="retained replay row")
        source_sha = near._require_sha256(row.get("source_sha256"), "replay source SHA")
        image_sha = near._require_sha256(row.get("image_sha256"), "replay crop SHA")
        source = lineage._resolve_existing_path(lineage._decode_source_path(row["source_path_b64"]),
            manifest_dir=manifest.parent, remaps=remaps if replay else (), kind="source")
        crop = lineage._resolve_existing_path(row["filepath"], manifest_dir=manifest.parent,
            remaps=remaps if replay else (), kind="crop")
        derived = lineage._source_reference_bbox(row, location="retained replay bbox")
        values = {k: row.get(k, "") or derived.get(k, "") for k in fields}
        values.update(source_path=path(source).as_posix(), crop_path=path(crop).as_posix(),
            source_sha256=source_sha, image_sha256=image_sha, split=split, role=role, fold=row.get("fold") or role)
        return (role, near._sha256_bytes(near._canonical_bytes(values)))
    original = Counter(fingerprint(row, replay_path, replay=True) for row in replay_rows)
    retained = Counter(fingerprint(row, combined_path, replay=False) for row in combined)
    require(all(count <= original[key] for key, count in retained.items()),
            "retained lineage row changed replay proposal/GT/state/source evidence")
    require(all(count == 1 for count in retained.values()), "lineage retained duplicate replay rows")
    missing = set(original) - set(retained)
    require(all(role == "model_validation" for role, _ in missing), "lineage discarded a unique training replay row")
    require(type(receipt.get("duplicates_removed")) is int
            and receipt["duplicates_removed"] == len(replay_rows) - len(original), "lineage duplicate count mismatch")
    quarantine = receipt.get("near_phash_quarantine", {})
    require(type(quarantine.get("validation_rows_removed")) is int
            and quarantine["validation_rows_removed"] == len(missing)
            and type(quarantine.get("training_rows_removed")) is int and quarantine["training_rows_removed"] == 0,
            "lineage excluded row count mismatch")
    require(not missing or (quarantine.get("enabled") is True and quarantine.get("distance") == 4),
            "missing validation rows lack declared existing quarantine")


def _verify_candidate_chain(files, payloads, pins):
    combined = rows(payloads["lineage_manifest"])
    partitions = {role: rows(payloads[role]) for role in sorted(near.CANDIDATE_ROLES)}
    require(Counter(_row_key(row, files["lineage_manifest"].parent) for row in combined) ==
            Counter(_row_key(row, files[role].parent) for role, items in partitions.items() for row in items),
            "candidate role partitions differ from lineage rows")
    for role, items in partitions.items():
        require(all(row.get("role") == row.get("fold") == role for row in items), "candidate role/fold mismatch")
    receipt = parse(payloads["lineage_report"])
    require(type(receipt.get("schema_version")) is int and receipt["schema_version"] == 1
            and receipt.get("builder") == "scripts/upgrade_proposal_manifest_lineage.py"
            and receipt.get("dry_run") is False and receipt.get("blind_test_eligible") is False,
            "completed non-authorizing lineage report required")
    require(type(receipt.get("rows")) is int and receipt["rows"] == len(combined), "lineage row count mismatch")
    require(receipt.get("role_counts") == {k: len(v) for k, v in partitions.items()}, "lineage role counts mismatch")
    require(receipt.get("outputs", {}).get("csv") == dict(path=files["lineage_manifest"].as_posix(),
            sha256=pins[files["lineage_manifest"]]), "lineage manifest binding mismatch")
    require(len(receipt.get("inputs", [])) == len(receipt.get("validator_reports", [])) == 1,
            "one strict replay input required")
    input_entry, report_entry = receipt["inputs"][0], receipt["validator_reports"][0]
    require(report_entry.get("path") == files["replay_report"].as_posix()
            and report_entry.get("sha256") == pins[files["replay_report"]], "lineage replay report binding mismatch")
    replay_path = path(input_entry["path"])
    raw, actual, _ = near._read_regular_file(replay_path, expected_sha256=near._require_sha256(input_entry["sha256"], "replay manifest"), max_bytes=MAX_METADATA_BYTES)
    pins[replay_path] = actual
    replay_rows = rows(raw)
    replay, _ = lineage._load_validator_report(files["replay_report"],
        expected_report_sha256=pins[files["replay_report"]], validated_manifest_sha256=actual,
        validated_manifest_rows=len(replay_rows))
    require(parse(payloads["replay_report"]) == replay, "replay report reread mismatch")
    require(type(replay.get("schema_version")) is int and type(replay.get("rows")) is int, "strict replay integer types required")
    for name in ("ready_for_lineage_upgrade", "blind_test_eligible", "production_deployment_authorized"):
        require(type(replay.get(name)) is bool, "strict replay boolean types required")
    provenance = replay["contract"]["proposal_provenance"]
    expected = dict(provider_kind="frozen_yolo_runtime", runtime_detector_executed=True,
        runtime_top1_replayed=True, provided_top1_predictions_matched=True,
        proposal_class_confidence_bbox_matched=True, confidence_abs_tolerance=1e-6, bbox_abs_tolerance=1e-4,
        production_or_blind_authority=False, detector_artifact_bytes_bound=True,
        inference_spec_bytes_bound=True, dataset_info_bytes_bound=True, source_bbox_crop_bytes_recomputed=True)
    require(all(type(provenance.get(k)) is type(v) and provenance[k] == v for k, v in expected.items()),
            "strict runtime replay evidence required")
    require(replay["bindings"]["detector_model_sha256"] == pins[files["detector_model"]]
            and replay["bindings"]["inference_spec_sha256"] == pins[files["inference_spec"]],
            "replay model/spec binding mismatch")
    _verify_retained_replay_rows(replay_rows, combined, replay_path, files["lineage_manifest"], receipt)
    return {role: len(items) for role, items in partitions.items()}


def _publish(output, value, publications):
    payload = near._report_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".publishing-", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
            state = os.fstat(stream.fileno())
        publications[output] = (state.st_dev, state.st_ino, near._sha256_bytes(payload))
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def audit_research(*, candidate_manifests, candidate_manifest_sha256, reference_inventory,
                   reference_inventory_sha256, lineage_manifest, lineage_manifest_sha256,
                   lineage_report, lineage_report_sha256, replay_report, replay_report_sha256,
                   detector_model, detector_model_sha256, inference_spec, inference_spec_sha256,
                   code_sha256, output_dir):
    """Return a passed/blocked research report; failures retain failed.json, never ready."""
    require(set(candidate_manifests) == set(candidate_manifest_sha256) == near.CANDIDATE_ROLES,
            "exact train/model_validation manifest pair required")
    require(set(code_sha256) == set(CODE_FILES), "exact auditor/reader/shared/lineage code pins required")
    directory = path(output_dir, missing=True)
    require(not directory.exists(), "fresh output directory required")
    files = {role: path(p) for role, p in candidate_manifests.items()}
    files.update(reference_inventory=path(reference_inventory), lineage_manifest=path(lineage_manifest),
                 lineage_report=path(lineage_report), replay_report=path(replay_report),
                 detector_model=path(detector_model), inference_spec=path(inference_spec))
    expected = dict(candidate_manifest_sha256, reference_inventory=reference_inventory_sha256,
        lineage_manifest=lineage_manifest_sha256, lineage_report=lineage_report_sha256,
        replay_report=replay_report_sha256, detector_model=detector_model_sha256,
        inference_spec=inference_spec_sha256)
    pins, payloads, publications = {}, {}, {}
    owned = None
    stage = "inputs"
    try:
        code_root = path(Path(__file__).absolute().parent)
        for name in CODE_FILES:
            p = path(code_root / name)
            _, actual, _ = near._read_regular_file(p, expected_sha256=near._require_sha256(code_sha256[name], "code SHA"))
            pins[p] = actual
        for name, p in files.items():
            raw, actual, _ = near._read_regular_file(p, expected_sha256=near._require_sha256(expected[name], name), max_bytes=MAX_METADATA_BYTES)
            pins[p] = actual
            if name != "detector_model":
                payloads[name] = raw
        references_module = importlib.import_module("scripts.assemble_research_protected_references")
        for module in (near, lineage, references_module):
            require(path(Path(module.__file__).absolute()).parent == code_root, "import escaped pinned code root")
        partition_counts = _verify_candidate_chain(files, payloads, pins)
        references = references_module.load_reference_inventory(files["reference_inventory"], reference_inventory_sha256,
            expected_model_sha256=detector_model_sha256, expected_spec_sha256=inference_spec_sha256)
        inventory = parse(payloads["reference_inventory"])
        payloads.clear()  # Keep only row counts after the complete chain check.
        memberships = {row["source_sha256"]: row["roles"] for row in inventory["sources"]}
        protected = tuple(references.records)
        require(protected and all(r.role == r.cohort == "protected_reference" for r in protected), "invalid research reference roles")
        require({r.source_sha256 for r in protected} == set(memberships), "reference coverage mismatch")
        candidate_assets = []
        for role in sorted(near.CANDIDATE_ROLES):
            assets, actual = near._load_candidate_manifest(role, files[role], max_bytes=MAX_METADATA_BYTES,
                                                          allow_absolute_crop_paths=True)
            require(actual == expected[role], "candidate manifest changed during load")
            candidate_assets.extend(assets)
        counts = Counter((r.role, r.source_sha256, r.view_kind) for r in candidate_assets)
        keys = {(r.role, r.source_sha256) for r in candidate_assets}
        require(all(counts[(role, sha, kind)] == 1 for role, sha in keys for kind in near.VIEW_KINDS),
                "one candidate source and crop per role/source required")
        inputs = [path(p) for p in (*pins, *references.input_paths, *(r.path for r in candidate_assets), *(r.path for r in protected))]
        require(all(not directory.is_relative_to(p.parent) and not p.is_relative_to(directory) for p in inputs),
                "output overlaps an input tree")
        require(not any((p.parent / "failed.json").exists() for p in inputs), "input failure marker present")
        input_identities = {p: _identity(p) for p in inputs}
        directory.mkdir(parents=True, exist_ok=False)
        owned = (directory.stat().st_dev, directory.stat().st_ino)
        stage = "image_verification"
        candidate = tuple(near._verify_audit_asset(asset, protected=False) for asset in candidate_assets)
        records = candidate + protected
        require(len({near._asset_id(r) for r in records}) == len(records), "duplicate asset identities")

        def recheck():
            path(directory)
            require((directory.stat().st_dev, directory.stat().st_ino) == owned, "output ownership changed")
            require(not any((p.parent / "failed.json").exists() for p in inputs), "input failure marker appeared")
            for p, sha in pins.items():
                near._read_regular_file(p, expected_sha256=sha, max_bytes=MAX_METADATA_BYTES)
            references.recheck()
            near.reverify_assets(candidate)
            for p, (device, inode, sha) in publications.items():
                _, _, current = near._read_regular_file(p, expected_sha256=sha, max_bytes=MAX_METADATA_BYTES)
                require((current.device, current.inode) == (device, inode), "publication ownership changed")
            # Image scans can be long. Check every consumed metadata/code/model
            # and asset identity again after them without decoding images twice.
            require(all(_identity(p) == before for p, before in input_identities.items()),
                    "input changed during terminal image verification")
            require(not any((p.parent / "failed.json").exists() for p in inputs)
                    and not (directory / "failed.json").exists(), "failure marker appeared during verification")
            require((path(directory).stat().st_dev, directory.stat().st_ino) == owned, "output ownership changed")

        stage = "graph"
        edges, clusters = near._graph_evidence(records)
        blocking = [c for c in clusters if c["blocking"]]
        protected_ids = {near._asset_id(r) for r in protected}
        report = dict(schema=SCHEMA, status="blocked" if blocking else "passed", ok=not blocking,
            artifact_role="research_dataset_separation_evidence_only", **AUTHORITY,
            algorithm=dict(id=near.ALGORITHM_ID, threshold=near.PHASH_DISTANCE,
                           shared_auditor=near._auditor_binding()),
            bindings=dict(candidate_manifest_sha256=dict(candidate_manifest_sha256),
                reference_inventory=references.binding(),
                candidate_payload_set_sha256=near._payload_set_sha(candidate),
                protected_payload_set_sha256=near._payload_set_sha(protected),
                files={k: dict(path=p.as_posix(), sha256=pins[p]) for k, p in files.items()},
                code_sha256=dict(code_sha256)),
            coverage=dict(candidate_assets=len(candidate), protected_assets=len(protected),
                protected_sources=len(memberships), protected_crops=sum(r.view_kind == "crop" for r in protected),
                candidate_rows=partition_counts,
                supplied_reference_inventory_complete=True, formal_protected_coverage=False),
            summary=dict(edges=len(edges), clusters=len(clusters), blocking_multi_role_clusters=len(blocking),
                protected_internal_edges_nonblocking=sum(not e["blocking"] and e["left_asset_id"] in
                    protected_ids for e in edges)),
            reference_memberships=memberships,
            entries=sorted((near._entry(r) for r in records), key=lambda r: r["asset_id"]), edges=edges, clusters=clusters)
        recheck()
        stage = "publication"
        _publish(directory / "report.json", report, publications)
        recheck()
        marker = "research_audit_ready.json" if not blocking else "blocked.json"
        _publish(directory / marker, dict(schema=SCHEMA, status=report["status"],
            report_sha256=publications[directory / "report.json"][2], **AUTHORITY), publications)
        recheck()
        return report
    except BaseException as error:
        if owned is not None and directory.exists() and not directory.is_symlink() and (directory.stat().st_dev, directory.stat().st_ino) == owned:
            ready = directory / "research_audit_ready.json"
            if ready in publications:
                try:
                    device, inode, sha = publications[ready]
                    _, _, current = near._read_regular_file(ready, expected_sha256=sha)
                    if (current.device, current.inode) == (device, inode):
                        ready.unlink()
                except (OSError, ValueError):
                    pass
            try:
                _publish(directory / "failed.json", dict(schema=SCHEMA, status="failed", stage=stage,
                    exception_type=type(error).__name__, **AUTHORITY), {})
            except (OSError, ValueError):
                pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "model-validation", "reference-inventory", "lineage-manifest", "lineage-report",
                 "replay-report", "detector-model", "inference-spec"):
        parser.add_argument("--" + name, required=True, type=Path)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--code-pin", action="append", required=True, metavar="FILENAME=SHA256")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = vars(parser.parse_args(argv))
    code = {}
    for item in args.pop("code_pin"):
        name, separator, sha = item.partition("=")
        require(separator and name not in code, "invalid/duplicate code pin")
        code[name] = sha
    manifests = {role: args.pop(role) for role in sorted(near.CANDIDATE_ROLES)}
    hashes = {role: args.pop(role + "_sha256") for role in sorted(near.CANDIDATE_ROLES)}
    try:
        report = audit_research(candidate_manifests=manifests, candidate_manifest_sha256=hashes, code_sha256=code, **args)
        print(json.dumps(dict(status=report["status"], **AUTHORITY)))
        return 0 if report["ok"] else 2
    except Exception as error:
        print(json.dumps(dict(status="failed", exception_type=type(error).__name__, **AUTHORITY)), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
