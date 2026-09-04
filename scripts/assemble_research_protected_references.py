"""Assemble protection-only references; never grant formal, label or training authority.

The v1 auditor and its coverage rules remain unchanged. This small adapter uses
its byte reader, image decoder, rot4 pHash and terminal asset revalidation only.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

try:
    from scripts import audit_v4_near_duplicate_leakage as near
except ModuleNotFoundError:
    import audit_v4_near_duplicate_leakage as near

INVENTORY_SCHEMA = "research_protected_reference_inventory.v2"
REPORT_SCHEMA = "research_protected_reference_assembly.v2"
BOUNDARY_ROLE = "protected_reference"
ROLES = {"qx3", "capture", "known_audit"}
AUTHORITY = {"research_only": True, "formal_protected_coverage": False,
             "training_authorized": False, "blind_test_authorized": False,
             "deployment_authorized": False, "label_authority": False,
             "selection_authorized": False, "candidate_leakage_passed": False,
             "objectness_targets_emitted": False}
PROOF_NAMES = ("fingerprint", "reuse", "roi", "observation")
OBS_CONFIG = {"device": "0", "batch": 1, "imgsz": 640, "conf": .1, "nms_iou": .7,
              "selection": "highest_confidence_then_original_order", "crop_size": 320,
              "padding": .08, "letterbox_fill": 114, "jpeg_quality": 92}
ROI_CONFIG = {"size": 320, "padding": .08, "letterbox_fill": 114, "jpeg_quality": 92}


def require(ok, message):
    if not ok:
        raise near.AuditError(message)


def path(value, *, exists=True):
    require(isinstance(value, (str, Path)), "path must be textual")
    result = Path(value)
    require(result.is_absolute() and ".." not in result.parts, "absolute traversal-free path required")
    near._reject_symlink_chain(result if exists else result.parent)
    return result


def encode(value):
    return base64.urlsafe_b64encode(os.fsencode(value)).decode("ascii")


def decode(value):
    require(type(value) is str, "encoded path required")
    raw = base64.b64decode(value, altchars=b"-_", validate=True)
    require(base64.urlsafe_b64encode(raw).decode() == value, "noncanonical encoded path")
    return path(Path(os.fsdecode(raw)))


def sha(value):
    require(type(value) is str, "SHA256 must be a string")
    return near._require_sha256(value, "reference SHA256")


def integer(value, *, minimum=0):
    require(type(value) is int and value >= minimum, "invalid exact integer")
    return value


def json_value(content):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def invalid(_):
        raise near.AuditError("nonfinite JSON value")
    return json.loads(content, object_pairs_hook=pairs, parse_constant=invalid)


def rendered(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def code_paths():
    return {p.name: path(p) for p in (Path(__file__).absolute(), Path(near.__file__).absolute())}


def roles(value):
    require(type(value) is list and value and all(type(r) is str and r in ROLES for r in value)
            and len(set(value)) == len(value), "invalid original source memberships")
    return sorted(value)


def false_fields(value, fields):
    require(type(value) is dict and all(value.get(k) is False for k in fields), "proof grants or omits forbidden authority")


def proof_rows(value):
    require(type(value) is list, "proof records must be an array")
    result = {}
    for row in value:
        require(type(row) is dict, "proof record must be an object")
        key = sha(row.get("source_sha256"))
        require(key not in result, "duplicate proof source")
        result[key] = row
    return result


def _build(proofs, model_sha, spec_sha, code_pins, *, output=None):
    """Reconstruct the inventory from pinned reports and verified image bytes."""
    model_sha, spec_sha = sha(model_sha), sha(spec_sha)
    require(type(proofs) is dict and set(proofs) == set(PROOF_NAMES), "exact four report bindings required")
    code = code_paths()
    require(type(code_pins) is dict and set(code_pins) == set(code), "exact assembler/shared-auditor code pins required")
    pins, documents, proof_paths = {}, {}, {}

    def pin(p, expected, limit=near.MAX_ENCODED_BYTES):
        p, expected = path(p), sha(expected)
        require(output is None or not (output.is_relative_to(p.parent) or p.is_relative_to(output)), "output overlaps a proof/source tree")
        require(p not in pins or pins[p][0] == expected, "conflicting file pins")
        payload, _, _ = near._read_regular_file(p, expected_sha256=expected, max_bytes=limit)
        require(not (p.parent / "failed.json").exists(), "proof/source has failure marker")
        pins[p] = (expected, limit)
        return payload

    for name, p in code.items():
        pin(p, code_pins[name])
    for name in PROOF_NAMES:
        ref = proofs[name]
        require(type(ref) is dict and set(ref) == {"path_b64", "sha256"}, "invalid report binding")
        p = decode(ref["path_b64"])
        require(p not in proof_paths.values(), "reports must be distinct")
        proof_paths[name] = p
        documents[name] = json_value(pin(p, ref["sha256"]))
    fp, reuse, roi, observation = (documents[k] for k in PROOF_NAMES)
    false_fields(fp, ("training_authorized", "deployment_authorized", "blind_test_authorized", "selection_authorized"))
    require(fp.get("schema") == "protected_image_fingerprint_snapshot.v1" and fp.get("status") == "snapshot_complete"
            and fp.get("snapshot_only") is True and fp.get("consumer_must_rehash_sources") is True, "invalid fingerprint snapshot")
    sources = proof_rows(fp.get("records"))
    require(sources and integer(fp.get("missing_sources")) == 0
            and all(integer(fp.get(k), minimum=1) == len(sources) for k in ("expected_sources", "verified_sources")), "incomplete source fingerprint union")
    require(type(fp.get("metadata_bindings")) is list, "fingerprint metadata bindings required")
    for ref in fp["metadata_bindings"]:
        require(type(ref) is dict and set(ref) == {"path_b64", "sha256"}, "invalid fingerprint metadata binding")
        pin(decode(ref["path_b64"]), ref["sha256"])

    common_false = ("training_authorized", "deployment_authorized", "blind_test_authorized", "formal_protected_coverage")
    false_fields(reuse, common_false + ("original_sources_rehashed", "crop_transform_recomputed", "detector_inference_executed"))
    require(reuse.get("schema") == "proposal_crop_reuse_audit.v1" and reuse.get("status") == "reuse_candidates_verified", "invalid reuse report")
    rb = reuse["bindings"]
    require(rb.get("declared_model_sha256") == model_sha and rb.get("declared_spec_sha256") == spec_sha, "reuse model/spec mismatch")
    sha(rb.get("audit_code_sha256"))
    for kind in ("manifest", "selection"):
        pin(path(rb[kind + "_path"]), rb[kind + "_sha256"])
    crops = proof_rows(reuse.get("records"))
    missing_reuse = proof_rows(reuse.get("missing_selection_sources"))
    qx3 = {key for key, row in sources.items() if "qx3" in roles(row.get("roles"))}
    require(not set(crops) & set(missing_reuse) and set(crops) | set(missing_reuse) == qx3, "reuse must cover the exact QX3 source membership")
    for key, row in missing_reuse.items():
        require(path(row.get("source_path")) == decode(sources[key]["source_path_b64"])
                and row.get("reason") == "no_manifest_crop", "missing reuse reference differs from source membership")
    require(integer(reuse.get("selection_sources")) == len(qx3)
            and integer(reuse.get("verified_crop_rows")) == len(crops)
            and integer(reuse.get("missing_sources")) == len(missing_reuse), "reuse counts mismatch")

    false_fields(roi, common_false + ("label_authority", "selection_authorized", "semantic_truth_established", "runtime_detector_executed", "state_targets_emitted"))
    require(roi.get("schema") == "protected_reference_roi.v1" and roi.get("status") == "reference_materialization_complete"
            and rendered(roi.get("crop_configuration")) == rendered(ROI_CONFIG), "invalid reference ROI report/configuration")
    false_fields(observation, common_false + ("label_authority", "selection_authorized", "semantic_truth_established"))
    require(observation.get("schema") == "protected_proposal_observation.v1" and observation.get("status") == "observation_complete", "invalid observation report")
    runtime = observation.get("runtime", {})
    require(runtime.get("runtime_detector_executed") is True and runtime.get("provider_kind") == "frozen_yolo_runtime"
            and rendered(runtime.get("requested_configuration")) == rendered(OBS_CONFIG), "absence requires actual frozen detector execution")
    require(observation["bindings"].get("model_sha256") == model_sha
            and observation["bindings"].get("inference_spec_sha256") == spec_sha, "observation model/spec mismatch")
    for name in ("roi", "observation"):
        bindings = documents[name]["bindings"]
        require(bindings.get("protected_report_sha256") == proofs["fingerprint"]["sha256"], "proof belongs to another fingerprint snapshot")
        file_refs = bindings.get("input_files")
        require(type(file_refs) is list and file_refs, "proof input file bindings required")
        file_shas = set()
        by_name = {}
        for ref in file_refs:
            require(type(ref) is dict and set(ref) == {"path_b64", "sha256"}, "invalid transitive proof binding")
            p, digest = decode(ref["path_b64"]), sha(ref["sha256"])
            pin(p, digest, 512 * 1024**2 if digest == model_sha else near.MAX_ENCODED_BYTES)
            file_shas.add(digest)
            by_name.setdefault(p.name, set()).add(digest)
        require(proofs["fingerprint"]["sha256"] in file_shas, "proof omits fingerprint file binding")
        code_refs = bindings.get("code_sha256")
        expected_code = {"audit_proposal_crop_reuse.py", "verifier_preprocessing_contract.py",
                         "materialize_protected_reference_crops.py" if name == "roi" else "observe_protected_proposals.py"}
        if name == "observation": expected_code.add("prepare_proposal_verifier_dataset.py")
        require(type(code_refs) is dict and set(code_refs) == expected_code, "producer code set mismatch")
        for basename, digest in code_refs.items():
            require(sha(digest) in by_name.get(basename, set()), "producer code is not bound to actual input bytes")
        if name == "observation":
            require({model_sha, spec_sha} <= file_shas, "observation model/spec files not bound")
        else:
            require({sha(bindings.get("known_audit_sha256")), sha(bindings.get("capture_inventory_sha256"))} <= file_shas,
                    "reference metadata files not bound")
    roi_rows, observed = proof_rows(roi.get("records")), proof_rows(observation.get("records"))
    raw = {key for key, row in sources.items() if set(roles(row.get("roles"))) & {"capture", "known_audit"}}
    require(set(roi_rows) == raw, "ROI report must cover the complete raw source union")
    generated_roi = {key for key, row in roi_rows.items() if row.get("status") == "reference_roi_generated"}
    require(integer(roi.get("raw_source_count")) == len(raw)
            and integer(roi.get("reference_roi_count")) == len(generated_roi)
            and integer(roi.get("missing_reference_count")) == len(raw - generated_roi), "ROI counts mismatch")
    require(not set(crops) & generated_roi, "duplicate crop authorities for one source")
    needed = set(sources) - set(crops) - generated_roi
    require(set(observed) == needed, "observation must exactly resolve sources without reuse/reference crops")
    require(1 <= len(observed) <= 32, "observation exceeds the bounded producer scope")
    require(integer(observation.get("requested_sources"), minimum=1) == len(observed)
            and integer(observation.get("observed_sources"), minimum=1) == len(observed), "incomplete observation source union")
    verified, assembled, seen_paths = [], [], set()

    def image_asset(p, digest, size, width, height, source_sha, kind):
        p, digest = path(p), sha(digest)
        require(p not in seen_paths, "duplicate source/crop asset path")
        require(output is None or not (output.is_relative_to(p.parent) or p.is_relative_to(output)), "output overlaps image tree")
        seen_paths.add(p)
        payload, _, identity = near._read_regular_file(p, expected_sha256=digest)
        signature, actual_width, actual_height = near._phash_signature(payload)
        require((identity.size, actual_width, actual_height) == (integer(size, minimum=1), integer(width, minimum=1), integer(height, minimum=1)), "source/crop size or dimensions mismatch")
        require(not (p.parent / "failed.json").exists(), "image has failure marker")
        asset = near.VerifiedAsset(p, BOUNDARY_ROLE, BOUNDARY_ROLE, kind,
            f"research-protected:{source_sha}:{kind}", source_sha, digest, identity.size,
            actual_width, actual_height, signature, identity)
        verified.append(asset)
        return {"path_b64": encode(p), "image_sha256": digest, "size": identity.size,
                "width": actual_width, "height": actual_height, "phash_rot4": [f"{v:016x}" for v in signature]}

    def matching(row, source):
        require(row.get("source_sha256") == source["source_sha256"]
                and decode(row.get("source_path_b64")) == decode(source["source_path_b64"])
                and roles(row.get("roles")) == roles(source["roles"])
                and all(type(row.get(k)) is int and row[k] == source[k] for k in ("source_bytes", "image_width", "image_height")), "proof source membership/shape mismatch")

    def bbox(value, source, *, bounded=False):
        require(type(value) is list and len(value) == 4 and all(type(v) in (int, float) and math.isfinite(v) for v in value), "invalid reference bbox")
        x1, y1, x2, y2 = value
        require(x1 < x2 and y1 < y2 and x2 > 0 and y2 > 0 and x1 < source["image_width"] and y1 < source["image_height"], "reference bbox does not intersect image")
        if bounded:
            require(all(type(v) is int for v in value) and 0 <= x1 < x2 <= source["image_width"] and 0 <= y1 < y2 <= source["image_height"], "invalid crop bounds")
        return value

    def output_crop(proof_name, row, source, provenance):
        crop = row.get("crop")
        require(type(crop) is dict and crop.get("provenance") == provenance, "crop provenance mismatch")
        relative = near._normalized_relative_path(crop.get("path"), "proof crop path")
        require(relative == PurePosixPath("crops") / (source["source_sha256"] + ".jpg"), "unexpected proof crop filename")
        bounds = bbox(crop.get("bounds_xyxy"), source, bounded=True)
        result = image_asset(proof_paths[proof_name].parent.joinpath(*relative.parts), crop["sha256"], crop["bytes"], crop["width"], crop["height"], source["source_sha256"], "crop")
        require(result["width"] == result["height"] == 320, "protected crop must be 320 square")
        return {**result, "provenance": provenance, "proof_sha256": proofs[proof_name]["sha256"], "bounds_xyxy": bounds}

    for key, source in sorted(sources.items()):
        require(source.get("source_sha256") == key, "source SHA mismatch")
        source_asset = image_asset(decode(source["source_path_b64"]), key, source["source_bytes"], source["image_width"], source["image_height"], key, "source")
        item = {"source_sha256": key, "roles": roles(source["roles"]), "boundary_role": BOUNDARY_ROLE,
                "source": source_asset, "crop": None, "absence": None}
        if key in roi_rows:
            row = roi_rows[key]; matching(row, source)
            require(row.get("object_absence_established") is False, "reference cannot establish object absence")
            if key not in generated_roi:
                require(row.get("status") == "missing_reference" and row.get("reference") is None and row.get("crop") is None, "invalid missing reference")
        if key in crops:
            row = crops[key]
            require(path(row["source_path"]) == decode(source["source_path_b64"])
                    and row.get("declared_source_size_wh") == [source["image_width"], source["image_height"]], "reuse source differs from fingerprint")
            bounds = bbox(row.get("declared_crop_xyxy"), source, bounded=True)
            crop_path = path(row["crop_path"])
            require(crop_path.is_relative_to(path(rb["crop_root"])), "reuse crop escapes declared root")
            crop = image_asset(crop_path, row["crop_sha256"], row["crop_bytes"], 320, 320, key, "crop")
            item["crop"] = {**crop, "provenance": "existing_qx3_replay_crop", "proof_sha256": proofs["reuse"]["sha256"], "bounds_xyxy": bounds}
        elif key in generated_roi:
            row, reference = roi_rows[key], roi_rows[key].get("reference")
            require(type(reference) is dict and set(reference) == {"kind", "bbox_source", "bbox_xyxy", "metadata_sha256", "field"}, "invalid reference provenance")
            kind = reference["kind"]
            require(kind in {"known_audit_reference", "historical_deployed_reference"}
                    and type(reference["bbox_source"]) is str and bool(reference["bbox_source"]), "invalid reference kind")
            bind_key, expected_field = ("known_audit_sha256", "bbox") if kind == "known_audit_reference" else ("capture_inventory_sha256", "deployed.bbox")
            require(sha(reference["metadata_sha256"]) == roi["bindings"].get(bind_key) and reference["field"] == expected_field, "reference metadata binding mismatch")
            bbox(reference["bbox_xyxy"], source)
            item["crop"] = {**output_crop("roi", row, source, kind), "reference": reference}
        else:
            row = observed[key]; matching(row, source)
            require(row.get("object_absence_established") is False, "observation cannot establish semantic absence")
            returned = integer(row.get("returned_proposals_after_model_confidence_nms"))
            eligible, below = integer(row.get("eligible_proposals")), integer(row.get("below_confidence_floor"))
            require(returned == eligible + below, "observation proposal count mismatch")
            if row.get("observation_status") == "no_eligible_proposal":
                require(eligible == 0 and row.get("crop") is None and row.get("selected_proposal") is None, "forged crop absence")
                item["absence"] = {"reason": "no_eligible_proposal", "observation_report_sha256": proofs["observation"]["sha256"],
                                   "source_sha256": key, "returned_proposals_after_model_confidence_nms": returned,
                                   "object_absence_established": False}
            else:
                selected = row.get("selected_proposal")
                require(row.get("observation_status") == "crop_generated" and eligible > 0 and type(selected) is dict, "invalid successful crop observation")
                require(integer(selected.get("index")) < returned and type(selected.get("confidence")) in (int, float)
                        and math.isfinite(selected["confidence"]) and .1 <= selected["confidence"] <= 1, "invalid selected observation")
                bbox(selected.get("bbox_xyxy"), source)
                item["crop"] = {**output_crop("observation", row, source, "actual_yolo_runtime_top1"), "selected_proposal": selected}
        assembled.append(item)
    absence_count = sum(r["absence"] is not None for r in assembled)
    require(integer(observation.get("no_eligible_proposal")) == absence_count
            and integer(observation.get("crop_generated")) == len(observed) - absence_count, "observation summary count mismatch")
    inventory = {"schema": INVENTORY_SCHEMA, "status": "reference_snapshot_complete", **AUTHORITY,
        "algorithm": {"id": near.ALGORITHM_ID, "distance": near.PHASH_DISTANCE, "crop_invariant": False},
        "bindings": {"proofs": proofs, "model_sha256": model_sha, "inference_spec_sha256": spec_sha,
                     "code_sha256": dict(sorted(code_pins.items())), "shared_auditor": near._auditor_binding()},
        "coverage": {"sources": len(sources), "crops": len(sources) - absence_count, "observed_crop_absences": absence_count,
                     "all_sources_present": True, "all_available_crops_present": True, "full_source_crop_coverage": absence_count == 0},
        "sources": assembled}
    return inventory, tuple(verified), pins


@dataclass
class ResearchReferences:
    records: tuple[near.VerifiedAsset, ...]
    inventory: dict
    _pins: dict
    _binding: dict

    @property
    def input_paths(self):
        return tuple(sorted(set(self._pins) | {record.path for record in self.records}, key=str))

    def binding(self):
        return json_value(rendered(self._binding))

    def recheck(self):
        def metadata():
            for p, (digest, limit) in self._pins.items():
                require(not (p.parent / "failed.json").exists(), "reference input failed")
                near._read_regular_file(p, expected_sha256=digest, max_bytes=limit)
        metadata()
        require(near._auditor_binding() == self.inventory["bindings"]["shared_auditor"], "loaded shared auditor changed")
        near.reverify_assets(self.records)
        require(not any((r.path.parent / "failed.json").exists() for r in self.records), "asset failure marker appeared")
        metadata()  # A proof mutation during the longer image pass must also fail.
        require(near._auditor_binding() == self.inventory["bindings"]["shared_auditor"], "loaded shared auditor changed")


def _publish(p, payload, recheck):
    near._atomic_no_overwrite(p, payload, pre_publish=recheck)


def assemble(*, fingerprint_report, fingerprint_report_sha256, reuse_report, reuse_report_sha256,
             roi_report, roi_report_sha256, observation_report, observation_report_sha256,
             model_sha256, inference_spec_sha256, code_pins, output):
    output = path(output, exists=False)
    require(not output.exists(), "fresh output required")
    proofs = {name: {"path_b64": encode(path(p)), "sha256": sha(digest)} for name, p, digest in
              zip(PROOF_NAMES, (fingerprint_report, reuse_report, roi_report, observation_report),
                  (fingerprint_report_sha256, reuse_report_sha256, roi_report_sha256, observation_report_sha256), strict=True)}
    inventory, records, pins = _build(proofs, model_sha256, inference_spec_sha256, code_pins, output=output)
    bundle = ResearchReferences(records, inventory, pins, {})
    bundle.recheck()
    output.mkdir(exist_ok=False)
    owned = (output.stat().st_dev, output.stat().st_ino)
    publications = {}
    try:
        data = rendered(inventory)
        report = {"schema": REPORT_SCHEMA, "status": "reference_assembly_complete", **AUTHORITY,
                  "inventory_sha256": hashlib.sha256(data).hexdigest(), "coverage": inventory["coverage"],
                  "bindings": inventory["bindings"]}
        for name, payload in (("reference_inventory.json", data), ("report.json", rendered(report))):
            p = output / name
            _publish(p, payload, bundle.recheck)
            publications[p] = payload
        bundle.recheck()
        for p, payload in publications.items():
            near._read_regular_file(p, expected_sha256=hashlib.sha256(payload).hexdigest())
        return report
    except BaseException:
        near._reject_symlink_chain(output)
        if (output.stat().st_dev, output.stat().st_ino) == owned:
            for p, payload in publications.items():
                if p.is_file() and not p.is_symlink() and p.read_bytes() == payload:
                    p.unlink()
            with (output / "failed.json").open("xb") as stream:
                stream.write(rendered({"status": "failed", **AUTHORITY}))
        raise


def load_reference_inventory(inventory_path: Path, expected_sha256: str, *, expected_model_sha256: str,
                             expected_spec_sha256: str) -> ResearchReferences:
    inventory_path = path(inventory_path)
    require(inventory_path.name == "reference_inventory.json", "canonical reference inventory filename required")
    content, inventory_sha, _ = near._read_regular_file(inventory_path, expected_sha256=sha(expected_sha256))
    value = json_value(content)
    require(type(value) is dict and value.get("schema") == INVENTORY_SCHEMA, "research v2 inventory required")
    binding = value["bindings"]
    require(binding["model_sha256"] == sha(expected_model_sha256) and binding["inference_spec_sha256"] == sha(expected_spec_sha256), "consumer model/spec pin mismatch")
    rebuilt, records, pins = _build(binding["proofs"], expected_model_sha256, expected_spec_sha256, binding["code_sha256"])
    require(rendered(value) == rendered(rebuilt), "inventory differs from its actual bound proofs/images")
    report_path = inventory_path.parent / "report.json"
    report_bytes, report_sha, _ = near._read_regular_file(report_path, expected_sha256=None)
    report = json_value(report_bytes)
    expected_report = {"schema": REPORT_SCHEMA, "status": "reference_assembly_complete", **AUTHORITY,
                       "inventory_sha256": inventory_sha, "coverage": value["coverage"], "bindings": value["bindings"]}
    require(rendered(report) == rendered(expected_report), "assembly report/inventory binding mismatch")
    pins[inventory_path], pins[report_path] = (inventory_sha, near.MAX_ENCODED_BYTES), (report_sha, near.MAX_ENCODED_BYTES)
    result = ResearchReferences(records, rebuilt, pins, {"inventory_path": str(inventory_path), "inventory_sha256": inventory_sha,
        "assembly_report_path": str(report_path), "assembly_report_sha256": report_sha,
        "model_sha256": expected_model_sha256, "inference_spec_sha256": expected_spec_sha256,
        "assembler_code_sha256": binding["code_sha256"][Path(__file__).name], "shared_auditor": near._auditor_binding()})
    result.recheck()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in PROOF_NAMES:
        parser.add_argument("--" + name + "-report", type=Path, required=True)
        parser.add_argument("--" + name + "-report-sha256", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--inference-spec-sha256", required=True)
    parser.add_argument("--code-pin", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    values = vars(parser.parse_args(argv)); pins = {}
    for item in values.pop("code_pin"):
        name, separator, digest = item.partition("=")
        require(separator and name not in pins, "invalid duplicate code pin")
        pins[name] = sha(digest)
    report = assemble(**values, code_pins=pins)
    print(json.dumps({"status": report["status"], "coverage": report["coverage"], **AUTHORITY}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, **AUTHORITY}), flush=True)
        raise SystemExit(1) from None
