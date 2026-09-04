"""Plan a leakage-filtered original cohort from pinned image audit snapshots.

This stage reads metadata only. It does not turn snapshots into training authority;
the materializer must recheck original image/annotation bytes before consumption.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

try:
    from scripts import audit_aihub_original_annotations as original
except ModuleNotFoundError:
    import audit_aihub_original_annotations as original

SHA = re.compile(r"[0-9a-f]{64}")
SOURCE_ID = re.compile(r"[0-9a-f]{20}")
PHASH = re.compile(r"[0-9a-f]{16}")
PHASH_CONVENTION = "direct-grayscale-imdecode_area32_dct8_median-exclude-dc_64bit"


class PlanError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise PlanError(message)


def checked_hex(value, pattern, description):
    require(type(value) is str and pattern.fullmatch(value) is not None, description)
    return value


class HammingIndex:
    """Exact Hamming-radius lookup using distance+1 disjoint bit bands."""
    def __init__(self, distance=4):
        require(type(distance) is int and 0 <= distance <= 7, "invalid pHash distance")
        self.distance = distance
        self.bands = []
        offset = 0
        for i in range(distance + 1):
            width = 64 // (distance + 1) + (i < 64 % (distance + 1))
            self.bands.append((offset, (1 << width) - 1))
            offset += width
        self.buckets = [defaultdict(set) for _ in self.bands]
        self.values = defaultdict(set)

    @staticmethod
    def validate(value):
        require(type(value) is int and 0 <= value < 2**64, "invalid 64-bit pHash")

    def add(self, value, key):
        self.validate(value)
        if value not in self.values:
            for bucket, (offset, mask) in zip(self.buckets, self.bands):
                bucket[(value >> offset) & mask].add(value)
        self.values[value].add(key)

    def _candidates(self, value):
        self.validate(value)
        candidates = set()
        for bucket, (offset, mask) in zip(self.buckets, self.bands):
            candidates.update(bucket.get((value >> offset) & mask, ()))
        return candidates

    def matches(self, value):
        return {key for other in self._candidates(value) if (value ^ other).bit_count() <= self.distance
                for key in self.values[other]}

    def has_match(self, value):
        return any((value ^ other).bit_count() <= self.distance for other in self._candidates(value))


def validate_original_row(row):
    require(type(row) is dict and row.get("status") == "verified_pair", "not a verified original pair")
    checked_hex(row.get("source_id"), SOURCE_ID, "invalid source_id")
    checked_hex(row.get("source_sha256"), SHA, "invalid source SHA")
    checked_hex(row.get("label_sha256"), SHA, "invalid label SHA")
    checked_hex(row.get("source_phash64"), PHASH, "invalid source pHash")
    require(row.get("split") in ("training", "validation"), "invalid official split")
    cls = row.get("class_id")
    require(type(cls) is int and 0 <= cls < 9 and row.get("class_name") == original.CLASS_NAMES[cls], "class mapping mismatch")
    h, w = row.get("image_height"), row.get("image_width")
    require(type(h) is int and type(w) is int and min(h, w) > 0, "invalid source dimensions")
    require(type(row.get("source_bytes")) is int and row["source_bytes"] > 0, "invalid source byte count")
    try:
        require(type(row.get("bbox_xywh")) is list, "invalid source bbox")
        original.strict_bbox([row["bbox_xywh"]], w, h)
    except original.AnnotationError as exc:
        raise PlanError(str(exc)) from exc
    conditions = row.get("conditions")
    require(type(conditions) is dict and set(conditions) == {"dent", "label", "foreign_material"}
            and all(type(v) is int and v == -1 for v in conditions.values()), "unverified state must remain masked")


def plan_records(originals: list[dict], protected: list[dict], protected_source_ids: set[str],
                 *, quarantine_annotation_conflicts: bool = False) -> dict:
    require(type(originals) is list and bool(originals), "empty original cohort")
    require(type(protected) is list, "invalid protected reference set")
    require(type(quarantine_annotation_conflicts) is bool, "annotation conflict opt-in must be boolean")
    for sid in protected_source_ids:
        checked_hex(sid, SOURCE_ID, "invalid protected source_id")
    by_id, by_sha = {}, defaultdict(list)
    split_indexes = {split: HammingIndex() for split in ("training", "validation")}
    for row in originals:
        validate_original_row(row)
        sid = row["source_id"]
        require(sid not in by_id, "duplicate source_id")
        by_id[sid] = row
        by_sha[row["source_sha256"]].append(row)
        split_indexes[row["split"]].add(int(row["source_phash64"], 16), sid)
    protected_shas, protected_index = set(), HammingIndex()
    for row in protected:
        require(type(row) is dict, "invalid protected record")
        sha = checked_hex(row.get("source_sha256"), SHA, "invalid protected SHA")
        phash = checked_hex(row.get("source_phash64"), PHASH, "invalid protected pHash")
        require(sha not in protected_shas, "duplicate protected SHA")
        protected_shas.add(sha)
        protected_index.add(int(phash, 16), sha)
    reasons, conflict_groups = defaultdict(set), []
    for sha, group in by_sha.items():
        image_signatures = {(r["image_width"], r["image_height"], r["source_phash64"], r["source_bytes"]) for r in group}
        require(len(image_signatures) == 1, "same image SHA has conflicting image evidence")
        annotation_conflict = len({(r["class_id"], tuple(r["bbox_xywh"])) for r in group}) > 1
        if annotation_conflict:
            require(quarantine_annotation_conflicts, "same image SHA has conflicting ground truth or image evidence")
            for row in group:
                reasons[row["source_id"]].add("annotation_conflict_same_sha256")
            conflict_groups.append({
                "source_sha256": sha,
                "image_evidence": {key: group[0][key] for key in
                                   ("image_width", "image_height", "source_phash64", "source_bytes")},
                "members": [{key: r[key] for key in
                             ("source_id", "split", "label_sha256", "class_id", "class_name", "bbox_xywh",
                              "source_path_b64", "label_path_b64")}
                            for r in sorted(group, key=lambda r: r["source_id"])],
            })
        if len({r["split"] for r in group}) > 1:
            for row in group:
                reasons[row["source_id"]].add("cross_split_duplicate")
        elif len(group) > 1 and not annotation_conflict:
            for row in sorted(group, key=lambda r: r["source_id"])[1:]:
                reasons[row["source_id"]].add("same_split_duplicate")
    for sid, row in by_id.items():
        phash = int(row["source_phash64"], 16)
        if row["source_sha256"] in protected_shas:
            reasons[sid].add("protected_exact_sha256")
        if sid in protected_source_ids:
            reasons[sid].add("protected_source_id")
        if protected_index.has_match(phash):
            reasons[sid].add("protected_near_phash")
        other_split = "validation" if row["split"] == "training" else "training"
        if split_indexes[other_split].has_match(phash):
            reasons[sid].add("cross_split_duplicate")
    accepted = [by_id[sid] for sid in sorted(by_id) if not reasons[sid]]
    excluded = [{"source_id": sid, "reasons": sorted(reasons[sid])} for sid in sorted(by_id) if reasons[sid]]
    counts = {"originals": len(originals), "accepted": len(accepted), "excluded": len(excluded),
              "accepted_by_split_class": dict(sorted(Counter(f'{r["split"]}/{r["class_name"]}' for r in accepted).items())),
              "exclusion_reason_counts_overlapping": dict(sorted(Counter(reason for r in excluded for reason in r["reasons"]).items()))}
    result = {"records": accepted, "exclusions": excluded, "counts": counts}
    if quarantine_annotation_conflicts:
        result["annotation_conflict_quarantine"] = {
            "enabled": True, "group_count": len(conflict_groups),
            "source_count": sum(len(group["members"]) for group in conflict_groups),
            "groups": sorted(conflict_groups, key=lambda group: group["source_sha256"]),
        }
    return result


def decode_path(value) -> PurePosixPath:
    require(type(value) is str, "missing path encoding")
    try:
        raw = os.fsdecode(base64.b64decode(value, altchars=b"-_", validate=True))
    except (ValueError, UnicodeError) as exc:
        raise PlanError("invalid encoded source path") from exc
    path = PurePosixPath(raw)
    require(path.is_absolute() and ".." not in path.parts and "\x00" not in raw, "invalid source path")
    return path


def protected_original_ids(protected, selected: Path, originals_by_id):
    protected_stems = set()
    commercial = PurePosixPath('/app/yolo_commercial_single_v1_20260813/images/train')
    for row in protected:
        path = decode_path(row["source_path_b64"])
        if path.parent == commercial:
            protected_stems.add(path.stem)
    matched, seen = set(), set()
    with selected.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            stem, sid = row["stem"], checked_hex(row["source_id"], SOURCE_ID, "invalid commercial source_id")
            require(stem not in seen and stem == f'{row["category"]}_{sid}', "invalid or duplicate selected stem")
            seen.add(stem)
            if stem in protected_stems:
                matched.add(sid)
                if sid in originals_by_id:
                    source = originals_by_id[sid]
                    require(source["split"] == "training" and source["class_name"] == row["category"]
                            and str(source["class_id"]) == row["class_id"]
                            and str(decode_path(source["source_path_b64"])) == row["source_path"], "commercial/original identity mismatch")
    unmapped = protected_stems - seen
    require(all(stem.startswith("hardware_r") for stem in unmapped), "unresolved protected commercial source")
    return matched, len(unmapped)


def load_pinned(path: Path, sha: str):
    checked_hex(sha, SHA, "invalid input pin")
    require(path.is_file() and not path.is_symlink(), "input must be regular file")
    content = original.read_stable(path, 768 * 1024**2)
    require(hashlib.sha256(content).hexdigest() == sha, "input SHA mismatch")
    return json.loads(content, object_pairs_hook=original.unique_object)


def _encoded_bytes(value):
    require(type(value) is str, "missing encoded evidence path")
    raw = base64.b64decode(value, altchars=b"-_", validate=True)
    require(b"\x00" not in raw, "invalid encoded evidence path")
    path = Path(os.fsdecode(raw))
    posix = PurePosixPath(os.fsdecode(raw))
    require((path.is_absolute() or posix.is_absolute()) and ".." not in path.parts
            and ".." not in posix.parts, "invalid encoded evidence path")
    return raw


def _legacy_identity(encoded):
    raw = _encoded_bytes(encoded)
    path = PurePosixPath(os.fsdecode(raw))
    if path.parent.name != "images" or path.parent.parent.parent.name != "yolo_dataset_9class_v2":
        return None
    match = re.fullmatch(r"(train_r|train_|val_)([0-9]{7})\.jpg", path.name)
    require(match is not None, "unrecognized protected legacy filename")
    kind, index = match[1].rstrip("_"), int(match[2])
    require(path.parent.parent.name == ("val" if kind == "val" else "train"), "legacy official split mismatch")
    return raw, kind, index


def _evidence_bindings(rows):
    require(type(rows) is list and bool(rows), "missing evidence input bindings")
    result = {}
    for row in rows:
        require(type(row) is dict, "invalid evidence binding")
        raw = _encoded_bytes(row.get("path_b64"))
        require(raw not in result, "duplicate evidence input binding")
        result[raw] = checked_hex(row.get("sha256"), SHA, "invalid evidence input SHA")
    return result


def recheck_legacy_bindings(bindings, reports):
    for report in reports:
        failed = report.parent / "failed.json"
        require(not failed.exists() and not failed.is_symlink(), "legacy evidence has a failure marker")
    for binding in bindings:
        path = Path(binding["path"])
        require(path.is_absolute() and ".." not in path.parts
                and not any(p.is_symlink() for p in (path, *path.parents)), "unsafe legacy metadata path")
        require(original.digest_file(path) == binding["sha256"], "legacy metadata changed")


def validate_legacy_link_report(*, link_report: Path, link_sha256: str,
                                protected_report: Path, protected_sha256: str,
                                protected: list[dict]) -> dict:
    """Validate full reference coverage; unresolved candidate fields confer no source link."""
    reports = (link_report, protected_report)
    pins = {p.absolute(): checked_hex(s, SHA, "invalid legacy report pin") for p, s in
            ((link_report, link_sha256), (protected_report, protected_sha256))}
    require(len(pins) == 2, "legacy evidence reports must be distinct")
    bindings = lambda: [{"path": str(p), "sha256": s} for p, s in sorted(pins.items())]
    recheck_legacy_bindings(bindings(), reports)
    links = load_pinned(link_report, link_sha256)
    snapshot = load_pinned(protected_report, protected_sha256)
    require(type(snapshot) is dict and snapshot.get("records") == protected, "protected rows differ from pinned snapshot")
    require(snapshot.get("schema") == "protected_image_fingerprint_snapshot.v1"
            and snapshot.get("status") == "snapshot_complete" and snapshot.get("snapshot_only") is True
            and snapshot.get("training_authorized") is False and snapshot.get("deployment_authorized") is False,
            "protected snapshot is not complete non-authoritative evidence")
    for field, count in (("missing_sources", 0), ("expected_sources", len(protected)), ("verified_sources", len(protected))):
        require(type(snapshot.get(field)) is int and snapshot[field] == count, "protected snapshot coverage mismatch")
    require(type(links) is dict and links.get("schema") == "legacy_aihub_source_link_probe.v1"
            and links.get("status") == "probe_complete" and links.get("partial_selection") is False
            and type(links.get("max_per_kind")) is int and links["max_per_kind"] == 0
            and links.get("candidate_index_is_search_only") is True, "full legacy link probe required")
    for field in ("training_authorized", "blind_test_authorized", "deployment_authorized",
                  "complete_original_lineage", "original_alias_uniqueness_proven"):
        require(links.get(field) is False, "legacy links must not claim authority or unique original lineage")
    consumed = _evidence_bindings(links.get("metadata_and_consumed_inputs"))
    require(consumed.get(os.fsencode(protected_report.absolute())) == protected_sha256,
            "legacy links do not bind the supplied protected snapshot")
    expected, kind_counts, protected_shas = {}, Counter(), set()
    for row in protected:
        require(type(row) is dict, "invalid protected reference")
        sha = checked_hex(row.get("source_sha256"), SHA, "invalid protected SHA")
        require(sha not in protected_shas, "duplicate protected SHA")
        protected_shas.add(sha)
        identity = _legacy_identity(row["source_path_b64"])
        if identity is not None:
            raw, kind, index = identity
            key = (raw, checked_hex(row.get("source_sha256"), SHA, "invalid protected legacy SHA"))
            require(key not in expected, "duplicate protected legacy reference")
            expected[key] = (kind, index)
            kind_counts[kind] += 1
    require(bool(expected), "protected snapshot contains no legacy references")
    declared_counts = links.get("protected_legacy_counts")
    require(type(declared_counts) is dict and all(type(v) is int for v in declared_counts.values())
            and {k: v for k, v in declared_counts.items() if v} == dict(kind_counts)
            and set(declared_counts) <= {"train", "train_r", "val"}, "legacy reference counts mismatch")
    rows = links.get("records")
    require(type(rows) is list and len(rows) == len(expected), "legacy reference coverage mismatch")
    seen, recovered, unresolved = set(), {}, []
    for row in rows:
        require(type(row) is dict, "invalid legacy link row")
        key = (_encoded_bytes(row.get("legacy_path_b64")), checked_hex(row.get("legacy_sha256"), SHA, "invalid legacy SHA"))
        require(key in expected and key not in seen, "legacy reference missing, extra or duplicated")
        seen.add(key)
        kind, index = expected[key]
        require(row.get("kind") == kind and type(row.get("index")) is int and row["index"] == index,
                "legacy index/path identity mismatch")
        require(consumed.get(key[0]) == key[1], "legacy image is not bound to consumed bytes")
        if row.get("status") == "unresolved":
            require(type(row.get("reason")) is str and bool(row["reason"]), "unresolved legacy reason missing")
            unresolved.append({"legacy_sha256": key[1], "reason": row["reason"]})
            continue  # Candidate fields on unresolved rows never become exclusions.
        require(row.get("status") == "verified_source_link" and row.get("reason") == "exact_legacy_jpeg_bytes"
                and row.get("regenerated_legacy_sha256") == key[1], "unverified legacy reproduction")
        source, annotation = _encoded_bytes(row.get("source_path_b64")), _encoded_bytes(row.get("annotation_path_b64"))
        source_sha = checked_hex(row.get("source_sha256"), SHA, "invalid recovered source SHA")
        annotation_sha = checked_hex(row.get("annotation_sha256"), SHA, "invalid recovered annotation SHA")
        require(consumed.get(source) == source_sha and consumed.get(annotation) == annotation_sha,
                "recovered pair is not bound to consumed bytes")
        evidence = (source_sha, annotation, annotation_sha, "validation" if kind == "val" else "training")
        require(source not in recovered or recovered[source] == evidence, "conflicting recovered source evidence")
        recovered[source] = evidence
    status_counts = dict(Counter(row["status"] for row in rows))
    require(type(links.get("status_counts")) is dict
            and all(type(v) is int for v in links["status_counts"].values())
            and links["status_counts"] == status_counts, "legacy status counts mismatch")
    recheck_legacy_bindings(bindings(), reports)
    return {"verified_records": [dict(row) for row in rows if row["status"] == "verified_source_link"],
            "recovered": recovered, "unresolved": unresolved, "status_counts": status_counts,
            "expected_legacy_references": len(expected), "bindings": bindings()}


def load_legacy_exclusions(*, link_report: Path, link_sha256: str,
                           fingerprint_report: Path, fingerprint_sha256: str,
                           protected_report: Path, protected_sha256: str,
                           protected: list[dict], originals: list[dict], original_auditor_sha256: str):
    """Join pinned snapshots only; image decoding/GT and training approval stay upstream/downstream."""
    evidence = validate_legacy_link_report(link_report=link_report, link_sha256=link_sha256,
        protected_report=protected_report, protected_sha256=protected_sha256, protected=protected)
    reports = (link_report, fingerprint_report, protected_report)
    pins = {Path(row["path"]): row["sha256"] for row in evidence["bindings"]}
    require(fingerprint_report.absolute() not in pins, "legacy evidence reports must be distinct")
    pins[fingerprint_report.absolute()] = checked_hex(fingerprint_sha256, SHA, "invalid fingerprint report pin")
    bindings = lambda: [{"path": str(p), "sha256": s} for p, s in sorted(pins.items())]
    recheck_legacy_bindings(bindings(), reports)
    fingerprints = load_pinned(fingerprint_report, fingerprint_sha256)
    recovered, unresolved, status_counts = evidence["recovered"], evidence["unresolved"], evidence["status_counts"]
    require(type(fingerprints) is dict and fingerprints.get("schema") == "protected_image_fingerprint_snapshot.v1"
            and fingerprints.get("status") == "snapshot_complete" and fingerprints.get("snapshot_only") is True,
            "invalid recovered-original fingerprint snapshot")
    for field in ("training_authorized", "deployment_authorized", "blind_test_authorized", "selection_authorized"):
        require(fingerprints.get(field) is False, "fingerprints must not grant authority")
    require(fingerprints.get("code_sha256", {}).get("audit_aihub_original_annotations.py") == original_auditor_sha256,
            "recovered-original perceptual implementation mismatch")
    checked_hex(fingerprints.get("inventory_sha256"), SHA, "invalid recovered fingerprint inventory pin")
    fingerprint_bindings = _evidence_bindings(fingerprints.get("metadata_bindings"))
    require(fingerprint_bindings.get(os.fsencode(link_report.absolute())) == link_sha256,
            "recovered fingerprints do not bind the legacy link report")
    for raw, sha in fingerprint_bindings.items():
        path = Path(os.fsdecode(raw))
        require(path not in pins or pins[path] == sha, "conflicting legacy metadata pins")
        pins[path] = sha
    fp_rows, by_sha = fingerprints.get("records"), {}
    require(type(fp_rows) is list and bool(fp_rows), "recovered fingerprints are missing")
    for field, count in (("missing_sources", 0), ("expected_sources", len(fp_rows)), ("verified_sources", len(fp_rows))):
        require(type(fingerprints.get(field)) is int and fingerprints[field] == count, "recovered fingerprint coverage mismatch")
    for row in fp_rows:
        require(type(row) is dict, "invalid recovered fingerprint row")
        sha = checked_hex(row.get("source_sha256"), SHA, "invalid recovered fingerprint SHA")
        path = _encoded_bytes(row.get("source_path_b64"))
        require(sha not in by_sha and path in recovered and recovered[path][0] == sha, "fingerprint source is not a verified legacy link")
        checked_hex(row.get("source_phash64"), PHASH, "invalid recovered pHash")
        require(all(type(row.get(k)) is int and row[k] > 0 for k in ("image_height", "image_width", "source_bytes")),
                "invalid recovered image dimensions/size")
        roles = row.get("roles")
        require(type(roles) is list and bool(roles) and all(type(role) is str and role in {"qx3", "capture", "known_audit"} for role in roles)
                and len(set(roles)) == len(roles), "invalid recovered fingerprint roles")
        by_sha[sha] = row
    require(set(by_sha) == {row[0] for row in recovered.values()}, "verified legacy source fingerprints missing")
    protected_ids, path_matches, matched_shas = set(), set(), set()
    for row in originals:
        path, sha = _encoded_bytes(row["source_path_b64"]), row["source_sha256"]
        if path in recovered:
            r_sha, annotation, annotation_sha, split = recovered[path]
            require(sha == r_sha and _encoded_bytes(row["label_path_b64"]) == annotation
                    and row["label_sha256"] == annotation_sha and row["split"] == split,
                    "legacy/original path, pair SHA or official split mismatch")
            path_matches.add(path)
        if sha in by_sha:
            fp = by_sha[sha]
            require(all(row.get(k) == fp[k] and type(row.get(k)) is type(fp[k]) for k in
                        ("source_phash64", "image_height", "image_width", "source_bytes")),
                    "same original SHA has conflicting image evidence")
            protected_ids.add(row["source_id"])
            matched_shas.add(sha)
    union = {row["source_sha256"]: row for row in protected}
    require(len(union) == len(protected), "duplicate base protected SHA")
    for sha, row in by_sha.items():
        if sha in union:
            require(union[sha]["source_phash64"] == row["source_phash64"], "protected fingerprint conflict")
        else:
            union[sha] = row
    summary = {"expected_legacy_references": evidence["expected_legacy_references"], "verified_source_links": status_counts.get("verified_source_link", 0),
               "unresolved_legacy_references": len(unresolved), "unresolved": unresolved,
               "recovered_original_unique_shas": len(by_sha), "original_path_matches": len(path_matches),
               "recovered_paths_outside_original_pool": len(set(recovered) - path_matches),
               "verified_links_without_original_pool_id": sum(row["source_sha256"] not in matched_shas for row in evidence["verified_records"]),
               "protected_original_ids_bound": len(protected_ids), "complete_original_lineage": False,
               "original_alias_uniqueness_proven": False, "training_authorized": False, "deployment_authorized": False}
    recheck_legacy_bindings(bindings(), reports)
    return protected_ids, list(union.values()), summary, bindings()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-report", "protected-report", "selected-manifest", "original-auditor"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    for name in ("legacy-link-report", "legacy-original-fingerprint-report"):
        parser.add_argument(f"--{name}", type=Path)
        parser.add_argument(f"--{name}-sha256")
    parser.add_argument("--quarantine-annotation-conflicts", action="store_true",
                        help="Exclude every same-SHA class/bbox conflict member; image-evidence conflicts still fail")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    legacy_args = (args.legacy_link_report, args.legacy_link_report_sha256,
                   args.legacy_original_fingerprint_report, args.legacy_original_fingerprint_report_sha256)
    require(not any(v is not None for v in legacy_args) or all(v is not None for v in legacy_args),
            "legacy link and original fingerprint reports with both SHA pins are required together")
    bindings = [{"path": str(getattr(args, key).resolve()), "sha256": getattr(args, key + "_sha256")}
                for key in ("original_report", "protected_report", "selected_manifest", "original_auditor")]
    for binding in bindings:
        checked_hex(binding["sha256"], SHA, "invalid metadata pin")
        require(original.digest_file(Path(binding["path"])) == binding["sha256"], "metadata pin mismatch")
    raw = load_pinned(args.original_report, args.original_report_sha256)
    protected = load_pinned(args.protected_report, args.protected_report_sha256)
    require(not (args.protected_report.parent / 'failed.json').exists(), "protected snapshot failed")
    require(raw.get("schema") == "aihub_original_annotation_audit_v1"
            and raw.get("perceptual_hash") == PHASH_CONVENTION, "original snapshot contract mismatch")
    require(raw.get("training_authorized") is False and raw.get("deployment_authorized") is False, "invalid original authority")
    require(raw.get("selected") == len(raw["records"]) == sum(raw["manifest_counts"].values()), "partial original audit cannot define full cohort")
    require(protected.get("status") == "snapshot_complete" and protected.get("snapshot_only") is True
            and protected.get("training_authorized") is False and protected.get("deployment_authorized") is False
            and protected.get("missing_sources") == 0, "protected snapshot incomplete")
    require(protected.get("expected_sources") == protected.get("verified_sources") == len(protected["records"]), "protected source coverage mismatch")
    require(bool(protected["records"]), "empty protected snapshot cannot authorize a cohort plan")
    require(protected["code_sha256"].get("audit_aihub_original_annotations.py") == args.original_auditor_sha256,
            "original/protected perceptual implementation mismatch")
    for encoded in protected["metadata_bindings"]:
        path = Path(str(decode_path(encoded["path_b64"])))
        sha = checked_hex(encoded["sha256"], SHA, "invalid protected metadata SHA")
        require(original.digest_file(path) == sha, "protected metadata changed")
        bindings.append({"path": str(path), "sha256": sha})
    records = [r for r in raw["records"] if r.get("status") == "verified_pair"]
    quarantined = [r for r in raw["records"] if r.get("status") == "quarantined"]
    require(len(records) == raw["verified"] and len(quarantined) == raw["quarantined"]
            and len(records) + len(quarantined) == len(raw["records"]), "original audit coverage mismatch")
    infrastructure_failures = {"total read budget exhausted", "PermissionError", "TimeoutError", "OSError", "source file changed during read"}
    require(not any(r.get("reason") in infrastructure_failures for r in quarantined), "original audit has unresolved infrastructure failures")
    ids, unmapped = protected_original_ids(protected["records"], args.selected_manifest, {r["source_id"]: r for r in records})
    commercial_id_count = len(ids)
    references, legacy_summary, legacy_bindings = protected["records"], None, []
    if args.legacy_link_report is not None:
        legacy_ids, references, legacy_summary, legacy_bindings = load_legacy_exclusions(
            link_report=args.legacy_link_report, link_sha256=args.legacy_link_report_sha256,
            fingerprint_report=args.legacy_original_fingerprint_report,
            fingerprint_sha256=args.legacy_original_fingerprint_report_sha256,
            protected_report=args.protected_report, protected_sha256=args.protected_report_sha256,
            protected=protected["records"], originals=records, original_auditor_sha256=args.original_auditor_sha256,
        )
        ids |= legacy_ids
        existing = {b["path"]: b["sha256"] for b in bindings}
        for binding in legacy_bindings:
            require(binding["path"] not in existing or existing[binding["path"]] == binding["sha256"], "conflicting cohort metadata pin")
            if binding["path"] not in existing:
                bindings.append(binding)
                existing[binding["path"]] = binding["sha256"]
    planned = plan_records(records, references, ids,
                           quarantine_annotation_conflicts=args.quarantine_annotation_conflicts)
    require(bool(planned["records"]), "no eligible original sources")
    require(len(planned["counts"]["accepted_by_split_class"]) == 18, "accepted cohort lacks an official split/class")
    for binding in bindings:
        require(original.digest_file(Path(binding["path"])) == binding["sha256"], "input metadata changed during planning")
    report = {"schema": "aihub_original_cohort_v1", "status": "cohort_planned", **planned,
              "full_cohort": True,
              "metadata_bindings": bindings, "protected_original_ids": sorted(ids), "protected_hardware_stems": unmapped,
              "original_audit_quarantined": len(quarantined), "phash_distance": 4,
              "policy": "exclude both sides of cross-split exact/near duplicates; all protected matches; same-split exact keep lexical first",
              "snapshot_only": True, "consumer_must_rehash_source_and_annotation": True,
              "protected_identity_scope": {
                  "commercial_original_ids_bound": commercial_id_count,
                  "legacy_v2_references_without_original_id": sum(
                      str(decode_path(r["source_path_b64"])).startswith('/app/yolo_dataset_9class_v2/')
                      for r in protected["records"]),
                  "complete_original_lineage": False,
                  "note": "Legacy references have exact SHA and pHash protection only; reencoding identity and materialized-image leakage need separate verification."},
              "pending_checks": ["legacy_transformation_identity", "materialized_image_leakage", "raw_proposal_replay", "independent_hardware_gate"],
              "training_authorized": False, "deployment_authorized": False}
    if args.quarantine_annotation_conflicts:
        report["policy"] += "; quarantine every member of same-SHA class/bbox conflicts; image-evidence conflicts fail"
    if legacy_summary is not None:
        report["legacy_exclusion_evidence"] = legacy_summary
        report["protected_identity_scope"]["legacy_v2_references_without_original_id"] = (
            legacy_summary["unresolved_legacy_references"] + legacy_summary["verified_links_without_original_pool_id"])
        report["protected_identity_scope"]["note"] = "Verified legacy source links extend exact SHA/pHash and original-ID exclusions; unresolved and transformation aliases remain explicit."
        report["pending_checks"] = ["legacy_transform_aliases", "materialized_image_leakage", "raw_proposal_replay", "independent_hardware_gate"]
        if legacy_summary["unresolved_legacy_references"]:
            report["pending_checks"].insert(0, "unresolved_legacy_source_links")
        output = args.output.absolute()
        require(not any(p.is_symlink() for p in (output, *output.parents))
                and ".." not in output.parts, "unsafe legacy cohort output")
        require(not any(output.is_relative_to(p.parent.resolve()) for p in
                        (args.original_report, args.protected_report, args.legacy_link_report, args.legacy_original_fingerprint_report)),
                "legacy cohort output overlaps immutable input evidence")
        recheck_legacy_bindings(legacy_bindings, (args.legacy_link_report, args.legacy_original_fingerprint_report))
    report_bytes = (json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    args.output.mkdir(parents=True, exist_ok=False)
    cohort_path = args.output / "cohort.json"
    if legacy_summary is None:
        with cohort_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
    else:
        with cohort_path.open("xb") as handle:
            handle.write(report_bytes)
    if legacy_summary is not None:
        try:
            recheck_legacy_bindings(legacy_bindings, (args.legacy_link_report, args.legacy_original_fingerprint_report))
        except BaseException:
            if cohort_path.is_file() and not cohort_path.is_symlink() and cohort_path.read_bytes() == report_bytes:
                cohort_path.unlink()  # Only this run's exact invalid publication.
            with (args.output / "failed.json").open("x", encoding="utf-8") as handle:
                json.dump({"status": "failed", "training_authorized": False, "deployment_authorized": False}, handle)
            raise
    print(json.dumps({"status": report["status"], "counts": report["counts"], "training_authorized": False}), flush=True)


if __name__ == '__main__':
    main()
