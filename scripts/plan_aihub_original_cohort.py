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
    try:
        require(type(row.get("bbox_xywh")) is list, "invalid source bbox")
        original.strict_bbox([row["bbox_xywh"]], w, h)
    except original.AnnotationError as exc:
        raise PlanError(str(exc)) from exc
    conditions = row.get("conditions")
    require(type(conditions) is dict and set(conditions) == {"dent", "label", "foreign_material"}
            and all(type(v) is int and v == -1 for v in conditions.values()), "unverified state must remain masked")


def plan_records(originals: list[dict], protected: list[dict], protected_source_ids: set[str]) -> dict:
    require(type(originals) is list and bool(originals), "empty original cohort")
    require(type(protected) is list, "invalid protected reference set")
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
    reasons = defaultdict(set)
    for sha, group in by_sha.items():
        signatures = {(r["class_id"], tuple(r["bbox_xywh"]), r["image_width"], r["image_height"], r["source_phash64"]) for r in group}
        require(len(signatures) == 1, "same image SHA has conflicting ground truth or image evidence")
        if len({r["split"] for r in group}) > 1:
            for row in group:
                reasons[row["source_id"]].add("cross_split_duplicate")
        elif len(group) > 1:
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
    return {"records": accepted, "exclusions": excluded, "counts": counts}


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-report", "protected-report", "selected-manifest", "original-auditor"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
    planned = plan_records(records, protected["records"], ids)
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
                  "commercial_original_ids_bound": len(ids),
                  "legacy_v2_references_without_original_id": sum(
                      str(decode_path(r["source_path_b64"])).startswith('/app/yolo_dataset_9class_v2/')
                      for r in protected["records"]),
                  "complete_original_lineage": False,
                  "note": "Legacy references have exact SHA and pHash protection only; reencoding identity and materialized-image leakage need separate verification."},
              "pending_checks": ["legacy_transformation_identity", "materialized_image_leakage", "raw_proposal_replay", "independent_hardware_gate"],
              "training_authorized": False, "deployment_authorized": False}
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "cohort.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "training_authorized": False}), flush=True)


if __name__ == '__main__':
    main()
