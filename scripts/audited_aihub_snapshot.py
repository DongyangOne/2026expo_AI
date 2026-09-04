"""Read original-to-materialized AIHub evidence without granting training authority.

The replay source remains the resized JPEG. Original identities are additional
provenance only, never replacements for its SHA or invented physical groups.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

try:
    from scripts import materialize_audited_aihub_sources as materializer
except ModuleNotFoundError:
    import materialize_audited_aihub_sources as materializer

METADATA_FIELDS = (
    "original_source_id", "original_source_sha256", "original_annotation_sha256",
    "original_source_path_b64", "original_annotation_path_b64", "materializer_report_sha256",
)
_SIZE_LIMIT = 1024**3


def _require(value, message):
    if not value:
        raise ValueError(message)


def _equal(a, b):
    # bool/int and int/float are not interchangeable in an evidence document.
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _integer(value, field):
    _require(type(value) is int and value >= 0, f"invalid {field}")
    return value


def _no_authority(value, *, blind=True):
    fields = ["training_authorized", "deployment_authorized"]
    if blind:
        fields.append("blind_test_authorized")
    _require(isinstance(value, dict) and all(value.get(k) is False for k in fields),
             "snapshot must not grant training, deployment or blind authority")


def _tree(root):
    root = materializer._path(root)
    files = set()
    for dirname in ("images", "labels"):
        folder = root / dirname
        if not folder.exists():
            continue
        materializer._path(folder)
        _require(folder.is_dir(), "snapshot image/label directory is invalid")
        for path in folder.rglob("*"):
            path = materializer._path(path)
            _require(path.is_relative_to(root), "snapshot file escapes root")
            if path.is_file():
                files.add(path)
            else:
                _require(path.is_dir(), "non-regular snapshot entry")
    return files


class AuditedAIHubSnapshot:
    """Pinned image linkage evidence; a load or lookup is not training approval."""

    def __init__(self, *, report_path, report_sha256, cohort_path, cohort_sha256,
                 require_full_cohort, bindings, metadata, splits, files, failure_paths):
        self._report_path = report_path
        self._report_sha256 = report_sha256
        self._cohort_path = cohort_path
        self._cohort_sha256 = cohort_sha256
        self._require_full_cohort = require_full_cohort
        self._bindings = dict(bindings)
        self._metadata = copy.deepcopy(metadata)
        self._splits = {key: frozenset(paths) for key, paths in splits.items()}
        self._files = frozenset(files)
        self._failure_paths = tuple(failure_paths)

    def binding(self) -> dict:
        return {"report_path": self._report_path.as_posix(), "report_sha256": self._report_sha256,
                "cohort_path": self._cohort_path.as_posix(), "cohort_sha256": self._cohort_sha256,
                "require_full_cohort": self._require_full_cohort}

    def metadata_for(self, image_path: Path) -> dict[str, str]:
        path = materializer._path(image_path)
        _require(path in self._metadata, "image is not in the audited AIHub snapshot")
        return dict(self._metadata[path])

    def split_for(self, image_path: Path) -> str:
        path = materializer._path(image_path)
        matches = [split for split, paths in self._splits.items() if path in paths]
        _require(path in self._metadata and len(matches) == 1,
                 "image has no unique audited official split")
        return matches[0]

    def assert_source_membership(self, split_images: Mapping[str, Sequence[Path]]) -> None:
        _require(isinstance(split_images, Mapping) and set(split_images) == set(self._splits),
                 "official training/validation image lists required")
        for split, expected in self._splits.items():
            paths = [materializer._path(path) for path in split_images[split]]
            _require(len(paths) == len(set(paths)), "duplicate audited source image")
            _require(set(paths) == expected, "source list differs from exact audited official split")

    def recheck(self) -> None:
        for path in self._failure_paths:
            _require(not path.exists() and not path.is_symlink(), "audited snapshot failure marker exists")
        for path, expected in self._bindings.items():
            _require(materializer._digest(path) == expected, "audited snapshot input changed")
        _require(_tree(self._report_path.parent) == set(self._files), "audited image/label membership changed")


def load_audited_aihub_snapshot(report_path: Path, report_sha256: str, *, cohort_path: Path,
                               require_full_cohort: bool = True) -> AuditedAIHubSnapshot:
    _require(type(require_full_cohort) is bool, "require_full_cohort must be boolean")
    report_path, cohort_path = materializer._path(report_path), materializer._path(cohort_path)
    root = report_path.parent
    _require(report_path.name == "report.json", "materializer report.json required")
    pins = {}

    def pin(path, expected=None, *, content=False, limit=_SIZE_LIMIT):
        path = materializer._path(path)
        if content:
            raw = materializer.audit.read_stable(path, limit)
            actual = hashlib.sha256(raw).hexdigest()
        else:
            actual, raw = materializer._digest(path), None
        if expected is not None:
            _require(actual == materializer._sha(expected), "audited snapshot SHA256 mismatch")
        _require(path not in pins or pins[path] == actual, "audited input changed between reads")
        pins[path] = actual
        return raw

    for module in (Path(__file__), Path(materializer.__file__), Path(materializer.audit.__file__)):
        pin(module.resolve())
    report = materializer._parse(pin(report_path, report_sha256, content=True))
    _no_authority(report)
    _require(report.get("schema") == "audited_aihub_source_snapshot_v1"
             and report.get("status") == "snapshot_complete" and report.get("snapshot_only") is True,
             "invalid materialized AIHub snapshot")
    _require(_equal(report.get("quality_policy"), materializer.POLICY), "materialization quality policy mismatch")
    failures = [root / "failed.json", cohort_path.parent / "failed.json"]
    _require(all(not p.exists() and not p.is_symlink() for p in failures), "audited snapshot failure marker exists")
    ready = materializer._parse(pin(root / "snapshot_ready.json", content=True))
    _no_authority(ready)
    _require(ready.get("status") == "snapshot_complete" and ready.get("snapshot_only") is True
             and ready.get("report_sha256") == report_sha256, "materializer ready/report binding mismatch")
    cohort_sha = materializer._sha(report.get("cohort_sha256"))
    cohort = materializer._parse(pin(cohort_path, cohort_sha, content=True))
    _no_authority(cohort, blind=False)
    _require(cohort.get("schema") == "aihub_original_cohort_v1" and cohort.get("status") == "cohort_planned",
             "invalid original cohort")
    metadata_bindings = report.get("metadata_bindings")
    _require(isinstance(metadata_bindings, list) and metadata_bindings
             and _equal(metadata_bindings, cohort.get("metadata_bindings")), "cohort metadata bindings mismatch")
    seen_metadata = set()
    for item in metadata_bindings:
        _require(isinstance(item, dict) and set(item) == {"path", "sha256"}, "invalid cohort metadata entry")
        path = materializer._path(item["path"])
        _require(path not in seen_metadata and path not in {report_path, cohort_path}, "duplicate/recursive metadata binding")
        seen_metadata.add(path)
        pin(path, item["sha256"])
    rows = cohort.get("records")
    _require(isinstance(rows, list) and rows, "empty original cohort")
    max_sources = _integer(report.get("requested_max_sources"), "requested_max_sources")
    selected = rows[:max_sources] if max_sources else rows
    _require(_integer(report.get("cohort_records"), "cohort_records") == len(rows)
             and _integer(report.get("verified_sources"), "verified_sources") == len(selected)
             and _integer(report.get("unprocessed_sources"), "unprocessed_sources") == len(rows) - len(selected),
             "materializer/cohort coverage mismatch")
    _require(type(report.get("full_cohort")) is bool
             and report["full_cohort"] == (cohort.get("full_cohort") is True and not max_sources),
             "materializer full-cohort claim mismatch")
    if require_full_cohort:
        _require(report["full_cohort"] is True and len(selected) == len(rows),
                 "partial cohort requires explicit diagnostic-only opt-in")
    expected_pending = cohort.get("pending_checks", ["source_coverage_and_leakage_not_proven"])
    _require(_equal(report.get("pending_checks"), expected_pending), "pending checks changed")

    def json_lines(path, digest):
        raw = pin(path, digest, content=True)
        return [materializer._parse(line) for line in raw.splitlines() if line.strip()]

    accepted = json_lines(root / "lineage.jsonl", report.get("lineage_sha256"))
    excluded = json_lines(root / "excluded.jsonl", report.get("excluded_sha256"))
    _require(_integer(report.get("materialized_sources"), "materialized_sources") == len(accepted)
             and _integer(report.get("quality_excluded_sources"), "quality_excluded_sources") == len(excluded)
             and len(accepted) + len(excluded) == len(selected), "materializer result coverage mismatch")
    cohort_by_id, seen_shas, seen_paths = {}, set(), set()
    for row in rows:
        # Path layout/class/bbox validation is the same as the source materializer;
        # actual original and JSON bytes below independently reproduce each output.
        source = materializer._decode(row.get("source_path_b64"), None)
        original_root = source.parent.parent.parent.parent
        source, annotation = materializer._validated_row(row, original_root)
        sid = row["source_id"]
        _require(sid not in cohort_by_id and row["source_sha256"] not in seen_shas and source not in seen_paths,
                 "duplicate original cohort identity")
        cohort_by_id[sid] = (row, source, annotation)
        seen_shas.add(row["source_sha256"])
        seen_paths.add(source)
    selected_ids = {row["source_id"] for row in selected}
    accepted_by_id, excluded_by_id = {}, {}
    for target, values in ((accepted_by_id, accepted), (excluded_by_id, excluded)):
        for row in values:
            _require(isinstance(row, dict) and row.get("source_id") in selected_ids
                     and row["source_id"] not in target, "duplicate or extra materializer result")
            target[row["source_id"]] = row
    _require(not (set(accepted_by_id) & set(excluded_by_id))
             and set(accepted_by_id) | set(excluded_by_id) == selected_ids, "materializer membership mismatch")
    metadata, splits, expected_files = {}, {"training": set(), "validation": set()}, set()
    counts, reason_counts = Counter(), Counter()
    for original_row in selected:
        sid = original_row["source_id"]
        original_row, source, annotation = cohort_by_id[sid]
        pin(source, original_row["source_sha256"])
        pin(annotation, original_row["label_sha256"])
        jpeg, yolo, measures, reason, shape = materializer._prepare(original_row, source, annotation)
        if sid in excluded_by_id:
            row = excluded_by_id[sid]
            _require(reason is not None and row.get("reason") == reason and _equal(row.get("quality"), measures)
                     and row.get("source_sha256") == original_row["source_sha256"]
                     and row.get("label_sha256") == original_row["label_sha256"], "quality exclusion does not reproduce")
            reason_counts[reason] += 1
            continue
        row = accepted_by_id[sid]
        _no_authority(row)
        _require(reason is None and row.get("annotation_authority") == "original_aihub_json"
                 and all(key in row and _equal(row[key], value) for key, value in original_row.items()),
                 "materialized lineage is not an exact original cohort member")
        split = "train" if original_row["split"] == "training" else "val"
        stem = f"{original_row['class_name']}_{sid}"
        image_ref, label_ref = f"images/{split}/{stem}.jpg", f"labels/{split}/{stem}.txt"
        _require(row.get("image_ref") == image_ref and row.get("label_ref") == label_ref,
                 "materialized image/label official split or identity mismatch")
        image_path, label_path = root / image_ref, root / label_ref
        _require((row.get("materialized_height"), row.get("materialized_width")) == shape
                 and all(type(row.get(k)) is int for k in ("materialized_height", "materialized_width"))
                 and _equal(row.get("quality"), measures), "materialized dimensions/quality mismatch")
        image_bytes = pin(image_path, row.get("image_sha256"), content=True, limit=64 * 1024**2)
        label_bytes = pin(label_path, row.get("yolo_label_sha256"), content=True, limit=1024**2)
        _require(image_bytes == jpeg and label_bytes == yolo, "original-to-JPEG/YOLO reproduction mismatch")
        expected_files.update((image_path, label_path))
        splits[original_row["split"]].add(image_path)
        counts[f"{original_row['split']}/{original_row['class_name']}"] += 1
        metadata[image_path] = dict(zip(METADATA_FIELDS, (
            sid, original_row["source_sha256"], original_row["label_sha256"],
            original_row["source_path_b64"], original_row["label_path_b64"], report_sha256,
        )))
    _require(_equal(report.get("counts"), dict(counts)) and _equal(report.get("exclusions"), dict(reason_counts)),
             "materializer counts differ from verified membership")
    yaml = f"path: {json.dumps(root.as_posix(), ensure_ascii=True)}\ntrain: images/train\nval: images/val\nnames:\n"
    yaml += "".join(f"  {i}: {name}\n" for i, name in enumerate(materializer.audit.CLASS_NAMES))
    _require(pin(root / "dataset.yaml", content=True) == yaml.encode(), "materializer YAML mismatch")
    snapshot = AuditedAIHubSnapshot(report_path=report_path, report_sha256=report_sha256,
        cohort_path=cohort_path, cohort_sha256=cohort_sha, require_full_cohort=require_full_cohort,
        bindings=pins, metadata=metadata, splits=splits, files=expected_files, failure_paths=failures)
    snapshot.recheck()
    return snapshot
