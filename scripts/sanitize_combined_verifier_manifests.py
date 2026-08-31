"""Resolve cross-manifest leakage without changing any source manifest.

This sanitizer is intentionally evidence-aware instead of using a broad
"first input wins" rule.  It writes adjacent, immutable copies of the proposal,
hardware, and operational manifests and applies only these deterministic
precedence rules:

* model-validation beats train for source/object/session partition collisions;
* hardware provenance beats proposal provenance for exact sample/image copies;
* ground-truth-derived rows beat pseudo-labelled rows for near-pHash label
  conflicts;
* otherwise role precedence, provenance precedence, then a stable source key
  break a cross-partition tie.

Retained rows are copied field-for-field.  In particular, ``role``, ``fold``,
``split``, and relative ``filepath`` text are never rewritten.  Outputs must be
adjacent to inputs so those relative paths keep the same anchor.  Complete
temporary files are published with exclusive hard links and the hash-bound
report is published last.  There is deliberately no overwrite option.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np


KINDS = ("proposal", "hardware", "operational")
KIND_PRIORITY = {"operational": 1, "proposal": 2, "hardware": 3}
ROLE_PRIORITY = {
    "train": 1,
    "model_validation": 2,
    "calibration": 3,
    "blind_test": 4,
}
ROLE_TO_SPLIT = {"train": "training", "model_validation": "validation"}
TRAINING_ROLES = set(ROLE_TO_SPLIT)
MATERIAL_CLASS_NAMES = (
    "can", "pet", "paper", "plastic", "styrofoam",
    "vinyl", "glass", "battery", "fluorescent", "background",
)
REQUIRED_FIELDS = {
    "filepath", "split", "source_id", "material", "category",
    "dent", "label", "foreign_material", "source_object_count",
    "sample_id", "role", "fold", "source_sha256", "image_sha256",
    "object_group", "capture_session", "origin",
}
PARTITION_GUARD_FIELDS = ("source_sha256", "object_group", "capture_session")
EXACT_GUARD_FIELDS = ("sample_id", "image_sha256")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_REPORT_MATCHES = 8


@dataclass(frozen=True)
class ManifestTarget:
    kind: str
    input_path: Path
    output_path: Path


@dataclass(frozen=True)
class _LoadedManifest:
    target: ManifestTarget
    raw_bytes: bytes
    sha256: str
    fields: tuple[str, ...]
    rows: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class _Candidate:
    index: int
    kind: str
    manifest: Path
    line: int
    row: Mapping[str, str]
    image_path: Path

    @property
    def sample_id(self) -> str:
        return self.row["sample_id"].strip()

    @property
    def role(self) -> str:
        return self.row["role"].strip()

    @property
    def fold(self) -> str:
        return self.row["fold"].strip()

    @property
    def partition(self) -> tuple[str, str]:
        return self.role, self.fold

    @property
    def material(self) -> int:
        return int(self.row["material"])

    @property
    def pseudo_label(self) -> bool:
        raw = str(self.row.get("pseudo_label", "")).strip().casefold()
        return raw in {"1", "true", "yes", "y"} or self.kind == "operational"

    @property
    def stable_key(self) -> tuple[str, str, int, str]:
        return self.kind, self.manifest.as_posix(), self.line, self.sample_id

    def ref(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "manifest": self.manifest.as_posix(),
            "line": self.line,
            "sample_id": self.sample_id,
            "role": self.role,
            "fold": self.fold,
            "material": self.material,
        }


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object, *, pretty: bool = False) -> bytes:
    options = {
        "ensure_ascii": False,
        "sort_keys": True,
        "allow_nan": False,
    }
    if pretty:
        rendered = json.dumps(value, indent=2, **options)
    else:
        rendered = json.dumps(value, separators=(",", ":"), **options)
    return (rendered + "\n").encode("utf-8")


def _identity(field: str, value: str) -> str:
    stripped = value.strip()
    if field in {"source_sha256", "image_sha256"}:
        return stripped.lower()
    return stripped.casefold()


def _resolve_image(manifest: Path, value: str, line: int) -> Path:
    raw = value.strip()
    if not raw:
        raise ValueError(f"{manifest}:{line}: filepath is empty")
    path = Path(raw)
    if not path.is_absolute():
        path = manifest.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{manifest}:{line}: image does not exist: {path}")
    return path


def _validate_row(
    row: Mapping[str, str],
    *,
    manifest: Path,
    line: int,
    image_hash_cache: dict[Path, str],
) -> Path:
    location = f"{manifest}:{line}"
    for field in REQUIRED_FIELDS:
        if not str(row.get(field, "")).strip():
            raise ValueError(f"{location}: missing required {field}")
    role = row["role"].strip()
    if role not in TRAINING_ROLES:
        raise ValueError(
            f"{location}: role {role!r} is not a training role; calibration/blind_test "
            "evidence must never enter combined training manifests"
        )
    expected_split = ROLE_TO_SPLIT.get(role)
    if expected_split and row["split"].strip().casefold() != expected_split:
        raise ValueError(
            f"{location}: split {row['split']!r} conflicts with role {role!r}"
        )
    try:
        material = int(row["material"])
        object_count = int(row["source_object_count"])
        conditions = [int(row[name]) for name in ("dent", "label", "foreign_material")]
    except ValueError as error:
        raise ValueError(f"{location}: invalid integer label") from error
    if material not in range(len(MATERIAL_CLASS_NAMES)):
        raise ValueError(f"{location}: material must be between 0 and 9")
    if row["category"].strip() != MATERIAL_CLASS_NAMES[material]:
        raise ValueError(f"{location}: category does not match material {material}")
    if object_count not in {0, 1}:
        raise ValueError(
            f"{location}: source_object_count must be 0 or 1"
        )
    expected_crop_count = 0 if material == 9 else 1
    raw_crop_count = str(row.get("crop_object_count", "")).strip()
    if raw_crop_count:
        try:
            crop_count = int(raw_crop_count)
        except ValueError as error:
            raise ValueError(
                f"{location}: crop_object_count must be an integer"
            ) from error
    else:
        # Backward compatibility is limited to the two unambiguous legacy
        # contracts.  A source=1 background needs the explicit crop=0 field.
        crop_count = expected_crop_count
        if object_count != expected_crop_count:
            raise ValueError(
                f"{location}: crop_object_count is required when "
                f"source_object_count={object_count} for material {material}"
            )
    if crop_count != expected_crop_count:
        raise ValueError(
            f"{location}: crop_object_count must be {expected_crop_count} "
            f"for material {material}"
        )
    if crop_count > object_count:
        raise ValueError(
            f"{location}: crop_object_count cannot exceed source_object_count"
        )
    if any(value not in {-1, 0, 1} for value in conditions):
        raise ValueError(f"{location}: condition labels must be -1, 0, or 1")
    for field in ("source_sha256", "image_sha256"):
        if not SHA256_RE.fullmatch(row[field].strip()):
            raise ValueError(f"{location}: {field} must be lowercase SHA-256")
    image_path = _resolve_image(manifest, row["filepath"], line)
    actual = image_hash_cache.get(image_path)
    if actual is None:
        actual = _sha256_file(image_path)
        image_hash_cache[image_path] = actual
    if actual != row["image_sha256"].strip():
        raise ValueError(f"{location}: image_sha256 does not match image content")
    return image_path


def _load_manifest(
    target: ManifestTarget,
    *,
    image_hash_cache: dict[Path, str],
    start_index: int,
) -> tuple[_LoadedManifest, list[_Candidate]]:
    path = target.input_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {path}")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"manifest is not UTF-8: {path}") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fields = tuple(reader.fieldnames or ())
    if not fields:
        raise ValueError(f"manifest has no CSV header: {path}")
    if any(not field for field in fields) or len(fields) != len(set(fields)):
        raise ValueError(f"manifest has empty or duplicate CSV columns: {path}")
    missing = sorted(REQUIRED_FIELDS - set(fields))
    if missing:
        raise ValueError(f"manifest is missing strict fields {missing}: {path}")
    rows: list[dict[str, str]] = []
    candidates: list[_Candidate] = []
    for offset, raw_row in enumerate(reader):
        line = offset + 2
        if None in raw_row:
            raise ValueError(f"{path}:{line}: unnamed extra CSV column")
        row = {
            field: "" if raw_row.get(field) is None else str(raw_row[field])
            for field in fields
        }
        image_path = _validate_row(
            row,
            manifest=path,
            line=line,
            image_hash_cache=image_hash_cache,
        )
        rows.append(row)
        candidates.append(
            _Candidate(
                index=start_index + offset,
                kind=target.kind,
                manifest=path,
                line=line,
                row=row,
                image_path=image_path,
            )
        )
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return (
        _LoadedManifest(
            target=ManifestTarget(target.kind, path, target.output_path.resolve(strict=False)),
            raw_bytes=raw,
            sha256=_sha256_bytes(raw),
            fields=fields,
            rows=tuple(rows),
        ),
        candidates,
    )


def _stable_winner(candidates: Sequence[_Candidate]) -> _Candidate:
    """Provenance first, then role; final tie uses ascending stable key."""
    best_score = max(
        (KIND_PRIORITY[item.kind], ROLE_PRIORITY[item.role]) for item in candidates
    )
    return min(
        (
            item
            for item in candidates
            if (KIND_PRIORITY[item.kind], ROLE_PRIORITY[item.role]) == best_score
        ),
        key=lambda item: item.stable_key,
    )


def _record_drop(
    dropped: dict[int, dict[str, object]],
    *,
    loser: _Candidate,
    winner: _Candidate,
    reason: str,
    field: str | None = None,
    value: str | None = None,
    distance: int | None = None,
) -> None:
    if loser.index in dropped:
        return
    record: dict[str, object] = {
        "reason": reason,
        "loser": loser.ref(),
        "winner": winner.ref(),
        "_winner_index": winner.index,
    }
    if field is not None:
        record["field"] = field
    if value is not None:
        record["value"] = value
    if distance is not None:
        record["phash_distance"] = distance
    dropped[loser.index] = record


def _resolve_exact_and_partition_conflicts(
    candidates: Sequence[_Candidate],
) -> dict[int, dict[str, object]]:
    dropped: dict[int, dict[str, object]] = {}

    # Exact guards form a graph: a row can share sample_id with one input and
    # image_sha256 with another.  Resolve each connected component once so a
    # reported winner can never itself be dropped by a later guard pass.
    parent = {candidate.index: candidate.index for candidate in candidates}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    exact_groups: list[tuple[str, str, list[_Candidate]]] = []
    for field in EXACT_GUARD_FIELDS:
        groups: dict[str, list[_Candidate]] = defaultdict(list)
        for candidate in candidates:
            groups[_identity(field, candidate.row[field])].append(candidate)
        for value in sorted(groups):
            group = groups[value]
            if len(group) < 2:
                continue
            manifests = [candidate.manifest for candidate in group]
            if len(manifests) != len(set(manifests)):
                raise ValueError(
                    f"within-manifest exact duplicate {field}={value!r}; "
                    "source manifests must be strict before combined sanitation"
                )
            exact_groups.append((field, value, group))
            anchor = group[0].index
            for candidate in group[1:]:
                union(anchor, candidate.index)

    components: dict[int, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        components[find(candidate.index)].append(candidate)
    component_guards: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for field, value, group in exact_groups:
        component_guards[find(group[0].index)].append((field, value))
    for root in sorted(components):
        component = components[root]
        if len(component) < 2:
            continue
        winner = _stable_winner(component)
        guards = sorted(set(component_guards[root]))
        for loser in sorted(component, key=lambda item: item.stable_key):
            if loser.index == winner.index:
                continue
            direct = [
                (field, value)
                for field, value in guards
                if _identity(field, loser.row[field]) == value
            ]
            _record_drop(
                dropped,
                loser=loser,
                winner=winner,
                reason="exact_identity_precedence",
                field=",".join(field for field, _ in direct) or "connected_component",
                value="|".join(value for _, value in direct) or None,
            )

    # For physical source/object/session leakage, the validation partition is
    # authoritative over train.  Multiple views inside the winning partition
    # are retained; no role/fold text is rewritten.
    for field in PARTITION_GUARD_FIELDS:
        groups: dict[str, list[_Candidate]] = defaultdict(list)
        for candidate in candidates:
            if candidate.index not in dropped:
                groups[_identity(field, candidate.row[field])].append(candidate)
        for value in sorted(groups):
            group = groups[value]
            partitions: dict[tuple[str, str], list[_Candidate]] = defaultdict(list)
            for candidate in group:
                partitions[candidate.partition].append(candidate)
            if len(partitions) < 2:
                continue
            partitions_by_manifest: dict[Path, set[tuple[str, str]]] = defaultdict(set)
            for candidate in group:
                partitions_by_manifest[candidate.manifest].add(candidate.partition)
            corrupt = [
                manifest
                for manifest, manifest_partitions in partitions_by_manifest.items()
                if len(manifest_partitions) > 1
            ]
            if corrupt:
                raise ValueError(
                    f"within-manifest partition leakage for {field}={value!r}: "
                    f"{[path.as_posix() for path in sorted(corrupt)]}"
                )
            if len(partitions_by_manifest) < 2:
                raise ValueError(
                    f"within-manifest partition leakage for {field}={value!r}"
                )

            def partition_score(
                item: tuple[tuple[str, str], list[_Candidate]],
            ) -> tuple[int, int, str, str]:
                partition, rows = item
                return (
                    ROLE_PRIORITY[partition[0]],
                    max(KIND_PRIORITY[row.kind] for row in rows),
                    partition[0],
                    partition[1],
                )

            winning_partition, winners = max(partitions.items(), key=partition_score)
            representative = _stable_winner(winners)
            for partition, rows in sorted(partitions.items()):
                if partition == winning_partition:
                    continue
                for loser in sorted(rows, key=lambda item: item.stable_key):
                    _record_drop(
                        dropped,
                        loser=loser,
                        winner=representative,
                        reason="validation_partition_precedence",
                        field=field,
                        value=value,
                    )
    return dropped


def _perceptual_hash(path: Path) -> int:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot decode image for pHash: {path}")
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(np.float32(resized))[:8, :8].reshape(-1)
    median = float(np.median(coefficients[1:]))
    bits = coefficients > median
    bits[0] = False
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def _bucket_keys(value: int, threshold: int) -> tuple[tuple[int, int], ...]:
    widths = [64 // (threshold + 1)] * (threshold + 1)
    for index in range(64 % (threshold + 1)):
        widths[index] += 1
    keys = []
    offset = 0
    for index, width in enumerate(widths):
        keys.append((index, (value >> offset) & ((1 << width) - 1)))
        offset += width
    return tuple(keys)


def _pair_winner(left: _Candidate, right: _Candidate) -> tuple[_Candidate, str]:
    if left.material != right.material and left.pseudo_label != right.pseudo_label:
        return (right, "ground_truth_over_pseudo") if left.pseudo_label else (
            left,
            "ground_truth_over_pseudo",
        )
    if left.partition != right.partition and left.role != right.role:
        if ROLE_PRIORITY[left.role] != ROLE_PRIORITY[right.role]:
            return (
                (left, "validation_partition_precedence")
                if ROLE_PRIORITY[left.role] > ROLE_PRIORITY[right.role]
                else (right, "validation_partition_precedence")
            )
    return _stable_winner((left, right)), "stable_cross_partition_precedence"


def _near_phash_edges(
    candidates: Sequence[_Candidate],
    *,
    threshold: int,
) -> list[tuple[int, int, int]]:
    records: dict[int, list[_Candidate]] = defaultdict(list)
    buckets: dict[tuple[int, int], set[int]] = defaultdict(set)
    cache: dict[Path, int] = {}
    edges: set[tuple[int, int, int]] = set()
    for current in sorted(candidates, key=lambda item: item.stable_key):
        current_hash = cache.get(current.image_path)
        if current_hash is None:
            current_hash = _perceptual_hash(current.image_path)
            cache[current.image_path] = current_hash
        candidate_hashes: set[int] = set()
        for key in _bucket_keys(current_hash, threshold):
            candidate_hashes.update(buckets.get(key, ()))
        candidate_hashes.add(current_hash)
        for other_hash in candidate_hashes:
            distance = int((current_hash ^ other_hash).bit_count())
            if distance > threshold:
                continue
            for other in records.get(other_hash, ()):
                if other.row["image_sha256"] == current.row["image_sha256"]:
                    continue
                if other.partition == current.partition and other.material == current.material:
                    continue
                low, high = sorted((other.index, current.index))
                edges.add((distance, low, high))
        records[current_hash].append(current)
        for key in _bucket_keys(current_hash, threshold):
            buckets[key].add(current_hash)
    return sorted(edges)


def _resolve_near_phash_conflicts(
    candidates: Sequence[_Candidate],
    *,
    threshold: int,
    dropped: dict[int, dict[str, object]],
) -> None:
    by_index = {candidate.index: candidate for candidate in candidates}
    active = [candidate for candidate in candidates if candidate.index not in dropped]
    edges = _near_phash_edges(active, threshold=threshold)
    # Label conflicts are resolved before same-label partition leakage.
    edges.sort(
        key=lambda edge: (
            by_index[edge[1]].material == by_index[edge[2]].material,
            edge[0],
            by_index[edge[1]].stable_key,
            by_index[edge[2]].stable_key,
        )
    )
    # Source partition leakage is never a sanitation decision.  Inspect every
    # cross-partition edge before greedy cross-manifest resolution so a row
    # dropped by an earlier edge cannot mask it.  A same-manifest,
    # same-partition label difference is valid hard data: coarse 64-bit pHash
    # can collide for genuinely different objects (for example can vs PET), so
    # that case must not be quarantined or rejected.
    for distance, left_index, right_index in edges:
        left, right = by_index[left_index], by_index[right_index]
        if left.manifest == right.manifest and left.partition != right.partition:
            raise ValueError(
                "within-manifest cross-partition near-pHash conflict: "
                f"{left.manifest}:{left.line} and {right.manifest}:{right.line}, "
                f"distance={distance}"
            )
    for distance, left_index, right_index in edges:
        if left_index in dropped or right_index in dropped:
            continue
        left, right = by_index[left_index], by_index[right_index]
        if left.manifest == right.manifest:
            continue
        winner, reason = _pair_winner(left, right)
        loser = right if winner.index == left.index else left
        _record_drop(
            dropped,
            loser=loser,
            winner=winner,
            reason=f"near_phash_{reason}",
            distance=distance,
        )


def _render_csv(fields: Sequence[str], rows: Sequence[Mapping[str, str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return output.getvalue().encode("utf-8")


def _validate_targets(
    targets: Sequence[ManifestTarget], report_json: Path, *, dry_run: bool
) -> tuple[list[ManifestTarget], Path]:
    if [target.kind for target in targets] != list(KINDS):
        raise ValueError(f"targets must be ordered exactly as {KINDS}")
    inputs: set[Path] = set()
    outputs: set[Path] = set()
    normalized = []
    for target in targets:
        source = target.input_path.resolve()
        output = target.output_path.resolve(strict=False)
        if source.suffix.lower() != ".csv" or output.suffix.lower() != ".csv":
            raise ValueError("all manifest inputs and outputs must use .csv")
        if source == output:
            raise ValueError("sanitized output must not replace its source manifest")
        if source.parent != output.parent:
            raise ValueError(
                "each output must be adjacent to its input so relative filepath semantics remain unchanged"
            )
        if source in inputs or output in outputs:
            raise ValueError("manifest input/output paths must be unique")
        inputs.add(source)
        outputs.add(output)
        normalized.append(ManifestTarget(target.kind, source, output))
    overlap = sorted(inputs & outputs, key=lambda path: path.as_posix())
    if overlap:
        raise ValueError(f"sanitized outputs must not alias any source manifest: {overlap}")
    report = report_json.resolve(strict=False)
    if report.suffix.lower() != ".json":
        raise ValueError("report_json must use .json")
    if report in inputs or report in outputs:
        raise ValueError("report path must be distinct from manifest paths")
    if not dry_run:
        existing = sorted(
            (path for path in (*outputs, report) if path.exists()),
            key=lambda path: path.as_posix(),
        )
        if existing:
            raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")
    return normalized, report


def _write_temp(target: Path, content: bytes) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        temporary.chmod(0o644)
        return temporary
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _publish(
    artifacts: Sequence[tuple[Path, bytes]], source_hashes: Mapping[Path, str]
) -> None:
    staged: list[tuple[Path, Path, str]] = []
    published: list[tuple[Path, str]] = []
    try:
        for target, content in artifacts:
            if target.exists():
                raise FileExistsError(f"refusing to overwrite existing output: {target}")
            staged.append((target, _write_temp(target, content), _sha256_bytes(content)))
        for source, digest in source_hashes.items():
            if _sha256_file(source) != digest:
                raise RuntimeError(f"source manifest changed during sanitation: {source}")
        for target, temporary, digest in staged:
            os.link(temporary, target)  # exclusive atomic create; never replaces target
            temporary.unlink()
            published.append((target, digest))
    except BaseException:
        for target, digest in reversed(published):
            try:
                if target.is_file() and _sha256_file(target) == digest:
                    target.unlink()
            except OSError:
                pass
        raise
    finally:
        for _, temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def sanitize_combined_manifests(
    *,
    proposal_manifest: Path,
    proposal_output: Path,
    hardware_manifest: Path,
    hardware_output: Path,
    operational_manifest: Path,
    operational_output: Path,
    report_json: Path,
    phash_distance: int = 4,
    dry_run: bool = False,
) -> dict[str, object]:
    if not 0 <= phash_distance <= 4:
        raise ValueError("phash_distance must be between 0 and 4")
    targets, report_path = _validate_targets(
        [
            ManifestTarget("proposal", proposal_manifest, proposal_output),
            ManifestTarget("hardware", hardware_manifest, hardware_output),
            ManifestTarget("operational", operational_manifest, operational_output),
        ],
        report_json,
        dry_run=dry_run,
    )
    image_hash_cache: dict[Path, str] = {}
    loaded: list[_LoadedManifest] = []
    candidates: list[_Candidate] = []
    for target in targets:
        item, rows = _load_manifest(
            target,
            image_hash_cache=image_hash_cache,
            start_index=len(candidates),
        )
        loaded.append(item)
        candidates.extend(rows)

    dropped = _resolve_exact_and_partition_conflicts(candidates)
    _resolve_near_phash_conflicts(
        candidates,
        threshold=phash_distance,
        dropped=dropped,
    )

    outputs: list[tuple[Path, bytes]] = []
    output_report: list[dict[str, object]] = []
    candidate_by_location = {
        (candidate.kind, candidate.line): candidate for candidate in candidates
    }
    for item in loaded:
        kept_rows = [
            row
            for line, row in enumerate(item.rows, start=2)
            if candidate_by_location[(item.target.kind, line)].index not in dropped
        ]
        content = _render_csv(item.fields, kept_rows)
        outputs.append((item.target.output_path, content))
        output_report.append(
            {
                "kind": item.target.kind,
                "input": {
                    "path": item.target.input_path.as_posix(),
                    "sha256": item.sha256,
                    "rows": len(item.rows),
                },
                "output": {
                    "path": item.target.output_path.as_posix(),
                    "sha256": _sha256_bytes(content),
                    "rows": len(kept_rows),
                },
                "dropped_rows": len(item.rows) - len(kept_rows),
                "retained_row_fields_rewritten": False,
                "relative_filepath_anchor_preserved": True,
            }
        )

    # A winner selected by one guard may itself lose to stronger evidence in a
    # later guard.  Collapse those chains so every persisted report reference
    # names a row that is actually present in the sanitized outputs.
    by_index = {candidate.index: candidate for candidate in candidates}
    for record in dropped.values():
        winner_index = int(record["_winner_index"])
        seen = set()
        while winner_index in dropped:
            if winner_index in seen:
                raise RuntimeError("cyclic sanitation winner chain")
            seen.add(winner_index)
            winner_index = int(dropped[winner_index]["_winner_index"])
        record["winner"] = by_index[winner_index].ref()
        del record["_winner_index"]
    drop_records = sorted(
        dropped.values(),
        key=lambda item: (
            str(item["loser"]["kind"]),
            str(item["loser"]["manifest"]),
            int(item["loser"]["line"]),
            str(item["reason"]),
        ),
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "builder": "scripts/sanitize_combined_verifier_manifests.py",
        "policy": {
            "retained_row_fields_rewritten": False,
            "relative_filepath_policy": "adjacent_output_no_path_rewrite",
            "source_manifests_modified": False,
            "overwrite_allowed": False,
            "partition_guard_fields": list(PARTITION_GUARD_FIELDS),
            "exact_guard_fields": list(EXACT_GUARD_FIELDS),
            "validation_precedence_over_train": True,
            "hardware_precedence_for_exact_duplicates": True,
            "ground_truth_precedence_over_pseudo_label_conflicts": True,
            "phash_distance": phash_distance,
            "phash_scope": "cross_manifest_label_conflict_or_cross_partition",
        },
        "manifests": output_report,
        "dropped": {
            "rows": len(drop_records),
            "reason_counts": dict(
                sorted(Counter(str(item["reason"]) for item in drop_records).items())
            ),
            "records_sha256": _sha256_bytes(_json_bytes(drop_records)),
            "records": drop_records,
        },
        "combined_output_rows": sum(
            int(item["output"]["rows"]) for item in output_report
        ),
        "publication": {
            "atomic_file_publication": True,
            "exclusive_no_overwrite": True,
            "report_published_last": True,
            "dry_run": dry_run,
        },
    }
    report_bytes = _json_bytes(report, pretty=True)
    if not dry_run:
        _publish(
            [*outputs, (report_path, report_bytes)],
            {item.target.input_path: item.sha256 for item in loaded},
        )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sanitize the three v3 verifier manifests before combined audit."
    )
    for kind in KINDS:
        parser.add_argument(f"--{kind}-manifest", required=True, type=Path)
        parser.add_argument(f"--{kind}-output", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--phash-distance", type=int, default=4, metavar="0..4")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = sanitize_combined_manifests(
        proposal_manifest=args.proposal_manifest,
        proposal_output=args.proposal_output,
        hardware_manifest=args.hardware_manifest,
        hardware_output=args.hardware_output,
        operational_manifest=args.operational_manifest,
        operational_output=args.operational_output,
        report_json=args.report_json,
        phash_distance=args.phash_distance,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
