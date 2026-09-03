"""Train the v3 proposal verifier without a background material class.

The v2 proposal verifier could add ``background`` as a tenth material class.
That makes objectness and material identity compete in one softmax.  This
training path deliberately gives the two questions different heads:

* ``objectness`` is a two-class head trained on every proposal crop;
* ``material`` is a nine-class head trained only when objectness is positive;
* condition heads are optional and are also trained only on positive crops
  with a non-negative label.

The script consumes strict, pre-partitioned manifests.  It never invents a
random row split: ``role=train`` and ``role=model_validation`` are mandatory,
and source/group/session/content identities may not cross those roles.  This
file is intentionally separate from the production verifier/runtime so a v3
candidate cannot silently change the deployed output contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import stat
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

try:  # ``python -m scripts...`` and pytest imports.
    from scripts.train_verifier import (
        CLASS_NAMES as _LEGACY_CLASS_NAMES,
        CLASS_WEIGHT_MODES,
        DEFAULT_CLASS_WEIGHT_BETA,
        IMAGENET_MEAN,
        IMAGENET_STD,
        _build_backbone as _build_legacy_backbone,
        class_weights,
    )
except ModuleNotFoundError:  # Direct ``python scripts/train_multitask_verifier.py``.
    from train_verifier import (  # type: ignore[no-redef]
        CLASS_NAMES as _LEGACY_CLASS_NAMES,
        CLASS_WEIGHT_MODES,
        DEFAULT_CLASS_WEIGHT_BETA,
        IMAGENET_MEAN,
        IMAGENET_STD,
        _build_backbone as _build_legacy_backbone,
        class_weights,
    )


MATERIAL_CLASS_NAMES = tuple(_LEGACY_CLASS_NAMES)
if len(MATERIAL_CLASS_NAMES) != 9:  # Fail before producing a wrong checkpoint.
    raise RuntimeError("the v3 verifier requires the existing nine material classes")

BACKGROUND_MATERIAL_ID = 9
BACKGROUND_CLASS_NAME = "background"
OBJECTNESS_CLASS_NAMES = (BACKGROUND_CLASS_NAME, "material")
CONDITION_NAMES = ("dent", "label", "foreign_material")
CONDITION_CLASS_NAMES = {
    "dent": ("not_dented", "dented"),
    "label": ("no_label", "has_label"),
    "foreign_material": ("no_foreign_material", "has_foreign_material"),
}
OUTPUT_CONTRACT_VERSION = "multitask_verifier.v3"
IMAGE_CONSUMPTION_CONTRACT_VERSION = "multitask_verifier.image_consumption.v1"
IMAGE_CONSUMPTION_MAX_BYTES = 64 * 1024 * 1024
IMAGE_CONSUMPTION_MAX_PIXELS = 16 * 1024 * 1024
IMAGE_CONSUMPTION_READ_BYTES = 1024 * 1024
TRAIN_ROLE = "train"
VALIDATION_ROLE = "model_validation"
TRAINING_ROLES = (TRAIN_ROLE, VALIDATION_ROLE)
CALIBRATION_ROLE = "calibration"
BLIND_TEST_ROLE = "blind_test"
ALL_ROLES = (*TRAINING_ROLES, CALIBRATION_ROLE, BLIND_TEST_ROLE)
ROLE_TO_SPLIT = {TRAIN_ROLE: "training", VALIDATION_ROLE: "validation"}
LINEAGE_FIELDS = (
    "sample_id",
    "source_sha256",
    "object_group",
    "capture_session",
    "role",
    "fold",
)
REQUIRED_MANIFEST_FIELDS = (
    "filepath",
    "split",
    "source_id",
    "material",
    "category",
    "dent",
    "label",
    "foreign_material",
    "source_object_count",
    "sample_id",
    "role",
    "fold",
    "source_sha256",
    "image_sha256",
    "object_group",
    "capture_session",
    "origin",
)
IMAGE_PATH_FIELDS = ("filepath", "image_path", "crop_path", "path")
HEX_SHA256_LENGTH = 64
CHECKPOINT_NAME = "best_multitask_verifier.pt"
METADATA_NAME = "multitask_verifier_metadata.json"
ONNX_NAME = "multitask_verifier.onnx"


def build_image_consumption_contract(
    *,
    trainer_sha256: str,
    dataset_snapshot_report_sha256: str | None,
    dataset_snapshot_tree_sha256: str | None,
    manifest_payload_set_sha256: str,
) -> dict[str, Any]:
    """Return the exact per-access image-byte contract bound into artifacts."""

    if not _is_sha256(trainer_sha256):
        raise ValueError("trainer SHA must be lowercase SHA-256")
    authority_values = (
        dataset_snapshot_report_sha256,
        dataset_snapshot_tree_sha256,
    )
    if any(value is None for value in authority_values) and not all(
        value is None for value in authority_values
    ):
        raise ValueError("dataset authority SHA values must be supplied together")
    for name, value in (
        ("dataset snapshot report", dataset_snapshot_report_sha256),
        ("dataset snapshot tree", dataset_snapshot_tree_sha256),
    ):
        if value is not None and not _is_sha256(value):
            raise ValueError(f"{name} SHA must be lowercase SHA-256")
    if not _is_sha256(manifest_payload_set_sha256):
        raise ValueError("manifest payload-set SHA must be lowercase SHA-256")
    return {
        "schema": "v4_candidate_dataset_consumption.v1",
        "version": IMAGE_CONSUMPTION_CONTRACT_VERSION,
        "evidence_scope": "per_access_fail_closed_no_complete_access_receipt",
        "authority_platform": (
            "linux_qnap"
            if dataset_snapshot_report_sha256 is not None
            else "unbound_local_validation"
        ),
        "read_semantics": "single_descriptor_fstat_sha256_then_bytesio_decode",
        "trainer_path": "scripts/train_multitask_verifier.py",
        "trainer_sha256": trainer_sha256,
        "dataset_snapshot_report_sha256": dataset_snapshot_report_sha256,
        "dataset_snapshot_tree_sha256": dataset_snapshot_tree_sha256,
        "manifest_payload_set_sha256": manifest_payload_set_sha256,
        "max_image_bytes": IMAGE_CONSUMPTION_MAX_BYTES,
        "max_image_pixels": IMAGE_CONSUMPTION_MAX_PIXELS,
        "complete_access_receipt": False,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns) if os.name != "nt" else 0,
    )


def _lstat_image_chain(root: Path, path: Path) -> list[tuple[Path, tuple[int, ...]]]:
    lexical_root = Path(os.path.abspath(root))
    lexical_path = Path(os.path.abspath(path))
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError as exc:
        raise ValueError(f"image path escapes its manifest directory: {lexical_path}") from exc
    if not relative.parts:
        raise ValueError("image path must be a file beneath its manifest directory")
    result: list[tuple[Path, tuple[int, ...]]] = []
    current = lexical_root
    components = (
        lexical_root,
        *(
            lexical_root / Path(*relative.parts[:index])
            for index in range(1, len(relative.parts) + 1)
        ),
    )
    for index, current in enumerate(components):
        try:
            current_stat = os.lstat(current)
        except OSError as exc:
            raise ValueError(f"image path component cannot be inspected: {current}") from exc
        if stat.S_ISLNK(current_stat.st_mode):
            raise ValueError(f"image path contains a symlink: {current}")
        is_leaf = index == len(components) - 1
        if is_leaf:
            if not stat.S_ISREG(current_stat.st_mode):
                raise ValueError(f"image payload must be a regular file: {current}")
            if current_stat.st_nlink != 1:
                raise ValueError(f"image payload must have exactly one hard link: {current}")
        elif not stat.S_ISDIR(current_stat.st_mode):
            raise ValueError(f"image path ancestor must be a directory: {current}")
        result.append((current, _stat_identity(current_stat)))
    return result


def _verified_image_stream(
    path: Path,
    *,
    manifest_root: Path,
    expected_sha256: str,
    expected_size: int | None = None,
    expected_identity: tuple[int, ...] | None = None,
) -> tuple[io.BytesIO, int, tuple[int, ...]]:
    """Read, hash, and return exactly the descriptor bytes later given to PIL."""

    if not _is_sha256(expected_sha256):
        raise ValueError("expected image SHA must be lowercase SHA-256")
    lexical_path = Path(os.path.abspath(path))
    before_chain = _lstat_image_chain(manifest_root, lexical_path)
    leaf_identity = before_chain[-1][1]
    if expected_identity is not None and leaf_identity != expected_identity:
        raise ValueError("image payload identity changed after manifest validation")
    leaf_size = leaf_identity[4]
    if leaf_size <= 0 or leaf_size > IMAGE_CONSUMPTION_MAX_BYTES:
        raise ValueError(
            f"image payload size must be in 1..{IMAGE_CONSUMPTION_MAX_BYTES} bytes"
        )
    if expected_size is not None and leaf_size != expected_size:
        raise ValueError("image payload size changed after manifest validation")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    if os.name == "posix":
        no_follow = getattr(os, "O_NOFOLLOW", None)
        if no_follow is None:
            raise RuntimeError("POSIX image consumption requires O_NOFOLLOW")
        flags |= no_follow | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(lexical_path, flags)
    except OSError as exc:
        raise ValueError(f"image payload secure open failed: {lexical_path}") from exc

    stream = io.BytesIO()
    try:
        opened_before = os.fstat(descriptor)
        if _stat_identity(opened_before) != leaf_identity:
            raise ValueError("image path identity changed during secure open")
        digest = hashlib.sha256()
        total = 0
        while True:
            try:
                chunk = os.read(descriptor, IMAGE_CONSUMPTION_READ_BYTES)
            except InterruptedError:
                continue
            if not chunk:
                break
            total += len(chunk)
            if total > IMAGE_CONSUMPTION_MAX_BYTES:
                raise ValueError("image payload exceeds the per-object byte cap")
            digest.update(chunk)
            stream.write(chunk)
        opened_after = os.fstat(descriptor)
        if _stat_identity(opened_after) != leaf_identity:
            raise ValueError("image payload identity changed while being consumed")
    except Exception:
        stream.close()
        raise
    finally:
        os.close(descriptor)

    try:
        after_chain = _lstat_image_chain(manifest_root, lexical_path)
        if after_chain != before_chain:
            raise ValueError("image path chain changed while being consumed")
        if total != leaf_size:
            raise ValueError("image payload read was partial")
        if digest.hexdigest() != expected_sha256:
            raise ValueError("image_sha256 does not match consumed image bytes")
        stream.seek(0)
        return stream, total, leaf_identity
    except Exception:
        stream.close()
        raise


def _is_sha256(value: str) -> bool:
    if len(value) != HEX_SHA256_LENGTH:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _nonempty(row: Mapping[str, Any], name: str, location: str) -> str:
    value = str(row.get(name, "")).strip()
    if not value:
        raise ValueError(f"{location}: missing required {name}")
    return value


def _unique_path_value(row: Mapping[str, Any], location: str) -> tuple[str, str]:
    available = [
        (name, str(row[name]).strip())
        for name in IMAGE_PATH_FIELDS
        if name in row and str(row[name]).strip()
    ]
    if not available:
        raise ValueError(f"{location}: missing required filepath")
    values = {value for _, value in available}
    if len(values) != 1:
        raise ValueError(f"{location}: conflicting image path fields {available}")
    return available[0]


def _parse_integer_label(
    row: Mapping[str, Any],
    name: str,
    location: str,
    *,
    required: bool,
) -> int:
    raw = str(row.get(name, "")).strip()
    if not raw:
        if required:
            raise ValueError(f"{location}: missing required {name}")
        return -1
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{location}: {name} must be an integer, got {raw!r}") from exc
    return value


@dataclass(frozen=True)
class ManifestRow:
    """One strict manifest row with its lineage retained verbatim."""

    path: Path
    material: int
    sample_id: str
    source_sha256: str
    object_group: str
    capture_session: str
    role: str
    fold: str
    image_sha256: str
    image_size_bytes: int
    image_identity: tuple[int, ...] = field(repr=False)
    manifest_split: str
    source_id: str
    source_object_count: int
    crop_object_count: int
    category: str
    origin: str
    dent: int = -1
    label: int = -1
    foreign_material: int = -1
    manifest_path: Path = field(default=Path("."), repr=False)
    manifest_line: int = field(default=0, repr=False)
    raw: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)

    @property
    def objectness(self) -> int:
        return 0 if self.material == BACKGROUND_MATERIAL_ID else 1

    @property
    def is_positive(self) -> bool:
        return self.objectness == 1

    @property
    def split(self) -> str:
        return self.manifest_split

    def condition(self, name: str) -> int:
        if name not in CONDITION_NAMES:
            raise KeyError(name)
        return int(getattr(self, name))

    def lineage_record(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in LINEAGE_FIELDS}


def _read_source_rows(path: Path) -> tuple[list[dict[str, str]], str]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {path}")
    text = path.read_text(encoding="utf-8-sig")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if not reader.fieldnames:
            raise ValueError(f"manifest has no CSV header: {path}")
        names = [str(name).strip() for name in reader.fieldnames]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError(f"manifest has empty or duplicate CSV columns: {path}")
        rows: list[dict[str, str]] = []
        for number, raw in enumerate(reader, start=2):
            cleaned: dict[str, str] = {}
            for key, value in raw.items():
                if key is None:
                    raise ValueError(f"{path}:{number}: unnamed extra CSV column")
                cleaned[str(key).strip()] = "" if value is None else str(value).strip()
            rows.append(cleaned)
        if not rows:
            raise ValueError(f"manifest is empty: {path}")
        return rows, "csv"
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{number}: JSONL row must be an object")
            rows.append(
                {
                    str(key).strip(): "" if value is None else str(value).strip()
                    for key, value in raw.items()
                }
            )
        if not rows:
            raise ValueError(f"manifest is empty: {path}")
        return rows, "jsonl"
    raise ValueError(f"unsupported manifest type {path.suffix!r}; use CSV or JSONL")


def _parse_manifest_row(
    raw: Mapping[str, str],
    *,
    manifest_path: Path,
    line: int,
) -> ManifestRow:
    location = f"{manifest_path}:{line}"
    missing_fields = [name for name in REQUIRED_MANIFEST_FIELDS if name not in raw]
    if missing_fields:
        raise ValueError(f"{location}: missing strict manifest fields {missing_fields}")
    sample_id = _nonempty(raw, "sample_id", location)
    source_sha256 = _nonempty(raw, "source_sha256", location)
    if not _is_sha256(source_sha256):
        raise ValueError(
            f"{location}: source_sha256 must be 64 lowercase hexadecimal characters"
        )
    object_group = _nonempty(raw, "object_group", location)
    capture_session = _nonempty(raw, "capture_session", location)
    role = _nonempty(raw, "role", location)
    if role not in ALL_ROLES:
        raise ValueError(
            f"{location}: unsupported role={role!r}; expected {ALL_ROLES}"
        )
    fold = _nonempty(raw, "fold", location)
    source_id = _nonempty(raw, "source_id", location)
    origin = _nonempty(raw, "origin", location)

    split = _nonempty(raw, "split", location).lower()
    if role in ROLE_TO_SPLIT and split != ROLE_TO_SPLIT[role]:
        raise ValueError(
            f"{location}: split={split!r} conflicts with role={role!r} "
            f"(expected {ROLE_TO_SPLIT[role]!r})"
        )

    strict_filepath = _nonempty(raw, "filepath", location)
    _, raw_path = _unique_path_value(raw, location)
    if raw_path != strict_filepath:
        raise ValueError(f"{location}: filepath conflicts with an image path alias")
    image_path = Path(raw_path)
    if not image_path.is_absolute():
        image_path = manifest_path.parent / image_path
    image_path = Path(os.path.abspath(image_path))

    material = _parse_integer_label(raw, "material", location, required=True)
    if material not in range(len(MATERIAL_CLASS_NAMES) + 1):
        raise ValueError(
            f"{location}: material={material} must be 0..8 or "
            f"{BACKGROUND_MATERIAL_ID} ({BACKGROUND_CLASS_NAME})"
        )
    category = _nonempty(raw, "category", location)
    expected_category = (
        BACKGROUND_CLASS_NAME
        if material == BACKGROUND_MATERIAL_ID
        else MATERIAL_CLASS_NAMES[material]
    )
    if category != expected_category:
        raise ValueError(
            f"{location}: category={category!r} conflicts with material={material} "
            f"({expected_category!r})"
        )

    source_object_count = _parse_integer_label(
        raw, "source_object_count", location, required=True
    )
    if source_object_count not in {0, 1}:
        raise ValueError(
            f"{location}: source_object_count={source_object_count} conflicts with "
            "the zero-or-one-object source count required by the "
            "single-object verifier contract"
        )
    expected_crop_object_count = (
        0 if material == BACKGROUND_MATERIAL_ID else 1
    )
    raw_crop_object_count = str(raw.get("crop_object_count", "")).strip()
    if raw_crop_object_count:
        crop_object_count = _parse_integer_label(
            raw, "crop_object_count", location, required=True
        )
    else:
        # Legacy manifests did not distinguish source and crop counts.  They
        # remain valid only for the old unambiguous cases: empty background
        # (0/0) and one-object material crops (1/1).
        crop_object_count = expected_crop_object_count
        if source_object_count != expected_crop_object_count:
            raise ValueError(
                f"{location}: source_object_count={source_object_count} conflicts "
                f"with material={material}; crop_object_count is required for "
                "a hard-negative background under the single-object verifier contract"
            )
    if crop_object_count != expected_crop_object_count:
        raise ValueError(
            f"{location}: crop_object_count={crop_object_count} conflicts with "
            f"material={material}; expected {expected_crop_object_count}"
        )
    if crop_object_count > source_object_count:
        raise ValueError(
            f"{location}: crop_object_count={crop_object_count} exceeds "
            f"source_object_count={source_object_count} for the "
            "single-object verifier contract"
        )

    condition_values = {
        name: _parse_integer_label(raw, name, location, required=True)
        for name in CONDITION_NAMES
    }
    for name, value in condition_values.items():
        if value not in {-1, 0, 1}:
            raise ValueError(f"{location}: {name}={value} must be -1, 0, or 1")

    declared_image_sha256 = _nonempty(raw, "image_sha256", location)
    if not _is_sha256(declared_image_sha256):
        raise ValueError(
            f"{location}: image_sha256 must be 64 lowercase hexadecimal characters"
        )
    verified_stream, image_size_bytes, image_identity = _verified_image_stream(
        image_path,
        manifest_root=manifest_path.parent,
        expected_sha256=declared_image_sha256,
    )
    verified_stream.close()

    return ManifestRow(
        path=image_path,
        material=material,
        sample_id=sample_id,
        source_sha256=source_sha256,
        object_group=object_group,
        capture_session=capture_session,
        role=role,
        fold=fold,
        image_sha256=declared_image_sha256,
        image_size_bytes=image_size_bytes,
        image_identity=image_identity,
        manifest_split=split,
        source_id=source_id,
        source_object_count=source_object_count,
        crop_object_count=crop_object_count,
        category=category,
        origin=origin,
        manifest_path=manifest_path,
        manifest_line=line,
        raw=dict(raw),
        **condition_values,
    )


def validate_group_integrity(rows: Sequence[ManifestRow]) -> None:
    """Reject row, source, physical-object, session, or content leakage."""
    if not rows:
        raise ValueError("manifest is empty")

    sample_locations: dict[str, str] = {}
    for row in rows:
        location = f"{row.manifest_path}:{row.manifest_line}"
        previous = sample_locations.get(row.sample_id)
        if previous is not None:
            raise ValueError(
                f"duplicate sample_id {row.sample_id!r} at {previous} and {location}"
            )
        sample_locations[row.sample_id] = location

    role_guard_fields = (
        "source_sha256",
        "image_sha256",
        "object_group",
        "capture_session",
    )
    for field_name in role_guard_fields:
        seen: dict[str, tuple[str, str]] = {}
        for row in rows:
            identity = str(getattr(row, field_name)).lower()
            location = f"{row.manifest_path}:{row.manifest_line}"
            previous = seen.get(identity)
            if previous is not None and previous[0] != row.role:
                raise ValueError(
                    f"leakage: {field_name} {getattr(row, field_name)!r} crosses "
                    f"train/validation roles {previous[0]!r} ({previous[1]}) and "
                    f"{row.role!r} ({location})"
                )
            seen[identity] = (row.role, location)

    # Multiple views/crops from one identity may stay in a fold, never cross it.
    for field_name in role_guard_fields:
        seen_fold: dict[tuple[str, str], tuple[str, str]] = {}
        for row in rows:
            identity = str(getattr(row, field_name)).lower()
            key = (row.role, identity)
            location = f"{row.manifest_path}:{row.manifest_line}"
            previous = seen_fold.get(key)
            if previous is not None and previous[0] != row.fold:
                raise ValueError(
                    f"leakage: {field_name} {getattr(row, field_name)!r} crosses "
                    f"folds {previous[0]!r} ({previous[1]}) and "
                    f"{row.fold!r} ({location}) within role {row.role!r}"
                )
            seen_fold[key] = (row.fold, location)


def validate_training_coverage(rows: Sequence[ManifestRow]) -> None:
    role_counts = Counter(row.role for row in rows)
    missing_roles = [role for role in TRAINING_ROLES if role_counts[role] == 0]
    if missing_roles:
        raise ValueError(f"manifest is missing required roles: {missing_roles}")
    for role in TRAINING_ROLES:
        objectness_values = {row.objectness for row in rows if row.role == role}
        if objectness_values != {0, 1}:
            raise ValueError(
                f"role {role!r} must contain positive material and background rows; "
                f"observed objectness labels={sorted(objectness_values)}"
            )
        observed_materials = {
            row.material for row in rows if row.role == role and row.is_positive
        }
        required_materials = set(range(len(MATERIAL_CLASS_NAMES)))
        if observed_materials != required_materials:
            missing = sorted(required_materials - observed_materials)
            raise ValueError(
                f"role {role!r} is missing positive material classes {missing}; "
                "all nine classes are required independently in train and validation"
            )


def read_manifests(
    paths: Sequence[str | Path],
) -> list[ManifestRow]:
    """Read strict CSV/JSONL manifests and preserve all lineage fields."""
    rows: list[ManifestRow] = []
    for raw_path in paths:
        manifest_path = Path(raw_path).resolve()
        raw_rows, source_type = _read_source_rows(manifest_path)
        line_offset = 2 if source_type == "csv" else 1
        for index, raw in enumerate(raw_rows, start=line_offset):
            rows.append(
                _parse_manifest_row(
                    raw,
                    manifest_path=manifest_path,
                    line=index,
                )
            )
    validate_group_integrity(rows)
    validate_training_coverage(rows)
    return rows


def canonical_condition_heads(names: Sequence[str]) -> tuple[str, ...]:
    requested_names = tuple(names)
    unknown = sorted(set(requested_names) - set(CONDITION_NAMES))
    if unknown:
        raise ValueError(f"unknown condition heads: {unknown}")
    if len(requested_names) != len(set(requested_names)):
        raise ValueError("condition heads must not be repeated")
    requested_set = set(requested_names)
    return tuple(name for name in CONDITION_NAMES if name in requested_set)


def resolve_condition_heads(
    rows: Sequence[ManifestRow],
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    """Resolve auto/explicit condition heads against positive training labels."""
    if requested is None:
        candidates: Iterable[str] = CONDITION_NAMES
        explicit = False
    else:
        candidates = canonical_condition_heads(requested)
        explicit = True
    result: list[str] = []
    for name in candidates:
        values_by_role = {
            role: {
                row.condition(name)
                for row in rows
                if row.role == role and row.is_positive and row.condition(name) >= 0
            }
            for role in TRAINING_ROLES
        }
        complete = all(values_by_role[role] == {0, 1} for role in TRAINING_ROLES)
        if complete:
            result.append(name)
        elif explicit:
            raise ValueError(
                f"condition head {name!r} requires labels 0 and 1 on positive rows "
                f"in both roles; observed={values_by_role}"
            )
    return tuple(result)


class _TinyCpuBackbone(nn.Module):
    """Small offline backbone used only when explicitly requested or in smoke mode."""

    num_features = 32

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, self.num_features, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if name == "tiny_cnn":
        return _TinyCpuBackbone(), _TinyCpuBackbone.num_features
    return _build_legacy_backbone(name, pretrained)


class MultitaskCropVerifier(nn.Module):
    """Shared-backbone v3 verifier with separate objectness/material heads."""

    def __init__(
        self,
        backbone_name: str = "mobilenet_v3_small",
        *,
        pretrained: bool = True,
        condition_heads: Sequence[str] = CONDITION_NAMES,
    ) -> None:
        super().__init__()
        names = canonical_condition_heads(condition_heads)
        self.backbone_name = backbone_name
        self.condition_head_names = names
        self.backbone, features = _build_backbone(backbone_name, pretrained)
        self.objectness_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(features, 2))
        # Deliberately nine outputs. Background never enters this softmax.
        self.material_head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(features, len(MATERIAL_CLASS_NAMES))
        )
        self.condition_heads = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Dropout(0.2), nn.Linear(features, 2))
                for name in names
            }
        )

    @property
    def output_names(self) -> tuple[str, ...]:
        return ("objectness", "material", *self.condition_head_names)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(image)
        outputs = {
            "objectness": self.objectness_head(features),
            "material": self.material_head(features),
        }
        outputs.update(
            {name: self.condition_heads[name](features) for name in self.condition_head_names}
        )
        return outputs


class VerifierDataset(Dataset):
    def __init__(self, rows: Sequence[ManifestRow], transform: Any) -> None:
        self.rows = list(rows)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        stream, _, _ = _verified_image_stream(
            row.path,
            manifest_root=row.manifest_path.parent,
            expected_sha256=row.image_sha256,
            expected_size=row.image_size_bytes,
            expected_identity=row.image_identity,
        )
        with stream:
            with Image.open(stream) as image:
                width, height = image.size
                if (
                    type(width) is not int
                    or type(height) is not int
                    or width <= 0
                    or height <= 0
                    or width * height > IMAGE_CONSUMPTION_MAX_PIXELS
                ):
                    raise ValueError("decoded image exceeds the per-object pixel cap")
                image.load()
                rgb = image.convert("RGB")
                try:
                    tensor = self.transform(rgb)
                finally:
                    rgb.close()
        return {
            "image": tensor,
            "objectness": row.objectness,
            "material": row.material,
            "dent": row.dent,
            "label": row.label,
            "foreign_material": row.foreign_material,
            "sample_id": row.sample_id,
        }


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:  # torch/torchvision training itself does not require numpy here.
        np = None
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(worker_seed)


def _training_rows(rows: Sequence[ManifestRow]) -> list[ManifestRow]:
    return [row for row in rows if row.role == TRAIN_ROLE]


def parse_origin_weights(values: Sequence[str]) -> dict[str, float]:
    """Parse exact ``origin=weight`` entries without duplicating manifest rows."""
    result: dict[str, float] = {}
    for value in values:
        origin, separator, raw_weight = value.partition("=")
        origin = origin.strip()
        raw_weight = raw_weight.strip()
        if not separator or not origin or not raw_weight:
            raise ValueError(
                f"invalid origin weight {value!r}; expected nonempty ORIGIN=WEIGHT"
            )
        if origin in result:
            raise ValueError(f"origin weight repeated for {origin!r}")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"origin weight must be numeric for {origin!r}") from exc
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"origin weight must be finite and positive for {origin!r}")
        result[origin] = weight
    return result


def build_origin_sampling_plan(
    rows: Sequence[ManifestRow], origin_weights: Mapping[str, float]
) -> tuple[list[float] | None, dict[str, Any]]:
    """Return a deterministic replacement-sampling plan for training rows only."""
    train_rows = _training_rows(rows)
    observed = Counter(row.origin for row in train_rows)
    missing = sorted(set(origin_weights) - set(observed))
    if missing:
        raise ValueError(f"origin weights reference absent training origins: {missing}")
    weights = [float(origin_weights.get(row.origin, 1.0)) for row in train_rows]
    weighted_mass = Counter()
    for row, weight in zip(train_rows, weights):
        weighted_mass[row.origin] += weight
    total_mass = math.fsum(weights)
    # Sampling weights are relative.  Multiplying every origin by the same
    # constant must not silently switch a uniform epoch from shuffle-without-
    # replacement to replacement sampling (which would omit and repeat rows).
    custom = len(set(weights)) > 1
    metadata = {
        "mode": "weighted_replacement" if custom else "shuffle_without_replacement",
        "samples_per_epoch": len(train_rows),
        "configured_origin_weights": dict(sorted(origin_weights.items())),
        "row_counts_by_origin": dict(sorted(observed.items())),
        "weighted_mass_by_origin": {
            origin: float(weighted_mass[origin]) for origin in sorted(weighted_mass)
        },
        "expected_fraction_by_origin": {
            origin: float(weighted_mass[origin] / total_mass)
            for origin in sorted(weighted_mass)
        },
        "manifest_rows_remain_unique": True,
    }
    return (weights if custom else None), metadata


def build_class_weight_values(
    rows: Sequence[ManifestRow],
    condition_heads: Sequence[str],
    *,
    mode: str = "inverse",
    beta: float = DEFAULT_CLASS_WEIGHT_BETA,
) -> dict[str, torch.Tensor | None]:
    """Build balanced weights; material/condition counts exclude backgrounds."""
    train_rows = _training_rows(rows)
    values: dict[str, list[int]] = {
        "objectness": [row.objectness for row in train_rows],
        "material": [row.material for row in train_rows if row.is_positive],
    }
    canonical_heads = canonical_condition_heads(condition_heads)
    for name in canonical_heads:
        values[name] = [
            row.condition(name)
            for row in train_rows
            if row.is_positive and row.condition(name) >= 0
        ]
    return {
        name: class_weights(
            labels,
            2 if name != "material" else len(MATERIAL_CLASS_NAMES),
            mode=mode,
            beta=beta,
        )
        for name, labels in values.items()
    }


def build_criteria(
    weight_values: Mapping[str, torch.Tensor | None],
    device: torch.device,
    *,
    label_smoothing: float = 0.0,
) -> dict[str, nn.CrossEntropyLoss]:
    if not math.isfinite(label_smoothing) or not 0 <= label_smoothing < 1:
        raise ValueError("label smoothing must be finite and in [0, 1)")
    return {
        name: nn.CrossEntropyLoss(
            weight=(weight.to(device) if weight is not None else None),
            reduction="none",
            label_smoothing=label_smoothing,
        )
        for name, weight in weight_values.items()
    }


def compute_multitask_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    criteria: Mapping[str, nn.CrossEntropyLoss],
    task_weights: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, int]]:
    """Compute v3 losses with an explicit positive-only material mask."""
    required_outputs = set(criteria)
    missing = sorted(required_outputs - set(outputs))
    if missing:
        raise ValueError(f"model outputs are missing heads: {missing}")

    objectness = batch["objectness"].long()
    per_task: dict[str, torch.Tensor] = {
        "objectness": criteria["objectness"](
            outputs["objectness"], objectness
        ).mean()
    }
    counts = {"objectness": int(objectness.numel())}

    # Background rows have material=9, which is intentionally outside the
    # material head.  Slice before CE so they cannot affect its logits/gradient.
    positive_mask = objectness == 1
    if positive_mask.any():
        material_target = batch["material"].long()[positive_mask]
        if torch.any((material_target < 0) | (material_target >= len(MATERIAL_CLASS_NAMES))):
            raise ValueError("positive material targets must be in 0..8")
        per_task["material"] = criteria["material"](
            outputs["material"][positive_mask], material_target
        ).mean()
        counts["material"] = int(positive_mask.sum().item())
    else:
        counts["material"] = 0

    for name in criteria:
        if name in {"objectness", "material"}:
            continue
        targets = batch[name].long()
        condition_mask = positive_mask & (targets >= 0)
        if condition_mask.any():
            per_task[name] = criteria[name](
                outputs[name][condition_mask], targets[condition_mask]
            ).mean()
            counts[name] = int(condition_mask.sum().item())
        else:
            counts[name] = 0

    total: torch.Tensor | None = None
    for name, task_loss in per_task.items():
        weight = float(task_weights.get(name, 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"task weight for {name} must be finite and non-negative")
        weighted = task_loss * weight
        total = weighted if total is None else total + weighted
    if total is None:  # Objectness always has labels, kept as a defensive invariant.
        raise RuntimeError("batch has no supervised task")
    return total, per_task, counts


def _confusion_update(
    confusion: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    if targets.numel() == 0:
        return
    predictions = logits.argmax(dim=1)
    classes = confusion.shape[0]
    bins = torch.bincount(
        (targets.detach().cpu() * classes + predictions.detach().cpu()),
        minlength=classes * classes,
    ).reshape(classes, classes)
    confusion += bins


def _metrics_from_confusion(confusion: torch.Tensor) -> dict[str, Any]:
    total = int(confusion.sum().item())
    supports = confusion.sum(dim=1)
    recalls = []
    per_class_recall: list[float | None] = []
    for index, support in enumerate(supports.tolist()):
        value = float(confusion[index, index].item() / support) if support else None
        per_class_recall.append(value)
        if value is not None:
            recalls.append(value)
    return {
        "count": total,
        "accuracy": float(confusion.diag().sum().item() / total) if total else None,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else None,
        "support": [int(value) for value in supports.tolist()],
        "per_class_recall": per_class_recall,
    }


def run_epoch(
    model: MultitaskCropVerifier,
    loader: DataLoader,
    criteria: Mapping[str, nn.CrossEntropyLoss],
    task_weights: Mapping[str, float],
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> tuple[float, dict[str, dict[str, Any]]]:
    training = optimizer is not None
    model.train(training)
    confusions = {
        name: torch.zeros(
            (len(MATERIAL_CLASS_NAMES) if name == "material" else 2,) * 2,
            dtype=torch.int64,
        )
        for name in criteria
    }
    total_loss = 0.0
    total_examples = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        image = batch["image"].to(device)
        device_batch = {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in batch.items()
        }
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(image)
            loss, _, _ = compute_multitask_loss(
                outputs, device_batch, criteria, task_weights
            )
            if training:
                loss.backward()
                optimizer.step()

        objectness = device_batch["objectness"].long()
        _confusion_update(confusions["objectness"], outputs["objectness"], objectness)
        positive = objectness == 1
        if positive.any():
            _confusion_update(
                confusions["material"],
                outputs["material"][positive],
                device_batch["material"].long()[positive],
            )
        for name in criteria:
            if name in {"objectness", "material"}:
                continue
            target = device_batch[name].long()
            mask = positive & (target >= 0)
            if mask.any():
                _confusion_update(confusions[name], outputs[name][mask], target[mask])

        batch_size = int(image.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1), {
        name: _metrics_from_confusion(confusion)
        for name, confusion in confusions.items()
    }


def build_output_contract(condition_heads: Sequence[str]) -> dict[str, Any]:
    condition_heads = canonical_condition_heads(condition_heads)
    outputs: list[dict[str, Any]] = [
        {
            "name": "objectness",
            "kind": "logits",
            "shape": ["batch", len(OBJECTNESS_CLASS_NAMES)],
            "activation": "softmax",
            "class_names": list(OBJECTNESS_CLASS_NAMES),
            "class_ids": {name: index for index, name in enumerate(OBJECTNESS_CLASS_NAMES)},
            "trained_on": "all proposal rows",
        },
        {
            "name": "material",
            "kind": "logits",
            "shape": ["batch", len(MATERIAL_CLASS_NAMES)],
            "activation": "softmax",
            "class_names": list(MATERIAL_CLASS_NAMES),
            "class_ids": {name: index for index, name in enumerate(MATERIAL_CLASS_NAMES)},
            "valid_when": {"output": "objectness", "class_id": 1, "class_name": "material"},
            "trained_on": "positive material rows only; background is excluded from CE",
        },
    ]
    for name in condition_heads:
        class_names = CONDITION_CLASS_NAMES[name]
        outputs.append(
            {
                "name": name,
                "kind": "logits",
                "shape": ["batch", len(class_names)],
                "activation": "softmax",
                "class_names": list(class_names),
                "class_ids": {
                    class_name: index for index, class_name in enumerate(class_names)
                },
                "valid_when": {
                    "output": "objectness",
                    "class_id": 1,
                    "label_is_present": True,
                },
                "trained_on": "labeled positive material rows only",
            }
        )
    return {
        "version": OUTPUT_CONTRACT_VERSION,
        "output_order": [output["name"] for output in outputs],
        "outputs": outputs,
        "material_background_class_id": None,
        "decision_order": ["objectness", "material", "conditions"],
        "warning": "This v3 contract is not the legacy four-output production contract.",
    }


def _manifest_summary(
    rows: Sequence[ManifestRow], manifest_paths: Sequence[str | Path]
) -> dict[str, Any]:
    ordered_lineage = [
        {
            **row.lineage_record(),
            "image_sha256": row.image_sha256.lower(),
            "material": str(row.material),
        }
        for row in sorted(rows, key=lambda item: item.sample_id)
    ]
    lineage_bytes = json.dumps(
        ordered_lineage, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload_rows = [
        {
            "sample_id": row.sample_id,
            "role": row.role,
            "path": row.path.relative_to(row.manifest_path.parent).as_posix(),
            "size": row.image_size_bytes,
            "sha256": row.image_sha256.lower(),
        }
        for row in sorted(rows, key=lambda item: (item.role, item.sample_id))
    ]
    payload_set_bytes = (
        json.dumps(
            payload_rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    role_counts = Counter(row.role for row in rows)
    objectness_counts = Counter((row.role, row.objectness) for row in rows)
    material_counts = Counter(
        (row.role, row.material) for row in rows if row.is_positive
    )
    return {
        "strict": True,
        "required_lineage_fields": list(LINEAGE_FIELDS),
        "rows": len(rows),
        "lineage_sha256": hashlib.sha256(lineage_bytes).hexdigest(),
        "payload_set_sha256": hashlib.sha256(payload_set_bytes).hexdigest(),
        "input_manifests": [
            {
                "path": str(Path(path).resolve()),
                "sha256": _sha256_file(Path(path).resolve()),
            }
            for path in manifest_paths
        ],
        "role_counts": dict(sorted(role_counts.items())),
        "excluded_from_training_role_counts": {
            role: role_counts[role] for role in (CALIBRATION_ROLE, BLIND_TEST_ROLE)
        },
        "folds_by_role": {
            role: sorted({row.fold for row in rows if row.role == role})
            for role in ALL_ROLES
        },
        "unique": {
            field_name: len({str(getattr(row, field_name)).lower() for row in rows})
            for field_name in (
                "sample_id", "source_sha256", "image_sha256", "object_group", "capture_session"
            )
        },
        "objectness_counts": {
            f"{role}/{OBJECTNESS_CLASS_NAMES[class_id]}": objectness_counts[(role, class_id)]
            for role in ALL_ROLES
            for class_id in range(2)
        },
        "positive_material_counts": {
            f"{role}/{MATERIAL_CLASS_NAMES[class_id]}": material_counts[(role, class_id)]
            for role in ALL_ROLES
            for class_id in range(len(MATERIAL_CLASS_NAMES))
        },
    }


def _build_transforms(size: int, *, smoke: bool) -> tuple[Any, Any]:
    validation = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    if smoke:
        return validation, validation
    training = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(
                degrees=10, translate=(0.04, 0.04), scale=(0.92, 1.08)
            ),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.04),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, scale=(0.01, 0.05)),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return training, validation


def _make_loader(
    rows: Sequence[ManifestRow],
    transform: Any,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
    sample_weights: Sequence[float] | None = None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = None
    if sample_weights is not None:
        if not shuffle:
            raise ValueError("sample weights are allowed only for a shuffled training loader")
        if len(sample_weights) != len(rows):
            raise ValueError("sample weights must match the training row count")
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(rows),
            replacement=True,
            generator=generator,
        )
    loader_options: dict[str, Any] = {}
    if workers:
        # The authoritative NAS path eagerly initializes CUDA before manifest
        # hashing.  Linux's default ``fork`` would clone that live CUDA state
        # into workers and can deadlock or fail before the first epoch.  Spawn
        # gives every image-consumption worker a clean interpreter instead.
        loader_options["multiprocessing_context"] = "spawn"
    return DataLoader(
        VerifierDataset(rows, transform),
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker if workers else None,
        generator=generator,
        **loader_options,
    )


def _build_optimizer(
    model: MultitaskCropVerifier,
    *,
    lr: float,
    backbone_lr: float | None,
    head_lr: float | None,
) -> tuple[torch.optim.Optimizer, dict[str, float]]:
    learning_rates = {
        "base": lr,
        "backbone": lr if backbone_lr is None else backbone_lr,
        "heads": lr if head_lr is None else head_lr,
    }
    for name, value in learning_rates.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} learning rate must be finite and positive")
    backbone_parameters = list(model.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in backbone_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": learning_rates["backbone"],
                "name": "backbone",
            },
            {"params": head_parameters, "lr": learning_rates["heads"], "name": "heads"},
        ],
        lr=lr,
        weight_decay=1e-4,
    )
    return optimizer, learning_rates


def _selection_score(
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    require_complete_support: bool,
) -> float:
    selected: list[float] = []
    for name, task_metrics in metrics.items():
        support = task_metrics["support"]
        if require_complete_support and any(value <= 0 for value in support):
            raise RuntimeError(
                f"validation head {name!r} is missing class support: {support}"
            )
        balanced_accuracy = task_metrics["balanced_accuracy"]
        if balanced_accuracy is None:
            raise RuntimeError(f"validation head {name!r} has no labeled samples")
        selected.append(float(balanced_accuracy))
    if not selected:
        raise RuntimeError("validation has no scored heads")
    return float(sum(selected) / len(selected))


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


class _OnnxOutputWrapper(nn.Module):
    def __init__(self, model: MultitaskCropVerifier) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = self.model(image)
        return tuple(outputs[name] for name in self.model.output_names)


def export_onnx(model: MultitaskCropVerifier, path: Path, size: int) -> None:
    model.eval().cpu()
    wrapper = _OnnxOutputWrapper(model)
    dummy = torch.randn(1, 3, size, size)
    dynamic_axes = {"img": {0: "batch"}}
    dynamic_axes.update({name: {0: "batch"} for name in model.output_names})
    kwargs = {
        "input_names": ["img"],
        "output_names": list(model.output_names),
        "dynamic_axes": dynamic_axes,
        "opset_version": 17,
    }
    try:
        torch.onnx.export(wrapper, dummy, path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(wrapper, dummy, path, **kwargs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the v3 objectness + positive-only material verifier."
    )
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--backbone", default="mobilenet_v3_small")
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float)
    parser.add_argument("--head-lr", type=float)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--class-weight-mode", choices=CLASS_WEIGHT_MODES, default="inverse"
    )
    parser.add_argument("--class-weight-beta", type=float, default=DEFAULT_CLASS_WEIGHT_BETA)
    parser.add_argument("--objectness-weight", type=float, default=1.0)
    parser.add_argument("--material-weight", type=float, default=1.0)
    parser.add_argument("--condition-weight", type=float, default=0.5)
    parser.add_argument(
        "--origin-weight",
        action="append",
        default=[],
        metavar="ORIGIN=WEIGHT",
        help=(
            "Increase an exact training origin's sampling probability without "
            "duplicating strict manifest identities. Repeat for multiple origins."
        ),
    )
    parser.add_argument(
        "--condition-head",
        action="append",
        choices=CONDITION_NAMES,
        help="Repeat to require selected heads. Omit to auto-enable labeled heads.",
    )
    parser.add_argument(
        "--no-condition-heads",
        action="store_true",
        help="Produce only objectness and material heads.",
    )
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dataset-snapshot-report-sha256")
    parser.add_argument("--dataset-snapshot-tree-sha256")
    parser.add_argument("--manifest-payload-set-sha256")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-export-onnx", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate lineage/splits and print exact class weights without writing files.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one deterministic CPU epoch with tiny_cnn and no ONNX export.",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.dry_run and args.smoke:
        parser.error("--dry-run and --smoke are mutually exclusive")
    if not args.dry_run and not args.output_dir:
        parser.error("--output-dir is required unless --dry-run is used")
    if args.condition_head and args.no_condition_heads:
        parser.error("--condition-head and --no-condition-heads are mutually exclusive")
    for name in ("size", "epochs", "patience", "batch"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if not math.isfinite(args.class_weight_beta) or not 0 <= args.class_weight_beta < 1:
        parser.error("--class-weight-beta must be finite and in [0, 1)")
    if not math.isfinite(args.label_smoothing) or not 0 <= args.label_smoothing < 1:
        parser.error("--label-smoothing must be finite and in [0, 1)")
    for name in ("objectness_weight", "material_weight", "condition_weight"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.objectness_weight == 0 or args.material_weight == 0:
        parser.error("objectness and material task weights must both be positive")
    authority_values = (
        args.dataset_snapshot_report_sha256,
        args.dataset_snapshot_tree_sha256,
    )
    if any(value is None for value in authority_values) and not all(
        value is None for value in authority_values
    ):
        parser.error("dataset authority SHA arguments must be supplied together")
    if any(value is not None for value in authority_values) and (
        args.manifest_payload_set_sha256 is None
    ):
        parser.error(
            "--manifest-payload-set-sha256 is required with dataset authority SHAs"
        )
    if all(value is not None for value in authority_values) and not sys.platform.startswith(
        "linux"
    ):
        parser.error("authority-bound image consumption requires Linux/QNAP")
    for value in authority_values:
        if value is not None and not _is_sha256(value):
            parser.error("dataset authority SHA arguments must be lowercase SHA-256")
    if (
        args.manifest_payload_set_sha256 is not None
        and not _is_sha256(args.manifest_payload_set_sha256)
    ):
        parser.error("manifest payload-set SHA must be lowercase SHA-256")


def _weight_metadata(
    weights: Mapping[str, torch.Tensor | None]
) -> dict[str, list[float] | None]:
    return {
        name: value.detach().cpu().tolist() if value is not None else None
        for name, value in weights.items()
    }


def eager_initialize_cuda_context() -> torch.Tensor:
    """Reserve this training process's QNAP CUDA fault buffer before manifest I/O."""

    try:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "torch.cuda.is_available() is false "
                f"(device_count={torch.cuda.device_count()})"
            )
        guard = torch.ones(1, device="cuda:0") + 1
        torch.cuda.synchronize(0)
        if guard.item() != 2:
            raise RuntimeError("CUDA tensor smoke result was not 2")
        print(
            "eager CUDA context ready: " + torch.cuda.get_device_name(0),
            flush=True,
        )
        return guard
    except Exception as exc:
        raise RuntimeError(
            "failed to eagerly initialize CUDA in the verifier training process"
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    smoke = bool(args.smoke)
    output_dir: Path | None = None
    if not args.dry_run:
        output_dir = Path(args.output_dir).resolve()
        if output_dir.exists():
            if not output_dir.is_dir():
                parser.error(f"--output-dir is not a directory: {output_dir}")
            if any(output_dir.iterdir()):
                parser.error(
                    f"--output-dir must be absent or empty to avoid stale/overwritten "
                    f"artifacts: {output_dir}"
                )

    # Keep this tensor referenced in the main frame through manifest/image SHA
    # verification and all epochs.  Explicit CUDA training is the NAS path that
    # must fail closed; dry-run, smoke and CPU/auto paths preserve their prior
    # behavior and do not acquire a CUDA context here.
    _cuda_context_guard: torch.Tensor | None = None
    if not args.dry_run and not smoke and args.device == "cuda":
        try:
            _cuda_context_guard = eager_initialize_cuda_context()
        except RuntimeError as exc:
            parser.error(str(exc))

    try:
        rows = read_manifests(args.manifest)
        requested_heads: Sequence[str] | None
        if args.no_condition_heads:
            requested_heads = ()
        else:
            requested_heads = args.condition_head
        condition_heads = resolve_condition_heads(rows, requested_heads)
        if condition_heads and args.condition_weight <= 0:
            raise ValueError(
                "--condition-weight must be positive when condition heads are enabled"
            )
        class_weight_values = build_class_weight_values(
            rows,
            condition_heads,
            mode=args.class_weight_mode,
            beta=args.class_weight_beta,
        )
        origin_weights = parse_origin_weights(args.origin_weight)
        train_sample_weights, sampling_plan = build_origin_sampling_plan(
            rows, origin_weights
        )
        manifest_summary = _manifest_summary(rows, args.manifest)
        if (
            args.manifest_payload_set_sha256 is not None
            and args.manifest_payload_set_sha256
            != manifest_summary["payload_set_sha256"]
        ):
            raise ValueError("manifest payload-set SHA differs from consumed manifests")
        dataset_consumption_contract = build_image_consumption_contract(
            trainer_sha256=_sha256_file(Path(__file__).resolve()),
            dataset_snapshot_report_sha256=args.dataset_snapshot_report_sha256,
            dataset_snapshot_tree_sha256=args.dataset_snapshot_tree_sha256,
            manifest_payload_set_sha256=manifest_summary["payload_set_sha256"],
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    output_contract = build_output_contract(condition_heads)
    preflight = {
        "ok": True,
        "mode": "dry-run" if args.dry_run else ("smoke" if args.smoke else "train"),
        "seed": args.seed,
        "manifest": manifest_summary,
        "condition_heads": list(condition_heads),
        "class_weights": {
            "mode": args.class_weight_mode,
            "beta": args.class_weight_beta,
            "values": _weight_metadata(class_weight_values),
        },
        "sampling": sampling_plan,
        "output_contract": output_contract,
        "dataset_consumption_contract": dataset_consumption_contract,
    }
    if args.dry_run:
        print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    effective = {
        "backbone": "tiny_cnn" if smoke else args.backbone,
        "size": min(args.size, 64) if smoke else args.size,
        "epochs": 1 if smoke else args.epochs,
        "patience": 1 if smoke else args.patience,
        "batch": min(args.batch, 8) if smoke else args.batch,
        "workers": 0 if smoke else args.workers,
        "max_train_batches": 2 if smoke else None,
        "max_validation_batches": 2 if smoke else None,
        "pretrained": False if smoke else not args.no_pretrained,
        "export_onnx": False if smoke else not args.no_export_onnx,
    }
    assert output_dir is not None

    seed_everything(args.seed)
    if smoke or args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("--device cuda requested but CUDA is unavailable")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_rows = [row for row in rows if row.role == TRAIN_ROLE]
    validation_rows = [row for row in rows if row.role == VALIDATION_ROLE]
    train_transform, validation_transform = _build_transforms(
        effective["size"], smoke=smoke
    )
    train_loader = _make_loader(
        train_rows,
        train_transform,
        batch_size=effective["batch"],
        workers=effective["workers"],
        shuffle=True,
        seed=args.seed,
        pin_memory=device.type == "cuda",
        sample_weights=train_sample_weights,
    )
    validation_loader = _make_loader(
        validation_rows,
        validation_transform,
        batch_size=effective["batch"],
        workers=effective["workers"],
        shuffle=False,
        seed=args.seed + 1,
        pin_memory=device.type == "cuda",
    )

    model = MultitaskCropVerifier(
        effective["backbone"],
        pretrained=effective["pretrained"],
        condition_heads=condition_heads,
    ).to(device)
    criteria = build_criteria(
        class_weight_values, device, label_smoothing=args.label_smoothing
    )
    task_weights = {
        "objectness": args.objectness_weight,
        "material": args.material_weight,
        **{name: args.condition_weight for name in condition_heads},
    }
    try:
        optimizer, learning_rates = _build_optimizer(
            model,
            lr=args.lr,
            backbone_lr=args.backbone_lr,
            head_lr=args.head_lr,
        )
    except ValueError as exc:
        parser.error(str(exc))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, effective["epochs"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / CHECKPOINT_NAME
    best_score = -math.inf
    best_epoch = 0
    no_improve = 0
    best_metrics: dict[str, Any] | None = None

    print(
        f"device={device} mode={'smoke' if smoke else 'train'} "
        f"train={len(train_rows)} validation={len(validation_rows)} "
        f"heads={output_contract['output_order']}",
        flush=True,
    )
    for epoch in range(1, effective["epochs"] + 1):
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criteria,
            task_weights,
            device,
            optimizer=optimizer,
            max_batches=effective["max_train_batches"],
        )
        validation_loss, validation_metrics = run_epoch(
            model,
            validation_loader,
            criteria,
            task_weights,
            device,
            max_batches=effective["max_validation_batches"],
        )
        scheduler.step()
        score = _selection_score(
            validation_metrics, require_complete_support=not smoke
        )
        print(
            f"[{epoch:02d}/{effective['epochs']}] "
            f"loss={train_loss:.4f}/{validation_loss:.4f} "
            f"objectness_bal={validation_metrics['objectness']['balanced_accuracy']:.4f} "
            f"material_bal={validation_metrics['material']['balanced_accuracy']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            best_metrics = {
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            training_config = {
                "seed": args.seed,
                "deterministic_algorithms": True,
                "smoke": smoke,
                "learning_rates": learning_rates,
                "label_smoothing": args.label_smoothing,
                "class_weights": preflight["class_weights"],
                "task_weights": task_weights,
                "sampling": sampling_plan,
                "effective": effective,
            }
            torch.save(
                {
                    "format_version": 3,
                    "architecture": "multitask_crop_verifier",
                    "state_dict": _cpu_state_dict(model),
                    "model_config": {
                        "backbone": effective["backbone"],
                        "input_size": effective["size"],
                        "condition_heads": list(condition_heads),
                    },
                    "backbone": effective["backbone"],
                    "input_size": effective["size"],
                    "classes": list(MATERIAL_CLASS_NAMES),
                    "material_classes": list(MATERIAL_CLASS_NAMES),
                    "objectness_classes": list(OBJECTNESS_CLASS_NAMES),
                    "condition_classes": {
                        name: list(CONDITION_CLASS_NAMES[name])
                        for name in condition_heads
                    },
                    "output_contract": output_contract,
                    "preprocessing": {
                        "color_space": "RGB",
                        "resize": [effective["size"], effective["size"]],
                        "normalization": {
                            "mean": list(IMAGENET_MEAN),
                            "std": list(IMAGENET_STD),
                        },
                    },
                    "manifest_contract": {
                        "required_fields": list(REQUIRED_MANIFEST_FIELDS),
                        "lineage_fields": list(LINEAGE_FIELDS),
                        "allowed_roles": list(ALL_ROLES),
                        "optimization_role": TRAIN_ROLE,
                        "checkpoint_selection_role": VALIDATION_ROLE,
                        "excluded_roles": [CALIBRATION_ROLE, BLIND_TEST_ROLE],
                    },
                    "manifest_summary": manifest_summary,
                    "dataset_consumption_contract": dataset_consumption_contract,
                    "training_config": training_config,
                    "selection_contract": {
                        "metric": "mean balanced accuracy",
                        "heads": list(criteria),
                        "requires_every_class_in_validation": not smoke,
                    },
                    "epoch": epoch,
                    "selection_score": score,
                    "metrics": best_metrics,
                },
                checkpoint_path,
            )
        else:
            no_improve += 1
            if no_improve >= effective["patience"]:
                print(
                    f"early stop: {effective['patience']} epochs without improvement",
                    flush=True,
                )
                break

    if not checkpoint_path.is_file() or best_metrics is None:
        raise RuntimeError("training did not produce a checkpoint")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if effective["export_onnx"]:
        export_model = MultitaskCropVerifier(
            checkpoint["backbone"],
            pretrained=False,
            condition_heads=checkpoint["model_config"]["condition_heads"],
        )
        export_model.load_state_dict(checkpoint["state_dict"])
        export_onnx(export_model, output_dir / ONNX_NAME, checkpoint["input_size"])

    metadata = {
        "format_version": 3,
        "architecture": checkpoint["architecture"],
        "candidate_only": True,
        "production_runtime_modified": False,
        "checkpoint": CHECKPOINT_NAME,
        "onnx": ONNX_NAME if effective["export_onnx"] else None,
        "model_config": checkpoint["model_config"],
        "classes": checkpoint["classes"],
        "material_classes": checkpoint["material_classes"],
        "objectness_classes": checkpoint["objectness_classes"],
        "condition_classes": checkpoint["condition_classes"],
        "output_contract": checkpoint["output_contract"],
        "preprocessing": checkpoint["preprocessing"],
        "manifest_contract": checkpoint["manifest_contract"],
        "manifest_summary": checkpoint["manifest_summary"],
        "dataset_consumption_contract": checkpoint["dataset_consumption_contract"],
        "training_config": checkpoint["training_config"],
        "selection_contract": checkpoint["selection_contract"],
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "best_metrics": best_metrics,
    }
    (output_dir / METADATA_NAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"checkpoint: {checkpoint_path}", flush=True)
    print(f"metadata: {output_dir / METADATA_NAME}", flush=True)
    if effective["export_onnx"]:
        print(f"ONNX: {output_dir / ONNX_NAME}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
