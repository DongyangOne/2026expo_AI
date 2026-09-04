"""Fail-closed perceptual near-duplicate audit for v4 candidate data.

The report produced here is evidence about dataset separation only.  A pHash
match is deliberately never granted authority to delete, relabel, select,
promote, or deploy a model.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import io
import json
import os
import platform
import re
import stat
import sys
import tempfile
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np
import PIL
from PIL import Image, UnidentifiedImageError


# Bind the complete module execution plan as loaded, including imports and
# top-level statements.  Function-only fingerprints cannot see a modified
# module that changes dependency state before restoring its source file.
_RUNTIME_MODULE_CODE = sys._getframe().f_code


REPORT_SCHEMA = "v4_near_duplicate_leakage_audit.v1"
ALGORITHM_ID = "oneexpo_phash_rot4_v1"
PHASH_DISTANCE = 4
MAX_ENCODED_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_GRAPH_EDGES = 1_000_000

PROTECTED_INVENTORY_SCHEMA = "v4_near_duplicate_protected_inventory.v1"
PROTECTED_SOURCES_SCHEMA = "v4_candidate_protected_holdouts.v1"
PROTECTED_SOURCE_FIELDS = (
    "qx3_diagnostic_source_sha256",
    "qx3_validation_source_sha256",
    "hardware41_source_sha256",
    "known_audit_source_sha256",
    "calibration_source_sha256",
    "blind_test_source_sha256",
)
PROTECTED_REQUIRED_COUNTS = {
    "qx3_diagnostic_source_sha256": 3500,
    "qx3_validation_source_sha256": 1000,
    "hardware41_source_sha256": 41,
}
PROTECTED_COHORT_BY_FIELD = {
    "qx3_diagnostic_source_sha256": "qx3_diagnostic",
    "hardware41_source_sha256": "hardware41",
    "known_audit_source_sha256": "known_audit",
    "calibration_source_sha256": "calibration",
    "blind_test_source_sha256": "blind_test",
}
PROTECTED_COHORTS = frozenset(PROTECTED_COHORT_BY_FIELD.values())
CANDIDATE_ROLES = frozenset({"train", "model_validation"})
VIEW_KINDS = frozenset({"source", "crop"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUNTIME_CODE_FINGERPRINT_SCHEMA = "oneexpo_auditor_runtime_code.v1"

# Capture the immutable algorithm/coverage contract at import time. Unit tests
# may deliberately replace the production protected-count mapping after import;
# that fixture adjustment must not masquerade as a different loaded auditor.
# Conversely, loading altered source and then restoring the file still leaves
# the altered import-time snapshot (and bytecode) visible to this fingerprint.
_RUNTIME_FINGERPRINT_CONTRACT = (
    ("report_schema", REPORT_SCHEMA),
    ("algorithm_id", ALGORITHM_ID),
    ("phash_distance", PHASH_DISTANCE),
    ("max_encoded_bytes", MAX_ENCODED_BYTES),
    ("max_image_pixels", MAX_IMAGE_PIXELS),
    ("max_graph_edges", MAX_GRAPH_EDGES),
    ("protected_inventory_schema", PROTECTED_INVENTORY_SCHEMA),
    ("protected_sources_schema", PROTECTED_SOURCES_SCHEMA),
    ("protected_source_fields", PROTECTED_SOURCE_FIELDS),
    (
        "protected_required_counts",
        tuple(sorted(PROTECTED_REQUIRED_COUNTS.items())),
    ),
    (
        "protected_cohort_by_field",
        tuple(sorted(PROTECTED_COHORT_BY_FIELD.items())),
    ),
    ("protected_cohorts", tuple(sorted(PROTECTED_COHORTS))),
    ("candidate_roles", tuple(sorted(CANDIDATE_ROLES))),
    ("view_kinds", tuple(sorted(VIEW_KINDS))),
    ("sha256_pattern", SHA256_RE.pattern),
    ("sha256_flags", SHA256_RE.flags),
)


class AuditError(ValueError):
    """Raised when the audit cannot establish complete, trustworthy coverage."""


@dataclass(frozen=True)
class AuditAsset:
    path: Path
    role: str
    cohort: str
    view_kind: str
    sample_id: str
    source_sha256: str
    image_sha256: str


@dataclass(frozen=True)
class _ReadIdentity:
    device: int
    inode: int
    mode: int
    nlink: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class VerifiedAsset:
    path: Path
    role: str
    cohort: str
    view_kind: str
    sample_id: str
    source_sha256: str
    image_sha256: str
    size: int
    width: int
    height: int
    signature: tuple[int, int, int, int]
    identity: _ReadIdentity


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _report_bytes(value: object) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _runtime_fingerprint_value(value: object) -> object:
    """Normalize Python runtime values without source paths or line tables."""
    if value is None:
        return {"type": "none"}
    if value is Ellipsis:
        return {"type": "ellipsis"}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": str(value)}
    if type(value) is float:
        return {"type": "float", "value": value.hex()}
    if type(value) is complex:
        return {
            "type": "complex",
            "real": value.real.hex(),
            "imag": value.imag.hex(),
        }
    if type(value) is str:
        return {"type": "str", "value": value}
    if type(value) is bytes:
        return {
            "type": "bytes",
            "value": base64.b64encode(value).decode("ascii"),
        }
    if type(value) is tuple:
        return {
            "type": "tuple",
            "items": [_runtime_fingerprint_value(item) for item in value],
        }
    if type(value) is list:
        return {
            "type": "list",
            "items": [_runtime_fingerprint_value(item) for item in value],
        }
    if type(value) in {set, frozenset}:
        items = [_runtime_fingerprint_value(item) for item in value]
        items.sort(key=_canonical_bytes)
        return {
            "type": "frozenset" if type(value) is frozenset else "set",
            "items": items,
        }
    if type(value) is dict:
        items = [
            {
                "key": _runtime_fingerprint_value(key),
                "value": _runtime_fingerprint_value(item),
            }
            for key, item in value.items()
        ]
        items.sort(key=_canonical_bytes)
        return {"type": "dict", "items": items}
    if isinstance(value, types.CodeType):
        return {"type": "code", "value": _runtime_code_payload(value)}
    raise AuditError(
        "unsupported live-runtime fingerprint value: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _runtime_code_payload(code: types.CodeType) -> dict[str, object]:
    """Return stable semantic bytecode metadata for one live code object."""
    return {
        "name": code.co_name,
        "qualname": code.co_qualname,
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "bytecode": base64.b64encode(code.co_code).decode("ascii"),
        "constants": [
            _runtime_fingerprint_value(value) for value in code.co_consts
        ],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "exception_table": base64.b64encode(
            getattr(code, "co_exceptiontable", b"")
        ).decode("ascii"),
    }


def runtime_code_fingerprint_sha256() -> str:
    """Hash the auditor implementation actually loaded in this interpreter.

    Filenames, first-line numbers, and line tables are excluded so an identical
    module loaded from a different immutable root has the same fingerprint.
    Function bytecode (including nested code), defaults, and the fixed import-
    time contract are included, so restoring altered source bytes after import
    cannot make the already-loaded implementation look pristine.
    """
    functions: list[dict[str, object]] = []
    for name, value in sorted(globals().items()):
        if type(value) is not types.FunctionType or value.__module__ != __name__:
            continue
        functions.append(
            {
                "name": name,
                "code": _runtime_code_payload(value.__code__),
                "defaults": _runtime_fingerprint_value(value.__defaults__),
                "kwdefaults": _runtime_fingerprint_value(value.__kwdefaults__),
            }
        )
    classes: list[dict[str, object]] = []
    for name, value in sorted(globals().items()):
        if type(value) is not type or value.__module__ != __name__:
            continue
        methods: list[dict[str, object]] = []
        for member_name, member in sorted(vars(value).items()):
            function: types.FunctionType | None = None
            if type(member) is types.FunctionType:
                function = member
            elif type(member) in {staticmethod, classmethod}:
                function = member.__func__
            if (
                function is None
                or function.__module__ != __name__
                or function.__code__.co_filename != __file__
            ):
                continue
            methods.append(
                {
                    "name": member_name,
                    "code": _runtime_code_payload(function.__code__),
                    "defaults": _runtime_fingerprint_value(function.__defaults__),
                    "kwdefaults": _runtime_fingerprint_value(function.__kwdefaults__),
                }
            )
        classes.append(
            {
                "name": name,
                "qualname": value.__qualname__,
                "bases": [
                    f"{base.__module__}.{base.__qualname__}"
                    for base in value.__bases__
                ],
                "methods": methods,
            }
        )
    if not functions or not classes:
        raise AuditError("live-runtime auditor function inventory is empty")
    payload = {
        "schema": RUNTIME_CODE_FINGERPRINT_SCHEMA,
        "contract": _runtime_fingerprint_value(_RUNTIME_FINGERPRINT_CONTRACT),
        "module_code": _runtime_code_payload(_RUNTIME_MODULE_CODE),
        "functions": functions,
        "classes": classes,
    }
    return _sha256_bytes(_canonical_bytes(payload))


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise AuditError(f"{name} must be a lowercase SHA-256")
    return value


def _identity(value: os.stat_result) -> _ReadIdentity:
    return _ReadIdentity(
        device=int(value.st_dev),
        inode=int(value.st_ino),
        mode=int(value.st_mode),
        nlink=int(value.st_nlink),
        size=int(value.st_size),
        mtime_ns=int(value.st_mtime_ns),
        ctime_ns=int(value.st_ctime_ns),
    )


def _path_identity_matches(
    path_identity: _ReadIdentity,
    descriptor_identity: _ReadIdentity,
    *,
    expected_nlink: int,
) -> bool:
    """Compare stable path/handle identity fields across Windows and POSIX."""
    return (
        path_identity.device == descriptor_identity.device
        and path_identity.inode == descriptor_identity.inode
        and path_identity.mode == descriptor_identity.mode
        and path_identity.nlink == expected_nlink
        and path_identity.size == descriptor_identity.size
        and path_identity.mtime_ns == descriptor_identity.mtime_ns
    )


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_chain(path: Path, *, stop_at: Path | None = None) -> None:
    current = _absolute_without_resolving(path)
    stop = _absolute_without_resolving(stop_at) if stop_at is not None else None
    chain: list[Path] = []
    while True:
        chain.append(current)
        if stop is not None and current == stop:
            break
        parent = current.parent
        if parent == current:
            if stop is not None:
                raise AuditError(f"path is outside declared root: {path}")
            break
        current = parent
    for item in reversed(chain):
        try:
            item_stat = os.lstat(item)
        except OSError as error:
            raise AuditError(f"cannot inspect path component: {error}") from error
        file_attributes = int(getattr(item_stat, "st_file_attributes", 0))
        reparse_mask = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        if stat.S_ISLNK(item_stat.st_mode) or file_attributes & reparse_mask:
            raise AuditError(f"symlink/reparse path component is forbidden: {item}")


def _read_regular_file(
    path: Path,
    *,
    expected_sha256: str | None,
    max_bytes: int | None = None,
    root: Path | None = None,
) -> tuple[bytes, str, _ReadIdentity]:
    """Read a bounded regular file once and bind its bytes to one descriptor."""
    if max_bytes is None:
        max_bytes = MAX_ENCODED_BYTES
    if max_bytes <= 0:
        raise AuditError("max_bytes must be positive")
    absolute = _absolute_without_resolving(path)
    if root is not None:
        absolute_root = _absolute_without_resolving(root)
        try:
            common = Path(os.path.commonpath((absolute_root, absolute)))
        except ValueError as error:
            raise AuditError("path and root are on different volumes") from error
        if common != absolute_root:
            raise AuditError(f"path is outside declared root: {path}")
        _reject_symlink_chain(absolute, stop_at=absolute_root)
    else:
        _reject_symlink_chain(absolute)

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise AuditError(f"cannot open audited file: {error}") from error
    try:
        before_stat = os.fstat(descriptor)
        before = _identity(before_stat)
        if not stat.S_ISREG(before.mode):
            raise AuditError("audited path is not a regular file")
        if before.nlink != 1:
            raise AuditError("audited file must have exactly one hard link")
        if before.size <= 0:
            raise AuditError("audited file is empty")
        if before.size > max_bytes:
            raise AuditError(f"audited file exceeds byte cap {max_bytes}")

        remaining = before.size
        payload = bytearray()
        digest = hashlib.sha256()
        while remaining:
            try:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
            except OSError as error:
                raise AuditError(f"audited file read failed: {error}") from error
            if not chunk:
                raise AuditError("short read while consuming audited file")
            if len(chunk) > remaining:
                raise AuditError("audited file read exceeded declared size")
            payload.extend(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise AuditError("audited file grew while it was read")

        after = _identity(os.fstat(descriptor))
        if after != before:
            raise AuditError("audited file identity changed while it was read")
        try:
            path_stat = os.lstat(absolute)
        except OSError as error:
            raise AuditError("audited path disappeared after read") from error
        path_attributes = int(getattr(path_stat, "st_file_attributes", 0))
        reparse_mask = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        if stat.S_ISLNK(path_stat.st_mode) or path_attributes & reparse_mask:
            raise AuditError("audited path became a symlink/reparse point")
        path_identity = _identity(path_stat)
        if (
            path_identity.device != after.device
            or path_identity.inode != after.inode
            or path_identity.mode != after.mode
            or path_identity.nlink != after.nlink
            or path_identity.size != after.size
            or path_identity.mtime_ns != after.mtime_ns
        ):
            raise AuditError("audited path no longer names the opened file")
        _reject_symlink_chain(absolute, stop_at=root if root is not None else None)
    finally:
        os.close(descriptor)

    actual_sha = digest.hexdigest()
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise AuditError("audited file SHA-256 does not match its declaration")
    return bytes(payload), actual_sha, before


def _decode_verified_image(payload: bytes) -> tuple[np.ndarray, int, int]:
    try:
        with Image.open(io.BytesIO(payload)) as header:
            width, height = (int(header.size[0]), int(header.size[1]))
            if width <= 0 or height <= 0:
                raise AuditError("image dimensions must be positive")
            if width * height > MAX_IMAGE_PIXELS:
                raise AuditError(f"image exceeds pixel cap {MAX_IMAGE_PIXELS}")
            header.verify()
    except AuditError:
        raise
    except (OSError, SyntaxError, ValueError, UnidentifiedImageError) as error:
        raise AuditError(f"Pillow could not verify image header: {error}") from error

    encoded = np.frombuffer(payload, dtype=np.uint8)
    flags = int(cv2.IMREAD_GRAYSCALE) | int(cv2.IMREAD_IGNORE_ORIENTATION)
    try:
        image = cv2.imdecode(encoded, flags)
    except cv2.error as error:
        raise AuditError(f"OpenCV could not decode image: {error}") from error
    if image is None or image.dtype != np.uint8 or image.ndim != 2:
        raise AuditError("OpenCV image decode returned invalid pixels")
    if int(image.shape[1]) != width or int(image.shape[0]) != height:
        raise AuditError("Pillow/OpenCV decoded dimensions disagree")
    return image, width, height


def _phash64(image: np.ndarray) -> int:
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(np.asarray(resized, dtype=np.float32))[:8, :8].reshape(-1)
    median = float(np.median(coefficients[1:]))
    bits = coefficients > median
    bits[0] = False
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def _phash_signature(payload: bytes) -> tuple[tuple[int, int, int, int], int, int]:
    image, width, height = _decode_verified_image(payload)
    views = (
        image,
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(image, cv2.ROTATE_180),
        cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
    )
    signature = tuple(sorted(_phash64(view) for view in views))
    return signature, width, height


def _validate_asset_fields(asset: AuditAsset, *, protected: bool) -> None:
    if not isinstance(asset.path, Path):
        raise AuditError("asset.path must be pathlib.Path")
    if asset.view_kind not in VIEW_KINDS:
        raise AuditError("asset view_kind must be source or crop")
    if not asset.sample_id or asset.sample_id.strip() != asset.sample_id:
        raise AuditError("asset sample_id must be non-empty and trimmed")
    _require_sha256(asset.source_sha256, "asset.source_sha256")
    _require_sha256(asset.image_sha256, "asset.image_sha256")
    if asset.view_kind == "source" and asset.image_sha256 != asset.source_sha256:
        raise AuditError("source view image_sha256 must equal source_sha256")
    if protected:
        if asset.cohort not in PROTECTED_COHORTS or asset.role != asset.cohort:
            raise AuditError("protected asset role/cohort mapping is invalid")
    elif asset.role not in CANDIDATE_ROLES or asset.cohort != "candidate":
        raise AuditError("candidate asset role/cohort mapping is invalid")


def _verify_audit_asset(
    asset: AuditAsset,
    *,
    protected: bool,
    root: Path | None = None,
    declared_size: int | None = None,
) -> VerifiedAsset:
    _validate_asset_fields(asset, protected=protected)
    payload, actual_sha, identity = _read_regular_file(
        asset.path,
        expected_sha256=asset.image_sha256,
        root=root,
    )
    if declared_size is not None and identity.size != declared_size:
        raise AuditError("protected inventory size does not match audited file")
    signature, width, height = _phash_signature(payload)
    if actual_sha != asset.image_sha256:
        raise AuditError("verified image SHA unexpectedly changed")
    return VerifiedAsset(
        path=asset.path,
        role=asset.role,
        cohort=asset.cohort,
        view_kind=asset.view_kind,
        sample_id=asset.sample_id,
        source_sha256=asset.source_sha256,
        image_sha256=asset.image_sha256,
        size=identity.size,
        width=width,
        height=height,
        signature=signature,
        identity=identity,
    )


def reverify_assets(records: Iterable[VerifiedAsset]) -> None:
    """Re-read every asset and reject any byte, identity, or decode drift."""
    for record in records:
        payload, actual_sha, identity = _read_regular_file(
            record.path,
            expected_sha256=record.image_sha256,
        )
        signature, width, height = _phash_signature(payload)
        if actual_sha != record.image_sha256 or identity != record.identity:
            raise AuditError("asset byte SHA or identity changed after initial verification")
        if (
            identity.size != record.size
            or width != record.width
            or height != record.height
            or signature != record.signature
        ):
            raise AuditError("asset decode or pHash changed after initial verification")


def _normalized_relative_path(value: object, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise AuditError(f"{name} must be a normalized relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or value in {".", ".."} or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise AuditError(f"{name} must be a normalized relative POSIX path")
    if path.as_posix() != value:
        raise AuditError(f"{name} is not normalized")
    return path


def _path_below(root: Path, relative: PurePosixPath) -> Path:
    target = root.joinpath(*relative.parts)
    absolute_root = _absolute_without_resolving(root)
    absolute_target = _absolute_without_resolving(target)
    try:
        common = Path(os.path.commonpath((absolute_root, absolute_target)))
    except ValueError as error:
        raise AuditError("inventory path and root are on different volumes") from error
    if common != absolute_root:
        raise AuditError("inventory object path escapes root")
    return absolute_target


def _load_json_file(path: Path, name: str) -> tuple[Mapping[str, object], bytes, str]:
    payload, sha256, _ = _read_regular_file(path, expected_sha256=None)

    def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise AuditError(f"{name} contains duplicate object key {key!r}")
            result[key] = item
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=strict_object,
        )
    except AuditError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AuditError(f"{name} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, Mapping):
        raise AuditError(f"{name} root must be an object")
    return value, payload, sha256


def _load_protected_sources(
    path: Path,
) -> tuple[dict[str, str], set[str], bytes, str, str]:
    value, payload, file_sha = _load_json_file(path, "protected sources")
    expected_keys = {
        "schema",
        "artifact_role",
        "status",
        *PROTECTED_SOURCE_FIELDS,
    }
    if set(value) != expected_keys:
        raise AuditError("protected sources top-level fields mismatch")
    if value.get("schema") != PROTECTED_SOURCES_SCHEMA:
        raise AuditError("protected sources schema mismatch")
    if value.get("artifact_role") != "protected_holdouts_not_training_or_model_selection":
        raise AuditError("protected sources artifact_role mismatch")
    if value.get("status") != "protected_holdouts_ready":
        raise AuditError("protected sources status mismatch")

    per_field: dict[str, set[str]] = {}
    for field in PROTECTED_SOURCE_FIELDS:
        items = value.get(field)
        if type(items) is not list:
            raise AuditError(f"protected sources {field} must be an array")
        normalized = [_require_sha256(item, f"protected sources {field}") for item in items]
        if len(set(normalized)) != len(normalized):
            raise AuditError(f"protected sources {field} contains duplicates")
        per_field[field] = set(normalized)
    for field, expected_count in PROTECTED_REQUIRED_COUNTS.items():
        if len(per_field[field]) != expected_count:
            raise AuditError(
                f"protected sources {field} must contain exactly {expected_count} SHAs"
            )
    if not per_field["qx3_validation_source_sha256"].issubset(
        per_field["qx3_diagnostic_source_sha256"]
    ):
        raise AuditError("qx3 validation must be a qx3 diagnostic subset")

    primary_fields = tuple(PROTECTED_COHORT_BY_FIELD)
    for index, left in enumerate(primary_fields):
        for right in primary_fields[index + 1 :]:
            if per_field[left].intersection(per_field[right]):
                raise AuditError("protected primary cohorts overlap")
    cohort_by_sha: dict[str, str] = {}
    for field, cohort in PROTECTED_COHORT_BY_FIELD.items():
        for sha256 in per_field[field]:
            cohort_by_sha[sha256] = cohort
    union = set(cohort_by_sha)
    union_sha = _sha256_bytes(_canonical_bytes(sorted(union)))
    return cohort_by_sha, union, payload, file_sha, union_sha


def _load_protected_inventory(
    path: Path,
    *,
    cohort_by_sha: Mapping[str, str],
    expected_union: set[str],
) -> tuple[list[VerifiedAsset], bytes, str, str]:
    value, payload, file_sha = _load_json_file(path, "protected inventory")
    if set(value) != {"schema", "root", "objects"}:
        raise AuditError("protected inventory top-level fields mismatch")
    if value.get("schema") != PROTECTED_INVENTORY_SCHEMA:
        raise AuditError("protected inventory schema mismatch")
    root_relative = _normalized_relative_path(value.get("root"), "protected inventory root")
    root = _path_below(path.parent, root_relative)
    _reject_symlink_chain(root)
    objects = value.get("objects")
    if type(objects) is not list:
        raise AuditError("protected inventory objects must be an array")

    prepared: list[tuple[AuditAsset, int, str]] = []
    seen_paths: set[str] = set()
    for index, raw in enumerate(objects):
        if not isinstance(raw, Mapping) or set(raw) != {
            "cohort",
            "view_kind",
            "path",
            "size",
            "image_sha256",
            "source_sha256",
        }:
            raise AuditError(f"protected inventory object {index} fields mismatch")
        cohort = raw.get("cohort")
        view_kind = raw.get("view_kind")
        if not isinstance(cohort, str) or cohort not in PROTECTED_COHORTS:
            raise AuditError(f"protected inventory object {index} cohort is invalid")
        if not isinstance(view_kind, str) or view_kind not in VIEW_KINDS:
            raise AuditError(f"protected inventory object {index} view_kind is invalid")
        relative = _normalized_relative_path(raw.get("path"), f"inventory object {index} path")
        relative_text = relative.as_posix()
        if relative_text in seen_paths:
            raise AuditError("protected inventory contains duplicate object paths")
        seen_paths.add(relative_text)
        size = raw.get("size")
        if type(size) is not int or not 0 < size <= MAX_ENCODED_BYTES:
            raise AuditError(f"protected inventory object {index} size is invalid")
        image_sha = _require_sha256(raw.get("image_sha256"), "inventory image_sha256")
        source_sha = _require_sha256(raw.get("source_sha256"), "inventory source_sha256")
        if source_sha not in expected_union:
            raise AuditError("protected inventory contains an extra source SHA")
        if cohort_by_sha[source_sha] != cohort:
            raise AuditError("protected inventory source SHA is misclassified")
        if view_kind == "source" and image_sha != source_sha:
            raise AuditError("protected source view image SHA must equal source SHA")
        sample_suffix = _sha256_bytes(relative_text.encode("utf-8"))[:16]
        asset = AuditAsset(
            path=_path_below(root, relative),
            role=cohort,
            cohort=cohort,
            view_kind=view_kind,
            sample_id=f"protected:{cohort}:{view_kind}:{image_sha[:16]}:{sample_suffix}",
            source_sha256=source_sha,
            image_sha256=image_sha,
        )
        prepared.append((asset, size, relative_text))

    inventory_sources = {asset.source_sha256 for asset, _, _ in prepared}
    if inventory_sources != expected_union:
        missing = sorted(expected_union - inventory_sources)
        extra = sorted(inventory_sources - expected_union)
        raise AuditError(
            f"protected inventory source union mismatch: missing={missing[:4]}, extra={extra[:4]}"
        )
    source_view_counts: dict[str, int] = defaultdict(int)
    crop_view_counts: dict[str, int] = defaultdict(int)
    for asset, _, _ in prepared:
        if asset.view_kind == "source":
            source_view_counts[asset.source_sha256] += 1
        elif asset.view_kind == "crop":
            crop_view_counts[asset.source_sha256] += 1
    missing = sorted(expected_union - set(source_view_counts))
    duplicate_source_views = sorted(
        sha256 for sha256, count in source_view_counts.items() if count != 1
    )
    missing_crop_views = sorted(expected_union - set(crop_view_counts))
    duplicate_crop_views = sorted(
        sha256 for sha256, count in crop_view_counts.items() if count != 1
    )
    if (
        missing
        or duplicate_source_views
        or missing_crop_views
        or duplicate_crop_views
    ):
        raise AuditError(
            "protected source and crop views must cover every source exactly once: "
            f"missing_source={missing[:4]}, "
            f"duplicate_source={duplicate_source_views[:4]}, "
            f"missing_crop={missing_crop_views[:4]}, "
            f"duplicate_crop={duplicate_crop_views[:4]}"
        )

    listed_paths = {relative for _, _, relative in prepared}
    discovered_paths: set[str] = set()
    def walk_error(error: OSError) -> None:
        raise AuditError(f"cannot enumerate protected root: {error}") from error

    for directory, directory_names, file_names in os.walk(
        root,
        followlinks=False,
        onerror=walk_error,
    ):
        directory_path = Path(directory)
        for name in list(directory_names):
            component = directory_path / name
            component_stat = os.lstat(component)
            attributes = int(getattr(component_stat, "st_file_attributes", 0))
            reparse_mask = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
            if stat.S_ISLNK(component_stat.st_mode) or attributes & reparse_mask:
                raise AuditError("protected root contains a symlink/reparse directory")
        for name in file_names:
            component = directory_path / name
            component_stat = os.lstat(component)
            attributes = int(getattr(component_stat, "st_file_attributes", 0))
            reparse_mask = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
            if stat.S_ISLNK(component_stat.st_mode) or attributes & reparse_mask:
                raise AuditError("protected root contains a symlink/reparse file")
            if not stat.S_ISREG(component_stat.st_mode):
                raise AuditError("protected root contains a special file")
            discovered_paths.add(component.relative_to(root).as_posix())
    if discovered_paths != listed_paths:
        raise AuditError(
            "protected inventory does not exactly enumerate root files: "
            f"missing={sorted(discovered_paths - listed_paths)[:4]}, "
            f"extra={sorted(listed_paths - discovered_paths)[:4]}"
        )

    verified = [
        _verify_audit_asset(asset, protected=True, root=root, declared_size=size)
        for asset, size, _ in sorted(
            prepared,
            key=lambda item: (
                item[0].cohort,
                item[0].view_kind,
                item[0].source_sha256,
                item[0].image_sha256,
                item[2],
            ),
        )
    ]
    payload_sha = _sha256_bytes(_canonical_bytes(value))
    return verified, payload, file_sha, payload_sha


def _asset_id(record: VerifiedAsset) -> str:
    value = {
        "cohort": record.cohort,
        "image_sha256": record.image_sha256,
        "role": record.role,
        "sample_id": record.sample_id,
        "source_sha256": record.source_sha256,
        "view_kind": record.view_kind,
    }
    return _sha256_bytes(_canonical_bytes(value))


def _entry(record: VerifiedAsset) -> dict[str, object]:
    return {
        "asset_id": _asset_id(record),
        "role": record.role,
        "cohort": record.cohort,
        "view_kind": record.view_kind,
        "sample_id": record.sample_id,
        "source_sha256": record.source_sha256,
        "image_sha256": record.image_sha256,
        "size": record.size,
        "width": record.width,
        "height": record.height,
        "phash_rot4": [f"{value:016x}" for value in record.signature],
    }


def _bucket_keys(value: int, threshold: int) -> tuple[tuple[int, int], ...]:
    if not 0 <= threshold <= 7:
        raise AuditError("pHash threshold must be between 0 and 7")
    widths = [64 // (threshold + 1)] * (threshold + 1)
    for index in range(64 % (threshold + 1)):
        widths[index] += 1
    keys: list[tuple[int, int]] = []
    offset = 0
    for index, width in enumerate(widths):
        keys.append((index, (value >> offset) & ((1 << width) - 1)))
        offset += width
    return tuple(keys)


def _signature_distance(left: Sequence[int], right: Sequence[int]) -> int:
    return min((a ^ b).bit_count() for a in left for b in right)


def _near_pairs_from_signatures(
    signatures: Sequence[Sequence[int]], threshold: int = PHASH_DISTANCE,
    *, max_pairs: int = MAX_GRAPH_EDGES,
) -> list[tuple[int, int, int]]:
    """Return every pair within the radius using lossless pigeonhole buckets."""
    buckets: dict[tuple[int, int], set[int]] = defaultdict(set)
    pairs: list[tuple[int, int, int]] = []
    for current_index, signature in enumerate(signatures):
        unique_hashes = tuple(sorted(set(int(value) for value in signature)))
        if not unique_hashes or any(value < 0 or value >= 1 << 64 for value in unique_hashes):
            raise AuditError("pHash signatures must contain unsigned 64-bit values")
        candidates: set[int] = set()
        for value in unique_hashes:
            for key in _bucket_keys(value, threshold):
                candidates.update(buckets.get(key, ()))
        for other_index in sorted(candidates):
            distance = _signature_distance(signatures[other_index], unique_hashes)
            if distance <= threshold:
                pairs.append((other_index, current_index, distance))
                if len(pairs) > max_pairs:
                    raise AuditError(
                        f"near-duplicate graph exceeds edge cap {max_pairs}"
                    )
        for value in unique_hashes:
            for key in _bucket_keys(value, threshold):
                buckets[key].add(current_index)
    return pairs


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        self.parent[high] = low


def _graph_evidence(
    records: Sequence[VerifiedAsset],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    entries = [_entry(record) for record in records]
    edge_values: dict[tuple[int, int], dict[str, object]] = {}

    for left, right, distance in _near_pairs_from_signatures(
        [record.signature for record in records], PHASH_DISTANCE
    ):
        edge_values[(left, right)] = {
            "distance": distance,
            "evidence": {"perceptual_hash"},
        }

    by_image: dict[str, list[int]] = defaultdict(list)
    by_source: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_image[record.image_sha256].append(index)
        by_source[record.source_sha256].append(index)
    for evidence_name, groups in (
        ("exact_image_sha256", by_image),
        ("source_sha256", by_source),
    ):
        for indexes in groups.values():
            for position, left in enumerate(indexes):
                for right in indexes[position + 1 :]:
                    key = (left, right)
                    edge = edge_values.setdefault(
                        key,
                        {"distance": None, "evidence": set()},
                    )
                    if len(edge_values) > MAX_GRAPH_EDGES:
                        raise AuditError(
                            f"near-duplicate graph exceeds edge cap {MAX_GRAPH_EDGES}"
                        )
                    edge["evidence"].add(evidence_name)  # type: ignore[union-attr]
                    if evidence_name == "exact_image_sha256":
                        edge["distance"] = 0

    union_find = _UnionFind(len(records))
    for left, right in sorted(edge_values):
        union_find.union(left, right)

    rendered_edges: list[dict[str, object]] = []
    for (left, right), value in sorted(
        edge_values.items(),
        key=lambda item: (
            str(entries[item[0][0]]["asset_id"]),
            str(entries[item[0][1]]["asset_id"]),
        ),
    ):
        left_id, right_id = sorted(
            (str(entries[left]["asset_id"]), str(entries[right]["asset_id"]))
        )
        rendered_edges.append(
            {
                "left_asset_id": left_id,
                "right_asset_id": right_id,
                "distance": value["distance"],
                "evidence": sorted(value["evidence"]),
                "blocking": records[left].role != records[right].role,
            }
        )

    members_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        members_by_root[union_find.find(index)].append(index)
    edge_count_by_root: dict[int, int] = defaultdict(int)
    for left, _right in edge_values:
        edge_count_by_root[union_find.find(left)] += 1

    clusters: list[dict[str, object]] = []
    for root, indexes in members_by_root.items():
        member_image_shas = sorted({records[index].image_sha256 for index in indexes})
        cluster_preimage = {
            "algorithm_id": ALGORITHM_ID,
            "threshold": PHASH_DISTANCE,
            "member_image_sha256s": member_image_shas,
        }
        roles = sorted({records[index].role for index in indexes})
        clusters.append(
            {
                "cluster_id": _sha256_bytes(_canonical_bytes(cluster_preimage)),
                "member_asset_ids": sorted(str(entries[index]["asset_id"]) for index in indexes),
                "member_image_sha256s": member_image_shas,
                "roles": roles,
                "cohorts": sorted({records[index].cohort for index in indexes}),
                "view_kinds": sorted({records[index].view_kind for index in indexes}),
                "edge_count": edge_count_by_root[root],
                "multi_role": len(roles) > 1,
                "blocking": len(roles) > 1,
            }
        )
    clusters.sort(key=lambda item: str(item["cluster_id"]))
    return rendered_edges, clusters


def _payload_set_sha(records: Sequence[VerifiedAsset]) -> str:
    return _sha256_bytes(
        _canonical_bytes(sorted({record.image_sha256 for record in records}))
    )


def _auditor_binding() -> dict[str, str]:
    path = Path(__file__)
    _payload, sha256, _identity_value = _read_regular_file(
        path,
        expected_sha256=None,
    )
    return {
        "path": "scripts/audit_v4_near_duplicate_leakage.py",
        "sha256": sha256,
        "runtime_code_sha256": runtime_code_fingerprint_sha256(),
    }


def build_near_duplicate_report(
    candidate_assets: Sequence[AuditAsset],
    candidate_manifest_sha256: dict[str, str],
    protected_sources_path: Path,
    protected_inventory_path: Path,
) -> tuple[dict, bytes, tuple[VerifiedAsset, ...]]:
    """Build a deterministic, complete separation report from verified bytes."""
    if set(candidate_manifest_sha256) != CANDIDATE_ROLES:
        raise AuditError("candidate manifest bindings must contain train and model_validation")
    normalized_manifest_bindings = {
        role: _require_sha256(candidate_manifest_sha256[role], f"candidate manifest {role}")
        for role in sorted(CANDIDATE_ROLES)
    }
    if not candidate_assets:
        raise AuditError("candidate asset set is empty")
    if {asset.role for asset in candidate_assets} != CANDIDATE_ROLES:
        raise AuditError("candidate assets must cover train and model_validation")
    source_view_counts: dict[tuple[str, str], int] = defaultdict(int)
    crop_view_counts: dict[tuple[str, str], int] = defaultdict(int)
    referenced_sources: set[tuple[str, str]] = set()
    for asset in candidate_assets:
        referenced_sources.add((asset.role, asset.source_sha256))
        if asset.view_kind == "source":
            source_view_counts[(asset.role, asset.source_sha256)] += 1
        elif asset.view_kind == "crop":
            crop_view_counts[(asset.role, asset.source_sha256)] += 1
    missing_source_views = sorted(referenced_sources - set(source_view_counts))
    duplicate_source_views = sorted(
        key for key, count in source_view_counts.items() if count != 1
    )
    missing_crop_views = sorted(referenced_sources - set(crop_view_counts))
    duplicate_crop_views = sorted(
        key for key, count in crop_view_counts.items() if count != 1
    )
    if (
        missing_source_views
        or duplicate_source_views
        or missing_crop_views
        or duplicate_crop_views
    ):
        raise AuditError(
            "candidate source and crop views must cover every role/source exactly once: "
            f"missing_source={missing_source_views[:4]}, "
            f"duplicate_source={duplicate_source_views[:4]}, "
            f"missing_crop={missing_crop_views[:4]}, "
            f"duplicate_crop={duplicate_crop_views[:4]}"
        )

    candidate_records = [
        _verify_audit_asset(asset, protected=False) for asset in candidate_assets
    ]
    candidate_records.sort(key=lambda record: _canonical_bytes(_entry(record)))
    candidate_ids = [_asset_id(record) for record in candidate_records]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise AuditError("candidate assets contain duplicate stable identities")

    cohort_by_sha, protected_union, protected_sources_payload, protected_sources_file_sha, protected_union_sha = (
        _load_protected_sources(protected_sources_path)
    )
    protected_records, protected_inventory_payload, protected_inventory_file_sha, protected_inventory_payload_sha = (
        _load_protected_inventory(
            protected_inventory_path,
            cohort_by_sha=cohort_by_sha,
            expected_union=protected_union,
        )
    )
    protected_records.sort(key=lambda record: _canonical_bytes(_entry(record)))
    all_records = tuple(candidate_records + protected_records)
    reverify_assets(all_records)

    edges, clusters = _graph_evidence(all_records)
    entries = sorted((_entry(record) for record in all_records), key=lambda item: str(item["asset_id"]))
    blocking_clusters = [cluster for cluster in clusters if cluster["blocking"]]
    same_role_duplicate_clusters = [
        cluster
        for cluster in clusters
        if not cluster["blocking"] and int(cluster["edge_count"]) > 0
    ]
    algorithm = {
        "id": ALGORITHM_ID,
        "threshold": PHASH_DISTANCE,
        "decode": "verified_bytes_cv2_grayscale_ignore_exif_orientation",
        "views": ["rot0", "rot90", "rot180", "rot270"],
        "resize": {"width": 32, "height": 32, "interpolation": "INTER_AREA"},
        "dct": {"dtype": "float32", "low_frequency_block": [8, 8]},
        "bit_rule": "row_major_msb_first; median(coefficients[1:]); coefficient>median; dc=0",
        "byte_cap": MAX_ENCODED_BYTES,
        "pixel_cap": MAX_IMAGE_PIXELS,
        "graph_edge_cap": MAX_GRAPH_EDGES,
        "exact_right_angle_rotation_invariant": True,
        "crop_invariant": False,
        "runtime": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "opencv_build_information_sha256": _sha256_bytes(
                cv2.getBuildInformation().encode("utf-8")
            ),
        },
    }
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": "blocked" if blocking_clusters else "passed",
        "ok": not blocking_clusters,
        "artifact_role": "candidate_dataset_separation_evidence_only",
        "authority": {
            "candidate_only": True,
            "label_authority": False,
            "blind_authority": False,
            "promotion_authority": False,
            "deployment_authority": False,
            "automatic_delete_or_relabel": False,
        },
        "algorithm": algorithm,
        "bindings": {
            "candidate_manifest_sha256": normalized_manifest_bindings,
            "candidate_payload_set_sha256": _payload_set_sha(candidate_records),
            "protected_payload_set_sha256": _payload_set_sha(protected_records),
            "protected_sources": {
                "file_sha256": protected_sources_file_sha,
                "payload_sha256": _sha256_bytes(_canonical_bytes(json.loads(protected_sources_payload))),
                "canonical_union_sha256": protected_union_sha,
            },
            "protected_inventory": {
                "file_sha256": protected_inventory_file_sha,
                "payload_sha256": protected_inventory_payload_sha,
            },
            "auditor": _auditor_binding(),
        },
        "coverage": {
            "candidate_assets": len(candidate_records),
            "protected_assets": len(protected_records),
            "protected_source_union": len(protected_union),
            "verified_assets": len(all_records),
            "complete": True,
        },
        "summary": {
            "edges": len(edges),
            "clusters": len(clusters),
            "blocking_multi_role_clusters": len(blocking_clusters),
            "same_role_duplicate_clusters_nonblocking": len(same_role_duplicate_clusters),
        },
        "entries": entries,
        "edges": edges,
        "clusters": clusters,
    }
    rendered = _report_bytes(report)
    return report, rendered, all_records


def _manifest_relative_path(root: Path, value: str, name: str) -> Path:
    relative = _normalized_relative_path(value, name)
    return _path_below(root, relative)


def _candidate_source_path(root: Path, value: str, name: str) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return _absolute_without_resolving(raw)
    return _manifest_relative_path(root, value, name)


def _decode_source_path(value: str) -> str:
    try:
        raw = base64.urlsafe_b64decode(value.encode("ascii"))
        return os.fsdecode(raw)
    except (UnicodeError, ValueError, binascii.Error) as error:
        raise AuditError("candidate source_path_b64 is invalid") from error


def _load_candidate_manifest(
    role: str, path: Path, *, max_bytes: int = MAX_ENCODED_BYTES,
    allow_absolute_crop_paths: bool = False,
) -> tuple[list[AuditAsset], str]:
    if role not in CANDIDATE_ROLES:
        raise AuditError(f"unsupported candidate role: {role}")
    if type(allow_absolute_crop_paths) is not bool:
        raise AuditError("allow_absolute_crop_paths must be boolean")
    payload, manifest_sha, _ = _read_regular_file(path, expected_sha256=None, max_bytes=max_bytes)
    try:
        text = payload.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        fields = tuple(reader.fieldnames or ())
        if len(set(fields)) != len(fields):
            raise AuditError("candidate manifest contains duplicate columns")
        rows = list(reader)
    except (UnicodeError, csv.Error) as error:
        raise AuditError(f"candidate manifest is invalid CSV: {error}") from error
    required = {"filepath", "sample_id", "source_sha256", "image_sha256"}
    if not required.issubset(fields) or not rows:
        raise AuditError("candidate manifest is empty or missing required fields")
    root = path.parent
    crop_path = _candidate_source_path if allow_absolute_crop_paths else _manifest_relative_path
    assets: list[AuditAsset] = []
    source_seen: set[tuple[str, str]] = set()
    explicit_views = "view_kind" in fields
    for line, row in enumerate(rows, start=2):
        if None in row:
            raise AuditError(f"candidate manifest line {line} has extra columns")
        row_role = (row.get("role") or role).strip()
        if row_role != role:
            raise AuditError(f"candidate manifest line {line} role mismatch")
        sample_id = (row.get("sample_id") or "").strip()
        source_sha = _require_sha256(row.get("source_sha256"), f"line {line} source_sha256")
        image_sha = _require_sha256(row.get("image_sha256"), f"line {line} image_sha256")
        if explicit_views:
            view_kind = (row.get("view_kind") or "").strip()
            assets.append(
                AuditAsset(
                    path=crop_path(root, row["filepath"], f"line {line} filepath"),
                    role=role,
                    cohort="candidate",
                    view_kind=view_kind,
                    sample_id=sample_id,
                    source_sha256=source_sha,
                    image_sha256=image_sha,
                )
            )
            continue

        assets.append(
            AuditAsset(
                path=crop_path(root, row["filepath"], f"line {line} filepath"),
                role=role,
                cohort="candidate",
                view_kind="crop",
                sample_id=sample_id,
                source_sha256=source_sha,
                image_sha256=image_sha,
            )
        )
        source_value = (row.get("source_filepath") or "").strip()
        if not source_value and (row.get("source_path_b64") or "").strip():
            source_value = _decode_source_path(row["source_path_b64"].strip())
        if not source_value:
            raise AuditError(
                f"candidate manifest line {line} lacks source_filepath/source_path_b64"
            )
        source_path = _candidate_source_path(root, source_value, f"line {line} source filepath")
        source_key = (source_sha, os.path.normcase(os.fspath(source_path)))
        if source_key not in source_seen:
            source_seen.add(source_key)
            assets.append(
                AuditAsset(
                    path=source_path,
                    role=role,
                    cohort="candidate",
                    view_kind="source",
                    sample_id=f"source:{role}:{source_sha}",
                    source_sha256=source_sha,
                    image_sha256=source_sha,
                )
            )
    return assets, manifest_sha


def _parse_candidate_manifest(value: str) -> tuple[str, Path]:
    role, separator, path = value.partition("=")
    if not separator or role not in CANDIDATE_ROLES or not path:
        raise argparse.ArgumentTypeError("candidate manifest must be role=path")
    return role, Path(path)


def _atomic_no_overwrite(
    path: Path,
    payload: bytes,
    *,
    pre_publish: Callable[[], None] | None = None,
) -> None:
    if path.exists():
        raise AuditError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_chain(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    published = False
    temporary_identity: _ReadIdentity | None = None
    try:
        with os.fdopen(descriptor, "w+b") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
            temporary_identity = _identity(os.fstat(file.fileno()))
            if not stat.S_ISREG(temporary_identity.mode) or temporary_identity.nlink != 1:
                raise AuditError("temporary output identity is invalid")
            if temporary_identity.size != len(payload):
                raise AuditError("temporary output size is invalid")
            _reject_symlink_chain(path.parent)
            try:
                temporary_path_identity = _identity(os.lstat(temporary))
            except OSError as error:
                raise AuditError("temporary output path changed before publication") from error
            if (
                _identity(os.fstat(file.fileno())) != temporary_identity
                or not _path_identity_matches(
                    temporary_path_identity,
                    temporary_identity,
                    expected_nlink=1,
                )
            ):
                raise AuditError("temporary output identity changed before publication")
            file.seek(0)
            if _sha256_bytes(file.read()) != _sha256_bytes(payload):
                raise AuditError("temporary output bytes changed before publication")
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise AuditError(f"refusing to overwrite output: {path}") from error
            except OSError as error:
                raise AuditError(f"could not publish output atomically: {error}") from error
            published = True
            published_identity = _identity(os.lstat(path))
            if not _path_identity_matches(
                published_identity,
                temporary_identity,
                expected_nlink=2,
            ):
                raise AuditError("published output does not name the verified temporary file")
            # Check all audited inputs again after the output name exists.  If
            # anything changed in the former pre-publication race window, the
            # exception path removes exactly this newly linked output.
            if pre_publish is not None:
                pre_publish()
    except Exception:
        if published and temporary_identity is not None:
            try:
                current = _identity(os.lstat(path))
                if (
                    current.device == temporary_identity.device
                    and current.inode == temporary_identity.inode
                ):
                    path.unlink()
            except OSError:
                pass
        raise
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            if not published:
                raise
    if published:
        try:
            _read_regular_file(
                path,
                expected_sha256=_sha256_bytes(payload),
                max_bytes=max(1, len(payload)),
                root=path.parent,
            )
        except Exception:
            try:
                current = _identity(os.lstat(path))
                if (
                    temporary_identity is not None
                    and current.device == temporary_identity.device
                    and current.inode == temporary_identity.inode
                ):
                    path.unlink()
            except OSError:
                pass
            raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-manifest",
        action="append",
        type=_parse_candidate_manifest,
        required=True,
        metavar="role=path",
    )
    parser.add_argument("--protected-sources", type=Path, required=True)
    parser.add_argument("--protected-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.candidate_manifest) != 2:
        parser.error("exactly two --candidate-manifest values are required")
    manifest_map = dict(args.candidate_manifest)
    if set(manifest_map) != CANDIDATE_ROLES:
        parser.error("candidate manifests must cover train and model_validation exactly once")

    assets: list[AuditAsset] = []
    bindings: dict[str, str] = {}
    for role in sorted(CANDIDATE_ROLES):
        current_assets, manifest_sha = _load_candidate_manifest(role, manifest_map[role])
        assets.extend(current_assets)
        bindings[role] = manifest_sha
    report, payload, records = build_near_duplicate_report(
        assets,
        bindings,
        args.protected_sources,
        args.protected_inventory,
    )

    def final_input_reverification() -> None:
        for role, manifest_path in manifest_map.items():
            _read_regular_file(
                manifest_path,
                expected_sha256=bindings[role],
            )
        cohort_by_sha, protected_union, _source_payload, source_file_sha, union_sha = (
            _load_protected_sources(args.protected_sources)
        )
        source_binding = report["bindings"]["protected_sources"]
        if (
            source_file_sha != source_binding["file_sha256"]
            or union_sha != source_binding["canonical_union_sha256"]
        ):
            raise AuditError("protected sources changed before report publication")
        protected_records, _inventory_payload, inventory_file_sha, inventory_payload_sha = (
            _load_protected_inventory(
                args.protected_inventory,
                cohort_by_sha=cohort_by_sha,
                expected_union=protected_union,
            )
        )
        inventory_binding = report["bindings"]["protected_inventory"]
        if (
            inventory_file_sha != inventory_binding["file_sha256"]
            or inventory_payload_sha != inventory_binding["payload_sha256"]
            or _payload_set_sha(protected_records)
            != report["bindings"]["protected_payload_set_sha256"]
        ):
            raise AuditError("protected inventory changed before report publication")
        if _auditor_binding() != report["bindings"]["auditor"]:
            raise AuditError("auditor bytes changed before report publication")
        # Make image verification the last substantive check so a mutation
        # during the preceding metadata/inventory checks is also detected.
        reverify_assets(records)
        for role, manifest_path in manifest_map.items():
            _read_regular_file(
                manifest_path,
                expected_sha256=bindings[role],
            )

    _atomic_no_overwrite(
        args.output,
        payload,
        pre_publish=final_input_reverification,
    )
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as error:
        print(f"near-duplicate audit failed closed: {error}", file=sys.stderr)
        raise SystemExit(2)
