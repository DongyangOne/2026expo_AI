"""Build immutable, candidate-only v4 verifier training inputs.

The qx3 pilot is bound only as generator reproducibility evidence. Its
diagnostic crops cannot become training rows. Nothing emitted here grants
blind-test, promotion, Pi, Spring, or production authority.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import errno
import hashlib
import io
import json
import math
import os
import posixpath
import re
import shutil
import stat
import sys
import tempfile
import types
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

try:
    from scripts import operational_quality_assembly_contract as quality_assembly_contract
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import operational_quality_assembly_contract as quality_assembly_contract


AUTHORITY_SCHEMA = "v4_candidate_training_authority.v3"
AUTHORITY_ROLE = "v4_candidate_training_input_authority_not_blind_or_deployment"
AUTHORITY_STATUS = "candidate_training_inputs_ready"
PREAUDIT_PROPOSAL_SCHEMA = "v4_candidate_preaudit_proposal.v1"
PREAUDIT_PROPOSAL_ROLE = (
    "v4_candidate_near_duplicate_preaudit_proposal_"
    "not_training_or_blind_or_deployment_authority"
)
PREAUDIT_PROPOSAL_STATUS = "candidate_preaudit_inputs_ready"
REPO_ROOT = Path(__file__).resolve().parents[1]
TRUSTED_POLICY_RELATIVE_PATH = Path(
    "configs/v4_candidate_training_trusted_policy.json"
)
APPROVED_TRUSTED_POLICY_SHA256 = "UNCONFIGURED"
UNCONFIGURED_TRUST_ROOT = "UNCONFIGURED"
TRUST_ROOT_METHOD = "git_bundled_code_sha256_pin"
FULL_DATA_REPORT_ROLE = "v4_development_candidates_not_blind_or_deployment_authority"
QX3_READY_ROLE = "v4_reproducibility_diagnostic_not_candidate_or_deployment_authority"
QX3_REPORT_ROLE = "v4_batch1_validator_reproducibility_diagnostic_only"
QUALITY_CONTRACT = quality_assembly_contract.QUALITY_CONTRACT
QUALITY_ROLE = quality_assembly_contract.QUALITY_ROLE
QUALITY_REASONS = quality_assembly_contract.QUALITY_REASONS
QUALITY_ASSEMBLY_SCHEMA = quality_assembly_contract.QUALITY_ASSEMBLY_SCHEMA
QUALITY_ASSEMBLY_ROLE = quality_assembly_contract.QUALITY_ASSEMBLY_ROLE
QUALITY_ASSEMBLY_STATUS = quality_assembly_contract.QUALITY_ASSEMBLY_STATUS
QUALITY_ASSEMBLY_MODE = quality_assembly_contract.QUALITY_ASSEMBLY_MODE
QUALITY_ASSEMBLY_TEACHER_SCHEMA = quality_assembly_contract.QUALITY_ASSEMBLY_TEACHER_SCHEMA
QUALITY_ASSEMBLY_FILES = quality_assembly_contract.QUALITY_ASSEMBLY_FILES
QUALITY_ASSEMBLY_INPUT_SHA_FIELDS = quality_assembly_contract.QUALITY_ASSEMBLY_INPUT_SHA_FIELDS
QUALITY_ASSEMBLY_CODE_SHA_FIELDS = quality_assembly_contract.QUALITY_ASSEMBLY_CODE_SHA_FIELDS
QUALITY_ASSEMBLY_SCOPE_FIELDS = quality_assembly_contract.QUALITY_ASSEMBLY_SCOPE_FIELDS
QUALITY_ASSEMBLY_RECEIPT_FIELDS = quality_assembly_contract.QUALITY_ASSEMBLY_RECEIPT_FIELDS
KST = quality_assembly_contract.KST
OPERATIONAL_CUTOFF = quality_assembly_contract.OPERATIONAL_CUTOFF
SHA256_RE = quality_assembly_contract.SHA256_RE
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
MATERIAL_CLASSES = (
    "can", "pet", "paper", "plastic", "styrofoam", "vinyl", "glass",
    "battery", "fluorescent",
)
CONDITION_HEADS = ("dent", "label", "foreign_material")
ROLE_SPLITS = {"train": "training", "model_validation": "validation"}
OUTPUT_FIELDS = (
    "filepath", "split", "source_id", "material", "category", "dent",
    "label", "foreign_material", "source_object_count", "crop_object_count",
    "sample_id", "role", "fold", "source_sha256", "image_sha256",
    "object_group", "capture_session", "origin", "source_filepath",
    "captured_at", "auditor_sha256", "teacher_output_sha256",
    "localizer_output_sha256",
)
REQUIRED_INPUT_FIELDS = set(OUTPUT_FIELDS)
PROTECTED_FIELDS = (
    "qx3_diagnostic_source_sha256", "qx3_validation_source_sha256",
    "hardware41_source_sha256",
    "known_audit_source_sha256", "calibration_source_sha256",
    "blind_test_source_sha256",
)
FALSE_AUTHORITY_FIELDS = quality_assembly_contract.FALSE_AUTHORITY_FIELDS
CONFIG_FIELDS = {
    "schema", "backbone", "pretrained", "input_size", "epochs", "patience",
    "batch", "workers", "lr", "backbone_lr", "head_lr", "label_smoothing",
    "class_weight_mode", "class_weight_beta", "objectness_weight",
    "material_weight", "condition_weight", "condition_heads",
    "origin_weights", "seed", "optimizer", "optimizer_betas",
    "weight_decay", "scheduler", "sampling_mode",
    "scheduler_t_max", "scheduler_eta_min",
    "sampling_samples_per_epoch", "sampling_expected_fraction_by_origin",
    "image_consumption_contract_version", "image_max_bytes", "image_max_pixels",
}
REQUIRED_CONTAINER_ENV = {
    "RUN_ROOT", "RUN_DIR", "GLOBAL_ROOT", "CODE_ROOT", "AUTHORITY_JSON",
    "AUTHORITY_MARKER", "CODE_INVENTORY", "TRAINING_CONFIG",
    "HOST_LAUNCH_CONTRACT", "PRETRAINED_BACKBONE", "CONTAINER_IMAGE_ID",
}
FORBIDDEN_CONTAINER_ENV = {
    "PYTHON_BIN", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP",
    "PYTHONINSPECT", "PYTHONWARNINGS", "PYTHONBREAKPOINT",
    "PYTHONUSERBASE", "LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT",
    "BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS", "PS4", "BASH_XTRACEFD",
}
BASH_EXPORTED_FUNCTION_PREFIX = "BASH_FUNC_"
CLEAN_CONTAINER_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
POLICY_BINDING_FIELDS = {
    "qx3_diagnostic_ready_sha256", "qx3_diagnostic_report_sha256",
    "license_allowlist_sha256", "quality_exclusions_sha256",
    "quality_exclusion_assembly_receipt_sha256",
    "protected_sources_sha256", "code_inventory_sha256",
    "training_config_sha256", "host_launch_contract_sha256",
    "raw_container_inspect_sha256", "pretrained_backbone_sha256",
    "container_image_id", "candidate_train_manifest_sha256",
    "candidate_model_validation_manifest_sha256",
    "candidate_dataset_snapshot_sha256",
    "candidate_dataset_consumption_contract_sha256",
    "candidate_near_duplicate_audit_sha256",
    "protected_reference_inventory_sha256",
}
NEAR_DUPLICATE_REPORT_SCHEMA = "v4_near_duplicate_leakage_audit.v1"
NEAR_DUPLICATE_AUDITOR_PATH = "scripts/audit_v4_near_duplicate_leakage.py"
NEAR_DUPLICATE_ALGORITHM_ID = "oneexpo_phash_rot4_v1"
NEAR_DUPLICATE_PHASH_DISTANCE = 4
NEAR_DUPLICATE_PIXEL_CAP = 16_000_000
NEAR_DUPLICATE_GRAPH_EDGE_CAP = 1_000_000
PROTECTED_REFERENCE_INVENTORY_SCHEMA = (
    "v4_near_duplicate_protected_inventory.v1"
)
TRUST_ROOT_CODE_PATHS = {
    "configs/v4_candidate_training_trusted_policy.json",
    "scripts/build_v4_candidate_training_authority.py",
    "scripts/nas/run_v4_candidate_training.sh",
}
EXPECTED_NVIDIA_DEVICES = (
    "/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset",
    "/dev/nvidia-caps/nvidia-cap1", "/dev/nvidia-caps/nvidia-cap2",
)
ALLOWED_QNAP_LIBRARY_MOUNTS = {
    (
        "/share/CACHEDEV1_DATA/.qpkg/NVIDIA_GPU_DRV/usr/lib",
        "/qnap/nvidia/lib",
    ),
    (
        "/share/CACHEDEV1_DATA/.qpkg/NVIDIA_GPU_DRV/cuda-12.9/lib64",
        "/qnap/cuda/lib64",
    ),
}
QNAP_LIBRARY_INVENTORY_SCHEMA = "v4_qnap_library_inventory.v1"
QNAP_LIBRARY_SNAPSHOT_MAX_BYTES = 3221225472
DATASET_SNAPSHOT_SCHEMA = "v4_candidate_dataset_snapshot.v2"
DATASET_SNAPSHOT_ROLE = (
    "candidate_training_crop_bytes_not_blind_or_deployment_authority"
)
DATASET_SNAPSHOT_MAX_BYTES = 68719476736
IMAGE_CONSUMPTION_CONTRACT_VERSION = "multitask_verifier.image_consumption.v1"
IMAGE_CONSUMPTION_MAX_BYTES = 67108864
IMAGE_CONSUMPTION_MAX_PIXELS = 16777216
DATASET_CONSUMPTION_CONTRACT_SCHEMA = "v4_candidate_dataset_consumption.v1"


def _dataset_consumption_contract(
    *,
    trainer_sha256: str,
    snapshot_report_sha256: str,
    snapshot_tree_sha256: str,
    manifest_payload_set_sha256: str,
) -> dict[str, object]:
    return {
        "schema": DATASET_CONSUMPTION_CONTRACT_SCHEMA,
        "version": IMAGE_CONSUMPTION_CONTRACT_VERSION,
        "evidence_scope": "per_access_fail_closed_no_complete_access_receipt",
        "authority_platform": "linux_qnap",
        "read_semantics": "single_descriptor_fstat_sha256_then_bytesio_decode",
        "trainer_path": "scripts/train_multitask_verifier.py",
        "trainer_sha256": _require_sha256(trainer_sha256, "consumption trainer SHA"),
        "dataset_snapshot_report_sha256": _require_sha256(
            snapshot_report_sha256, "consumption snapshot report SHA"
        ),
        "dataset_snapshot_tree_sha256": _require_sha256(
            snapshot_tree_sha256, "consumption snapshot tree SHA"
        ),
        "manifest_payload_set_sha256": _require_sha256(
            manifest_payload_set_sha256, "consumption manifest payload-set SHA"
        ),
        "max_image_bytes": IMAGE_CONSUMPTION_MAX_BYTES,
        "max_image_pixels": IMAGE_CONSUMPTION_MAX_PIXELS,
        "complete_access_receipt": False,
    }


def _reject_duplicate_keys(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _canonical_compact_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _require_bool(value: object, expected: bool, field: str) -> None:
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{field} must be {expected!r}")


def _exact_json_equal(left: object, right: object) -> bool:
    """Compare JSON values without Python's bool/int equality aliasing."""

    if type(left) is not type(right):
        return False
    if type(left) is dict:
        assert type(right) is dict
        if set(left) != set(right):
            return False
        return all(_exact_json_equal(left[key], right[key]) for key in left)
    if type(left) is list:
        assert type(right) is list
        return len(left) == len(right) and all(
            _exact_json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _reject_symlink_components(path: Path, description: str) -> Path:
    absolute = Path(os.path.abspath(path))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{description} path contains a symlink: {cursor}")
    return absolute


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without ever replacing a destination."""

    if os.name == "nt":
        # Windows rename fails if the destination already exists.
        os.rename(source, destination)
        return
    if not sys.platform.startswith("linux"):
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory publication is unsupported",
            os.fspath(destination),
        )
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(
            errno.ENOTSUP,
            "renameat2 is required for atomic no-replace publication",
            os.fspath(destination),
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,  # AT_FDCWD
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            os.fspath(destination),
        )


def _regular_file(path: Path, description: str) -> Path:
    absolute = _reject_symlink_components(path, description)
    if absolute.is_symlink() or not absolute.is_file():
        raise ValueError(f"{description} must be a regular non-symlink file: {path}")
    return absolute


def _stable_bytes(path: Path, description: str) -> bytes:
    resolved = _regular_file(path, description)
    before = resolved.stat()
    with resolved.open("rb") as handle:
        opened_before = os.fstat(handle.fileno())
        content = handle.read()
        opened_after = os.fstat(handle.fileno())
    after = resolved.stat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if not (
        identity(before) == identity(opened_before) == identity(opened_after) == identity(after)
    ):
        raise RuntimeError(f"{description} changed while being read: {resolved}")
    return content


def _load_auditor_runtime_fingerprint(
    path: Path, expected_sha256: str,
) -> str:
    """Execute the exact verified auditor bytes and fingerprint live code.

    Loading from the already verified byte string, instead of importing the
    path, closes the load/restore race where Python could execute one auditor
    while the report and code inventory name another.
    """
    content = _stable_bytes(path, "near-duplicate auditor runtime source")
    if _sha256_bytes(content) != expected_sha256:
        raise ValueError("near-duplicate auditor runtime source SHA mismatch")
    module_name = f"_v4_near_duplicate_auditor_{expected_sha256}"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        code = compile(
            content,
            str(path),
            "exec",
            dont_inherit=True,
        )
        exec(code, module.__dict__)
        fingerprint_function = module.__dict__.get(
            "runtime_code_fingerprint_sha256"
        )
        if not callable(fingerprint_function):
            raise ValueError(
                "near-duplicate auditor runtime fingerprint function is absent"
            )
        fingerprint = fingerprint_function()
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
    result = _require_sha256(
        fingerprint, "near-duplicate auditor runtime code SHA"
    )
    if _sha256_bytes(
        _stable_bytes(path, "near-duplicate auditor runtime source final rehash")
    ) != expected_sha256:
        raise ValueError("near-duplicate auditor changed during runtime validation")
    return result


def _dataset_stat_contract(value: os.stat_result) -> dict[str, int]:
    return {
        "dev": int(value.st_dev),
        "ino": int(value.st_ino),
        "size": int(value.st_size),
        "mode": stat.S_IMODE(value.st_mode),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
        "nlink": int(value.st_nlink),
    }


def _dataset_stat_core(value: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(value[key] for key in (
        "dev", "ino", "size", "mode", "mtime_ns", "nlink"
    ))


def _read_dataset_input(
    path: Path, description: str, expected_sha256: str
) -> dict[str, object]:
    """Hash one payload through one descriptor and bind its path identity."""

    absolute = _reject_symlink_components(path, description)
    before = absolute.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or absolute.is_symlink():
        raise ValueError(f"{description} must be a regular non-symlink file")
    if before.st_nlink != 1:
        raise ValueError(f"{description} hardlink aliases are forbidden")
    digest = hashlib.sha256()
    with absolute.open("rb") as handle:
        opened_before = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        opened_after = os.fstat(handle.fileno())
    after = absolute.stat(follow_symlinks=False)
    path_before = _dataset_stat_contract(before)
    handle_before = _dataset_stat_contract(opened_before)
    handle_after = _dataset_stat_contract(opened_after)
    path_after = _dataset_stat_contract(after)
    if (
        path_before != path_after
        or handle_before != handle_after
        or len({
            _dataset_stat_core(path_before),
            _dataset_stat_core(handle_before),
            _dataset_stat_core(handle_after),
            _dataset_stat_core(path_after),
        }) != 1
    ):
        raise RuntimeError(f"{description} changed while being hashed: {absolute}")
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{description} content hash mismatch")
    return {
        "path": absolute.as_posix(),
        "sha256": actual_sha256,
        **path_before,
    }


def _verify_dataset_input(record: Mapping[str, object], description: str) -> None:
    expected_fields = {
        "path", "sha256", "dev", "ino", "size", "mode", "mtime_ns",
        "ctime_ns", "nlink",
    }
    if set(record) != expected_fields:
        raise ValueError(f"{description} input record schema mismatch")
    path = Path(str(record["path"]))
    current = _read_dataset_input(
        path, description, _require_sha256(record.get("sha256"), f"{description} SHA")
    )
    if current != dict(record):
        raise RuntimeError(f"{description} identity changed")


def _snapshot_relative_path(image_sha256: str) -> str:
    digest = _require_sha256(image_sha256, "dataset snapshot crop SHA")
    return f"dataset_snapshot/objects/{digest[:2]}/{digest}"


def _dataset_snapshot_report(
    entries: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], bytes]:
    expected_fields = {"sample_id", "role", "path", "size", "sha256"}
    normalized: list[dict[str, object]] = []
    seen_samples: set[str] = set()
    seen_paths: set[str] = set()
    seen_shas: set[str] = set()
    total = 0
    for index, raw in enumerate(entries):
        if set(raw) != expected_fields:
            raise ValueError(f"dataset snapshot plan row {index} schema mismatch")
        sample_id = raw.get("sample_id")
        role = raw.get("role")
        relative = raw.get("path")
        size = raw.get("size")
        digest = _require_sha256(
            raw.get("sha256"), f"dataset snapshot plan row {index} SHA"
        )
        if type(sample_id) is not str or not sample_id or role not in ROLE_SPLITS:
            raise ValueError(f"dataset snapshot plan row {index} identity mismatch")
        if type(relative) is not str or relative != _snapshot_relative_path(digest):
            raise ValueError(f"dataset snapshot plan row {index} path mismatch")
        if type(size) is not int or size <= 0:
            raise ValueError(f"dataset snapshot plan row {index} size mismatch")
        if size > IMAGE_CONSUMPTION_MAX_BYTES:
            raise ValueError(
                f"dataset snapshot plan row {index} exceeds image_max_bytes"
            )
        if sample_id in seen_samples:
            raise ValueError("duplicate dataset snapshot sample_id")
        if relative in seen_paths:
            raise ValueError("duplicate dataset snapshot canonical path")
        if digest in seen_shas:
            raise ValueError("duplicate dataset snapshot crop SHA")
        seen_samples.add(sample_id)
        seen_paths.add(relative)
        seen_shas.add(digest)
        total += size
        normalized.append({
            "sample_id": sample_id,
            "role": role,
            "path": relative,
            "size": size,
            "sha256": digest,
        })
    normalized.sort(key=lambda row: (str(row["role"]), str(row["sample_id"])))
    if not normalized:
        raise ValueError("dataset snapshot plan is empty")
    if total > DATASET_SNAPSHOT_MAX_BYTES:
        raise ValueError("dataset snapshot exceeds snapshot_max_bytes")
    tree_rows = [
        {"path": row["path"], "size": row["size"], "sha256": row["sha256"]}
        for row in sorted(normalized, key=lambda row: str(row["path"]))
    ]
    payload_set_sha256 = _sha256_bytes(_canonical_compact_json(normalized))
    report: dict[str, object] = {
        "schema": DATASET_SNAPSHOT_SCHEMA,
        "artifact_role": DATASET_SNAPSHOT_ROLE,
        "status": "candidate_dataset_snapshot_ready",
        "candidate_only": True,
        "production_deployment_authorized": False,
        "payload_kind": "training_crop_only",
        "source_lineage_bytes_snapshotted": False,
        "snapshot_root_relative": "dataset_snapshot",
        "snapshot_max_bytes": DATASET_SNAPSHOT_MAX_BYTES,
        "object_max_bytes": IMAGE_CONSUMPTION_MAX_BYTES,
        "object_count": len(normalized),
        "total_regular_bytes": total,
        "tree_sha256": _sha256_bytes(_canonical_compact_json(tree_rows)),
        "payload_set_sha256": payload_set_sha256,
        "objects": normalized,
    }
    return report, _canonical_json(report)


def _copy_dataset_snapshot_object(
    record: Mapping[str, object], destination: Path, description: str
) -> dict[str, int]:
    _verify_dataset_input(record, f"{description} pre-copy")
    source = Path(str(record["path"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"dataset snapshot object already exists: {destination}")
    digest = hashlib.sha256()
    with source.open("rb") as source_handle, destination.open("xb") as output:
        source_before = os.fstat(source_handle.fileno())
        source_before_contract = _dataset_stat_contract(source_before)
        if _dataset_stat_core(source_before_contract) != _dataset_stat_core({
            key: record[key]
            for key in ("dev", "ino", "size", "mode", "mtime_ns", "ctime_ns", "nlink")
        }):
            raise RuntimeError(f"{description} changed before copy")
        for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
            digest.update(chunk)
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
        source_after = os.fstat(source_handle.fileno())
    path_after = source.stat(follow_symlinks=False)
    expected_identity = {
        key: record[key]
        for key in ("dev", "ino", "size", "mode", "mtime_ns", "ctime_ns", "nlink")
    }
    source_after_contract = _dataset_stat_contract(source_after)
    path_after_contract = _dataset_stat_contract(path_after)
    if (
        source_after_contract != source_before_contract
        or
        _dataset_stat_core(source_after_contract) != _dataset_stat_core(expected_identity)
        or path_after_contract != expected_identity
    ):
        raise RuntimeError(f"{description} changed during copy")
    if digest.hexdigest() != record["sha256"]:
        raise RuntimeError(f"{description} copied bytes differ from the approved SHA")
    os.chmod(destination, 0o444, follow_symlinks=False)
    copied = destination.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(copied.st_mode)
        or copied.st_nlink != 1
        or stat.S_IMODE(copied.st_mode) != 0o444
    ):
        raise RuntimeError(f"{description} snapshot identity/mode mismatch")
    return _dataset_stat_contract(copied)


def _dataset_snapshot_tree_contract(
    root: Path, report: Mapping[str, object], *, logical_root: Path
) -> dict[str, object]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("dataset snapshot root must be a non-symlink directory")
    if (
        report.get("schema") != DATASET_SNAPSHOT_SCHEMA
        or report.get("object_max_bytes") != IMAGE_CONSUMPTION_MAX_BYTES
    ):
        raise ValueError("dataset snapshot report consumption contract mismatch")
    objects = report.get("objects")
    if type(objects) is not list:
        raise ValueError("dataset snapshot report objects must be a list")
    expected: dict[Path, Mapping[str, object]] = {}
    expected_directories = {root}
    for index, raw in enumerate(objects):
        if type(raw) is not dict:
            raise ValueError(f"dataset snapshot report object {index} is not an object")
        relative = Path(str(raw.get("path", "")))
        if relative.parts[:1] != ("dataset_snapshot",) or relative.is_absolute():
            raise ValueError("dataset snapshot object path escapes the snapshot root")
        object_relative = Path(*relative.parts[1:])
        path = root / object_relative
        if path in expected:
            raise ValueError("duplicate dataset snapshot object path")
        expected[path] = raw
        cursor = path.parent
        while cursor != root:
            expected_directories.add(cursor)
            cursor = cursor.parent
        expected_directories.add(root)
    actual_entries = list(root.rglob("*"))
    if any(path.is_symlink() for path in actual_entries):
        raise ValueError("dataset snapshot tree contains a symlink")
    for path in actual_entries:
        entry_mode = path.stat(follow_symlinks=False).st_mode
        if not (stat.S_ISREG(entry_mode) or stat.S_ISDIR(entry_mode)):
            raise ValueError(
                "dataset snapshot tree entries must be regular files or directories"
            )
    actual_files = {path for path in actual_entries if path.is_file()}
    actual_directories = {root, *(path for path in actual_entries if path.is_dir())}
    if actual_files != set(expected):
        raise ValueError("dataset snapshot regular-file set mismatch")
    if actual_directories != expected_directories:
        raise ValueError("dataset snapshot directory set mismatch")
    root_stat = root.stat(follow_symlinks=False)
    if os.name != "nt" and stat.S_IMODE(root_stat.st_mode) != 0o555:
        raise ValueError("dataset snapshot root mode must be 0555")
    for directory in actual_directories:
        current = directory.stat(follow_symlinks=False)
        if not stat.S_ISDIR(current.st_mode) or (
            os.name != "nt" and stat.S_IMODE(current.st_mode) != 0o555
        ):
            raise ValueError("dataset snapshot directories must be mode 0555")
    rows: list[dict[str, object]] = []
    total = 0
    for path in sorted(actual_files, key=lambda value: value.as_posix()):
        raw = expected[path]
        current = _read_dataset_input(
            path,
            "published dataset snapshot object",
            _require_sha256(raw.get("sha256"), "dataset snapshot object SHA"),
        )
        if current["size"] != raw.get("size") or current["mode"] != 0o444:
            raise ValueError("dataset snapshot object size/mode mismatch")
        if int(current["size"]) > IMAGE_CONSUMPTION_MAX_BYTES:
            raise ValueError("dataset snapshot object exceeds image_max_bytes")
        relative = path.relative_to(root).as_posix()
        total += int(current["size"])
        rows.append({
            "path": f"dataset_snapshot/{relative}",
            "dev": current["dev"],
            "ino": current["ino"],
            "mode": current["mode"],
            "nlink": current["nlink"],
            "size": current["size"],
            "sha256": current["sha256"],
        })
    if total != report.get("total_regular_bytes"):
        raise ValueError("dataset snapshot total bytes mismatch")
    tree_rows = [
        {"path": row["path"], "size": row["size"], "sha256": row["sha256"]}
        for row in rows
    ]
    if _sha256_bytes(_canonical_compact_json(tree_rows)) != report.get("tree_sha256"):
        raise ValueError("dataset snapshot tree SHA mismatch")
    payload_rows = [
        {
            "sample_id": raw["sample_id"],
            "role": raw["role"],
            "path": raw["path"],
            "size": raw["size"],
            "sha256": raw["sha256"],
        }
        for raw in sorted(
            expected.values(), key=lambda value: (str(value["role"]), str(value["sample_id"]))
        )
    ]
    if _sha256_bytes(_canonical_compact_json(payload_rows)) != report.get(
        "payload_set_sha256"
    ):
        raise ValueError("dataset snapshot payload-set SHA mismatch")
    return {
        "schema": "v4_candidate_dataset_snapshot_publish_receipt.v1",
        "snapshot_root": logical_root.as_posix(),
        "root_dev": int(root_stat.st_dev),
        "root_ino": int(root_stat.st_ino),
        "root_mode": stat.S_IMODE(root_stat.st_mode),
        "files": rows,
    }


def _load_json(path: Path, description: str) -> tuple[dict[str, object], bytes]:
    content = _stable_bytes(path, description)
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {description}: {error}") from error
    if type(value) is not dict:
        raise ValueError(f"{description} root must be an object")
    return value, content


def _load_manifest(path: Path) -> tuple[list[dict[str, str]], bytes]:
    content = _stable_bytes(path, "source manifest")
    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError(f"invalid UTF-8 source manifest: {path}") from error
    fields = list(reader.fieldnames or [])
    if len(fields) != len(set(fields)) or not REQUIRED_INPUT_FIELDS.issubset(fields):
        raise ValueError(f"source manifest columns are missing or duplicated: {path}")
    if not rows:
        raise ValueError(f"source manifest is empty: {path}")
    for number, row in enumerate(rows, start=2):
        if None in row or set(row) != set(fields):
            raise ValueError(f"malformed source manifest row: {path}:{number}")
    return rows, content


def _render_manifest(rows: Sequence[Mapping[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


_canonical_quality_entries = quality_assembly_contract._canonical_quality_entries
_validate_quality_manifest = quality_assembly_contract._validate_quality_manifest
_QualityAssemblyBundle = quality_assembly_contract._QualityAssemblyBundle
_quality_assembly_code_paths = quality_assembly_contract._quality_assembly_code_paths
_quality_assembly_code_hashes = quality_assembly_contract._quality_assembly_code_hashes
_quality_assembly_marker_bytes = quality_assembly_contract._quality_assembly_marker_bytes
_validate_operational_quality_assembly = quality_assembly_contract._validate_operational_quality_assembly
_rehash_operational_quality_assembly = quality_assembly_contract._rehash_operational_quality_assembly


def _validate_full_data_report(
    value: Mapping[str, object], manifest_sha256: str,
    manifest_rows: Sequence[Mapping[str, str]],
) -> None:
    row_count = len(manifest_rows)
    expected_fields = {
        "schema_version", "artifact_role", "ready_for_lineage_upgrade",
        "blind_test_eligible", "production_deployment_authorized", "rows",
        "counts", "contract", "bindings",
    }
    if set(value) != expected_fields:
        raise ValueError("full-data validator top-level schema mismatch")
    if type(value.get("schema_version")) is not int or value.get("schema_version") != 1:
        raise ValueError("full-data validator schema_version mismatch")
    if value.get("artifact_role") != FULL_DATA_REPORT_ROLE:
        raise ValueError("full-data validator is not authoritative")
    _require_bool(value.get("ready_for_lineage_upgrade"), True, "ready_for_lineage_upgrade")
    _require_bool(value.get("blind_test_eligible"), False, "blind_test_eligible")
    _require_bool(
        value.get("production_deployment_authorized"), False,
        "production_deployment_authorized",
    )
    if type(value.get("rows")) is not int or value.get("rows") != row_count:
        raise ValueError("full-data validator row count mismatch")
    expected_counts = dict(sorted(Counter(
        f"{row.get('split', '')}/{row.get('category', '')}"
        for row in manifest_rows
    ).items()))
    counts = value.get("counts")
    if type(counts) is not dict or counts != expected_counts:
        raise ValueError("full-data validator counts mismatch")
    bindings = value.get("bindings")
    binding_fields = {
        "input_manifest_sha256", "dataset_info_sha256", "detector_model_sha256",
        "inference_spec_sha256", "validated_manifest_sha256",
    }
    if type(bindings) is not dict or set(bindings) != binding_fields:
        raise ValueError("full-data validator bindings schema mismatch")
    for field in binding_fields:
        _require_sha256(bindings.get(field), f"full-data validator bindings.{field}")
    if bindings.get("validated_manifest_sha256") != manifest_sha256:
        raise ValueError("full-data validator validated manifest binding mismatch")
    contract = value.get("contract")
    expected_contract = {
        "manifest_schema_version": "proposal_verifier.v4.bgfix.v1",
        "background_policy": "strict-zero-intersection",
        "background_gt_margin": 0.10,
        "explicit_label_file_required": True,
        "source_object_count_semantics": "complete_source_frame",
        "crop_object_count_semantics": "final_padded_verifier_crop",
        "visual_judge_still_required": True,
    }
    if type(contract) is not dict or set(contract) != {
        *expected_contract, "proposal_provenance"
    }:
        raise ValueError("full-data validator contract schema mismatch")
    for field, expected in expected_contract.items():
        actual = contract.get(field)
        if type(expected) is bool:
            _require_bool(actual, expected, f"full-data validator contract.{field}")
        elif actual != expected or (type(expected) is int and type(actual) is not int):
            raise ValueError(f"full-data validator contract.{field} mismatch")
    provenance = contract.get("proposal_provenance")
    provenance_fields = {
        "sources", "provider_kind", "runtime_detector_executed",
        "runtime_top1_replayed", "provided_top1_predictions_matched",
        "proposal_class_confidence_bbox_matched", "confidence_abs_tolerance",
        "bbox_abs_tolerance", "original_generation_event_cryptographically_attested",
        "authority", "cuda_client_initialized_before_source_crop_scan",
        "detector_artifact_bytes_bound", "detector_replay_used_unique_snapshot",
        "source_and_label_replay_used_unique_snapshots",
        "replay_snapshots_verified_after_inference",
        "original_detector_bytes_unchanged_through_validation",
        "inference_spec_bytes_bound", "dataset_info_bytes_bound",
        "source_bbox_crop_bytes_recomputed", "production_or_blind_authority",
    }
    if type(provenance) is not dict or set(provenance) != provenance_fields:
        raise ValueError("full-data validator proposal provenance schema mismatch")
    expected_sources = len({
        _require_sha256(row.get("source_sha256"), "full-data manifest source_sha256")
        for row in manifest_rows
    })
    if type(provenance.get("sources")) is not int or provenance["sources"] != expected_sources:
        raise ValueError("full-data validator proposal source count mismatch")
    if provenance.get("provider_kind") != "frozen_yolo_runtime":
        raise ValueError("full-data validator did not execute the frozen YOLO runtime")
    if provenance.get("authority") != "development_only_current_detector_reproduction":
        raise ValueError("full-data validator proposal authority mismatch")
    for field in (
        "runtime_detector_executed", "runtime_top1_replayed",
        "provided_top1_predictions_matched", "proposal_class_confidence_bbox_matched",
        "cuda_client_initialized_before_source_crop_scan",
        "detector_artifact_bytes_bound", "detector_replay_used_unique_snapshot",
        "source_and_label_replay_used_unique_snapshots",
        "replay_snapshots_verified_after_inference",
        "original_detector_bytes_unchanged_through_validation",
        "inference_spec_bytes_bound", "dataset_info_bytes_bound",
        "source_bbox_crop_bytes_recomputed",
    ):
        _require_bool(provenance.get(field), True, f"full-data validator provenance.{field}")
    for field in (
        "original_generation_event_cryptographically_attested",
        "production_or_blind_authority",
    ):
        _require_bool(provenance.get(field), False, f"full-data validator provenance.{field}")
    for field, expected in (
        ("confidence_abs_tolerance", 1e-6), ("bbox_abs_tolerance", 1e-4)
    ):
        actual = provenance.get(field)
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or actual != expected:
            raise ValueError(f"full-data validator provenance.{field} mismatch")


def _validate_qx3_quality_bindings(
    value: Mapping[str, object], *, description: str,
    quality_exclusions_sha256: str,
    quality_exclusion_assembly_receipt_sha256: str,
) -> None:
    bindings = value.get("bindings")
    if type(bindings) is not dict:
        raise ValueError(f"{description} bindings are missing")
    manifest_sha = _require_sha256(
        bindings.get("quality_exclusions_sha256"),
        f"{description} quality_exclusions_sha256",
    )
    legacy_alias_sha = _require_sha256(
        bindings.get("quality_exclusion_manifest_sha256"),
        f"{description} quality_exclusion_manifest_sha256",
    )
    receipt_sha = _require_sha256(
        bindings.get("quality_exclusion_assembly_receipt_sha256"),
        f"{description} quality_exclusion_assembly_receipt_sha256",
    )
    validator_sha = _require_sha256(
        bindings.get("quality_assembly_validator_sha256"),
        f"{description} quality_assembly_validator_sha256",
    )
    if manifest_sha != legacy_alias_sha:
        raise ValueError(f"{description} quality manifest aliases mismatch")
    if manifest_sha != quality_exclusions_sha256:
        raise ValueError(f"{description} quality exclusions binding mismatch")
    if receipt_sha != quality_exclusion_assembly_receipt_sha256:
        raise ValueError(f"{description} quality assembly receipt binding mismatch")
    current_validator_sha = _sha256_bytes(
        _stable_bytes(
            Path(quality_assembly_contract.__file__).resolve(),
            "quality assembly validator code",
        )
    )
    if validator_sha != current_validator_sha:
        raise ValueError(f"{description} quality assembly validator binding mismatch")


def _validate_qx3_ready(
    value: Mapping[str, object], *, quality_exclusions_sha256: str,
    quality_exclusion_assembly_receipt_sha256: str,
) -> None:
    if type(value.get("schema_version")) is not int or value.get("schema_version") != 1:
        raise ValueError("qx3 ready schema_version mismatch")
    if value.get("artifact_role") != QX3_READY_ROLE:
        raise ValueError("qx3 ready is not the frozen diagnostic artifact")
    if value.get("status") != "batch1_validator_ab_reproducibility_passed":
        raise ValueError("qx3 ready status mismatch")
    for field in (
        "lineage_execution_authorized", "judge_authority", "training_authority",
        "blind_test_authority", "candidate_promotion_authorized",
        "production_deployment_authorized",
    ):
        _require_bool(value.get(field), False, f"qx3 ready.{field}")
    if type(value.get("selected_sources")) is not int or value.get("selected_sources") != 3500:
        raise ValueError("qx3 ready must bind exactly 3500 selected sources")
    if type(value.get("validated_rows")) is not int or value.get("validated_rows") < 3465:
        raise ValueError("qx3 ready validated row coverage is below 99 percent")
    coverage = value.get("selected_source_coverage")
    if isinstance(coverage, bool) or not isinstance(coverage, (int, float)):
        raise ValueError("qx3 ready selected_source_coverage is invalid")
    if not math.isfinite(float(coverage)) or not 0.99 <= float(coverage) <= 1:
        raise ValueError("qx3 ready selected_source_coverage is below 99 percent")
    _validate_qx3_quality_bindings(
        value,
        description="qx3 ready",
        quality_exclusions_sha256=quality_exclusions_sha256,
        quality_exclusion_assembly_receipt_sha256=(
            quality_exclusion_assembly_receipt_sha256
        ),
    )


def _validate_qx3_report(
    value: Mapping[str, object], *, quality_exclusions_sha256: str,
    quality_exclusion_assembly_receipt_sha256: str,
) -> None:
    if type(value.get("schema_version")) is not int or value.get("schema_version") != 1:
        raise ValueError("qx3 report schema_version mismatch")
    if value.get("artifact_role") != QX3_REPORT_ROLE:
        raise ValueError("qx3 report is not validator A/B diagnostic evidence")
    if value.get("status") != "validator_ab_exact_reproduction":
        raise ValueError("qx3 report status mismatch")
    _require_bool(value.get("validated_manifest_bytes_equal"), True, "qx3 report bytes equal")
    _require_bool(
        value.get("report_core_contract_and_bindings_equal"), True,
        "qx3 report contract equal",
    )
    for field in (
        "lineage_execution_authorized", "training_authority",
        "blind_test_authority", "production_deployment_authorized",
    ):
        _require_bool(value.get(field), False, f"qx3 report.{field}")
    _validate_qx3_quality_bindings(
        value,
        description="qx3 report",
        quality_exclusions_sha256=quality_exclusions_sha256,
        quality_exclusion_assembly_receipt_sha256=(
            quality_exclusion_assembly_receipt_sha256
        ),
    )


def _validate_license_allowlist(value: Mapping[str, object]) -> Mapping[str, object]:
    if set(value) != {
        "schema", "artifact_role", "status", "commercial_training_allowlist",
        "origins",
    }:
        raise ValueError("license allowlist top-level fields mismatch")
    if value.get("schema") != "v4_commercial_training_license_allowlist.v1":
        raise ValueError("license allowlist schema mismatch")
    if value.get("artifact_role") != "commercial_training_license_evidence":
        raise ValueError("license allowlist artifact_role mismatch")
    if value.get("status") != "commercial_training_allowed":
        raise ValueError("license allowlist status mismatch")
    _require_bool(
        value.get("commercial_training_allowlist"), True,
        "commercial_training_allowlist",
    )
    origins = value.get("origins")
    if type(origins) is not dict or not origins:
        raise ValueError("license allowlist origins must be a nonempty object")
    exact_fields = {
        "kind", "dataset_id", "commercial_training_allowed",
        "redistribution_allowed", "evidence_sha256",
    }
    for origin, rule in origins.items():
        if type(origin) is not str or not origin or type(rule) is not dict:
            raise ValueError("license origin rule is invalid")
        if set(rule) != exact_fields:
            raise ValueError("license origin rule fields mismatch")
        if rule.get("kind") not in {"aihub", "operational"}:
            raise ValueError("license origin kind must be aihub or operational")
        dataset_id = rule.get("dataset_id")
        if type(dataset_id) is not str or not dataset_id:
            raise ValueError("license dataset_id must be nonempty")
        if rule.get("kind") == "aihub" and dataset_id not in {"71362", "AIHUB_71362"}:
            raise ValueError("only AI Hub dataset 71362 is commercially allowlisted")
        _require_bool(
            rule.get("commercial_training_allowed"), True,
            f"license origin {origin}.commercial_training_allowed",
        )
        if type(rule.get("redistribution_allowed")) is not bool:
            raise ValueError("license redistribution_allowed must be boolean")
        _require_sha256(rule.get("evidence_sha256"), f"license origin {origin} evidence")
    return origins


def _validate_protected_sources(value: Mapping[str, object]) -> set[str]:
    if set(value) != {"schema", "artifact_role", "status", *PROTECTED_FIELDS}:
        raise ValueError("protected holdout top-level fields mismatch")
    if value.get("schema") != "v4_candidate_protected_holdouts.v1":
        raise ValueError("protected holdout schema mismatch")
    if value.get("artifact_role") != "protected_holdouts_not_training_or_model_selection":
        raise ValueError("protected holdout artifact_role mismatch")
    if value.get("status") != "protected_holdouts_ready":
        raise ValueError("protected holdout status mismatch")
    per_field: dict[str, set[str]] = {}
    for field in PROTECTED_FIELDS:
        items = value.get(field)
        if type(items) is not list:
            raise ValueError(f"protected_sources.{field} must be an array")
        current = {_require_sha256(item, f"protected_sources.{field}") for item in items}
        if len(current) != len(items):
            raise ValueError(f"protected_sources.{field} contains duplicates")
        per_field[field] = current
    if len(per_field["qx3_diagnostic_source_sha256"]) != 3500:
        raise ValueError("qx3 diagnostic cohort must contain exactly 3500 source SHAs")
    if len(per_field["qx3_validation_source_sha256"]) != 1000:
        raise ValueError("qx3 validation holdout must contain exactly 1000 source SHAs")
    if not per_field["qx3_validation_source_sha256"].issubset(
        per_field["qx3_diagnostic_source_sha256"]
    ):
        raise ValueError("qx3 validation holdout must be a subset of the diagnostic cohort")
    if len(per_field["hardware41_source_sha256"]) != 41:
        raise ValueError("hardware holdout must contain exactly 41 source SHAs")
    allowed_overlap = {
        ("qx3_diagnostic_source_sha256", "qx3_validation_source_sha256")
    }
    for index, left in enumerate(PROTECTED_FIELDS):
        for right in PROTECTED_FIELDS[index + 1:]:
            if (left, right) in allowed_overlap:
                continue
            if per_field[left].intersection(per_field[right]):
                raise ValueError("protected holdout lists overlap")
    protected: set[str] = set()
    for current in per_field.values():
        protected.update(current)
    return protected


def _canonical_compact_payload(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _validate_protected_reference_inventory(
    value: Mapping[str, object], protected_sources: set[str],
) -> int:
    if set(value) != {"schema", "root", "objects"}:
        raise ValueError("protected reference inventory top-level fields mismatch")
    if value.get("schema") != PROTECTED_REFERENCE_INVENTORY_SCHEMA:
        raise ValueError("protected reference inventory schema mismatch")
    _require_normalized_posix_relative_path(
        value.get("root"), "protected reference inventory root"
    )
    objects = value.get("objects")
    if type(objects) is not list or not objects:
        raise ValueError("protected reference inventory objects must be nonempty")
    allowed_cohorts = {
        "qx3_diagnostic", "hardware41", "known_audit", "calibration", "blind_test",
    }
    seen_paths: set[str] = set()
    source_view_counts: Counter[str] = Counter()
    crop_view_counts: Counter[str] = Counter()
    for index, row in enumerate(objects):
        if type(row) is not dict or set(row) != {
            "cohort", "view_kind", "path", "size", "image_sha256", "source_sha256",
        }:
            raise ValueError(
                f"protected reference inventory object {index} fields mismatch"
            )
        cohort = row.get("cohort")
        if type(cohort) is not str or cohort not in allowed_cohorts:
            raise ValueError(
                f"protected reference inventory object {index} cohort is invalid"
            )
        view_kind = row.get("view_kind")
        if type(view_kind) is not str or view_kind not in {"source", "crop"}:
            raise ValueError(
                f"protected reference inventory object {index} view_kind is invalid"
            )
        relative = _require_normalized_posix_relative_path(
            row.get("path"), f"protected reference inventory object {index} path"
        )
        if relative in seen_paths:
            raise ValueError("protected reference inventory contains duplicate paths")
        seen_paths.add(relative)
        size = row.get("size")
        if type(size) is not int or not 0 < size <= IMAGE_CONSUMPTION_MAX_BYTES:
            raise ValueError(
                f"protected reference inventory object {index} size is invalid"
            )
        image_sha = _require_sha256(
            row.get("image_sha256"),
            f"protected reference inventory object {index} image SHA",
        )
        source_sha = _require_sha256(
            row.get("source_sha256"),
            f"protected reference inventory object {index} source SHA",
        )
        if source_sha not in protected_sources:
            raise ValueError(
                "protected holdout reference inventory contains an extra source SHA"
            )
        if view_kind == "source":
            if image_sha != source_sha:
                raise ValueError(
                    "protected reference source view image SHA must equal source SHA"
                )
            source_view_counts[source_sha] += 1
        else:
            crop_view_counts[source_sha] += 1
    if (
        set(source_view_counts) != protected_sources
        or any(count != 1 for count in source_view_counts.values())
        or set(crop_view_counts) != protected_sources
        or any(count != 1 for count in crop_view_counts.values())
    ):
        raise ValueError(
            "protected reference inventory must cover every protected source with "
            "exactly one source and one crop view"
        )
    return len(objects)


def _phash_bucket_keys(value: int) -> tuple[tuple[int, int], ...]:
    """Return the auditor's lossless pigeonhole keys for distance <= 4."""

    widths = [64 // (NEAR_DUPLICATE_PHASH_DISTANCE + 1)] * (
        NEAR_DUPLICATE_PHASH_DISTANCE + 1
    )
    for index in range(64 % (NEAR_DUPLICATE_PHASH_DISTANCE + 1)):
        widths[index] += 1
    keys: list[tuple[int, int]] = []
    offset = 0
    for index, width in enumerate(widths):
        keys.append((index, (value >> offset) & ((1 << width) - 1)))
        offset += width
    return tuple(keys)


def _reconstruct_near_duplicate_graph(
    entries: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Rebuild every auditor edge and connected component from report entries.

    Merely checking supplied edges would let a forged report omit the one edge
    that crosses train, validation, or a protected cohort.  This uses the same
    complete radius search and exact/source identity joins as the pinned
    auditor, while avoiding a quadratic scan for normal datasets.
    """

    ordered = sorted(entries, key=lambda entry: str(entry["asset_id"]))
    signatures: list[tuple[int, ...]] = [
        tuple(int(str(value), 16) for value in entry["phash_rot4"])  # type: ignore[index]
        for entry in ordered
    ]
    edge_values: dict[tuple[int, int], dict[str, object]] = {}
    buckets: dict[tuple[int, int], set[int]] = {}
    for right, signature in enumerate(signatures):
        unique_hashes = tuple(sorted(set(signature)))
        candidates: set[int] = set()
        for value in unique_hashes:
            for key in _phash_bucket_keys(value):
                candidates.update(buckets.get(key, ()))
        for left in sorted(candidates):
            distance = min(
                (left_value ^ right_value).bit_count()
                for left_value in signatures[left]
                for right_value in unique_hashes
            )
            if distance <= NEAR_DUPLICATE_PHASH_DISTANCE:
                edge_values[(left, right)] = {
                    "distance": distance,
                    "evidence": {"perceptual_hash"},
                }
                if len(edge_values) > NEAR_DUPLICATE_GRAPH_EDGE_CAP:
                    raise ValueError("near-duplicate audit graph exceeds edge cap")
        for value in unique_hashes:
            for key in _phash_bucket_keys(value):
                buckets.setdefault(key, set()).add(right)

    for evidence_name, field in (
        ("exact_image_sha256", "image_sha256"),
        ("source_sha256", "source_sha256"),
    ):
        groups: dict[str, list[int]] = {}
        for index, entry in enumerate(ordered):
            groups.setdefault(str(entry[field]), []).append(index)
        for indexes in groups.values():
            for position, left in enumerate(indexes):
                for right in indexes[position + 1:]:
                    edge = edge_values.setdefault(
                        (left, right), {"distance": None, "evidence": set()}
                    )
                    if len(edge_values) > NEAR_DUPLICATE_GRAPH_EDGE_CAP:
                        raise ValueError("near-duplicate audit graph exceeds edge cap")
                    evidence = edge["evidence"]
                    assert type(evidence) is set
                    evidence.add(evidence_name)
                    if evidence_name == "exact_image_sha256":
                        edge["distance"] = 0

    parent = list(range(len(ordered)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            low, high = sorted((left_root, right_root))
            parent[high] = low

    for left, right in sorted(edge_values):
        union(left, right)

    rendered_edges: list[dict[str, object]] = []
    for (left, right), raw in sorted(
        edge_values.items(),
        key=lambda item: (
            str(ordered[item[0][0]]["asset_id"]),
            str(ordered[item[0][1]]["asset_id"]),
        ),
    ):
        left_id, right_id = sorted(
            (
                str(ordered[left]["asset_id"]),
                str(ordered[right]["asset_id"]),
            )
        )
        evidence = raw["evidence"]
        assert type(evidence) is set
        rendered_edges.append({
            "left_asset_id": left_id,
            "right_asset_id": right_id,
            "distance": raw["distance"],
            "evidence": sorted(evidence),
            "blocking": ordered[left]["role"] != ordered[right]["role"],
        })

    members_by_root: dict[int, list[int]] = {}
    for index in range(len(ordered)):
        members_by_root.setdefault(find(index), []).append(index)
    edge_count_by_root: Counter[int] = Counter()
    for left, _right in edge_values:
        edge_count_by_root[find(left)] += 1

    clusters: list[dict[str, object]] = []
    for root, indexes in members_by_root.items():
        member_image_shas = sorted(
            {str(ordered[index]["image_sha256"]) for index in indexes}
        )
        roles = sorted({str(ordered[index]["role"]) for index in indexes})
        clusters.append({
            "cluster_id": _sha256_bytes(
                _canonical_compact_payload({
                    "algorithm_id": NEAR_DUPLICATE_ALGORITHM_ID,
                    "threshold": NEAR_DUPLICATE_PHASH_DISTANCE,
                    "member_image_sha256s": member_image_shas,
                })
            ),
            "member_asset_ids": sorted(
                str(ordered[index]["asset_id"]) for index in indexes
            ),
            "member_image_sha256s": member_image_shas,
            "roles": roles,
            "cohorts": sorted(
                {str(ordered[index]["cohort"]) for index in indexes}
            ),
            "view_kinds": sorted(
                {str(ordered[index]["view_kind"]) for index in indexes}
            ),
            "edge_count": edge_count_by_root[root],
            "multi_role": len(roles) > 1,
            "blocking": len(roles) > 1,
        })
    clusters.sort(key=lambda item: str(item["cluster_id"]))
    return rendered_edges, clusters


def _validate_near_duplicate_audit(
    value: Mapping[str, object], content: bytes, *,
    candidate_manifest_sha256: Mapping[str, str],
    candidate_rows_by_role: Mapping[str, Sequence[Mapping[str, str]]],
    candidate_asset_count: int,
    protected_sources_value: Mapping[str, object],
    protected_sources_content: bytes,
    protected_sources: set[str],
    protected_inventory_value: Mapping[str, object],
    protected_inventory_content: bytes,
    protected_asset_count: int,
    auditor_sha256: str,
    auditor_runtime_code_sha256: str,
) -> None:
    expected_fields = {
        "schema", "status", "ok", "artifact_role", "authority", "algorithm",
        "bindings", "coverage", "summary", "entries", "edges", "clusters",
    }
    if set(value) != expected_fields:
        raise ValueError("near-duplicate audit top-level fields mismatch")
    # The auditor has a deterministic canonical renderer. Requiring those exact
    # bytes lets downstream consumers recompute the policy-pinned report hash
    # from the report embedded in this authority.
    if content != _canonical_compact_payload(value) + b"\n":
        raise ValueError("near-duplicate audit is not canonical auditor output")
    if value.get("schema") != NEAR_DUPLICATE_REPORT_SCHEMA:
        raise ValueError("near-duplicate audit schema mismatch")
    if value.get("status") != "passed" or value.get("ok") is not True:
        raise ValueError("near-duplicate audit did not pass")
    if value.get("artifact_role") != "candidate_dataset_separation_evidence_only":
        raise ValueError("near-duplicate audit artifact_role mismatch")

    authority = value.get("authority")
    expected_authority = {
        "candidate_only": True,
        "label_authority": False,
        "blind_authority": False,
        "promotion_authority": False,
        "deployment_authority": False,
        "automatic_delete_or_relabel": False,
    }
    if authority != expected_authority:
        raise ValueError("near-duplicate audit authority must be separation evidence only")

    algorithm = value.get("algorithm")
    if type(algorithm) is not dict or set(algorithm) != {
        "id", "threshold", "decode", "views", "resize", "dct", "bit_rule",
        "byte_cap", "pixel_cap", "exact_right_angle_rotation_invariant",
        "crop_invariant", "graph_edge_cap", "runtime",
    }:
        raise ValueError("near-duplicate audit algorithm schema mismatch")
    fixed_algorithm = {
        "id": NEAR_DUPLICATE_ALGORITHM_ID,
        "threshold": NEAR_DUPLICATE_PHASH_DISTANCE,
        "decode": "verified_bytes_cv2_grayscale_ignore_exif_orientation",
        "views": ["rot0", "rot90", "rot180", "rot270"],
        "resize": {"width": 32, "height": 32, "interpolation": "INTER_AREA"},
        "dct": {"dtype": "float32", "low_frequency_block": [8, 8]},
        "bit_rule": (
            "row_major_msb_first; median(coefficients[1:]); "
            "coefficient>median; dc=0"
        ),
        "byte_cap": IMAGE_CONSUMPTION_MAX_BYTES,
        "pixel_cap": NEAR_DUPLICATE_PIXEL_CAP,
        "exact_right_angle_rotation_invariant": True,
        "crop_invariant": False,
        "graph_edge_cap": NEAR_DUPLICATE_GRAPH_EDGE_CAP,
    }
    for field, expected in fixed_algorithm.items():
        if algorithm.get(field) != expected:
            raise ValueError(f"near-duplicate audit algorithm {field} mismatch")
    runtime = algorithm.get("runtime")
    if type(runtime) is not dict or set(runtime) != {
        "python", "opencv", "numpy", "pillow", "opencv_build_information_sha256",
    }:
        raise ValueError("near-duplicate audit runtime schema mismatch")
    for field in ("python", "opencv", "numpy", "pillow"):
        if type(runtime.get(field)) is not str or not runtime[field]:
            raise ValueError(f"near-duplicate audit runtime {field} is invalid")
    _require_sha256(
        runtime.get("opencv_build_information_sha256"),
        "near-duplicate audit OpenCV build information SHA",
    )

    bindings = value.get("bindings")
    if type(bindings) is not dict or set(bindings) != {
        "candidate_manifest_sha256", "candidate_payload_set_sha256",
        "protected_payload_set_sha256", "protected_sources",
        "protected_inventory", "auditor",
    }:
        raise ValueError("near-duplicate audit binding schema mismatch")
    expected_candidate_bindings = {
        role: _require_sha256(digest, f"candidate {role} manifest SHA")
        for role, digest in sorted(candidate_manifest_sha256.items())
    }
    if set(expected_candidate_bindings) != set(ROLE_SPLITS):
        raise ValueError("near-duplicate audit producer candidate role schema mismatch")
    if bindings.get("candidate_manifest_sha256") != expected_candidate_bindings:
        raise ValueError("near-duplicate audit candidate manifest binding mismatch")
    _require_sha256(
        bindings.get("candidate_payload_set_sha256"),
        "near-duplicate audit candidate payload-set SHA",
    )
    _require_sha256(
        bindings.get("protected_payload_set_sha256"),
        "near-duplicate audit protected payload-set SHA",
    )

    protected_binding = bindings.get("protected_sources")
    if type(protected_binding) is not dict or set(protected_binding) != {
        "file_sha256", "payload_sha256", "canonical_union_sha256",
    }:
        raise ValueError("near-duplicate audit protected-source binding schema mismatch")
    expected_union_sha = _sha256_bytes(
        _canonical_compact_payload(sorted(protected_sources))
    )
    if protected_binding != {
        "file_sha256": _sha256_bytes(protected_sources_content),
        "payload_sha256": _sha256_bytes(
            _canonical_compact_payload(protected_sources_value)
        ),
        "canonical_union_sha256": expected_union_sha,
    }:
        raise ValueError("near-duplicate audit protected-source binding mismatch")

    inventory_binding = bindings.get("protected_inventory")
    if type(inventory_binding) is not dict or set(inventory_binding) != {
        "file_sha256", "payload_sha256",
    }:
        raise ValueError("near-duplicate audit protected-inventory binding schema mismatch")
    if inventory_binding != {
        "file_sha256": _sha256_bytes(protected_inventory_content),
        "payload_sha256": _sha256_bytes(
            _canonical_compact_payload(protected_inventory_value)
        ),
    }:
        raise ValueError("near-duplicate audit protected-inventory binding mismatch")
    if bindings.get("auditor") != {
        "path": NEAR_DUPLICATE_AUDITOR_PATH,
        "sha256": auditor_sha256,
        "runtime_code_sha256": auditor_runtime_code_sha256,
    }:
        raise ValueError("near-duplicate audit auditor binding mismatch")

    coverage = value.get("coverage")
    expected_coverage = {
        "candidate_assets": candidate_asset_count,
        "protected_assets": protected_asset_count,
        "protected_source_union": len(protected_sources),
        "verified_assets": candidate_asset_count + protected_asset_count,
        "complete": True,
    }
    if coverage != expected_coverage or not protected_sources or protected_asset_count < 1:
        raise ValueError("near-duplicate audit coverage is incomplete")

    entries = value.get("entries")
    edges = value.get("edges")
    clusters = value.get("clusters")
    summary = value.get("summary")
    if type(entries) is not list or type(edges) is not list or type(clusters) is not list:
        raise ValueError("near-duplicate audit evidence arrays are invalid")
    if len(entries) != expected_coverage["verified_assets"]:
        raise ValueError("near-duplicate audit entry coverage is incomplete")
    if len(edges) > NEAR_DUPLICATE_GRAPH_EDGE_CAP:
        raise ValueError("near-duplicate supplied edge array exceeds its graph edge cap")
    entries_by_id: dict[str, dict[str, object]] = {}
    candidate_source_views: Counter[tuple[str, str]] = Counter()
    candidate_crop_views: Counter[tuple[str, str]] = Counter()
    protected_source_views: Counter[str] = Counter()
    protected_crop_views: Counter[str] = Counter()
    candidate_roles = set(ROLE_SPLITS)
    protected_roles = {
        "qx3_diagnostic", "hardware41", "known_audit", "calibration", "blind_test",
    }
    expected_candidate_source_entries: set[tuple[str, str, str, str]] = set()
    expected_candidate_crop_entries: set[tuple[str, str, str, str]] = set()
    expected_candidate_rows = 0
    if set(candidate_rows_by_role) != candidate_roles:
        raise ValueError("near-duplicate audit producer candidate rows are incomplete")
    for role, rows in candidate_rows_by_role.items():
        for row in rows:
            expected_candidate_rows += 1
            source_sha = _require_sha256(
                row.get("source_sha256"), "candidate manifest source SHA"
            )
            image_sha = _require_sha256(
                row.get("image_sha256"), "candidate manifest image SHA"
            )
            sample_id = row.get("sample_id")
            if type(sample_id) is not str or not sample_id:
                raise ValueError("candidate manifest sample ID is invalid")
            expected_candidate_source_entries.add(
                (role, f"source:{role}:{source_sha}", source_sha, source_sha)
            )
            expected_candidate_crop_entries.add(
                (role, sample_id, source_sha, image_sha)
            )
    if (
        candidate_asset_count != 2 * expected_candidate_rows
        or len(expected_candidate_source_entries) != expected_candidate_rows
        or len(expected_candidate_crop_entries) != expected_candidate_rows
    ):
        raise ValueError(
            "near-duplicate audit producer candidate rows violate one-source/one-crop"
        )

    expected_protected_entries: set[
        tuple[str, str, str, str, str, int]
    ] = set()
    protected_objects = protected_inventory_value.get("objects")
    if type(protected_objects) is not list:
        raise ValueError("protected reference inventory objects are invalid")
    for index, row in enumerate(protected_objects):
        assert type(row) is dict
        cohort = str(row["cohort"])
        view_kind = str(row["view_kind"])
        relative = _require_normalized_posix_relative_path(
            row.get("path"), f"protected reference inventory object {index} path"
        )
        source_sha = _require_sha256(
            row.get("source_sha256"),
            f"protected reference inventory object {index} source SHA",
        )
        image_sha = _require_sha256(
            row.get("image_sha256"),
            f"protected reference inventory object {index} image SHA",
        )
        size = row.get("size")
        assert type(size) is int
        sample_suffix = _sha256_bytes(relative.encode("utf-8"))[:16]
        expected_protected_entries.add((
            cohort,
            view_kind,
            f"protected:{cohort}:{view_kind}:{image_sha[:16]}:{sample_suffix}",
            source_sha,
            image_sha,
            size,
        ))
    if len(expected_protected_entries) != protected_asset_count:
        raise ValueError("protected reference inventory identities are duplicated")

    candidate_source_entries: set[tuple[str, str, str, str]] = set()
    candidate_crop_entries: set[tuple[str, str, str, str]] = set()
    protected_entries: set[tuple[str, str, str, str, str, int]] = set()
    for index, entry in enumerate(entries):
        if type(entry) is not dict or set(entry) != {
            "asset_id", "role", "cohort", "view_kind", "sample_id",
            "source_sha256", "image_sha256", "size", "width", "height",
            "phash_rot4",
        }:
            raise ValueError(f"near-duplicate audit entry {index} fields mismatch")
        role = entry.get("role")
        cohort = entry.get("cohort")
        view_kind = entry.get("view_kind")
        sample_id = entry.get("sample_id")
        if type(role) is not str or role not in candidate_roles | protected_roles:
            raise ValueError(f"near-duplicate audit entry {index} role is invalid")
        if type(cohort) is not str or (
            (role in candidate_roles and cohort != "candidate")
            or (role in protected_roles and cohort != role)
        ):
            raise ValueError(f"near-duplicate audit entry {index} cohort is invalid")
        if type(view_kind) is not str or view_kind not in {"source", "crop"}:
            raise ValueError(f"near-duplicate audit entry {index} view_kind is invalid")
        if type(sample_id) is not str or not sample_id:
            raise ValueError(f"near-duplicate audit entry {index} sample_id is invalid")
        source_sha = _require_sha256(
            entry.get("source_sha256"), f"near-duplicate audit entry {index} source SHA"
        )
        image_sha = _require_sha256(
            entry.get("image_sha256"), f"near-duplicate audit entry {index} image SHA"
        )
        asset_id = _require_sha256(
            entry.get("asset_id"), f"near-duplicate audit entry {index} asset ID"
        )
        expected_asset_id = _sha256_bytes(
            _canonical_compact_payload(
                {
                    "cohort": cohort,
                    "image_sha256": image_sha,
                    "role": role,
                    "sample_id": sample_id,
                    "source_sha256": source_sha,
                    "view_kind": view_kind,
                }
            )
        )
        if asset_id != expected_asset_id or asset_id in entries_by_id:
            raise ValueError("near-duplicate audit asset identity is invalid or duplicated")
        entries_by_id[asset_id] = entry
        for field in ("size", "width", "height"):
            number = entry.get(field)
            if type(number) is not int or number < 1:
                raise ValueError(f"near-duplicate audit entry {index} {field} is invalid")
        if entry["size"] > IMAGE_CONSUMPTION_MAX_BYTES or (
            entry["width"] * entry["height"] > NEAR_DUPLICATE_PIXEL_CAP
        ):
            raise ValueError("near-duplicate audit entry exceeds resource limits")
        signatures = entry.get("phash_rot4")
        if type(signatures) is not list or len(signatures) != 4 or any(
            type(signature) is not str
            or re.fullmatch(r"[0-9a-f]{16}", signature) is None
            for signature in signatures
        ):
            raise ValueError(f"near-duplicate audit entry {index} pHash is invalid")
        if role in candidate_roles:
            if source_sha in protected_sources:
                raise ValueError("near-duplicate audit candidate uses a protected source")
            if view_kind == "source":
                if image_sha != source_sha:
                    raise ValueError("near-duplicate candidate source view SHA mismatch")
                candidate_source_views[(role, source_sha)] += 1
                candidate_source_entries.add(
                    (role, sample_id, source_sha, image_sha)
                )
            else:
                candidate_crop_views[(role, source_sha)] += 1
                candidate_crop_entries.add(
                    (role, sample_id, source_sha, image_sha)
                )
        else:
            if source_sha not in protected_sources:
                raise ValueError("near-duplicate audit protected entry source is unknown")
            if view_kind == "source":
                if image_sha != source_sha:
                    raise ValueError("near-duplicate protected source view SHA mismatch")
                protected_source_views[source_sha] += 1
            else:
                protected_crop_views[source_sha] += 1
            protected_entries.add((
                role,
                view_kind,
                sample_id,
                source_sha,
                image_sha,
                int(entry["size"]),
            ))
    if (
        len(candidate_source_views) != candidate_asset_count // 2
        or any(count != 1 for count in candidate_source_views.values())
        or candidate_crop_views != candidate_source_views
        or set(protected_source_views) != protected_sources
        or any(count != 1 for count in protected_source_views.values())
        or protected_crop_views != protected_source_views
    ):
        raise ValueError("near-duplicate audit source/crop evidence coverage is incomplete")
    if (
        candidate_source_entries != expected_candidate_source_entries
        or candidate_crop_entries != expected_candidate_crop_entries
    ):
        raise ValueError(
            "near-duplicate audit candidate entries differ from exact manifest rows"
        )
    if protected_entries != expected_protected_entries:
        raise ValueError(
            "near-duplicate audit protected entries differ from exact inventory rows"
        )
    if entries != sorted(entries, key=lambda entry: str(entry["asset_id"])):
        raise ValueError("near-duplicate audit entries are not deterministically ordered")
    reconstructed_edges, reconstructed_clusters = _reconstruct_near_duplicate_graph(
        entries
    )
    if edges != reconstructed_edges:
        raise ValueError("near-duplicate audit edge set is incomplete or inconsistent")
    if clusters != reconstructed_clusters:
        raise ValueError("near-duplicate audit cluster graph is incomplete or inconsistent")
    candidate_payload_sha = _sha256_bytes(
        _canonical_compact_payload(
            sorted(
                {
                    str(entry["image_sha256"])
                    for entry in entries
                    if entry["role"] in candidate_roles
                }
            )
        )
    )
    protected_payload_sha = _sha256_bytes(
        _canonical_compact_payload(
            sorted(
                {
                    str(entry["image_sha256"])
                    for entry in entries
                    if entry["role"] in protected_roles
                }
            )
        )
    )
    if (
        bindings["candidate_payload_set_sha256"] != candidate_payload_sha
        or bindings["protected_payload_set_sha256"] != protected_payload_sha
    ):
        raise ValueError("near-duplicate audit payload-set binding mismatch")
    if type(summary) is not dict or set(summary) != {
        "edges", "clusters", "blocking_multi_role_clusters",
        "same_role_duplicate_clusters_nonblocking",
    }:
        raise ValueError("near-duplicate audit summary schema mismatch")
    for field in summary:
        if type(summary[field]) is not int or summary[field] < 0:
            raise ValueError(f"near-duplicate audit summary {field} is invalid")
    if summary["edges"] != len(edges) or summary["clusters"] != len(clusters):
        raise ValueError("near-duplicate audit summary evidence counts mismatch")
    edge_pairs: set[tuple[str, str]] = set()
    for edge in edges:
        if type(edge) is not dict or set(edge) != {
            "left_asset_id", "right_asset_id", "distance", "evidence", "blocking",
        }:
            raise ValueError("near-duplicate audit edge is invalid")
        left = edge.get("left_asset_id")
        right = edge.get("right_asset_id")
        if (
            type(left) is not str
            or type(right) is not str
            or left not in entries_by_id
            or right not in entries_by_id
            or left >= right
            or (left, right) in edge_pairs
        ):
            raise ValueError("near-duplicate audit edge endpoints are invalid")
        edge_pairs.add((left, right))
        distance = edge.get("distance")
        if distance is not None and (
            type(distance) is not int
            or not 0 <= distance <= NEAR_DUPLICATE_PHASH_DISTANCE
        ):
            raise ValueError("near-duplicate audit edge distance is invalid")
        evidence = edge.get("evidence")
        if (
            type(evidence) is not list
            or not evidence
            or evidence != sorted(evidence)
            or len(set(evidence)) != len(evidence)
            or set(evidence).difference(
                {"perceptual_hash", "exact_image_sha256", "source_sha256"}
            )
        ):
            raise ValueError("near-duplicate audit edge evidence is invalid")
        cross_role = entries_by_id[left]["role"] != entries_by_id[right]["role"]
        if edge.get("blocking") is not cross_role or cross_role:
            raise ValueError("near-duplicate audit contains a forbidden cross-role edge")
    blocking_clusters = 0
    same_role_clusters = 0
    clustered_asset_ids: set[str] = set()
    for cluster in clusters:
        if type(cluster) is not dict or set(cluster) != {
            "cluster_id", "member_asset_ids", "member_image_sha256s", "roles",
            "cohorts", "view_kinds", "edge_count", "multi_role", "blocking",
        }:
            raise ValueError("near-duplicate audit cluster is invalid")
        member_ids = cluster.get("member_asset_ids")
        if (
            type(member_ids) is not list
            or not member_ids
            or any(type(member) is not str for member in member_ids)
            or member_ids != sorted(member_ids)
            or len(set(member_ids)) != len(member_ids)
            or any(member not in entries_by_id for member in member_ids)
            or clustered_asset_ids.intersection(member_ids)
        ):
            raise ValueError("near-duplicate audit cluster membership is invalid")
        clustered_asset_ids.update(member_ids)
        member_entries = [entries_by_id[member] for member in member_ids]
        roles = cluster.get("roles")
        expected_roles = sorted({str(entry["role"]) for entry in member_entries})
        expected_cohorts = sorted({str(entry["cohort"]) for entry in member_entries})
        expected_views = sorted({str(entry["view_kind"]) for entry in member_entries})
        expected_images = sorted({str(entry["image_sha256"]) for entry in member_entries})
        if (
            roles != expected_roles
            or cluster.get("cohorts") != expected_cohorts
            or cluster.get("view_kinds") != expected_views
            or cluster.get("member_image_sha256s") != expected_images
        ):
            raise ValueError("near-duplicate audit cluster roles are invalid")
        expected_cluster_id = _sha256_bytes(
            _canonical_compact_payload(
                {
                    "algorithm_id": NEAR_DUPLICATE_ALGORITHM_ID,
                    "threshold": NEAR_DUPLICATE_PHASH_DISTANCE,
                    "member_image_sha256s": expected_images,
                }
            )
        )
        if cluster.get("cluster_id") != expected_cluster_id:
            raise ValueError("near-duplicate audit cluster ID is invalid")
        blocking = cluster.get("blocking")
        multi_role = cluster.get("multi_role")
        edge_count = cluster.get("edge_count")
        if type(blocking) is not bool or type(multi_role) is not bool:
            raise ValueError("near-duplicate audit cluster flags are invalid")
        if type(edge_count) is not int or edge_count < 0:
            raise ValueError("near-duplicate audit cluster edge_count is invalid")
        if blocking or multi_role or len(set(roles)) > 1:
            blocking_clusters += 1
        elif edge_count > 0:
            same_role_clusters += 1
    if blocking_clusters or summary["blocking_multi_role_clusters"] != 0:
        raise ValueError("near-duplicate audit contains a forbidden cross-role cluster")
    if clustered_asset_ids != set(entries_by_id):
        raise ValueError("near-duplicate audit clusters do not partition all entries")
    if summary["same_role_duplicate_clusters_nonblocking"] != same_role_clusters:
        raise ValueError("near-duplicate audit same-role cluster count mismatch")


def _validate_training_config(value: Mapping[str, object]) -> None:
    if set(value) != CONFIG_FIELDS or value.get("schema") != "v4_candidate_training_config.v2":
        raise ValueError("training config fields/schema mismatch")
    if value.get("backbone") != "mobilenet_v3_small":
        raise ValueError("training backbone must be mobilenet_v3_small")
    _require_bool(value.get("pretrained"), True, "training config pretrained")
    if value.get("condition_heads") != list(CONDITION_HEADS):
        raise ValueError("training condition head order mismatch")
    if type(value.get("input_size")) is not int or value.get("input_size") != 320:
        raise ValueError("training input_size must be integer 320")
    for field in ("epochs", "patience", "batch"):
        if type(value.get(field)) is not int or value[field] < 1:
            raise ValueError(f"training {field} must be a positive integer")
    if type(value.get("workers")) is not int or value["workers"] < 0:
        raise ValueError("training workers must be a non-negative integer")
    if type(value.get("seed")) is not int or value["seed"] < 0:
        raise ValueError("training seed must be a non-negative integer")
    if value.get("image_consumption_contract_version") != (
        IMAGE_CONSUMPTION_CONTRACT_VERSION
    ):
        raise ValueError("training image consumption contract version mismatch")
    if value.get("image_max_bytes") != IMAGE_CONSUMPTION_MAX_BYTES:
        raise ValueError("training image_max_bytes mismatch")
    if value.get("image_max_pixels") != IMAGE_CONSUMPTION_MAX_PIXELS:
        raise ValueError("training image_max_pixels mismatch")
    for field in (
        "lr", "backbone_lr", "head_lr", "objectness_weight",
        "material_weight", "condition_weight",
    ):
        number = value.get(field)
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ValueError(f"training {field} must be numeric")
        if not math.isfinite(float(number)) or float(number) <= 0:
            raise ValueError(f"training {field} must be finite and positive")
    for field in ("label_smoothing", "class_weight_beta"):
        number = value.get(field)
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ValueError(f"training {field} must be numeric")
        if not math.isfinite(float(number)) or not 0 <= float(number) < 1:
            raise ValueError(f"training {field} must be finite in [0,1)")
    if value.get("class_weight_mode") not in {"none", "inverse", "effective-number"}:
        raise ValueError("training class_weight_mode is unsupported")
    if value.get("optimizer") != "AdamW":
        raise ValueError("training optimizer must be AdamW")
    if value.get("optimizer_betas") != [0.9, 0.999]:
        raise ValueError("training optimizer_betas must be [0.9, 0.999]")
    weight_decay = value.get("weight_decay")
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, (int, float))
        or float(weight_decay) != 0.0001
    ):
        raise ValueError("training weight_decay must be exactly 0.0001")
    if value.get("scheduler") != "CosineAnnealingLR":
        raise ValueError("training scheduler must be CosineAnnealingLR")
    if type(value.get("scheduler_t_max")) is not int or value.get(
        "scheduler_t_max"
    ) != value.get("epochs"):
        raise ValueError("training scheduler_t_max must exactly equal epochs")
    scheduler_eta_min = value.get("scheduler_eta_min")
    if (
        isinstance(scheduler_eta_min, bool)
        or not isinstance(scheduler_eta_min, (int, float))
        or float(scheduler_eta_min) != 0.0
    ):
        raise ValueError("training scheduler_eta_min must be exactly zero")
    if value.get("sampling_mode") not in {
        "weighted_replacement", "shuffle_without_replacement"
    }:
        raise ValueError("training sampling_mode is unsupported")
    if (
        type(value.get("sampling_samples_per_epoch")) is not int
        or value["sampling_samples_per_epoch"] < 1
    ):
        raise ValueError("training sampling_samples_per_epoch must be positive")
    expected_fractions = value.get("sampling_expected_fraction_by_origin")
    if type(expected_fractions) is not dict or not expected_fractions:
        raise ValueError("training sampling fractions must be a nonempty object")
    for origin, fraction in expected_fractions.items():
        if (
            type(origin) is not str or not origin
            or isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction))
            or not 0 < float(fraction) <= 1
        ):
            raise ValueError("training sampling fraction is invalid")
    if not math.isclose(
        math.fsum(float(value) for value in expected_fractions.values()),
        1.0, rel_tol=0.0, abs_tol=1e-12,
    ):
        raise ValueError("training sampling fractions must sum to one")
    weights = value.get("origin_weights")
    if type(weights) is not dict:
        raise ValueError("training origin_weights must be an object")
    for origin, weight in weights.items():
        if type(origin) is not str or not origin or "=" in origin:
            raise ValueError("training origin weight name is invalid")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError("training origin weight must be numeric")
        if not math.isfinite(float(weight)) or float(weight) <= 0:
            raise ValueError("training origin weight must be finite and positive")


def _validate_code_inventory(value: Mapping[str, object]) -> None:
    if set(value) != {"schema", "root", "file_count", "files"}:
        raise ValueError("code inventory top-level fields mismatch")
    if value.get("schema") != "v4_candidate_code_inventory.v1":
        raise ValueError("code inventory schema mismatch")
    root_text = value.get("root")
    files = value.get("files")
    count = value.get("file_count")
    if type(root_text) is not str or type(files) is not list or type(count) is not int:
        raise ValueError("code inventory root/files/count types are invalid")
    root = _reject_symlink_components(Path(root_text), "code inventory root")
    if not root.is_dir() or root.is_symlink() or count != len(files) or not files:
        raise ValueError("code inventory root/count is invalid")
    expected: set[Path] = set()
    for row in files:
        if type(row) is not dict or set(row) != {"path", "size", "sha256"}:
            raise ValueError("code inventory row schema mismatch")
        relative = row.get("path")
        if type(relative) is not str or not relative or Path(relative).is_absolute():
            raise ValueError("code inventory path must be relative")
        relative_path = Path(relative)
        if ".." in relative_path.parts or relative_path.as_posix() != relative:
            raise ValueError("code inventory path must be normalized POSIX relative")
        if relative in TRUST_ROOT_CODE_PATHS:
            raise ValueError("trust-root scripts must not be inside the policy-bound inventory")
        path = _regular_file(root / relative, "inventoried code file")
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("code inventory path escapes root") from error
        if path in expected:
            raise ValueError("duplicate code inventory path")
        content = _stable_bytes(path, "inventoried code file")
        if type(row.get("size")) is not int or row.get("size") != len(content):
            raise ValueError("code inventory size mismatch")
        if _require_sha256(row.get("sha256"), "code inventory SHA") != _sha256_bytes(content):
            raise ValueError("code inventory SHA mismatch")
        expected.add(path)
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("code inventory root must not contain symlinks")
    actual = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in TRUST_ROOT_CODE_PATHS
    }
    if actual != expected:
        raise ValueError("code inventory does not cover the exact regular-file set")


def _require_normalized_posix_relative_path(value: object, field: str) -> str:
    if (
        type(value) is not str
        or not value
        or "\x00" in value
        or "\\" in value
        or posixpath.isabs(value)
        or value == "."
        or posixpath.normpath(value) != value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError(f"{field} must be a normalized POSIX relative path")
    return value


def _resolve_qnap_inventory_target(link_path: str, target: object) -> str:
    if (
        type(target) is not str
        or not target
        or "\x00" in target
        or "\\" in target
        or posixpath.isabs(target)
        or posixpath.normpath(target) != target
    ):
        raise ValueError(
            "QNAP library inventory symlink target must be normalized and relative"
        )
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(link_path), target))
    if resolved == ".." or resolved.startswith("../") or posixpath.isabs(resolved):
        raise ValueError("QNAP library inventory symlink target escapes its tree")
    return resolved


def _validate_qnap_library_inventory(value: object) -> None:
    if type(value) is not dict or set(value) != {
        "schema", "snapshot_max_bytes", "trees", "required_mapped_libraries"
    }:
        raise ValueError("QNAP library inventory top-level schema mismatch")
    if value.get("schema") != QNAP_LIBRARY_INVENTORY_SCHEMA:
        raise ValueError("QNAP library inventory schema mismatch")
    if (
        type(value.get("snapshot_max_bytes")) is not int
        or value.get("snapshot_max_bytes") != QNAP_LIBRARY_SNAPSHOT_MAX_BYTES
    ):
        raise ValueError("QNAP library inventory snapshot_max_bytes mismatch")
    trees = value.get("trees")
    if type(trees) is not list or len(trees) != len(ALLOWED_QNAP_LIBRARY_MOUNTS):
        raise ValueError("QNAP library inventory must contain exactly two trees")
    ordered_roots = [
        (tree.get("source_root"), tree.get("container_root"))
        for tree in trees
        if type(tree) is dict
    ]
    if ordered_roots != sorted(ALLOWED_QNAP_LIBRARY_MOUNTS):
        raise ValueError("QNAP library inventory trees must be source-root sorted")

    seen_roots: set[tuple[str, str]] = set()
    tree_entries: dict[
        str, tuple[dict[str, Mapping[str, object]], dict[str, str]]
    ] = {}
    combined_regular_bytes = 0
    for tree in trees:
        if type(tree) is not dict or set(tree) != {
            "source_root", "container_root", "total_regular_bytes",
            "tree_sha256", "entries",
        }:
            raise ValueError("QNAP library inventory tree schema mismatch")
        source_root = tree.get("source_root")
        container_root = tree.get("container_root")
        if type(source_root) is not str or type(container_root) is not str:
            raise ValueError("QNAP library inventory tree roots must be strings")
        roots = (source_root, container_root)
        if roots not in ALLOWED_QNAP_LIBRARY_MOUNTS or roots in seen_roots:
            raise ValueError("QNAP library inventory tree roots mismatch")
        seen_roots.add(roots)

        entries = tree.get("entries")
        if type(entries) is not list or not entries:
            raise ValueError("QNAP library inventory tree entries must be nonempty")
        paths: list[str] = []
        entries_by_path: dict[str, Mapping[str, object]] = {}
        resolved_targets: dict[str, str] = {}
        regular_bytes = 0
        for entry in entries:
            if type(entry) is not dict:
                raise ValueError("QNAP library inventory entry must be an object")
            entry_type = entry.get("type")
            if entry_type == "file":
                if set(entry) != {"path", "type", "size", "sha256"}:
                    raise ValueError("QNAP library inventory file schema mismatch")
                size = entry.get("size")
                if type(size) is not int or size < 0:
                    raise ValueError("QNAP library inventory file size is invalid")
                _require_sha256(
                    entry.get("sha256"), "QNAP library inventory file SHA"
                )
                regular_bytes += size
            elif entry_type == "symlink":
                if set(entry) != {"path", "type", "target"}:
                    raise ValueError("QNAP library inventory symlink schema mismatch")
            else:
                raise ValueError("QNAP library inventory entry type is unsupported")

            path = _require_normalized_posix_relative_path(
                entry.get("path"), "QNAP library inventory entry path"
            )
            if path in entries_by_path:
                raise ValueError("duplicate QNAP library inventory entry path")
            paths.append(path)
            entries_by_path[path] = entry
            if entry_type == "symlink":
                resolved_targets[path] = _resolve_qnap_inventory_target(
                    path, entry.get("target")
                )

        if paths != sorted(paths):
            raise ValueError("QNAP library inventory entries must be path-sorted")
        for link_path, target_path in resolved_targets.items():
            if target_path not in entries_by_path:
                raise ValueError(
                    "QNAP library inventory symlink target is not inventoried"
                )
            seen_links: set[str] = set()
            cursor = link_path
            while entries_by_path[cursor].get("type") == "symlink":
                if cursor in seen_links:
                    raise ValueError("QNAP library inventory symlink cycle")
                seen_links.add(cursor)
                cursor = resolved_targets[cursor]
            if entries_by_path[cursor].get("type") != "file":
                raise ValueError(
                    "QNAP library inventory symlink must resolve to a regular file"
                )

        declared_total = tree.get("total_regular_bytes")
        if type(declared_total) is not int or declared_total != regular_bytes:
            raise ValueError("QNAP library inventory total_regular_bytes mismatch")
        expected_tree_sha = _sha256_bytes(_canonical_compact_json(entries))
        if _require_sha256(
            tree.get("tree_sha256"), "QNAP library inventory tree SHA"
        ) != expected_tree_sha:
            raise ValueError("QNAP library inventory tree SHA mismatch")
        tree_entries[container_root] = (entries_by_path, resolved_targets)
        combined_regular_bytes += regular_bytes

    if seen_roots != ALLOWED_QNAP_LIBRARY_MOUNTS:
        raise ValueError("QNAP library inventory does not cover the exact allowed roots")
    if combined_regular_bytes > QNAP_LIBRARY_SNAPSHOT_MAX_BYTES:
        raise ValueError("QNAP library inventory exceeds snapshot_max_bytes")

    required = value.get("required_mapped_libraries")
    if type(required) is not list or not required:
        raise ValueError(
            "QNAP library inventory required_mapped_libraries must be nonempty"
        )
    required_keys: list[tuple[str, str]] = []
    for item in required:
        if type(item) is not dict or set(item) != {"container_root", "path"}:
            raise ValueError("QNAP required mapped library schema mismatch")
        container_root = item.get("container_root")
        if type(container_root) is not str or container_root not in tree_entries:
            raise ValueError("QNAP required mapped library root is not inventoried")
        path = _require_normalized_posix_relative_path(
            item.get("path"), "QNAP required mapped library path"
        )
        key = (container_root, path)
        if key in required_keys:
            raise ValueError("duplicate QNAP required mapped library")
        required_keys.append(key)
        entries_by_path, resolved_targets = tree_entries[container_root]
        if path not in entries_by_path:
            raise ValueError("QNAP required mapped library is not inventoried")
        seen_links: set[str] = set()
        cursor = path
        while entries_by_path[cursor].get("type") == "symlink":
            if cursor in seen_links:
                raise ValueError("QNAP required mapped library symlink cycle")
            seen_links.add(cursor)
            cursor = resolved_targets[cursor]
        if entries_by_path[cursor].get("type") != "file":
            raise ValueError(
                "QNAP required mapped library must resolve to a regular file"
            )
    if required_keys != sorted(required_keys):
        raise ValueError("QNAP required mapped libraries must be path-sorted")
    if not any(
        root == "/qnap/nvidia/lib" and posixpath.basename(path) == "libcuda.so.1"
        for root, path in required_keys
    ):
        raise ValueError(
            "QNAP required mapped libraries must include nvidia libcuda.so.1"
        )
    required_roots = {root for root, _ in required_keys}
    expected_required_roots = {
        destination for _, destination in ALLOWED_QNAP_LIBRARY_MOUNTS
    }
    if required_roots != expected_required_roots:
        raise ValueError(
            "QNAP required mapped libraries must include at least one library from each tree"
        )


def _validate_host_contract(
    value: Mapping[str, object], image_id: str
) -> tuple[Path, bytes]:
    expected_fields = {
        "schema", "container_id", "container_name", "container_image_id",
        "network_mode", "restart_policy", "shm_size_bytes", "privileged",
        "device_requests", "devices", "mounts", "command",
        "raw_inspect_path", "raw_inspect_sha256", "qnap_library_inventory",
    }
    if set(value) != expected_fields:
        raise ValueError("host launch contract top-level schema mismatch")
    if value.get("schema") != "v4_candidate_training_host_launch.v1":
        raise ValueError("host launch contract schema mismatch")
    if value.get("container_image_id") != image_id:
        raise ValueError("host launch image ID mismatch")
    _validate_qnap_library_inventory(value.get("qnap_library_inventory"))
    container_id = value.get("container_id")
    if type(container_id) is not str or CONTAINER_ID_RE.fullmatch(container_id) is None:
        raise ValueError("host launch container_id must be a full lowercase ID")
    container_name = value.get("container_name")
    if (
        type(container_name) is not str
        or re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,62}", container_name) is None
    ):
        raise ValueError("host launch container_name is invalid")
    exact = {
        "network_mode": "none", "restart_policy": "no",
        "shm_size_bytes": 8589934592, "privileged": False,
        "device_requests": None,
    }
    for field, expected in exact.items():
        actual = value.get(field)
        if type(expected) is bool:
            _require_bool(actual, expected, f"host launch {field}")
        elif actual != expected or (type(expected) is int and type(actual) is not int):
            raise ValueError(f"host launch {field} mismatch")
    devices = value.get("devices")
    if type(devices) is not list or sorted(devices) != sorted(EXPECTED_NVIDIA_DEVICES):
        raise ValueError("host launch NVIDIA device contract mismatch")
    command = value.get("command")
    if type(command) is not list or not all(type(item) is str for item in command):
        raise ValueError("host launch command contract mismatch")
    mounts = value.get("mounts")
    if type(mounts) is not list or len(mounts) < 2:
        raise ValueError("host launch mount contract is incomplete")
    normalized_mounts: dict[str, tuple[str, bool]] = {}
    for mount in mounts:
        if type(mount) is not dict or set(mount) != {
            "source", "destination", "read_only"
        }:
            raise ValueError("host launch mount row schema mismatch")
        source = mount.get("source")
        destination = mount.get("destination")
        read_only = mount.get("read_only")
        if (
            type(source) is not str or not source
            or type(destination) is not str or not destination
            or type(read_only) is not bool or destination in normalized_mounts
        ):
            raise ValueError("host launch mount row value mismatch")
        normalized_mounts[destination] = (source, read_only)
    read_write_mounts = [
        (source, destination)
        for destination, (source, read_only) in normalized_mounts.items()
        if not read_only
    ]
    expected_run_source = f"/share/Container/runs/{container_name}-workspace"
    if len(read_write_mounts) != 1 or read_write_mounts[0][0] != expected_run_source:
        raise ValueError(
            "host launch must mount the dedicated per-container run workspace read-write"
        )
    global_mounts = [
        (source, destination)
        for destination, (source, read_only) in normalized_mounts.items()
        if source == "/share/Container" and read_only
    ]
    if len(global_mounts) != 1:
        raise ValueError("host launch requires exact global read-only and run read-write mounts")
    core_destinations = {global_mounts[0][1], read_write_mounts[0][1]}
    for destination, (source, read_only) in normalized_mounts.items():
        if destination in core_destinations:
            continue
        if not read_only or (source, destination) not in ALLOWED_QNAP_LIBRARY_MOUNTS:
            raise ValueError("host launch contains an unapproved extra mount")
    qnap_mounts = {
        (source, destination)
        for destination, (source, read_only) in normalized_mounts.items()
        if (source, destination) in ALLOWED_QNAP_LIBRARY_MOUNTS and read_only
    }
    if qnap_mounts != ALLOWED_QNAP_LIBRARY_MOUNTS:
        raise ValueError("host launch requires both exact read-only QNAP library mounts")

    inspect_path_text = value.get("raw_inspect_path")
    if type(inspect_path_text) is not str or not Path(inspect_path_text).is_absolute():
        raise ValueError("host launch raw_inspect_path must be absolute")
    inspect_path = _regular_file(Path(inspect_path_text), "raw docker inspect evidence")
    inspect_content = _stable_bytes(inspect_path, "raw docker inspect evidence")
    if _require_sha256(
        value.get("raw_inspect_sha256"), "host launch raw_inspect_sha256"
    ) != _sha256_bytes(inspect_content):
        raise ValueError("raw docker inspect evidence SHA mismatch")
    try:
        inspected = json.loads(
            inspect_content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid raw docker inspect evidence: {error}") from error
    if type(inspected) is not list or len(inspected) != 1 or type(inspected[0]) is not dict:
        raise ValueError("raw docker inspect evidence must contain one object")
    item = inspected[0]
    if item.get("Id") != container_id or item.get("Image") != image_id:
        raise ValueError("raw docker inspect identity mismatch")
    if item.get("Name") != f"/{container_name}":
        raise ValueError("raw docker inspect container name mismatch")
    config = item.get("Config")
    host = item.get("HostConfig")
    raw_mounts = item.get("Mounts")
    if type(config) is not dict or config.get("Cmd") != command:
        raise ValueError("raw docker inspect command mismatch")
    if type(host) is not dict:
        raise ValueError("raw docker inspect HostConfig is missing")
    if host.get("NetworkMode") != "none" or host.get("ShmSize") != 8589934592:
        raise ValueError("raw docker inspect network/shm mismatch")
    _require_bool(host.get("Privileged"), False, "raw docker inspect Privileged")
    if host.get("DeviceRequests") is not None:
        raise ValueError("raw docker inspect DeviceRequests mismatch")
    restart = host.get("RestartPolicy")
    if (
        type(restart) is not dict
        or restart.get("Name") != "no"
        or restart.get("MaximumRetryCount") != 0
    ):
        raise ValueError("raw docker inspect restart policy mismatch")
    if config.get("Hostname") != container_id[:12]:
        raise ValueError("raw docker inspect hostname is not the default container ID")
    if config.get("Entrypoint") not in (None, []):
        raise ValueError("raw docker inspect Entrypoint must be empty")
    if config.get("User") != "" or config.get("WorkingDir") != "":
        raise ValueError("raw docker inspect User/WorkingDir must use frozen defaults")
    raw_environment = config.get("Env")
    if type(raw_environment) is not list:
        raise ValueError("raw docker inspect Config.Env must be a list")
    environment: dict[str, str] = {}
    for entry in raw_environment:
        if type(entry) is not str or "=" not in entry:
            raise ValueError("raw docker inspect Config.Env entry is malformed")
        name, content = entry.split("=", 1)
        if not name or name in environment:
            raise ValueError("raw docker inspect Config.Env contains an empty/duplicate name")
        environment[name] = content
    forbidden_environment = sorted(
        name
        for name in environment
        if name in FORBIDDEN_CONTAINER_ENV
        or name.startswith(BASH_EXPORTED_FUNCTION_PREFIX)
    )
    if forbidden_environment:
        raise ValueError(
            f"raw docker inspect Config.Env contains forbidden injection variables: "
            f"{forbidden_environment}"
        )
    if "V4_CLEAN_REEXEC" in environment:
        raise ValueError("raw docker inspect may not inject the internal clean-env marker")
    missing_environment = sorted(REQUIRED_CONTAINER_ENV.difference(environment))
    if missing_environment:
        raise ValueError(
            f"raw docker inspect Config.Env lacks required launcher variables: "
            f"{missing_environment}"
        )
    if environment["CONTAINER_IMAGE_ID"] != image_id:
        raise ValueError("raw docker inspect CONTAINER_IMAGE_ID mismatch")
    expected_wrapper = (
        Path(environment["CODE_ROOT"]) / "scripts" / "nas" /
        "run_v4_candidate_training.sh"
    ).as_posix()
    expected_command = [
        "/usr/bin/env", "-i", f"PATH={CLEAN_CONTAINER_PATH}",
        "V4_CLEAN_REEXEC=1",
        *[
            f"{name}={environment[name]}"
            for name in sorted(REQUIRED_CONTAINER_ENV)
        ],
        "/bin/sh", expected_wrapper,
    ]
    if command != expected_command:
        raise ValueError("host launch command does not clean and reconstruct the environment")
    for field in ("CapAdd", "CapDrop", "SecurityOpt"):
        if host.get(field) not in (None, []):
            raise ValueError(f"raw docker inspect HostConfig.{field} must be empty")
    for field in ("PidMode", "UTSMode", "UsernsMode"):
        if host.get(field) != "":
            raise ValueError(f"raw docker inspect HostConfig.{field} must be empty")
    if host.get("IpcMode") != "private":
        raise ValueError("raw docker inspect HostConfig.IpcMode must be private")
    raw_devices = host.get("Devices")
    if type(raw_devices) is not list:
        raise ValueError("raw docker inspect Devices is missing")
    observed_devices: list[str] = []
    for device in raw_devices:
        if type(device) is not dict:
            raise ValueError("raw docker inspect device row mismatch")
        host_path = device.get("PathOnHost")
        container_path = device.get("PathInContainer")
        permissions = device.get("CgroupPermissions")
        if host_path != container_path or permissions != "rwm" or type(host_path) is not str:
            raise ValueError("raw docker inspect NVIDIA device mapping mismatch")
        observed_devices.append(host_path)
    if sorted(observed_devices) != sorted(EXPECTED_NVIDIA_DEVICES):
        raise ValueError("raw docker inspect NVIDIA device set mismatch")
    if type(raw_mounts) is not list or len(raw_mounts) != len(normalized_mounts):
        raise ValueError("raw docker inspect mount count mismatch")
    observed_mounts: dict[str, tuple[str, bool]] = {}
    for mount in raw_mounts:
        if type(mount) is not dict:
            raise ValueError("raw docker inspect mount row mismatch")
        source = mount.get("Source")
        destination = mount.get("Destination")
        writable = mount.get("RW")
        if (
            type(source) is not str or type(destination) is not str
            or type(writable) is not bool or destination in observed_mounts
        ):
            raise ValueError("raw docker inspect mount value mismatch")
        if mount.get("Type") != "bind" or mount.get("Propagation") != "rprivate":
            raise ValueError("raw docker inspect mount type/propagation mismatch")
        expected_mode = "rw" if writable else "ro"
        if mount.get("Mode") != expected_mode:
            raise ValueError("raw docker inspect mount mode mismatch")
        observed_mounts[destination] = (source, not writable)
    if observed_mounts != normalized_mounts:
        raise ValueError("raw docker inspect mounts differ from host contract")
    return inspect_path, inspect_content


def _audit_trusted_policy_trust_root(path: Path, actual_sha256: str) -> dict[str, object]:
    """Require the policy path and digest that were pinned by code review."""

    relative = TRUSTED_POLICY_RELATIVE_PATH
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("trusted policy code path must be a safe relative path")
    repository = REPO_ROOT.resolve(strict=False)
    expected_path = (repository / relative).resolve(strict=False)
    if expected_path == repository or not expected_path.is_relative_to(repository):
        raise ValueError("trusted policy code path must stay inside the repository")
    if path.resolve(strict=False) != expected_path:
        raise ValueError(
            "trusted policy path differs from the repository-pinned trust root"
        )
    approved = APPROVED_TRUSTED_POLICY_SHA256.strip().casefold()
    if approved == UNCONFIGURED_TRUST_ROOT.casefold():
        raise ValueError("trusted policy trust root is UNCONFIGURED")
    _require_sha256(approved, "approved trusted policy code pin")
    if actual_sha256 != approved:
        raise ValueError(
            "trusted policy SHA-256 differs from the repository-pinned trust root"
        )
    return {
        "method": TRUST_ROOT_METHOD,
        "repository_relative_policy_path": relative.as_posix(),
        "approved_policy_sha256": approved,
        "actual_policy_sha256": actual_sha256,
        "verified": True,
    }


def _validate_trusted_policy(
    value: Mapping[str, object], *, source_manifest_sha256: list[str],
    full_data_report_sha256: list[str], bindings: Mapping[str, str],
    operational_sources: Mapping[str, object],
    license_origins: Mapping[str, object],
    candidate_counts: Mapping[str, object],
) -> None:
    if set(bindings) != POLICY_BINDING_FIELDS:
        raise ValueError("trusted policy producer binding schema mismatch")
    expected_fields = {
        "schema", "artifact_role", "status", "approved", "operational_cutoff_kst",
        "source_manifest_sha256", "full_data_validator_report_sha256",
        "operational_sources", "license_origins", "candidate_counts",
        *POLICY_BINDING_FIELDS,
    }
    if set(value) != expected_fields:
        raise ValueError("trusted policy top-level fields mismatch")
    if value.get("schema") != "v4_candidate_training_trusted_policy.v1":
        raise ValueError("trusted policy schema mismatch")
    if value.get("artifact_role") != "approved_v4_candidate_training_policy":
        raise ValueError("trusted policy artifact_role mismatch")
    if value.get("status") != "approved":
        raise ValueError("trusted policy status mismatch")
    _require_bool(value.get("approved"), True, "trusted policy approved")
    if value.get("operational_cutoff_kst") != OPERATIONAL_CUTOFF.isoformat():
        raise ValueError("trusted policy operational cutoff mismatch")
    if value.get("source_manifest_sha256") != source_manifest_sha256:
        raise ValueError("trusted policy source manifest binding mismatch")
    if value.get("full_data_validator_report_sha256") != full_data_report_sha256:
        raise ValueError("trusted policy validator report binding mismatch")
    for field, digest in bindings.items():
        if value.get(field) != digest:
            raise ValueError(f"trusted policy {field} binding mismatch")
    if type(value.get("operational_sources")) is not dict:
        raise ValueError("trusted policy operational_sources must be an object")
    if value.get("operational_sources") != operational_sources:
        raise ValueError("trusted policy operational source evidence mismatch")
    if value.get("license_origins") != license_origins:
        raise ValueError("trusted policy license origin contract mismatch")
    if value.get("candidate_counts") != candidate_counts:
        raise ValueError("trusted policy candidate count contract mismatch")


def _parse_timestamp(value: object, location: str) -> datetime:
    if type(value) is not str or not value:
        raise ValueError(f"{location} operational captured_at is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{location} captured_at is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{location} captured_at timezone is required")
    return parsed


def _artifact(path: Path, content: bytes) -> dict[str, str]:
    return {"path": path.as_posix(), "sha256": _sha256_bytes(content)}


def _verify_dataset_content_inventory(
    entries: Sequence[Mapping[str, object]], description: str,
    *, path_aliases: Mapping[str, Path] | None = None,
) -> str:
    expected_fields = {
        "sample_id", "role", "source_path", "source_size", "source_sha256",
        "crop_path", "crop_size", "crop_sha256",
    }
    previous_key: tuple[str, str] | None = None
    for index, entry in enumerate(entries):
        if set(entry) != expected_fields:
            raise ValueError(f"{description} row {index} schema mismatch")
        sample_id = entry.get("sample_id")
        role = entry.get("role")
        if type(sample_id) is not str or not sample_id or role not in ROLE_SPLITS:
            raise ValueError(f"{description} row {index} identity mismatch")
        key = (str(role), sample_id)
        if previous_key is not None and key <= previous_key:
            raise ValueError(f"{description} rows must be strictly role/sample sorted")
        previous_key = key
        for prefix, path_field, size_field, sha_field in (
            ("source", "source_path", "source_size", "source_sha256"),
            ("crop", "crop_path", "crop_size", "crop_sha256"),
        ):
            path_text = entry.get(path_field)
            size = entry.get(size_field)
            digest = _require_sha256(
                entry.get(sha_field), f"{description} row {index} {prefix} SHA"
            )
            if type(path_text) is not str or not Path(path_text).is_absolute():
                raise ValueError(f"{description} row {index} {prefix} path is not absolute")
            declared_path = Path(path_text)
            if declared_path.as_posix() != path_text:
                raise ValueError(f"{description} row {index} {prefix} path is not normalized")
            verification_path = (
                path_aliases.get(path_text, declared_path)
                if path_aliases is not None
                else declared_path
            )
            path = _regular_file(
                verification_path, f"{description} row {index} {prefix}"
            )
            content = _stable_bytes(path, f"{description} row {index} {prefix}")
            if type(size) is not int or size != len(content):
                raise RuntimeError(f"{description} row {index} {prefix} size changed")
            if _sha256_bytes(content) != digest:
                raise RuntimeError(f"{description} row {index} {prefix} bytes changed")
    return _sha256_bytes(_canonical_json(list(entries)))


def _publish_preaudit_proposal(
    *, final: Path, source_manifests: Sequence[Path], manifest_shas: Sequence[str],
    full_data_validator_reports: Sequence[Path], full_report_shas: Sequence[str],
    license_allowlist: Path, quality_exclusions: Path, protected_sources: Path,
    training_config: Path, quality_assembly_bundle: _QualityAssemblyBundle,
    selection_bindings: Mapping[str, str],
    train_content: bytes, validation_content: bytes,
    dataset_snapshot_value: Mapping[str, object], dataset_snapshot_content: bytes,
    dataset_content_inventory: Sequence[Mapping[str, object]],
    dataset_content_inventory_sha: str,
    dataset_source_inputs: Sequence[Mapping[str, object]],
    dataset_crop_inputs: Sequence[Mapping[str, object]],
    candidate_counts: Mapping[str, object],
) -> dict[str, object]:
    """Publish audit inputs atomically without granting training authority."""

    train_path = (final / "train_manifest.csv").absolute()
    validation_path = (final / "model_validation_manifest.csv").absolute()
    proposal_path = (final / "preaudit_proposal.json").absolute()
    snapshot_report_path = (final / "candidate_dataset_snapshot.json").absolute()
    snapshot_root = (final / "dataset_snapshot").absolute()
    stage = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    marker_rows: tuple[tuple[Path, str], ...]
    snapshot_publish_receipt: dict[str, object]
    proposal: dict[str, object]
    try:
        crop_inputs_by_sha = {
            str(record["sha256"]): record for record in dataset_crop_inputs
        }
        if len(crop_inputs_by_sha) != len(dataset_crop_inputs):
            raise ValueError("duplicate dataset crop input SHA")
        snapshot_stage_root = stage / "dataset_snapshot"
        snapshot_stage_root.mkdir()
        objects = dataset_snapshot_value.get("objects")
        if type(objects) is not list:
            raise RuntimeError("dataset snapshot report objects schema changed")
        for index, raw in enumerate(objects):
            if type(raw) is not dict:
                raise RuntimeError("dataset snapshot report object schema changed")
            digest = str(raw["sha256"])
            record = crop_inputs_by_sha.get(digest)
            if record is None:
                raise RuntimeError("dataset snapshot plan lacks its crop input")
            _copy_dataset_snapshot_object(
                record, stage / str(raw["path"]),
                f"preaudit dataset snapshot object {index}",
            )
        if os.name != "nt":
            for directory in sorted(
                (path for path in snapshot_stage_root.rglob("*") if path.is_dir()),
                key=lambda value: len(value.parts), reverse=True,
            ):
                os.chmod(directory, 0o555, follow_symlinks=False)
            os.chmod(snapshot_stage_root, 0o555, follow_symlinks=False)
        snapshot_publish_receipt = _dataset_snapshot_tree_contract(
            snapshot_stage_root, dataset_snapshot_value, logical_root=snapshot_root
        )
        snapshot_aliases = {
            (final / str(raw["path"])).absolute().as_posix(): stage / str(raw["path"])
            for raw in objects
        }
        if _verify_dataset_content_inventory(
            dataset_content_inventory,
            "preaudit dataset content inventory construction",
            path_aliases=snapshot_aliases,
        ) != dataset_content_inventory_sha:
            raise RuntimeError("preaudit dataset content inventory changed")

        artifacts = {
            "manifests": [
                {"role": "train", **_artifact(train_path, train_content)},
                {
                    "role": "model_validation",
                    **_artifact(validation_path, validation_content),
                },
            ],
            "dataset_snapshot_report": _artifact(
                snapshot_report_path, dataset_snapshot_content
            ),
        }
        proposal = {
            "schema": PREAUDIT_PROPOSAL_SCHEMA,
            "artifact_role": PREAUDIT_PROPOSAL_ROLE,
            "status": PREAUDIT_PROPOSAL_STATUS,
            "candidate_only": True,
            "proposal_only": True,
            "near_duplicate_audit_input_ready": True,
            "near_duplicate_audit_completed": False,
            "candidate_training_input_authorized": False,
            "training_authority": False,
            "lineage_execution_authorized": False,
            "ready_for_lineage_upgrade": False,
            "blind_test_authority": False,
            "candidate_promotion_authorized": False,
            "production_deployment_authorized": False,
            "pi_deployment_authorized": False,
            "spring_contract_modified": False,
            "local_only": True,
            "portable": False,
            "selection_contract": {
                "operational_cutoff_kst": OPERATIONAL_CUTOFF.isoformat(),
                "quality_contract": QUALITY_CONTRACT,
                "quality_reasons": sorted(QUALITY_REASONS),
                "dent_or_crush_is_condition_target_not_quality_exclusion": True,
            },
            "artifacts": artifacts,
            "dataset_content_inventory": list(dataset_content_inventory),
            "dataset_snapshot_publish_receipt": snapshot_publish_receipt,
            "counts": dict(candidate_counts),
            "bindings": {
                "source_manifest_sha256": list(manifest_shas),
                "full_data_validator_report_sha256": list(full_report_shas),
                "dataset_content_inventory_sha256": dataset_content_inventory_sha,
                **dict(selection_bindings),
            },
        }
        proposal_content = _canonical_json(proposal)
        marker_rows = (
            (proposal_path, _sha256_bytes(proposal_content)),
            (train_path, _sha256_bytes(train_content)),
            (validation_path, _sha256_bytes(validation_content)),
            (snapshot_report_path, _sha256_bytes(dataset_snapshot_content)),
        )
        marker_content = "".join(
            f"{digest}  {path.as_posix()}\n" for path, digest in marker_rows
        ).encode("utf-8")
        for name, content in (
            ("train_manifest.csv", train_content),
            ("model_validation_manifest.csv", validation_content),
            ("candidate_dataset_snapshot.json", dataset_snapshot_content),
            ("preaudit_proposal.json", proposal_content),
        ):
            with (stage / name).open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        staged_paths = {
            proposal_path: stage / "preaudit_proposal.json",
            train_path: stage / "train_manifest.csv",
            validation_path: stage / "model_validation_manifest.csv",
            snapshot_report_path: stage / "candidate_dataset_snapshot.json",
        }
        for path, expected in marker_rows:
            if _sha256_bytes(
                _stable_bytes(staged_paths[path], "preaudit artifact pre-publish")
            ) != expected:
                raise RuntimeError(f"preaudit artifact changed before publish: {path}")
        for path, expected in zip(source_manifests, manifest_shas, strict=True):
            if _sha256_bytes(_stable_bytes(path, "source manifest final rehash")) != expected:
                raise RuntimeError("source manifest changed before preaudit publish")
        for path, expected in zip(
            full_data_validator_reports, full_report_shas, strict=True
        ):
            if _sha256_bytes(
                _stable_bytes(path, "validator report final rehash")
            ) != expected:
                raise RuntimeError("validator report changed before preaudit publish")
        for path, binding, description in (
            (license_allowlist, "license_allowlist_sha256", "license allowlist"),
            (quality_exclusions, "quality_exclusions_sha256", "quality exclusions"),
            (protected_sources, "protected_sources_sha256", "protected sources"),
            (training_config, "training_config_sha256", "training config"),
        ):
            if _sha256_bytes(_stable_bytes(path, f"{description} final rehash")) != (
                selection_bindings[binding]
            ):
                raise RuntimeError(f"{description} changed before preaudit publish")
        _rehash_operational_quality_assembly(quality_assembly_bundle)
        for index, record in enumerate(dataset_source_inputs):
            _verify_dataset_input(record, f"preaudit source payload {index}")
        for index, record in enumerate(dataset_crop_inputs):
            _verify_dataset_input(record, f"preaudit crop payload {index}")
        if _dataset_snapshot_tree_contract(
            snapshot_stage_root, dataset_snapshot_value, logical_root=snapshot_root
        ) != snapshot_publish_receipt:
            raise RuntimeError("preaudit dataset snapshot changed before publish")
        _publish_directory_no_replace(stage, final)
    except Exception:
        for path in sorted(
            stage.rglob("*"), key=lambda value: len(value.parts), reverse=True
        ):
            try:
                os.chmod(path, 0o700 if path.is_dir() else 0o600)
            except OSError:
                pass
        shutil.rmtree(stage, ignore_errors=True)
        raise

    final_paths = {
        proposal_path: final / "preaudit_proposal.json",
        train_path: final / "train_manifest.csv",
        validation_path: final / "model_validation_manifest.csv",
        snapshot_report_path: final / "candidate_dataset_snapshot.json",
    }
    for path, expected in marker_rows:
        if _sha256_bytes(
            _stable_bytes(final_paths[path], "preaudit artifact post-publish")
        ) != expected:
            raise RuntimeError(f"preaudit artifact changed after publish: {path}")
    if _dataset_snapshot_tree_contract(
        final / "dataset_snapshot", dataset_snapshot_value, logical_root=snapshot_root
    ) != snapshot_publish_receipt:
        raise RuntimeError("preaudit dataset snapshot changed after publish")
    if _verify_dataset_content_inventory(
        dataset_content_inventory, "preaudit dataset content inventory post-publish"
    ) != dataset_content_inventory_sha:
        raise RuntimeError("preaudit dataset content inventory changed after publish")
    for index, record in enumerate(dataset_source_inputs):
        _verify_dataset_input(record, f"preaudit source payload post-publish {index}")
    for index, record in enumerate(dataset_crop_inputs):
        _verify_dataset_input(record, f"preaudit crop payload post-publish {index}")
    _rehash_operational_quality_assembly(quality_assembly_bundle)
    with (final / "preaudit_proposal.sha256").open("xb") as handle:
        handle.write(marker_content)
        handle.flush()
        os.fsync(handle.fileno())
    if _stable_bytes(
        final / "preaudit_proposal.sha256", "preaudit completion seal post-write"
    ) != marker_content:
        raise RuntimeError("preaudit completion seal changed after publication")
    return proposal


def _validate_preaudit_proposal(
    proposal_dir: Path, *, source_manifest_sha256: Sequence[str],
    full_data_report_sha256: Sequence[str], selection_bindings: Mapping[str, str],
    train_content: bytes, validation_content: bytes,
    dataset_snapshot_value: Mapping[str, object], dataset_snapshot_content: bytes,
    candidate_counts: Mapping[str, object],
) -> None:
    """Require a sealed proposal and regenerated byte-identical audit inputs."""

    root = _reject_symlink_components(proposal_dir, "preaudit proposal directory")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("preaudit proposal directory must be a non-symlink directory")
    expected_names = {
        "preaudit_proposal.json", "preaudit_proposal.sha256",
        "train_manifest.csv", "model_validation_manifest.csv",
        "candidate_dataset_snapshot.json", "dataset_snapshot",
    }
    if {path.name for path in root.iterdir()} != expected_names:
        raise ValueError("preaudit proposal top-level file set mismatch")
    proposal, proposal_content = _load_json(
        root / "preaudit_proposal.json", "preaudit proposal"
    )
    expected_fields = {
        "schema", "artifact_role", "status", "candidate_only", "proposal_only",
        "near_duplicate_audit_input_ready", "near_duplicate_audit_completed",
        "candidate_training_input_authorized", "training_authority",
        "lineage_execution_authorized", "ready_for_lineage_upgrade",
        "blind_test_authority", "candidate_promotion_authorized",
        "production_deployment_authorized", "pi_deployment_authorized",
        "spring_contract_modified", "local_only", "portable",
        "selection_contract", "artifacts", "dataset_content_inventory",
        "dataset_snapshot_publish_receipt", "counts", "bindings",
    }
    if set(proposal) != expected_fields or proposal_content != _canonical_json(proposal):
        raise ValueError("preaudit proposal schema or canonical bytes mismatch")
    if (
        proposal.get("schema") != PREAUDIT_PROPOSAL_SCHEMA
        or proposal.get("artifact_role") != PREAUDIT_PROPOSAL_ROLE
        or proposal.get("status") != PREAUDIT_PROPOSAL_STATUS
    ):
        raise ValueError("preaudit proposal identity mismatch")
    for field in (
        "candidate_only", "proposal_only", "near_duplicate_audit_input_ready",
        "local_only",
    ):
        _require_bool(proposal.get(field), True, f"preaudit proposal {field}")
    for field in (
        "near_duplicate_audit_completed", "candidate_training_input_authorized",
        "training_authority", "lineage_execution_authorized",
        "ready_for_lineage_upgrade", "blind_test_authority",
        "candidate_promotion_authorized", "production_deployment_authorized",
        "pi_deployment_authorized", "spring_contract_modified", "portable",
    ):
        _require_bool(proposal.get(field), False, f"preaudit proposal {field}")
    if proposal.get("selection_contract") != {
        "operational_cutoff_kst": OPERATIONAL_CUTOFF.isoformat(),
        "quality_contract": QUALITY_CONTRACT,
        "quality_reasons": sorted(QUALITY_REASONS),
        "dent_or_crush_is_condition_target_not_quality_exclusion": True,
    }:
        raise ValueError("preaudit proposal selection contract mismatch")
    bindings = proposal.get("bindings")
    if type(bindings) is not dict:
        raise ValueError("preaudit proposal bindings are invalid")
    proposal_inventory_sha = _require_sha256(
        bindings.get("dataset_content_inventory_sha256"),
        "preaudit proposal dataset content inventory SHA",
    )
    expected_bindings = {
        "source_manifest_sha256": list(source_manifest_sha256),
        "full_data_validator_report_sha256": list(full_data_report_sha256),
        "dataset_content_inventory_sha256": proposal_inventory_sha,
        **dict(selection_bindings),
    }
    if bindings != expected_bindings:
        raise ValueError("preaudit proposal input binding mismatch")
    if proposal.get("counts") != dict(candidate_counts):
        raise ValueError("preaudit proposal selection counts mismatch")

    train_path = root / "train_manifest.csv"
    validation_path = root / "model_validation_manifest.csv"
    snapshot_report_path = root / "candidate_dataset_snapshot.json"
    if _stable_bytes(train_path, "preaudit train manifest") != train_content:
        raise ValueError("preaudit train manifest differs from regenerated bytes")
    if _stable_bytes(
        validation_path, "preaudit validation manifest"
    ) != validation_content:
        raise ValueError(
            "preaudit model-validation manifest differs from regenerated bytes"
        )
    if _stable_bytes(
        snapshot_report_path, "preaudit snapshot report"
    ) != dataset_snapshot_content:
        raise ValueError(
            "preaudit dataset snapshot report differs from regenerated bytes"
        )
    expected_artifacts = {
        "manifests": [
            {"role": "train", **_artifact(train_path.absolute(), train_content)},
            {
                "role": "model_validation",
                **_artifact(validation_path.absolute(), validation_content),
            },
        ],
        "dataset_snapshot_report": _artifact(
            snapshot_report_path.absolute(), dataset_snapshot_content
        ),
    }
    if proposal.get("artifacts") != expected_artifacts:
        raise ValueError("preaudit proposal artifact binding mismatch")
    receipt = _dataset_snapshot_tree_contract(
        root / "dataset_snapshot", dataset_snapshot_value,
        logical_root=(root / "dataset_snapshot").absolute(),
    )
    if proposal.get("dataset_snapshot_publish_receipt") != receipt:
        raise ValueError("preaudit proposal snapshot receipt mismatch")
    inventory = proposal.get("dataset_content_inventory")
    if type(inventory) is not list:
        raise ValueError("preaudit proposal dataset content inventory is invalid")
    inventory_sha = _verify_dataset_content_inventory(
        inventory, "preaudit proposal dataset content inventory"
    )
    if inventory_sha != proposal_inventory_sha:
        raise ValueError("preaudit proposal dataset content inventory binding mismatch")

    marker_content = _stable_bytes(
        root / "preaudit_proposal.sha256", "preaudit proposal completion seal"
    )
    expected_marker = "".join(
        f"{_sha256_bytes(content)}  {path.absolute().as_posix()}\n"
        for path, content in (
            (root / "preaudit_proposal.json", proposal_content),
            (train_path, train_content),
            (validation_path, validation_content),
            (snapshot_report_path, dataset_snapshot_content),
        )
    ).encode("utf-8")
    if marker_content != expected_marker:
        raise ValueError("preaudit proposal completion seal mismatch")


def _build_candidate_bundle(
    *, source_manifests: Sequence[Path],
    full_data_validator_reports: Sequence[Path], license_allowlist: Path,
    quality_exclusions: Path, quality_exclusion_assembly_receipt: Path,
    protected_sources: Path, training_config: Path,
    output_dir: Path, proposal_only: bool,
    qx3_diagnostic_ready: Path | None = None,
    qx3_diagnostic_report: Path | None = None,
    trusted_policy: Path | None = None,
    near_duplicate_audit_report: Path | None = None,
    protected_reference_inventory: Path | None = None,
    code_inventory: Path | None = None,
    pretrained_backbone: Path | None = None,
    host_launch_contract: Path | None = None,
    container_image_id: str | None = None,
    preaudit_proposal_dir: Path | None = None,
) -> dict[str, object]:
    """Build either the immutable pre-audit proposal or final authority."""

    if not source_manifests or len(source_manifests) != len(full_data_validator_reports):
        raise ValueError("each source manifest requires one full-data validator report")
    if not proposal_only and (
        type(container_image_id) is not str
        or IMAGE_ID_RE.fullmatch(container_image_id) is None
    ):
        raise ValueError("container_image_id must be a full sha256 image ID")
    final = _reject_symlink_components(output_dir, "output directory")
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to reuse immutable output directory: {final}")
    parent = _reject_symlink_components(final.parent, "output parent")
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("output parent must be an existing non-symlink directory")

    manifest_groups: list[list[dict[str, str]]] = []
    manifest_contents: list[bytes] = []
    manifest_shas: list[str] = []
    for path in source_manifests:
        rows, content = _load_manifest(path)
        manifest_groups.append(rows)
        manifest_contents.append(content)
        manifest_shas.append(_sha256_bytes(content))
    if len(set(manifest_shas)) != len(manifest_shas):
        raise ValueError("duplicate source manifest bytes")

    full_report_shas: list[str] = []
    for report_path, manifest_sha, rows in zip(
        full_data_validator_reports, manifest_shas, manifest_groups, strict=True
    ):
        report, content = _load_json(report_path, "full-data validator report")
        _validate_full_data_report(report, manifest_sha, rows)
        full_report_shas.append(_sha256_bytes(content))

    license_value, license_content = _load_json(license_allowlist, "license allowlist")
    quality_value, quality_content = _load_json(quality_exclusions, "quality exclusions")
    protected_value, protected_content = _load_json(protected_sources, "protected sources")
    config_value, config_content = _load_json(training_config, "training config")

    origins = _validate_license_allowlist(license_value)
    excluded_sources = _validate_quality_manifest(quality_value)
    quality_assembly_bundle = _validate_operational_quality_assembly(
        receipt_path=quality_exclusion_assembly_receipt,
        quality_path=quality_exclusions,
        quality_value=quality_value,
        quality_content=quality_content,
        output_dir=final,
    )
    protected = _validate_protected_sources(protected_value)
    _validate_training_config(config_value)

    # These inputs grant candidate-training authority and therefore must not be
    # needed to materialize the earlier, non-authoritative audit proposal.
    qx3_ready_content = b""
    qx3_report_content = b""
    near_duplicate_value: dict[str, object] = {}
    near_duplicate_content = b""
    protected_inventory_value: dict[str, object] = {}
    protected_inventory_content = b""
    protected_asset_count = 0
    inventory_value: dict[str, object] = {}
    inventory_content = b""
    host_content = b""
    policy_value: dict[str, object] = {}
    policy_content = b""
    trust_root_evidence: dict[str, object] = {}
    backbone_content = b""
    trainer_sha256 = ""
    auditor_sha256 = ""
    auditor_runtime_code_sha256 = ""
    auditor_path = Path()
    raw_inspect_path = Path()
    raw_inspect_content = b""
    backbone_path = Path()
    if not proposal_only:
        required_final_paths = {
            "qx3_diagnostic_ready": qx3_diagnostic_ready,
            "qx3_diagnostic_report": qx3_diagnostic_report,
            "trusted_policy": trusted_policy,
            "near_duplicate_audit_report": near_duplicate_audit_report,
            "protected_reference_inventory": protected_reference_inventory,
            "code_inventory": code_inventory,
            "pretrained_backbone": pretrained_backbone,
            "host_launch_contract": host_launch_contract,
        }
        missing = sorted(
            name for name, path in required_final_paths.items()
            if not isinstance(path, Path)
        )
        if missing:
            raise ValueError(f"final authority inputs are missing: {missing}")
        assert qx3_diagnostic_ready is not None
        assert qx3_diagnostic_report is not None
        assert trusted_policy is not None
        assert near_duplicate_audit_report is not None
        assert protected_reference_inventory is not None
        assert code_inventory is not None
        assert pretrained_backbone is not None
        assert host_launch_contract is not None
        assert type(container_image_id) is str

        qx3_ready, qx3_ready_content = _load_json(
            qx3_diagnostic_ready, "qx3 diagnostic ready"
        )
        qx3_report, qx3_report_content = _load_json(
            qx3_diagnostic_report, "qx3 diagnostic report"
        )
        quality_exclusions_sha256 = _sha256_bytes(quality_content)
        quality_exclusion_assembly_receipt_sha256 = _sha256_bytes(
            quality_assembly_bundle.receipt_content
        )
        _validate_qx3_ready(
            qx3_ready,
            quality_exclusions_sha256=quality_exclusions_sha256,
            quality_exclusion_assembly_receipt_sha256=(
                quality_exclusion_assembly_receipt_sha256
            ),
        )
        _validate_qx3_report(
            qx3_report,
            quality_exclusions_sha256=quality_exclusions_sha256,
            quality_exclusion_assembly_receipt_sha256=(
                quality_exclusion_assembly_receipt_sha256
            ),
        )
        qx3_bindings = qx3_ready.get("bindings")
        if type(qx3_bindings) is not dict or qx3_bindings.get(
            "comparison_sha256"
        ) != _sha256_bytes(qx3_report_content):
            raise ValueError(
                "qx3 ready does not bind the diagnostic comparison report"
            )
        quality_validator_path = Path(quality_assembly_contract.__file__).resolve()
        quality_validator_sha256 = qx3_bindings["quality_assembly_validator_sha256"]
        near_duplicate_value, near_duplicate_content = _load_json(
            near_duplicate_audit_report, "near-duplicate audit report"
        )
        protected_inventory_value, protected_inventory_content = _load_json(
            protected_reference_inventory, "protected reference inventory"
        )
        protected_asset_count = _validate_protected_reference_inventory(
            protected_inventory_value, protected
        )
        inventory_value, inventory_content = _load_json(
            code_inventory, "code inventory"
        )
        host_value, host_content = _load_json(
            host_launch_contract, "host launch contract"
        )
        policy_value, policy_content = _load_json(trusted_policy, "trusted policy")
        trust_root_evidence = _audit_trusted_policy_trust_root(
            trusted_policy, _sha256_bytes(policy_content)
        )
        backbone_content = _stable_bytes(pretrained_backbone, "pretrained backbone")

        _validate_code_inventory(inventory_value)
        inventory_files = inventory_value.get("files")
        assert type(inventory_files) is list
        trainer_inventory_rows = [
            row
            for row in inventory_files
            if type(row) is dict
            and row.get("path") == "scripts/train_multitask_verifier.py"
        ]
        if len(trainer_inventory_rows) != 1:
            raise ValueError("code inventory must contain exactly one candidate trainer")
        trainer_sha256 = _require_sha256(
            trainer_inventory_rows[0].get("sha256"), "code inventory trainer SHA"
        )
        auditor_inventory_rows = [
            row
            for row in inventory_files
            if type(row) is dict and row.get("path") == NEAR_DUPLICATE_AUDITOR_PATH
        ]
        if len(auditor_inventory_rows) != 1:
            raise ValueError(
                "code inventory must contain exactly one near-duplicate auditor"
            )
        auditor_sha256 = _require_sha256(
            auditor_inventory_rows[0].get("sha256"),
            "code inventory near-duplicate auditor SHA",
        )
        inventory_root_text = inventory_value.get("root")
        assert type(inventory_root_text) is str
        auditor_path = _regular_file(
            Path(inventory_root_text) / NEAR_DUPLICATE_AUDITOR_PATH,
            "near-duplicate auditor",
        )
        auditor_runtime_code_sha256 = _load_auditor_runtime_fingerprint(
            auditor_path, auditor_sha256
        )
        raw_inspect_path, raw_inspect_content = _validate_host_contract(
            host_value, container_image_id
        )
        backbone_path = _regular_file(pretrained_backbone, "pretrained backbone")
        if backbone_path.name != "mobilenet_v3_small-047dcff4.pth":
            raise ValueError("unexpected pretrained MobileNetV3 checkpoint filename")
        if (
            backbone_path.parent.name != "checkpoints"
            or backbone_path.parent.parent.name != "hub"
        ):
            raise ValueError(
                "pretrained backbone must be beneath TORCH_HOME/hub/checkpoints"
            )

    all_rows = [row for group in manifest_groups for row in group]
    all_source_shas = {
        _require_sha256(row.get("source_sha256"), "manifest source_sha256")
        for row in all_rows
    }
    if set(excluded_sources).difference(all_source_shas):
        raise ValueError("quality exclusion SHA is absent from the full-data manifests")

    operational_evidence: dict[str, object] = {}
    for row in all_rows:
        rule = origins.get(row.get("origin", ""))
        if type(rule) is dict and rule.get("kind") == "operational":
            source_sha = _require_sha256(row.get("source_sha256"), "operational source")
            timestamp = _parse_timestamp(row.get("captured_at"), source_sha)
            if timestamp.astimezone(KST) >= OPERATIONAL_CUTOFF:
                operational_evidence[source_sha] = {
                    "auditor_sha256": _require_sha256(
                        row.get("auditor_sha256"), "operational auditor_sha256"
                    ),
                    "teacher_output_sha256": _require_sha256(
                        row.get("teacher_output_sha256"),
                        "operational teacher_output_sha256",
                    ),
                    "localizer_output_sha256": _require_sha256(
                        row.get("localizer_output_sha256"),
                        "operational localizer_output_sha256",
                    ),
                }

    policy_bindings: dict[str, str] = {
        "license_allowlist_sha256": _sha256_bytes(license_content),
        "quality_exclusions_sha256": _sha256_bytes(quality_content),
        "quality_exclusion_assembly_receipt_sha256": _sha256_bytes(
            quality_assembly_bundle.receipt_content
        ),
        "protected_sources_sha256": _sha256_bytes(protected_content),
        "training_config_sha256": _sha256_bytes(config_content),
    }
    if not proposal_only:
        assert type(container_image_id) is str
        policy_bindings.update({
            "qx3_diagnostic_ready_sha256": _sha256_bytes(qx3_ready_content),
            "qx3_diagnostic_report_sha256": _sha256_bytes(qx3_report_content),
            "candidate_near_duplicate_audit_sha256": _sha256_bytes(
                near_duplicate_content
            ),
            "protected_reference_inventory_sha256": _sha256_bytes(
                protected_inventory_content
            ),
            "code_inventory_sha256": _sha256_bytes(inventory_content),
            "host_launch_contract_sha256": _sha256_bytes(host_content),
            "raw_container_inspect_sha256": _sha256_bytes(raw_inspect_content),
            "pretrained_backbone_sha256": _sha256_bytes(backbone_content),
            "container_image_id": container_image_id,
        })
    identities: dict[str, dict[str, str]] = {
        field: {} for field in (
            "source_sha256", "image_sha256", "object_group", "capture_session"
        )
    }
    sample_ids: set[str] = set()
    selected: dict[str, list[dict[str, str]]] = {role: [] for role in ROLE_SPLITS}
    dataset_content_inventory: list[dict[str, object]] = []
    dataset_snapshot_plan: list[dict[str, object]] = []
    dataset_source_inputs: list[dict[str, object]] = []
    dataset_crop_inputs: list[dict[str, object]] = []
    seen_source_paths: set[str] = set()
    seen_source_shas: set[str] = set()
    seen_crop_paths: set[str] = set()
    seen_crop_shas: set[str] = set()
    excluded_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    condition_counts = {head: Counter() for head in CONDITION_HEADS}
    material_counts_by_role = {role: Counter() for role in ROLE_SPLITS}
    objectness_counts_by_role = {role: Counter() for role in ROLE_SPLITS}
    origin_counts_by_role = {role: Counter() for role in ROLE_SPLITS}
    condition_counts_by_role = {
        role: {head: Counter() for head in CONDITION_HEADS}
        for role in ROLE_SPLITS
    }
    license_kind_counts_by_role = {role: Counter() for role in ROLE_SPLITS}
    dataset_counts_by_role = {role: Counter() for role in ROLE_SPLITS}

    for number, row in enumerate(all_rows, start=2):
        location = f"source row {number}"
        source_sha = _require_sha256(row.get("source_sha256"), f"{location} source_sha256")
        image_sha = _require_sha256(row.get("image_sha256"), f"{location} image_sha256")
        if source_sha in protected:
            raise ValueError(f"{location} uses a protected holdout source")
        source_path_text = row.get("source_filepath", "")
        image_path_text = row.get("filepath", "")
        if not Path(source_path_text).is_absolute() or not Path(image_path_text).is_absolute():
            raise ValueError(f"{location} source/crop paths must be absolute")
        source_path = _regular_file(Path(source_path_text), f"{location} source")
        image_path = _regular_file(Path(image_path_text), f"{location} crop")

        origin = row.get("origin", "")
        rule = origins.get(origin)
        if type(rule) is not dict:
            raise ValueError(f"{location} origin is not commercially allowlisted")
        if any(token in origin.casefold() for token in ("qx3", "diagnostic", "repro_pilot")):
            raise ValueError(f"{location} uses diagnostic qx3 evidence as a row source")
        forbidden_path_tokens = (
            "qx1", "qx2", "qx3", "diagnostic", "repro_pilot",
            "repro_validation", "v4_batch1_repro",
        )
        for path_text in (source_path_text, image_path_text):
            normalized_path = Path(path_text).as_posix().casefold()
            if any(token in normalized_path for token in forbidden_path_tokens):
                raise ValueError(
                    f"{location} points into a diagnostic qx artifact tree"
                )
        role = row.get("role", "")
        if role not in ROLE_SPLITS:
            raise ValueError(f"{location} role must be train or model_validation")
        if row.get("split") != ROLE_SPLITS[role] or row.get("fold") != role:
            raise ValueError(f"{location} role/fold/split mismatch")
        if source_sha in excluded_sources:
            excluded_counts[f"quality/{excluded_sources[source_sha]}"] += 1
            continue

        kind = rule.get("kind")
        if kind == "operational":
            captured = _parse_timestamp(row.get("captured_at"), location)
            if captured.astimezone(KST) < OPERATIONAL_CUTOFF:
                excluded_counts["operational/before_2026_08_01_kst"] += 1
                continue
            if row.get("role") != "train":
                raise ValueError("operational teacher/localizer evidence is train-only")
            if operational_evidence.get(source_sha) != {
                "auditor_sha256": row.get("auditor_sha256"),
                "teacher_output_sha256": row.get("teacher_output_sha256"),
                "localizer_output_sha256": row.get("localizer_output_sha256"),
            }:
                raise ValueError("operational trust evidence mismatch")
        elif kind != "aihub":
            raise ValueError(f"{location} license kind is unsupported")

        for field, value in (("source_sha256", source_sha), ("image_sha256", image_sha)):
            previous_role = identities[field].get(value)
            if previous_role is not None and previous_role != role:
                raise ValueError(f"leakage: {field} crosses train/model_validation")

        source_input = _read_dataset_input(
            source_path, f"{location} source", source_sha
        )
        crop_input = _read_dataset_input(
            image_path, f"{location} crop", image_sha
        )
        source_input_path = str(source_input["path"])
        crop_input_path = str(crop_input["path"])
        if source_input_path in seen_source_paths:
            raise ValueError("duplicate selected source payload path")
        if source_sha in seen_source_shas:
            raise ValueError("duplicate selected source SHA")
        if crop_input_path in seen_crop_paths:
            raise ValueError("duplicate selected crop payload path")
        if image_sha in seen_crop_shas:
            raise ValueError("duplicate selected crop SHA")
        seen_source_paths.add(source_input_path)
        seen_source_shas.add(source_sha)
        seen_crop_paths.add(crop_input_path)
        seen_crop_shas.add(image_sha)

        sample_id = row.get("sample_id", "")
        if not sample_id or sample_id in sample_ids:
            raise ValueError(f"{location} sample_id is empty or duplicated")
        sample_ids.add(sample_id)
        try:
            material = int(row.get("material", ""))
            source_count = int(row.get("source_object_count", ""))
            crop_count = int(row.get("crop_object_count", ""))
        except ValueError as error:
            raise ValueError(f"{location} numeric manifest contract is invalid") from error
        if material not in range(10):
            raise ValueError(f"{location} material must be 0..9")
        expected_category = "background" if material == 9 else MATERIAL_CLASSES[material]
        expected_crop_count = 0 if material == 9 else 1
        if row.get("category") != expected_category:
            raise ValueError(f"{location} material/category mismatch")
        if source_count not in {0, 1} or crop_count != expected_crop_count or crop_count > source_count:
            raise ValueError(f"{location} object-count contract mismatch")
        for head in CONDITION_HEADS:
            target = row.get(head, "")
            if target not in {"-1", "0", "1"}:
                raise ValueError(f"{location} invalid {head}; unknown targets must remain -1")
            condition_counts[head][target] += 1
            condition_counts_by_role[role][head][target] += 1
        material_counts_by_role[role][expected_category] += 1
        objectness_counts_by_role[role][
            "background" if material == 9 else "material"
        ] += 1
        origin_counts_by_role[role][origin] += 1
        license_kind_counts_by_role[role][str(rule["kind"])] += 1
        dataset_counts_by_role[role][str(rule["dataset_id"])] += 1
        for field, value in (
            ("source_sha256", source_sha), ("image_sha256", image_sha),
            ("object_group", row.get("object_group", "")),
            ("capture_session", row.get("capture_session", "")),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{location} has empty {field}")
            previous_role = identities[field].get(value)
            if previous_role is not None and previous_role != role:
                raise ValueError(f"leakage: {field} crosses train/model_validation")
            identities[field][value] = role
        selected_row = {field: row.get(field, "") for field in OUTPUT_FIELDS}
        selected_row["source_filepath"] = source_path.as_posix()
        snapshot_relative_path = _snapshot_relative_path(image_sha)
        selected_row["filepath"] = snapshot_relative_path
        selected[role].append(selected_row)
        snapshot_final_path = (final / snapshot_relative_path).absolute()
        dataset_content_inventory.append(
            {
                "sample_id": sample_id,
                "role": role,
                "source_path": source_path.as_posix(),
                "source_size": source_input["size"],
                "source_sha256": source_sha,
                "crop_path": snapshot_final_path.as_posix(),
                "crop_size": crop_input["size"],
                "crop_sha256": image_sha,
            }
        )
        dataset_snapshot_plan.append(
            {
                "sample_id": sample_id,
                "role": role,
                "path": snapshot_relative_path,
                "size": crop_input["size"],
                "sha256": image_sha,
            }
        )
        dataset_source_inputs.append(source_input)
        dataset_crop_inputs.append(crop_input)
        origin_counts[origin] += 1

    if any(not selected[role] for role in ROLE_SPLITS):
        raise ValueError("filtering left a required train/model_validation role empty")
    for role, rows in selected.items():
        observed_materials = {
            int(row["material"]) for row in rows if int(row["material"]) != 9
        }
        if observed_materials != set(range(9)):
            missing = sorted(set(range(9)).difference(observed_materials))
            raise ValueError(
                f"role {role!r} is missing positive material classes {missing}"
            )
        if not any(int(row["material"]) == 9 for row in rows):
            raise ValueError(f"role {role!r} is missing a background row")
        for head in CONDITION_HEADS:
            observed = {
                int(row[head])
                for row in rows
                if int(row["material"]) != 9 and row[head] in {"0", "1"}
            }
            if observed != {0, 1}:
                raise ValueError(
                    f"condition head {head!r} lacks 0/1 support in role {role!r}"
                )
    for rows in selected.values():
        rows.sort(key=lambda row: row["sample_id"])
    train_origin_counts = Counter(row["origin"] for row in selected["train"])
    configured_origin_weights = config_value["origin_weights"]
    assert type(configured_origin_weights) is dict
    missing_weighted_origins = sorted(
        set(configured_origin_weights).difference(train_origin_counts)
    )
    if missing_weighted_origins:
        raise ValueError(
            f"training origin weights reference absent selected origins: "
            f"{missing_weighted_origins}"
        )
    weighted_mass = {
        origin: count * float(configured_origin_weights.get(origin, 1.0))
        for origin, count in sorted(train_origin_counts.items())
    }
    total_weighted_mass = math.fsum(weighted_mass.values())
    derived_fractions = {
        origin: mass / total_weighted_mass
        for origin, mass in weighted_mass.items()
    }
    observed_row_weights = {
        float(configured_origin_weights.get(origin, 1.0))
        for origin in train_origin_counts
    }
    derived_sampling_mode = (
        "weighted_replacement"
        if len(observed_row_weights) > 1
        else "shuffle_without_replacement"
    )
    if config_value["sampling_mode"] != derived_sampling_mode:
        raise ValueError("training sampling_mode differs from selected rows")
    if config_value["sampling_samples_per_epoch"] != len(selected["train"]):
        raise ValueError("training sampling_samples_per_epoch differs from selected rows")
    configured_fractions = config_value["sampling_expected_fraction_by_origin"]
    assert type(configured_fractions) is dict
    if set(configured_fractions) != set(derived_fractions) or any(
        not math.isclose(
            float(configured_fractions[origin]), fraction,
            rel_tol=0.0, abs_tol=1e-12,
        )
        for origin, fraction in derived_fractions.items()
    ):
        raise ValueError("training sampling fractions differ from selected rows")
    dataset_content_inventory.sort(
        key=lambda row: (str(row["role"]), str(row["sample_id"]))
    )
    dataset_snapshot_plan.sort(
        key=lambda row: (str(row["role"]), str(row["sample_id"]))
    )
    dataset_snapshot_value, dataset_snapshot_content = _dataset_snapshot_report(
        dataset_snapshot_plan
    )
    dataset_content_inventory_sha = _sha256_bytes(
        _canonical_json(dataset_content_inventory)
    )
    train_content = _render_manifest(selected["train"])
    validation_content = _render_manifest(selected["model_validation"])
    policy_bindings.update({
        "candidate_train_manifest_sha256": _sha256_bytes(train_content),
        "candidate_model_validation_manifest_sha256": _sha256_bytes(
            validation_content
        ),
        "candidate_dataset_snapshot_sha256": _sha256_bytes(
            dataset_snapshot_content
        ),
    })
    dataset_consumption_contract: dict[str, object] = {}
    dataset_consumption_contract_sha = ""
    if not proposal_only:
        dataset_consumption_contract = _dataset_consumption_contract(
            trainer_sha256=trainer_sha256,
            snapshot_report_sha256=_sha256_bytes(dataset_snapshot_content),
            snapshot_tree_sha256=str(dataset_snapshot_value["tree_sha256"]),
            manifest_payload_set_sha256=str(
                dataset_snapshot_value["payload_set_sha256"]
            ),
        )
        dataset_consumption_contract_sha = _sha256_bytes(
            _canonical_json(dataset_consumption_contract)
        )
        policy_bindings["candidate_dataset_consumption_contract_sha256"] = (
            dataset_consumption_contract_sha
        )
        _validate_near_duplicate_audit(
            near_duplicate_value,
            near_duplicate_content,
            candidate_manifest_sha256={
                "train": policy_bindings["candidate_train_manifest_sha256"],
                "model_validation": policy_bindings[
                    "candidate_model_validation_manifest_sha256"
                ],
            },
            candidate_rows_by_role=selected,
            candidate_asset_count=2 * sum(len(rows) for rows in selected.values()),
            protected_sources_value=protected_value,
            protected_sources_content=protected_content,
            protected_sources=protected,
            protected_inventory_value=protected_inventory_value,
            protected_inventory_content=protected_inventory_content,
            protected_asset_count=protected_asset_count,
            auditor_sha256=auditor_sha256,
            auditor_runtime_code_sha256=auditor_runtime_code_sha256,
        )
    candidate_counts: dict[str, object] = {
        "selected_by_role": {role: len(selected[role]) for role in ROLE_SPLITS},
        "selected_by_origin": dict(sorted(origin_counts.items())),
        "material_by_role": {
            role: dict(sorted(material_counts_by_role[role].items()))
            for role in ROLE_SPLITS
        },
        "objectness_by_role": {
            role: dict(sorted(objectness_counts_by_role[role].items()))
            for role in ROLE_SPLITS
        },
        "origin_by_role": {
            role: dict(sorted(origin_counts_by_role[role].items()))
            for role in ROLE_SPLITS
        },
        "condition_targets_by_role": {
            role: {
                head: dict(sorted(condition_counts_by_role[role][head].items()))
                for head in CONDITION_HEADS
            }
            for role in ROLE_SPLITS
        },
        "license_kind_by_role": {
            role: dict(sorted(license_kind_counts_by_role[role].items()))
            for role in ROLE_SPLITS
        },
        "dataset_by_role": {
            role: dict(sorted(dataset_counts_by_role[role].items()))
            for role in ROLE_SPLITS
        },
        "excluded": dict(sorted(excluded_counts.items())),
        "condition_targets": {
            head: dict(sorted(counts.items()))
            for head, counts in condition_counts.items()
        },
    }
    proposal_selection_bindings = {
        field: policy_bindings[field]
        for field in (
            "license_allowlist_sha256", "quality_exclusions_sha256",
            "quality_exclusion_assembly_receipt_sha256",
            "protected_sources_sha256", "training_config_sha256",
            "candidate_train_manifest_sha256",
            "candidate_model_validation_manifest_sha256",
            "candidate_dataset_snapshot_sha256",
        )
    }
    if proposal_only:
        return _publish_preaudit_proposal(
            final=final,
            source_manifests=source_manifests,
            manifest_shas=manifest_shas,
            full_data_validator_reports=full_data_validator_reports,
            full_report_shas=full_report_shas,
            license_allowlist=license_allowlist,
            quality_exclusions=quality_exclusions,
            quality_assembly_bundle=quality_assembly_bundle,
            protected_sources=protected_sources,
            training_config=training_config,
            selection_bindings=proposal_selection_bindings,
            train_content=train_content,
            validation_content=validation_content,
            dataset_snapshot_value=dataset_snapshot_value,
            dataset_snapshot_content=dataset_snapshot_content,
            dataset_content_inventory=dataset_content_inventory,
            dataset_content_inventory_sha=dataset_content_inventory_sha,
            dataset_source_inputs=dataset_source_inputs,
            dataset_crop_inputs=dataset_crop_inputs,
            candidate_counts=candidate_counts,
        )
    if preaudit_proposal_dir is not None:
        _validate_preaudit_proposal(
            preaudit_proposal_dir,
            source_manifest_sha256=manifest_shas,
            full_data_report_sha256=full_report_shas,
            selection_bindings=proposal_selection_bindings,
            train_content=train_content,
            validation_content=validation_content,
            dataset_snapshot_value=dataset_snapshot_value,
            dataset_snapshot_content=dataset_snapshot_content,
            candidate_counts=candidate_counts,
        )
    # The repository-pinned policy approves the exact deterministic, filtered
    # candidate manifests.  A self-issued authority file cannot substitute
    # different rows or labels while retaining the same trust root.
    _validate_trusted_policy(
        policy_value, source_manifest_sha256=manifest_shas,
        full_data_report_sha256=full_report_shas, bindings=policy_bindings,
        operational_sources=operational_evidence, license_origins=origins,
        candidate_counts=candidate_counts,
    )
    train_path = (final / "train_manifest.csv").absolute()
    validation_path = (final / "model_validation_manifest.csv").absolute()
    authority_path = (final / "training_authority.json").absolute()
    snapshot_report_path = (final / "candidate_dataset_snapshot.json").absolute()
    snapshot_root = (final / "dataset_snapshot").absolute()
    assert code_inventory is not None
    assert host_launch_contract is not None
    inventory_path = _regular_file(code_inventory, "code inventory")
    config_path = _regular_file(training_config, "training config")
    host_path = _regular_file(host_launch_contract, "host launch contract")
    stage = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=parent))
    authority: dict[str, object]
    marker_rows: tuple[tuple[Path, str], ...]
    snapshot_publish_receipt: dict[str, object]
    try:
        crop_inputs_by_sha = {
            str(record["sha256"]): record for record in dataset_crop_inputs
        }
        if len(crop_inputs_by_sha) != len(dataset_crop_inputs):
            raise ValueError("duplicate dataset crop input SHA")
        snapshot_stage_root = stage / "dataset_snapshot"
        snapshot_stage_root.mkdir()
        objects = dataset_snapshot_value["objects"]
        if type(objects) is not list:
            raise RuntimeError("dataset snapshot report objects schema changed")
        for index, raw in enumerate(objects):
            if type(raw) is not dict:
                raise RuntimeError("dataset snapshot report object schema changed")
            digest = str(raw["sha256"])
            record = crop_inputs_by_sha.get(digest)
            if record is None:
                raise RuntimeError("dataset snapshot plan lacks its crop input")
            destination = stage / str(raw["path"])
            _copy_dataset_snapshot_object(
                record, destination, f"dataset snapshot object {index}"
            )
        if os.name != "nt":
            for directory in sorted(
                (path for path in snapshot_stage_root.rglob("*") if path.is_dir()),
                key=lambda value: len(value.parts), reverse=True,
            ):
                os.chmod(directory, 0o555, follow_symlinks=False)
            os.chmod(snapshot_stage_root, 0o555, follow_symlinks=False)
        snapshot_publish_receipt = _dataset_snapshot_tree_contract(
            snapshot_stage_root, dataset_snapshot_value, logical_root=snapshot_root
        )
        snapshot_publish_receipt_sha = _sha256_bytes(
            _canonical_json(snapshot_publish_receipt)
        )
        snapshot_aliases = {
            (final / str(raw["path"])).absolute().as_posix(): stage / str(raw["path"])
            for raw in objects
        }
        if _verify_dataset_content_inventory(
            dataset_content_inventory,
            "dataset content inventory construction",
            path_aliases=snapshot_aliases,
        ) != dataset_content_inventory_sha:
            raise RuntimeError("dataset content inventory construction changed")

        artifacts = {
            "manifests": [
                {"role": "train", **_artifact(train_path, train_content)},
                {"role": "model_validation", **_artifact(validation_path, validation_content)},
            ],
            "dataset_snapshot_report": _artifact(
                snapshot_report_path, dataset_snapshot_content
            ),
            "code_inventory": _artifact(inventory_path, inventory_content),
            "training_config": _artifact(config_path, config_content),
            "host_launch_contract": _artifact(host_path, host_content),
            "pretrained_backbone": _artifact(backbone_path, backbone_content),
        }
        authority = {
            "schema": AUTHORITY_SCHEMA, "artifact_role": AUTHORITY_ROLE,
            "status": AUTHORITY_STATUS, "candidate_only": True,
            "candidate_training_input_authorized": True, "training_authority": True,
            "lineage_execution_authorized": True, "ready_for_lineage_upgrade": True,
            "diagnostic_only": False, "production_runtime_modified": False,
            "blind_test_authority": False, "candidate_promotion_authorized": False,
            "production_deployment_authorized": False,
            "pi_deployment_authorized": False, "spring_contract_modified": False,
            "local_only": True, "portable": False,
            "operational_cutoff_kst": OPERATIONAL_CUTOFF.isoformat(),
            "material_classes": list(MATERIAL_CLASSES),
            "objectness_classes": ["background", "material"],
            "condition_heads": list(CONDITION_HEADS), "artifacts": artifacts,
            "trust_root": trust_root_evidence,
            "dataset_content_inventory": dataset_content_inventory,
            "dataset_snapshot_publish_receipt": snapshot_publish_receipt,
            "dataset_consumption_contract": dataset_consumption_contract,
            "near_duplicate_audit": near_duplicate_value,
            "counts": candidate_counts,
            "bindings": {
                "source_manifest_sha256": manifest_shas,
                "full_data_validator_report_sha256": full_report_shas,
                "trusted_policy_sha256": _sha256_bytes(policy_content),
                "dataset_content_inventory_sha256": dataset_content_inventory_sha,
                "dataset_snapshot_publish_receipt_sha256": (
                    snapshot_publish_receipt_sha
                ),
                "dataset_consumption_contract_sha256": (
                    dataset_consumption_contract_sha
                ),
                **policy_bindings,
            },
        }
        authority_content = _canonical_json(authority)
        marker_rows = (
            (authority_path, _sha256_bytes(authority_content)),
            (train_path, _sha256_bytes(train_content)),
            (validation_path, _sha256_bytes(validation_content)),
            (snapshot_report_path, _sha256_bytes(dataset_snapshot_content)),
            (inventory_path, _sha256_bytes(inventory_content)),
            (config_path, _sha256_bytes(config_content)),
            (host_path, _sha256_bytes(host_content)),
            (backbone_path, _sha256_bytes(backbone_content)),
        )
        marker_content = "".join(
            f"{digest}  {path.as_posix()}\n" for path, digest in marker_rows
        ).encode("utf-8")
        for name, content in (
            ("train_manifest.csv", train_content),
            ("model_validation_manifest.csv", validation_content),
            ("candidate_dataset_snapshot.json", dataset_snapshot_content),
            ("training_authority.json", authority_content),
        ):
            with (stage / name).open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        staged_paths = {
            authority_path: stage / "training_authority.json",
            train_path: stage / "train_manifest.csv",
            validation_path: stage / "model_validation_manifest.csv",
            snapshot_report_path: stage / "candidate_dataset_snapshot.json",
        }
        for path, expected in marker_rows:
            current = staged_paths.get(path, path)
            if _sha256_bytes(_stable_bytes(current, "pre-publish authority artifact")) != expected:
                raise RuntimeError(f"authority artifact changed before publish: {path}")
        for path, expected in zip(source_manifests, manifest_shas, strict=True):
            if _sha256_bytes(_stable_bytes(path, "source manifest final rehash")) != expected:
                raise RuntimeError("source manifest changed before publish")
        for path, expected in zip(full_data_validator_reports, full_report_shas, strict=True):
            if _sha256_bytes(_stable_bytes(path, "validator report final rehash")) != expected:
                raise RuntimeError("full-data validator report changed before publish")
        for path, expected, description in (
            (qx3_diagnostic_ready, policy_bindings["qx3_diagnostic_ready_sha256"], "qx3 ready"),
            (qx3_diagnostic_report, policy_bindings["qx3_diagnostic_report_sha256"], "qx3 report"),
            (quality_validator_path, quality_validator_sha256, "quality assembly validator"),
            (trusted_policy, _sha256_bytes(policy_content), "trusted policy"),
            (license_allowlist, policy_bindings["license_allowlist_sha256"], "license allowlist"),
            (quality_exclusions, policy_bindings["quality_exclusions_sha256"], "quality exclusions"),
            (protected_sources, policy_bindings["protected_sources_sha256"], "protected sources"),
            (
                near_duplicate_audit_report,
                policy_bindings["candidate_near_duplicate_audit_sha256"],
                "near-duplicate audit report",
            ),
            (
                protected_reference_inventory,
                policy_bindings["protected_reference_inventory_sha256"],
                "protected reference inventory",
            ),
            (auditor_path, auditor_sha256, "near-duplicate auditor"),
            (
                raw_inspect_path,
                policy_bindings["raw_container_inspect_sha256"],
                "raw docker inspect evidence",
            ),
        ):
            if _sha256_bytes(_stable_bytes(path, f"{description} final rehash")) != expected:
                raise RuntimeError(f"{description} changed before publish")
        if _load_auditor_runtime_fingerprint(
            auditor_path, auditor_sha256
        ) != auditor_runtime_code_sha256:
            raise RuntimeError(
                "near-duplicate auditor runtime fingerprint changed before publish"
            )
        for index, record in enumerate(dataset_source_inputs):
            _verify_dataset_input(record, f"source payload pre-publish {index}")
        for index, record in enumerate(dataset_crop_inputs):
            _verify_dataset_input(record, f"crop payload pre-publish {index}")
        if _dataset_snapshot_tree_contract(
            snapshot_stage_root, dataset_snapshot_value, logical_root=snapshot_root
        ) != snapshot_publish_receipt:
            raise RuntimeError("dataset snapshot changed before publish")
        if _verify_dataset_content_inventory(
            dataset_content_inventory,
            "dataset content inventory pre-publish",
            path_aliases=snapshot_aliases,
        ) != dataset_content_inventory_sha:
            raise RuntimeError("dataset content inventory changed before publish")
        _rehash_operational_quality_assembly(quality_assembly_bundle)
        _publish_directory_no_replace(stage, final)
    except Exception:
        for path in sorted(
            stage.rglob("*"), key=lambda value: len(value.parts), reverse=True
        ):
            try:
                os.chmod(path, 0o700 if path.is_dir() else 0o600)
            except OSError:
                pass
        shutil.rmtree(stage, ignore_errors=True)
        raise

    final_mapping = {
        authority_path: final / "training_authority.json",
        train_path: final / "train_manifest.csv",
        validation_path: final / "model_validation_manifest.csv",
        snapshot_report_path: final / "candidate_dataset_snapshot.json",
    }
    for path, expected in marker_rows:
        current = final_mapping.get(path, path)
        if _sha256_bytes(_stable_bytes(current, "post-publish authority artifact")) != expected:
            raise RuntimeError(f"authority artifact changed after publish: {path}")
    if _dataset_snapshot_tree_contract(
        final / "dataset_snapshot", dataset_snapshot_value, logical_root=snapshot_root
    ) != snapshot_publish_receipt:
        raise RuntimeError("dataset snapshot changed after publish")
    if _verify_dataset_content_inventory(
        dataset_content_inventory, "dataset content inventory post-publish"
    ) != dataset_content_inventory_sha:
        raise RuntimeError("dataset content inventory changed after publish")
    for index, record in enumerate(dataset_source_inputs):
        _verify_dataset_input(record, f"source payload post-publish {index}")
    for index, record in enumerate(dataset_crop_inputs):
        _verify_dataset_input(record, f"crop payload post-publish {index}")
    for path, expected, description in (
        (
            quality_validator_path,
            quality_validator_sha256,
            "quality assembly validator",
        ),
        (
            qx3_diagnostic_ready,
            policy_bindings["qx3_diagnostic_ready_sha256"],
            "qx3 ready",
        ),
        (
            qx3_diagnostic_report,
            policy_bindings["qx3_diagnostic_report_sha256"],
            "qx3 report",
        ),
    ):
        if _sha256_bytes(
            _stable_bytes(path, f"{description} post-publish rehash")
        ) != expected:
            raise RuntimeError(f"{description} changed after publish")
    _rehash_operational_quality_assembly(quality_assembly_bundle)
    # The marker is the atomic completion seal. Until every post-publication
    # check succeeds, the exposed directory is intentionally not consumable by
    # the fail-closed launcher.
    with (final / "training_authority.sha256").open("xb") as handle:
        handle.write(marker_content)
        handle.flush()
        os.fsync(handle.fileno())
    return authority


def build_preaudit_candidate_proposal(
    *, source_manifests: Sequence[Path],
    full_data_validator_reports: Sequence[Path], license_allowlist: Path,
    quality_exclusions: Path, quality_exclusion_assembly_receipt: Path,
    protected_sources: Path, training_config: Path, output_dir: Path,
) -> dict[str, object]:
    """Create immutable inputs for the near-duplicate audit, with no authority."""

    return _build_candidate_bundle(
        source_manifests=source_manifests,
        full_data_validator_reports=full_data_validator_reports,
        license_allowlist=license_allowlist,
        quality_exclusions=quality_exclusions,
        quality_exclusion_assembly_receipt=quality_exclusion_assembly_receipt,
        protected_sources=protected_sources,
        training_config=training_config,
        output_dir=output_dir,
        proposal_only=True,
    )


def build_training_authority(
    *, source_manifests: Sequence[Path],
    full_data_validator_reports: Sequence[Path], qx3_diagnostic_ready: Path,
    qx3_diagnostic_report: Path, trusted_policy: Path, license_allowlist: Path,
    quality_exclusions: Path, quality_exclusion_assembly_receipt: Path,
    protected_sources: Path,
    near_duplicate_audit_report: Path, protected_reference_inventory: Path,
    code_inventory: Path, pretrained_backbone: Path, training_config: Path,
    host_launch_contract: Path, container_image_id: str, output_dir: Path,
    preaudit_proposal_dir: Path | None = None,
) -> dict[str, object]:
    """Create a sealed candidate-training input directory."""

    return _build_candidate_bundle(
        source_manifests=source_manifests,
        full_data_validator_reports=full_data_validator_reports,
        license_allowlist=license_allowlist,
        quality_exclusions=quality_exclusions,
        quality_exclusion_assembly_receipt=quality_exclusion_assembly_receipt,
        protected_sources=protected_sources,
        training_config=training_config,
        output_dir=output_dir,
        proposal_only=False,
        qx3_diagnostic_ready=qx3_diagnostic_ready,
        qx3_diagnostic_report=qx3_diagnostic_report,
        trusted_policy=trusted_policy,
        near_duplicate_audit_report=near_duplicate_audit_report,
        protected_reference_inventory=protected_reference_inventory,
        code_inventory=code_inventory,
        pretrained_backbone=pretrained_backbone,
        host_launch_contract=host_launch_contract,
        container_image_id=container_image_id,
        preaudit_proposal_dir=preaudit_proposal_dir,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("final", "preaudit-proposal"), default="final"
    )
    parser.add_argument("--source-manifest", action="append", type=Path, required=True)
    parser.add_argument(
        "--full-data-validator-report", action="append", type=Path, required=True
    )
    for name in (
        "license_allowlist", "quality_exclusions",
        "quality_exclusion_assembly_receipt", "protected_sources",
        "training_config", "output_dir",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    for name in (
        "qx3_diagnostic_ready", "qx3_diagnostic_report", "trusted_policy",
        "near_duplicate_audit_report", "protected_reference_inventory",
        "code_inventory", "pretrained_backbone", "host_launch_contract",
        "preaudit_proposal_dir",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path)
    parser.add_argument("--container-image-id")
    return parser


def main() -> int:
    parser = _parser()
    args = vars(parser.parse_args())
    mode = args.pop("mode")
    args["source_manifests"] = args.pop("source_manifest")
    args["full_data_validator_reports"] = args.pop("full_data_validator_report")
    if mode == "preaudit-proposal":
        forbidden = {
            name: args.pop(name)
            for name in (
                "qx3_diagnostic_ready", "qx3_diagnostic_report", "trusted_policy",
                "near_duplicate_audit_report", "protected_reference_inventory",
                "code_inventory", "pretrained_backbone", "host_launch_contract",
                "container_image_id", "preaudit_proposal_dir",
            )
        }
        supplied = sorted(name for name, value in forbidden.items() if value is not None)
        if supplied:
            parser.error(
                "preaudit-proposal mode rejects authority-only inputs: "
                + ", ".join(supplied)
            )
        result = build_preaudit_candidate_proposal(**args)
    else:
        required = (
            "qx3_diagnostic_ready", "qx3_diagnostic_report", "trusted_policy",
            "near_duplicate_audit_report", "protected_reference_inventory",
            "code_inventory", "pretrained_backbone", "host_launch_contract",
            "container_image_id",
        )
        missing = [name for name in required if args.get(name) is None]
        if missing:
            parser.error(
                "final mode requires: "
                + ", ".join("--" + name.replace("_", "-") for name in missing)
            )
        result = build_training_authority(**args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
