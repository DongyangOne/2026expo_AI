"""Replay a candidate ONNX on the strict model-validation partition.

This is an immutable evidence builder, not a threshold tuner.  It validates
the same metadata/spec/output/preprocessing contracts used by the existing
multitask evidence path, re-reads and hashes every strict crop through the
training manifest parser, runs the actual ONNX, and publishes sample-level
logits plus independently calculated objectness/material confusion metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verifier_preprocessing_contract import validate_crop_preprocessing_spec


EVIDENCE_SCHEMA = "v4_candidate_validation_replay.v1"
MATERIAL_CLASS_NAMES = (
    "can", "pet", "paper", "plastic", "styrofoam", "vinyl", "glass",
    "battery", "fluorescent",
)
OBJECTNESS_CLASS_NAMES = ("background", "material")
VALIDATION_ROLE = "model_validation"
KNOWN_ROLES = {"train", VALIDATION_ROLE, "calibration", "blind_test"}
REQUIRED_FIELDS = {
    "filepath", "split", "source_id", "material", "category", "dent", "label",
    "foreign_material", "source_object_count", "sample_id", "source_sha256",
    "image_sha256", "object_group", "capture_session", "role", "fold", "origin",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, field: str) -> dict[str, Any]:
    return _load_json_bytes(path.read_bytes(), field=field, path=path)


def _load_json_bytes(
    raw: bytes, *, field: str, path: Path | None = None
) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        suffix = f": {path}" if path is not None else ""
        raise ValueError(f"{field} is not valid UTF-8 JSON{suffix}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _stage(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return temp


def _publish_pair_no_replace(*, output_jsonl: Path, jsonl_bytes: bytes, output_attestation: Path, attestation_bytes: bytes) -> None:
    if output_jsonl.resolve() == output_attestation.resolve():
        raise ValueError("output JSONL and attestation paths must differ")
    for path in (output_jsonl, output_attestation):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
    staged_jsonl = _stage(output_jsonl, jsonl_bytes)
    staged_attestation = _stage(output_attestation, attestation_bytes)
    published = False
    try:
        os.link(staged_jsonl, output_jsonl)
        published = True
        os.link(staged_attestation, output_attestation)
    except BaseException:
        if published:
            output_jsonl.unlink(missing_ok=True)
        raise
    finally:
        staged_jsonl.unlink(missing_ok=True)
        staged_attestation.unlink(missing_ok=True)


def _default_session_factory(path: Path) -> Any:
    return _default_session_factory_from_bytes(path.read_bytes())


def _default_session_factory_from_bytes(model_bytes: bytes) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("onnxruntime is required for candidate replay") from error
    return ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])


def _validate_spec(spec: Mapping[str, Any]):
    if spec.get("format_version") != 1 or spec.get("artifact_role") != "offline_candidate_spec_not_production_authorization":
        raise ValueError("inference spec is not the frozen offline-candidate contract")
    verifier = spec.get("verifier_contract")
    safety = spec.get("safety")
    if not isinstance(verifier, Mapping) or not isinstance(safety, Mapping):
        raise ValueError("inference spec is missing verifier/safety contract")
    if verifier.get("objectness") != "binary_material_vs_background" or verifier.get("material") != "positive_only_9_class_softmax":
        raise ValueError("inference spec verifier contract is unsupported")
    if verifier.get("blind_gate_threshold_overrides") is not False:
        raise ValueError("inference spec permits threshold overrides")
    if safety.get("production_model_replacement") is not False:
        raise ValueError("inference spec claims production replacement")
    if spec.get("detector_classes") != list(MATERIAL_CLASS_NAMES):
        raise ValueError("inference spec material class tuple is invalid")
    return validate_crop_preprocessing_spec(spec)


def _descriptor(contract: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    outputs = contract.get("outputs")
    matches = [item for item in outputs or [] if isinstance(item, Mapping) and item.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"metadata must define exactly one {name} output")
    return matches[0]


def _validate_metadata(metadata: Mapping[str, Any], *, onnx_path: Path, crop_contract: Any) -> tuple[int, tuple[str, ...]]:
    if metadata.get("format_version") != 3 or metadata.get("architecture") != "multitask_crop_verifier":
        raise ValueError("verifier metadata architecture/version is invalid")
    if metadata.get("candidate_only") is not True or metadata.get("production_runtime_modified") is not False:
        raise ValueError("verifier metadata candidate safety flags are invalid")
    if Path(str(metadata.get("onnx", ""))).name != onnx_path.name:
        raise ValueError("metadata.onnx does not name the supplied ONNX")
    if metadata.get("material_classes") != list(MATERIAL_CLASS_NAMES) or metadata.get("objectness_classes") != list(OBJECTNESS_CLASS_NAMES):
        raise ValueError("metadata class contract is invalid")
    model_config = metadata.get("model_config")
    if not isinstance(model_config, Mapping) or model_config.get("input_size") != crop_contract.size:
        raise ValueError("metadata input size differs from inference spec")
    preprocessing = metadata.get("preprocessing")
    if not isinstance(preprocessing, Mapping) or preprocessing.get("color_space") != "RGB" or preprocessing.get("resize") != [crop_contract.size, crop_contract.size]:
        raise ValueError("metadata preprocessing contract is invalid")
    normalization = preprocessing.get("normalization")
    if not isinstance(normalization, Mapping) or normalization.get("mean") != list(crop_contract.mean) or normalization.get("std") != list(crop_contract.std):
        raise ValueError("metadata normalization differs from inference spec")
    contract = metadata.get("output_contract")
    if not isinstance(contract, Mapping) or contract.get("version") != "multitask_verifier.v3" or contract.get("material_background_class_id") is not None:
        raise ValueError("metadata output contract is invalid")
    order = tuple(contract.get("output_order", ()))
    if order[:2] != ("objectness", "material") or len(order) != len(set(order)):
        raise ValueError("metadata output order is invalid")
    objectness = _descriptor(contract, "objectness")
    material = _descriptor(contract, "material")
    if objectness.get("shape") != ["batch", 2] or objectness.get("class_names") != list(OBJECTNESS_CLASS_NAMES):
        raise ValueError("metadata objectness output is invalid")
    if material.get("shape") != ["batch", 9] or material.get("class_names") != list(MATERIAL_CLASS_NAMES):
        raise ValueError("metadata material output is invalid")
    return crop_contract.size, order


def _validate_onnx_session(session: Any, *, input_size: int, metadata_output_names: Sequence[str]) -> str:
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or inputs[0].name != "img" or inputs[0].type != "tensor(float)":
        raise ValueError("ONNX input contract is invalid")
    shape = list(inputs[0].shape)
    if len(shape) != 4 or shape[1:] != [3, input_size, input_size]:
        raise ValueError("ONNX input shape is invalid")
    if tuple(output.name for output in outputs) != tuple(metadata_output_names):
        raise ValueError("ONNX output order differs from metadata")
    for output, classes in zip(outputs[:2], (2, 9), strict=True):
        shape = list(output.shape)
        if output.type != "tensor(float)" or len(shape) != 2 or shape[1] != classes:
            raise ValueError(f"ONNX {output.name} output contract is invalid")
    return inputs[0].name


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and value == value.lower() and all(
        character in "0123456789abcdef" for character in value
    )


def _read_strict_manifests(
    paths: Sequence[Path],
    *,
    validation_image_snapshots: Mapping[str, Path] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if not paths:
        raise ValueError("at least one strict CSV manifest is required")
    rows: list[dict[str, Any]] = []
    manifest_artifacts: list[dict[str, str]] = []
    for manifest in paths:
        manifest = manifest.resolve()
        raw_manifest = manifest.read_bytes()
        try:
            manifest_text = raw_manifest.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError(f"{manifest}: strict manifest must be UTF-8") from error
        manifest_artifacts.append(
            {"path": str(manifest), "sha256": _sha256_bytes(raw_manifest)}
        )
        with io.StringIO(manifest_text, newline="") as file:
            reader = csv.DictReader(file)
            missing = sorted(REQUIRED_FIELDS - set(reader.fieldnames or ()))
            if missing:
                raise ValueError(f"{manifest}: missing strict fields {missing}")
            for line, raw in enumerate(reader, start=2):
                location = f"{manifest}:{line}"
                row = {key: str(value or "").strip() for key, value in raw.items()}
                for field in ("sample_id", "source_sha256", "image_sha256", "object_group", "capture_session", "role", "fold", "origin"):
                    if not row.get(field):
                        raise ValueError(f"{location}: missing {field}")
                for field in ("source_sha256", "image_sha256"):
                    row[field] = row[field].lower()
                    if not _is_sha256(row[field]):
                        raise ValueError(f"{location}: invalid {field}")
                role = row["role"]
                if role not in KNOWN_ROLES:
                    raise ValueError(f"{location}: unsupported role {role}")
                expected_split = {"train": "training", VALIDATION_ROLE: "validation"}.get(role)
                if expected_split and row["split"] != expected_split:
                    raise ValueError(f"{location}: role/split mismatch")
                try:
                    material = int(row["material"])
                    source_count = int(row["source_object_count"])
                except ValueError as error:
                    raise ValueError(f"{location}: invalid integer field") from error
                if material not in range(10):
                    raise ValueError(f"{location}: material must be 0..9")
                expected_category = "background" if material == 9 else MATERIAL_CLASS_NAMES[material]
                if row["category"] != expected_category:
                    raise ValueError(f"{location}: category/material mismatch")
                if source_count not in {0, 1}:
                    raise ValueError(f"{location}: source_object_count must be 0 or 1")
                expected_crop_count = 0 if material == 9 else 1
                raw_crop = row.get("crop_object_count", "")
                if raw_crop:
                    try:
                        crop_count = int(raw_crop)
                    except ValueError as error:
                        raise ValueError(f"{location}: invalid crop_object_count") from error
                else:
                    crop_count = expected_crop_count
                    if source_count != expected_crop_count:
                        raise ValueError(f"{location}: hard negative requires crop_object_count")
                if crop_count != expected_crop_count or crop_count > source_count:
                    raise ValueError(f"{location}: invalid source/crop object count contract")
                image_path = Path(row["filepath"])
                if not image_path.is_absolute():
                    image_path = manifest.parent / image_path
                image_path = image_path.resolve()
                validation_snapshot: Path | None = None
                if validation_image_snapshots is not None and role == VALIDATION_ROLE:
                    validation_snapshot = validation_image_snapshots.get(
                        str(image_path)
                    )
                    if validation_snapshot is None:
                        raise ValueError(
                            f"{location}: validation image snapshot is missing"
                        )
                    if not validation_snapshot.is_file():
                        raise FileNotFoundError(
                            f"{location}: validation image snapshot does not exist"
                        )
                    if sha256_file(validation_snapshot) != row["image_sha256"]:
                        raise ValueError(
                            f"{location}: validation image snapshot SHA-256 mismatch"
                        )
                elif validation_image_snapshots is None:
                    if not image_path.is_file():
                        raise FileNotFoundError(
                            f"{location}: image does not exist: {image_path}"
                        )
                    if sha256_file(image_path) != row["image_sha256"]:
                        raise ValueError(
                            f"{location}: image_sha256 does not match image content"
                        )
                row.update(
                    {
                        "material": material,
                        "objectness": 0 if material == 9 else 1,
                        "_path": validation_snapshot or image_path,
                    }
                )
                rows.append(row)
    if not rows:
        raise ValueError("strict manifests contain no rows")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("strict manifests contain duplicate sample_id")
    for field in ("source_sha256", "image_sha256", "object_group", "capture_session"):
        roles_by_identity: dict[str, set[str]] = {}
        for row in rows:
            roles_by_identity.setdefault(str(row[field]).lower(), set()).add(row["role"])
        if any(len(roles) > 1 for roles in roles_by_identity.values()):
            raise ValueError(f"strict manifest leakage across roles: {field}")
    for role in ("train", VALIDATION_ROLE):
        role_rows = [row for row in rows if row["role"] == role]
        if {row["objectness"] for row in role_rows} != {0, 1}:
            raise ValueError(f"{role} lacks objectness class coverage")
        if {row["material"] for row in role_rows if row["objectness"] == 1} != set(range(9)):
            raise ValueError(f"{role} lacks nine-class material coverage")
    return rows, manifest_artifacts


def _validation_tensor(
    path: Path,
    *,
    expected_sha256: str,
    size: int,
    input_scale: float,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    raw_image = path.read_bytes()
    if _sha256_bytes(raw_image) != expected_sha256:
        raise ValueError(f"image changed before tensor conversion: {path}")
    with Image.open(io.BytesIO(raw_image)) as image:
        rgb = image.convert("RGB")
        if rgb.size != (size, size):
            rgb = rgb.resize((size, size), resample=Image.Resampling.BILINEAR)
        array = np.asarray(rgb, dtype=np.float32) / np.float32(input_scale)
    normalized = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32)


def _metrics(confusion: np.ndarray) -> dict[str, object]:
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion matrix must be square")
    support = confusion.sum(axis=1).astype(np.int64)
    if np.any(support <= 0):
        raise ValueError(f"every replay class needs support, got {support.tolist()}")
    recalls = [
        float(confusion[index, index] / support[index])
        for index in range(confusion.shape[0])
    ]
    count = int(confusion.sum())
    return {
        "count": count,
        "support": support.tolist(),
        "per_class_recall": recalls,
        "balanced_accuracy": float(sum(recalls) / len(recalls)),
        "accuracy": float(np.trace(confusion) / count),
        "confusion": confusion.astype(np.int64).tolist(),
    }


def _manifest_lineage(rows: Sequence[Any]) -> str:
    inventory = [
        {
            "sample_id": row["sample_id"],
            "source_sha256": row["source_sha256"],
            "image_sha256": row["image_sha256"],
            "object_group": row["object_group"],
            "capture_session": row["capture_session"],
            "role": row["role"],
            "fold": row["fold"],
            "material": str(row["material"]),
        }
        for row in sorted(rows, key=lambda item: item["sample_id"])
    ]
    return _sha256_bytes(
        json.dumps(
            inventory, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )


def _validate_logits(
    value: object, *, batch: int, classes: int, field: str
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (batch, classes):
        raise ValueError(
            f"{field} logits must have shape {(batch, classes)}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field} logits contain non-finite values")
    sorted_logits = np.sort(array, axis=1)
    if np.any(sorted_logits[:, -1] == sorted_logits[:, -2]):
        raise ValueError(f"{field} logits contain an unstable top-1 tie")
    return array


def replay_validation(
    *,
    manifest_paths: Sequence[Path],
    verifier_onnx: Path,
    verifier_metadata: Path,
    inference_spec: Path,
    output_jsonl: Path,
    output_attestation: Path,
    batch_size: int = 64,
    session_factory: Callable[[Path], Any] | None = None,
    validation_image_snapshots: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    """Run actual model-validation inference and publish immutable evidence."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    for path in (output_jsonl, output_attestation):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
    for description, path in (
        ("verifier ONNX", verifier_onnx),
        ("verifier metadata", verifier_metadata),
        ("inference spec", inference_spec),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} does not exist: {path}")

    model_snapshot = verifier_onnx.read_bytes()
    metadata_snapshot = verifier_metadata.read_bytes()
    spec_snapshot = inference_spec.read_bytes()
    model_sha256 = _sha256_bytes(model_snapshot)
    metadata_sha256 = _sha256_bytes(metadata_snapshot)
    spec_sha256 = _sha256_bytes(spec_snapshot)
    metadata = _load_json_bytes(
        metadata_snapshot, field="verifier metadata", path=verifier_metadata
    )
    spec = _load_json_bytes(spec_snapshot, field="inference spec", path=inference_spec)
    crop_contract = _validate_spec(spec)
    size, output_order = _validate_metadata(
        metadata, onnx_path=verifier_onnx, crop_contract=crop_contract
    )
    custom_session_factory_used = session_factory is not None
    session = (
        session_factory(verifier_onnx)
        if session_factory is not None
        else _default_session_factory_from_bytes(model_snapshot)
    )
    input_name = _validate_onnx_session(
        session, input_size=size, metadata_output_names=output_order
    )

    all_rows, manifest_artifacts = _read_strict_manifests(
        manifest_paths,
        validation_image_snapshots=validation_image_snapshots,
    )
    validation_rows = sorted(
        (row for row in all_rows if row["role"] == VALIDATION_ROLE),
        key=lambda row: row["sample_id"],
    )
    if not validation_rows:
        raise ValueError("strict manifests contain no model_validation rows")

    evidence_rows: list[dict[str, object]] = []
    objectness_confusion = np.zeros((2, 2), dtype=np.int64)
    material_confusion = np.zeros((9, 9), dtype=np.int64)
    objectness_predictions: Counter[int] = Counter()
    material_predictions: Counter[int] = Counter()
    for start in range(0, len(validation_rows), batch_size):
        stop = min(start + batch_size, len(validation_rows))
        input_array = np.stack(
            [
                _validation_tensor(
                    validation_rows[index]["_path"],
                    expected_sha256=validation_rows[index]["image_sha256"],
                    size=size,
                    input_scale=crop_contract.input_scale,
                    mean=crop_contract.mean,
                    std=crop_contract.std,
                )
                for index in range(start, stop)
            ]
        ).astype(np.float32, copy=False)
        values = session.run(
            ["objectness", "material"], {input_name: input_array}
        )
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError("ONNX replay must return objectness and material logits")
        objectness_logits = _validate_logits(
            values[0], batch=stop - start, classes=2, field="objectness"
        )
        material_logits = _validate_logits(
            values[1], batch=stop - start, classes=9, field="material"
        )
        for offset, row in enumerate(validation_rows[start:stop]):
            predicted_objectness = int(np.argmax(objectness_logits[offset]))
            predicted_material_head = int(np.argmax(material_logits[offset]))
            objectness_confusion[row["objectness"], predicted_objectness] += 1
            objectness_predictions[predicted_objectness] += 1
            if row["objectness"] == 1:
                material_confusion[row["material"], predicted_material_head] += 1
                material_predictions[predicted_material_head] += 1
            tensor_sha = _sha256_bytes(
                np.ascontiguousarray(input_array[offset], dtype="<f4").tobytes()
            )
            evidence_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "source_sha256": row["source_sha256"],
                    "image_sha256": row["image_sha256"],
                    "object_group": row["object_group"],
                    "capture_session": row["capture_session"],
                    "fold": row["fold"],
                    "role": VALIDATION_ROLE,
                    "truth_objectness": row["objectness"],
                    "truth_material": row["material"] if row["objectness"] == 1 else None,
                    "input_tensor_sha256": tensor_sha,
                    "objectness_logits": [float(value) for value in objectness_logits[offset]],
                    "material_logits": [float(value) for value in material_logits[offset]],
                    "predicted_objectness": predicted_objectness,
                    "predicted_material_head": predicted_material_head,
                    "cascaded_material": (
                        predicted_material_head if predicted_objectness == 1 else None
                    ),
                }
            )

    metrics = {
        "objectness": _metrics(objectness_confusion),
        "material": _metrics(material_confusion),
    }
    jsonl_bytes = _canonical_jsonl(evidence_rows)
    inventory = [
        {
            "sample_id": row["sample_id"],
            "source_sha256": row["source_sha256"],
            "image_sha256": row["image_sha256"],
        }
        for row in evidence_rows
    ]
    attestation: dict[str, object] = {
        "schema_version": 1,
        "evidence_schema": EVIDENCE_SCHEMA,
        "evaluation_role": VALIDATION_ROLE,
        "prediction_count": len(evidence_rows),
        "predictions_sha256": _sha256_bytes(jsonl_bytes),
        "prediction_inventory_sha256": _sha256_bytes(
            json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
        ),
        "model_sha256": model_sha256,
        "verifier_metadata_sha256": metadata_sha256,
        "inference_spec_sha256": spec_sha256,
        "manifest_artifacts": manifest_artifacts,
        "manifest_lineage_sha256": _manifest_lineage(all_rows),
        "objectness_classes": list(OBJECTNESS_CLASS_NAMES),
        "material_classes": list(MATERIAL_CLASS_NAMES),
        "preprocessing": {
            "input_size": size,
            "implementation": "replay_v4_candidate_metrics._validation_tensor",
            "contract": metadata["preprocessing"],
        },
        "metrics": metrics,
        "prediction_histogram": {
            "objectness": {
                str(index): objectness_predictions[index] for index in range(2)
            },
            "material_head_on_positive_truth": {
                str(index): material_predictions[index] for index in range(9)
            },
        },
        "thresholds_applied": False,
        "custom_session_factory_used": custom_session_factory_used,
        "artifact_snapshot_contract": "read_bytes_hash_before_use_and_hash_after.v1",
        "production_deployment_authorized": False,
        "generated_by": "scripts/replay_v4_candidate_metrics.py",
    }
    final_hashes = {
        "model": sha256_file(verifier_onnx),
        "metadata": sha256_file(verifier_metadata),
        "spec": sha256_file(inference_spec),
    }
    if final_hashes != {
        "model": model_sha256,
        "metadata": metadata_sha256,
        "spec": spec_sha256,
    }:
        raise RuntimeError("model, metadata, or inference spec changed during replay")
    for artifact in manifest_artifacts:
        if sha256_file(Path(artifact["path"])) != artifact["sha256"]:
            raise RuntimeError("strict manifest changed during replay")
    for row in all_rows:
        if (
            validation_image_snapshots is not None
            and row["role"] != VALIDATION_ROLE
        ):
            continue
        if sha256_file(row["_path"]) != row["image_sha256"]:
            raise RuntimeError(
                f"strict image changed during replay: {row['sample_id']}"
            )
    attestation["runtime_artifact_hashes"] = final_hashes
    attestation["runtime_artifact_hashes_match_snapshots"] = True
    attestation_bytes = _canonical_json(attestation)
    _publish_pair_no_replace(
        output_jsonl=output_jsonl,
        jsonl_bytes=jsonl_bytes,
        output_attestation=output_attestation,
        attestation_bytes=attestation_bytes,
    )
    return attestation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, action="append", type=Path)
    parser.add_argument("--verifier-onnx", required=True, type=Path)
    parser.add_argument("--verifier-metadata", required=True, type=Path)
    parser.add_argument("--inference-spec", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-attestation", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    attestation = replay_validation(
        manifest_paths=args.manifest,
        verifier_onnx=args.verifier_onnx,
        verifier_metadata=args.verifier_metadata,
        inference_spec=args.inference_spec,
        output_jsonl=args.output_jsonl,
        output_attestation=args.output_attestation,
        batch_size=args.batch_size,
    )
    print(json.dumps(attestation, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
