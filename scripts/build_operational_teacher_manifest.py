"""Accept conservative VLM pseudo-labels into strict operational manifests.

The output is development data only.  A positive teacher decision may be used
for ``train`` or ``calibration`` but is never promoted to blind-test ground
truth.  High-confidence empty-scene decisions are preserved separately as
train-only source inventory; they are not verifier crops until a later frozen
detector pass emits a real runtime proposal.  Every accepted row is revalidated
against the image bytes, the independent teacher passes, geometry, and capture
lineage before any artifact is written.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np

try:
    from scripts.operational_teacher_contract import (
        TEACHER_LABEL_BASE_FIELDS, TEACHER_LABEL_SCHEMA_VERSION,
        build_teacher_contract, load_known_audit, valid_sha256,
    )
    from scripts.build_independent_localization_consensus import (
        AGGREGATE_METHOD, AGGREGATE_TOLERANCE, CONTRACT_SCHEMA_VERSION,
        IOU_THRESHOLD, LOCALIZATION_SCHEMA_VERSION, bbox_iou,
        canonical_json as localization_canonical_json, provider_output_core,
        _provider_rows,
    )
except ModuleNotFoundError:  # direct ``python scripts/...py`` execution
    from operational_teacher_contract import (  # type: ignore[no-redef]
        TEACHER_LABEL_BASE_FIELDS, TEACHER_LABEL_SCHEMA_VERSION,
        build_teacher_contract, load_known_audit, valid_sha256,
    )
    from build_independent_localization_consensus import (  # type: ignore[no-redef]
        AGGREGATE_METHOD, AGGREGATE_TOLERANCE, CONTRACT_SCHEMA_VERSION,
        IOU_THRESHOLD, LOCALIZATION_SCHEMA_VERSION, bbox_iou,
        canonical_json as localization_canonical_json, provider_output_core,
        _provider_rows,
    )


CLASS_NAMES = (
    "can", "pet", "paper", "plastic", "styrofoam",
    "vinyl", "glass", "battery", "fluorescent",
)
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}
ACCEPTED_TEACHER_MATERIALS = frozenset((*CLASS_NAMES, "negative"))
ALLOWED_ROLES = frozenset({"train", "calibration"})
SHA256_LENGTH = 64
QUALITY_REASONS = frozenset(
    {
        "usable",
        "severe_frame_crop",
        "person_occlusion_or_dominance",
        "clutter_or_multiple_objects",
        "boundary_unreadable",
    }
)
KST = timezone(timedelta(hours=9))
OPERATIONAL_CAPTURE_CUTOFF_KST = datetime(2026, 8, 1, 0, 0, 0, tzinfo=KST)
OPERATIONAL_CAPTURE_CUTOFF_UTC = OPERATIONAL_CAPTURE_CUTOFF_KST.astimezone(
    timezone.utc
)
MINIMUM_IMAGE_WIDTH = 160
MINIMUM_IMAGE_HEIGHT = 120
EXTREME_EXPOSURE_FRACTION = 0.995
UNDEREXPOSED_LUMA_MAX = 5
OVEREXPOSED_LUMA_MIN = 250

MANIFEST_FIELDS = (
    "sample_id", "role", "split_role", "fold", "filepath", "split",
    "source_id", "source_sha256", "image_sha256", "content_identity",
    "object_group", "capture_session", "origin", "selection_reason",
    "material", "category", "teacher_material", "teacher_training_usable",
    "teacher_quality_reason", "dent", "label",
    "foreign_material", "source_object_count", "source_path_b64",
    "source_bbox_x", "source_bbox_y", "source_bbox_w", "source_bbox_h",
    "source_width", "source_height", "bbox_x1", "bbox_y1", "bbox_x2",
    "bbox_y2", "bbox_area_ratio", "bbox_source", "capture_timestamp",
    "lineage_key_source", "teacher_model", "teacher_minimum_confidence",
    "teacher_consensus_votes", "teacher_pass_count", "pseudo_label",
    "ground_truth_authority", "blind_test_eligible",
)
EMPTY_SCENE_INVENTORY_FIELDS = (*MANIFEST_FIELDS, "training_crop_ready")
ARTIFACT_NAMES = {
    "csv": "operational_teacher_manifest.csv",
    "jsonl": "operational_teacher_manifest.jsonl",
    "empty_scene_csv": "operational_empty_scene_inventory.csv",
    "empty_scene_jsonl": "operational_empty_scene_inventory.jsonl",
    "rejections": "operational_teacher_rejections.json",
    "lineage": "operational_teacher_lineage.json",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha(value: object) -> str | None:
    normalized = str(value or "").strip().lower()
    if len(normalized) != SHA256_LENGTH:
        return None
    try:
        int(normalized, 16)
    except ValueError:
        return None
    return normalized


def _canonical_json(value: object, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\n"


def _load_jsonl(path: Path, *, kind: str) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"{kind} input does not exist: {path}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{number}: {error.msg}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{kind} row must be an object at {path}:{number}")
        row = dict(row)
        row["_input_line"] = number
        rows.append(row)
    return rows


def _load_inventory(path: Path | None) -> list[dict]:
    if path is None:
        return []
    if not path.is_file():
        raise FileNotFoundError(f"capture inventory does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(value, dict):
        for field in ("captures", "inventory", "rows", "items"):
            if isinstance(value.get(field), list):
                value = value[field]
                break
        else:
            if all(isinstance(item, dict) for item in value.values()):
                value = [dict(item, sha256=item.get("sha256", key)) for key, item in value.items()]
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("capture inventory must be a JSON array or SHA-keyed object")
    return [dict(row) for row in value]


def _nested(row: Mapping[str, object], *paths: str) -> object | None:
    for path in paths:
        current: object = row
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                break
            current = current[part]
        else:
            if current not in (None, "", [], {}):
                return current
    return None


def _timestamp(row: Mapping[str, object]) -> datetime | None:
    value = _nested(
        row, "timestamp", "captured_at", "created_at", "metadata.timestamp"
    )
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _private_identity(prefix: str, value: object) -> str:
    normalized = str(value).strip().casefold().encode("utf-8")
    return f"{prefix}_{_sha256_bytes(normalized)[:20]}"


def _cluster_by_gap(
    members: Sequence[tuple[str, datetime | None]], *, gap_seconds: float, prefix: str
) -> dict[str, str]:
    """Create deterministic IDs without including role, split, or raw client IDs."""
    ordered = sorted(members, key=lambda item: (item[1] is None, item[1] or datetime.max.replace(tzinfo=timezone.utc), item[0]))
    clusters: list[list[str]] = []
    current: list[str] = []
    previous_time: datetime | None = None
    for sha, captured_at in ordered:
        if (
            current
            and (
                captured_at is None
                or previous_time is None
                or (captured_at - previous_time).total_seconds() > gap_seconds
            )
        ):
            clusters.append(current)
            current = []
        current.append(sha)
        previous_time = captured_at
    if current:
        clusters.append(current)
    result = {}
    for cluster in clusters:
        identity = _sha256_bytes("\n".join(sorted(cluster)).encode("ascii"))[:24]
        for sha in cluster:
            result[sha] = f"{prefix}_{identity}"
    return result


def _inventory_by_sha(rows: Sequence[Mapping[str, object]]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        sha = _valid_sha(_nested(row, "sha256", "image.sha256", "metadata.image.sha256"))
        if sha:
            result[sha].append(dict(row))
    return result


def _lineage_rows(
    queue_by_sha: Mapping[str, dict],
    inventory_by_sha: Mapping[str, Sequence[dict]],
    *,
    burst_gap_seconds: float,
) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, object]] = {}
    actor_members: dict[str, list[tuple[str, datetime | None]]] = defaultdict(list)
    session_members: dict[str, list[tuple[str, datetime | None]]] = defaultdict(list)
    stream_members: dict[str, list[tuple[str, datetime | None]]] = defaultdict(list)

    for sha, queue_row in queue_by_sha.items():
        inventory = list(inventory_by_sha.get(sha, ()))
        candidates = [*inventory, queue_row]
        explicit_group = next(
            (
                _nested(row, "object_group", "physical_object_id", "group_id", "metadata.object_group")
                for row in candidates
                if _nested(row, "object_group", "physical_object_id", "group_id", "metadata.object_group")
            ),
            None,
        )
        explicit_session = next(
            (
                _nested(row, "capture_session", "session_id", "capture_batch", "metadata.capture_session")
                for row in candidates
                if _nested(row, "capture_session", "session_id", "capture_batch", "metadata.capture_session")
            ),
            None,
        )
        client_id = next(
            (
                _nested(row, "client_id", "request.client_id", "metadata.request.client_id")
                for row in candidates
                if _nested(row, "client_id", "request.client_id", "metadata.request.client_id")
            ),
            None,
        )
        device_id = next(
            (
                _nested(
                    row,
                    "device_id",
                    "camera_id",
                    "hardware_id",
                    "request.device_id",
                    "metadata.request.device_id",
                )
                for row in candidates
                if _nested(
                    row,
                    "device_id",
                    "camera_id",
                    "hardware_id",
                    "request.device_id",
                    "metadata.request.device_id",
                )
            ),
            None,
        )
        captured_at = next((_timestamp(row) for row in candidates if _timestamp(row)), None)
        metadata[sha] = {
            "explicit_group": explicit_group,
            "explicit_session": explicit_session,
            "client_id": client_id,
            "device_id": device_id,
            "timestamp": captured_at,
        }
        if client_id is not None:
            actor_members[_private_identity("actor", client_id)].append((sha, captured_at))
        elif explicit_session is not None:
            session_members[_private_identity("sessionkey", explicit_session)].append((sha, captured_at))
        elif captured_at is not None:
            # When the privacy-safe queue has redacted client_id, adjacent
            # frames from the same camera stream are still conservatively kept
            # together.  Over-grouping costs some data efficiency but prevents
            # optimistic validation leakage from a burst of the same object.
            stream_key = (
                _private_identity("device", device_id)
                if device_id is not None
                else "device_unknown"
            )
            stream_members[stream_key].append((sha, captured_at))

    object_clusters: dict[str, str] = {}
    session_clusters: dict[str, str] = {}
    for key, members in sorted(actor_members.items()):
        object_clusters.update(
            _cluster_by_gap(members, gap_seconds=burst_gap_seconds, prefix=f"object_{key}")
        )
        session_clusters.update(
            _cluster_by_gap(
                members,
                gap_seconds=max(burst_gap_seconds * 5.0, burst_gap_seconds),
                prefix=f"capture_{key}",
            )
        )
    for key, members in sorted(session_members.items()):
        object_clusters.update(
            _cluster_by_gap(members, gap_seconds=burst_gap_seconds, prefix=f"object_{key}")
        )
        session_clusters.update(
            _cluster_by_gap(
                members,
                gap_seconds=max(burst_gap_seconds * 5.0, burst_gap_seconds),
                prefix=f"capture_{key}",
            )
        )
    for key, members in sorted(stream_members.items()):
        object_clusters.update(
            _cluster_by_gap(members, gap_seconds=burst_gap_seconds, prefix=f"object_{key}")
        )
        session_clusters.update(
            _cluster_by_gap(
                members,
                gap_seconds=max(burst_gap_seconds * 5.0, burst_gap_seconds),
                prefix=f"capture_{key}",
            )
        )

    result = {}
    for sha, values in metadata.items():
        if values["explicit_group"] is not None:
            object_group = _private_identity("object_explicit", values["explicit_group"])
            key_source = "explicit_object_group"
        elif sha in object_clusters:
            object_group = object_clusters[sha]
            if values["client_id"] is not None:
                key_source = "client_id_time"
            elif values["explicit_session"] is not None:
                key_source = "capture_session_time"
            elif values["device_id"] is not None:
                key_source = "device_time"
            else:
                key_source = "timestamp_burst"
        else:
            object_group = f"object_sha_{sha[:24]}"
            key_source = "content_sha256"

        if values["explicit_session"] is not None:
            capture_session = _private_identity("capture_explicit", values["explicit_session"])
        elif sha in session_clusters:
            capture_session = session_clusters[sha]
        else:
            capture_session = f"capture_sha_{sha[:24]}"
        timestamp = values["timestamp"]
        result[sha] = {
            "object_group": object_group,
            "capture_session": capture_session,
            "lineage_key_source": key_source,
            "capture_timestamp": timestamp.isoformat().replace("+00:00", "Z") if timestamp else "",
        }
    return result


def _image_path(
    row: Mapping[str, object], image_root: Path
) -> tuple[Path | None, str | None]:
    value = row.get("image_ref")
    if not isinstance(value, str) or not value.strip():
        return None, "image_ref_missing"
    relative = Path(value.strip())
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        return None, "image_ref_invalid_or_outside_root"
    resolved = (image_root / relative).resolve(strict=False)
    try:
        resolved.relative_to(image_root)
    except ValueError:
        return None, "image_ref_invalid_or_outside_root"
    return resolved, None


def _read_image(path: Path) -> np.ndarray | None:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        return None
    if image is None or image.ndim != 3:
        return None
    return image


def _image_quality_reasons(image: np.ndarray) -> list[str]:
    """Apply objective pre-teacher checks without blur/model self-filtering."""
    reasons = []
    height, width = image.shape[:2]
    if width < MINIMUM_IMAGE_WIDTH or height < MINIMUM_IMAGE_HEIGHT:
        reasons.append("image_resolution_below_minimum")

    luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = float(luma.size)
    if (
        pixels
        and np.count_nonzero(luma <= UNDEREXPOSED_LUMA_MAX) / pixels
        >= EXTREME_EXPOSURE_FRACTION
    ):
        reasons.append("image_extreme_underexposure")
    if (
        pixels
        and np.count_nonzero(luma >= OVEREXPOSED_LUMA_MIN) / pixels
        >= EXTREME_EXPOSURE_FRACTION
    ):
        reasons.append("image_extreme_overexposure")
    return sorted(set(reasons))


def _independent_localization(
    row: Mapping[str, object], *, source_sha256: str, width: int, height: int,
    trusted_provider_rows: Sequence[Mapping[str, object]],
    trusted_provider_evidence: Sequence[Mapping[str, object]],
) -> tuple[object | None, list[str]]:
    value = row.get("independent_localization")
    if not isinstance(value, Mapping):
        return None, ["independent_localization_missing"]
    reasons = []
    required = {
        "schema_version",
        "source_image_sha256",
        "bbox_xyxy",
        "providers",
        "provider_iou",
        "contract",
        "contract_sha256",
        "deployed_prediction_used",
        "consensus",
    }
    if set(value) != required:
        reasons.append("independent_localization_invalid_shape")
    if value.get("schema_version") != LOCALIZATION_SCHEMA_VERSION:
        reasons.append("independent_localization_schema_version_mismatch")
    if _valid_sha(value.get("source_image_sha256")) != source_sha256:
        reasons.append("independent_localization_source_sha256_mismatch")
    if value.get("deployed_prediction_used") is not False:
        reasons.append("independent_localization_used_deployed_prediction")
    if value.get("consensus") is not True:
        reasons.append("independent_localization_no_consensus")
    bbox_xyxy = value.get("bbox_xyxy")
    bbox_shape_valid = not (
        not isinstance(bbox_xyxy, list)
        or len(bbox_xyxy) != 4
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in bbox_xyxy
        )
    )
    if not bbox_shape_valid:
        reasons.append("independent_localization_bbox_invalid_shape")
    providers = value.get("providers")
    if not isinstance(providers, list) or len(providers) != 2:
        reasons.append("independent_localization_requires_two_providers")
    else:
        provider_names = []
        provider_models = []
        provider_outputs = []
        normalized_providers = []
        for provider in providers:
            if not isinstance(provider, Mapping):
                reasons.append("independent_localization_invalid_provider")
                continue
            try:
                provider_box = [float(item) for item in provider.get("bbox_xyxy", [])]
            except (TypeError, ValueError):
                provider_box = []
            if (
                len(provider_box) != 4
                or not all(math.isfinite(item) for item in provider_box)
                or min(provider_box[0], provider_box[1]) < 0
                or provider_box[2] <= provider_box[0]
                or provider_box[3] <= provider_box[1]
                or provider_box[2] > width
                or provider_box[3] > height
            ):
                reasons.append("independent_localization_invalid_provider")
                continue
            core = provider_output_core(
                provider=str(provider.get("provider") or ""),
                source_sha=source_sha256,
                box=provider_box,
                model_sha=str(provider.get("model_sha256") or ""),
                spec_sha=str(provider.get("inference_spec_sha256") or ""),
            )
            if set(provider) != {*core, "provider_output_sha256"}:
                reasons.append("independent_localization_invalid_provider")
            if any(provider.get(key) != expected for key, expected in core.items()):
                reasons.append("independent_localization_invalid_provider")
            expected_output = _sha256_bytes(
                localization_canonical_json(core).encode("utf-8")
            )
            if provider.get("provider_output_sha256") != expected_output:
                reasons.append("independent_localization_provider_output_sha256_mismatch")
            name = provider.get("provider")
            if not isinstance(name, str) or not name.strip():
                reasons.append("independent_localization_invalid_provider")
            else:
                provider_names.append(name.strip().casefold())
            model_sha = _valid_sha(provider.get("model_sha256"))
            if model_sha is None:
                reasons.append("independent_localization_invalid_provider_model_sha256")
            else:
                provider_models.append(model_sha)
            if _valid_sha(provider.get("inference_spec_sha256")) is None:
                reasons.append("independent_localization_invalid_provider_inference_spec_sha256")
            provider_outputs.append(str(provider.get("provider_output_sha256") or ""))
            normalized_providers.append(provider)
        if len(provider_names) == 2 and len(set(provider_names)) != 2:
            reasons.append("independent_localization_providers_not_distinct")
        if len(provider_models) == 2 and len(set(provider_models)) != 2:
            reasons.append("independent_localization_provider_models_not_distinct")
        if len(provider_outputs) == 2 and len(set(provider_outputs)) != 2:
            reasons.append("independent_localization_provider_outputs_not_distinct")
        if len(normalized_providers) == 2:
            if list(normalized_providers) != [dict(item) for item in trusted_provider_rows]:
                reasons.append("independent_localization_provider_files_mismatch")
            first_box = [float(item) for item in normalized_providers[0]["bbox_xyxy"]]
            second_box = [float(item) for item in normalized_providers[1]["bbox_xyxy"]]
            actual_iou = bbox_iou(first_box, second_box)
            reported_iou = value.get("provider_iou")
            if (
                isinstance(reported_iou, bool)
                or not isinstance(reported_iou, (int, float))
                or not math.isfinite(float(reported_iou))
                or abs(float(reported_iou) - actual_iou) > AGGREGATE_TOLERANCE
            ):
                reasons.append("independent_localization_provider_iou_mismatch")
            if actual_iou < IOU_THRESHOLD:
                reasons.append("independent_localization_iou_below_threshold")
            expected_bbox = [
                (first_box[index] + second_box[index]) / 2.0 for index in range(4)
            ]
            if bbox_shape_valid:
                if any(
                    abs(float(bbox_xyxy[index]) - expected_bbox[index])
                    > AGGREGATE_TOLERANCE
                    for index in range(4)
                ):
                    reasons.append("independent_localization_aggregate_mismatch")

            contract = value.get("contract")
            if not isinstance(contract, Mapping):
                reasons.append("independent_localization_contract_invalid")
            else:
                provider_evidence = contract.get("providers")
                expected_evidence = []
                if isinstance(provider_evidence, list) and len(provider_evidence) == 2:
                    for evidence, provider in zip(provider_evidence, normalized_providers):
                        if not isinstance(evidence, Mapping):
                            expected_evidence = []
                            break
                        expected_evidence.append(
                            {
                                "provider": provider["provider"],
                                "manifest_sha256": evidence.get("manifest_sha256"),
                                "model_sha256": provider["model_sha256"],
                                "inference_spec_sha256": provider["inference_spec_sha256"],
                            }
                        )
                expected_contract = {
                    "schema_version": CONTRACT_SCHEMA_VERSION,
                    "method": AGGREGATE_METHOD,
                    "iou_threshold": IOU_THRESHOLD,
                    "aggregate_tolerance": AGGREGATE_TOLERANCE,
                    "source_image_sha256": source_sha256,
                    "providers": [dict(item) for item in trusted_provider_evidence],
                }
                if dict(contract) != expected_contract:
                    reasons.append("independent_localization_contract_invalid")
                expected_contract_sha = _sha256_bytes(
                    localization_canonical_json(expected_contract).encode("utf-8")
                )
                if value.get("contract_sha256") != expected_contract_sha:
                    reasons.append("independent_localization_contract_sha256_mismatch")
    return bbox_xyxy, sorted(set(reasons))


def _bbox(
    value: object,
    *,
    width: int,
    height: int,
    min_area_ratio: float,
    max_area_ratio: float,
) -> tuple[list[float] | None, float | None, str | None]:
    if isinstance(value, Mapping):
        if all(field in value for field in ("x1", "y1", "x2", "y2")):
            values = [value[field] for field in ("x1", "y1", "x2", "y2")]
        elif all(field in value for field in ("x", "y", "width", "height")):
            x, y = value["x"], value["y"]
            values = [x, y, float(x) + float(value["width"]), float(y) + float(value["height"])]
        else:
            return None, None, "bbox_invalid_shape"
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 4:
        values = list(value)
    else:
        return None, None, "bbox_missing"
    try:
        x1, y1, x2, y2 = (float(item) for item in values)
    except (TypeError, ValueError):
        return None, None, "bbox_non_numeric"
    if not all(math.isfinite(item) for item in (x1, y1, x2, y2)):
        return None, None, "bbox_non_finite"
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
        return None, None, "bbox_out_of_bounds"
    area_ratio = (x2 - x1) * (y2 - y1) / float(width * height)
    if area_ratio < min_area_ratio:
        return None, area_ratio, "bbox_area_too_small"
    if area_ratio > max_area_ratio:
        return None, area_ratio, "bbox_area_too_large"
    return [x1, y1, x2, y2], area_ratio, None


def _decision_tuple(
    row: Mapping[str, object],
) -> tuple[str, bool, bool, bool, str] | None:
    if not isinstance(row.get("material"), str):
        return None
    if not isinstance(row.get("single_object"), bool):
        return None
    if not isinstance(row.get("foreign_material"), bool):
        return None
    if not isinstance(row.get("training_usable"), bool):
        return None
    if row.get("quality_reason") not in QUALITY_REASONS:
        return None
    if bool(row["training_usable"]) != (row["quality_reason"] == "usable"):
        return None
    return (
        str(row["material"]).strip().casefold(),
        bool(row["single_object"]),
        bool(row["foreign_material"]),
        bool(row["training_usable"]),
        str(row["quality_reason"]),
    )


def _teacher_contract_reasons(row: Mapping[str, object]) -> list[str]:
    contract = row.get("teacher_contract")
    reported_sha = _valid_sha(row.get("teacher_contract_sha256"))
    if not isinstance(contract, Mapping):
        return ["teacher_contract_missing_or_invalid"]
    model = row.get("model")
    model_digest = valid_sha256(row.get("model_digest"))
    if not isinstance(model, str) or not model.strip() or model_digest is None:
        return ["teacher_contract_model_or_digest_invalid"]
    try:
        expected, expected_sha = build_teacher_contract(model, model_digest)
    except ValueError:
        return ["teacher_contract_model_or_digest_invalid"]
    reasons = []
    if dict(contract) != expected:
        reasons.append("teacher_contract_not_exact_trusted_contract")
    if reported_sha != expected_sha:
        reasons.append("teacher_contract_sha256_mismatch")
    if row.get("model_digest") != model_digest:
        reasons.append("teacher_model_digest_not_normalized")
    return sorted(set(reasons))


def _confidence(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or not 0 <= normalized <= 1:
        return None
    return normalized


def _teacher_consensus(row: Mapping[str, object]) -> tuple[dict | None, list[str]]:
    reasons = _teacher_contract_reasons(row)
    base_fields = set(TEACHER_LABEL_BASE_FIELDS)
    raw_fields = frozenset(set(row) - {"_input_line"})
    if raw_fields not in {
        frozenset(base_fields), frozenset(base_fields | {"independent_localization"})
    }:
        reasons.append("teacher_label_top_level_shape_mismatch")
    if {"deployed", "verifier", "bbox"}.intersection(row):
        reasons.append("teacher_label_contains_forbidden_prediction_or_bbox")
    if row.get("schema_version") != TEACHER_LABEL_SCHEMA_VERSION:
        reasons.append("teacher_label_schema_version_mismatch")
    errors = row.get("errors")
    if not isinstance(errors, list) or errors:
        reasons.append("teacher_errors")
    if row.get("consensus") is not True:
        reasons.append("no_exact_tuple_consensus")
    decision = row.get("consensus_decision")
    passes = row.get("passes")
    if not isinstance(decision, Mapping) or not isinstance(passes, list):
        return None, sorted(set((*reasons, "invalid_consensus_payload")))
    expected_decision_fields = {
        "material", "single_object", "foreign_material", "training_usable",
        "quality_reason", "votes", "pass_count",
    }
    if set(decision) != expected_decision_fields:
        reasons.append("invalid_consensus_payload")
    if len(passes) not in {2, 3}:
        reasons.append("invalid_teacher_pass_count")
    decision_tuple = _decision_tuple(decision)
    if decision_tuple is None:
        return None, sorted(set((*reasons, "invalid_consensus_payload")))
    valid_passes = []
    for item in passes:
        if (
            not isinstance(item, Mapping)
            or set(item)
            != {
                "material", "confidence", "single_object", "foreign_material",
                "training_usable", "quality_reason",
            }
            or _decision_tuple(item) is None
        ):
            reasons.append("invalid_teacher_pass")
            continue
        confidence = _confidence(item.get("confidence"))
        if confidence is None:
            reasons.append("invalid_teacher_pass")
            continue
        valid_passes.append((_decision_tuple(item), confidence))
    supporting = [confidence for value, confidence in valid_passes if value == decision_tuple]
    if (
        len(passes) == 3
        and len(valid_passes) >= 2
        and valid_passes[0][0] == valid_passes[1][0]
    ):
        reasons.append("unexpected_teacher_adjudication")
    if len(supporting) < 2:
        reasons.append("no_exact_tuple_consensus")
    reported_votes = decision.get("votes")
    if (
        isinstance(reported_votes, bool)
        or not isinstance(reported_votes, int)
        or reported_votes != len(supporting)
    ):
        reasons.append("consensus_vote_mismatch")
    reported_pass_count = decision.get("pass_count")
    if (
        isinstance(reported_pass_count, bool)
        or not isinstance(reported_pass_count, int)
        or reported_pass_count != len(passes)
    ):
        reasons.append("consensus_pass_count_mismatch")
    reported_minimum = _confidence(row.get("minimum_confidence"))
    if reported_minimum is None:
        reasons.append("invalid_minimum_confidence")
    actual_minimum = min(supporting) if supporting else -1.0
    if reported_minimum is not None and reported_minimum != actual_minimum:
        reasons.append("minimum_confidence_mismatch")
    return {
        "material": decision_tuple[0],
        "single_object": decision_tuple[1],
        "foreign_material": decision_tuple[2],
        "training_usable": decision_tuple[3],
        "quality_reason": decision_tuple[4],
        "votes": len(supporting),
        "pass_count": len(passes),
        "minimum_confidence": actual_minimum,
    }, sorted(set(reasons))


def _csv_text(
    rows: Sequence[Mapping[str, object]],
    *,
    fieldnames: Sequence[str] = MANIFEST_FIELDS,
) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    os.replace(temporary, path)


def build_operational_teacher_manifest(
    *,
    teacher_queue: Path,
    teacher_labels: Path,
    image_root: Path,
    known_audit: Path,
    provider_a_manifest: Path,
    provider_a_name: str,
    provider_a_model: Path,
    provider_a_spec: Path,
    provider_b_manifest: Path,
    provider_b_name: str,
    provider_b_model: Path,
    provider_b_spec: Path,
    output_dir: Path,
    capture_inventory: Path,
    role: str = "train",
    fold: str = "operational_teacher_v1",
    minimum_confidence: float = 0.80,
    burst_gap_seconds: float = 45.0,
    minimum_bbox_area_ratio: float = 0.003,
    maximum_bbox_area_ratio: float = 0.98,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    role = role.strip().casefold()
    if role not in ALLOWED_ROLES:
        raise ValueError("role must be train or calibration; VLM labels can never be blind_test")
    if not fold.strip():
        raise ValueError("fold must not be empty")
    if not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum_confidence must be between 0 and 1")
    if burst_gap_seconds < 0:
        raise ValueError("burst_gap_seconds must be non-negative")
    if not 0 < minimum_bbox_area_ratio < maximum_bbox_area_ratio <= 1:
        raise ValueError("bbox area ratios must satisfy 0 < minimum < maximum <= 1")
    image_root = image_root.resolve(strict=True)
    if not image_root.is_dir():
        raise NotADirectoryError(f"image_root is not a directory: {image_root}")

    input_paths = {
        "teacher_queue": teacher_queue, "teacher_labels": teacher_labels,
        "capture_inventory": capture_inventory, "known_audit": known_audit,
        "provider_a_manifest": provider_a_manifest,
        "provider_a_model": provider_a_model, "provider_a_spec": provider_a_spec,
        "provider_b_manifest": provider_b_manifest,
        "provider_b_model": provider_b_model, "provider_b_spec": provider_b_spec,
    }
    input_snapshot = {name: _sha256_file(path) for name, path in input_paths.items()}

    queue_rows = _load_jsonl(teacher_queue, kind="teacher queue")
    label_rows = _load_jsonl(teacher_labels, kind="teacher labels")
    inventory_rows = _load_inventory(capture_inventory)
    inventory_map = _inventory_by_sha(inventory_rows)
    known = load_known_audit(known_audit)
    if provider_a_name.strip().casefold() == provider_b_name.strip().casefold():
        raise ValueError("provider names must be distinct")
    provider_a_rows, provider_a_evidence = _provider_rows(
        provider_a_manifest, provider=provider_a_name,
        model_file=provider_a_model, spec_file=provider_a_spec,
    )
    provider_b_rows, provider_b_evidence = _provider_rows(
        provider_b_manifest, provider=provider_b_name,
        model_file=provider_b_model, spec_file=provider_b_spec,
    )
    if provider_a_evidence["model_sha256"] == provider_b_evidence["model_sha256"]:
        raise ValueError("provider model SHA values must be distinct")

    queue_groups: dict[str, list[dict]] = defaultdict(list)
    invalid_queue = []
    for row in queue_rows:
        sha = _valid_sha(row.get("sha256"))
        if sha is None:
            invalid_queue.append(
                {"sha256": None, "queue_line": row["_input_line"], "reasons": ["invalid_queue_sha256"]}
            )
        else:
            queue_groups[sha].append(row)
    label_groups: dict[str, list[dict]] = defaultdict(list)
    invalid_labels = []
    for row in label_rows:
        sha = _valid_sha(row.get("sha256"))
        if sha is None:
            invalid_labels.append(
                {"sha256": None, "label_line": row["_input_line"], "reasons": ["invalid_label_sha256"]}
            )
        else:
            label_groups[sha].append(row)

    unique_queue = {sha: rows[0] for sha, rows in queue_groups.items() if len(rows) == 1}
    lineage_by_sha = _lineage_rows(
        unique_queue, inventory_map, burst_gap_seconds=burst_gap_seconds
    )
    accepted = []
    empty_scene_inventory = []
    rejected = [*invalid_queue, *invalid_labels]

    for sha in sorted(queue_groups):
        reasons = []
        queue_candidates = queue_groups[sha]
        if len(queue_candidates) != 1:
            rejected.append(
                {
                    "sha256": sha,
                    "queue_lines": [row["_input_line"] for row in queue_candidates],
                    "reasons": ["duplicate_queue_sha256"],
                }
            )
            continue
        queue_row = queue_candidates[0]
        if sha in known:
            reasons.append("known_train_or_validation_sha_forbidden")
        if set(queue_row) - {"_input_line"} != {
            "sha256", "timestamp", "image_ref", "decision"
        }:
            reasons.append("teacher_queue_row_shape_not_exact")
        if queue_row.get("decision") != "teacher_required":
            reasons.append("teacher_queue_decision_mismatch")
        if {
            "client_id", "device_id", "deployed", "verifier", "image_path", "filepath"
        }.intersection(queue_row):
            reasons.append("teacher_queue_contains_forbidden_private_prediction_or_path")
        teacher_candidates = label_groups.get(sha, [])
        if not teacher_candidates:
            rejected.append({"sha256": sha, "reasons": ["missing_teacher_label"]})
            continue
        if len(teacher_candidates) != 1:
            rejected.append(
                {
                    "sha256": sha,
                    "label_lines": [row["_input_line"] for row in teacher_candidates],
                    "reasons": ["duplicate_teacher_label_sha256"],
                }
            )
            continue
        teacher_row = teacher_candidates[0]
        decision, consensus_reasons = _teacher_consensus(teacher_row)
        reasons.extend(consensus_reasons)
        if _valid_sha(teacher_row.get("input_image_sha256")) != sha:
            reasons.append("teacher_input_image_sha256_mismatch")
        if decision is not None:
            is_negative_decision = decision["material"] == "negative"
            if decision["minimum_confidence"] < minimum_confidence:
                reasons.append("minimum_confidence_below_threshold")
            # ``single_object=False`` is the coherent answer for an empty
            # scene.  Only positive material labels require one primary item.
            if not is_negative_decision and decision["single_object"] is not True:
                reasons.append("not_single_object")
            if is_negative_decision and decision["single_object"] is not False:
                reasons.append("negative_single_object_must_be_false")
            if is_negative_decision and decision["foreign_material"] is not False:
                reasons.append("negative_foreign_material_must_be_false")
            if decision["material"] not in ACCEPTED_TEACHER_MATERIALS:
                reasons.append("unsupported_teacher_material")
            if is_negative_decision and role != "train":
                reasons.append("negative_source_must_be_train")
            if decision["training_usable"] is not True:
                reasons.append(
                    f"training_unusable_{decision['quality_reason']}"
                )

        queue_timestamp = _timestamp(queue_row)
        if queue_timestamp is None:
            reasons.append("queue_capture_timestamp_missing_invalid_or_naive")
        elif queue_timestamp < OPERATIONAL_CAPTURE_CUTOFF_UTC:
            reasons.append("queue_capture_before_operational_cutoff")

        matching_inventory = inventory_map.get(sha, [])
        if len(matching_inventory) != 1:
            reasons.append("capture_inventory_row_missing_or_duplicate")
        for inventory_row in matching_inventory:
            if inventory_row.get("decision") != "teacher_required":
                reasons.append("capture_inventory_decision_mismatch")
            for field in ("sha256", "timestamp", "image_ref"):
                if inventory_row.get(field) != queue_row.get(field):
                    reasons.append(f"capture_inventory_{field}_mismatch")
            inventory_timestamp = _timestamp(inventory_row)
            if inventory_timestamp is None:
                reasons.append(
                    "inventory_capture_timestamp_missing_invalid_or_naive"
                )
            elif inventory_timestamp < OPERATIONAL_CAPTURE_CUTOFF_UTC:
                reasons.append("inventory_capture_before_operational_cutoff")

        image_path, image_path_problem = _image_path(queue_row, image_root)
        if image_path_problem:
            reasons.append(image_path_problem)
        if image_path is None or not image_path.is_file():
            reasons.append("image_missing")
            actual_sha = None
            image = None
            dimensions = None
        else:
            actual_sha = _sha256_file(image_path)
            if actual_sha != sha:
                reasons.append("image_sha256_mismatch")
            image = _read_image(image_path)
            if image is None:
                reasons.append("image_unreadable")
                dimensions = None
            else:
                height, width = image.shape[:2]
                dimensions = (int(width), int(height))
                reasons.extend(_image_quality_reasons(image))

        box = None
        area_ratio = None
        bbox_source = ""
        if decision is not None and decision["material"] in CLASS_IDS and dimensions:
            raw_box, localization_reasons = _independent_localization(
                teacher_row,
                source_sha256=sha,
                width=dimensions[0],
                height=dimensions[1],
                trusted_provider_rows=[
                    provider_a_rows.get(sha, {}), provider_b_rows.get(sha, {})
                ],
                trusted_provider_evidence=[
                    provider_a_evidence, provider_b_evidence
                ],
            )
            reasons.extend(localization_reasons)
            bbox_source = "independent_localization_consensus"
            box, area_ratio, bbox_problem = _bbox(
                raw_box,
                width=dimensions[0],
                height=dimensions[1],
                min_area_ratio=minimum_bbox_area_ratio,
                max_area_ratio=maximum_bbox_area_ratio,
            )
            if bbox_problem:
                reasons.append(bbox_problem)

        if reasons:
            rejection = {
                "sha256": sha,
                "queue_line": queue_row["_input_line"],
                "label_line": teacher_row["_input_line"],
                "reasons": sorted(set(reasons)),
            }
            if decision is not None:
                rejection["teacher_training_usable"] = decision["training_usable"]
                rejection["teacher_quality_reason"] = decision["quality_reason"]
            rejected.append(rejection)
            continue

        assert decision is not None and image_path is not None and dimensions is not None
        width, height = dimensions
        is_negative = decision["material"] == "negative"
        lineage = lineage_by_sha[sha]
        if is_negative:
            # A full camera frame is not shaped like the verifier input seen
            # at runtime.  Preserve the independently teacher-labelled source
            # as train-only inventory; a separate detector pass must produce a
            # real runtime-top1 proposal before this can become a background
            # training crop.
            empty_scene_inventory.append(
                {
                    "sample_id": f"opempty_{sha[:24]}",
                    "role": "train",
                    "split_role": "train",
                    "fold": fold.strip(),
                    "filepath": image_path.resolve().as_posix(),
                    "split": "training",
                    "source_id": sha,
                    "source_sha256": sha,
                    "image_sha256": sha,
                    "content_identity": f"sha256:{sha}",
                    "object_group": lineage["object_group"],
                    "capture_session": lineage["capture_session"],
                    "origin": "operational_empty_scene_vlm_teacher_source",
                    "selection_reason": (
                        "exact_tuple_high_confidence_negative_source_inventory"
                    ),
                    "material": 9,
                    "category": "background",
                    "teacher_material": "negative",
                    "teacher_training_usable": "true",
                    "teacher_quality_reason": decision["quality_reason"],
                    "dent": -1,
                    "label": -1,
                    "foreign_material": int(decision["foreign_material"]),
                    "source_object_count": 0,
                    "source_path_b64": base64.urlsafe_b64encode(
                        os.fsencode(image_path.resolve())
                    ).decode("ascii"),
                    "source_bbox_x": "",
                    "source_bbox_y": "",
                    "source_bbox_w": "",
                    "source_bbox_h": "",
                    "source_width": width,
                    "source_height": height,
                    "bbox_x1": "",
                    "bbox_y1": "",
                    "bbox_x2": "",
                    "bbox_y2": "",
                    "bbox_area_ratio": "",
                    "bbox_source": "",
                    "capture_timestamp": lineage["capture_timestamp"],
                    "lineage_key_source": lineage["lineage_key_source"],
                    "teacher_model": str(teacher_row.get("model") or ""),
                    "teacher_minimum_confidence": (
                        f"{decision['minimum_confidence']:.8f}"
                    ),
                    "teacher_consensus_votes": decision["votes"],
                    "teacher_pass_count": decision["pass_count"],
                    "pseudo_label": "true",
                    "ground_truth_authority": (
                        "vlm_teacher_pseudo_label_train_only"
                    ),
                    "blind_test_eligible": "false",
                    "training_crop_ready": "false",
                }
            )
            continue

        material = CLASS_IDS[decision["material"]]
        category = decision["material"]
        source_box = box
        assert source_box is not None and area_ratio is not None
        x1, y1, x2, y2 = source_box
        accepted.append(
            {
                "sample_id": f"opteacher_{sha[:24]}",
                "role": role,
                "split_role": role,
                "fold": fold.strip(),
                "filepath": image_path.resolve().as_posix(),
                "split": "training" if role == "train" else "calibration",
                "source_id": sha,
                "source_sha256": sha,
                "image_sha256": sha,
                "content_identity": f"sha256:{sha}",
                "object_group": lineage["object_group"],
                "capture_session": lineage["capture_session"],
                "origin": "operational_capture_vlm_teacher",
                "selection_reason": "exact_tuple_consensus_high_confidence",
                "material": material,
                "category": category,
                "teacher_material": decision["material"],
                "teacher_training_usable": "true",
                "teacher_quality_reason": decision["quality_reason"],
                "dent": -1,
                "label": -1,
                "foreign_material": int(decision["foreign_material"]),
                "source_object_count": 0 if is_negative else 1,
                "source_path_b64": base64.urlsafe_b64encode(os.fsencode(image_path.resolve())).decode("ascii"),
                "source_bbox_x": f"{x1:.6f}",
                "source_bbox_y": f"{y1:.6f}",
                "source_bbox_w": f"{x2 - x1:.6f}",
                "source_bbox_h": f"{y2 - y1:.6f}",
                "source_width": width,
                "source_height": height,
                "bbox_x1": f"{x1:.6f}",
                "bbox_y1": f"{y1:.6f}",
                "bbox_x2": f"{x2:.6f}",
                "bbox_y2": f"{y2:.6f}",
                "bbox_area_ratio": f"{area_ratio:.8f}",
                "bbox_source": bbox_source,
                "capture_timestamp": lineage["capture_timestamp"],
                "lineage_key_source": lineage["lineage_key_source"],
                "teacher_model": str(teacher_row.get("model") or ""),
                "teacher_minimum_confidence": f"{decision['minimum_confidence']:.8f}",
                "teacher_consensus_votes": decision["votes"],
                "teacher_pass_count": decision["pass_count"],
                "pseudo_label": "true",
                "ground_truth_authority": "vlm_teacher_pseudo_label_train_only",
                "blind_test_eligible": "false",
            }
        )

    for sha in sorted(set(label_groups) - set(queue_groups)):
        rejected.append({"sha256": sha, "reasons": ["teacher_label_not_in_queue"]})

    accepted.sort(key=lambda row: row["source_sha256"])
    empty_scene_inventory.sort(key=lambda row: row["source_sha256"])
    rejected.sort(
        key=lambda row: (
            str(row.get("sha256") or ""),
            row.get("queue_line", -1),
            row.get("label_line", -1),
            tuple(row["reasons"]),
        )
    )
    reason_counts = Counter(reason for row in rejected for reason in row["reasons"])

    csv_content = _csv_text(accepted)
    jsonl_content = "".join(_canonical_json(row) for row in accepted)
    empty_scene_csv_content = _csv_text(
        empty_scene_inventory, fieldnames=EMPTY_SCENE_INVENTORY_FIELDS
    )
    empty_scene_jsonl_content = "".join(
        _canonical_json(row) for row in empty_scene_inventory
    )
    rejection_report = {
        "accepted": len(accepted) + len(empty_scene_inventory),
        "crop_ready_manifest_rows": len(accepted),
        "empty_scene_inventory_rows": len(empty_scene_inventory),
        "queue_rows": len(queue_rows),
        "rejected_records": len(rejected),
        "reason_counts": dict(sorted(reason_counts.items())),
        "rejections": rejected,
    }
    rejection_content = _canonical_json(rejection_report, pretty=True)
    output_digests = {
        "csv_sha256": _sha256_bytes(csv_content.encode("utf-8")),
        "jsonl_sha256": _sha256_bytes(jsonl_content.encode("utf-8")),
        "empty_scene_csv_sha256": _sha256_bytes(
            empty_scene_csv_content.encode("utf-8")
        ),
        "empty_scene_jsonl_sha256": _sha256_bytes(
            empty_scene_jsonl_content.encode("utf-8")
        ),
        "rejections_sha256": _sha256_bytes(rejection_content.encode("utf-8")),
    }
    lineage_report = {
        "builder": "scripts/build_operational_teacher_manifest.py",
        "portable": False,
        "local_only_contains_absolute_paths": True,
        "policy": {
            "allowed_roles": sorted(ALLOWED_ROLES),
            "blind_test_eligible": False,
            "ground_truth_authority": "vlm_teacher_pseudo_label_train_only",
            "teacher_label_schema_version": TEACHER_LABEL_SCHEMA_VERSION,
            "negative_source_role": "train",
            "negative_source_is_training_crop": False,
            "negative_source_requires_runtime_proposal_mining": True,
            "role": role,
            "fold": fold.strip(),
            "minimum_confidence": minimum_confidence,
            "burst_gap_seconds": burst_gap_seconds,
            "minimum_bbox_area_ratio": minimum_bbox_area_ratio,
            "maximum_bbox_area_ratio": maximum_bbox_area_ratio,
            "operational_capture_cutoff_kst": (
                OPERATIONAL_CAPTURE_CUTOFF_KST.isoformat()
            ),
            "operational_capture_cutoff_utc": (
                OPERATIONAL_CAPTURE_CUTOFF_UTC.isoformat().replace("+00:00", "Z")
            ),
            "minimum_image_width": MINIMUM_IMAGE_WIDTH,
            "minimum_image_height": MINIMUM_IMAGE_HEIGHT,
            "extreme_exposure_fraction": EXTREME_EXPOSURE_FRACTION,
            "underexposed_luma_max": UNDEREXPOSED_LUMA_MAX,
            "overexposed_luma_min": OVEREXPOSED_LUMA_MIN,
            "blur_filter_enabled": False,
            "deployed_prediction_filter_enabled": False,
        },
        "inputs": {
            **{f"{name}_sha256": digest for name, digest in input_snapshot.items()},
            "provider_a_name": provider_a_name,
            "provider_b_name": provider_b_name,
        },
        "counts": {
            "accepted": len(accepted) + len(empty_scene_inventory),
            "crop_ready_manifest_rows": len(accepted),
            "empty_scene_inventory_rows": len(empty_scene_inventory),
            "unique_accepted_sha256": len(
                {
                    row["source_sha256"]
                    for row in (*accepted, *empty_scene_inventory)
                }
            ),
            "rejected_records": len(rejected),
            "accepted_by_teacher_material": dict(
                sorted(
                    Counter(
                        row["teacher_material"]
                        for row in (*accepted, *empty_scene_inventory)
                    ).items()
                )
            ),
            "unique_object_groups": len(
                {row["object_group"] for row in (*accepted, *empty_scene_inventory)}
            ),
            "unique_capture_sessions": len(
                {row["capture_session"] for row in (*accepted, *empty_scene_inventory)}
            ),
        },
        "output_digests": output_digests,
    }
    lineage_content = _canonical_json(lineage_report, pretty=True)

    targets = {name: output_dir / filename for name, filename in ARTIFACT_NAMES.items()}
    existing = sorted(str(path) for path in targets.values() if path.exists())
    if not dry_run and existing and not overwrite:
        raise FileExistsError(f"output artifacts already exist; use --overwrite: {existing}")
    if not dry_run:
        final_snapshot = {
            name: _sha256_file(path) for name, path in input_paths.items()
        }
        if final_snapshot != input_snapshot:
            raise ValueError("builder input changed before final publish")
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, content in (
            ("csv", csv_content),
            ("jsonl", jsonl_content),
            ("empty_scene_csv", empty_scene_csv_content),
            ("empty_scene_jsonl", empty_scene_jsonl_content),
            ("rejections", rejection_content),
            ("lineage", lineage_content),
        ):
            _atomic_write(targets[name], content)

    return {
        "dry_run": dry_run,
        "output_dir": str(output_dir.resolve()),
        "accepted": len(accepted) + len(empty_scene_inventory),
        "crop_ready_manifest_rows": len(accepted),
        "empty_scene_inventory_rows": len(empty_scene_inventory),
        "rejected_records": len(rejected),
        "reason_counts": dict(sorted(reason_counts.items())),
        "unique_object_groups": lineage_report["counts"]["unique_object_groups"],
        "output_digests": output_digests,
        "blind_test_eligible": False,
        "operational_capture_cutoff_kst": OPERATIONAL_CAPTURE_CUTOFF_KST.isoformat(),
        "teacher_label_schema_version": TEACHER_LABEL_SCHEMA_VERSION,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe train/calibration manifests from VLM teacher labels"
    )
    parser.add_argument("--teacher-queue", required=True, type=Path)
    parser.add_argument("--teacher-labels", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--known-audit", required=True, type=Path)
    parser.add_argument("--capture-inventory", required=True, type=Path)
    for prefix in ("a", "b"):
        parser.add_argument(f"--provider-{prefix}-manifest", required=True, type=Path)
        parser.add_argument(f"--provider-{prefix}-name", required=True)
        parser.add_argument(f"--provider-{prefix}-model", required=True, type=Path)
        parser.add_argument(f"--provider-{prefix}-spec", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--role", choices=sorted(ALLOWED_ROLES), default="train")
    parser.add_argument("--fold", default="operational_teacher_v1")
    parser.add_argument("--minimum-confidence", type=float, default=0.80)
    parser.add_argument("--burst-gap-seconds", type=float, default=45.0)
    parser.add_argument("--minimum-bbox-area-ratio", type=float, default=0.003)
    parser.add_argument("--maximum-bbox-area-ratio", type=float, default=0.98)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = build_operational_teacher_manifest(
        teacher_queue=args.teacher_queue,
        teacher_labels=args.teacher_labels,
        image_root=args.image_root,
        known_audit=args.known_audit,
        provider_a_manifest=args.provider_a_manifest,
        provider_a_name=args.provider_a_name,
        provider_a_model=args.provider_a_model,
        provider_a_spec=args.provider_a_spec,
        provider_b_manifest=args.provider_b_manifest,
        provider_b_name=args.provider_b_name,
        provider_b_model=args.provider_b_model,
        provider_b_spec=args.provider_b_spec,
        capture_inventory=args.capture_inventory,
        output_dir=args.output_dir,
        role=args.role,
        fold=args.fold,
        minimum_confidence=args.minimum_confidence,
        burst_gap_seconds=args.burst_gap_seconds,
        minimum_bbox_area_ratio=args.minimum_bbox_area_ratio,
        maximum_bbox_area_ratio=args.maximum_bbox_area_ratio,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
