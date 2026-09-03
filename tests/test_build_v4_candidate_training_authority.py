from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest

import scripts.audit_v4_near_duplicate_leakage as near_duplicate_auditor
import scripts.build_v4_candidate_training_authority as authority_builder

from scripts.build_v4_candidate_training_authority import (
    AUTHORITY_ROLE,
    AUTHORITY_SCHEMA,
    AUTHORITY_STATUS,
    FALSE_AUTHORITY_FIELDS,
    FULL_DATA_REPORT_ROLE,
    MATERIAL_CLASSES,
    OPERATIONAL_CUTOFF,
    OUTPUT_FIELDS,
    PROTECTED_FIELDS,
    QUALITY_CONTRACT,
    QUALITY_ROLE,
    QX3_READY_ROLE,
    QX3_REPORT_ROLE,
    TRUST_ROOT_CODE_PATHS,
    build_training_authority,
)


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _fake_sha(label: str) -> str:
    return _sha(label.encode("utf-8"))


def _dump(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _dump_compact(path: Path, value: object) -> Path:
    path.write_bytes(authority_builder._canonical_compact_payload(value) + b"\n")
    return path


def _near_duplicate_entry(
    *, role: str, cohort: str, view_kind: str, sample_id: str,
    source_sha256: str, image_sha256: str, size: int,
) -> dict[str, object]:
    identity = {
        "cohort": cohort,
        "image_sha256": image_sha256,
        "role": role,
        "sample_id": sample_id,
        "source_sha256": source_sha256,
        "view_kind": view_kind,
    }
    asset_id = _sha(authority_builder._canonical_compact_payload(identity))
    phash = _fake_sha(f"phash:{asset_id}")
    return {
        "asset_id": asset_id,
        "role": role,
        "cohort": cohort,
        "view_kind": view_kind,
        "sample_id": sample_id,
        "source_sha256": source_sha256,
        "image_sha256": image_sha256,
        "size": size,
        "width": 1,
        "height": 1,
        "phash_rot4": [phash[index:index + 16] for index in range(0, 64, 16)],
    }

def _qnap_inventory_tree(
    source_root: str, container_root: str, library_name: str
) -> dict[str, object]:
    entries = [
        {
            "path": f"{library_name}.so",
            "type": "symlink",
            "target": f"{library_name}.so.1",
        },
        {
            "path": f"{library_name}.so.1",
            "type": "file",
            "size": 16,
            "sha256": _fake_sha(f"{library_name}-bytes"),
        },
    ]
    return {
        "source_root": source_root,
        "container_root": container_root,
        "total_regular_bytes": 16,
        "tree_sha256": _sha(authority_builder._canonical_compact_json(entries)),
        "entries": entries,
    }


def _quality_value(entries: list[dict[str, str]]) -> dict[str, object]:
    entries = sorted(entries, key=lambda row: row["source_sha256"])
    canonical = (
        json.dumps(
            entries,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    reason_counts: dict[str, int] = {}
    for row in entries:
        reason_counts[row["reason"]] = reason_counts.get(row["reason"], 0) + 1
    return {
        "schema_version": 1,
        "artifact_role": QUALITY_ROLE,
        "quality_exclusion_contract": QUALITY_CONTRACT,
        "status": "quality_exclusions_ready",
        "excluded_source_count": len(entries),
        "max_excluded_sources": 100,
        "reason_counts": dict(sorted(reason_counts.items())),
        "source_list_sha256": _sha(canonical),
        "entries": entries,
        "authority": {field: False for field in FALSE_AUTHORITY_FIELDS},
    }


def _write_manifest(fixture: dict[str, object]) -> None:
    path = fixture["manifest"]
    assert isinstance(path, Path)
    rows = fixture["rows"]
    assert isinstance(rows, list)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _policy_bindings(fixture: dict[str, object]) -> dict[str, str]:
    names = {
        "qx3_diagnostic_ready_sha256": "qx3_ready",
        "qx3_diagnostic_report_sha256": "qx3_report",
        "license_allowlist_sha256": "license_path",
        "quality_exclusions_sha256": "quality",
        "protected_sources_sha256": "protected",
        "candidate_near_duplicate_audit_sha256": "near_duplicate_audit",
        "protected_reference_inventory_sha256": "protected_inventory",
        "code_inventory_sha256": "inventory",
        "training_config_sha256": "config",
        "host_launch_contract_sha256": "host",
        "raw_container_inspect_sha256": "raw_inspect",
        "pretrained_backbone_sha256": "backbone",
    }
    result = {
        field: _sha(fixture[name].read_bytes())  # type: ignore[union-attr]
        for field, name in names.items()
    }
    result["container_image_id"] = fixture["image_id"]  # type: ignore[assignment]
    return result


def _snapshot_report_bytes_for_policy(entries: list[dict[str, object]]) -> bytes:
    """Render valid bytes, while allowing intentionally invalid leakage fixtures."""

    try:
        return authority_builder._dataset_snapshot_report(entries)[1]
    except ValueError:
        normalized = sorted(entries, key=lambda row: (str(row["role"]), str(row["sample_id"])))
        tree_rows = [
            {"path": row["path"], "size": row["size"], "sha256": row["sha256"]}
            for row in sorted(normalized, key=lambda row: str(row["path"]))
        ]
        report = {
            "schema": authority_builder.DATASET_SNAPSHOT_SCHEMA,
            "artifact_role": authority_builder.DATASET_SNAPSHOT_ROLE,
            "status": "candidate_dataset_snapshot_ready",
            "candidate_only": True,
            "production_deployment_authorized": False,
            "payload_kind": "training_crop_only",
            "source_lineage_bytes_snapshotted": False,
            "snapshot_root_relative": "dataset_snapshot",
            "snapshot_max_bytes": authority_builder.DATASET_SNAPSHOT_MAX_BYTES,
            "object_max_bytes": authority_builder.IMAGE_CONSUMPTION_MAX_BYTES,
            "object_count": len(normalized),
            "total_regular_bytes": sum(int(row["size"]) for row in normalized),
            "tree_sha256": _sha(authority_builder._canonical_compact_json(tree_rows)),
            "payload_set_sha256": _sha(
                authority_builder._canonical_compact_json(normalized)
            ),
            "objects": normalized,
        }
        return authority_builder._canonical_json(report)


def _refresh(fixture: dict[str, object]) -> None:
    _write_manifest(fixture)
    manifest = fixture["manifest"]
    report = fixture["full_report"]
    rows = fixture["rows"]
    assert isinstance(manifest, Path) and isinstance(report, Path) and isinstance(rows, list)
    _dump(
        report,
        {
            "schema_version": 1,
            "artifact_role": FULL_DATA_REPORT_ROLE,
            "ready_for_lineage_upgrade": True,
            "blind_test_eligible": False,
            "production_deployment_authorized": False,
            "rows": len(rows),
            "counts": dict(sorted({
                key: sum(
                    1 for row in rows
                    if f"{row['split']}/{row['category']}" == key
                )
                for key in {
                    f"{row['split']}/{row['category']}" for row in rows
                }
            }.items())),
            "contract": {
                "manifest_schema_version": "proposal_verifier.v4.bgfix.v1",
                "background_policy": "strict-zero-intersection",
                "background_gt_margin": 0.10,
                "explicit_label_file_required": True,
                "source_object_count_semantics": "complete_source_frame",
                "crop_object_count_semantics": "final_padded_verifier_crop",
                "visual_judge_still_required": True,
                "proposal_provenance": {
                    "sources": len({row["source_sha256"] for row in rows}),
                    "provider_kind": "frozen_yolo_runtime",
                    "runtime_detector_executed": True,
                    "runtime_top1_replayed": True,
                    "provided_top1_predictions_matched": True,
                    "proposal_class_confidence_bbox_matched": True,
                    "confidence_abs_tolerance": 1e-6,
                    "bbox_abs_tolerance": 1e-4,
                    "original_generation_event_cryptographically_attested": False,
                    "authority": "development_only_current_detector_reproduction",
                    "cuda_client_initialized_before_source_crop_scan": True,
                    "detector_artifact_bytes_bound": True,
                    "detector_replay_used_unique_snapshot": True,
                    "source_and_label_replay_used_unique_snapshots": True,
                    "replay_snapshots_verified_after_inference": True,
                    "original_detector_bytes_unchanged_through_validation": True,
                    "inference_spec_bytes_bound": True,
                    "dataset_info_bytes_bound": True,
                    "source_bbox_crop_bytes_recomputed": True,
                    "production_or_blind_authority": False,
                },
            },
            "bindings": {
                "input_manifest_sha256": _fake_sha("raw-input-manifest"),
                "dataset_info_sha256": _fake_sha("dataset-info"),
                "detector_model_sha256": _fake_sha("detector-model"),
                "inference_spec_sha256": _fake_sha("inference-spec"),
                "validated_manifest_sha256": _sha(manifest.read_bytes()),
            },
        },
    )
    operational_sources: dict[str, object] = {}
    for row in rows:
        if row["origin"] != "ops" or not row["captured_at"]:
            continue
        timestamp = row["captured_at"].replace("Z", "+00:00")
        from datetime import datetime

        parsed = datetime.fromisoformat(timestamp)
        if parsed.astimezone(OPERATIONAL_CUTOFF.tzinfo) >= OPERATIONAL_CUTOFF:
            operational_sources[row["source_sha256"]] = {
                "auditor_sha256": row["auditor_sha256"],
                "teacher_output_sha256": row["teacher_output_sha256"],
                "localizer_output_sha256": row["localizer_output_sha256"],
            }
    quality_path = fixture["quality"]
    assert isinstance(quality_path, Path)
    quality_value = json.loads(quality_path.read_text(encoding="utf-8"))
    exclusion_reasons = {
        entry["source_sha256"]: entry["reason"]
        for entry in quality_value["entries"]
    }
    excluded_sources = set(exclusion_reasons)
    excluded_counts: Counter[str] = Counter()
    selected: dict[str, list[dict[str, str]]] = {
        "train": [], "model_validation": []
    }
    snapshot_plan: list[dict[str, object]] = []
    from datetime import datetime

    for row in rows:
        role = row.get("role")
        if role not in selected:
            continue
        if row["source_sha256"] in excluded_sources:
            excluded_counts[f"quality/{exclusion_reasons[row['source_sha256']]}"] += 1
            continue
        if row["origin"] == "ops":
            captured = datetime.fromisoformat(row["captured_at"].replace("Z", "+00:00"))
            if captured.astimezone(OPERATIONAL_CUTOFF.tzinfo) < OPERATIONAL_CUTOFF:
                excluded_counts["operational/before_2026_08_01_kst"] += 1
                continue
        selected_row = {field: row.get(field, "") for field in OUTPUT_FIELDS}
        selected_row["source_filepath"] = Path(row["source_filepath"]).resolve().as_posix()
        selected_row["filepath"] = authority_builder._snapshot_relative_path(
            row["image_sha256"]
        )
        selected[role].append(selected_row)
        snapshot_plan.append({
            "sample_id": row["sample_id"],
            "role": role,
            "path": selected_row["filepath"],
            "size": Path(row["filepath"]).stat().st_size,
            "sha256": row["image_sha256"],
        })
    for selected_rows in selected.values():
        selected_rows.sort(key=lambda row: row["sample_id"])
    config_path = fixture["config"]
    assert isinstance(config_path, Path)
    config_value = json.loads(config_path.read_text(encoding="utf-8"))
    train_origin_counts = Counter(row["origin"] for row in selected["train"])
    weights = config_value["origin_weights"]
    weighted_mass = {
        origin: count * float(weights.get(origin, 1.0))
        for origin, count in sorted(train_origin_counts.items())
    }
    total_mass = sum(weighted_mass.values())
    config_value["sampling_mode"] = (
        "weighted_replacement"
        if len({float(weights.get(origin, 1.0)) for origin in train_origin_counts}) > 1
        else "shuffle_without_replacement"
    )
    config_value["sampling_samples_per_epoch"] = len(selected["train"])
    config_value["sampling_expected_fraction_by_origin"] = {
        origin: mass / total_mass for origin, mass in weighted_mass.items()
    }
    _dump(config_path, config_value)
    snapshot_report_bytes = _snapshot_report_bytes_for_policy(snapshot_plan)
    snapshot_report_value = json.loads(snapshot_report_bytes)
    trainer_path = fixture["trainer"]
    assert isinstance(trainer_path, Path)
    consumption_contract = authority_builder._dataset_consumption_contract(
        trainer_sha256=_sha(trainer_path.read_bytes()),
        snapshot_report_sha256=_sha(snapshot_report_bytes),
        snapshot_tree_sha256=snapshot_report_value["tree_sha256"],
        manifest_payload_set_sha256=snapshot_report_value["payload_set_sha256"],
    )
    candidate_manifest_bindings = {
        "candidate_train_manifest_sha256": _sha(
            authority_builder._render_manifest(selected["train"])
        ),
        "candidate_model_validation_manifest_sha256": _sha(
            authority_builder._render_manifest(selected["model_validation"])
        ),
        "candidate_dataset_snapshot_sha256": _sha(snapshot_report_bytes),
        "candidate_dataset_consumption_contract_sha256": _sha(
            authority_builder._canonical_json(consumption_contract)
        ),
    }
    protected_path = fixture["protected"]
    protected_inventory_path = fixture["protected_inventory"]
    auditor_path = fixture["auditor"]
    near_duplicate_path = fixture["near_duplicate_audit"]
    assert all(
        isinstance(path, Path)
        for path in (
            protected_path,
            protected_inventory_path,
            auditor_path,
            near_duplicate_path,
        )
    )
    protected_value = json.loads(protected_path.read_text(encoding="utf-8"))
    protected_union = sorted(
        {
            digest
            for field in PROTECTED_FIELDS
            for digest in protected_value[field]
        }
    )
    protected_inventory_value = json.loads(
        protected_inventory_path.read_text(encoding="utf-8")
    )
    crop_sizes = {
        str(item["sha256"]): int(item["size"])
        for item in snapshot_plan
    }
    candidate_entries: list[dict[str, object]] = []
    for role, role_rows in selected.items():
        for row in role_rows:
            source_sha = row["source_sha256"]
            image_sha = row["image_sha256"]
            candidate_entries.append(
                _near_duplicate_entry(
                    role=role,
                    cohort="candidate",
                    view_kind="source",
                    sample_id=f"source:{role}:{source_sha}",
                    source_sha256=source_sha,
                    image_sha256=source_sha,
                    size=Path(row["source_filepath"]).stat().st_size,
                )
            )
            candidate_entries.append(
                _near_duplicate_entry(
                    role=role,
                    cohort="candidate",
                    view_kind="crop",
                    sample_id=row["sample_id"],
                    source_sha256=source_sha,
                    image_sha256=image_sha,
                    size=crop_sizes[image_sha],
                )
            )
    protected_entries = [
        _near_duplicate_entry(
            role=str(item["cohort"]),
            cohort=str(item["cohort"]),
            view_kind=str(item["view_kind"]),
            sample_id=(
                f"protected:{item['cohort']}:{item['view_kind']}:"
                f"{str(item['image_sha256'])[:16]}:"
                f"{_sha(str(item['path']).encode('utf-8'))[:16]}"
            ),
            source_sha256=str(item["source_sha256"]),
            image_sha256=str(item["image_sha256"]),
            size=int(item["size"]),
        )
        for item in protected_inventory_value["objects"]
    ]
    near_duplicate_entries = sorted(
        candidate_entries + protected_entries,
        key=lambda item: str(item["asset_id"]),
    )
    near_duplicate_edges, near_duplicate_clusters = (
        authority_builder._reconstruct_near_duplicate_graph(
            near_duplicate_entries
        )
    )
    near_duplicate_report = {
        "schema": authority_builder.NEAR_DUPLICATE_REPORT_SCHEMA,
        "status": "passed",
        "ok": True,
        "artifact_role": "candidate_dataset_separation_evidence_only",
        "authority": {
            "candidate_only": True,
            "label_authority": False,
            "blind_authority": False,
            "promotion_authority": False,
            "deployment_authority": False,
            "automatic_delete_or_relabel": False,
        },
        "algorithm": {
            "id": authority_builder.NEAR_DUPLICATE_ALGORITHM_ID,
            "threshold": authority_builder.NEAR_DUPLICATE_PHASH_DISTANCE,
            "decode": "verified_bytes_cv2_grayscale_ignore_exif_orientation",
            "views": ["rot0", "rot90", "rot180", "rot270"],
            "resize": {"width": 32, "height": 32, "interpolation": "INTER_AREA"},
            "dct": {"dtype": "float32", "low_frequency_block": [8, 8]},
            "bit_rule": (
                "row_major_msb_first; median(coefficients[1:]); "
                "coefficient>median; dc=0"
            ),
            "byte_cap": authority_builder.IMAGE_CONSUMPTION_MAX_BYTES,
            "pixel_cap": authority_builder.NEAR_DUPLICATE_PIXEL_CAP,
            "exact_right_angle_rotation_invariant": True,
            "crop_invariant": False,
            "graph_edge_cap": authority_builder.NEAR_DUPLICATE_GRAPH_EDGE_CAP,
            "runtime": {
                "python": "3.12.0",
                "opencv": "4.10.0",
                "numpy": "2.0.0",
                "pillow": "11.0.0",
                "opencv_build_information_sha256": _fake_sha("opencv-build"),
            },
        },
        "bindings": {
            "candidate_manifest_sha256": {
                "train": candidate_manifest_bindings[
                    "candidate_train_manifest_sha256"
                ],
                "model_validation": candidate_manifest_bindings[
                    "candidate_model_validation_manifest_sha256"
                ],
            },
            "candidate_payload_set_sha256": _sha(
                authority_builder._canonical_compact_payload(
                    sorted({entry["image_sha256"] for entry in candidate_entries})
                )
            ),
            "protected_payload_set_sha256": _sha(
                authority_builder._canonical_compact_payload(
                    sorted({entry["image_sha256"] for entry in protected_entries})
                )
            ),
            "protected_sources": {
                "file_sha256": _sha(protected_path.read_bytes()),
                "payload_sha256": _sha(
                    authority_builder._canonical_compact_payload(protected_value)
                ),
                "canonical_union_sha256": _sha(
                    authority_builder._canonical_compact_payload(protected_union)
                ),
            },
            "protected_inventory": {
                "file_sha256": _sha(protected_inventory_path.read_bytes()),
                "payload_sha256": _sha(
                    authority_builder._canonical_compact_payload(
                        protected_inventory_value
                    )
                ),
            },
            "auditor": {
                "path": authority_builder.NEAR_DUPLICATE_AUDITOR_PATH,
                "sha256": _sha(auditor_path.read_bytes()),
                "runtime_code_sha256": (
                    near_duplicate_auditor.runtime_code_fingerprint_sha256()
                ),
            },
        },
        "coverage": {
            "candidate_assets": len(candidate_entries),
            "protected_assets": len(protected_entries),
            "protected_source_union": len(protected_union),
            "verified_assets": (
                len(candidate_entries) + len(protected_entries)
            ),
            "complete": True,
        },
        "summary": {
            "edges": len(near_duplicate_edges),
            "clusters": len(near_duplicate_clusters),
            "blocking_multi_role_clusters": 0,
            "same_role_duplicate_clusters_nonblocking": sum(
                1
                for cluster in near_duplicate_clusters
                if not cluster["blocking"] and int(cluster["edge_count"]) > 0
            ),
        },
        "entries": near_duplicate_entries,
        "edges": near_duplicate_edges,
        "clusters": near_duplicate_clusters,
    }
    _dump_compact(near_duplicate_path, near_duplicate_report)
    license_path = fixture["license_path"]
    assert isinstance(license_path, Path)
    license_origins = json.loads(
        license_path.read_text(encoding="utf-8")
    )["origins"]
    material_by_role = {role: Counter() for role in selected}
    objectness_by_role = {role: Counter() for role in selected}
    origin_by_role = {role: Counter() for role in selected}
    condition_by_role = {
        role: {head: Counter() for head in ("dent", "label", "foreign_material")}
        for role in selected
    }
    license_by_role = {role: Counter() for role in selected}
    dataset_by_role = {role: Counter() for role in selected}
    all_origins: Counter[str] = Counter()
    all_conditions = {
        head: Counter() for head in ("dent", "label", "foreign_material")
    }
    for role, selected_rows in selected.items():
        for row in selected_rows:
            material_by_role[role][row["category"]] += 1
            objectness_by_role[role][
                "background" if row["material"] == "9" else "material"
            ] += 1
            origin_by_role[role][row["origin"]] += 1
            all_origins[row["origin"]] += 1
            rule = license_origins.get(
                row["origin"], {"kind": "unknown", "dataset_id": "unknown"}
            )
            license_by_role[role][rule["kind"]] += 1
            dataset_by_role[role][rule["dataset_id"]] += 1
            for head in ("dent", "label", "foreign_material"):
                condition_by_role[role][head][row[head]] += 1
                all_conditions[head][row[head]] += 1
    candidate_counts = {
        "selected_by_role": {role: len(selected[role]) for role in selected},
        "selected_by_origin": dict(sorted(all_origins.items())),
        "material_by_role": {
            role: dict(sorted(material_by_role[role].items())) for role in selected
        },
        "objectness_by_role": {
            role: dict(sorted(objectness_by_role[role].items())) for role in selected
        },
        "origin_by_role": {
            role: dict(sorted(origin_by_role[role].items())) for role in selected
        },
        "condition_targets_by_role": {
            role: {
                head: dict(sorted(condition_by_role[role][head].items()))
                for head in condition_by_role[role]
            }
            for role in selected
        },
        "license_kind_by_role": {
            role: dict(sorted(license_by_role[role].items())) for role in selected
        },
        "dataset_by_role": {
            role: dict(sorted(dataset_by_role[role].items())) for role in selected
        },
        "excluded": dict(sorted(excluded_counts.items())),
        "condition_targets": {
            head: dict(sorted(all_conditions[head].items()))
            for head in all_conditions
        },
    }
    _dump(
        fixture["policy"],  # type: ignore[arg-type]
        {
            "schema": "v4_candidate_training_trusted_policy.v1",
            "artifact_role": "approved_v4_candidate_training_policy",
            "status": "approved",
            "approved": True,
            "operational_cutoff_kst": OPERATIONAL_CUTOFF.isoformat(),
            "source_manifest_sha256": [_sha(manifest.read_bytes())],
            "full_data_validator_report_sha256": [_sha(report.read_bytes())],
            **_policy_bindings(fixture),
            **candidate_manifest_bindings,
            "operational_sources": operational_sources,
            "license_origins": license_origins,
            "candidate_counts": candidate_counts,
        },
    )


def _fixture(tmp_path: Path) -> dict[str, object]:
    global_root = tmp_path / "global"
    data_root = global_root / "data"
    data_root.mkdir(parents=True)
    code_root = global_root / "code"
    wrapper = code_root / "scripts" / "nas" / "run_v4_candidate_training.sh"
    wrapper.parent.mkdir(parents=True)
    source_wrapper = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "nas"
        / "run_v4_candidate_training.sh"
    )
    wrapper.write_bytes(source_wrapper.read_bytes())
    trainer = code_root / "scripts" / "train_multitask_verifier.py"
    trainer.write_text(
        """from __future__ import annotations
import csv,hashlib,json,math,sys
from collections import Counter
from pathlib import Path

def _build_optimizer(model, *, lr, backbone_lr, head_lr):
    return torch.optim.AdamW([], lr=lr, weight_decay=1e-4), {}

def _scheduler_contract(optimizer, effective):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, effective['epochs']
    )

def values(flag):
    return [sys.argv[i + 1] for i, value in enumerate(sys.argv[:-1]) if value == flag]
def one(flag):
    found = values(flag)
    if len(found) != 1: raise SystemExit(64)
    return found[0]
manifests = values('--manifest')
train_origins = Counter()
for manifest in manifests:
    with Path(manifest).open(encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if row['role'] == 'train': train_origins[row['origin']] += 1
origin_weights = {}
for value in values('--origin-weight'):
    origin, weight = value.split('=', 1); origin_weights[origin] = float(weight)
weighted_mass = {
    origin: count * origin_weights.get(origin, 1.0)
    for origin, count in sorted(train_origins.items())
}
total_mass = math.fsum(weighted_mass.values())
sampling = {
    'mode': 'weighted_replacement' if len({origin_weights.get(origin, 1.0) for origin in train_origins}) > 1 else 'shuffle_without_replacement',
    'samples_per_epoch': sum(train_origins.values()),
    'configured_origin_weights': dict(sorted(origin_weights.items())),
    'row_counts_by_origin': dict(sorted(train_origins.items())),
    'weighted_mass_by_origin': {origin: float(value) for origin, value in weighted_mass.items()},
    'expected_fraction_by_origin': {origin: float(value / total_mass) for origin, value in weighted_mass.items()},
    'manifest_rows_remain_unique': True,
}
material_classes = ['can','pet','paper','plastic','styrofoam','vinyl','glass','battery','fluorescent']
objectness_classes = ['background','material']
condition_classes = {
    'dent': ['not_dented','dented'], 'label': ['no_label','has_label'],
    'foreign_material': ['no_foreign_material','has_foreign_material'],
}
outputs = [
    {
        'name': 'objectness', 'kind': 'logits', 'shape': ['batch', 2],
        'activation': 'softmax', 'class_names': objectness_classes,
        'class_ids': {'background': 0, 'material': 1},
        'trained_on': 'all proposal rows',
    },
    {
        'name': 'material', 'kind': 'logits', 'shape': ['batch', 9],
        'activation': 'softmax', 'class_names': material_classes,
        'class_ids': {name: index for index, name in enumerate(material_classes)},
        'valid_when': {'output': 'objectness', 'class_id': 1, 'class_name': 'material'},
        'trained_on': 'positive material rows only; background is excluded from CE',
    },
]
for name in ('dent', 'label', 'foreign_material'):
    names = condition_classes[name]
    outputs.append({
        'name': name, 'kind': 'logits', 'shape': ['batch', 2],
        'activation': 'softmax', 'class_names': names,
        'class_ids': {value: index for index, value in enumerate(names)},
        'valid_when': {'output': 'objectness', 'class_id': 1, 'label_is_present': True},
        'trained_on': 'labeled positive material rows only',
    })
output_contract = {
    'version': 'multitask_verifier.v3',
    'output_order': ['objectness','material','dent','label','foreign_material'],
    'outputs': outputs, 'material_background_class_id': None,
    'decision_order': ['objectness','material','conditions'],
    'warning': 'This v3 contract is not the legacy four-output production contract.',
}
dataset_consumption_contract = {
    'schema': 'v4_candidate_dataset_consumption.v1',
    'version': 'multitask_verifier.image_consumption.v1',
    'evidence_scope': 'per_access_fail_closed_no_complete_access_receipt',
    'authority_platform': 'linux_qnap',
    'read_semantics': 'single_descriptor_fstat_sha256_then_bytesio_decode',
    'trainer_path': 'scripts/train_multitask_verifier.py',
    'trainer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'dataset_snapshot_report_sha256': one('--dataset-snapshot-report-sha256'),
    'dataset_snapshot_tree_sha256': one('--dataset-snapshot-tree-sha256'),
    'manifest_payload_set_sha256': one('--manifest-payload-set-sha256'),
    'max_image_bytes': 67108864, 'max_image_pixels': 16777216,
    'complete_access_receipt': False,
}
if '--dry-run' in sys.argv:
    print(json.dumps({
        'ok': True, 'mode': 'dry-run', 'seed': int(one('--seed')),
        'manifest': {}, 'condition_heads': values('--condition-head'),
        'class_weights': {
            'mode': one('--class-weight-mode'),
            'beta': float(one('--class-weight-beta')), 'values': {},
        },
        'sampling': sampling, 'output_contract': output_contract,
        'dataset_consumption_contract': dataset_consumption_contract,
    }))
    raise SystemExit(0)
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper
from torchvision import models
output = Path(one('--output-dir')); output.mkdir(exist_ok=True)
manifest_summary = {
    'input_manifests': [
        {'path': Path(path).resolve().as_posix(), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        for path in manifests
    ],
    'strict': True,
    'required_lineage_fields': ['sample_id','source_sha256','object_group','capture_session','role','fold'],
    'rows': 0, 'lineage_sha256': '0' * 64,
    'payload_set_sha256': one('--manifest-payload-set-sha256'), 'role_counts': {},
    'excluded_from_training_role_counts': {'calibration': 0, 'blind_test': 0},
    'folds_by_role': {}, 'unique': {}, 'objectness_counts': {},
    'positive_material_counts': {},
}
manifest_contract = {
    'required_fields': [
        'filepath','split','source_id','material','category','dent','label',
        'foreign_material','source_object_count','sample_id','role','fold',
        'source_sha256','image_sha256','object_group','capture_session','origin',
    ],
    'lineage_fields': ['sample_id','source_sha256','object_group','capture_session','role','fold'],
    'allowed_roles': ['train','model_validation','calibration','blind_test'],
    'optimization_role': 'train', 'checkpoint_selection_role': 'model_validation',
    'excluded_roles': ['calibration','blind_test'],
}
selection_contract = {
    'metric': 'mean balanced accuracy',
    'heads': ['objectness','material','dent','label','foreign_material'],
    'requires_every_class_in_validation': True,
}
best_metrics = {}
config = {
    'effective': {
        'epochs': int(one('--epochs')), 'patience': int(one('--patience')),
        'batch': int(one('--batch')), 'workers': int(one('--workers')),
        'backbone': one('--backbone'), 'size': int(one('--size')),
        'pretrained': True, 'export_onnx': True,
        'max_train_batches': None, 'max_validation_batches': None,
    },
    'seed': int(one('--seed')), 'label_smoothing': float(one('--label-smoothing')),
    'deterministic_algorithms': True, 'smoke': False,
    'learning_rates': {
        'base': float(one('--lr')), 'backbone': float(one('--backbone-lr')),
        'heads': float(one('--head-lr')),
    },
    'class_weights': {
        'mode': one('--class-weight-mode'),
        'beta': float(one('--class-weight-beta')), 'values': {},
    },
    'task_weights': {
        'objectness': float(one('--objectness-weight')),
        'material': float(one('--material-weight')),
        'dent': float(one('--condition-weight')),
        'label': float(one('--condition-weight')),
        'foreign_material': float(one('--condition-weight')),
    },
    'sampling': sampling,
}
metadata = {
    'format_version': 3, 'architecture': 'multitask_crop_verifier', 'candidate_only': True,
    'production_runtime_modified': False,
    'checkpoint': 'best_multitask_verifier.pt', 'onnx': 'multitask_verifier.onnx',
    'model_config': {
        'backbone': one('--backbone'), 'input_size': int(one('--size')),
        'condition_heads': values('--condition-head'),
    },
    'classes': material_classes, 'material_classes': material_classes,
    'objectness_classes': objectness_classes, 'condition_classes': condition_classes,
    'output_contract': output_contract,
    'preprocessing': {
        'color_space': 'RGB', 'resize': [320, 320],
        'normalization': {
            'mean': [0.485,0.456,0.406], 'std': [0.229,0.224,0.225],
        },
    },
    'manifest_contract': manifest_contract, 'manifest_summary': manifest_summary,
    'dataset_consumption_contract': dataset_consumption_contract,
    'training_config': config, 'selection_contract': selection_contract,
    'best_epoch': 1, 'best_selection_score': 0.5, 'best_metrics': best_metrics,
}
class ExpectedMultitaskVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights=None)
        features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        self.objectness_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(features, 2))
        self.material_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(features, 9))
        self.condition_heads = nn.ModuleDict({
            name: nn.Sequential(nn.Dropout(0.2), nn.Linear(features, 2))
            for name in ('dent', 'label', 'foreign_material')
        })
    def forward(self, image):
        features = self.backbone(image)
        return {
            'objectness': self.objectness_head(features),
            'material': self.material_head(features),
            **{
                name: self.condition_heads[name](features)
                for name in ('dent', 'label', 'foreign_material')
            },
        }
mock_model = ExpectedMultitaskVerifier().eval()
torch.save(
    {
        'format_version': 3,
        'architecture': 'multitask_crop_verifier',
        'model_config': metadata['model_config'],
        'backbone': metadata['model_config']['backbone'],
        'input_size': metadata['model_config']['input_size'],
        'classes': material_classes, 'material_classes': material_classes,
        'objectness_classes': objectness_classes,
        'condition_classes': condition_classes,
        'output_contract': metadata['output_contract'],
        'preprocessing': metadata['preprocessing'],
        'manifest_contract': manifest_contract,
        'manifest_summary': manifest_summary,
        'dataset_consumption_contract': dataset_consumption_contract,
        'training_config': config,
        'selection_contract': selection_contract,
        'epoch': 1, 'selection_score': 0.5, 'metrics': best_metrics,
        'state_dict': mock_model.state_dict(),
    },
    output / 'best_multitask_verifier.pt',
)
class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        values = self.model(image)
        return tuple(values[name] for name in metadata['output_contract']['output_order'])
torch.onnx.export(
    ExportWrapper(mock_model).eval(),
    torch.zeros(1, 3, 320, 320, dtype=torch.float32),
    output / 'multitask_verifier.onnx',
    input_names=['img'],
    output_names=metadata['output_contract']['output_order'],
    dynamic_axes={
        'img': {0: 'batch'},
        **{name: {0: 'batch'} for name in metadata['output_contract']['output_order']},
    },
    opset_version=17,
    dynamo=False,
)
(output / 'multitask_verifier_metadata.json').write_text(
    json.dumps(metadata, sort_keys=True) + '\\n', encoding='utf-8'
)
""",
        encoding="utf-8",
        newline="\n",
    )
    (code_root / "scripts" / "train_verifier.py").write_text(
        "# Minimal inventoried dependency for verified-byte launcher tests.\n",
        encoding="utf-8",
        newline="\n",
    )
    auditor = code_root / authority_builder.NEAR_DUPLICATE_AUDITOR_PATH
    auditor.parent.mkdir(parents=True, exist_ok=True)
    auditor.write_bytes(
        (
            Path(__file__).resolve().parents[1]
            / authority_builder.NEAR_DUPLICATE_AUDITOR_PATH
        ).read_bytes()
    )

    rows: list[dict[str, str]] = []

    def add_row(
        role: str,
        material: int,
        suffix: str,
        *,
        origin: str = "aihub",
        captured_at: str = "",
        evidence: bool = False,
    ) -> dict[str, str]:
        source = data_root / f"source-{suffix}.jpg"
        crop = data_root / f"crop-{suffix}.jpg"
        source.write_bytes(f"source:{suffix}".encode())
        crop.write_bytes(f"crop:{suffix}".encode())
        positive = material != 9
        row = {
            "filepath": str(crop.resolve()),
            "split": "training" if role == "train" else "validation",
            "source_id": f"source-id-{suffix}",
            "material": str(material),
            "category": "background" if material == 9 else MATERIAL_CLASSES[material],
            "dent": "-1" if not positive else str(material % 2),
            "label": "-1" if not positive else str((material + 1) % 2),
            "foreign_material": "-1" if not positive else str(material % 2),
            "source_object_count": "0" if not positive else "1",
            "crop_object_count": "0" if not positive else "1",
            "sample_id": f"sample-{suffix}",
            "role": role,
            "fold": role,
            "source_sha256": _sha(source.read_bytes()),
            "image_sha256": _sha(crop.read_bytes()),
            "object_group": f"group-{suffix}",
            "capture_session": f"session-{suffix}",
            "origin": origin,
            "source_filepath": str(source.resolve()),
            "captured_at": captured_at,
            "auditor_sha256": _fake_sha(f"audit-{suffix}") if evidence else "",
            "teacher_output_sha256": _fake_sha(f"teacher-{suffix}") if evidence else "",
            "localizer_output_sha256": _fake_sha(f"localizer-{suffix}") if evidence else "",
        }
        rows.append(row)
        return row

    for role in ("train", "model_validation"):
        for material in range(10):
            add_row(role, material, f"{role}-{material}")
    bad_row = add_row("train", 5, "bad-shot")
    old_row = add_row(
        "train",
        3,
        "old-operation",
        origin="ops",
        captured_at="2026-07-31T23:59:59+09:00",
    )
    manifest = global_root / "source_manifest.csv"
    full_report = global_root / "full_report.json"
    qx3_report = _dump(
        global_root / "qx3_report.json",
        {
            "schema_version": 1,
            "artifact_role": QX3_REPORT_ROLE,
            "status": "validator_ab_exact_reproduction",
            "validated_manifest_bytes_equal": True,
            "report_core_contract_and_bindings_equal": True,
            "rows": 3498,
            "lineage_execution_authorized": False,
            "training_authority": False,
            "blind_test_authority": False,
            "production_deployment_authorized": False,
        },
    )
    qx3_ready = _dump(
        global_root / "qx3_ready.json",
        {
            "schema_version": 1,
            "artifact_role": QX3_READY_ROLE,
            "status": "batch1_validator_ab_reproducibility_passed",
            "selected_sources": 3500,
            "validated_rows": 3498,
            "selected_source_coverage": 3498 / 3500,
            "lineage_execution_authorized": False,
            "judge_authority": False,
            "training_authority": False,
            "blind_test_authority": False,
            "candidate_promotion_authorized": False,
            "production_deployment_authorized": False,
            "bindings": {"comparison_sha256": _sha(qx3_report.read_bytes())},
        },
    )
    license_path = _dump(
        global_root / "license.json",
        {
            "schema": "v4_commercial_training_license_allowlist.v1",
            "artifact_role": "commercial_training_license_evidence",
            "status": "commercial_training_allowed",
            "commercial_training_allowlist": True,
            "origins": {
                "aihub": {
                    "kind": "aihub",
                    "dataset_id": "AIHUB_71362",
                    "commercial_training_allowed": True,
                    "redistribution_allowed": False,
                    "evidence_sha256": _fake_sha("aihub-license"),
                },
                "ops": {
                    "kind": "operational",
                    "dataset_id": "pi5-operational",
                    "commercial_training_allowed": True,
                    "redistribution_allowed": False,
                    "evidence_sha256": _fake_sha("ops-license"),
                },
            },
        },
    )
    quality = _dump(
        global_root / "quality.json",
        _quality_value(
            [
                {
                    "source_sha256": bad_row["source_sha256"],
                    "reason": "severe_frame_crop",
                }
            ]
        ),
    )
    qx3_diagnostic_sources = [
        _fake_sha(f"qx3-{index}") for index in range(3500)
    ]
    protected_value = {
        "schema": "v4_candidate_protected_holdouts.v1",
        "artifact_role": "protected_holdouts_not_training_or_model_selection",
        "status": "protected_holdouts_ready",
        "qx3_diagnostic_source_sha256": qx3_diagnostic_sources,
        "qx3_validation_source_sha256": qx3_diagnostic_sources[:1000],
        "hardware41_source_sha256": [_fake_sha(f"hardware-{index}") for index in range(41)],
        "known_audit_source_sha256": [],
        "calibration_source_sha256": [],
        "blind_test_source_sha256": [],
    }
    protected = _dump(global_root / "protected.json", protected_value)
    protected_inventory = _dump(
        global_root / "protected_reference_inventory.json",
        {
            "schema": authority_builder.PROTECTED_REFERENCE_INVENTORY_SCHEMA,
            "root": "protected_objects",
            "objects": [
                {
                    "cohort": "qx3_diagnostic",
                    "view_kind": view_kind,
                    "path": f"qx3/{index:04d}-{view_kind}-{digest}.jpg",
                    "size": 1,
                    "image_sha256": (
                        digest
                        if view_kind == "source"
                        else _fake_sha(f"qx3-crop-{index}")
                    ),
                    "source_sha256": digest,
                }
                for index, digest in enumerate(qx3_diagnostic_sources)
                for view_kind in ("source", "crop")
            ]
            + [
                {
                    "cohort": "hardware41",
                    "view_kind": view_kind,
                    "path": f"hardware41/{index:04d}-{view_kind}-{digest}.jpg",
                    "size": 1,
                    "image_sha256": (
                        digest
                        if view_kind == "source"
                        else _fake_sha(f"hardware-crop-{index}")
                    ),
                    "source_sha256": digest,
                }
                for index, digest in enumerate(
                    protected_value["hardware41_source_sha256"]
                )
                for view_kind in ("source", "crop")
            ],
        },
    )
    near_duplicate_audit = global_root / "near_duplicate_audit.json"
    code_files = sorted(
        path
        for path in code_root.rglob("*")
        if path.is_file()
        and path.relative_to(code_root).as_posix() not in TRUST_ROOT_CODE_PATHS
    )
    inventory = _dump(
        global_root / "code_inventory.json",
        {
            "schema": "v4_candidate_code_inventory.v1",
            "root": str(code_root.resolve()),
            "file_count": len(code_files),
            "files": [
                {
                    "path": path.relative_to(code_root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": _sha(path.read_bytes()),
                }
                for path in code_files
            ],
        },
    )
    torch_home = global_root / "torch" / "hub" / "checkpoints"
    torch_home.mkdir(parents=True)
    backbone = torch_home / "mobilenet_v3_small-047dcff4.pth"
    backbone.write_bytes(b"frozen-pretrained-backbone")
    config = _dump(
        global_root / "training_config.json",
        {
            "schema": "v4_candidate_training_config.v2",
            "backbone": "mobilenet_v3_small",
            "pretrained": True,
            "input_size": 320,
            "epochs": 50,
            "patience": 10,
            "batch": 64,
            "workers": 2,
            "lr": 0.001,
            "backbone_lr": 0.0001,
            "head_lr": 0.001,
            "label_smoothing": 0.0,
            "class_weight_mode": "inverse",
            "class_weight_beta": 0.999,
            "objectness_weight": 1.0,
            "material_weight": 1.0,
            "condition_weight": 0.5,
            "condition_heads": ["dent", "label", "foreign_material"],
            "origin_weights": {"aihub": 1.0},
            "optimizer": "AdamW",
            "optimizer_betas": [0.9, 0.999],
            "weight_decay": 0.0001,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": 50,
            "scheduler_eta_min": 0.0,
            "sampling_mode": "shuffle_without_replacement",
            "sampling_samples_per_epoch": 10,
            "sampling_expected_fraction_by_origin": {"aihub": 1.0},
            "image_consumption_contract_version": (
                authority_builder.IMAGE_CONSUMPTION_CONTRACT_VERSION
            ),
            "image_max_bytes": authority_builder.IMAGE_CONSUMPTION_MAX_BYTES,
            "image_max_pixels": authority_builder.IMAGE_CONSUMPTION_MAX_PIXELS,
            "seed": 20260902,
        },
    )
    image_id = "sha256:" + "a" * 64
    container_id = "b" * 64
    container_name = "v4-candidate-test"
    run_root = tmp_path / "run-root"
    run_root.mkdir()
    devices = [
        "/dev/nvidia0",
        "/dev/nvidiactl",
        "/dev/nvidia-uvm",
        "/dev/nvidia-uvm-tools",
        "/dev/nvidia-modeset",
        "/dev/nvidia-caps/nvidia-cap1",
        "/dev/nvidia-caps/nvidia-cap2",
    ]
    contract_mounts = [
        {
            "source": "/share/Container",
            "destination": global_root.resolve().as_posix(),
            "read_only": True,
        },
        {
            "source": f"/share/Container/runs/{container_name}-workspace",
            "destination": run_root.resolve().as_posix(),
            "read_only": False,
        },
        *[
            {
                "source": source,
                "destination": destination,
                "read_only": True,
            }
            for source, destination in sorted(
                authority_builder.ALLOWED_QNAP_LIBRARY_MOUNTS
            )
        ],
    ]
    qnap_library_inventory = {
        "schema": authority_builder.QNAP_LIBRARY_INVENTORY_SCHEMA,
        "snapshot_max_bytes": authority_builder.QNAP_LIBRARY_SNAPSHOT_MAX_BYTES,
        "required_mapped_libraries": [
            {
                "container_root": "/qnap/cuda/lib64",
                "path": "libcudart.so.1",
            },
            {
                "container_root": "/qnap/nvidia/lib",
                "path": "libcuda.so.1",
            },
        ],
        "trees": [
            _qnap_inventory_tree(source, destination, library_name)
            for (source, destination), library_name in zip(
                sorted(authority_builder.ALLOWED_QNAP_LIBRARY_MOUNTS),
                ("libcudart", "libcuda"),
                strict=True,
            )
        ],
    }
    container_environment = {
        "RUN_ROOT": run_root.resolve().as_posix(),
        "RUN_DIR": (run_root / "candidate-run").resolve().as_posix(),
        "GLOBAL_ROOT": global_root.resolve().as_posix(),
        "CODE_ROOT": code_root.resolve().as_posix(),
        "AUTHORITY_JSON": (
            global_root / "authority" / "training_authority.json"
        ).resolve().as_posix(),
        "AUTHORITY_MARKER": (
            global_root / "authority" / "training_authority.sha256"
        ).resolve().as_posix(),
        "CODE_INVENTORY": inventory.resolve().as_posix(),
        "TRAINING_CONFIG": config.resolve().as_posix(),
        "HOST_LAUNCH_CONTRACT": (global_root / "host.json").resolve().as_posix(),
        "PRETRAINED_BACKBONE": backbone.resolve().as_posix(),
        "CONTAINER_IMAGE_ID": image_id,
    }
    clean_command = [
        "/usr/bin/env", "-i",
        f"PATH={authority_builder.CLEAN_CONTAINER_PATH}",
        "V4_CLEAN_REEXEC=1",
        *[
            f"{name}={container_environment[name]}"
            for name in sorted(authority_builder.REQUIRED_CONTAINER_ENV)
        ],
        "/bin/sh", wrapper.resolve().as_posix(),
    ]
    raw_inspect = _dump(
        global_root / "raw_inspect.json",
        [
            {
                "Id": container_id,
                "Image": image_id,
                "Name": f"/{container_name}",
                "Config": {
                    "Cmd": clean_command,
                    "Hostname": container_id[:12],
                    "Entrypoint": None,
                    "User": "",
                    "WorkingDir": "",
                    "Env": [
                        f"{name}={container_environment[name]}"
                        for name in sorted(container_environment)
                    ],
                },
                "HostConfig": {
                    "NetworkMode": "none",
                    "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
                    "ShmSize": 8589934592,
                    "Privileged": False,
                    "DeviceRequests": None,
                    "CapAdd": None,
                    "CapDrop": None,
                    "SecurityOpt": None,
                    "PidMode": "",
                    "IpcMode": "private",
                    "UTSMode": "",
                    "UsernsMode": "",
                    "Devices": [
                        {
                            "PathOnHost": device,
                            "PathInContainer": device,
                            "CgroupPermissions": "rwm",
                        }
                        for device in devices
                    ],
                },
                "Mounts": [
                    {
                        "Type": "bind",
                        "Source": mount["source"],
                        "Destination": mount["destination"],
                        "Mode": "ro" if mount["read_only"] else "rw",
                        "RW": not mount["read_only"],
                        "Propagation": "rprivate",
                    }
                    for mount in contract_mounts
                ],
            }
        ],
    )
    host = _dump(
        global_root / "host.json",
        {
            "schema": "v4_candidate_training_host_launch.v1",
            "container_id": container_id,
            "container_name": container_name,
            "container_image_id": image_id,
            "network_mode": "none",
            "restart_policy": "no",
            "shm_size_bytes": 8589934592,
            "privileged": False,
            "device_requests": None,
            "devices": devices,
            "mounts": contract_mounts,
            "command": clean_command,
            "raw_inspect_path": raw_inspect.resolve().as_posix(),
            "raw_inspect_sha256": _sha(raw_inspect.read_bytes()),
            "qnap_library_inventory": qnap_library_inventory,
        },
    )
    policy = code_root / "configs" / "v4_candidate_training_trusted_policy.json"
    policy.parent.mkdir(parents=True)
    result: dict[str, object] = locals()
    _refresh(result)
    return result


def _run(
    fixture: dict[str, object], *, output_name: str = "authority",
    preaudit_proposal_dir: Path | None = None,
) -> dict[str, object]:
    policy = fixture["policy"]
    global_root = fixture["global_root"]
    code_root = fixture["code_root"]
    assert (
        isinstance(policy, Path)
        and isinstance(global_root, Path)
        and isinstance(code_root, Path)
    )
    with (
        patch.object(authority_builder, "REPO_ROOT", code_root),
        patch.object(
            authority_builder,
            "TRUSTED_POLICY_RELATIVE_PATH",
            policy.relative_to(code_root),
        ),
        patch.object(
            authority_builder,
            "APPROVED_TRUSTED_POLICY_SHA256",
            _sha(policy.read_bytes()),
        ),
    ):
        return build_training_authority(
            source_manifests=[fixture["manifest"]],  # type: ignore[list-item]
            full_data_validator_reports=[fixture["full_report"]],  # type: ignore[list-item]
            qx3_diagnostic_ready=fixture["qx3_ready"],  # type: ignore[arg-type]
            qx3_diagnostic_report=fixture["qx3_report"],  # type: ignore[arg-type]
            trusted_policy=policy,
            license_allowlist=fixture["license_path"],  # type: ignore[arg-type]
            quality_exclusions=fixture["quality"],  # type: ignore[arg-type]
            protected_sources=fixture["protected"],  # type: ignore[arg-type]
            near_duplicate_audit_report=fixture["near_duplicate_audit"],  # type: ignore[arg-type]
            protected_reference_inventory=fixture["protected_inventory"],  # type: ignore[arg-type]
            code_inventory=fixture["inventory"],  # type: ignore[arg-type]
            pretrained_backbone=fixture["backbone"],  # type: ignore[arg-type]
            training_config=fixture["config"],  # type: ignore[arg-type]
            host_launch_contract=fixture["host"],  # type: ignore[arg-type]
            container_image_id=fixture["image_id"],  # type: ignore[arg-type]
            output_dir=global_root / output_name,
            preaudit_proposal_dir=preaudit_proposal_dir,
        )


def _build_preaudit_proposal(
    fixture: dict[str, object], *, output_name: str = "preaudit-proposal",
) -> dict[str, object]:
    global_root = fixture["global_root"]
    assert isinstance(global_root, Path)
    return authority_builder.build_preaudit_candidate_proposal(
        source_manifests=[fixture["manifest"]],  # type: ignore[list-item]
        full_data_validator_reports=[fixture["full_report"]],  # type: ignore[list-item]
        license_allowlist=fixture["license_path"],  # type: ignore[arg-type]
        quality_exclusions=fixture["quality"],  # type: ignore[arg-type]
        protected_sources=fixture["protected"],  # type: ignore[arg-type]
        training_config=fixture["config"],  # type: ignore[arg-type]
        output_dir=global_root / output_name,
    )


def test_happy_path_filters_old_and_bad_but_keeps_dented(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    authority = _run(fixture)
    output = fixture["global_root"] / "authority"  # type: ignore[operator]
    assert set(path.name for path in output.iterdir()) == {
        "train_manifest.csv",
        "model_validation_manifest.csv",
        "candidate_dataset_snapshot.json",
        "dataset_snapshot",
        "training_authority.json",
        "training_authority.sha256",
    }
    assert authority["schema"] == AUTHORITY_SCHEMA
    assert authority["artifact_role"] == AUTHORITY_ROLE
    assert authority["status"] == AUTHORITY_STATUS
    assert authority["candidate_only"] is True
    assert authority["ready_for_lineage_upgrade"] is True
    assert authority["production_runtime_modified"] is False
    assert authority["trust_root"] == {
        "method": "git_bundled_code_sha256_pin",
        "repository_relative_policy_path": "configs/v4_candidate_training_trusted_policy.json",
        "approved_policy_sha256": _sha(fixture["policy"].read_bytes()),  # type: ignore[union-attr]
        "actual_policy_sha256": _sha(fixture["policy"].read_bytes()),  # type: ignore[union-attr]
        "verified": True,
    }
    assert authority["counts"]["excluded"] == {  # type: ignore[index]
        "operational/before_2026_08_01_kst": 1,
        "quality/severe_frame_crop": 1,
    }
    train_rows = list(csv.DictReader((output / "train_manifest.csv").open(encoding="utf-8")))
    assert any(row["category"] == "can" and row["dent"] == "0" for row in train_rows)
    assert all(row["sample_id"] not in {"sample-bad-shot", "sample-old-operation"} for row in train_rows)
    marker_rows = (output / "training_authority.sha256").read_text(encoding="utf-8").splitlines()
    assert len(marker_rows) == 8
    marker_paths = set()
    for line in marker_rows:
        digest, path_text = line.split("  ", 1)
        path = Path(path_text)
        assert digest == _sha(path.read_bytes())
        marker_paths.add(path)
    expected_paths = {
        output / "training_authority.json",
        output / "train_manifest.csv",
        output / "model_validation_manifest.csv",
        output / "candidate_dataset_snapshot.json",
        fixture["inventory"],
        fixture["config"],
        fixture["host"],
        fixture["backbone"],
    }
    assert marker_paths == {Path(path).absolute() for path in expected_paths}
    content_inventory = authority["dataset_content_inventory"]
    assert isinstance(content_inventory, list)
    assert len(content_inventory) == sum(authority["counts"]["selected_by_role"].values())  # type: ignore[index,union-attr]
    assert authority["bindings"]["dataset_content_inventory_sha256"] == _sha(  # type: ignore[index]
        (
            json.dumps(
                content_inventory,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    )
    policy_value = json.loads(fixture["policy"].read_text(encoding="utf-8"))  # type: ignore[union-attr]
    for role, filename in (
        ("train", "train_manifest.csv"),
        ("model_validation", "model_validation_manifest.csv"),
    ):
        binding = f"candidate_{role}_manifest_sha256"
        emitted_sha = _sha((output / filename).read_bytes())
        assert policy_value[binding] == emitted_sha
        assert authority["bindings"][binding] == emitted_sha  # type: ignore[index]
    near_duplicate_path = fixture["near_duplicate_audit"]
    protected_inventory_path = fixture["protected_inventory"]
    assert isinstance(near_duplicate_path, Path)
    assert isinstance(protected_inventory_path, Path)
    assert authority["near_duplicate_audit"] == json.loads(
        near_duplicate_path.read_text(encoding="utf-8")
    )
    assert authority["bindings"]["candidate_near_duplicate_audit_sha256"] == _sha(  # type: ignore[index]
        near_duplicate_path.read_bytes()
    )
    assert authority["bindings"]["protected_reference_inventory_sha256"] == _sha(  # type: ignore[index]
        protected_inventory_path.read_bytes()
    )
    for entry in content_inventory:
        assert set(entry) == {
            "sample_id", "role", "source_path", "source_size", "source_sha256",
            "crop_path", "crop_size", "crop_sha256",
        }
        assert _sha(Path(entry["source_path"]).read_bytes()) == entry["source_sha256"]
        assert _sha(Path(entry["crop_path"]).read_bytes()) == entry["crop_sha256"]
        assert Path(entry["crop_path"]).is_relative_to(output / "dataset_snapshot")
        assert Path(entry["crop_path"]).stat().st_nlink == 1
        assert Path(entry["crop_path"]).stat().st_mode & 0o777 == 0o444
    snapshot_report = json.loads(
        (output / "candidate_dataset_snapshot.json").read_text(encoding="utf-8")
    )
    assert snapshot_report["schema"] == authority_builder.DATASET_SNAPSHOT_SCHEMA
    assert snapshot_report["object_count"] == len(content_inventory)
    assert snapshot_report["payload_kind"] == "training_crop_only"
    assert snapshot_report["object_max_bytes"] == (
        authority_builder.IMAGE_CONSUMPTION_MAX_BYTES
    )
    contract = authority["dataset_consumption_contract"]
    assert contract == authority_builder._dataset_consumption_contract(
        trainer_sha256=next(
            item["sha256"]
            for item in json.loads(
                fixture["inventory"].read_text(encoding="utf-8")  # type: ignore[union-attr]
            )["files"]
            if item["path"] == "scripts/train_multitask_verifier.py"
        ),
        snapshot_report_sha256=_sha(
            (output / "candidate_dataset_snapshot.json").read_bytes()
        ),
        snapshot_tree_sha256=snapshot_report["tree_sha256"],
        manifest_payload_set_sha256=snapshot_report["payload_set_sha256"],
    )
    assert authority["bindings"]["candidate_dataset_consumption_contract_sha256"] == (  # type: ignore[index]
        _sha(authority_builder._canonical_json(contract))
    )
    assert authority["bindings"]["candidate_dataset_snapshot_sha256"] == _sha(  # type: ignore[index]
        (output / "candidate_dataset_snapshot.json").read_bytes()
    )
    receipt = authority["dataset_snapshot_publish_receipt"]
    assert authority["bindings"]["dataset_snapshot_publish_receipt_sha256"] == _sha(  # type: ignore[index]
        authority_builder._canonical_json(receipt)
    )


def test_preaudit_proposal_is_non_authoritative_and_final_bytes_match(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    proposal = _build_preaudit_proposal(fixture)
    proposal_root = fixture["global_root"] / "preaudit-proposal"  # type: ignore[operator]
    assert set(path.name for path in proposal_root.iterdir()) == {
        "preaudit_proposal.json",
        "preaudit_proposal.sha256",
        "train_manifest.csv",
        "model_validation_manifest.csv",
        "candidate_dataset_snapshot.json",
        "dataset_snapshot",
    }
    assert proposal["schema"] == authority_builder.PREAUDIT_PROPOSAL_SCHEMA
    assert proposal["near_duplicate_audit_input_ready"] is True
    for field in (
        "candidate_training_input_authorized", "training_authority",
        "lineage_execution_authorized", "ready_for_lineage_upgrade",
        "blind_test_authority", "candidate_promotion_authorized",
        "production_deployment_authorized", "pi_deployment_authorized",
    ):
        assert proposal[field] is False
    assert proposal["counts"]["excluded"] == {  # type: ignore[index]
        "operational/before_2026_08_01_kst": 1,
        "quality/severe_frame_crop": 1,
    }

    authority = _run(
        fixture,
        output_name="authority-from-proposal",
        preaudit_proposal_dir=proposal_root,
    )
    authority_root = fixture["global_root"] / "authority-from-proposal"  # type: ignore[operator]
    for filename in (
        "train_manifest.csv", "model_validation_manifest.csv",
        "candidate_dataset_snapshot.json",
    ):
        assert (proposal_root / filename).read_bytes() == (
            authority_root / filename
        ).read_bytes()
    snapshot = json.loads(
        (proposal_root / "candidate_dataset_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    for item in snapshot["objects"]:
        relative = Path(item["path"])
        assert (proposal_root / relative).read_bytes() == (
            authority_root / relative
        ).read_bytes()
    assert authority["bindings"]["candidate_train_manifest_sha256"] == _sha(  # type: ignore[index]
        (proposal_root / "train_manifest.csv").read_bytes()
    )


def test_preaudit_proposal_cli_publishes_the_same_sealed_inputs(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = fixture["global_root"] / "preaudit-cli"  # type: ignore[operator]
    script = Path(authority_builder.__file__).resolve()
    result = subprocess.run(
        [
            sys.executable, os.fspath(script),
            "--mode", "preaudit-proposal",
            "--source-manifest", os.fspath(fixture["manifest"]),
            "--full-data-validator-report", os.fspath(fixture["full_report"]),
            "--license-allowlist", os.fspath(fixture["license_path"]),
            "--quality-exclusions", os.fspath(fixture["quality"]),
            "--protected-sources", os.fspath(fixture["protected"]),
            "--training-config", os.fspath(fixture["config"]),
            "--output-dir", os.fspath(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rendered = json.loads(result.stdout)
    assert rendered["training_authority"] is False
    assert (output / "preaudit_proposal.sha256").is_file()


def test_final_builder_rejects_tampered_preaudit_manifest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _build_preaudit_proposal(fixture)
    proposal_root = fixture["global_root"] / "preaudit-proposal"  # type: ignore[operator]
    train = proposal_root / "train_manifest.csv"
    train.write_bytes(train.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="differs from regenerated bytes"):
        _run(
            fixture,
            output_name="authority-tampered-proposal",
            preaudit_proposal_dir=proposal_root,
        )


def test_dataset_snapshot_rejects_an_object_above_the_consumption_cap() -> None:
    digest = _fake_sha("oversized-object")
    with pytest.raises(ValueError, match="exceeds image_max_bytes"):
        authority_builder._dataset_snapshot_report(
            [
                {
                    "sample_id": "oversized",
                    "role": "train",
                    "path": authority_builder._snapshot_relative_path(digest),
                    "size": authority_builder.IMAGE_CONSUMPTION_MAX_BYTES + 1,
                    "sha256": digest,
                }
            ]
        )


@pytest.mark.parametrize(
    "field", ("source_sha256", "image_sha256", "object_group", "capture_session")
)
def test_cross_role_leakage_fails_closed(tmp_path: Path, field: str) -> None:
    fixture = _fixture(tmp_path)
    rows = fixture["rows"]
    assert isinstance(rows, list)
    train = next(row for row in rows if row["role"] == "train" and row["material"] == "0")
    validation = next(
        row for row in rows if row["role"] == "model_validation" and row["material"] == "0"
    )
    validation[field] = train[field]
    if field == "source_sha256":
        validation["source_filepath"] = train["source_filepath"]
    if field == "image_sha256":
        validation["filepath"] = train["filepath"]
    _refresh(fixture)
    with pytest.raises(ValueError, match="leakage"):
        _run(fixture)


def test_post_cutoff_operational_is_train_only_and_aihub_has_no_cutoff(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    rows = fixture["rows"]
    assert isinstance(rows, list)
    operational = next(row for row in rows if row["role"] == "train" and row["material"] == "1")
    operational.update(
        origin="ops",
        captured_at="2026-08-01T00:00:00+09:00",
        auditor_sha256=_fake_sha("audit-current"),
        teacher_output_sha256=_fake_sha("teacher-current"),
        localizer_output_sha256=_fake_sha("localizer-current"),
    )
    aihub = next(
        row for row in rows if row["role"] == "model_validation" and row["material"] == "1"
    )
    aihub["captured_at"] = "2020-01-01T00:00:00+09:00"
    _refresh(fixture)
    authority = _run(fixture)
    assert authority["counts"]["selected_by_origin"]["ops"] == 1  # type: ignore[index]

    fixture = _fixture(tmp_path / "validation-ops")
    rows = fixture["rows"]
    assert isinstance(rows, list)
    forbidden = next(row for row in rows if row["role"] == "model_validation")
    forbidden.update(
        origin="ops",
        captured_at="2026-08-02T00:00:00+09:00",
        auditor_sha256=_fake_sha("audit-v"),
        teacher_output_sha256=_fake_sha("teacher-v"),
        localizer_output_sha256=_fake_sha("localizer-v"),
    )
    _refresh(fixture)
    with pytest.raises(ValueError, match="train-only"):
        _run(fixture)


def test_quality_reason_and_unknown_sha_fail_but_dent_is_preserved(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    quality = fixture["quality"]
    assert isinstance(quality, Path)
    bad_source = fixture["bad_row"]["source_sha256"]  # type: ignore[index]
    _dump(
        quality,
        _quality_value([{"source_sha256": bad_source, "reason": "object_dented"}]),
    )
    _refresh(fixture)
    with pytest.raises(ValueError, match="condition is not"):
        _run(fixture)

    fixture = _fixture(tmp_path / "legacy-reason")
    bad_source = fixture["bad_row"]["source_sha256"]  # type: ignore[index]
    _dump(
        fixture["quality"],  # type: ignore[arg-type]
        _quality_value(
            [{"source_sha256": bad_source, "reason": "clutter_or_multiple_objects"}]
        ),
    )
    _refresh(fixture)
    with pytest.raises(ValueError, match="condition is not"):
        _run(fixture)

    fixture = _fixture(tmp_path / "unknown")
    _dump(
        fixture["quality"],  # type: ignore[arg-type]
        _quality_value(
            [{"source_sha256": _fake_sha("not-in-data"), "reason": "severe_frame_crop"}]
        ),
    )
    _refresh(fixture)
    with pytest.raises(ValueError, match="absent from the full-data"):
        _run(fixture)


def test_quality_manifest_limit_is_exactly_100(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    quality = json.loads(fixture["quality"].read_text(encoding="utf-8"))  # type: ignore[union-attr]
    quality["max_excluded_sources"] = 101
    _dump(fixture["quality"], quality)  # type: ignore[arg-type]
    _refresh(fixture)
    with pytest.raises(ValueError, match="exactly 100"):
        _run(fixture)

    fixture = _fixture(tmp_path / "too-many")
    _dump(
        fixture["quality"],  # type: ignore[arg-type]
        _quality_value(
            [
                {
                    "source_sha256": _fake_sha(f"quality-entry-{index}"),
                    "reason": "extreme_exposure",
                }
                for index in range(101)
            ]
        ),
    )
    _refresh(fixture)
    with pytest.raises(ValueError, match="at most 100"):
        _run(fixture)


@pytest.mark.parametrize("row_name", ("bad_row", "old_row"))
def test_filtered_rows_cannot_smuggle_calibration_or_blind_roles(
    tmp_path: Path, row_name: str
) -> None:
    fixture = _fixture(tmp_path)
    row = fixture[row_name]
    assert isinstance(row, dict)
    row["role"] = "blind" if row_name == "bad_row" else "calibration"
    row["fold"] = row["role"]
    row["split"] = row["role"]
    _refresh(fixture)
    with pytest.raises(ValueError, match="role must be train or model_validation"):
        _run(fixture)


def test_protected_and_diagnostic_rows_never_enter_training(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    rows = fixture["rows"]
    assert isinstance(rows, list)
    protected_row = next(row for row in rows if row["role"] == "train" and row["material"] == "0")
    value = json.loads(fixture["protected"].read_text(encoding="utf-8"))  # type: ignore[union-attr]
    value["qx3_diagnostic_source_sha256"][0] = protected_row["source_sha256"]
    value["qx3_validation_source_sha256"][0] = protected_row["source_sha256"]
    _dump(fixture["protected"], value)  # type: ignore[arg-type]
    _refresh(fixture)
    with pytest.raises(ValueError, match="protected holdout"):
        _run(fixture)

    fixture = _fixture(tmp_path / "diagnostic")
    rows = fixture["rows"]
    assert isinstance(rows, list)
    row = next(item for item in rows if item["role"] == "train" and item["material"] == "0")
    row["origin"] = "qx3_diagnostic"
    license_value = json.loads(fixture["license_path"].read_text(encoding="utf-8"))  # type: ignore[union-attr]
    license_value["origins"]["qx3_diagnostic"] = {
        "kind": "aihub",
        "dataset_id": "AIHUB_71362",
        "commercial_training_allowed": True,
        "redistribution_allowed": False,
        "evidence_sha256": _fake_sha("diagnostic-license"),
    }
    _dump(fixture["license_path"], license_value)  # type: ignore[arg-type]
    _refresh(fixture)
    with pytest.raises(ValueError, match="diagnostic qx3"):
        _run(fixture)


def test_license_unknown_and_boolean_forgery_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    rows = fixture["rows"]
    assert isinstance(rows, list)
    rows[0]["origin"] = "unknown"
    _refresh(fixture)
    with pytest.raises(ValueError, match="commercially allowlisted"):
        _run(fixture)

    fixture = _fixture(tmp_path / "bool")
    report = json.loads(fixture["full_report"].read_text(encoding="utf-8"))  # type: ignore[union-attr]
    report["ready_for_lineage_upgrade"] = 1
    _dump(fixture["full_report"], report)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be True"):
        _run(fixture)

    fixture = _fixture(tmp_path / "provenance")
    report = json.loads(fixture["full_report"].read_text(encoding="utf-8"))  # type: ignore[union-attr]
    report["contract"]["proposal_provenance"]["runtime_detector_executed"] = False
    _dump(fixture["full_report"], report)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="runtime_detector_executed"):
        _run(fixture)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("escape", "target escapes"),
        ("cycle", "symlink cycle"),
        ("unregistered", "target is not inventoried"),
        ("bad_digest", "tree SHA mismatch"),
        ("special_type", "entry type is unsupported"),
        ("unsorted", "path-sorted"),
    ),
)
def test_qnap_library_inventory_fails_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    fixture = _fixture(tmp_path)
    host_path = fixture["host"]
    assert isinstance(host_path, Path)
    host = json.loads(host_path.read_text(encoding="utf-8"))
    tree = host["qnap_library_inventory"]["trees"][0]
    entries = tree["entries"]
    link_path = entries[0]["path"]
    file_path = entries[1]["path"]
    if mutation == "escape":
        entries[0]["target"] = "../outside.so"
    elif mutation == "cycle":
        entries[1] = {
            "path": file_path,
            "type": "symlink",
            "target": link_path,
        }
    elif mutation == "unregistered":
        entries[0]["target"] = "missing.so"
    elif mutation == "bad_digest":
        tree["tree_sha256"] = _fake_sha("forged-tree-digest")
    elif mutation == "special_type":
        entries[1] = {"path": file_path, "type": "fifo"}
    elif mutation == "unsorted":
        tree["entries"] = list(reversed(entries))
        entries = tree["entries"]
    else:  # pragma: no cover - protects the parametrization itself
        raise AssertionError(mutation)
    if mutation != "bad_digest":
        tree["tree_sha256"] = _sha(
            authority_builder._canonical_compact_json(entries)
        )
    _dump(host_path, host)
    _refresh(fixture)

    with pytest.raises(ValueError, match=message):
        _run(fixture)
    assert not (fixture["global_root"] / "authority").exists()  # type: ignore[operator]


def test_qnap_library_inventory_and_mount_roots_are_exact(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    host_path = fixture["host"]
    assert isinstance(host_path, Path)
    host = json.loads(host_path.read_text(encoding="utf-8"))
    host["qnap_library_inventory"]["trees"][0]["source_root"] = "/unapproved"
    _dump(host_path, host)
    _refresh(fixture)
    with pytest.raises(ValueError, match="source-root sorted|tree roots mismatch"):
        _run(fixture)

    fixture = _fixture(tmp_path / "tree-order")
    host_path = fixture["host"]
    assert isinstance(host_path, Path)
    host = json.loads(host_path.read_text(encoding="utf-8"))
    host["qnap_library_inventory"]["trees"].reverse()
    _dump(host_path, host)
    _refresh(fixture)
    with pytest.raises(ValueError, match="source-root sorted"):
        _run(fixture)

    fixture = _fixture(tmp_path / "missing-mount")
    host_path = fixture["host"]
    raw_path = fixture["raw_inspect"]
    assert isinstance(host_path, Path) and isinstance(raw_path, Path)
    host = json.loads(host_path.read_text(encoding="utf-8"))
    removed = host["mounts"].pop()
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw[0]["Mounts"] = [
        mount
        for mount in raw[0]["Mounts"]
        if not (
            mount["Source"] == removed["source"]
            and mount["Destination"] == removed["destination"]
        )
    ]
    _dump(raw_path, raw)
    host["raw_inspect_sha256"] = _sha(raw_path.read_bytes())
    _dump(host_path, host)
    _refresh(fixture)
    with pytest.raises(ValueError, match="both exact read-only QNAP library mounts"):
        _run(fixture)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("empty", "must be nonempty"),
        ("duplicate", "duplicate QNAP required mapped library"),
        ("unregistered", "is not inventoried"),
        ("bad_root", "root is not inventoried"),
        ("missing_libcuda", "include nvidia libcuda.so.1"),
        ("missing_cuda_tree", "at least one library from each tree"),
        ("unsorted", "must be path-sorted"),
    ),
)
def test_qnap_required_mapped_libraries_fail_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    fixture = _fixture(tmp_path)
    host_path = fixture["host"]
    assert isinstance(host_path, Path)
    host = json.loads(host_path.read_text(encoding="utf-8"))
    required = host["qnap_library_inventory"]["required_mapped_libraries"]
    if mutation == "empty":
        required.clear()
    elif mutation == "duplicate":
        required.append(dict(required[-1]))
    elif mutation == "unregistered":
        required[-1]["path"] = "missing-libcuda.so.1"
    elif mutation == "bad_root":
        required[-1]["container_root"] = "/qnap/unapproved"
    elif mutation == "missing_libcuda":
        required.pop()
    elif mutation == "missing_cuda_tree":
        required.pop(0)
    elif mutation == "unsorted":
        required.reverse()
    else:  # pragma: no cover - protects the parametrization itself
        raise AssertionError(mutation)
    _dump(host_path, host)
    _refresh(fixture)

    with pytest.raises(ValueError, match=message):
        _run(fixture)


def test_qnap_inventory_mutation_is_transitively_policy_bound(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    host_path = fixture["host"]
    assert isinstance(host_path, Path)
    host = json.loads(host_path.read_text(encoding="utf-8"))
    tree = host["qnap_library_inventory"]["trees"][0]
    file_entry = next(entry for entry in tree["entries"] if entry["type"] == "file")
    file_entry["sha256"] = _fake_sha("reviewed-but-not-policy-resealed-library")
    tree["tree_sha256"] = _sha(
        authority_builder._canonical_compact_json(tree["entries"])
    )
    _dump(host_path, host)

    with pytest.raises(
        ValueError, match="host_launch_contract_sha256 binding mismatch"
    ):
        _run(fixture)


def test_trusted_policy_is_repository_pinned_and_unconfigured_by_default(tmp_path: Path) -> None:
    expected = tmp_path / "configs" / "trusted.json"
    expected.parent.mkdir()
    expected.write_bytes(b"{}\n")
    digest = _sha(expected.read_bytes())
    with (
        patch.object(authority_builder, "REPO_ROOT", tmp_path),
        patch.object(
            authority_builder,
            "TRUSTED_POLICY_RELATIVE_PATH",
            Path("configs/trusted.json"),
        ),
    ):
        with pytest.raises(ValueError, match="UNCONFIGURED"):
            authority_builder._audit_trusted_policy_trust_root(expected, digest)
        with patch.object(
            authority_builder, "APPROVED_TRUSTED_POLICY_SHA256", digest
        ):
            other = tmp_path / "caller-policy.json"
            other.write_bytes(expected.read_bytes())
            with pytest.raises(ValueError, match="repository-pinned"):
                authority_builder._audit_trusted_policy_trust_root(other, digest)


def test_tamper_duplicate_json_symlink_and_reuse_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    row = fixture["rows"][0]  # type: ignore[index]
    Path(row["filepath"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="content hash mismatch"):
        _run(fixture)

    fixture = _fixture(tmp_path / "duplicate")
    fixture["license_path"].write_text(  # type: ignore[union-attr]
        '{"schema":"x","schema":"x"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _run(fixture)

    fixture = _fixture(tmp_path / "symlink")
    link = fixture["global_root"] / "license-link.json"  # type: ignore[operator]
    try:
        os.symlink(fixture["license_path"], link)  # type: ignore[arg-type]
    except OSError:
        pytest.skip("symlink creation is unavailable")
    fixture["license_path"] = link
    with pytest.raises(ValueError, match="symlink"):
        _run(fixture)

    fixture = _fixture(tmp_path / "reuse")
    _run(fixture)
    with pytest.raises(FileExistsError, match="reuse"):
        _run(fixture)


def test_authority_matches_strict_launcher_schema(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    authority = _run(fixture)
    for field in (
        "candidate_only",
        "candidate_training_input_authorized",
        "training_authority",
        "lineage_execution_authorized",
        "ready_for_lineage_upgrade",
    ):
        assert authority[field] is True
    for field in (
        "diagnostic_only",
        "production_runtime_modified",
        "blind_test_authority",
        "candidate_promotion_authorized",
        "production_deployment_authorized",
        "pi_deployment_authorized",
        "spring_contract_modified",
    ):
        assert authority[field] is False
    artifacts = authority["artifacts"]
    assert isinstance(artifacts, dict)
    assert set(artifacts) == {
        "manifests",
        "dataset_snapshot_report",
        "code_inventory",
        "training_config",
        "host_launch_contract",
        "pretrained_backbone",
    }
    manifests = artifacts["manifests"]
    assert isinstance(manifests, list)
    assert {tuple(sorted(item)) for item in manifests} == {
        ("path", "role", "sha256")
    }
    assert {item["role"] for item in manifests} == {"train", "model_validation"}
    for name in (
        "code_inventory",
        "dataset_snapshot_report",
        "training_config",
        "host_launch_contract",
        "pretrained_backbone",
    ):
        assert set(artifacts[name]) == {"path", "sha256"}
    assert authority["bindings"]["container_image_id"] == fixture["image_id"]  # type: ignore[index]


def test_pinned_policy_rejects_different_sanitized_manifest_hash(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    policy = fixture["policy"]
    assert isinstance(policy, Path)
    value = json.loads(policy.read_text(encoding="utf-8"))
    value["candidate_train_manifest_sha256"] = _fake_sha("different-train-manifest")
    _dump(policy, value)
    with pytest.raises(ValueError, match="candidate_train_manifest_sha256 binding mismatch"):
        _run(fixture)
    assert not (fixture["global_root"] / "authority").exists()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("candidate_manifest", "candidate manifest binding mismatch"),
        ("protected_sources", "protected-source binding mismatch"),
        ("auditor", "auditor binding mismatch"),
        ("auditor_runtime", "auditor binding mismatch"),
        ("coverage", "coverage is incomplete"),
        ("candidate_entry", "candidate entries differ from exact manifest rows"),
        ("protected_entry", "protected entries differ from exact inventory rows"),
        ("blocked", "did not pass"),
        ("authority", "separation evidence only"),
        ("omitted_cross_role_edge", "edge set is incomplete"),
    ),
)
def test_near_duplicate_audit_fails_closed(
    tmp_path: Path, mutation: str, message: str,
) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["near_duplicate_audit"]
    assert isinstance(path, Path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "candidate_manifest":
        value["bindings"]["candidate_manifest_sha256"]["train"] = _fake_sha(
            "stale-candidate-manifest"
        )
    elif mutation == "protected_sources":
        value["bindings"]["protected_sources"]["canonical_union_sha256"] = _fake_sha(
            "stale-protected-union"
        )
    elif mutation == "auditor":
        value["bindings"]["auditor"]["sha256"] = _fake_sha("different-auditor")
    elif mutation == "auditor_runtime":
        value["bindings"]["auditor"]["runtime_code_sha256"] = _fake_sha(
            "different-runtime-auditor"
        )
    elif mutation == "coverage":
        value["coverage"]["complete"] = False
    elif mutation == "candidate_entry":
        entry = next(
            item
            for item in value["entries"]
            if item["cohort"] == "candidate" and item["view_kind"] == "crop"
        )
        entry["sample_id"] = str(entry["sample_id"]) + "-tampered"
        entry["asset_id"] = _sha(authority_builder._canonical_compact_payload({
            "cohort": entry["cohort"],
            "image_sha256": entry["image_sha256"],
            "role": entry["role"],
            "sample_id": entry["sample_id"],
            "source_sha256": entry["source_sha256"],
            "view_kind": entry["view_kind"],
        }))
    elif mutation == "protected_entry":
        entry = next(
            item
            for item in value["entries"]
            if item["cohort"] != "candidate" and item["view_kind"] == "crop"
        )
        entry["image_sha256"] = _fake_sha("tampered-protected-crop")
        entry["asset_id"] = _sha(authority_builder._canonical_compact_payload({
            "cohort": entry["cohort"],
            "image_sha256": entry["image_sha256"],
            "role": entry["role"],
            "sample_id": entry["sample_id"],
            "source_sha256": entry["source_sha256"],
            "view_kind": entry["view_kind"],
        }))
    elif mutation == "blocked":
        value["status"] = "blocked"
        value["ok"] = False
    elif mutation == "omitted_cross_role_edge":
        train_entry = next(
            entry for entry in value["entries"] if entry["role"] == "train"
        )
        validation_entry = next(
            entry
            for entry in value["entries"]
            if entry["role"] == "model_validation"
        )
        validation_entry["phash_rot4"] = list(train_entry["phash_rot4"])
    else:
        value["authority"]["automatic_delete_or_relabel"] = True
    _dump_compact(path, value)
    with pytest.raises(ValueError, match=message):
        _run(fixture)
    assert not (fixture["global_root"] / "authority").exists()  # type: ignore[operator]


def test_near_duplicate_inventory_and_policy_hashes_are_pinned(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    inventory = fixture["protected_inventory"]
    assert isinstance(inventory, Path)
    inventory.write_bytes(inventory.read_bytes() + b" \n")
    with pytest.raises(ValueError, match="protected-inventory binding mismatch"):
        _run(fixture)

    fixture = _fixture(tmp_path / "policy")
    policy = fixture["policy"]
    assert isinstance(policy, Path)
    value = json.loads(policy.read_text(encoding="utf-8"))
    value["candidate_near_duplicate_audit_sha256"] = _fake_sha("different-audit")
    _dump(policy, value)
    with pytest.raises(
        ValueError, match="candidate_near_duplicate_audit_sha256 binding mismatch"
    ):
        _run(fixture)


def test_near_duplicate_protected_inventory_requires_source_and_crop(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    inventory = fixture["protected_inventory"]
    assert isinstance(inventory, Path)
    value = json.loads(inventory.read_text(encoding="utf-8"))
    value["objects"] = [
        item
        for item in value["objects"]
        if not (
            item["view_kind"] == "crop"
            and item["source_sha256"]
            == fixture["qx3_diagnostic_sources"][0]  # type: ignore[index]
        )
    ]
    _dump(inventory, value)
    with pytest.raises(ValueError, match="exactly one source and one crop"):
        _run(fixture)


def test_producer_requires_private_ipc_namespace(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    raw_path = fixture["raw_inspect"]
    host_path = fixture["host"]
    assert isinstance(raw_path, Path) and isinstance(host_path, Path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw[0]["HostConfig"]["IpcMode"] = ""
    _dump(raw_path, raw)
    host = json.loads(host_path.read_text(encoding="utf-8"))
    host["raw_inspect_sha256"] = _sha(raw_path.read_bytes())
    _dump(host_path, host)
    _refresh(fixture)

    with pytest.raises(ValueError, match="IpcMode must be private"):
        _run(fixture)


def test_producer_rejects_hash_bound_exported_bash_function_env(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    raw_path = fixture["raw_inspect"]
    host_path = fixture["host"]
    assert isinstance(raw_path, Path) and isinstance(host_path, Path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw[0]["Config"]["Env"].append("BASH_FUNC_sha256sum%%=() { :; }")
    _dump(raw_path, raw)
    host = json.loads(host_path.read_text(encoding="utf-8"))
    host["raw_inspect_sha256"] = _sha(raw_path.read_bytes())
    _dump(host_path, host)
    _refresh(fixture)

    with pytest.raises(ValueError, match="forbidden injection variables"):
        _run(fixture)
    assert not (fixture["global_root"] / "authority").exists()  # type: ignore[operator]


def _integration_bash(tmp_path: Path) -> str:
    candidates = [
        Path("C:/Program Files/Git/bin/bash.exe"),
        Path(shutil.which("bash") or ""),
    ]
    for candidate in candidates:
        if candidate.is_file() and subprocess.run(
            [str(candidate), "-c", 'test -d "$1"', "bash", tmp_path.as_posix()],
            check=False,
        ).returncode == 0:
            return str(candidate)
    pytest.skip("no bash can access the pytest temp path")


def _prepare_test_launcher(fixture: dict[str, object]) -> tuple[Path, dict[str, str], Path]:
    wrapper = fixture["wrapper"]
    policy = fixture["policy"]
    global_root = fixture["global_root"]
    run_root = fixture["run_root"]
    assert all(isinstance(path, Path) for path in (wrapper, policy, global_root, run_root))
    launcher_text = wrapper.read_text(encoding="utf-8")
    unconfigured = 'APPROVED_TRUSTED_POLICY_SHA256="UNCONFIGURED"'
    assert launcher_text.count(unconfigured) == 1
    wrapper.write_text(
        launcher_text.replace(
            unconfigured,
            f'APPROVED_TRUSTED_POLICY_SHA256="{_sha(policy.read_bytes())}"',
        ).replace(
            "PYTHON_BIN=/usr/local/bin/python3",
            f"PYTHON_BIN='{Path(sys.executable).as_posix()}'",
        ).replace(
            "PYTHONNOUSERSITE=1",
            "PYTHONNOUSERSITE=0",
        ),
        encoding="utf-8",
        newline="\n",
    )
    output = global_root / "authority"
    run_dir = run_root / "candidate-run"
    env = {
        **os.environ,
        "RUN_ROOT": run_root.resolve().as_posix(),
        "RUN_DIR": run_dir.resolve().as_posix(),
        "GLOBAL_ROOT": global_root.resolve().as_posix(),
        "CODE_ROOT": fixture["code_root"].resolve().as_posix(),  # type: ignore[union-attr]
        "AUTHORITY_JSON": (output / "training_authority.json").as_posix(),
        "AUTHORITY_MARKER": (output / "training_authority.sha256").as_posix(),
        "CODE_INVENTORY": fixture["inventory"].as_posix(),  # type: ignore[union-attr]
        "TRAINING_CONFIG": fixture["config"].as_posix(),  # type: ignore[union-attr]
        "HOST_LAUNCH_CONTRACT": fixture["host"].as_posix(),  # type: ignore[union-attr]
        "PRETRAINED_BACKBONE": fixture["backbone"].as_posix(),  # type: ignore[union-attr]
        "CONTAINER_IMAGE_ID": fixture["image_id"],  # type: ignore[dict-item]
        "V4_CLEAN_REEXEC": "1",
    }
    return wrapper, env, run_dir


def _reseal_authority_marker(output: Path) -> None:
    marker = output / "training_authority.sha256"
    paths = [Path(line.split("  ", 1)[1]) for line in marker.read_text().splitlines()]
    marker.write_text(
        "".join(f"{_sha(path.read_bytes())}  {path.as_posix()}\n" for path in paths),
        encoding="utf-8",
        newline="\n",
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "extra_authority_field",
        "forged_train_manifest",
        "forged_snapshot_report",
        "forged_consumption_contract",
    ),
)
def test_launcher_rejects_consistently_rehashed_authority_forgery(
    tmp_path: Path, mutation: str
) -> None:
    bash = _integration_bash(tmp_path)
    fixture = _fixture(tmp_path)
    _run(fixture)
    output = fixture["global_root"] / "authority"  # type: ignore[operator]
    authority_path = output / "training_authority.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    if mutation == "extra_authority_field":
        authority["unexpected"] = True
    elif mutation == "forged_train_manifest":
        train_path = output / "train_manifest.csv"
        rows = list(csv.DictReader(train_path.open(encoding="utf-8")))
        rows[0]["captured_at"] = "2026-08-02T00:00:00+09:00"
        train_path.write_bytes(authority_builder._render_manifest(rows))
        digest = _sha(train_path.read_bytes())
        manifest = next(
            item for item in authority["artifacts"]["manifests"]
            if item["role"] == "train"
        )
        manifest["sha256"] = digest
        authority["bindings"]["candidate_train_manifest_sha256"] = digest
    elif mutation == "forged_snapshot_report":
        report_path = output / "candidate_dataset_snapshot.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["snapshot_max_bytes"] += 1
        report_path.write_bytes(authority_builder._canonical_json(report))
        digest = _sha(report_path.read_bytes())
        authority["artifacts"]["dataset_snapshot_report"]["sha256"] = digest
        authority["bindings"]["candidate_dataset_snapshot_sha256"] = digest
    else:
        authority["dataset_consumption_contract"]["max_image_pixels"] += 1
        authority["bindings"]["dataset_consumption_contract_sha256"] = _sha(
            authority_builder._canonical_json(
                authority["dataset_consumption_contract"]
            )
        )
    authority_path.write_bytes(authority_builder._canonical_json(authority))
    _reseal_authority_marker(output)
    wrapper, env, run_dir = _prepare_test_launcher(fixture)
    result = subprocess.run(
        [bash, wrapper.as_posix()], env=env, text=True,
        capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert (run_dir / "control" / "failed.txt").is_file()
    assert not (run_dir / "control" / "candidate_training_ready.json").exists()


def test_producer_output_runs_through_launcher_candidate_only(tmp_path: Path) -> None:
    bash = _integration_bash(tmp_path)
    fixture = _fixture(tmp_path)
    authority = _run(fixture)
    wrapper, env, run_dir = _prepare_test_launcher(fixture)
    artifacts = authority["artifacts"]
    assert isinstance(artifacts, dict)
    result = subprocess.run(
        [bash, wrapper.as_posix()],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    control = run_dir / "control"
    preflight = json.loads((control / "preflight.json").read_text())
    assert set(preflight["runtime_dependency_contract"]) == {
        "onnx", "onnxruntime", "onnxruntime_providers", "torch", "torchvision",
    }
    assert "CPUExecutionProvider" in preflight["runtime_dependency_contract"][
        "onnxruntime_providers"
    ]
    assert preflight["cuda_runtime_contract"]["required"] is False
    trainer_path = fixture["trainer"]
    assert isinstance(trainer_path, Path)
    assert preflight["trainer_sha256"] == _sha(trainer_path.read_bytes())
    ready = json.loads((control / "candidate_training_ready.json").read_text())
    assert ready["dataset_consumption_contract"] == preflight[
        "dataset_consumption_contract"
    ]
    assert ready["bindings"]["dataset_consumption_contract_sha256"] == _sha(
        authority_builder._canonical_json(preflight["dataset_consumption_contract"])
    )
    assert ready["candidate_only"] is True
    assert ready["bindings"]["trainer_sha256"] == preflight["trainer_sha256"]
    assert ready["requires_independent_blind_hardware_gate"] is True
    assert ready["production_deployment_authorized"] is False
    assert not (control / "failed.txt").exists()


def test_dataset_snapshot_is_content_addressed_and_isolated_from_live_crop(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    authority = _run(fixture)
    output = fixture["global_root"] / "authority"  # type: ignore[operator]
    manifest_rows = list(
        csv.DictReader((output / "train_manifest.csv").open(encoding="utf-8"))
    )
    assert manifest_rows
    for row in manifest_rows:
        assert not Path(row["filepath"]).is_absolute()
        assert row["filepath"] == authority_builder._snapshot_relative_path(
            row["image_sha256"]
        )
        assert _sha((output / row["filepath"]).read_bytes()) == row["image_sha256"]
    from scripts.train_multitask_verifier import read_manifests

    trainer_rows = read_manifests(
        [output / "train_manifest.csv", output / "model_validation_manifest.csv"]
    )
    assert trainer_rows
    assert all(
        item.path.is_relative_to(output / "dataset_snapshot")
        for item in trainer_rows
    )

    original = fixture["rows"][0]  # type: ignore[index]
    snapshot_row = next(
        row for row in authority["dataset_content_inventory"]  # type: ignore[index]
        if row["sample_id"] == original["sample_id"]
    )
    snapshot_path = Path(snapshot_row["crop_path"])
    frozen = snapshot_path.read_bytes()
    Path(original["filepath"]).write_bytes(b"live source changed after publication")
    assert snapshot_path.read_bytes() == frozen
    assert _sha(frozen) == snapshot_row["crop_sha256"]


@pytest.mark.parametrize("duplicate_kind", ("path", "sha"))
def test_dataset_snapshot_rejects_duplicate_crop_path_or_sha(
    tmp_path: Path, duplicate_kind: str
) -> None:
    fixture = _fixture(tmp_path)
    rows = fixture["rows"]
    assert isinstance(rows, list)
    first, second = rows[0], rows[1]
    if duplicate_kind == "path":
        second["filepath"] = first["filepath"]
    else:
        Path(second["filepath"]).write_bytes(Path(first["filepath"]).read_bytes())
    second["image_sha256"] = first["image_sha256"]
    _refresh(fixture)
    with pytest.raises(ValueError, match="duplicate selected crop"):
        _run(fixture)
    assert not (fixture["global_root"] / "authority").exists()  # type: ignore[operator]


def test_dataset_snapshot_rejects_hardlink_crop_alias(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    rows = fixture["rows"]
    assert isinstance(rows, list)
    first, second = rows[0], rows[1]
    first_path = Path(first["filepath"])
    second_path = Path(second["filepath"])
    second_path.unlink()
    try:
        os.link(first_path, second_path)
    except OSError:
        pytest.skip("hardlink creation is unavailable")
    second["image_sha256"] = first["image_sha256"]
    _refresh(fixture)
    with pytest.raises(ValueError, match="hardlink aliases are forbidden"):
        _run(fixture)
    assert not (fixture["global_root"] / "authority").exists()  # type: ignore[operator]


@pytest.mark.parametrize("alias_kind", ("leaf", "ancestor"))
def test_dataset_snapshot_rejects_crop_symlink_leaf_or_ancestor(
    tmp_path: Path, alias_kind: str
) -> None:
    fixture = _fixture(tmp_path)
    row = fixture["rows"][0]  # type: ignore[index]
    crop = Path(row["filepath"])
    if alias_kind == "leaf":
        real = crop.with_name("real-crop.jpg")
        crop.replace(real)
        try:
            os.symlink(real, crop)
        except OSError:
            pytest.skip("symlink creation is unavailable")
    else:
        alias = crop.parent.parent / "data-alias"
        try:
            os.symlink(crop.parent, alias, target_is_directory=True)
        except OSError:
            pytest.skip("directory symlink creation is unavailable")
        row["filepath"] = str((alias / crop.name).absolute())
    _refresh(fixture)
    with pytest.raises(ValueError, match="symlink"):
        _run(fixture)
    assert not (fixture["global_root"] / "authority").exists()  # type: ignore[operator]


def test_dataset_snapshot_copy_detects_poison_then_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    row = fixture["rows"][0]  # type: ignore[index]
    target = Path(row["filepath"])
    original = (b"A" * (1024 * 1024)) + (b"B" * (1024 * 1024)) + b"tail"
    target.write_bytes(original)
    row["image_sha256"] = _sha(original)
    _refresh(fixture)
    original_open = Path.open
    target_key = target.resolve()
    matching_opens = 0

    class PoisoningReader:
        def __init__(self, handle) -> None:
            self.handle = handle
            self.reads = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            self.handle.close()

        def fileno(self) -> int:
            return self.handle.fileno()

        def read(self, size: int = -1) -> bytes:
            self.reads += 1
            if self.reads != 2:
                return self.handle.read(size)
            before = target.stat()
            try:
                with original_open(target, "r+b") as mutable:
                    mutable.seek(1024 * 1024)
                    mutable.write(b"Z" * (1024 * 1024))
                    mutable.flush()
                    os.fsync(mutable.fileno())
                poisoned = self.handle.read(size)
                with original_open(target, "r+b") as mutable:
                    mutable.seek(0)
                    mutable.write(original)
                    mutable.truncate()
                    mutable.flush()
                    os.fsync(mutable.fileno())
                os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
                return poisoned
            except PermissionError:
                pytest.skip("filesystem does not permit deterministic concurrent mutation")

    def hooked_open(self: Path, *args, **kwargs):
        nonlocal matching_opens
        handle = original_open(self, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        if self.resolve() == target_key and mode == "rb":
            matching_opens += 1
            if matching_opens == 3:
                return PoisoningReader(handle)
        return handle

    monkeypatch.setattr(Path, "open", hooked_open)
    with pytest.raises(RuntimeError, match="changed during copy|approved SHA"):
        _run(fixture)
    global_root = fixture["global_root"]
    assert not (global_root / "authority").exists()  # type: ignore[operator]
    assert not list(global_root.glob(".authority.*"))  # type: ignore[union-attr]


def test_dataset_snapshot_publish_failure_is_atomic(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with patch.object(
        authority_builder,
        "_publish_directory_no_replace",
        side_effect=OSError("injected rename failure"),
    ):
        with pytest.raises(OSError, match="injected rename failure"):
            _run(fixture)
    global_root = fixture["global_root"]
    assert not (global_root / "authority").exists()  # type: ignore[operator]
    assert not list(global_root.glob(".authority.*"))  # type: ignore[union-attr]


def test_dataset_snapshot_publish_rejects_destination_race(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    global_root = fixture["global_root"]
    final = global_root / "authority"  # type: ignore[operator]
    original = authority_builder._publish_directory_no_replace
    collision_inode: list[int] = []

    def collide(stage: Path, destination: Path) -> None:
        assert destination == final
        destination.mkdir()
        collision_inode.append(destination.stat().st_ino)
        original(stage, destination)

    with patch.object(authority_builder, "_publish_directory_no_replace", collide):
        with pytest.raises(OSError):
            _run(fixture)
    assert collision_inode
    assert final.is_dir()
    assert final.stat().st_ino == collision_inode[0]
    assert not list(final.iterdir())
    assert not list(global_root.glob(".authority.*"))  # type: ignore[union-attr]


def test_post_publish_failure_does_not_emit_completion_seal(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    original = authority_builder._verify_dataset_content_inventory

    def fail_post_publish(
        entries: object, description: str, **kwargs: object
    ) -> str:
        if description == "dataset content inventory post-publish":
            raise RuntimeError("injected post-publish failure")
        return original(entries, description, **kwargs)  # type: ignore[arg-type]

    with patch.object(
        authority_builder,
        "_verify_dataset_content_inventory",
        fail_post_publish,
    ):
        with pytest.raises(RuntimeError, match="injected post-publish failure"):
            _run(fixture)
    output = fixture["global_root"] / "authority"  # type: ignore[operator]
    assert output.is_dir()
    assert not (output / "training_authority.sha256").exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO requires POSIX")
def test_dataset_snapshot_tree_rejects_special_entries(tmp_path: Path) -> None:
    snapshot = tmp_path / "dataset_snapshot"
    snapshot.mkdir()
    os.mkfifo(snapshot / "extra")
    if os.name != "nt":
        snapshot.chmod(0o555)
    report = {
        "objects": [],
        "total_regular_bytes": 0,
        "tree_sha256": _sha(authority_builder._canonical_compact_json([])),
    }
    with pytest.raises(ValueError, match="regular files or directories"):
        authority_builder._dataset_snapshot_tree_contract(
            snapshot, report, logical_root=snapshot
        )


@pytest.mark.parametrize("mutation", ("mode", "inode"))
def test_launcher_rejects_dataset_snapshot_mode_or_inode_change(
    tmp_path: Path, mutation: str
) -> None:
    bash = _integration_bash(tmp_path)
    fixture = _fixture(tmp_path)
    authority = _run(fixture)
    snapshot = Path(authority["dataset_content_inventory"][0]["crop_path"])  # type: ignore[index]
    content = snapshot.read_bytes()
    os.chmod(snapshot, 0o644)
    if mutation == "inode":
        snapshot.unlink()
        snapshot.write_bytes(content)
        os.chmod(snapshot, 0o444)
    wrapper, env, run_dir = _prepare_test_launcher(fixture)
    result = subprocess.run(
        [bash, wrapper.as_posix()], env=env, text=True,
        capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert (run_dir / "control" / "failed.txt").is_file()
    assert not (run_dir / "control" / "candidate_training_ready.json").exists()
