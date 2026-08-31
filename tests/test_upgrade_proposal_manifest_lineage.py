from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import upgrade_proposal_manifest_lineage as lineage_upgrade
from scripts.upgrade_proposal_manifest_lineage import (
    UpgradeRejected,
    _apply_path_remaps,
    parse_path_remap,
    upgrade_proposal_manifests as _upgrade_proposal_manifests,
)


FIELDS = (
    "filepath",
    "split",
    "source_id",
    "material",
    "category",
    "dent",
    "label",
    "foreign_material",
    "source_object_count",
    "crop_object_count",
    "source_path_b64",
    "source_sha256",
    "image_sha256",
    "proposal_index",
    "predicted_confidence",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
    "source_width",
    "source_height",
    "custom_note",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _b64(value: str | Path) -> str:
    return base64.urlsafe_b64encode(os.fsencode(str(value))).decode("ascii")


def _row(
    *,
    filepath: str,
    source: str | Path,
    split: str = "training",
    source_id: str = "legacy-source-1",
    material: int = 0,
    category: str = "can",
    custom_note: str = "preserve-me",
) -> dict[str, object]:
    return {
        "filepath": filepath,
        "split": split,
        "source_id": source_id,
        "material": material,
        "category": category,
        "dent": -1,
        "label": -1,
        "foreign_material": -1,
        "source_object_count": 1,
        "crop_object_count": 0 if material == 9 else 1,
        "source_path_b64": _b64(source),
        "source_sha256": "",
        "image_sha256": "",
        "proposal_index": 0,
        "predicted_confidence": "0.875",
        "crop_x1": "",
        "crop_y1": "",
        "crop_x2": "",
        "crop_y2": "",
        "source_width": "",
        "source_height": "",
        "custom_note": custom_note,
    }


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _outputs(root: Path, suffix: str = "") -> dict[str, Path]:
    return {
        "output_csv": root / f"strict{suffix}.csv",
        "output_jsonl": root / f"strict{suffix}.jsonl",
        "lineage_path": root / f"lineage{suffix}.json",
        "rejections_path": root / f"rejections{suffix}.json",
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _validator_report(path: Path, manifest: Path) -> tuple[Path, str]:
    manifest_sha = _sha(manifest)
    report = {
        "schema_version": 1,
        "artifact_role": "v4_development_candidates_not_blind_or_deployment_authority",
        "ready_for_lineage_upgrade": True,
        "blind_test_eligible": False,
        "production_deployment_authorized": False,
        "rows": len(_read_csv(manifest)),
        "contract": {
            "manifest_schema_version": "proposal_verifier.v4.bgfix.v1",
            "background_policy": "strict-zero-intersection",
            "background_gt_margin": 0.10,
            "explicit_label_file_required": True,
            "source_object_count_semantics": "complete_source_frame",
            "crop_object_count_semantics": "final_padded_verifier_crop",
            "visual_judge_still_required": True,
            "proposal_provenance": {
                "detector_artifact_bytes_bound": True,
                "inference_spec_bytes_bound": True,
                "dataset_info_bytes_bound": True,
                "source_bbox_crop_bytes_recomputed": True,
                "production_or_blind_authority": False,
            },
        },
        "bindings": {
            "input_manifest_sha256": "1" * 64,
            "dataset_info_sha256": "2" * 64,
            "detector_model_sha256": "3" * 64,
            "inference_spec_sha256": "4" * 64,
            "validated_manifest_sha256": manifest_sha,
        },
    }
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return path, _sha(path)


def _automatic_group_map(path: Path, inputs: list[Path]) -> tuple[Path, str]:
    groups: dict[str, dict[str, str]] = {}
    for manifest in inputs:
        for row in _read_csv(manifest):
            try:
                decoded = os.fsdecode(base64.urlsafe_b64decode(row["source_path_b64"]))
            except Exception:
                continue
            identity = hashlib.sha256(decoded.encode()).hexdigest()[:16]
            groups[decoded] = {
                "object_group": f"test-object-{identity}",
                "capture_session": f"test-session-{identity}",
            }
    path.write_text(json.dumps(groups, sort_keys=True), encoding="utf-8")
    return path, _sha(path)


def upgrade_proposal_manifests(*, inputs, **kwargs):
    inputs = [Path(path) for path in inputs]
    reports = []
    pins = []
    for index, manifest in enumerate(inputs):
        report, pin = _validator_report(
            manifest.parent / f"validator-{manifest.stem}-{index}.json", manifest
        )
        reports.append(report)
        pins.append(pin)
    kwargs.setdefault("validator_report_paths", reports)
    kwargs.setdefault("validator_report_sha256s", pins)
    if kwargs.get("group_map_path") is not None:
        kwargs.setdefault("group_map_sha256", _sha(Path(kwargs["group_map_path"])))
    elif "quarantine_validation_near_phash_distance" not in kwargs:
        group_map, pin = _automatic_group_map(
            inputs[0].parent / f"automatic-groups-{hashlib.sha256(str(inputs).encode()).hexdigest()[:12]}.json",
            inputs,
        )
        kwargs["group_map_path"] = group_map
        kwargs["group_map_sha256"] = pin
    return _upgrade_proposal_manifests(inputs=inputs, **kwargs)


def test_recomputes_hashes_preserves_legacy_fields_and_marks_trusted_group(tmp_path):
    source = tmp_path / "sources" / "source.jpg"
    crop = tmp_path / "training" / "can" / "crop.jpg"
    source.parent.mkdir()
    crop.parent.mkdir(parents=True)
    source.write_bytes(b"source-image-bytes")
    crop.write_bytes(b"crop-image-bytes")
    manifest = tmp_path / "legacy.csv"
    _write_manifest(
        manifest,
        [_row(filepath="training/can/crop.jpg", source="sources/source.jpg")],
    )

    report = upgrade_proposal_manifests(inputs=[manifest], **_outputs(tmp_path))

    rows = _read_csv(tmp_path / "strict.csv")
    assert len(rows) == 1
    row = rows[0]
    assert row["source_sha256"] == _sha(source)
    assert row["image_sha256"] == _sha(crop)
    assert row["content_identity"] == f"sha256:{_sha(crop)}"
    assert row["sample_id"].startswith("proposal_")
    assert row["role"] == row["split_role"] == "train"
    assert row["fold"] == "train"
    assert row["object_group"].startswith("test-object-")
    assert row["capture_session"].startswith("test-session-")
    assert row["selection_reason"] == "trusted_group_map"
    assert row["custom_note"] == "preserve-me"
    assert row["legacy_filepath"] == "training/can/crop.jpg"
    assert Path(row["filepath"]) == crop.resolve()
    jsonl_row = json.loads((tmp_path / "strict.jsonl").read_text(encoding="utf-8"))
    assert jsonl_row == row
    assert json.loads((tmp_path / "rejections.json").read_text(encoding="utf-8"))[
        "rejected_count"
    ] == 0
    assert report["blind_test_eligible"] is False
    assert report["selection_reason_counts"] == {"trusted_group_map": 1}
    assert report["blind_test_eligible"] is False
    assert report["validator_reports"][0]["bindings"][
        "validated_manifest_sha256"
    ] == _sha(manifest)
    assert report["group_map"]["sha256"] == _sha(
        Path(report["group_map"]["path"])
    )

    # The upgraded row is accepted directly by the strict multitask trainer's
    # row parser; full train/validation class coverage is a later dataset gate.
    from scripts.train_multitask_verifier import _parse_manifest_row

    parsed = _parse_manifest_row(row, manifest_path=tmp_path / "strict.csv", line=2)
    assert parsed.sample_id == row["sample_id"]
    assert parsed.source_sha256 == row["source_sha256"]


@pytest.mark.parametrize("legacy_root", [r"Z:\\proposal\\old", "/share/old/proposal"])
def test_windows_and_posix_path_remap_resolve_source_and_crop(tmp_path, legacy_root):
    actual_root = tmp_path / "moved"
    actual_root.mkdir()
    source = actual_root / "source.jpg"
    crop = actual_root / "crop.jpg"
    source.write_bytes(b"source")
    crop.write_bytes(b"crop")
    separator = "\\" if ":" in legacy_root else "/"
    legacy_source = legacy_root.rstrip("/\\") + separator + "source.jpg"
    legacy_crop = legacy_root.rstrip("/\\") + separator + "crop.jpg"
    manifest = tmp_path / "legacy.csv"
    _write_manifest(manifest, [_row(filepath=legacy_crop, source=legacy_source)])

    upgrade_proposal_manifests(
        inputs=[manifest],
        path_remaps=[(legacy_root, str(actual_root))],
        **_outputs(tmp_path),
    )

    row = _read_csv(tmp_path / "strict.csv")[0]
    assert Path(row["filepath"]) == crop.resolve()
    decoded_source = os.fsdecode(base64.urlsafe_b64decode(row["source_path_b64"]))
    assert Path(decoded_source) == source.resolve()


def test_longest_path_remap_wins_and_parser_rejects_malformed():
    remapped = _apply_path_remaps(
        "/share/old/proposal/crop.jpg",
        [("/share/old", "/wrong"), ("/share/old/proposal", "/right")],
    )
    assert remapped.replace("\\", "/") == "/right/crop.jpg"
    assert parse_path_remap("C:\\old=D:\\new") == (r"C:\old", r"D:\new")
    with pytest.raises(Exception, match="FROM=TO"):
        parse_path_remap("missing-separator")


def test_optional_group_map_by_source_sha_is_used_but_never_claims_blind(tmp_path):
    source = tmp_path / "source.jpg"
    crop = tmp_path / "crop.jpg"
    source.write_bytes(b"trusted-source")
    crop.write_bytes(b"trusted-crop")
    manifest = tmp_path / "legacy.csv"
    _write_manifest(manifest, [_row(filepath="crop.jpg", source="source.jpg")])
    mapping = tmp_path / "groups.json"
    mapping.write_text(
        json.dumps(
            {
                _sha(source): {
                    "object_group": "hardware-object-42",
                    "capture_session": "hardware-session-7",
                }
            }
        ),
        encoding="utf-8",
    )

    report = upgrade_proposal_manifests(
        inputs=[manifest],
        group_map_path=mapping,
        **_outputs(tmp_path),
    )

    row = _read_csv(tmp_path / "strict.csv")[0]
    assert row["object_group"] == "hardware-object-42"
    assert row["capture_session"] == "hardware-session-7"
    assert row["selection_reason"] == "trusted_group_map"
    assert report["blind_test_eligible"] is False


def test_group_map_can_use_decoded_relative_source_path(tmp_path):
    source = tmp_path / "source.jpg"
    crop = tmp_path / "crop.jpg"
    source.write_bytes(b"mapped-source")
    crop.write_bytes(b"mapped-crop")
    manifest = tmp_path / "legacy.csv"
    _write_manifest(manifest, [_row(filepath="crop.jpg", source="source.jpg")])
    mapping = tmp_path / "groups.json"
    mapping.write_text(
        json.dumps(
            {
                "source.jpg": {
                    "object_group": "decoded-path-object",
                    "capture_session": "decoded-path-session",
                }
            }
        ),
        encoding="utf-8",
    )

    upgrade_proposal_manifests(
        inputs=[manifest],
        group_map_path=mapping,
        **_outputs(tmp_path),
    )

    row = _read_csv(tmp_path / "strict.csv")[0]
    assert row["object_group"] == "decoded-path-object"
    assert row["capture_session"] == "decoded-path-session"


def test_derives_strict_source_bbox_from_proposal_crop_bounds(tmp_path):
    source = tmp_path / "source.jpg"
    crop = tmp_path / "crop.jpg"
    source.write_bytes(b"source")
    crop.write_bytes(b"crop")
    row = _row(filepath="crop.jpg", source="source.jpg")
    row.update(
        {
            "crop_x1": "10",
            "crop_y1": "20",
            "crop_x2": "110",
            "crop_y2": "220",
            "source_width": "1920",
            "source_height": "1080",
        }
    )
    manifest = tmp_path / "legacy.csv"
    _write_manifest(manifest, [row])

    upgrade_proposal_manifests(inputs=[manifest], **_outputs(tmp_path))

    upgraded = _read_csv(tmp_path / "strict.csv")[0]
    assert upgraded["source_bbox_x"] == "10"
    assert upgraded["source_bbox_y"] == "20"
    assert upgraded["source_bbox_w"] == "100"
    assert upgraded["source_bbox_h"] == "200"


def test_cross_role_source_group_and_session_leakage_refuses_all_manifests(tmp_path):
    source = tmp_path / "source.jpg"
    train_crop = tmp_path / "train.jpg"
    validation_crop = tmp_path / "validation.jpg"
    source.write_bytes(b"same physical source")
    train_crop.write_bytes(b"train crop")
    validation_crop.write_bytes(b"validation crop")
    manifest = tmp_path / "legacy.csv"
    _write_manifest(
        manifest,
        [
            _row(filepath="train.jpg", source="source.jpg", split="training"),
            _row(
                filepath="validation.jpg",
                source="source.jpg",
                split="validation",
                source_id="legacy-source-2",
            ),
        ],
    )

    with pytest.raises(UpgradeRejected, match="crosses mutually exclusive roles"):
        upgrade_proposal_manifests(inputs=[manifest], **_outputs(tmp_path))

    assert not (tmp_path / "strict.csv").exists()
    assert not (tmp_path / "strict.jsonl").exists()
    rejections = json.loads((tmp_path / "rejections.json").read_text(encoding="utf-8"))
    fields = {item.get("field") for item in rejections["rejections"]}
    assert {"source_sha256", "object_group", "capture_session"} <= fields
    assert rejections["blind_test_eligible"] is False


def test_quarantines_only_validation_near_train_when_sequence_id_is_unavailable(tmp_path):
    train_source = tmp_path / "train-source.jpg"
    validation_source = tmp_path / "validation-source.jpg"
    train_source.write_bytes(b"train-source")
    validation_source.write_bytes(b"validation-source")
    pixels = np.zeros((48, 48, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(48, dtype=np.uint8)[None, :]
    pixels[:, :, 1] = np.arange(48, dtype=np.uint8)[:, None]
    pixels[:, :, 2] = 180
    train_crop = tmp_path / "train.png"
    validation_crop = tmp_path / "validation.png"
    for path, compression in ((train_crop, 1), (validation_crop, 9)):
        ok, encoded = cv2.imencode(
            ".png", pixels, [cv2.IMWRITE_PNG_COMPRESSION, compression]
        )
        assert ok
        path.write_bytes(encoded.tobytes())
    assert train_crop.read_bytes() != validation_crop.read_bytes()

    manifest = tmp_path / "legacy.csv"
    _write_manifest(
        manifest,
        [
            _row(filepath="train.png", source="train-source.jpg", split="training"),
            _row(
                filepath="validation.png",
                source="validation-source.jpg",
                split="validation",
                source_id="validation-source",
            ),
        ],
    )

    report = upgrade_proposal_manifests(
        inputs=[manifest],
        quarantine_validation_near_phash_distance=0,
        **_outputs(tmp_path),
    )

    rows = _read_csv(tmp_path / "strict.csv")
    rejection_report = json.loads(
        (tmp_path / "rejections.json").read_text(encoding="utf-8")
    )
    assert [row["role"] for row in rows] == ["train"]
    assert report["near_phash_quarantine"] == {
        "enabled": True,
        "distance": 0,
        "policy": "drop_model_validation_near_train_only",
        "training_rows_removed": 0,
        "validation_rows_removed": 1,
        "removed_by_category": {"can": 1},
    }
    assert rejection_report["rejected_count"] == 0
    assert rejection_report["quarantined_validation_count"] == 1
    quarantined = rejection_report["quarantined_validation"][0]
    assert quarantined["minimum_distance"] == 0
    assert quarantined["sample_id"].startswith("proposal_")
    assert quarantined["matching_train_sample_ids"][0].startswith("proposal_")


def test_malformed_base64_and_missing_files_are_rejected_without_partial_output(tmp_path):
    source = tmp_path / "source.jpg"
    crop = tmp_path / "crop.jpg"
    source.write_bytes(b"source")
    crop.write_bytes(b"crop")
    manifest = tmp_path / "legacy.csv"
    malformed = _row(filepath="crop.jpg", source="source.jpg")
    malformed["source_path_b64"] = "%%%not-base64%%%"
    missing = _row(
        filepath="missing.jpg",
        source="source.jpg",
        source_id="missing-crop",
    )
    _write_manifest(manifest, [malformed, missing])

    with pytest.raises(UpgradeRejected) as caught:
        upgrade_proposal_manifests(inputs=[manifest], **_outputs(tmp_path))

    messages = " ".join(str(item["error"]) for item in caught.value.rejections)
    assert "source_path_b64" in messages
    assert "crop file does not exist" in messages
    assert not (tmp_path / "strict.csv").exists()
    assert (tmp_path / "rejections.json").is_file()


def test_declared_full_sha_is_verified_and_core_labels_are_validated(tmp_path):
    source = tmp_path / "source.jpg"
    crop = tmp_path / "crop.jpg"
    source.write_bytes(b"source")
    crop.write_bytes(b"crop")
    hash_mismatch = _row(filepath="crop.jpg", source="source.jpg")
    hash_mismatch["source_sha256"] = "0" * 64
    invalid_label = _row(
        filepath="crop.jpg",
        source="source.jpg",
        source_id="invalid-label",
    )
    invalid_label["material"] = 2
    invalid_label["category"] = "plastic"
    manifest = tmp_path / "legacy.csv"
    _write_manifest(manifest, [hash_mismatch, invalid_label])

    with pytest.raises(UpgradeRejected) as caught:
        upgrade_proposal_manifests(inputs=[manifest], **_outputs(tmp_path))

    messages = " ".join(str(item["error"]) for item in caught.value.rejections)
    assert "source SHA-256 does not match" in messages
    assert "category 'plastic' does not match material 2 ('paper')" in messages


def test_duplicate_conflicting_sample_is_rejected(tmp_path):
    source = tmp_path / "source.jpg"
    crop = tmp_path / "crop.jpg"
    source.write_bytes(b"source")
    crop.write_bytes(b"crop")
    first = _row(filepath="crop.jpg", source="source.jpg", custom_note="first")
    second = dict(first)
    second["custom_note"] = "second"
    manifest = tmp_path / "legacy.csv"
    _write_manifest(manifest, [first, second])

    with pytest.raises(UpgradeRejected, match="duplicate conflicting sample"):
        upgrade_proposal_manifests(inputs=[manifest], **_outputs(tmp_path))


def test_deterministic_csv_jsonl_and_sample_id_are_role_fold_independent(tmp_path):
    source_a = tmp_path / "a-source.jpg"
    source_b = tmp_path / "b-source.jpg"
    crop_a = tmp_path / "a-crop.jpg"
    crop_b = tmp_path / "b-crop.jpg"
    source_a.write_bytes(b"source-a")
    source_b.write_bytes(b"source-b")
    crop_a.write_bytes(b"crop-a")
    crop_b.write_bytes(b"crop-b")
    manifest = tmp_path / "legacy.csv"
    _write_manifest(
        manifest,
        [
            _row(filepath="b-crop.jpg", source="b-source.jpg", source_id="b"),
            _row(filepath="a-crop.jpg", source="a-source.jpg", source_id="a"),
        ],
    )

    first = upgrade_proposal_manifests(inputs=[manifest], **_outputs(tmp_path, "-1"))
    second = upgrade_proposal_manifests(inputs=[manifest], **_outputs(tmp_path, "-2"))

    assert (tmp_path / "strict-1.csv").read_bytes() == (tmp_path / "strict-2.csv").read_bytes()
    assert (tmp_path / "strict-1.jsonl").read_bytes() == (
        tmp_path / "strict-2.jsonl"
    ).read_bytes()
    assert first["outputs"]["csv"]["sha256"] == second["outputs"]["csv"]["sha256"]
    rows = _read_csv(tmp_path / "strict-1.csv")
    original_id = rows[0]["sample_id"]
    payload = {
        "image_sha256": rows[0]["image_sha256"],
        "object_group": rows[0]["object_group"],
        "source_sha256": rows[0]["source_sha256"],
    }
    expected = "proposal_" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    assert original_id == expected


def test_same_content_identity_has_same_sample_id_when_role_and_fold_change(tmp_path):
    source = tmp_path / "source.jpg"
    crop = tmp_path / "crop.jpg"
    source.write_bytes(b"source")
    crop.write_bytes(b"crop")
    training = tmp_path / "training.csv"
    validation = tmp_path / "validation.csv"
    _write_manifest(training, [_row(filepath="crop.jpg", source="source.jpg")])
    _write_manifest(
        validation,
        [
            _row(
                filepath="crop.jpg",
                source="source.jpg",
                split="validation",
            )
        ],
    )

    upgrade_proposal_manifests(inputs=[training], **_outputs(tmp_path, "-train"))
    upgrade_proposal_manifests(inputs=[validation], **_outputs(tmp_path, "-validation"))

    train_row = _read_csv(tmp_path / "strict-train.csv")[0]
    validation_row = _read_csv(tmp_path / "strict-validation.csv")[0]
    assert train_row["role"] == "train"
    assert validation_row["role"] == "model_validation"
    assert train_row["fold"] != validation_row["fold"]
    assert train_row["sample_id"] == validation_row["sample_id"]
    assert train_row["content_identity"] == validation_row["content_identity"]


def test_dry_run_writes_nothing_and_overwrite_requires_opt_in(tmp_path):
    source = tmp_path / "source.jpg"
    crop = tmp_path / "crop.jpg"
    source.write_bytes(b"source")
    crop.write_bytes(b"crop")
    manifest = tmp_path / "legacy.csv"
    _write_manifest(manifest, [_row(filepath="crop.jpg", source="source.jpg")])
    outputs = _outputs(tmp_path)

    report = upgrade_proposal_manifests(inputs=[manifest], dry_run=True, **outputs)
    assert report["dry_run"] is True
    assert not any(path.exists() for path in outputs.values())

    outputs["output_csv"].write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        upgrade_proposal_manifests(inputs=[manifest], **outputs)
    assert outputs["output_csv"].read_text(encoding="utf-8") == "do not overwrite"


def test_validator_report_is_required_pinned_and_bound_to_manifest(tmp_path):
    source = tmp_path / "source.jpg"
    crop = tmp_path / "crop.jpg"
    source.write_bytes(b"source")
    crop.write_bytes(b"crop")
    manifest = tmp_path / "validated.csv"
    _write_manifest(manifest, [_row(filepath="crop.jpg", source="source.jpg")])
    report, pin = _validator_report(tmp_path / "validator.json", manifest)
    group_map, group_pin = _automatic_group_map(tmp_path / "groups.json", [manifest])
    common = {
        "inputs": [manifest],
        "validator_report_paths": [report],
        "output_csv": tmp_path / "strict.csv",
        "output_jsonl": tmp_path / "strict.jsonl",
        "lineage_path": tmp_path / "lineage.json",
        "rejections_path": tmp_path / "rejections.json",
        "group_map_path": group_map,
        "group_map_sha256": group_pin,
        "dry_run": True,
    }
    with pytest.raises(ValueError, match="trusted pin"):
        _upgrade_proposal_manifests(
            validator_report_sha256s=["0" * 64], **common
        )
    parsed = json.loads(report.read_text(encoding="utf-8"))
    parsed["ready_for_lineage_upgrade"] = False
    report.write_text(json.dumps(parsed), encoding="utf-8")
    with pytest.raises(ValueError, match="ready_for_lineage_upgrade"):
        _upgrade_proposal_manifests(
            validator_report_sha256s=[_sha(report)], **common
        )
    parsed["ready_for_lineage_upgrade"] = True
    parsed["bindings"]["validated_manifest_sha256"] = "f" * 64
    report.write_text(json.dumps(parsed), encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind"):
        _upgrade_proposal_manifests(
            validator_report_sha256s=[_sha(report)], **common
        )


def test_group_map_requires_pin_and_unmapped_fallback_requires_phash(tmp_path):
    source = tmp_path / "source.jpg"
    crop = tmp_path / "crop.jpg"
    source.write_bytes(b"source")
    crop.write_bytes(b"crop")
    manifest = tmp_path / "validated.csv"
    _write_manifest(manifest, [_row(filepath="crop.jpg", source="source.jpg")])
    report, report_pin = _validator_report(tmp_path / "validator.json", manifest)
    outputs = _outputs(tmp_path)
    with pytest.raises(ValueError, match="pHash quarantine is required"):
        _upgrade_proposal_manifests(
            inputs=[manifest],
            validator_report_paths=[report],
            validator_report_sha256s=[report_pin],
            **outputs,
        )
    empty_map = tmp_path / "groups.json"
    empty_map.write_text("{}", encoding="utf-8")
    with pytest.raises(UpgradeRejected, match="does not cover every source"):
        _upgrade_proposal_manifests(
            inputs=[manifest],
            validator_report_paths=[report],
            validator_report_sha256s=[report_pin],
            group_map_path=empty_map,
            group_map_sha256=_sha(empty_map),
            **outputs,
        )


def test_cli_origin_overrides_legacy_origin_and_preserves_it(tmp_path):
    source = tmp_path / "source.jpg"
    crop = tmp_path / "crop.jpg"
    source.write_bytes(b"source")
    crop.write_bytes(b"crop")
    row = _row(filepath="crop.jpg", source="source.jpg")
    row["origin"] = "spoofed_trusted_origin"
    manifest = tmp_path / "validated.csv"
    fields = [*FIELDS, "origin"]
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    upgrade_proposal_manifests(
        inputs=[manifest], origin="pinned_v4_origin", **_outputs(tmp_path)
    )
    upgraded = _read_csv(tmp_path / "strict.csv")[0]
    assert upgraded["origin"] == "pinned_v4_origin"
    assert upgraded["legacy_origin"] == "spoofed_trusted_origin"


def test_bundle_publish_rolls_back_when_any_target_races(tmp_path, monkeypatch):
    targets = [(tmp_path / f"artifact-{index}", str(index).encode()) for index in range(4)]
    real_link = os.link
    calls = 0

    def fail_second(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FileExistsError("raced target")
        return real_link(source, destination)

    monkeypatch.setattr(lineage_upgrade.os, "link", fail_second)
    with pytest.raises(FileExistsError, match="raced target"):
        lineage_upgrade._publish_exclusive_bundle(targets)
    assert all(not path.exists() for path, _ in targets)
    assert not list(tmp_path.glob(".*.tmp"))


def test_crop_object_count_must_match_material_contract(tmp_path):
    source = tmp_path / "source.jpg"
    crop = tmp_path / "crop.jpg"
    source.write_bytes(b"source")
    crop.write_bytes(b"crop")
    row = _row(filepath="crop.jpg", source="source.jpg")
    row["crop_object_count"] = 0
    manifest = tmp_path / "validated.csv"
    _write_manifest(manifest, [row])
    with pytest.raises(UpgradeRejected, match="crop_object_count=0"):
        upgrade_proposal_manifests(inputs=[manifest], **_outputs(tmp_path))


def test_source_or_crop_change_between_hash_and_publish_is_rejected(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.jpg"
    crop = tmp_path / "crop.jpg"
    source.write_bytes(b"source-a")
    crop.write_bytes(b"crop-a")
    manifest = tmp_path / "validated.csv"
    _write_manifest(manifest, [_row(filepath="crop.jpg", source="source.jpg")])
    report, report_pin = _validator_report(tmp_path / "validator.json", manifest)
    group_map, group_pin = _automatic_group_map(tmp_path / "groups.json", [manifest])
    real_stable_hash = lineage_upgrade._stable_sha256_file
    crop_calls = 0

    def mutate_before_second_crop_hash(path):
        nonlocal crop_calls
        if Path(path).resolve() == crop.resolve():
            crop_calls += 1
            if crop_calls == 2:
                crop.write_bytes(b"crop-b")
        return real_stable_hash(path)

    monkeypatch.setattr(
        lineage_upgrade, "_stable_sha256_file", mutate_before_second_crop_hash
    )
    with pytest.raises(ValueError, match="crop file changed during upgrade"):
        _upgrade_proposal_manifests(
            inputs=[manifest],
            validator_report_paths=[report],
            validator_report_sha256s=[report_pin],
            group_map_path=group_map,
            group_map_sha256=group_pin,
            **_outputs(tmp_path),
        )
    assert not any(path.exists() for path in _outputs(tmp_path).values())
