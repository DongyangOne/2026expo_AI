from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.build_v4_repro_pilot_inputs import (
    ScannedSource,
    _selection_rows,
    _source_score,
    build_pilot_inputs,
)


CLASS_NAMES = (
    "can", "pet", "paper", "plastic", "styrofoam",
    "vinyl", "glass", "battery", "fluorescent",
)


def _write_source(
    root: Path,
    *,
    split: str,
    name: str,
    content: bytes,
    label: str | None,
    decodable: bool = True,
) -> Path:
    image = root / split / "images" / f"{name}.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    if decodable:
        digest = hashlib.sha256(content).digest()
        pixels = np.frombuffer(digest * 24, dtype=np.uint8).reshape(16, 16, 3)
        encoded_ok, encoded = cv2.imencode(
            ".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        assert encoded_ok
        image.write_bytes(encoded.tobytes())
    else:
        image.write_bytes(content)
    if label is not None:
        label_path = root / split / "labels" / f"{name}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(label, encoding="utf-8")
    return image


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    root = tmp_path / "dataset"
    sources = {
        "train_can_a": _write_source(
            root, split="train", name="can_a", content=b"can-a", label="0 .5 .5 .4 .4\n"
        ),
        "train_can_b": _write_source(
            root, split="train", name="can_b", content=b"can-b", label="0 .5 .5 .4 .4\n"
        ),
        "train_paper": _write_source(
            root, split="train", name="paper", content=b"paper", label="2 .5 .5 .4 .4\n"
        ),
        "train_pet": _write_source(
            root, split="train", name="pet", content=b"pet-t", label="1 .5 .5 .4 .4\n"
        ),
        "train_empty": _write_source(
            root, split="train", name="empty", content=b"empty-t", label=""
        ),
        "train_multi": _write_source(
            root,
            split="train",
            name="multi",
            content=b"multi",
            label="0 .5 .5 .4 .4\n2 .5 .5 .2 .2\n",
        ),
        "train_malformed": _write_source(
            root, split="train", name="malformed", content=b"bad", label="oops\n"
        ),
        "train_missing": _write_source(
            root, split="train", name="missing", content=b"missing", label=None
        ),
        "train_dup_a": _write_source(
            root, split="train", name="dup_a", content=b"same-train", label="3 .5 .5 .4 .4\n"
        ),
        "train_dup_b": _write_source(
            root, split="train", name="dup_b", content=b"same-train", label="3 .5 .5 .4 .4\n"
        ),
        "train_cross": _write_source(
            root, split="train", name="cross", content=b"cross", label="4 .5 .5 .4 .4\n"
        ),
        "train_styrofoam": _write_source(
            root, split="train", name="styrofoam", content=b"styrofoam-t", label="4 .5 .5 .4 .4\n"
        ),
        "train_vinyl": _write_source(
            root, split="train", name="vinyl", content=b"vinyl-t", label="5 .5 .5 .4 .4\n"
        ),
        "train_glass": _write_source(
            root, split="train", name="glass", content=b"glass-t", label="6 .5 .5 .4 .4\n"
        ),
        "train_battery": _write_source(
            root, split="train", name="battery", content=b"battery-t", label="7 .5 .5 .4 .4\n"
        ),
        "train_fluorescent": _write_source(
            root, split="train", name="fluorescent", content=b"fluorescent-t", label="8 .5 .5 .4 .4\n"
        ),
        "train_unreadable": _write_source(
            root,
            split="train",
            name="unreadable",
            content=b"not-an-image",
            label="1 .5 .5 .4 .4\n",
            decodable=False,
        ),
        "val_can": _write_source(
            root, split="val", name="can", content=b"can-v", label="0 .5 .5 .4 .4\n"
        ),
        "val_pet": _write_source(
            root, split="val", name="pet", content=b"pet-v", label="1 .5 .5 .4 .4\n"
        ),
        "val_paper": _write_source(
            root, split="val", name="paper", content=b"paper-v", label="2 .5 .5 .4 .4\n"
        ),
        "val_plastic": _write_source(
            root, split="val", name="plastic", content=b"plastic-v", label="3 .5 .5 .4 .4\n"
        ),
        "val_styrofoam": _write_source(
            root, split="val", name="styrofoam", content=b"styrofoam-v", label="4 .5 .5 .4 .4\n"
        ),
        "val_vinyl": _write_source(
            root, split="val", name="vinyl", content=b"vinyl-v", label="5 .5 .5 .4 .4\n"
        ),
        "val_glass": _write_source(
            root, split="val", name="glass", content=b"glass-v", label="6 .5 .5 .4 .4\n"
        ),
        "val_battery": _write_source(
            root, split="val", name="battery", content=b"battery-v", label="7 .5 .5 .4 .4\n"
        ),
        "val_fluorescent": _write_source(
            root, split="val", name="fluorescent", content=b"fluorescent-v", label="8 .5 .5 .4 .4\n"
        ),
        "val_empty": _write_source(
            root, split="val", name="empty", content=b"empty-v", label=""
        ),
        "val_cross": _write_source(
            root, split="val", name="cross", content=b"cross", label="4 .5 .5 .4 .4\n"
        ),
    }
    train_list = root / "train.txt"
    train_list.write_text(
        "\n".join(path.as_posix() for key, path in sources.items() if key.startswith("train_"))
        + "\n",
        encoding="utf-8",
    )
    yaml = root / "dataset.yaml"
    yaml.write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                f"train: {train_list.as_posix()}",
                f"val: {(root / 'val' / 'images').as_posix()}",
                "names:",
                *(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return yaml, root, sources


def _inventory(output: Path) -> dict:
    return json.loads((output / "selection_inventory.json").read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quality_manifest(tmp_path: Path, entries: list[dict[str, str]] | None = None) -> Path:
    if entries is None:
        default_source = tmp_path / "dataset" / "train" / "images" / "unreadable.jpg"
        assert default_source.is_file()
        entries = [
            {
                "source_sha256": _sha(default_source),
                "reason": "objective_unreadable",
            }
        ]
    entries = sorted(entries, key=lambda row: row["source_sha256"])
    reason_counts: dict[str, int] = {}
    for row in entries:
        reason_counts[row["reason"]] = reason_counts.get(row["reason"], 0) + 1
    path = tmp_path / "quality-exclusions.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_role": (
                    "v4_capture_quality_exclusion_manifest_selection_only_"
                    "not_ground_truth_or_authority"
                ),
                "quality_exclusion_contract": (
                    "v4_capture_quality_exclusions.sha256_reason_only.v1"
                ),
                "status": "quality_exclusions_ready",
                "excluded_source_count": len(entries),
                "max_excluded_sources": 100,
                "reason_counts": dict(sorted(reason_counts.items())),
                "source_list_sha256": hashlib.sha256(
                    (
                        json.dumps(
                            entries,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                ).hexdigest(),
                "entries": entries,
                "authority": {
                    "selection": False,
                    "ground_truth": False,
                    "replay": False,
                    "training": False,
                    "calibration": False,
                    "blind_test": False,
                    "deployment": False,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_deterministic_selection_and_invalid_source_quarantine(tmp_path: Path) -> None:
    data, dataset_dir, sources = _fixture(tmp_path)
    first = tmp_path / "pilot-a"
    second = tmp_path / "pilot-b"

    build_pilot_inputs(
        data_path=data,
        dataset_dir=dataset_dir,
        output_dir=first,
        quality_exclusion_manifest=_quality_manifest(tmp_path),
        seed=77,
        training_quota=1,
        validation_quota=1,
    )
    build_pilot_inputs(
        data_path=data,
        dataset_dir=dataset_dir,
        output_dir=second,
        quality_exclusion_manifest=_quality_manifest(tmp_path),
        seed=77,
        training_quota=1,
        validation_quota=1,
    )

    assert (first / "train_pilot.txt").read_bytes() == (
        second / "train_pilot.txt"
    ).read_bytes()
    assert (first / "validation_pilot.txt").read_bytes() == (
        second / "validation_pilot.txt"
    ).read_bytes()
    first_selected = _inventory(first)["selected_sources"]
    second_selected = _inventory(second)["selected_sources"]
    assert first_selected == second_selected

    selected_paths = {row["path"] for row in first_selected}
    assert sources["train_cross"].resolve().as_posix() not in selected_paths
    assert sources["val_cross"].resolve().as_posix() not in selected_paths
    assert sources["train_multi"].resolve().as_posix() not in selected_paths
    assert sources["train_malformed"].resolve().as_posix() not in selected_paths
    assert sources["train_missing"].resolve().as_posix() not in selected_paths
    duplicate_paths = {
        sources["train_dup_a"].resolve().as_posix(),
        sources["train_dup_b"].resolve().as_posix(),
    }
    assert len(selected_paths & duplicate_paths) == 1

    counts = _inventory(first)["rejections"]["counts"]
    assert counts["duplicate_source_content_cross_split"] == 2
    assert counts["duplicate_source_content_same_split"] == 1
    assert counts["multi_object_label"] == 1
    assert counts["malformed_label/invalid_column_count"] == 1
    assert counts["missing_label_file"] == 1
    assert counts["unreadable_image"] == 1


def test_historical_drift_anchor_is_selection_only_and_has_priority(tmp_path: Path) -> None:
    data, dataset_dir, sources = _fixture(tmp_path)
    can_sources = [sources["train_can_a"], sources["train_can_b"]]
    default_output = tmp_path / "without-anchor"
    build_pilot_inputs(
        data_path=data,
        dataset_dir=dataset_dir,
        output_dir=default_output,
        quality_exclusion_manifest=_quality_manifest(tmp_path),
        seed=11,
        training_quota=1,
        validation_quota=1,
    )
    chosen_can = next(
        row for row in _inventory(default_output)["selected_sources"]
        if row["split"] == "training" and row["stratum"] == "can"
    )
    anchor_path = next(
        path for path in can_sources if path.resolve().as_posix() != chosen_can["path"]
    )
    anchor_sha = _sha(anchor_path)

    old_manifest = tmp_path / "historical.csv"
    with old_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "split", "category"])
        writer.writeheader()
        for path in can_sources:
            writer.writerow(
                {"source_id": _sha(path), "split": "training", "category": "can"}
            )
    drift = tmp_path / "drift.json"
    unrelated_sha = _sha(next(path for path in can_sources if path != anchor_path))
    drift.write_text(
        json.dumps(
            {
                "source_id": unrelated_sha,
                "replay": {
                    "hard_semantic_mismatch_examples": {
                        "crop_bounds_changed": [{"source_id": anchor_sha}]
                    },
                    "unrelated_nested_diagnostic": {"source_id": unrelated_sha},
                    "fixed_threshold_diagnostics": {"crossing_counts": {}},
                },
            }
        ),
        encoding="utf-8",
    )

    anchored_output = tmp_path / "with-anchor"
    build_pilot_inputs(
        data_path=data,
        dataset_dir=dataset_dir,
        output_dir=anchored_output,
        quality_exclusion_manifest=_quality_manifest(tmp_path),
        seed=11,
        training_quota=1,
        validation_quota=1,
        old_manifest=old_manifest,
        drift_report=drift,
    )
    inventory = _inventory(anchored_output)
    selected_can = next(
        row for row in inventory["selected_sources"]
        if row["split"] == "training" and row["stratum"] == "can"
    )
    assert selected_can["source_sha256"] == anchor_sha
    assert selected_can["drift_anchor"] is True
    assert selected_can["historical_categories_selection_only"] == ["can"]
    assert inventory["historical_selection_evidence"]["ground_truth_authority"] is False
    assert inventory["historical_selection_evidence"]["drift_report"]["anchor_source_ids"] == 1
    assert inventory["authority"]["training_authorized"] is False
    assert inventory["authority"]["production_deployment_authorized"] is False


def test_capture_quality_exclusion_removes_bad_drift_anchor_but_keeps_normal_anchor(
    tmp_path: Path,
) -> None:
    data, dataset_dir, sources = _fixture(tmp_path)
    bad_anchor = sources["train_can_a"]
    good_candidate = sources["train_can_b"]
    old_manifest = tmp_path / "historical-quality.csv"
    old_manifest.write_text(
        "source_id,split,category\n"
        f"{_sha(bad_anchor)},training,can\n"
        f"{_sha(good_candidate)},training,can\n",
        encoding="utf-8",
    )
    drift = tmp_path / "quality-drift.json"
    drift.write_text(
        json.dumps(
            {
                "replay": {
                    "hard_semantic_mismatch_examples": {
                        "crop_bounds_changed": [
                            {"source_id": _sha(bad_anchor)},
                            {"source_id": _sha(good_candidate)},
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    exclusions = _quality_manifest(
        tmp_path,
        [
            {
                "source_sha256": _sha(bad_anchor),
                "reason": "severe_frame_crop",
            }
        ],
    )
    output = tmp_path / "quality-filtered-pilot"
    build_pilot_inputs(
        data_path=data,
        dataset_dir=dataset_dir,
        output_dir=output,
        quality_exclusion_manifest=exclusions,
        training_quota=1,
        validation_quota=1,
        old_manifest=old_manifest,
        drift_report=drift,
    )

    inventory = _inventory(output)
    selected_hashes = {row["source_sha256"] for row in inventory["selected_sources"]}
    assert _sha(bad_anchor) not in selected_hashes
    assert _sha(good_candidate) in selected_hashes
    assert inventory["rejections"]["counts"][
        "quality_excluded/severe_frame_crop"
    ] == 1
    assert inventory["quality_exclusion"] == {
        "required": True,
        "manifest_contract": "v4_capture_quality_exclusions.sha256_reason_only.v1",
        "manifest_path": exclusions.resolve().as_posix(),
        "manifest_sha256": _sha(exclusions),
        "source_list_sha256": hashlib.sha256(
            (
                json.dumps(
                    [
                        {
                            "source_sha256": _sha(bad_anchor),
                            "reason": "severe_frame_crop",
                        }
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
        ).hexdigest(),
        "excluded_source_count": 1,
        "max_excluded_sources": 100,
        "matched_resolved_sources": 1,
        "reason_counts": {"severe_frame_crop": 1},
        "selection_authority": False,
        "ground_truth_authority": False,
        "replay_authority": False,
        "training_authority": False,
        "calibration_authority": False,
        "blind_test_authority": False,
        "deployment_authority": False,
    }
    assert inventory["source_contract"][
        "object_dent_or_crush_is_not_a_capture_quality_exclusion"
    ] is True
    assert inventory["bindings"]["quality_exclusion_manifest_sha256"] == _sha(
        exclusions
    )


def test_quality_exclusion_sha_must_exist_in_resolved_dataset(tmp_path: Path) -> None:
    data, dataset_dir, _ = _fixture(tmp_path)
    exclusions = _quality_manifest(
        tmp_path,
        [
            {
                "source_sha256": "f" * 64,
                "reason": "captured_before_2026_08_01",
            }
        ],
    )
    output = tmp_path / "unknown-quality-source"
    with pytest.raises(ValueError, match="absent from the resolved current dataset"):
        build_pilot_inputs(
            data_path=data,
            dataset_dir=dataset_dir,
            output_dir=output,
            quality_exclusion_manifest=exclusions,
            training_quota=1,
            validation_quota=1,
        )
    assert (output / "failed.txt").is_file()
    assert not (output / "input_ready.json").exists()


def test_quality_matched_count_is_unique_sha_not_duplicate_path_count(
    tmp_path: Path,
) -> None:
    data, dataset_dir, sources = _fixture(tmp_path)
    duplicate_sha = _sha(sources["train_dup_a"])
    assert duplicate_sha == _sha(sources["train_dup_b"])
    exclusions = _quality_manifest(
        tmp_path,
        [{"source_sha256": duplicate_sha, "reason": "boundary_unreadable"}],
    )
    replacement = _write_source(
        dataset_dir,
        split="train",
        name="plastic_replacement",
        content=b"plastic-replacement",
        label="3 .5 .5 .4 .4\n",
    )
    train_list = dataset_dir / "train.txt"
    with train_list.open("a", encoding="utf-8") as handle:
        handle.write(replacement.as_posix() + "\n")
    output = tmp_path / "duplicate-quality-sha"
    build_pilot_inputs(
        data_path=data,
        dataset_dir=dataset_dir,
        output_dir=output,
        quality_exclusion_manifest=exclusions,
        training_quota=1,
        validation_quota=1,
    )
    inventory = _inventory(output)
    assert inventory["quality_exclusion"]["excluded_source_count"] == 1
    assert inventory["quality_exclusion"]["matched_resolved_sources"] == 1
    assert inventory["rejections"]["counts"][
        "quality_excluded/boundary_unreadable"
    ] == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "duplicates source SHA"),
        ("unknown_reason", "unknown reason"),
        ("malformed", "malformed"),
        ("canonical_hash", "canonical source-list hash is inconsistent"),
        ("too_many", "bounded maximum"),
        ("duplicate_key", "contains duplicate key"),
        ("authority_zero", "contract or authority is invalid"),
        ("schema_true", "contract or authority is invalid"),
        ("max_float", "contract or authority is invalid"),
        ("reason_count_bool", "reason counts are inconsistent"),
    ],
)
def test_quality_exclusion_manifest_fails_closed_on_invalid_entries(
    tmp_path: Path, mutation: str, message: str
) -> None:
    data, dataset_dir, sources = _fixture(tmp_path)
    source_sha = _sha(sources["train_can_a"])
    path = _quality_manifest(
        tmp_path,
        [{"source_sha256": source_sha, "reason": "severe_frame_crop"}],
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "duplicate":
        value["entries"].append(dict(value["entries"][0]))
        value["excluded_source_count"] = 2
        value["reason_counts"] = {"severe_frame_crop": 2}
    elif mutation == "unknown_reason":
        value["entries"][0]["reason"] = "dented_object"
        value["reason_counts"] = {"dented_object": 1}
    else:
        if mutation == "malformed":
            value["entries"][0]["path"] = "private.jpg"
        elif mutation == "too_many":
            value["entries"] = [
                {
                    "source_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
                    "reason": "severe_frame_crop",
                }
                for index in range(101)
            ]
            value["entries"].sort(key=lambda row: row["source_sha256"])
            value["excluded_source_count"] = 101
            value["reason_counts"] = {"severe_frame_crop": 101}
            value["source_list_sha256"] = hashlib.sha256(
                (
                    json.dumps(
                        value["entries"], sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                ).encode()
            ).hexdigest()
        elif mutation == "duplicate_key":
            pass
        elif mutation == "authority_zero":
            value["authority"]["training"] = 0
        elif mutation == "schema_true":
            value["schema_version"] = True
        elif mutation == "max_float":
            value["max_excluded_sources"] = 100.0
        elif mutation == "reason_count_bool":
            value["reason_counts"] = {"severe_frame_crop": True}
        else:
            value["source_list_sha256"] = "f" * 64
    rendered = json.dumps(value)
    if mutation == "duplicate_key":
        rendered = '{"status":"forged",' + rendered[1:]
    path.write_text(rendered, encoding="utf-8")
    output = tmp_path / f"invalid-quality-{mutation}"
    with pytest.raises(ValueError, match=message):
        build_pilot_inputs(
            data_path=data,
            dataset_dir=dataset_dir,
            output_dir=output,
            quality_exclusion_manifest=path,
            training_quota=1,
            validation_quota=1,
        )
    assert (output / "failed.txt").is_file()
    assert not (output / "input_ready.json").exists()


def test_selector_rejects_symlink_quality_manifest_before_resolve(tmp_path: Path) -> None:
    data, dataset_dir, _ = _fixture(tmp_path)
    manifest = _quality_manifest(tmp_path)
    linked = tmp_path / "quality-link.json"
    try:
        os.symlink(manifest, linked)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="regular non-symlink file"):
        build_pilot_inputs(
            data_path=data,
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "symlink-quality-output",
            quality_exclusion_manifest=linked,
            training_quota=1,
            validation_quota=1,
        )


def test_selector_rejects_quality_manifest_ancestor_symlink(tmp_path: Path) -> None:
    data, dataset_dir, _ = _fixture(tmp_path)
    manifest = _quality_manifest(tmp_path)
    real_parent = tmp_path / "real-quality"
    real_parent.mkdir()
    copied = real_parent / manifest.name
    copied.write_bytes(manifest.read_bytes())
    linked_parent = tmp_path / "linked-quality"
    try:
        os.symlink(real_parent, linked_parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="quality exclusion manifest path.*symlink"):
        build_pilot_inputs(
            data_path=data,
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "ancestor-quality-output",
            quality_exclusion_manifest=linked_parent / manifest.name,
            training_quota=1,
            validation_quota=1,
        )


def test_historical_background_probe_fills_quota_without_relabeling(
    tmp_path: Path,
) -> None:
    data, dataset_dir, sources = _fixture(tmp_path)
    probe_sources = {
        "training": sources["train_empty"],
        "validation": sources["val_empty"],
    }
    for source in probe_sources.values():
        label = source.parent.parent / "labels" / f"{source.stem}.txt"
        label.write_text("0 .5 .5 .4 .4\n", encoding="utf-8")

    old_manifest = tmp_path / "historical-background.csv"
    with old_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "split", "category"])
        writer.writeheader()
        for split, source in probe_sources.items():
            writer.writerow(
                {"source_id": _sha(source), "split": split, "category": "background"}
            )

    output = tmp_path / "probe-pilot"
    build_pilot_inputs(
        data_path=data,
        dataset_dir=dataset_dir,
        output_dir=output,
        quality_exclusion_manifest=_quality_manifest(tmp_path),
        seed=17,
        training_quota=1,
        validation_quota=1,
        old_manifest=old_manifest,
    )

    inventory = _inventory(output)
    background_rows = [
        row for row in inventory["selected_sources"] if row["stratum"] == "background"
    ]
    assert len(background_rows) == 2
    assert {row["split"] for row in background_rows} == {"training", "validation"}
    assert all(row["selection_stratum"] == "background" for row in background_rows)
    assert all(row["current_gt_stratum"] == "can" for row in background_rows)
    assert all(row["selection_cohort"] == "historical_background_probe" for row in background_rows)
    assert all(row["explicit_empty_label"] is False for row in background_rows)
    assert all(
        row["historical_background_probe_selection_only"] is True
        for row in background_rows
    )
    assert all(row["gt_class_id"] == 0 and row["gt_xywhn"] for row in background_rows)
    assert all(
        row["historical_categories_selection_only"] == ["background"]
        for row in background_rows
    )
    selected_paths = [row["path"] for row in inventory["selected_sources"]]
    assert len(selected_paths) == len(set(selected_paths)) == 20
    assert inventory["background_quota_composition"] == {
        "training": {
            "current_explicit_empty_label": 0,
            "historical_background_probe": 1,
            "total": 1,
        },
        "validation": {
            "current_explicit_empty_label": 0,
            "historical_background_probe": 1,
            "total": 1,
        },
    }
    assert inventory["historical_selection_evidence"][
        "eligible_historical_background_probe_counts"
    ] == {"training": 1, "validation": 1}
    assert inventory["historical_selection_evidence"][
        "background_category_authority"
    ] is False


def test_background_probe_preserves_scarce_material_quota(tmp_path: Path) -> None:
    records: list[ScannedSource] = []

    def add_record(
        *, split: str, material: str, suffix: str, probe: bool = False,
        anchor: bool = False,
    ) -> None:
        class_id = CLASS_NAMES.index(material)
        identity = f"{split}-{material}-{suffix}"
        records.append(
            ScannedSource(
                path=tmp_path / f"{identity}.jpg",
                split=split,
                source_sha256=hashlib.sha256(identity.encode()).hexdigest(),
                label_path=tmp_path / f"{identity}.txt",
                label_sha256=hashlib.sha256(f"label-{identity}".encode()).hexdigest(),
                stratum=material,
                gt_class_id=class_id,
                gt_xywhn=(0.5, 0.5, 0.2, 0.2),
                historical_categories=("background",) if probe else (),
                anchor=anchor,
            )
        )

    for material in CLASS_NAMES:
        add_record(
            split="training",
            material=material,
            suffix="base",
            probe=material == "can",
            anchor=material == "can",
        )
        add_record(split="validation", material=material, suffix="base")
    add_record(split="training", material="paper", suffix="surplus", probe=True)
    validation_background = "validation-background"
    records.append(
        ScannedSource(
            path=tmp_path / f"{validation_background}.jpg",
            split="validation",
            source_sha256=hashlib.sha256(validation_background.encode()).hexdigest(),
            label_path=tmp_path / f"{validation_background}.txt",
            label_sha256=hashlib.sha256(
                f"label-{validation_background}".encode()
            ).hexdigest(),
            stratum="background",
            gt_class_id=None,
            gt_xywhn=None,
        )
    )

    selected, _, selected_counts, shortages, observed_counts = _selection_rows(
        records, seed=0, training_quota=1, validation_quota=1
    )

    training_background = next(
        row
        for row in selected
        if row["split"] == "training" and row["stratum"] == "background"
    )
    assert training_background["current_gt_stratum"] == "paper"
    assert selected_counts == {
        **{f"training/{stratum}": 1 for stratum in (*CLASS_NAMES, "background")},
        **{f"validation/{stratum}": 1 for stratum in (*CLASS_NAMES, "background")},
    }
    assert set(shortages.values()) == {0}
    selected_paths = [row["path"] for row in selected]
    selected_hashes = [row["source_sha256"] for row in selected]
    assert len(selected_paths) == len(set(selected_paths)) == 20
    assert len(selected_hashes) == len(set(selected_hashes)) == 20
    assert observed_counts["training/can"] == 1
    assert observed_counts["training/paper"] == 0
def test_historical_background_probe_requires_same_split_membership(
    tmp_path: Path,
) -> None:
    data, dataset_dir, sources = _fixture(tmp_path)
    val_source = sources["val_empty"]
    val_label = val_source.parent.parent / "labels" / f"{val_source.stem}.txt"
    val_label.write_text("0 .5 .5 .4 .4\n", encoding="utf-8")
    old_manifest = tmp_path / "wrong-split-background.csv"
    old_manifest.write_text(
        "source_id,split,category\n"
        f"{_sha(val_source)},training,background\n",
        encoding="utf-8",
    )

    output = tmp_path / "wrong-split-pilot"
    with pytest.raises(RuntimeError, match="validation/background=1"):
        build_pilot_inputs(
            data_path=data,
            dataset_dir=dataset_dir,
            output_dir=output,
            quality_exclusion_manifest=_quality_manifest(tmp_path),
            training_quota=1,
            validation_quota=1,
            old_manifest=old_manifest,
        )
    assert (output / "failed.txt").is_file()
    assert not (output / "input_ready.json").exists()


def test_historical_manifest_mutation_is_detected_before_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.build_v4_repro_pilot_inputs as builder

    data, dataset_dir, sources = _fixture(tmp_path)
    old_manifest = tmp_path / "historical.csv"
    old_manifest.write_text(
        "source_id,split,category\n"
        f"{_sha(sources['train_can_a'])},training,can\n",
        encoding="utf-8",
    )
    original = builder._verify_selected_bindings
    mutated = False

    def verify_then_mutate(selected: list[dict[str, object]]) -> None:
        nonlocal mutated
        original(selected)
        if not mutated:
            old_manifest.write_text(
                old_manifest.read_text(encoding="utf-8")
                + f"{_sha(sources['train_paper'])},training,paper\n",
                encoding="utf-8",
            )
            mutated = True

    monkeypatch.setattr(builder, "_verify_selected_bindings", verify_then_mutate)
    output = tmp_path / "mutated-history"
    with pytest.raises(RuntimeError, match="historical manifest changed"):
        builder.build_pilot_inputs(
            data_path=data,
            dataset_dir=dataset_dir,
            output_dir=output,
            quality_exclusion_manifest=_quality_manifest(tmp_path),
            training_quota=1,
            validation_quota=1,
            old_manifest=old_manifest,
        )
    assert (output / "failed.txt").is_file()
    assert not (output / "input_ready.json").exists()


def test_anchor_priority_is_capped_and_observed_remainder_uses_blake_order(
    tmp_path: Path,
) -> None:
    records = []
    for index in range(10):
        source_sha = hashlib.sha256(f"source-{index}".encode()).hexdigest()
        records.append(
            ScannedSource(
                path=tmp_path / f"source-{index}" / "images" / "item.jpg",
                split="training",
                source_sha256=source_sha,
                label_path=tmp_path / f"source-{index}" / "labels" / "item.txt",
                label_sha256=hashlib.sha256(f"label-{index}".encode()).hexdigest(),
                stratum="can",
                gt_class_id=0,
                gt_xywhn=(0.5, 0.5, 0.4, 0.4),
                historical_categories=("can",),
                anchor=index < 5,
            )
        )

    selected, _, _, _, _ = _selection_rows(
        records, seed=123, training_quota=5, validation_quota=1
    )
    selected_can = [
        row for row in selected
        if row["split"] == "training" and row["stratum"] == "can"
    ]
    assert len(selected_can) == 5
    assert sum(
        row["selection_reason"] == "drift_anchor_priority" for row in selected_can
    ) == 1
    assert sum(
        row["selection_reason"] == "historical_observation_priority_blake2"
        for row in selected_can
    ) == 4

    def score(record: ScannedSource) -> tuple[str, str, str]:
        return (
            _source_score(
                seed=123,
                split="training",
                stratum="can",
                source_sha256=record.source_sha256,
            ),
            record.source_sha256,
            record.path.resolve().as_posix(),
        )

    priority = sorted((record for record in records if record.anchor), key=score)[:1]
    expected = [
        *priority,
        *sorted((record for record in records if record not in priority), key=score)[:4],
    ]
    assert {row["source_sha256"] for row in selected_can} == {
        record.source_sha256 for record in expected
    }


def test_same_split_historical_observation_is_selection_only_priority(
    tmp_path: Path,
) -> None:
    records: list[ScannedSource] = []
    for index in range(8):
        identity = f"candidate-{index}"
        records.append(
            ScannedSource(
                path=tmp_path / identity / "images" / "item.jpg",
                split="training",
                source_sha256=hashlib.sha256(identity.encode()).hexdigest(),
                label_path=tmp_path / identity / "labels" / "item.txt",
                label_sha256=hashlib.sha256(f"label-{identity}".encode()).hexdigest(),
                stratum="can",
                gt_class_id=0,
                gt_xywhn=(0.5, 0.5, 0.4, 0.4),
            )
        )

    def score(record: ScannedSource) -> tuple[str, str, str]:
        return (
            _source_score(
                seed=321,
                split="training",
                stratum="can",
                source_sha256=record.source_sha256,
            ),
            record.source_sha256,
            record.path.resolve().as_posix(),
        )

    observed = max(records, key=score)
    unseen_blake_first = min((record for record in records if record is not observed), key=score)
    assert score(unseen_blake_first) < score(observed)
    observed.historical_categories = ("paper",)

    selected, _, _, _, observed_counts = _selection_rows(
        records, seed=321, training_quota=1, validation_quota=1
    )
    selected_can = next(
        row
        for row in selected
        if row["split"] == "training" and row["stratum"] == "can"
    )
    assert selected_can["source_sha256"] == observed.source_sha256
    assert selected_can["selection_reason"] == (
        "historical_observation_priority_blake2"
    )
    assert selected_can["historical_categories_selection_only"] == ["paper"]
    assert selected_can["current_gt_stratum"] == "can"
    assert selected_can["selection_cohort"] == "current_yolo_ground_truth"
    assert observed_counts["training/can"] == 1


def test_ready_marker_binds_every_input_and_is_published_last(tmp_path: Path) -> None:
    data, dataset_dir, _ = _fixture(tmp_path)
    output = tmp_path / "pilot"
    ready = build_pilot_inputs(
        data_path=data,
        dataset_dir=dataset_dir,
        output_dir=output,
        quality_exclusion_manifest=_quality_manifest(tmp_path),
        seed=99,
        training_quota=1,
        validation_quota=1,
    )

    marker = output / "inputs.sha256"
    parsed = {}
    for line in marker.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        parsed[name] = digest
        assert _sha(output / name) == digest
    assert parsed == ready["bindings"]["artifacts"]
    assert _sha(marker) == ready["bindings"]["inputs_marker_sha256"]
    on_disk = json.loads((output / "input_ready.json").read_text(encoding="utf-8"))
    assert on_disk == ready
    assert on_disk["full_quota_met"] is True
    assert len(on_disk["selected_counts"]) == 20
    assert set(on_disk["selected_counts"].values()) == {1}
    assert on_disk["validator_authority"] is False
    assert on_disk["training_authorized"] is False
    assert on_disk["blind_test_authorized"] is False
    assert on_disk["production_deployment_authorized"] is False
    assert not (output / "failed.txt").exists()


def test_failure_after_directory_creation_publishes_failed_without_ready(
    tmp_path: Path,
) -> None:
    data, dataset_dir, sources = _fixture(tmp_path)
    # Leave validation with only a source that is quarantined across splits.
    val_list = dataset_dir / "val-only-cross.txt"
    val_list.write_text(sources["val_cross"].as_posix() + "\n", encoding="utf-8")
    text = data.read_text(encoding="utf-8")
    data.write_text(
        text.replace((dataset_dir / "val" / "images").as_posix(), val_list.as_posix()),
        encoding="utf-8",
    )
    output = tmp_path / "failed-pilot"

    with pytest.raises(RuntimeError, match="balanced pilot quota shortage"):
        build_pilot_inputs(
            data_path=data,
            dataset_dir=dataset_dir,
            output_dir=output,
            quality_exclusion_manifest=_quality_manifest(tmp_path),
            training_quota=1,
            validation_quota=1,
        )

    assert (output / "failed.txt").is_file()
    assert not (output / "input_ready.json").exists()


def test_unreadable_required_source_causes_shortage_failure(tmp_path: Path) -> None:
    data, dataset_dir, sources = _fixture(tmp_path)
    valid = sources["val_fluorescent"]
    valid.unlink()
    valid.with_name(valid.name).parent.parent.joinpath("labels", "fluorescent.txt").unlink()
    _write_source(
        dataset_dir,
        split="val",
        name="fluorescent_unreadable",
        content=b"broken-required-image",
        label="8 .5 .5 .4 .4\n",
        decodable=False,
    )
    output = tmp_path / "unreadable-shortage"

    with pytest.raises(RuntimeError, match="validation/fluorescent=1"):
        build_pilot_inputs(
            data_path=data,
            dataset_dir=dataset_dir,
            output_dir=output,
            quality_exclusion_manifest=_quality_manifest(tmp_path),
            training_quota=1,
            validation_quota=1,
        )

    assert (output / "failed.txt").is_file()
    assert not (output / "input_ready.json").exists()


def test_ready_publish_race_also_publishes_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.build_v4_repro_pilot_inputs as builder

    data, dataset_dir, _ = _fixture(tmp_path)
    output = tmp_path / "ready-race"
    original = builder._publish_exclusive

    def publish_then_fail(path: Path, content: bytes) -> None:
        original(path, content)
        if path.name == "input_ready.json":
            raise RuntimeError("simulated failure after ready publication")

    monkeypatch.setattr(builder, "_publish_exclusive", publish_then_fail)
    with pytest.raises(RuntimeError, match="simulated failure"):
        builder.build_pilot_inputs(
            data_path=data,
            dataset_dir=dataset_dir,
            output_dir=output,
            quality_exclusion_manifest=_quality_manifest(tmp_path),
            training_quota=1,
            validation_quota=1,
        )

    assert (output / "input_ready.json").is_file()
    assert (output / "failed.txt").is_file()
    assert not (
        (output / "input_ready.json").is_file()
        and not (output / "failed.txt").exists()
    )


def test_existing_output_directory_is_never_reused(tmp_path: Path) -> None:
    data, dataset_dir, _ = _fixture(tmp_path)
    output = tmp_path / "already-there"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_pilot_inputs(
            data_path=data,
            dataset_dir=dataset_dir,
            output_dir=output,
            quality_exclusion_manifest=_quality_manifest(tmp_path),
            training_quota=1,
            validation_quota=1,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (output / "input_ready.json").exists()
    assert not (output / "failed.txt").exists()


@pytest.mark.parametrize("seed", [True, False, 1.5])
def test_builder_rejects_non_exact_integer_seed(tmp_path: Path, seed: object) -> None:
    data, dataset_dir, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        build_pilot_inputs(
            data_path=data,
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "pilot-invalid-seed",
            quality_exclusion_manifest=_quality_manifest(tmp_path),
            seed=seed,  # type: ignore[arg-type]
            training_quota=1,
            validation_quota=1,
        )
