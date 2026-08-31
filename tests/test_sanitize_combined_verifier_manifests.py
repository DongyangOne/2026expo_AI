import csv
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

import scripts.sanitize_combined_verifier_manifests as sanitizer
from scripts.audit_verifier_dataset import audit_manifests
from scripts.sanitize_combined_verifier_manifests import (
    main,
    sanitize_combined_manifests,
)


FIELDS = [
    "filepath", "split", "source_id", "material", "category", "dent",
    "label", "foreign_material", "source_object_count", "sample_id", "role",
    "fold", "source_sha256", "image_sha256", "object_group",
    "capture_session", "origin", "pseudo_label",
]


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _write_pattern(path: Path, seed: int, *, extension: str | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(72, 80, 3), dtype=np.uint8)
    suffix = extension or path.suffix
    ok, encoded = cv2.imencode(suffix, image)
    assert ok
    path.write_bytes(encoded.tobytes())
    return _sha_file(path)


def _write_same_pixels(path: Path, source: Path, extension: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = cv2.imdecode(np.frombuffer(source.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    ok, encoded = cv2.imencode(extension, image)
    assert ok
    path.write_bytes(encoded.tobytes())
    return _sha_file(path)


def _digest(label: str) -> str:
    return _sha_bytes(label.encode("utf-8"))


def _row(
    image: Path,
    *,
    manifest_dir: Path,
    sample: str,
    material: int = 2,
    role: str = "train",
    fold: str | None = None,
    source_sha: str | None = None,
    group: str | None = None,
    session: str | None = None,
    origin: str = "fixture",
    pseudo: bool = False,
    absolute_path: bool = False,
) -> dict[str, str]:
    categories = (
        "can", "pet", "paper", "plastic", "styrofoam",
        "vinyl", "glass", "battery", "fluorescent", "background",
    )
    return {
        "filepath": image.as_posix() if absolute_path else image.relative_to(manifest_dir).as_posix(),
        "split": "validation" if role == "model_validation" else "training",
        "source_id": f"source-{sample}",
        "material": str(material),
        "category": categories[material],
        "dent": "-1",
        "label": "-1",
        "foreign_material": "0" if pseudo else "-1",
        "source_object_count": "0" if material == 9 else "1",
        "sample_id": sample,
        "role": role,
        "fold": fold or ("validation" if role == "model_validation" else "train"),
        "source_sha256": source_sha or _digest(f"source-{sample}"),
        "image_sha256": _sha_file(image),
        "object_group": group or f"group-{sample}",
        "capture_session": session or f"session-{sample}",
        "origin": origin,
        "pseudo_label": "true" if pseudo else "false",
    }


def _write_manifest(
    path: Path,
    rows: list[dict[str, str]],
    fields: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=fields or FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _paths(root: Path) -> dict[str, Path]:
    return {
        "proposal_manifest": root / "proposal" / "proposal.csv",
        "proposal_output": root / "proposal" / "proposal.sanitized.csv",
        "hardware_manifest": root / "hardware" / "hardware.csv",
        "hardware_output": root / "hardware" / "hardware.sanitized.csv",
        "operational_manifest": root / "operational" / "operational.csv",
        "operational_output": root / "operational" / "operational.sanitized.csv",
        "report_json": root / "report.json",
    }


def _minimal_fixture(root: Path) -> dict[str, Path]:
    paths = _paths(root)
    for index, kind in enumerate(("proposal", "hardware", "operational"), start=1):
        image = paths[f"{kind}_manifest"].parent / f"{kind}.png"
        _write_pattern(image, index * 101)
        _write_manifest(
            paths[f"{kind}_manifest"],
            [
                _row(
                    image,
                    manifest_dir=image.parent,
                    sample=kind,
                    origin=f"{kind}-origin",
                    pseudo=kind == "operational",
                )
            ],
        )
    return paths


def test_accepts_legacy_empty_and_explicit_hard_negative_background_counts(tmp_path):
    paths = _minimal_fixture(tmp_path)
    proposal_rows = _read(paths["proposal_manifest"])
    proposal_rows[0].update(
        material="9",
        category="background",
        source_object_count="1",
        crop_object_count="0",
    )
    _write_manifest(
        paths["proposal_manifest"],
        proposal_rows,
        [*FIELDS, "crop_object_count"],
    )

    report = sanitize_combined_manifests(**paths, dry_run=True)

    assert report["combined_output_rows"] == 3


@pytest.mark.parametrize(
    ("source_count", "crop_count"),
    [(0, 1), (1, 1), (1, "")],
)
def test_rejects_background_source_crop_count_conflicts(
    tmp_path, source_count, crop_count
):
    paths = _minimal_fixture(tmp_path)
    proposal_rows = _read(paths["proposal_manifest"])
    proposal_rows[0].update(
        material="9",
        category="background",
        source_object_count=str(source_count),
        crop_object_count=str(crop_count),
    )
    _write_manifest(
        paths["proposal_manifest"],
        proposal_rows,
        [*FIELDS, "crop_object_count"],
    )

    with pytest.raises(ValueError, match="crop_object_count"):
        sanitize_combined_manifests(**paths, dry_run=True)


def test_exact_current_style_precedence_and_hash_bound_outputs(tmp_path):
    paths = _paths(tmp_path)
    proposal_dir = paths["proposal_manifest"].parent
    hardware_dir = paths["hardware_manifest"].parent
    operational_dir = paths["operational_manifest"].parent
    for directory in (proposal_dir, hardware_dir, operational_dir):
        directory.mkdir(parents=True)

    p_validation_loser = proposal_dir / "p-validation-loser.png"
    p_exact_loser = proposal_dir / "p-exact-loser.png"
    p_gt = proposal_dir / "p-gt.png"
    p_keep = proposal_dir / "p-keep.png"
    for seed, image in enumerate(
        (p_validation_loser, p_exact_loser, p_gt, p_keep), start=10
    ):
        _write_pattern(image, seed)
    shared_source = _digest("physical-source-shared")
    proposal_rows = [
        _row(
            p_validation_loser,
            manifest_dir=proposal_dir,
            sample="proposal-train-collision",
            source_sha=shared_source,
            group="proposal-group-a",
            session="proposal-session-a",
            origin="aihub_proposal_v3_lowconf",
        ),
        _row(
            p_exact_loser,
            manifest_dir=proposal_dir,
            sample="proposal-exact",
            origin="aihub_proposal_v3_lowconf",
        ),
        _row(
            p_gt,
            manifest_dir=proposal_dir,
            sample="proposal-ground-truth",
            material=2,
            origin="aihub_proposal_v3_lowconf",
        ),
        _row(
            p_keep,
            manifest_dir=proposal_dir,
            sample="proposal-keep",
            origin="aihub_proposal_v3_lowconf",
        ),
    ]
    _write_manifest(paths["proposal_manifest"], proposal_rows)

    h_validation = hardware_dir / "h-validation.png"
    _write_pattern(h_validation, 50)
    h_exact = hardware_dir / "h-exact.png"
    shutil.copyfile(p_exact_loser, h_exact)
    hardware_rows = [
        _row(
            h_validation,
            manifest_dir=hardware_dir,
            sample="hardware-validation",
            role="model_validation",
            fold="hardware-val-fold",
            source_sha=shared_source,
            group="hardware-group-a",
            session="hardware-session-a",
            origin="hardware_runtime_v3",
        ),
        _row(
            h_exact,
            manifest_dir=hardware_dir,
            sample="hardware-exact",
            origin="hardware_runtime_v3",
        ),
    ]
    _write_manifest(paths["hardware_manifest"], hardware_rows)

    o_conflict = operational_dir / "o-conflict.bmp"
    _write_same_pixels(o_conflict, p_gt, ".bmp")
    o_keep = operational_dir / "o-keep.png"
    _write_pattern(o_keep, 90)
    operational_rows = [
        _row(
            o_conflict,
            manifest_dir=operational_dir,
            sample="operational-label-conflict",
            material=3,
            fold="operational-provenance-fold",
            origin="operational_capture_vlm_teacher",
            pseudo=True,
        ),
        _row(
            o_keep,
            manifest_dir=operational_dir,
            sample="operational-keep",
            material=3,
            fold="operational-provenance-fold",
            origin="operational_capture_vlm_teacher",
            pseudo=True,
        ),
    ]
    _write_manifest(paths["operational_manifest"], operational_rows)
    source_bytes = {
        name: paths[name].read_bytes()
        for name in ("proposal_manifest", "hardware_manifest", "operational_manifest")
    }

    report = sanitize_combined_manifests(**paths, phash_distance=4)

    proposal_output = _read(paths["proposal_output"])
    hardware_output = _read(paths["hardware_output"])
    operational_output = _read(paths["operational_output"])
    assert [row["sample_id"] for row in proposal_output] == [
        "proposal-ground-truth",
        "proposal-keep",
    ]
    assert [row["sample_id"] for row in hardware_output] == [
        "hardware-validation",
        "hardware-exact",
    ]
    assert [row["sample_id"] for row in operational_output] == ["operational-keep"]
    # Retained role/fold/path values are not normalized or rebased.
    assert hardware_output[0] == hardware_rows[0]
    assert operational_output[0] == operational_rows[1]
    combined_audit = audit_manifests(
        [
            paths["proposal_output"],
            paths["hardware_output"],
            paths["operational_output"],
        ],
        phash_distance=4,
        allow_partial_class_coverage=True,
        fail_on_near_phash=True,
    )
    assert combined_audit["ok"] is True, combined_audit["problems"]
    assert report["combined_output_rows"] == 5
    assert [entry["output"]["rows"] for entry in report["manifests"]] == [2, 2, 1]
    assert report["dropped"]["reason_counts"] == {
        "exact_identity_precedence": 1,
        "near_phash_ground_truth_over_pseudo": 1,
        "validation_partition_precedence": 1,
    }
    assert report["policy"]["retained_row_fields_rewritten"] is False
    assert report["policy"]["source_manifests_modified"] is False

    persisted = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    assert persisted == report
    for entry in report["manifests"]:
        assert _sha_file(Path(entry["input"]["path"])) == entry["input"]["sha256"]
        assert _sha_file(Path(entry["output"]["path"])) == entry["output"]["sha256"]
    for name, content in source_bytes.items():
        assert paths[name].read_bytes() == content


def test_near_phash_only_drops_label_or_partition_conflicts(tmp_path):
    paths = _paths(tmp_path)
    proposal_dir = paths["proposal_manifest"].parent
    hardware_dir = paths["hardware_manifest"].parent
    operational_dir = paths["operational_manifest"].parent
    for directory in (proposal_dir, hardware_dir, operational_dir):
        directory.mkdir(parents=True)
    p_image = proposal_dir / "p.png"
    p_same_partition = proposal_dir / "p-same-partition.png"
    _write_pattern(p_image, 300)
    _write_pattern(p_same_partition, 301)
    h_image = hardware_dir / "h.bmp"
    o_image = operational_dir / "o.tiff"
    _write_same_pixels(h_image, p_image, ".bmp")
    _write_same_pixels(o_image, p_same_partition, ".tiff")
    proposal = _row(
        p_image,
        manifest_dir=proposal_dir,
        sample="proposal-train",
        role="train",
        origin="proposal",
    )
    proposal_same_partition = _row(
        p_same_partition,
        manifest_dir=proposal_dir,
        sample="proposal-train-same-partition",
        role="train",
        origin="proposal",
    )
    # Same visual and label, but validation must be retained over train.
    hardware = _row(
        h_image,
        manifest_dir=hardware_dir,
        sample="hardware-validation",
        role="model_validation",
        fold="real-validation-fold",
        origin="hardware",
    )
    # Same visual, label and partition as proposal. It does not create an audit
    # cross-partition pHash problem and is therefore retained.
    operational = _row(
        o_image,
        manifest_dir=operational_dir,
        sample="operational-train",
        role="train",
        fold="train",
        origin="operational",
        pseudo=True,
    )
    _write_manifest(paths["proposal_manifest"], [proposal, proposal_same_partition])
    _write_manifest(paths["hardware_manifest"], [hardware])
    _write_manifest(paths["operational_manifest"], [operational])

    report = sanitize_combined_manifests(**paths)

    assert _read(paths["proposal_output"]) == [proposal_same_partition]
    assert _read(paths["hardware_output"]) == [hardware]
    assert _read(paths["operational_output"]) == [operational]
    assert report["dropped"]["reason_counts"] == {
        "near_phash_validation_partition_precedence": 1
    }


def test_outputs_are_deterministic_and_dry_run_writes_nothing(tmp_path):
    first = _minimal_fixture(tmp_path / "fixture")
    source_hashes = {
        name: _sha_file(first[name])
        for name in ("proposal_manifest", "hardware_manifest", "operational_manifest")
    }
    dry_paths = dict(first)
    dry_paths["proposal_output"] = first["proposal_manifest"].with_name("dry-p.csv")
    dry_paths["hardware_output"] = first["hardware_manifest"].with_name("dry-h.csv")
    dry_paths["operational_output"] = first["operational_manifest"].with_name("dry-o.csv")
    dry_paths["report_json"] = tmp_path / "dry-report.json"
    dry_report = sanitize_combined_manifests(**dry_paths, dry_run=True)
    assert dry_report["publication"]["dry_run"] is True
    assert not any(dry_paths[name].exists() for name in (
        "proposal_output", "hardware_output", "operational_output", "report_json"
    ))

    sanitize_combined_manifests(**first)
    first_bytes = {
        name: first[name].read_bytes()
        for name in ("proposal_output", "hardware_output", "operational_output")
    }
    second = dict(first)
    second["proposal_output"] = first["proposal_manifest"].with_name("second-p.csv")
    second["hardware_output"] = first["hardware_manifest"].with_name("second-h.csv")
    second["operational_output"] = first["operational_manifest"].with_name("second-o.csv")
    second["report_json"] = tmp_path / "second-report.json"
    sanitize_combined_manifests(**second)
    assert first_bytes["proposal_output"] == second["proposal_output"].read_bytes()
    assert first_bytes["hardware_output"] == second["hardware_output"].read_bytes()
    assert first_bytes["operational_output"] == second["operational_output"].read_bytes()
    assert source_hashes == {
        name: _sha_file(first[name])
        for name in ("proposal_manifest", "hardware_manifest", "operational_manifest")
    }


def test_refuses_overwrite_and_nonadjacent_output_before_publication(tmp_path):
    paths = _minimal_fixture(tmp_path)
    paths["proposal_output"].write_text("do not overwrite", encoding="utf-8")
    source_bytes = paths["proposal_manifest"].read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        sanitize_combined_manifests(**paths)
    assert paths["proposal_output"].read_text(encoding="utf-8") == "do not overwrite"
    assert paths["proposal_manifest"].read_bytes() == source_bytes
    assert not paths["hardware_output"].exists()
    assert not paths["report_json"].exists()

    paths["proposal_output"].unlink()
    paths["operational_output"] = tmp_path / "elsewhere" / "op.csv"
    with pytest.raises(ValueError, match="adjacent"):
        sanitize_combined_manifests(**paths)
    assert not paths["proposal_output"].exists()


def test_publication_failure_rolls_back_only_new_outputs(tmp_path, monkeypatch):
    paths = _minimal_fixture(tmp_path)
    real_link = sanitizer.os.link
    calls = 0

    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        return real_link(source, target)

    monkeypatch.setattr(sanitizer.os, "link", fail_second)
    with pytest.raises(OSError, match="injected"):
        sanitize_combined_manifests(**paths)
    assert not any(paths[name].exists() for name in (
        "proposal_output", "hardware_output", "operational_output", "report_json"
    ))
    for kind in ("proposal", "hardware", "operational"):
        assert paths[f"{kind}_manifest"].is_file()


def test_rejects_hash_mismatch_invalid_distance_and_cli_writes_report(tmp_path, capsys):
    paths = _minimal_fixture(tmp_path)
    rows = _read(paths["hardware_manifest"])
    rows[0]["image_sha256"] = "0" * 64
    _write_manifest(paths["hardware_manifest"], rows)
    with pytest.raises(ValueError, match="image_sha256 does not match"):
        sanitize_combined_manifests(**paths)
    assert not paths["proposal_output"].exists()

    paths = _minimal_fixture(tmp_path / "cli")
    with pytest.raises(ValueError, match="between 0 and 4"):
        sanitize_combined_manifests(**paths, phash_distance=5)
    result = main(
        [
            "--proposal-manifest", str(paths["proposal_manifest"]),
            "--proposal-output", str(paths["proposal_output"]),
            "--hardware-manifest", str(paths["hardware_manifest"]),
            "--hardware-output", str(paths["hardware_output"]),
            "--operational-manifest", str(paths["operational_manifest"]),
            "--operational-output", str(paths["operational_output"]),
            "--report-json", str(paths["report_json"]),
            "--phash-distance", "4",
        ]
    )
    assert result == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads(paths["report_json"].read_text(encoding="utf-8"))


def test_exact_identity_chain_has_one_surviving_winner(tmp_path):
    paths = _paths(tmp_path)
    for kind in ("proposal", "hardware", "operational"):
        paths[f"{kind}_manifest"].parent.mkdir(parents=True)
    p_image = paths["proposal_manifest"].parent / "p.png"
    h_image = paths["hardware_manifest"].parent / "h.png"
    o_image = paths["operational_manifest"].parent / "o.png"
    _write_pattern(p_image, 700)
    _write_pattern(h_image, 701)
    shutil.copyfile(h_image, o_image)
    proposal = _row(
        p_image,
        manifest_dir=p_image.parent,
        sample="chain-sample",
        origin="proposal",
    )
    hardware = _row(
        h_image,
        manifest_dir=h_image.parent,
        sample="chain-sample",
        origin="hardware",
    )
    operational = _row(
        o_image,
        manifest_dir=o_image.parent,
        sample="different-sample",
        origin="operational",
        pseudo=True,
    )
    _write_manifest(paths["proposal_manifest"], [proposal])
    _write_manifest(paths["hardware_manifest"], [hardware])
    _write_manifest(paths["operational_manifest"], [operational])

    report = sanitize_combined_manifests(**paths)

    assert _read(paths["proposal_output"]) == []
    assert _read(paths["hardware_output"]) == [hardware]
    assert _read(paths["operational_output"]) == []
    assert report["dropped"]["rows"] == 2
    assert {
        record["winner"]["sample_id"] for record in report["dropped"]["records"]
    } == {"chain-sample"}
    assert all(
        record["winner"]["kind"] == "hardware"
        for record in report["dropped"]["records"]
    )


@pytest.mark.parametrize("forbidden_role", ["calibration", "blind_test"])
def test_rejects_evaluation_roles_from_training_manifests(tmp_path, forbidden_role):
    paths = _minimal_fixture(tmp_path)
    rows = _read(paths["operational_manifest"])
    rows[0]["role"] = forbidden_role
    rows[0]["split"] = "validation"
    _write_manifest(paths["operational_manifest"], rows)
    with pytest.raises(ValueError, match="must never enter combined training"):
        sanitize_combined_manifests(**paths)
    assert not paths["proposal_output"].exists()


def test_refuses_within_manifest_exact_and_partition_corruption(tmp_path):
    paths = _minimal_fixture(tmp_path / "exact")
    rows = _read(paths["proposal_manifest"])
    duplicate = dict(rows[0])
    duplicate["source_id"] = "second-source-id"
    _write_manifest(paths["proposal_manifest"], [rows[0], duplicate])
    with pytest.raises(ValueError, match="within-manifest exact duplicate"):
        sanitize_combined_manifests(**paths)

    paths = _minimal_fixture(tmp_path / "partition")
    rows = _read(paths["proposal_manifest"])
    second_image = paths["proposal_manifest"].parent / "validation.png"
    _write_pattern(second_image, 999)
    validation = _row(
        second_image,
        manifest_dir=second_image.parent,
        sample="proposal-validation",
        role="model_validation",
        source_sha=rows[0]["source_sha256"],
    )
    _write_manifest(paths["proposal_manifest"], [rows[0], validation])
    with pytest.raises(ValueError, match="within-manifest partition leakage"):
        sanitize_combined_manifests(**paths)


def test_source_manifest_race_aborts_before_any_publication(tmp_path, monkeypatch):
    paths = _minimal_fixture(tmp_path)
    real_write_temp = sanitizer._write_temp
    mutated = False

    def mutate_source_after_staging(target, content):
        nonlocal mutated
        temporary = real_write_temp(target, content)
        if not mutated:
            with paths["hardware_manifest"].open("ab") as file:
                file.write(b"\n")
            mutated = True
        return temporary

    monkeypatch.setattr(sanitizer, "_write_temp", mutate_source_after_staging)
    with pytest.raises(RuntimeError, match="changed during sanitation"):
        sanitize_combined_manifests(**paths)
    assert not any(paths[name].exists() for name in (
        "proposal_output", "hardware_output", "operational_output", "report_json"
    ))


def test_same_manifest_phash_corruption_cannot_be_masked_by_earlier_cross_drop(tmp_path):
    paths = _paths(tmp_path)
    for kind in ("proposal", "hardware", "operational"):
        paths[f"{kind}_manifest"].parent.mkdir(parents=True)
    proposal_dir = paths["proposal_manifest"].parent
    hardware_dir = paths["hardware_manifest"].parent
    operational_dir = paths["operational_manifest"].parent
    first = proposal_dir / "first.png"
    second = proposal_dir / "second.bmp"
    hardware_image = hardware_dir / "hardware.tiff"
    operational_image = operational_dir / "operational.png"
    _write_pattern(first, 1400)
    _write_same_pixels(second, first, ".bmp")
    _write_same_pixels(hardware_image, first, ".tiff")
    _write_pattern(operational_image, 1401)
    proposal_rows = [
        _row(
            first,
            manifest_dir=proposal_dir,
            sample="pseudo-proposal-a",
            material=2,
            pseudo=True,
        ),
        _row(
            second,
            manifest_dir=proposal_dir,
            sample="gt-proposal-b",
            material=3,
            role="model_validation",
        ),
    ]
    hardware = _row(
        hardware_image,
        manifest_dir=hardware_dir,
        sample="hardware-gt",
        material=3,
    )
    operational = _row(
        operational_image,
        manifest_dir=operational_dir,
        sample="operational-unrelated",
        material=3,
        pseudo=True,
    )
    _write_manifest(paths["proposal_manifest"], proposal_rows)
    _write_manifest(paths["hardware_manifest"], [hardware])
    _write_manifest(paths["operational_manifest"], [operational])

    with pytest.raises(ValueError, match="within-manifest cross-partition near-pHash"):
        sanitize_combined_manifests(**paths)
    assert not any(paths[name].exists() for name in (
        "proposal_output", "hardware_output", "operational_output", "report_json"
    ))


def test_same_manifest_same_partition_phash_collision_with_different_labels_is_kept(
    tmp_path,
):
    paths = _paths(tmp_path)
    for kind in ("proposal", "hardware", "operational"):
        paths[f"{kind}_manifest"].parent.mkdir(parents=True)
    proposal_dir = paths["proposal_manifest"].parent
    can_image = proposal_dir / "solid-can.png"
    pet_image = proposal_dir / "solid-pet.png"
    # Uniform but different colours have different bytes while the coarse DCT
    # pHash is identical.  This models the live can/PET false-near-duplicate.
    can_pixels = np.full((72, 80, 3), (220, 30, 30), dtype=np.uint8)
    pet_pixels = np.full((72, 80, 3), (30, 220, 30), dtype=np.uint8)
    assert cv2.imwrite(str(can_image), can_pixels)
    assert cv2.imwrite(str(pet_image), pet_pixels)
    assert sanitizer._perceptual_hash(can_image) == sanitizer._perceptual_hash(pet_image)
    proposal_rows = [
        _row(
            can_image,
            manifest_dir=proposal_dir,
            sample="can-hard-example",
            material=0,
            role="train",
            fold="train",
        ),
        _row(
            pet_image,
            manifest_dir=proposal_dir,
            sample="pet-hard-example",
            material=1,
            role="train",
            fold="train",
        ),
    ]
    _write_manifest(paths["proposal_manifest"], proposal_rows)
    for seed, kind in ((1701, "hardware"), (1702, "operational")):
        image = paths[f"{kind}_manifest"].parent / f"{kind}.png"
        _write_pattern(image, seed)
        _write_manifest(
            paths[f"{kind}_manifest"],
            [
                _row(
                    image,
                    manifest_dir=image.parent,
                    sample=f"{kind}-unrelated",
                    material=2,
                    pseudo=kind == "operational",
                )
            ],
        )

    report = sanitize_combined_manifests(**paths)

    assert _read(paths["proposal_output"]) == proposal_rows
    assert report["manifests"][0]["dropped_rows"] == 0
    assert all(
        record["loser"]["sample_id"] not in {"can-hard-example", "pet-hard-example"}
        for record in report["dropped"]["records"]
    )
