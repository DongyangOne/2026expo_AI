from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.build_v4_candidate_training_authority import (
    QUALITY_REASONS as AUTHORITY_QUALITY_REASONS,
    _validate_quality_manifest,
)
from scripts.build_v4_quality_exclusion_manifest import (
    QUALITY_EXCLUSION_REASON_ALIASES,
    QUALITY_EXCLUSION_REASONS,
    _manifest_value,
    build_quality_exclusion_manifest,
    build_single_quality_exclusion_manifest,
)
from scripts.build_v4_repro_pilot_inputs import (
    QUALITY_EXCLUSION_REASONS as REPRO_QUALITY_EXCLUSION_REASONS,
    _read_quality_exclusion_manifest,
)


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_v4_quality_exclusion_manifest.py"
)
EXPECTED_CANONICAL_REASONS = {
    "severe_frame_crop",
    "person_occlusion_or_dominance",
    "excessive_background_or_multi_object",
    "unreadable_boundary",
    "too_low_resolution",
    "extreme_exposure",
}
EXPECTED_REASON_ALIASES = {
    "clutter_or_multiple_objects": "excessive_background_or_multi_object",
    "boundary_unreadable": "unreadable_boundary",
    "objective_unreadable": "unreadable_boundary",
    "resolution_too_low": "too_low_resolution",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entries_sha(entries: list[dict[str, str]]) -> str:
    return hashlib.sha256(
        (
            json.dumps(
                entries,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()


def test_builder_emits_sha_reason_only_and_preserves_dented_object(tmp_path: Path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    low_resolution = root / "low-resolution.jpg"
    weird = root / "weird.jpg"
    dented = root / "dented-can.jpg"
    low_resolution.write_bytes(b"low-resolution")
    weird.write_bytes(b"bad-crop")
    dented.write_bytes(b"valid-dented-object")
    source_list = tmp_path / "quality.csv"
    source_list.write_text(
        "path,reason\n"
        "low-resolution.jpg,too_low_resolution\n"
        "weird.jpg,severe_frame_crop\n",
        encoding="utf-8",
    )
    output = tmp_path / "quality-exclusions.json"

    value = build_quality_exclusion_manifest(
        source_list=source_list, image_root=root, output_path=output
    )

    assert value["entries"] == sorted(
        [
            {"source_sha256": _sha(low_resolution), "reason": "too_low_resolution"},
            {"source_sha256": _sha(weird), "reason": "severe_frame_crop"},
        ],
        key=lambda row: row["source_sha256"],
    )
    assert _sha(dented) not in {row["source_sha256"] for row in value["entries"]}
    assert value["reason_counts"] == {
        "severe_frame_crop": 1,
        "too_low_resolution": 1,
    }
    assert value["source_list_sha256"] == _entries_sha(value["entries"])
    assert set(value["authority"].values()) == {False}
    rendered = output.read_text(encoding="utf-8")
    assert str(root) not in rendered
    assert "low-resolution.jpg" not in rendered
    assert "weird.jpg" not in rendered
    assert "dented-can.jpg" not in rendered


def test_single_source_helper_builds_manifest_without_csv_or_path_leak(
    tmp_path: Path,
) -> None:
    source = tmp_path / "failed-vinyl-anchor.jpg"
    source.write_bytes(b"failed-vinyl-anchor")
    output = tmp_path / "single-quality.json"
    value = build_single_quality_exclusion_manifest(
        source_path=source,
        reason="boundary_unreadable",
        output_path=output,
    )
    assert value["entries"] == [
        {"source_sha256": _sha(source), "reason": "unreadable_boundary"}
    ]
    assert value["excluded_source_count"] == 1
    assert value["max_excluded_sources"] == 100
    assert value["source_list_sha256"] == _entries_sha(value["entries"])
    assert source.name not in output.read_text(encoding="utf-8")


def test_single_source_cli_is_automation_ready(tmp_path: Path) -> None:
    source = tmp_path / "failed-vinyl-anchor.jpg"
    source.write_bytes(b"failed-vinyl-cli")
    output = tmp_path / "single-cli.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            str(source),
            "--reason",
            "severe_frame_crop",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["entries"] == [
        {"source_sha256": _sha(source), "reason": "severe_frame_crop"}
    ]


def test_single_source_helper_rejects_ancestor_symlink(tmp_path: Path) -> None:
    actual = tmp_path / "actual-source"
    actual.mkdir()
    (actual / "bad.jpg").write_bytes(b"bad")
    linked = tmp_path / "linked-source"
    try:
        os.symlink(actual, linked, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="single source path.*symlink components"):
        build_single_quality_exclusion_manifest(
            source_path=linked / "bad.jpg",
            reason="unreadable_boundary",
            output_path=tmp_path / "manifest.json",
        )


def test_producer_output_is_accepted_by_candidate_and_repro_consumers(
    tmp_path: Path,
) -> None:
    assert set(QUALITY_EXCLUSION_REASONS) == EXPECTED_CANONICAL_REASONS
    assert AUTHORITY_QUALITY_REASONS == EXPECTED_CANONICAL_REASONS
    assert QUALITY_EXCLUSION_REASON_ALIASES == EXPECTED_REASON_ALIASES
    assert EXPECTED_CANONICAL_REASONS <= set(REPRO_QUALITY_EXCLUSION_REASONS)
    root = tmp_path / "images"
    root.mkdir()
    image = root / "source.bin"
    image.write_bytes(b"legacy-alias-input")
    source_list = tmp_path / "source.csv"
    source_list.write_text(
        "path,reason\nsource.bin,clutter_or_multiple_objects\n", encoding="utf-8"
    )
    output = tmp_path / "quality.json"

    value = build_quality_exclusion_manifest(
        source_list=source_list, image_root=root, output_path=output
    )

    expected = {_sha(image): "excessive_background_or_multi_object"}
    published = json.loads(output.read_text(encoding="utf-8"))
    assert published == value
    assert _validate_quality_manifest(published) == expected
    assert _read_quality_exclusion_manifest(output)[0] == expected


@pytest.mark.parametrize(
    ("legacy", "canonical"), EXPECTED_REASON_ALIASES.items()
)
def test_legacy_reason_input_is_canonicalized_before_hashing(
    tmp_path: Path, legacy: str, canonical: str
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(legacy.encode())
    output = tmp_path / "quality.json"

    value = build_single_quality_exclusion_manifest(
        source_path=source, reason=legacy, output_path=output
    )

    assert value["entries"] == [{"source_sha256": _sha(source), "reason": canonical}]
    assert value["reason_counts"] == {canonical: 1}
    assert value["source_list_sha256"] == _entries_sha(value["entries"])
    assert legacy not in output.read_text(encoding="utf-8")


def test_aliases_that_share_a_canonical_reason_aggregate_counts(tmp_path: Path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    (root / "boundary.bin").write_bytes(b"boundary")
    (root / "objective.bin").write_bytes(b"objective")
    source_list = tmp_path / "source.csv"
    source_list.write_text(
        "path,reason\n"
        "boundary.bin,boundary_unreadable\n"
        "objective.bin,objective_unreadable\n",
        encoding="utf-8",
    )

    value = build_quality_exclusion_manifest(
        source_list=source_list,
        image_root=root,
        output_path=tmp_path / "quality.json",
    )

    assert value["reason_counts"] == {"unreadable_boundary": 2}
    assert {entry["reason"] for entry in value["entries"]} == {
        "unreadable_boundary"
    }
    assert value["source_list_sha256"] == _entries_sha(value["entries"])


def test_operational_cutoff_reason_is_rejected_in_favor_of_captured_at(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    root.mkdir()
    image = root / "source.bin"
    image.write_bytes(b"old-capture")
    source_list = tmp_path / "source.csv"
    source_list.write_text(
        "path,reason\nsource.bin,captured_before_2026_08_01\n", encoding="utf-8"
    )
    batch_output = tmp_path / "batch.json"
    single_output = tmp_path / "single.json"

    with pytest.raises(ValueError, match="captured_at"):
        build_quality_exclusion_manifest(
            source_list=source_list, image_root=root, output_path=batch_output
        )
    with pytest.raises(ValueError, match="captured_at"):
        build_single_quality_exclusion_manifest(
            source_path=image,
            reason="captured_before_2026_08_01",
            output_path=single_output,
        )
    assert not batch_output.exists()
    assert not single_output.exists()


@pytest.mark.parametrize("reason", ["dent", "crush", "object_dented"])
def test_object_condition_reason_is_rejected(tmp_path: Path, reason: str) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(reason.encode())
    output = tmp_path / "quality.json"
    with pytest.raises(ValueError, match="condition"):
        build_single_quality_exclusion_manifest(
            source_path=source, reason=reason, output_path=output
        )
    assert not output.exists()


@pytest.mark.parametrize("reason", QUALITY_EXCLUSION_REASONS)
def test_all_contract_reasons_are_accepted(tmp_path: Path, reason: str) -> None:
    root = tmp_path / "images"
    root.mkdir()
    image = root / "source.bin"
    image.write_bytes(reason.encode())
    source = tmp_path / "source.csv"
    source.write_text(f"path,reason\nsource.bin,{reason}\n", encoding="utf-8")
    value = build_quality_exclusion_manifest(
        source_list=source,
        image_root=root,
        output_path=tmp_path / f"{reason}.json",
    )
    assert value["entries"] == [{"source_sha256": _sha(image), "reason": reason}]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("a.bin,mystery_quality\n", "unknown reason"),
        (
            "a.bin,severe_frame_crop\na.bin,severe_frame_crop\n",
            "duplicates a path",
        ),
        (
            "a.bin,severe_frame_crop\nb.bin,extreme_exposure\n",
            "duplicates source bytes",
        ),
        ("../outside.bin,severe_frame_crop\n", "normalized and relative"),
    ],
)
def test_builder_rejects_unknown_duplicate_and_escaping_sources(
    tmp_path: Path, rows: str, message: str
) -> None:
    root = tmp_path / "images"
    root.mkdir()
    (root / "a.bin").write_bytes(b"same")
    (root / "b.bin").write_bytes(b"same")
    (tmp_path / "outside.bin").write_bytes(b"outside")
    source = tmp_path / "source.csv"
    source.write_text("path,reason\n" + rows, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        build_quality_exclusion_manifest(
            source_list=source,
            image_root=root,
            output_path=tmp_path / "manifest.json",
        )


def test_builder_requires_exact_header_and_immutable_output(tmp_path: Path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    (root / "a.bin").write_bytes(b"a")
    source = tmp_path / "source.csv"
    source.write_text("reason,path\nsevere_frame_crop,a.bin\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact path,reason header"):
        build_quality_exclusion_manifest(
            source_list=source,
            image_root=root,
            output_path=tmp_path / "manifest.json",
        )

    source.write_text("path,reason\na.bin,severe_frame_crop\n", encoding="utf-8")
    output = tmp_path / "existing.json"
    output.write_text(json.dumps({"sentinel": True}), encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite immutable output"):
        build_quality_exclusion_manifest(
            source_list=source, image_root=root, output_path=output
        )
    assert json.loads(output.read_text(encoding="utf-8")) == {"sentinel": True}


def test_builder_rejects_empty_adjudication(tmp_path: Path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    source = tmp_path / "source.csv"
    source.write_text("path,reason\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one source"):
        build_quality_exclusion_manifest(
            source_list=source,
            image_root=root,
            output_path=tmp_path / "manifest.json",
        )


def test_empty_serialization_is_not_standalone_evidence() -> None:
    value = _manifest_value([])
    assert value["entries"] == []
    assert value["excluded_source_count"] == 0
    assert value["reason_counts"] == {}
    assert value["source_list_sha256"] == _entries_sha([])
    with pytest.raises(ValueError, match="validated full assembly bundle"):
        _validate_quality_manifest(value)
    with pytest.raises(ValueError, match="validated full assembly bundle"):
        _validate_quality_manifest(value, assembly_bundle=True)


def test_builder_rejects_more_than_100_exclusions(tmp_path: Path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    rows = []
    for index in range(101):
        name = f"source-{index}.bin"
        (root / name).write_bytes(f"unique-{index}".encode())
        rows.append(f"{name},severe_frame_crop")
    source = tmp_path / "source.csv"
    source.write_text("path,reason\n" + "\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max_excluded_sources=100"):
        build_quality_exclusion_manifest(
            source_list=source,
            image_root=root,
            output_path=tmp_path / "manifest.json",
        )


def test_builder_rejects_symlink_source_list_and_source_ancestor(tmp_path: Path) -> None:
    root = tmp_path / "images"
    actual = root / "actual"
    actual.mkdir(parents=True)
    (actual / "a.bin").write_bytes(b"a")
    source = tmp_path / "source.csv"
    source.write_text("path,reason\nactual/a.bin,severe_frame_crop\n", encoding="utf-8")
    source_link = tmp_path / "source-link.csv"
    ancestor_link = root / "linked"
    try:
        os.symlink(source, source_link)
        os.symlink(actual, ancestor_link, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(ValueError, match="source_list must be a regular non-symlink"):
        build_quality_exclusion_manifest(
            source_list=source_link,
            image_root=root,
            output_path=tmp_path / "source-link-output.json",
        )

    source.write_text("path,reason\nlinked/a.bin,severe_frame_crop\n", encoding="utf-8")
    with pytest.raises(ValueError, match="symlink components"):
        build_quality_exclusion_manifest(
            source_list=source,
            image_root=root,
            output_path=tmp_path / "ancestor-link-output.json",
        )


def test_builder_rejects_source_list_and_image_root_ancestor_symlinks(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    root = real_parent / "images"
    root.mkdir(parents=True)
    (root / "a.bin").write_bytes(b"a")
    source = real_parent / "source.csv"
    source.write_text("path,reason\na.bin,severe_frame_crop\n", encoding="utf-8")
    linked_parent = tmp_path / "linked-parent"
    try:
        os.symlink(real_parent, linked_parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(ValueError, match="source_list path.*symlink components"):
        build_quality_exclusion_manifest(
            source_list=linked_parent / "source.csv",
            image_root=root,
            output_path=tmp_path / "source-list-ancestor.json",
        )
    with pytest.raises(ValueError, match="image_root path.*symlink components"):
        build_quality_exclusion_manifest(
            source_list=source,
            image_root=linked_parent / "images",
            output_path=tmp_path / "image-root-ancestor.json",
        )


def test_builder_rejects_symlink_output_parent(tmp_path: Path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    (root / "a.bin").write_bytes(b"a")
    source = tmp_path / "source.csv"
    source.write_text("path,reason\na.bin,severe_frame_crop\n", encoding="utf-8")
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    linked_output = tmp_path / "linked-output"
    try:
        os.symlink(real_output, linked_output, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="output parent"):
        build_quality_exclusion_manifest(
            source_list=source,
            image_root=root,
            output_path=linked_output / "manifest.json",
        )
