from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import scripts.audit_v4_near_duplicate_leakage as audit


@pytest.fixture(autouse=True)
def _small_protected_authority(monkeypatch):
    """Unit fixtures preserve the production count contract at a smaller scale."""
    monkeypatch.setattr(
        audit,
        "PROTECTED_REQUIRED_COUNTS",
        {
            "qx3_diagnostic_source_sha256": 1,
            "qx3_validation_source_sha256": 1,
            "hardware41_source_sha256": 0,
        },
    )


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pattern(seed: int, *, width: int = 96, height: int = 80) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    cv2.rectangle(
        image,
        (4 + seed % 11, 8),
        (width // 2, height // 2),
        ((seed * 17) % 256, 240, 20),
        -1,
    )
    cv2.circle(
        image,
        (width * 3 // 4, height * 2 // 3),
        8 + seed % 7,
        (250, (seed * 29) % 256, 180),
        -1,
    )
    return image


def _write_png(
    path: Path,
    pixels: np.ndarray,
    *,
    compression: int = 3,
) -> tuple[bytes, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(
        ".png",
        pixels,
        [cv2.IMWRITE_PNG_COMPRESSION, compression],
    )
    assert ok
    payload = encoded.tobytes()
    path.write_bytes(payload)
    return payload, _sha(payload)


def _dump(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def _protected_fixture(
    tmp_path: Path,
    *,
    seed: int = 9001,
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    root = tmp_path / "protected"
    payload, source_sha = _write_png(root / "qx3" / "source.png", _pattern(seed))
    crop_payload, crop_sha = _write_png(
        root / "qx3" / "crop.png", _pattern(seed + 1, width=72, height=64)
    )
    sources: dict[str, object] = {
        "schema": audit.PROTECTED_SOURCES_SCHEMA,
        "artifact_role": "protected_holdouts_not_training_or_model_selection",
        "status": "protected_holdouts_ready",
        "qx3_diagnostic_source_sha256": [source_sha],
        "qx3_validation_source_sha256": [source_sha],
        "hardware41_source_sha256": [],
        "known_audit_source_sha256": [],
        "calibration_source_sha256": [],
        "blind_test_source_sha256": [],
    }
    inventory: dict[str, object] = {
        "schema": audit.PROTECTED_INVENTORY_SCHEMA,
        "root": "protected",
        "objects": [
            {
                "cohort": "qx3_diagnostic",
                "view_kind": "source",
                "path": "qx3/source.png",
                "size": len(payload),
                "image_sha256": source_sha,
                "source_sha256": source_sha,
            },
            {
                "cohort": "qx3_diagnostic",
                "view_kind": "crop",
                "path": "qx3/crop.png",
                "size": len(crop_payload),
                "image_sha256": crop_sha,
                "source_sha256": source_sha,
            },
        ],
    }
    return (
        _dump(tmp_path / "protected_sources.json", sources),
        _dump(tmp_path / "protected_inventory.json", inventory),
        sources,
        inventory,
    )


def _asset(
    path: Path,
    role: str,
    sample_id: str,
    *,
    source_sha: str | None = None,
    image_sha: str | None = None,
    view_kind: str = "crop",
) -> audit.AuditAsset:
    actual = _sha(path.read_bytes())
    source = source_sha or actual
    image = image_sha or actual
    if view_kind == "source" and source_sha is None:
        source = image
    return audit.AuditAsset(
        path=path,
        role=role,
        cohort="candidate",
        view_kind=view_kind,
        sample_id=sample_id,
        source_sha256=source,
        image_sha256=image,
    )


def _complete_candidate_views(
    candidates: list[audit.AuditAsset],
) -> list[audit.AuditAsset]:
    result = list(candidates)
    existing = {
        (item.role, item.source_sha256)
        for item in result
        if item.view_kind == "source"
    }
    for item in list(result):
        key = (item.role, item.source_sha256)
        if item.view_kind != "crop" or key in existing:
            continue
        if _sha(item.path.read_bytes()) != item.source_sha256:
            raise AssertionError("test helper needs an explicit source view for this crop")
        result.append(
            audit.AuditAsset(
                path=item.path,
                role=item.role,
                cohort="candidate",
                view_kind="source",
                sample_id=f"source:{item.role}:{item.source_sha256}",
                source_sha256=item.source_sha256,
                image_sha256=item.source_sha256,
            )
        )
        existing.add(key)
    return result


def _bindings() -> dict[str, str]:
    return {
        "train": _sha(b"train manifest"),
        "model_validation": _sha(b"validation manifest"),
    }


def _base_candidates(tmp_path: Path) -> list[audit.AuditAsset]:
    _, _ = _write_png(tmp_path / "candidate" / "train.png", _pattern(11))
    _, _ = _write_png(tmp_path / "candidate" / "validation.png", _pattern(212))
    return [
        _asset(tmp_path / "candidate" / "train.png", "train", "train-1"),
        _asset(
            tmp_path / "candidate" / "validation.png",
            "model_validation",
            "validation-1",
        ),
    ]


def _build(
    tmp_path: Path,
    candidates: list[audit.AuditAsset] | None = None,
):
    protected_sources, protected_inventory, _, _ = _protected_fixture(tmp_path)
    return audit.build_near_duplicate_report(
        _complete_candidate_views(candidates or _base_candidates(tmp_path)),
        _bindings(),
        protected_sources,
        protected_inventory,
    )


def test_contract_constants_are_frozen():
    assert audit.REPORT_SCHEMA == "v4_near_duplicate_leakage_audit.v1"
    assert audit.ALGORITHM_ID == "oneexpo_phash_rot4_v1"
    assert audit.PHASH_DISTANCE == 4
    assert audit.MAX_ENCODED_BYTES == 64 * 1024 * 1024
    assert audit.MAX_IMAGE_PIXELS == 16_000_000
    assert audit.MAX_GRAPH_EDGES == 1_000_000
    assert (
        audit.RUNTIME_CODE_FINGERPRINT_SCHEMA
        == "oneexpo_auditor_runtime_code.v1"
    )


def test_auditor_binding_includes_deterministic_live_runtime_code_sha():
    first = audit._auditor_binding()
    second = audit._auditor_binding()

    assert first == second
    assert set(first) == {"path", "sha256", "runtime_code_sha256"}
    assert first["runtime_code_sha256"] == audit.runtime_code_fingerprint_sha256()
    assert len(first["runtime_code_sha256"]) == 64


def test_live_runtime_fingerprint_is_source_path_and_module_name_independent(tmp_path):
    module_path = tmp_path / "pristine_auditor_copy.py"
    module_path.write_bytes(Path(audit.__file__).read_bytes())
    module_name = "pristine_v4_near_duplicate_auditor_for_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = loaded
    try:
        spec.loader.exec_module(loaded)
        assert loaded.runtime_code_fingerprint_sha256() == (
            audit.runtime_code_fingerprint_sha256()
        )
    finally:
        sys.modules.pop(module_name, None)


def test_live_runtime_fingerprint_detects_loaded_code_after_file_restore(tmp_path):
    pristine = Path(audit.__file__).read_bytes()
    altered = pristine.replace(
        b"if distance <= threshold:",
        b"if distance < threshold: ",
        1,
    )
    assert altered != pristine
    module_path = tmp_path / "loaded_auditor.py"
    module_path.write_bytes(altered)
    module_name = "loaded_v4_near_duplicate_auditor_for_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = loaded
    try:
        spec.loader.exec_module(loaded)
        module_path.write_bytes(pristine)

        binding = loaded._auditor_binding()
        assert binding["sha256"] == _sha(pristine)
        assert binding["runtime_code_sha256"] == (
            loaded.runtime_code_fingerprint_sha256()
        )
        assert binding["runtime_code_sha256"] != (
            audit.runtime_code_fingerprint_sha256()
        )
    finally:
        sys.modules.pop(module_name, None)


def test_live_runtime_fingerprint_detects_top_level_side_effect_after_restore(
    tmp_path,
):
    pristine = Path(audit.__file__).read_bytes()
    altered = pristine.replace(
        b"from PIL import Image, UnidentifiedImageError\n",
        (
            b"from PIL import Image, UnidentifiedImageError\n"
            b"cv2.dct = cv2.idct\n"
        ),
        1,
    )
    assert altered != pristine
    module_path = tmp_path / "loaded_top_level_auditor.py"
    module_path.write_bytes(altered)
    module_name = "loaded_v4_top_level_auditor_for_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    original_dct = audit.cv2.dct
    sys.modules[module_name] = loaded
    try:
        spec.loader.exec_module(loaded)
        assert loaded.cv2.dct is loaded.cv2.idct
        module_path.write_bytes(pristine)

        binding = loaded._auditor_binding()
        assert binding["sha256"] == _sha(pristine)
        assert binding["runtime_code_sha256"] != (
            audit.runtime_code_fingerprint_sha256()
        )
    finally:
        audit.cv2.dct = original_dct
        sys.modules.pop(module_name, None)


def test_report_bytes_and_cluster_ids_are_candidate_order_independent(tmp_path):
    candidates = _base_candidates(tmp_path)
    complete_candidates = _complete_candidate_views(candidates)
    protected_sources, protected_inventory, _, _ = _protected_fixture(tmp_path)

    first, first_bytes, _ = audit.build_near_duplicate_report(
        complete_candidates,
        dict(reversed(list(_bindings().items()))),
        protected_sources,
        protected_inventory,
    )
    second, second_bytes, _ = audit.build_near_duplicate_report(
        list(reversed(complete_candidates)),
        _bindings(),
        protected_sources,
        protected_inventory,
    )

    assert first_bytes == second_bytes
    assert [item["cluster_id"] for item in first["clusters"]] == [
        item["cluster_id"] for item in second["clusters"]
    ]
    assert all("path" not in entry for entry in first["entries"])
    assert first["coverage"]["complete"] is True


def test_exact_image_across_candidate_roles_is_blocked(tmp_path):
    shared = tmp_path / "candidate" / "shared.png"
    _write_png(shared, _pattern(31))
    candidates = [
        _asset(shared, "train", "train-shared"),
        _asset(shared, "model_validation", "validation-shared"),
    ]

    report, _, _ = _build(tmp_path, candidates)

    assert report["ok"] is False
    assert report["status"] == "blocked"
    assert report["summary"]["blocking_multi_role_clusters"] >= 1
    assert any("exact_image_sha256" in edge["evidence"] for edge in report["edges"])


def test_reencoded_pixels_across_roles_are_blocked_by_phash(tmp_path):
    pixels = _pattern(41)
    _, first_sha = _write_png(tmp_path / "candidate" / "a.png", pixels, compression=1)
    _, second_sha = _write_png(tmp_path / "candidate" / "b.png", pixels, compression=9)
    assert first_sha != second_sha
    candidates = [
        _asset(tmp_path / "candidate" / "a.png", "train", "train-a"),
        _asset(
            tmp_path / "candidate" / "b.png",
            "model_validation",
            "validation-b",
        ),
    ]

    report, _, _ = _build(tmp_path, candidates)

    assert report["ok"] is False
    edge = next(
        item for item in report["edges"] if "perceptual_hash" in item["evidence"]
    )
    assert edge["distance"] == 0
    assert edge["blocking"] is True


def test_same_role_near_duplicate_is_counted_but_nonblocking(tmp_path):
    pixels = _pattern(51)
    _write_png(tmp_path / "candidate" / "a.png", pixels, compression=1)
    _write_png(tmp_path / "candidate" / "b.png", pixels, compression=9)
    _write_png(tmp_path / "candidate" / "validation.png", _pattern(852))
    candidates = [
        _asset(tmp_path / "candidate" / "a.png", "train", "train-a"),
        _asset(tmp_path / "candidate" / "b.png", "train", "train-b"),
        _asset(
            tmp_path / "candidate" / "validation.png",
            "model_validation",
            "validation",
        ),
    ]

    report, _, _ = _build(tmp_path, candidates)

    assert report["ok"] is True
    assert report["summary"]["same_role_duplicate_clusters_nonblocking"] >= 1
    assert report["authority"] == {
        "candidate_only": True,
        "label_authority": False,
        "blind_authority": False,
        "promotion_authority": False,
        "deployment_authority": False,
        "automatic_delete_or_relabel": False,
    }
    assert len(report["entries"]) == len(_complete_candidate_views(candidates)) + 2


def test_lossless_bucket_candidate_generation_honors_distance_boundary():
    at_boundary = (1 << 0) | (1 << 13) | (1 << 26) | (1 << 39)
    over_boundary = at_boundary | (1 << 52)

    assert audit._near_pairs_from_signatures([(0,), (at_boundary,)]) == [(0, 1, 4)]
    assert audit._near_pairs_from_signatures([(0,), (over_boundary,)]) == []


def test_rot4_bucket_search_matches_bruteforce_for_every_supported_radius():
    rng = np.random.default_rng(20260903)
    signatures = [
        tuple(sorted(int(value) for value in rng.integers(0, 1 << 64, 4, dtype=np.uint64)))
        for _ in range(32)
    ]
    signatures.extend(
        [
            (0, 1, 2, 3),
            (0xF, 0x10, 0x20, 0x40),
            (0xFF, 0x100, 0x200, 0x400),
        ]
    )
    for threshold in range(8):
        expected = sorted(
            (left, right, audit._signature_distance(signatures[left], signatures[right]))
            for left in range(len(signatures))
            for right in range(left + 1, len(signatures))
            if audit._signature_distance(signatures[left], signatures[right]) <= threshold
        )
        assert audit._near_pairs_from_signatures(signatures, threshold) == expected


def test_rot90_has_the_same_canonical_signature(tmp_path):
    pixels = _pattern(61, width=111, height=79)
    original, _ = _write_png(tmp_path / "original.png", pixels)
    rotated, _ = _write_png(
        tmp_path / "rotated.png",
        cv2.rotate(pixels, cv2.ROTATE_90_CLOCKWISE),
    )

    original_signature, _, _ = audit._phash_signature(original)
    rotated_signature, _, _ = audit._phash_signature(rotated)

    assert original_signature == rotated_signature
    assert [f"{value:016x}" for value in original_signature] == [
        "12933c6c6c6f3391",
        "14496bb669699716",
        "41e33e1c3cc3c6b8",
        "473969c639c4643b",
    ]


def test_near_edges_form_one_transitive_component(tmp_path, monkeypatch):
    paths = []
    payload_to_signature: dict[str, tuple[int, int, int, int]] = {}
    for name, seed, value in (("a", 71, 0), ("b", 72, 0xF), ("c", 73, 0xFF), ("v", 74, 0xFFFF0000)):
        payload, _ = _write_png(tmp_path / "candidate" / f"{name}.png", _pattern(seed))
        paths.append(tmp_path / "candidate" / f"{name}.png")
        payload_to_signature[_sha(payload)] = (value,) * 4
    protected_sources, protected_inventory, _, _ = _protected_fixture(tmp_path)
    for protected_name, value in (
        ("source.png", 0xAAAAAAAAAAAAAAAA),
        ("crop.png", 0x5555555555555555),
    ):
        protected_payload = (
            tmp_path / "protected" / "qx3" / protected_name
        ).read_bytes()
        payload_to_signature[_sha(protected_payload)] = (value,) * 4
    real_decode = audit._decode_verified_image

    def fake_signature(payload: bytes):
        _, width, height = real_decode(payload)
        return payload_to_signature[_sha(payload)], width, height

    monkeypatch.setattr(audit, "_phash_signature", fake_signature)
    candidates = [
        _asset(paths[0], "train", "a"),
        _asset(paths[1], "train", "b"),
        _asset(paths[2], "train", "c"),
        _asset(paths[3], "model_validation", "v"),
    ]

    report, _, _ = audit.build_near_duplicate_report(
        _complete_candidate_views(candidates),
        _bindings(),
        protected_sources,
        protected_inventory,
    )

    cluster = next(item for item in report["clusters"] if len(item["member_image_sha256s"]) == 3)
    assert len(cluster["member_asset_ids"]) == 6
    assert cluster["edge_count"] == 11
    assert cluster["roles"] == ["train"]
    assert cluster["blocking"] is False


def test_qx3_validation_subset_is_not_a_second_inventory_cohort(tmp_path):
    report, _, _ = _build(tmp_path)

    protected = [entry for entry in report["entries"] if entry["cohort"] != "candidate"]
    assert len(protected) == 2
    assert {entry["view_kind"] for entry in protected} == {"source", "crop"}
    assert {entry["cohort"] for entry in protected} == {"qx3_diagnostic"}


def test_empty_protected_authority_cannot_claim_complete(tmp_path):
    (tmp_path / "protected").mkdir()
    sources = {
        "schema": audit.PROTECTED_SOURCES_SCHEMA,
        "artifact_role": "protected_holdouts_not_training_or_model_selection",
        "status": "protected_holdouts_ready",
        **{field: [] for field in audit.PROTECTED_SOURCE_FIELDS},
    }
    inventory = {
        "schema": audit.PROTECTED_INVENTORY_SCHEMA,
        "root": "protected",
        "objects": [],
    }
    protected_sources = _dump(tmp_path / "protected_sources.json", sources)
    protected_inventory = _dump(tmp_path / "protected_inventory.json", inventory)

    with pytest.raises(audit.AuditError, match="must contain exactly"):
        audit.build_near_duplicate_report(
            _complete_candidate_views(_base_candidates(tmp_path)),
            _bindings(),
            protected_sources,
            protected_inventory,
        )


def test_candidate_crop_only_manifest_cannot_claim_complete(tmp_path):
    protected_sources, protected_inventory, _, _ = _protected_fixture(tmp_path)

    with pytest.raises(audit.AuditError, match="candidate source and crop views"):
        audit.build_near_duplicate_report(
            _base_candidates(tmp_path),
            _bindings(),
            protected_sources,
            protected_inventory,
        )


def test_candidate_source_without_its_crop_cannot_claim_complete(tmp_path):
    protected_sources, protected_inventory, _, _ = _protected_fixture(tmp_path)
    candidates = _complete_candidate_views(_base_candidates(tmp_path))
    source = tmp_path / "candidate" / "source-only.png"
    _write_png(source, _pattern(333))
    candidates.append(
        _asset(
            source,
            "train",
            "source-only",
            view_kind="source",
        )
    )

    with pytest.raises(audit.AuditError, match="missing_crop"):
        audit.build_near_duplicate_report(
            candidates,
            _bindings(),
            protected_sources,
            protected_inventory,
        )


def test_candidate_crop_with_protected_source_lineage_is_blocked(tmp_path):
    protected_sources, protected_inventory, sources, _ = _protected_fixture(tmp_path)
    protected_sha = sources["qx3_diagnostic_source_sha256"][0]  # type: ignore[index]
    candidates = _base_candidates(tmp_path)
    candidates[0] = _asset(
        candidates[0].path,
        "train",
        "protected-lineage-crop",
        source_sha=protected_sha,
    )
    candidates.append(
        audit.AuditAsset(
            path=tmp_path / "protected" / "qx3" / "source.png",
            role="train",
            cohort="candidate",
            view_kind="source",
            sample_id="protected-lineage-source",
            source_sha256=protected_sha,
            image_sha256=protected_sha,
        )
    )

    report, _, _ = audit.build_near_duplicate_report(
        _complete_candidate_views(candidates),
        _bindings(),
        protected_sources,
        protected_inventory,
    )

    assert report["ok"] is False
    assert any(
        edge["blocking"] and "source_sha256" in edge["evidence"]
        for edge in report["edges"]
    )


def test_missing_protected_source_view_fails_closed(tmp_path):
    protected_sources, protected_inventory, _, inventory = _protected_fixture(tmp_path)
    inventory["objects"][0]["view_kind"] = "crop"  # type: ignore[index]
    _dump(protected_inventory, inventory)

    with pytest.raises(audit.AuditError, match="cover every source exactly once"):
        audit.build_near_duplicate_report(
            _complete_candidate_views(_base_candidates(tmp_path)),
            _bindings(),
            protected_sources,
            protected_inventory,
        )


def test_missing_protected_crop_view_fails_closed(tmp_path):
    protected_sources, protected_inventory, _, inventory = _protected_fixture(tmp_path)
    crop = tmp_path / "protected" / "qx3" / "crop.png"
    crop.unlink()
    inventory["objects"] = [  # type: ignore[index]
        item for item in inventory["objects"]  # type: ignore[union-attr]
        if item["view_kind"] != "crop"
    ]
    _dump(protected_inventory, inventory)

    with pytest.raises(audit.AuditError, match="missing_crop"):
        audit.build_near_duplicate_report(
            _complete_candidate_views(_base_candidates(tmp_path)),
            _bindings(),
            protected_sources,
            protected_inventory,
        )


def test_duplicate_protected_source_views_fail_closed(tmp_path):
    protected_sources, protected_inventory, _, inventory = _protected_fixture(tmp_path)
    original = tmp_path / "protected" / "qx3" / "source.png"
    duplicate = tmp_path / "protected" / "qx3" / "source-copy.png"
    duplicate.write_bytes(original.read_bytes())
    duplicate_object = dict(inventory["objects"][0])  # type: ignore[index]
    duplicate_object["path"] = "qx3/source-copy.png"
    inventory["objects"].append(duplicate_object)  # type: ignore[union-attr]
    _dump(protected_inventory, inventory)

    with pytest.raises(audit.AuditError, match="exactly once"):
        audit.build_near_duplicate_report(
            _complete_candidate_views(_base_candidates(tmp_path)),
            _bindings(),
            protected_sources,
            protected_inventory,
        )


def test_unlisted_protected_root_file_fails_closed(tmp_path):
    protected_sources, protected_inventory, _, _ = _protected_fixture(tmp_path)
    _write_png(tmp_path / "protected" / "qx3" / "unlisted.png", _pattern(9002))

    with pytest.raises(audit.AuditError, match="exactly enumerate"):
        audit.build_near_duplicate_report(
            _complete_candidate_views(_base_candidates(tmp_path)),
            _bindings(),
            protected_sources,
            protected_inventory,
        )


def test_misclassified_protected_source_fails_closed(tmp_path):
    protected_sources, protected_inventory, _, inventory = _protected_fixture(tmp_path)
    inventory["objects"][0]["cohort"] = "hardware41"  # type: ignore[index]
    _dump(protected_inventory, inventory)

    with pytest.raises(audit.AuditError, match="misclassified"):
        audit.build_near_duplicate_report(
            _complete_candidate_views(_base_candidates(tmp_path)),
            _bindings(),
            protected_sources,
            protected_inventory,
        )


def test_extra_protected_source_fails_closed(tmp_path):
    protected_sources, protected_inventory, _, inventory = _protected_fixture(tmp_path)
    extra_payload, extra_sha = _write_png(
        tmp_path / "protected" / "qx3" / "extra.png",
        _pattern(9010),
    )
    inventory["objects"].append(  # type: ignore[union-attr]
        {
            "cohort": "qx3_diagnostic",
            "view_kind": "source",
            "path": "qx3/extra.png",
            "size": len(extra_payload),
            "image_sha256": extra_sha,
            "source_sha256": extra_sha,
        }
    )
    _dump(protected_inventory, inventory)

    with pytest.raises(audit.AuditError, match="extra source SHA"):
        audit.build_near_duplicate_report(
            _complete_candidate_views(_base_candidates(tmp_path)),
            _bindings(),
            protected_sources,
            protected_inventory,
        )


def test_duplicate_json_key_is_rejected_as_non_exact_schema(tmp_path):
    protected_sources, protected_inventory, _, _ = _protected_fixture(tmp_path)
    duplicate = protected_sources.read_text(encoding="utf-8")
    duplicate = duplicate[:-1] + ',"status":"protected_holdouts_ready"}'
    protected_sources.write_text(duplicate, encoding="utf-8")

    with pytest.raises(audit.AuditError, match="duplicate object key"):
        audit.build_near_duplicate_report(
            _complete_candidate_views(_base_candidates(tmp_path)),
            _bindings(),
            protected_sources,
            protected_inventory,
        )


def test_candidate_image_sha_mismatch_fails_closed(tmp_path):
    candidates = _base_candidates(tmp_path)
    candidates[0] = audit.AuditAsset(
        **{**candidates[0].__dict__, "image_sha256": "0" * 64}
    )

    with pytest.raises(audit.AuditError, match="SHA-256"):
        _build(tmp_path, candidates)


def test_non_image_payload_fails_decode(tmp_path):
    bad = tmp_path / "candidate" / "not-image.bin"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"this is not an image")
    candidates = _base_candidates(tmp_path)
    candidates[0] = _asset(bad, "train", "bad")

    with pytest.raises(audit.AuditError, match="Pillow"):
        _build(tmp_path, candidates)


def test_encoded_byte_cap_fails_before_decode(tmp_path, monkeypatch):
    candidates = _base_candidates(tmp_path)
    monkeypatch.setattr(audit, "MAX_ENCODED_BYTES", 16)

    with pytest.raises(audit.AuditError, match="byte cap"):
        _build(tmp_path, candidates)


def test_pixel_cap_fails_before_opencv_decode(tmp_path, monkeypatch):
    candidates = _base_candidates(tmp_path)
    monkeypatch.setattr(audit, "MAX_IMAGE_PIXELS", 100)

    with pytest.raises(audit.AuditError, match="pixel cap"):
        _build(tmp_path, candidates)


def test_final_symlink_is_rejected(tmp_path):
    target = tmp_path / "candidate" / "target.png"
    _write_png(target, _pattern(81))
    link = tmp_path / "candidate" / "link.png"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink unavailable: {error}")
    candidates = _base_candidates(tmp_path)
    candidates[0] = _asset(link, "train", "link")

    with pytest.raises(audit.AuditError, match="symlink"):
        _build(tmp_path, candidates)


def test_hardlink_is_rejected(tmp_path):
    target = tmp_path / "candidate" / "target.png"
    _write_png(target, _pattern(82))
    hardlink = tmp_path / "candidate" / "hardlink.png"
    try:
        os.link(target, hardlink)
    except OSError as error:
        pytest.skip(f"hardlink unavailable: {error}")
    candidates = _base_candidates(tmp_path)
    candidates[0] = _asset(hardlink, "train", "hardlink")

    with pytest.raises(audit.AuditError, match="hard link"):
        _build(tmp_path, candidates)


def test_normal_short_os_reads_are_accumulated(tmp_path, monkeypatch):
    candidates = _base_candidates(tmp_path)
    real_read = audit.os.read

    def short_read(descriptor: int, amount: int) -> bytes:
        return real_read(descriptor, min(amount, 7))

    monkeypatch.setattr(audit.os, "read", short_read)

    report, _, _ = _build(tmp_path, candidates)
    assert report["coverage"]["complete"] is True


@pytest.mark.skipif(os.name == "nt", reason="POSIX mutation semantics")
def test_poison_then_restore_during_descriptor_read_fails(tmp_path, monkeypatch):
    candidates = _base_candidates(tmp_path)
    target = candidates[0].path
    target_inode = target.stat().st_ino
    original = target.read_bytes()
    real_read = audit.os.read
    triggered = False

    def poisoning_read(descriptor: int, amount: int) -> bytes:
        nonlocal triggered
        chunk = real_read(descriptor, amount)
        if not triggered and os.fstat(descriptor).st_ino == target_inode:
            triggered = True
            target.write_bytes(b"x" * len(original))
            target.write_bytes(original)
        return chunk

    monkeypatch.setattr(audit.os, "read", poisoning_read)

    with pytest.raises(audit.AuditError, match="identity changed"):
        _build(tmp_path, candidates)


@pytest.mark.skipif(os.name == "nt", reason="POSIX FIFO")
def test_fifo_is_rejected_without_blocking(tmp_path):
    fifo = tmp_path / "candidate" / "fifo"
    fifo.parent.mkdir(parents=True)
    os.mkfifo(fifo)
    candidates = _base_candidates(tmp_path)
    fifo_sha = _sha(b"fifo")
    candidates[0] = audit.AuditAsset(
        path=fifo,
        role="train",
        cohort="candidate",
        view_kind="crop",
        sample_id="fifo",
        source_sha256=fifo_sha,
        image_sha256=fifo_sha,
    )
    candidates.append(
        audit.AuditAsset(
            path=fifo,
            role="train",
            cohort="candidate",
            view_kind="source",
            sample_id="fifo-source",
            source_sha256=fifo_sha,
            image_sha256=fifo_sha,
        )
    )

    with pytest.raises(audit.AuditError, match="regular file"):
        _build(tmp_path, candidates)


@pytest.mark.skipif(os.name == "nt", reason="POSIX ancestor symlink")
def test_ancestor_symlink_is_rejected(tmp_path):
    real = tmp_path / "real"
    image = real / "image.png"
    _write_png(image, _pattern(83))
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    candidates = _base_candidates(tmp_path)
    candidates[0] = _asset(linked / "image.png", "train", "ancestor-link")

    with pytest.raises(audit.AuditError, match="symlink"):
        _build(tmp_path, candidates)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction/reparse point")
def test_windows_ancestor_junction_is_rejected(tmp_path):
    real = tmp_path / "real"
    image = real / "image.png"
    _write_png(image, _pattern(84))
    junction = tmp_path / "junction"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(real)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"junction unavailable: {result.stderr or result.stdout}")
    candidates = _base_candidates(tmp_path)
    candidates[0] = _asset(junction / "image.png", "train", "junction")
    try:
        with pytest.raises(audit.AuditError, match="reparse"):
            _build(tmp_path, candidates)
    finally:
        os.rmdir(junction)


def test_phash_receives_verified_exact_bytes_not_a_path(tmp_path, monkeypatch):
    candidates = _base_candidates(tmp_path)
    protected_sources, protected_inventory, _, _ = _protected_fixture(tmp_path)
    expected_payloads = {
        item.path.read_bytes() for item in candidates
    } | {
        (tmp_path / "protected" / "qx3" / name).read_bytes()
        for name in ("source.png", "crop.png")
    }
    observed: list[bytes] = []
    real_signature = audit._phash_signature

    def spy(payload: bytes):
        assert type(payload) is bytes
        assert payload in expected_payloads
        observed.append(payload)
        return real_signature(payload)

    monkeypatch.setattr(audit, "_phash_signature", spy)

    report, _, records = audit.build_near_duplicate_report(
        _complete_candidate_views(candidates),
        _bindings(),
        protected_sources,
        protected_inventory,
    )

    assert report["ok"] is True
    assert len(observed) == 2 * len(records)


def test_reverify_rejects_same_bytes_replaced_at_new_identity(tmp_path):
    _, _, records = _build(tmp_path)
    target = records[0].path
    replacement = target.with_suffix(".replacement")
    replacement.write_bytes(target.read_bytes())
    os.replace(replacement, target)

    with pytest.raises(audit.AuditError, match="identity"):
        audit.reverify_assets(records)


def test_atomic_publish_rejects_temp_path_replacement(tmp_path):
    output = tmp_path / "audit.json"
    replacement_blocked_by_os = False

    def replace_temporary() -> None:
        nonlocal replacement_blocked_by_os
        temporary, = tmp_path.glob(".audit.json.*")
        try:
            temporary.unlink()
            temporary.write_bytes(b"forged\n")
        except PermissionError:
            replacement_blocked_by_os = True

    try:
        audit._atomic_no_overwrite(
            output,
            b'{"ok":true}\n',
            pre_publish=replace_temporary,
        )
    except audit.AuditError as error:
        assert "temporary output" in str(error)
        assert not output.exists()
    else:
        assert output.read_bytes() == b'{"ok":true}\n'


def test_atomic_publish_rolls_back_when_post_link_inputs_changed(tmp_path):
    output = tmp_path / "audit.json"

    def reject_changed_input() -> None:
        assert output.is_file()
        raise audit.AuditError("input changed in publication window")

    with pytest.raises(audit.AuditError, match="publication window"):
        audit._atomic_no_overwrite(
            output,
            b'{"ok":true}\n',
            pre_publish=reject_changed_input,
        )
    assert not output.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permits unlinking an open file")
def test_atomic_publish_rolls_back_if_temporary_link_is_removed(tmp_path):
    output = tmp_path / "audit.json"

    def remove_temporary_then_reject() -> None:
        temporary, = tmp_path.glob(".audit.json.*")
        temporary.unlink()
        raise audit.AuditError("input changed after temporary unlink")

    with pytest.raises(audit.AuditError, match="temporary unlink"):
        audit._atomic_no_overwrite(
            output,
            b'{"ok":true}\n',
            pre_publish=remove_temporary_then_reject,
        )
    assert not output.exists()


def test_duplicate_heavy_phash_graph_fails_at_fixed_edge_cap():
    signatures = [(0,)] * 20
    with pytest.raises(audit.AuditError, match="edge cap 100"):
        audit._near_pairs_from_signatures(signatures, max_pairs=100)


def _write_explicit_manifest(path: Path, role: str, image: Path) -> None:
    fields = [
        "filepath",
        "sample_id",
        "role",
        "view_kind",
        "source_sha256",
        "image_sha256",
    ]
    image_sha = _sha(image.read_bytes())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for view_kind in ("source", "crop"):
            writer.writerow(
                {
                    "filepath": image.relative_to(path.parent).as_posix(),
                    "sample_id": f"{role}-{view_kind}",
                    "role": role,
                    "view_kind": view_kind,
                    "source_sha256": image_sha,
                    "image_sha256": image_sha,
                }
            )


def test_legacy_candidate_row_materializes_crop_and_absolute_source(tmp_path):
    source = tmp_path / "source.png"
    crop = tmp_path / "crop.png"
    _write_png(source, _pattern(89))
    _write_png(crop, _pattern(90))
    manifest = tmp_path / "train.csv"
    fields = [
        "filepath",
        "source_filepath",
        "sample_id",
        "role",
        "source_sha256",
        "image_sha256",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "filepath": crop.name,
                "source_filepath": str(source.resolve()),
                "sample_id": "sample",
                "role": "train",
                "source_sha256": _sha(source.read_bytes()),
                "image_sha256": _sha(crop.read_bytes()),
            }
        )

    assets, manifest_sha = audit._load_candidate_manifest("train", manifest)

    assert manifest_sha == _sha(manifest.read_bytes())
    assert {item.view_kind for item in assets} == {"source", "crop"}
    assert next(item for item in assets if item.view_kind == "source").path == source.resolve()


def test_cli_writes_atomically_and_refuses_overwrite(tmp_path, monkeypatch):
    train_image = tmp_path / "train.png"
    validation_image = tmp_path / "validation.png"
    _write_png(train_image, _pattern(91))
    _write_png(validation_image, _pattern(192))
    train_manifest = tmp_path / "train.csv"
    validation_manifest = tmp_path / "validation.csv"
    _write_explicit_manifest(train_manifest, "train", train_image)
    _write_explicit_manifest(validation_manifest, "model_validation", validation_image)
    protected_sources, protected_inventory, _, _ = _protected_fixture(tmp_path)
    output = tmp_path / "audit.json"
    args = [
        "--candidate-manifest",
        f"train={train_manifest}",
        "--candidate-manifest",
        f"model_validation={validation_manifest}",
        "--protected-sources",
        str(protected_sources),
        "--protected-inventory",
        str(protected_inventory),
        "--output",
        str(output),
    ]
    reverify_calls = 0
    real_reverify = audit.reverify_assets

    def counting_reverify(records):
        nonlocal reverify_calls
        reverify_calls += 1
        return real_reverify(records)

    monkeypatch.setattr(audit, "reverify_assets", counting_reverify)

    assert audit.main(args) == 0
    assert reverify_calls == 2
    parsed = json.loads(output.read_bytes())
    assert parsed["schema"] == audit.REPORT_SCHEMA
    with pytest.raises(audit.AuditError, match="overwrite"):
        audit.main(args)
