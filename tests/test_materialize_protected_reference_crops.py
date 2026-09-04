"""Historical ROI byte checks only; fixtures do not establish labels or accuracy."""
import base64
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import materialize_protected_reference_crops as roi


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def data(tmp_path):
    metadata, captures, extra = (tmp_path / name for name in ("metadata", "captures", "supplements"))
    for directory in (metadata, captures, extra): directory.mkdir()
    known, captured, references = {}, [], []
    for index in range(4):
        path = (captures if index < 3 else extra) / f"source-{index}.jpg"
        pixels = np.full((60, 100, 3), (index * 40 + 30, 70, 100), np.uint8)
        pixels[15:35, 25:70] = (121, index * 50, 160)
        assert cv2.imwrite(str(path), pixels)
        digest = sha(path)
        roles = []
        if index != 2:
            known[digest] = {"sha256": digest, "bbox": [10., 10., 70., 40.] if index < 2 else None,
                            "bbox_source": "audited_override" if index == 0 else "low_conf_expected_class",
                            "label": "not-a-target", "source_image": "/old/moved-location.jpg"}
            roles.append("known_audit")
        if index < 3:
            captured.append({"sha256": digest, "image_ref": path.name,
                             "deployed": {"bbox": [50., 30., 90., 50.], "class_name": "not-a-target", "confidence": .99}})
            roles.append("capture")
        references.append({"source_sha256": digest, "source_path_b64": base64.urlsafe_b64encode(os.fsencode(path)).decode(),
                           "source_bytes": path.stat().st_size, "image_height": 60, "image_width": 100,
                           "roles": sorted(roles), "source_phash64": "0" * 16})
    references.append({"source_sha256": "f" * 64, "source_path_b64": base64.urlsafe_b64encode(b"/never/read/qx3.jpg").decode(),
                       "source_bytes": 1, "image_height": 1, "image_width": 1, "roles": ["qx3"], "source_phash64": "0" * 16})
    fingerprint = {"schema": "protected_image_fingerprint_snapshot.v1", "status": "snapshot_complete",
                   "snapshot_only": True, "consumer_must_rehash_sources": True,
                   "training_authorized": False, "deployment_authorized": False,
                   "blind_test_authorized": False, "selection_authorized": False,
                   "expected_sources": 5, "verified_sources": 5, "missing_sources": 0, "records": references}
    def save():
        paths = [metadata / name for name in ("known.json", "captures.json", "fingerprints.json")]
        for path, value in zip(paths, (known, captured, fingerprint)):
            path.write_text(json.dumps(value), encoding="utf-8")
        return dict(known_audit=paths[0], known_audit_sha256=sha(paths[0]), capture_inventory=paths[1],
                    capture_inventory_sha256=sha(paths[1]), capture_root=captures, protected_report=paths[2],
                    protected_report_sha256=sha(paths[2]), code_pins={name: sha(p) for name, p in roi.code_paths().items()},
                    output=tmp_path / "result", expected_sources=4, expected_crops=3, expected_missing=1)
    return save, known, captured, fingerprint


def test_priority_provenance_actual_crop_bytes_and_missing_sources(data):
    save, known, captures, fingerprint = data
    args = save()
    report = roi.materialize(**args)
    assert report == json.loads((args["output"] / "report.json").read_bytes())
    assert report["schema"] == "protected_reference_roi.v1"
    assert report["raw_source_count"] == 4 and report["reference_roi_count"] == 3 and report["missing_reference_count"] == 1
    assert report["reference_counts"] == {"known_audit_reference": 2, "historical_deployed_reference": 1}
    assert all(report[key] is False for key in roi.AUTHORITY)
    by_sha = {row["source_sha256"]: row for row in report["records"]}
    for reference in fingerprint["records"][:-1]:
        digest = reference["source_sha256"]
        row = by_sha[digest]
        assert row["roles"] == reference["roles"] and row["object_absence_established"] is False
        assert all(k not in row for k in ("label", "class_id", "class_name", "dent", "foreign_material", "objectness"))
        if row["crop"] is None:
            assert row["status"] == "missing_reference" and row["reference"] is None
            continue
        raw_path = roi.decode_path(row["source_path_b64"])
        input_pixels = cv2.imread(str(raw_path))
        expected, bounds = roi.crop_contract.crop_and_letterbox_bgr(input_pixels, row["reference"]["bbox_xyxy"], padding=.08, size=320, fill=114)
        ok, encoded = cv2.imencode(".jpg", expected, [cv2.IMWRITE_JPEG_QUALITY, 92])
        assert ok and (args["output"] / row["crop"]["path"]).read_bytes() == encoded.tobytes()
        assert row["crop"]["bounds_xyxy"] == list(bounds)
        if digest in known:
            assert row["reference"]["bbox_xyxy"] == known[digest]["bbox"]
            assert row["reference"]["bbox_source"] == known[digest]["bbox_source"]
            assert row["reference"]["metadata_sha256"] == args["known_audit_sha256"]
        else:
            assert row["reference"]["kind"] == "historical_deployed_reference"
            assert row["reference"]["metadata_sha256"] == args["capture_inventory_sha256"]
    assert not list(args["output"].rglob("*.csv"))


def test_all_missing_is_explicit_and_does_not_generate_empty_or_fake_crops(data):
    save, known, captures, _ = data
    for row in known.values(): row["bbox"] = None
    for row in captures: row["deployed"]["bbox"] = None
    args = save() | {"expected_crops": 0, "expected_missing": 4}
    report = roi.materialize(**args)
    assert report["reference_roi_count"] == 0 and report["missing_reference_count"] == 4
    assert all(row["reference"] is row["crop"] is None and row["object_absence_established"] is False for row in report["records"])
    assert not (args["output"] / "crops").exists()


@pytest.mark.parametrize("bbox", [[], [1, 2, 3], [1, 2, 1, 5], [1, 2, float("nan"), 5],
                                  [1, 2, float("inf"), 5], [True, 2, 4, 5], ["1", 2, 4, 5], [101, 61, 110, 70], "1,2,3,4"])
def test_invalid_known_reference_does_not_silently_use_deployed_fallback(data, bbox):
    save, known, _, _ = data
    next(iter(known.values()))["bbox"] = bbox
    args = save()
    with pytest.raises(ValueError): roi.materialize(**args)
    assert not args["output"].exists()


def test_missing_known_bbox_uses_historical_reference_without_label_authority(data):
    save, known, _, _ = data
    first_sha = next(iter(known))
    known[first_sha]["bbox"] = None
    args = save()
    report = roi.materialize(**args)
    row = next(row for row in report["records"] if row["source_sha256"] == first_sha)
    assert row["reference"]["kind"] == "historical_deployed_reference"
    assert row["crop"]["provenance"] == "historical_deployed_reference"
    assert report["runtime_detector_executed"] is report["label_authority"] is False


@pytest.mark.parametrize("fault", ["capture_duplicate", "unknown_source", "missing_source", "role", "path", "incomplete", "known_sha"])
def test_metadata_membership_path_role_and_coverage_mismatch_fail(data, fault):
    save, known, captured, fp = data
    if fault == "capture_duplicate": captured.append(captured[0].copy())
    elif fault == "unknown_source": known["e" * 64] = {"bbox": None}
    elif fault == "missing_source": known.pop(next(iter(known)))
    elif fault == "role": fp["records"][0]["roles"] = ["capture"]
    elif fault == "path": captured[0]["image_ref"] = captured[1]["image_ref"]
    elif fault == "incomplete": fp["missing_sources"] = 1
    else: next(iter(known.values()))["sha256"] = "e" * 64
    args = save()
    with pytest.raises(ValueError): roi.materialize(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("which", ["known_audit", "capture_inventory", "protected_report", "code"])
def test_input_pin_changes_fail_closed(data, which):
    save, _, _, _ = data
    args = save()
    if which == "code": args["code_pins"]["materialize_protected_reference_crops.py"] = "e" * 64
    else: args[which + "_sha256"] = "e" * 64
    with pytest.raises(ValueError): roi.materialize(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("which", ["source_bytes", "image_width", "pixels"])
def test_real_source_hash_size_and_decoded_dimensions_are_checked(data, which):
    save, _, _, fp = data
    row = fp["records"][0]
    if which in row: row[which] += 1
    else: roi.decode_path(row["source_path_b64"]).write_bytes(b"broken source")
    args = save()
    with pytest.raises(ValueError): roi.materialize(**args)
    assert not (args["output"] / "report.json").exists()
    assert (args["output"] / "failed.json").exists()


@pytest.mark.parametrize("key", ["expected_sources", "expected_crops", "expected_missing"])
def test_expected_counts_validate_but_do_not_choose_or_delete_sources(data, key):
    save, _, _, _ = data
    args = save()
    args[key] += 1
    with pytest.raises(ValueError, match="count mismatch"): roi.materialize(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("target", ["known_audit", "source", "crop"])
def test_postpublication_mutations_revoke_report(data, monkeypatch, target):
    save, _, _, fp = data
    args = save()
    read = roi.files.read_file
    changed = False
    def mutation(path, *a, **kw):
        nonlocal changed
        if not changed and (args["output"] / "report.json").exists():
            selected = (args["known_audit"] if target == "known_audit" else roi.decode_path(fp["records"][0]["source_path_b64"])
                        if target == "source" else next((args["output"] / "crops").glob("*.jpg")))
            selected.write_bytes(b"changed")
            changed = True
        return read(path, *a, **kw)
    monkeypatch.setattr(roi.files, "read_file", mutation)
    with pytest.raises(ValueError): roi.materialize(**args)
    assert changed and not (args["output"] / "report.json").exists()
    assert (args["output"] / "failed.json").exists()


@pytest.mark.parametrize("which", ["existing", "metadata", "capture"])
def test_output_reuse_overlap_and_original_protection(data, which):
    save, _, _, _ = data
    args = save()
    if which == "existing": args["output"].mkdir()
    elif which == "metadata": args["output"] = args["known_audit"].parent / "new"
    else: args["output"] = args["capture_root"] / "new"
    with pytest.raises(ValueError): roi.materialize(**args)
    assert not (args["output"] / "report.json").exists()


def test_cpu_cli_and_compact_output_expose_no_label_or_yolo_authority(data, capsys):
    save, _, _, _ = data
    args = save()
    cli = [part for key, value in args.items() if key != "code_pins" for part in ("--" + key.replace("_", "-"), str(value))]
    for name, value in args["code_pins"].items(): cli.extend(["--code-pin", name + "=" + value])
    assert roi.main(cli) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["raw_source_count"] == 4 and summary["reference_roi_count"] == 3
    assert all(summary[key] is False for key in roi.AUTHORITY)


@pytest.mark.parametrize("path", ["../elsewhere.jpg", "/outside.jpg", "day//a.jpg", "day/./a.jpg", "day\\a.jpg"])
def test_capture_relative_paths_cannot_escape_or_change_identity(data, path):
    save, _, captured, _ = data
    captured[0]["image_ref"] = path
    args = save()
    with pytest.raises(ValueError): roi.materialize(**args)
    assert not args["output"].exists()


def test_source_read_limit_is_checked_before_materialization(data, monkeypatch):
    save, _, _, _ = data
    args = save()
    monkeypatch.setattr(roi.files, "CROP_LIMIT", 10)
    with pytest.raises(ValueError, match="byte limit"): roi.materialize(**args)
    assert not args["output"].exists()


def test_code_file_mutation_is_detected_without_editing_real_source(data, tmp_path, monkeypatch):
    copied = tmp_path / "code"
    copied.mkdir()
    paths = {}
    for name, path in roi.code_paths().items():
        paths[name] = copied / name
        paths[name].write_bytes(path.read_bytes())
    monkeypatch.setattr(roi, "code_paths", lambda: paths)
    save, _, _, _ = data
    args = save()
    real_encode = cv2.imencode
    changed = False
    def change(*a, **kw):
        nonlocal changed
        if not changed:
            paths["materialize_protected_reference_crops.py"].write_bytes(b"changed code snapshot")
            changed = True
        return real_encode(*a, **kw)
    monkeypatch.setattr(cv2, "imencode", change)
    with pytest.raises(ValueError): roi.materialize(**args)
    assert changed and not (args["output"] / "report.json").exists()
    assert (args["output"] / "failed.json").exists()


def test_crop_symlink_swap_is_blocked_before_writing_elsewhere(data, tmp_path, monkeypatch):
    save, _, _, _ = data
    args = save()
    outside = tmp_path / "outside"
    outside.mkdir()
    probe = tmp_path / "link-probe"
    try: probe.symlink_to(outside, target_is_directory=True)
    except OSError: pytest.skip("symlink creation unavailable")
    probe.unlink()
    actual_encode = cv2.imencode
    def swap(*a, **kw):
        (args["output"] / "crops").symlink_to(outside, target_is_directory=True)
        return actual_encode(*a, **kw)
    monkeypatch.setattr(cv2, "imencode", swap)
    with pytest.raises(ValueError, match="symlink"): roi.materialize(**args)
    assert list(outside.iterdir()) == []
    assert not (args["output"] / "report.json").exists()
