"""Synthetic CPU lineage probes; these fixtures are not NAS source-link evidence."""
import base64
import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import convert_remainder, convert_v2


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def b64(path):
    return base64.urlsafe_b64encode(os.fsencode(path)).decode("ascii")


def _pair_tree(tmp_path, count=7):
    split = tmp_path / "Training"
    folder = "2.직접촬영_01.금속캔_001.철캔"
    sources = split / "01.원천데이터" / f"TS_{folder}_1"
    labels = split / "02.라벨링데이터" / f"TL_{folder}"
    sources.mkdir(parents=True)
    labels.mkdir(parents=True)
    for i in range(count):
        (labels / f"{i:02}.json").write_text("{}")
        if i != 2:
            (sources / f"{i:02}.png").write_bytes(b"metadata candidate only")
    return split, sources, labels


@pytest.mark.parametrize("remainder", [False, True])
@pytest.mark.parametrize("max_index", [None, 0, 2])
def test_candidate_sequence_matches_actual_legacy_collectors(tmp_path, remainder, max_index):
    from scripts import audit_legacy_aihub_links as links
    split, _, _ = _pair_tree(tmp_path)
    expected = (convert_remainder.collect_remainder_pairs(split, 2) if remainder
                else convert_v2.collect_pairs(split, 2))
    expected = [(Path(a), Path(b)) for a, b in expected]
    if max_index is not None:
        expected = expected[:max_index + 1]
    assert list(links.iter_pairs(split, 2, remainder=remainder, max_index=max_index)) == expected


def test_requested_ordinal_early_stop_never_scans_later_label_folder(tmp_path, monkeypatch):
    from scripts import audit_legacy_aihub_links as links
    split, _, labels = _pair_tree(tmp_path)
    late = labels.parent / "TL_9.never_scan"
    late.mkdir()
    original = Path.rglob
    def guard(path, pattern, *args, **kwargs):
        if path == late:
            raise AssertionError("scanned beyond the requested candidate ordinal")
        return original(path, pattern, *args, **kwargs)
    monkeypatch.setattr(Path, "rglob", guard)
    rows = list(links.iter_pairs(split, 0, max_index=0))
    assert len(rows) == 1 and rows[0][1].name == "00.json"


def test_source_directory_order_precedes_extension_order(tmp_path, monkeypatch):
    from scripts import audit_legacy_aihub_links as links
    split, sources, labels = _pair_tree(tmp_path, count=1)
    earlier = sources.with_name(sources.name[:-1] + "2")
    earlier.mkdir()
    (earlier / "00.png").write_bytes(b"first source directory png")
    (sources / "00.jpg").write_bytes(b"second source directory jpg")
    original = Path.iterdir
    def ordered(path):
        return iter([earlier, sources]) if path == sources.parent else original(path)
    monkeypatch.setattr(Path, "iterdir", ordered)
    expected = convert_v2.collect_pairs(split, 0)
    found = list(links.iter_pairs(split, 0, max_index=0))
    assert found == [(Path(a), Path(b)) for a, b in expected]
    assert found[0][0] == earlier / "00.png"


def test_jpg_extension_wins_before_png_in_same_source_directory(tmp_path):
    from scripts import audit_legacy_aihub_links as links
    split, sources, _ = _pair_tree(tmp_path, count=1)
    (sources / "00.jpg").write_bytes(b"jpg wins")
    assert list(links.iter_pairs(split, 0, max_index=0))[0][0].suffix == ".jpg"


@pytest.fixture
def probe(tmp_path):
    dataset = tmp_path / "aihub"
    split = dataset / "01-1.정식개방데이터" / "Training"
    folder = "2.직접촬영_01.금속캔_001.철캔"
    source = split / "01.원천데이터" / f"TS_{folder}_1" / "source.png"
    label = split / "02.라벨링데이터" / f"TL_{folder}" / "source.json"
    source.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    pixels = np.random.default_rng(20260904).integers(0, 256, (503, 1001, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", pixels)
    assert ok
    source.write_bytes(encoded.tobytes())
    payload = {"IMAGE_INFO": {"FILE_NAME": source.name, "IMAGE_WIDTH": 1001, "IMAGE_HEIGHT": 503},
               "ANNOTATION_INFO": [{"CLASS": "금속캔", "DETAILS": "철캔", "POINTS": [[200, 100, 300, 200]]}]}
    label.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    legacy = tmp_path / "yolo_dataset_9class_v2" / "train" / "images" / "train_0000000.jpg"
    legacy_label = legacy.parent.parent / "labels" / "train_0000000.txt"
    legacy.parent.mkdir(parents=True)
    legacy_label.parent.mkdir()
    # Actual legacy worker constructs this fixture using floor resize/JPEG90.
    result, _ = convert_v2.worker((str(source), str(label), str(legacy), str(legacy_label), 640))
    assert result == "ok"
    # The historical NAS worker ran on Linux; model its LF sidecar explicitly.
    legacy_label.write_bytes(legacy_label.read_bytes().replace(b"\r\n", b"\n"))
    protected = tmp_path / "protected" / "report.json"
    protected.parent.mkdir()
    data = {"schema": "protected_image_fingerprint_snapshot.v1", "status": "snapshot_complete",
            "expected_sources": 1, "verified_sources": 1, "missing_sources": 0,
            "training_authorized": False, "deployment_authorized": False,
            "records": [{"source_sha256": sha(legacy), "source_path_b64": b64(legacy), "roles": ["qx3"]}]}
    v3 = tmp_path / "v3" / "manifest.csv"
    v3.parent.mkdir()
    sid = "a" * 20
    with v3.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "split", "source_path_b64"])
        writer.writeheader()
        writer.writerow({"source_id": sid, "split": "training", "source_path_b64": b64(source)})
    output = tmp_path / "probe"
    def save():
        protected.write_text(json.dumps(data), encoding="utf-8")
        return {"protected_report": protected, "protected_report_sha256": sha(protected),
                "v3_manifest": v3, "v3_manifest_sha256": sha(v3),
                "converter": ROOT / "scripts/convert_v2.py", "converter_sha256": sha(ROOT / "scripts/convert_v2.py"),
                "remainder": ROOT / "scripts/convert_remainder.py", "remainder_sha256": sha(ROOT / "scripts/convert_remainder.py"),
                "dataset_dir": dataset, "output": output, "max_per_kind": 1}
    return save, data, source, label, legacy, legacy_label, v3, output


def test_real_legacy_worker_bytes_match_link_but_never_complete_lineage_or_training(probe):
    from scripts import audit_legacy_aihub_links as links
    save, data, source, label, legacy, legacy_label, v3, output = probe
    before = {p: sha(p) for p in (source, label, legacy, legacy_label, v3)}
    report = links.audit_links(**save())
    row = report["records"][0]
    assert row["status"] == "verified_source_link"
    assert row["legacy_sha256"] == sha(legacy) and row["source_sha256"] == sha(source)
    assert row["source_path_b64"] == b64(source) and row["annotation_sha256"] == sha(label)
    assert row["v3_source_ids"] == ["a" * 20] and row["membership"] == "in_v3"
    assert row["legacy_label_reproduction_matches"] is True
    assert report["complete_original_lineage"] is False
    assert report["original_alias_uniqueness_proven"] is False
    assert report["training_authorized"] is False and report["deployment_authorized"] is False
    assert all(sha(p) == digest for p, digest in before.items())
    assert json.loads((output / "report.json").read_text()) == report


def test_round_instead_of_legacy_floor_resize_is_unresolved_not_index_link(probe):
    from scripts import audit_legacy_aihub_links as links
    save, data, source, _, legacy, _, _, _ = probe
    pixels = cv2.imdecode(np.frombuffer(source.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    resized = cv2.resize(pixels, (640, round(503 * 640 / 1001)), interpolation=cv2.INTER_AREA)
    assert resized.shape[0] == 322  # Historical int-floor height is 321.
    ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    legacy.write_bytes(encoded.tobytes())
    data["records"][0]["source_sha256"] = sha(legacy)
    report = links.audit_links(**save())
    assert report["records"][0]["status"] == "unresolved"
    assert report["complete_original_lineage"] is False


def test_link_outside_v3_does_not_mean_complete_original_lineage(probe):
    from scripts import audit_legacy_aihub_links as links
    save, _, source, _, _, _, v3, _ = probe
    text = v3.read_text(encoding="utf-8")
    v3.write_text(text.replace(b64(source), b64(source.with_name("not_this_source.png"))), encoding="utf-8")
    report = links.audit_links(**save())
    row = report["records"][0]
    assert row["status"] == "verified_source_link"
    assert row["membership"] == "outside_v3" and row["v3_source_ids"] == []
    assert report["complete_original_lineage"] is False


@pytest.mark.parametrize("fault", ["protected", "v3", "converter", "remainder"])
def test_invalid_explicit_artifact_pins_fail_without_report(probe, fault):
    from scripts import audit_legacy_aihub_links as links
    save, _, _, _, _, _, _, output = probe
    args = save()
    key = {"protected": "protected_report_sha256", "v3": "v3_manifest_sha256",
           "converter": "converter_sha256", "remainder": "remainder_sha256"}[fault]
    args[key] = "0" * 64
    with pytest.raises(ValueError):
        links.audit_links(**args)
    assert not (output / "report.json").exists()


def test_changed_legacy_bytes_not_matching_protected_pin_fail(probe):
    from scripts import audit_legacy_aihub_links as links
    save, _, _, _, legacy, _, _, output = probe
    args = save()
    legacy.write_bytes(legacy.read_bytes() + b"drift")
    with pytest.raises(ValueError):
        links.audit_links(**args)
    assert not (output / "report.json").exists()


def test_missing_candidate_ordinal_does_not_become_source_evidence(probe):
    from scripts import audit_legacy_aihub_links as links
    save, data, _, _, legacy, legacy_label, _, _ = probe
    renamed = legacy.with_name("train_0000005.jpg")
    renamed_label = legacy_label.with_name("train_0000005.txt")
    legacy.rename(renamed)
    legacy_label.rename(renamed_label)
    data["records"][0]["source_path_b64"] = b64(renamed)
    report = links.audit_links(**save())
    row = report["records"][0]
    assert row["status"] == "unresolved" and row["reason"] == "candidate_index_unavailable"
    assert "source_sha256" not in row and report["complete_original_lineage"] is False


def test_label_sidecar_mismatch_is_not_ground_truth_approval(probe):
    from scripts import audit_legacy_aihub_links as links
    save, _, _, _, _, legacy_label, _, _ = probe
    legacy_label.write_bytes(b"3 0.5 0.5 0.5 0.5\n")
    report = links.audit_links(**save())
    row = report["records"][0]
    assert row["status"] == "verified_source_link"
    assert row["legacy_label_reproduction_matches"] is False
    assert report["training_authorized"] is False


def test_partial_protected_probe_never_claims_complete_original_mapping(probe):
    from scripts import audit_legacy_aihub_links as links
    save, data, _, _, legacy, _, _, _ = probe
    extra = legacy.with_name("train_0000005.jpg")
    extra.write_bytes(b"not consumed by max_per_kind=1")
    data["records"].append({"source_sha256": sha(extra), "source_path_b64": b64(extra), "roles": ["qx3"]})
    data["expected_sources"] = data["verified_sources"] = 2
    report = links.audit_links(**save())
    assert len(report["records"]) == 1 and report["partial_selection"] is True
    assert report["complete_original_lineage"] is False
    assert report["original_alias_uniqueness_proven"] is False


def test_output_nested_in_legacy_dataset_does_not_even_create_directory(probe):
    from scripts import audit_legacy_aihub_links as links
    save, _, _, _, legacy, _, _, _ = probe
    args = save()
    legacy_root = legacy.parent.parent.parent
    before = {p.relative_to(legacy_root).as_posix() for p in legacy_root.rglob("*")}
    output = legacy_root / "new_audit_output"
    with pytest.raises(ValueError, match="overlap"):
        links.audit_links(**{**args, "output": output})
    assert not output.exists()
    assert {p.relative_to(legacy_root).as_posix() for p in legacy_root.rglob("*")} == before


def test_source_change_at_report_publication_sets_failure_marker(probe, monkeypatch):
    from scripts import audit_legacy_aihub_links as links
    save, _, source, _, _, _, _, output = probe
    publish = links._publish
    def tamper(path, value):
        result = publish(path, value)
        if path.name == "report.json":
            source.write_bytes(source.read_bytes() + b"terminal mutation")
        return result
    monkeypatch.setattr(links, "_publish", tamper)
    with pytest.raises(ValueError, match="changed"):
        links.audit_links(**save())
    failed = json.loads((output / "failed.json").read_text())
    assert failed["status"] == "failed" and failed["training_authorized"] is False


def test_candidate_symlink_to_outside_source_is_never_followed(probe):
    from scripts import audit_legacy_aihub_links as links
    save, _, source, _, _, _, _, output = probe
    outside = output.parent / "outside_original.png"
    outside.write_bytes(source.read_bytes())
    source.unlink()
    try:
        source.symlink_to(outside)
    except OSError:
        pytest.skip("OS symlink creation not available")
    with pytest.raises(ValueError, match="unsafe"):
        links.audit_links(**save())
    assert not (output / "report.json").exists()


def test_traversing_encoded_legacy_path_fails_before_output_creation(probe):
    from scripts import audit_legacy_aihub_links as links
    save, data, _, _, legacy, _, _, output = probe
    data["records"][0]["source_path_b64"] = b64(legacy.parent / ".." / legacy.parent.name / legacy.name)
    with pytest.raises(ValueError, match="encoded path"):
        links.audit_links(**save())
    assert not output.exists()


def test_read_stable_detects_mutation_during_open_file_read(tmp_path, monkeypatch):
    from scripts import audit_legacy_aihub_links as links
    path = tmp_path / "source.bin"
    path.write_bytes(b"original")
    fstat = links.os.fstat
    calls = 0
    def mutate(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            path.write_bytes(b"changed source bytes")
        return fstat(fd)
    monkeypatch.setattr(links.os, "fstat", mutate)
    with pytest.raises(ValueError, match="changed during read"):
        links.read_stable(path)
