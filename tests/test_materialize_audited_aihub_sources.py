"""CPU original-GT materialization checks; no model accuracy or train authority."""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import audit_aihub_original_annotations as audit
from scripts import materialize_audited_aihub_sources as mat


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / "dataset"
    sources, labels, rows = [], [], []
    for index, (official, sp, lp, split) in enumerate([
            ("Training", "TS", "TL", "training"), ("Validation", "VS", "VL", "validation")]):
        folder = "2.직접촬영_01.금속캔_001.철캔"
        source = root / official / "01.원천데이터" / f"{sp}_{folder}_1" / "a.png"
        label = root / official / "02.라벨링데이터" / f"{lp}_{folder}" / "a.json"
        source.parent.mkdir(parents=True)
        label.parent.mkdir(parents=True)
        pixels = np.random.default_rng(index).integers(30, 220, (480, 800, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", pixels)
        assert ok
        source.write_bytes(encoded.tobytes())
        payload = {"IMAGE_INFO": {"FILE_NAME": source.name, "IMAGE_WIDTH": 800, "IMAGE_HEIGHT": 480},
                   "ANNOTATION_INFO": [{"CLASS": "금속캔", "DETAILS": "철캔", "DAMAGE": "찌그러짐",
                                        "POINTS": [[200, 120, 400, 240]], "DIRTINESS": "이물질(외부)"}]}
        label.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        rows.append({"source_id": hashlib.sha1(f"{official}/{lp}_{folder}/a.json".encode()).hexdigest()[:20],
                     "status": "verified_pair", "split": split, "class_id": 0, "class_name": "can",
                     "bbox_xywh": [200.0, 120.0, 400.0, 240.0], "annotation_dent": 1,
                     "source_path_b64": audit.encode_path(source), "label_path_b64": audit.encode_path(label),
                     "source_sha256": digest(source), "label_sha256": digest(label),
                     "image_height": 480, "image_width": 800, "conditions": dict(mat.UNKNOWN)})
        sources.append(source)
        labels.append(label)
    metadata = tmp_path / "protected" / "report.json"
    metadata.parent.mkdir()
    metadata.write_text('{"status":"snapshot_complete"}', encoding="utf-8")
    cohort = tmp_path / "planning" / "cohort.json"
    cohort.parent.mkdir()
    data = {"schema": "aihub_original_cohort_v1", "status": "cohort_planned",
            "training_authorized": False, "deployment_authorized": False, "records": rows,
            "metadata_bindings": [{"path": str(metadata), "sha256": digest(metadata)}]}
    output = tmp_path / "materialized"

    def save():
        cohort.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return {"cohort": cohort, "cohort_sha256": digest(cohort), "dataset_root": root,
                "output": output, "min_free_gib": 0, "max_output_gib": 1}

    return data, save, output, sources, labels, metadata


def test_real_original_pair_to_jpeg_yolo_lineage_and_non_authoritative_snapshot(fixture):
    data, save, output, sources, labels, metadata = fixture
    originals = {p: digest(p) for p in sources + labels + [metadata]}
    report = mat.materialize_sources(**save())
    assert report["verified_sources"] == report["materialized_sources"] == 2
    assert report["quality_excluded_sources"] == 0
    assert report["quality_policy"] == mat.POLICY
    assert report["snapshot_only"] and all(report[k] is False for k in mat.AUTHORITY)
    lineage = [json.loads(s) for s in (output / "lineage.jsonl").read_text().splitlines()]
    assert {r["split"] for r in lineage} == {"training", "validation"}
    for row in lineage:
        image = cv2.imdecode(np.frombuffer((output / row["image_ref"]).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        assert image.shape[:2] == (384, 640)
        assert (output / row["label_ref"]).read_text() == "0 0.50000000 0.50000000 0.50000000 0.50000000\n"
        assert row["conditions"] == mat.UNKNOWN and row["annotation_dent"] == 1
        assert digest(output / row["image_ref"]) == row["image_sha256"]
    yaml = (output / "dataset.yaml").read_text()
    assert output.as_posix() in yaml and "train: images/train" in yaml and "val: images/val" in yaml
    assert all(f"  {i}: {name}" in yaml for i, name in enumerate(audit.CLASS_NAMES))
    ready = json.loads((output / "snapshot_ready.json").read_text())
    assert ready["report_sha256"] == digest(output / "report.json")
    assert all(ready[k] is False for k in mat.AUTHORITY)
    assert all(digest(p) == sha for p, sha in originals.items())
    assert not (output / "failed.json").exists()


@pytest.mark.parametrize("fault", ["cohort_hash", "metadata_hash", "source_hash", "label_hash", "class_bool", "state_bool", "duplicate"])
def test_pinned_input_contract_mismatches_fail_without_ready(fixture, fault):
    data, save, output, sources, labels, metadata = fixture
    if fault == "metadata_hash":
        metadata.write_text("changed")
    elif fault == "source_hash":
        sources[0].write_bytes(sources[0].read_bytes() + b"changed")
    elif fault == "label_hash":
        labels[0].write_text("{}")
    elif fault == "class_bool":
        data["records"][0]["class_id"] = False
    elif fault == "state_bool":
        data["records"][0]["conditions"]["dent"] = False
    elif fault == "duplicate":
        data["records"].append(copy.deepcopy(data["records"][0]))
    args = save()
    if fault == "cohort_hash":
        args["cohort_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        mat.materialize_sources(**args)
    assert not (output / "snapshot_ready.json").exists()
    if output.exists():
        assert json.loads((output / "failed.json").read_text())["training_authorized"] is False


@pytest.mark.parametrize("fault", ["unknown_class", "multiple_objects", "bbox_changed"])
def test_original_json_is_revalidated_not_trusted_from_cohort(fixture, fault):
    data, save, output, sources, labels, metadata = fixture
    payload = json.loads(labels[0].read_text(encoding="utf-8"))
    if fault == "unknown_class":
        payload["ANNOTATION_INFO"][0].update(CLASS="알 수 없음", DETAILS="알 수 없음")
    elif fault == "multiple_objects":
        payload["ANNOTATION_INFO"].append(copy.deepcopy(payload["ANNOTATION_INFO"][0]))
    else:
        payload["ANNOTATION_INFO"][0]["POINTS"][0][0] += 1
    labels[0].write_text(json.dumps(payload), encoding="utf-8")
    data["records"][0]["label_sha256"] = digest(labels[0])
    with pytest.raises(ValueError):
        mat.materialize_sources(**save())
    assert (output / "failed.json").exists() and not (output / "snapshot_ready.json").exists()


def test_only_measured_quality_exclusions_are_recorded(fixture):
    data, save, output, sources, labels, metadata = fixture
    ok, encoded = cv2.imencode(".png", np.full((480, 800, 3), 128, dtype=np.uint8))
    assert ok
    sources[0].write_bytes(encoded.tobytes())
    data["records"][0]["source_sha256"] = digest(sources[0])
    report = mat.materialize_sources(**save())
    assert report["verified_sources"] == 2 and report["materialized_sources"] == 1
    excluded = json.loads((output / "excluded.jsonl").read_text())
    assert excluded["reason"] == "too_blurry"
    assert excluded["quality"]["resized_laplacian_variance"] == 0


def test_max_sources_is_explicit_partial_snapshot_not_full_coverage(fixture):
    data, save, output, *_ = fixture
    report = mat.materialize_sources(**save(), max_sources=1)
    assert report["verified_sources"] == 1 and report["unprocessed_sources"] == 1
    assert report["cohort_records"] == 2 and report["requested_max_sources"] == 1


def test_output_must_not_modify_existing_or_evidence_tree(fixture):
    data, save, output, *_ = fixture
    args = save()
    evidence = args["cohort"].parent
    before = list(evidence.iterdir())
    with pytest.raises(ValueError, match="overlaps"):
        mat.materialize_sources(**{**args, "output": evidence / "new"})
    assert list(evidence.iterdir()) == before
    output.mkdir()
    (output / "keep").write_text("existing")
    with pytest.raises(ValueError):
        mat.materialize_sources(**args)
    assert (output / "keep").read_text() == "existing"


def test_free_disk_and_output_byte_limits_fail_closed(fixture):
    data, save, output, *_ = fixture
    args = save()
    with pytest.raises(ValueError, match="disk"):
        mat.materialize_sources(**{**args, "min_free_gib": 10**12})
    assert not output.exists()
    with pytest.raises(ValueError, match="budget"):
        mat.materialize_sources(**{**args, "max_output_gib": 1 / 1024**3})
    assert (output / "failed.json").exists() and not (output / "snapshot_ready.json").exists()


def test_source_change_after_decode_is_fatal_not_quality_exclusion(fixture, monkeypatch):
    data, save, output, sources, *_ = fixture
    original = mat._prepare
    def change(*args):
        result = original(*args)
        sources[0].write_bytes(sources[0].read_bytes() + b"drift")
        return result
    monkeypatch.setattr(mat, "_prepare", change)
    with pytest.raises(ValueError, match="changed"):
        mat.materialize_sources(**save())
    assert (output / "failed.json").exists() and not (output / "snapshot_ready.json").exists()


def test_source_change_after_ready_publication_revokes_ready(fixture, monkeypatch):
    data, save, output, sources, *_ = fixture
    original = mat._digest
    changed = False
    def change(path):
        nonlocal changed
        if (output / "snapshot_ready.json").exists() and not changed:
            sources[0].write_bytes(sources[0].read_bytes() + b"late drift")
            changed = True
        return original(path)
    monkeypatch.setattr(mat, "_digest", change)
    with pytest.raises(ValueError, match="changed"):
        mat.materialize_sources(**save())
    assert changed and (output / "failed.json").exists()
    assert not (output / "snapshot_ready.json").exists()


@pytest.mark.parametrize("fault", ["outside", "traversal", "symlink"])
def test_source_paths_cannot_escape_or_alias_dataset(fixture, fault):
    data, save, output, sources, *_ = fixture
    source = sources[0]
    if fault == "outside":
        alias = output.parent / "outside.png"
        alias.write_bytes(source.read_bytes())
    elif fault == "traversal":
        alias = source.parent / ".." / source.parent.name / source.name
    else:
        alias = source.parent / "alias.png"
        try:
            alias.symlink_to(source)
        except OSError:
            pytest.skip("OS symlink creation not available")
    data["records"][0]["source_path_b64"] = audit.encode_path(alias)
    with pytest.raises(ValueError):
        mat.materialize_sources(**save())
    assert not output.exists()


def test_standalone_cli_happy_path_has_no_training_authority(fixture):
    data, save, output, *_ = fixture
    args = save()
    proc = subprocess.run([sys.executable, str(Path(mat.__file__)), "--cohort", str(args["cohort"]),
                           "--cohort-sha256", args["cohort_sha256"], "--dataset-root", str(args["dataset_root"]),
                           "--output", str(output), "--min-free-gib", "0", "--max-sources", "1"],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(proc.stdout.strip())
    assert result["materialized_sources"] == 1 and result["training_authorized"] is False


@pytest.mark.parametrize("module", [mat, audit], ids=["materializer", "annotation_validator"])
@pytest.mark.parametrize("matching", [False, True], ids=["stale_pin", "matching_pin"])
def test_explicit_code_metadata_pins_are_never_overwritten(fixture, module, matching):
    data, save, output, *_ = fixture
    path = Path(module.__file__).resolve()
    pinned = digest(path) if matching else "0" * 64
    data["metadata_bindings"].append({"path": str(path), "sha256": pinned})
    if not matching:
        with pytest.raises(ValueError, match="pinned code metadata SHA256 mismatch"):
            mat.materialize_sources(**save())
        assert not output.exists()
    else:
        report = mat.materialize_sources(**save())
        assert report["code_sha256"][path.name] == pinned
        assert report["metadata_bindings"][-1]["sha256"] == pinned
        assert (output / "snapshot_ready.json").exists()
