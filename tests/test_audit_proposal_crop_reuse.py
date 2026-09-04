"""Small real-file fixtures for metadata/crop-byte reuse; no model or original IO."""
import base64
import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts import audit_proposal_crop_reuse as audit


def digest(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def sample(tmp_path):
    metadata = tmp_path / "metadata"
    crops = tmp_path / "crops"
    metadata.mkdir()
    crops.mkdir()
    selected, rows = [], []
    for index, split in enumerate(("training", "validation", "validation")):
        source = f"/not-read/originals/{index}.jpg"
        source_sha = digest(source.encode())
        selected.append({"path": source, "source_sha256": source_sha, "split": split,
                         "explicit_empty_label": index == 2, "selection_cohort": "current_yolo_ground_truth"})
        if index == 2:
            continue
        relative = f"{split}/can/{index}.jpg"
        path = crops / relative
        path.parent.mkdir(parents=True)
        path.write_bytes(f"not-decoded-crop-{index}".encode())
        rows.append({"filepath": relative, "source_sha256": source_sha,
                     "source_path_b64": base64.urlsafe_b64encode(source.encode()).decode(), "split": split,
                     "image_sha256": digest(path.read_bytes()), "crop_bytes": str(path.stat().st_size),
                     "detector_model_sha256": "a" * 64, "inference_spec_sha256": "b" * 64,
                     "source_width": "640", "source_height": "480", "crop_x1": "10", "crop_y1": "20",
                     "crop_x2": "100", "crop_y2": "120"})
    manifest, selection = metadata / "manifest.csv", metadata / "selection.json"
    def save():
        selection.write_text(json.dumps({"selected_sources": selected}), encoding="utf-8")
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(audit.FIELDS))
            writer.writeheader()
            writer.writerows(rows)
        return {"manifest": manifest, "manifest_sha256": digest(manifest.read_bytes()),
                "selection": selection, "selection_sha256": digest(selection.read_bytes()),
                "crop_root": crops, "model_sha256": "a" * 64, "spec_sha256": "b" * 64,
                "output": tmp_path / "out"}
    return save, rows, selected, crops


def test_real_crop_bytes_join_and_missing_selection_are_reported_without_authority(sample, monkeypatch):
    save, rows, _, _ = sample
    args = save()
    seen = []
    actual = audit.read_file
    def read(path, *rest, **kwargs):
        seen.append(str(path))
        return actual(path, *rest, **kwargs)
    monkeypatch.setattr(audit, "read_file", read)
    report = audit.audit_reuse(**args)
    assert report == json.loads((args["output"] / "report.json").read_bytes())
    assert report["selection_sources"] == 3 and report["verified_crop_rows"] == 2 and report["missing_sources"] == 1
    assert report["missing_selection_sources"][0]["selection_explicit_empty_label"] is True
    assert report["missing_selection_sources"][0]["reason"] == "no_manifest_crop"
    assert report["duplicate_crop_sha_groups"] == []
    assert all(report[key] is False for key in audit.AUTHORITY)
    assert report["model_and_spec_files_rehashed"] is report["crop_pixels_decoded"] is False
    assert not any("not-read" in path for path in seen)
    for row in rows:
        assert seen.count(str((args["crop_root"] / row["filepath"]).resolve())) == 3


def test_identical_crop_bytes_across_distinct_sources_are_preserved_as_groups(sample):
    save, rows, _, crops = sample
    first, second = (crops / row["filepath"] for row in rows)
    second.write_bytes(first.read_bytes())
    rows[1]["image_sha256"] = rows[0]["image_sha256"]
    rows[1]["crop_bytes"] = rows[0]["crop_bytes"]
    report = audit.audit_reuse(**save())
    assert len(report["records"]) == 2
    assert report["duplicate_crop_sha_groups"] == [{"crop_sha256": rows[0]["image_sha256"], "record_indices": [0, 1]}]


@pytest.mark.parametrize("fault", ["crop_sha", "crop_size", "model", "spec", "source_path", "split", "geometry"])
def test_tampered_metadata_and_bytes_are_rejected(sample, fault):
    save, rows, _, crops = sample
    if fault == "crop_sha":
        path = crops / rows[0]["filepath"]
        path.write_bytes(b"x" * path.stat().st_size)
    elif fault == "crop_size": rows[0]["crop_bytes"] = "1"
    elif fault == "model": rows[0]["detector_model_sha256"] = "c" * 64
    elif fault == "spec": rows[0]["inference_spec_sha256"] = "c" * 64
    elif fault == "source_path": rows[0]["source_path_b64"] = base64.b64encode(b"/different/source.jpg").decode()
    elif fault == "split": rows[0]["split"] = "validation"
    else: rows[0]["crop_x2"] = "1000"
    args = save()
    with pytest.raises(ValueError): audit.audit_reuse(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("path", ["../outside.jpg", "/absolute.jpg", "C:/outside.jpg", "training/../outside.jpg",
                                  "training//can/0.jpg", "training/./can/0.jpg", "training\\can\\0.jpg"])
def test_crop_traversal_and_noncanonical_paths_are_rejected(sample, path):
    save, rows, _, _ = sample
    rows[0]["filepath"] = path
    args = save()
    with pytest.raises((ValueError, FileNotFoundError)): audit.audit_reuse(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("which", ["selected_sha", "selected_path", "manifest_source", "crop_filepath"])
def test_duplicate_source_or_crop_paths_are_not_silently_deduplicated(sample, which):
    save, rows, selected, _ = sample
    if which == "selected_sha": selected.append(copy.deepcopy(selected[0]))
    elif which == "selected_path": selected[1]["path"] = selected[0]["path"]
    elif which == "manifest_source": rows.append(copy.deepcopy(rows[0]))
    else:
        selected[1]["split"] = rows[1]["split"] = rows[0]["split"]
        rows[1]["filepath"] = rows[0]["filepath"]
    args = save()
    with pytest.raises(ValueError): audit.audit_reuse(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("target", ["manifest", "selection"])
def test_initial_metadata_sha_mismatch_is_rejected(sample, target):
    save, _, _, _ = sample
    args = save()
    args[target].write_bytes(b"changed")
    with pytest.raises(ValueError): audit.audit_reuse(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("which", ["metadata", "crop", "existing"])
def test_output_reuse_or_overlap_is_rejected(sample, which):
    save, _, _, _ = sample
    args = save()
    if which == "metadata": args["output"] = args["manifest"].parent / "new"
    elif which == "crop": args["output"] = args["crop_root"] / "new"
    else:
        args["output"].mkdir()
        (args["output"] / "keep").write_bytes(b"unchanged")
    with pytest.raises(ValueError): audit.audit_reuse(**args)
    assert not (args["output"] / "report.json").exists()
    if which == "existing": assert (args["output"] / "keep").read_bytes() == b"unchanged"


def test_crop_symlink_is_rejected_without_following_it(sample, tmp_path):
    save, rows, _, crops = sample
    path = crops / rows[0]["filepath"]
    target = tmp_path / "target.jpg"
    target.write_bytes(path.read_bytes())
    path.unlink()
    try: path.symlink_to(target)
    except OSError: pytest.skip("symlink creation unavailable")
    args = save()
    with pytest.raises(ValueError, match="symlink"): audit.audit_reuse(**args)


@pytest.mark.parametrize("target,late", [("manifest", False), ("selection", True), ("crop", True)])
def test_metadata_and_crop_mutation_cannot_leave_a_success_report(sample, monkeypatch, target, late):
    save, rows, _, crops = sample
    args = save()
    path = crops / rows[0]["filepath"] if target == "crop" else args[target]
    actual, mutated = audit.read_file, False
    def read(current, *rest, **kwargs):
        nonlocal mutated
        result = actual(current, *rest, **kwargs)
        if not mutated and Path(current) == path and ((args["output"] / "report.json").exists() if late else kwargs.get("keep")):
            path.write_bytes(b"changed")
            mutated = True
            # The current read succeeded; ensure the next remaining pinned read
            # observes this late mutation rather than depending on file mtimes.
            if late:
                return actual(current, *rest, **kwargs)
        return result
    monkeypatch.setattr(audit, "read_file", read)
    with pytest.raises(ValueError): audit.audit_reuse(**args)
    assert mutated
    assert not (args["output"] / "report.json").exists()
    if late: assert (args["output"] / "failed.json").exists()


def test_metadata_and_image_read_limits_are_enforced(sample, monkeypatch):
    save, _, _, _ = sample
    args = save()
    monkeypatch.setattr(audit, "METADATA_LIMIT", 4)
    with pytest.raises(ValueError, match="size/type"): audit.audit_reuse(**args)
    assert not args["output"].exists()


def test_cli_rejects_non_lowercase_sha_and_reports_no_authority(sample, capsys):
    save, _, _, _ = sample
    args = save()
    cli = [part for key, value in args.items() for part in ("--" + key.replace("_", "-"), str(value))]
    assert audit.main(cli) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["verified_crop_rows"] == 2
    assert all(printed[key] is False for key in audit.AUTHORITY)
    with pytest.raises(ValueError): audit.sha256_value("A" * 64)
