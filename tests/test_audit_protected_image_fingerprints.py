import base64
import copy
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts import audit_protected_image_fingerprints as audit


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    rows = []
    for i, roles in enumerate((["qx3", "known_audit"], ["capture"])):
        path = root / f"image{i}.png"
        Image.new("RGB", (48, 32), (30 + i * 30, 100, 170)).save(path)
        rows.append({"sha256": sha(path), "path": str(path), "roles": roles})
    metadata = root / "metadata.json"
    metadata.write_text('{"source_list":"test-only"}', encoding="utf-8")
    inventory = root / "inventory.json"
    data = {"records": rows, "metadata_bindings": [{"path": str(metadata), "sha256": sha(metadata)}]}
    inventory.write_text(json.dumps(data), encoding="utf-8")
    return {"inventory": inventory, "inventory_sha256": sha(inventory),
            "allowed_roots": [root], "output": tmp_path / "out"}, data


def update(case, data):
    args, _ = case
    args["inventory"].write_text(json.dumps(data), encoding="utf-8")
    args["inventory_sha256"] = sha(args["inventory"])


def test_all_unique_sources_have_actual_hash_shape_phash_and_no_authority(case):
    args, data = case
    original = {path: path.read_bytes() for path in args["allowed_roots"][0].iterdir()}
    report = audit.audit_fingerprints(**args)
    assert report["expected_sources"] == report["verified_sources"] == 2
    assert report["missing_sources"] == 0 and report["snapshot_only"] is True
    assert report["consumer_must_rehash_sources"] is True
    assert all(report[key] is False for key in audit.AUTHORITY)
    assert set(row["source_sha256"] for row in report["records"]) == {row["sha256"] for row in data["records"]}
    for row in report["records"]:
        path = Path(base64.urlsafe_b64decode(row["source_path_b64"]).decode())
        shape, expected, size, phash = audit.original.image_evidence(path)
        assert (row["image_height"], row["image_width"]) == shape
        assert (row["source_sha256"], row["source_bytes"], row["source_phash64"]) == (expected, size, phash)
    assert all(path.read_bytes() == content for path, content in original.items())
    assert json.loads((args["output"] / "report.json").read_bytes()) == report


@pytest.mark.parametrize("fault", ["duplicate_sha", "duplicate_path", "bad_sha", "empty_roles",
    "duplicate_roles", "unknown_role", "extra_gt", "empty_records", "duplicate_metadata"])
def test_invalid_union_or_role_schema_fails_without_output(case, fault):
    args, original = case
    data = copy.deepcopy(original)
    if fault == "duplicate_sha":
        data["records"][1]["sha256"] = data["records"][0]["sha256"]
    elif fault == "duplicate_path":
        data["records"][1]["path"] = data["records"][0]["path"]
    elif fault == "bad_sha":
        data["records"][0]["sha256"] = "not-a-sha"
    elif fault == "empty_roles":
        data["records"][0]["roles"] = []
    elif fault == "duplicate_roles":
        data["records"][0]["roles"] = ["qx3", "qx3"]
    elif fault == "unknown_role":
        data["records"][0]["roles"] = ["train"]
    elif fault == "extra_gt":
        data["records"][0]["class_id"] = 3
    elif fault == "empty_records":
        data["records"] = []
    else:
        data["metadata_bindings"].append(dict(data["metadata_bindings"][0]))
    update(case, data)
    with pytest.raises(ValueError):
        audit.audit_fingerprints(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("target", ["inventory", "metadata", "source", "missing_source", "outside", "traversal"])
def test_bad_bindings_paths_or_missing_images_hard_fail(case, tmp_path, target):
    args, data = case
    if target == "inventory":
        args["inventory_sha256"] = "f" * 64
    elif target == "metadata":
        Path(data["metadata_bindings"][0]["path"]).write_bytes(b"modified")
    elif target == "source":
        path = Path(data["records"][0]["path"])
        path.write_bytes(path.read_bytes() + b"modified")
    elif target == "missing_source":
        Path(data["records"][0]["path"]).unlink()
    else:
        source = Path(data["records"][0]["path"])
        if target == "outside":
            path = tmp_path / "outside.png"
            path.write_bytes(source.read_bytes())
        else:
            path = source.parent / "subdir" / ".." / source.name
        data["records"][0]["path"] = str(path)
        update(case, data)
    with pytest.raises((ValueError, OSError)):
        audit.audit_fingerprints(**args)
    assert not args["output"].exists()


def test_symlink_source_is_not_followed(case):
    args, data = case
    source = Path(data["records"][0]["path"])
    linked = source.with_name("linked.png")
    try:
        linked.symlink_to(source)
    except OSError:
        pytest.skip("symlinks unavailable")
    data["records"][0]["path"] = str(linked)
    update(case, data)
    with pytest.raises(ValueError, match="symlink"):
        audit.audit_fingerprints(**args)
    assert not args["output"].exists()


def test_source_change_after_image_decode_is_detected_at_end(case, monkeypatch):
    args, _ = case
    original = audit.original.image_evidence
    def changed(path, *values):
        result = original(path, *values)
        path.write_bytes(path.read_bytes() + b"modified")
        return result
    monkeypatch.setattr(audit.original, "image_evidence", changed)
    with pytest.raises(ValueError, match="changed"):
        audit.audit_fingerprints(**args)
    assert not args["output"].exists()


def test_metadata_change_after_image_decode_is_detected_at_end(case, monkeypatch):
    args, data = case
    metadata = Path(data["metadata_bindings"][0]["path"])
    original = audit.original.image_evidence
    def changed(path, *values):
        result = original(path, *values)
        metadata.write_bytes(b"changed")
        return result
    monkeypatch.setattr(audit.original, "image_evidence", changed)
    with pytest.raises(ValueError, match="changed"):
        audit.audit_fingerprints(**args)
    assert not args["output"].exists()


def test_read_budget_fails_instead_of_silently_skipping(case):
    args, _ = case
    with pytest.raises(ValueError, match="budget"):
        audit.audit_fingerprints(**args, max_read_bytes=10)
    assert not args["output"].exists()


def test_existing_output_remains_untouched(case):
    args, _ = case
    args["output"].mkdir()
    keep = args["output"] / "keep"
    keep.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        audit.audit_fingerprints(**args)
    assert list(args["output"].iterdir()) == [keep] and keep.read_bytes() == b"keep"


def test_source_change_at_publication_leaves_failure_without_success_report(case, monkeypatch):
    args, data = case
    source = Path(data["records"][0]["path"])
    original_open = Path.open
    class MutatingWriter:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            self.handle.__enter__()
            return self
        def write(self, value):
            result = self.handle.write(value)
            source.write_bytes(source.read_bytes() + b"changed")
            return result
        def __exit__(self, *values):
            return self.handle.__exit__(*values)
    def open_path(path, *values, **kwargs):
        handle = original_open(path, *values, **kwargs)
        if path == args["output"] / "report.json" and values == ("xb",):
            return MutatingWriter(handle)
        return handle
    monkeypatch.setattr(Path, "open", open_path)
    with pytest.raises(ValueError, match="changed"):
        audit.audit_fingerprints(**args)
    assert not (args["output"] / "report.json").exists()
    failure = json.loads((args["output"] / "failed.json").read_bytes())
    assert failure["status"] == "failed" and failure["training_authorized"] is False


def test_inventory_duplicate_keys_are_not_silently_overwritten(case):
    args, data = case
    content = json.dumps(data)[:-1] + ',"records":' + json.dumps(data["records"]) + '}'
    args["inventory"].write_text(content, encoding="utf-8")
    args["inventory_sha256"] = sha(args["inventory"])
    with pytest.raises(ValueError, match="duplicate JSON key"):
        audit.audit_fingerprints(**args)
    assert not args["output"].exists()
