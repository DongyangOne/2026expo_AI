"""Real metadata-loader tests with tiny synthetic reports, not image/GPU evidence."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts import build_legacy_protected_inventory as builder


def encoded(path):
    return base64.urlsafe_b64encode(os.fsencode(path)).decode()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def evidence(tmp_path):
    # Sources need to exist for path validation, but are intentionally not JPEGs:
    # this step must defer pixel hashing/decoding to the fingerprint auditor.
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    sources = tmp_path / "sources"
    sources.mkdir()
    originals = []
    for name, payload in (("a", b"same original"), ("b", b"same original"), ("c", b"other original")):
        path, annotation = sources / f"{name}.jpg", sources / f"{name}.json"
        path.write_bytes(payload)
        annotation.write_bytes(b"annotation metadata only")
        originals.append({"source_path_b64": encoded(path), "source_sha256": hashlib.sha256(payload).hexdigest(),
                          "annotation_path_b64": encoded(annotation),
                          "annotation_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest()})
    protected, rows, consumed = [], [], []
    for i, (kind, role) in enumerate((("train", "qx3"), ("train_r", "capture"), ("val", "known_audit")), 1):
        split = "val" if kind == "val" else "train"
        prefix = kind if kind == "train_r" else kind + "_"
        legacy = f"/app/yolo_dataset_9class_v2/{split}/images/{prefix}{i:07}.jpg"
        sha = hashlib.sha256(legacy.encode()).hexdigest()
        protected.append({"source_path_b64": encoded(legacy), "source_sha256": sha,
                          "source_phash64": "5" * 16, "roles": [role]})
        rows.append({"kind": kind, "index": i, "legacy_sha256": sha, "legacy_path_b64": encoded(legacy),
                     "status": "verified_source_link", "reason": "exact_legacy_jpeg_bytes",
                     "regenerated_legacy_sha256": sha, **originals[i - 1]})
        consumed.append({"path_b64": encoded(legacy), "sha256": sha})
        consumed.extend({"path_b64": originals[i - 1][key], "sha256": originals[i - 1][digest]}
                        for key, digest in (("source_path_b64", "source_sha256"),
                                            ("annotation_path_b64", "annotation_sha256")))
    protected_path, link_path = inputs / "protected.json", inputs / "links.json"
    protected_doc = {"schema": "protected_image_fingerprint_snapshot.v1", "status": "snapshot_complete",
                     "snapshot_only": True, "training_authorized": False, "deployment_authorized": False,
                     "missing_sources": 0, "expected_sources": 3, "verified_sources": 3, "records": protected}
    link_doc = {"schema": "legacy_aihub_source_link_probe.v1", "status": "probe_complete",
                "partial_selection": False, "max_per_kind": 0, "candidate_index_is_search_only": True,
                "protected_legacy_counts": {"train": 1, "train_r": 1, "val": 1},
                "status_counts": {"verified_source_link": 3}, "records": rows,
                **{key: False for key in ("training_authorized", "blind_test_authorized", "deployment_authorized",
                                         "complete_original_lineage", "original_alias_uniqueness_proven")}}

    def save():
        protected_sha = write_json(protected_path, protected_doc)
        link_doc["metadata_and_consumed_inputs"] = [*consumed,
            {"path_b64": encoded(protected_path), "sha256": protected_sha}]
        return {"link_report": link_path, "link_sha256": write_json(link_path, link_doc),
                "protected_report": protected_path, "protected_sha256": protected_sha, "output": tmp_path / "out"}

    return save, link_doc, protected_doc, originals


def test_inventory_uses_real_validated_links_and_defers_all_pixel_authority(evidence, monkeypatch):
    save, _, _, _ = evidence
    args = save()
    monkeypatch.setattr(builder.fingerprint, "audit_fingerprints", lambda **_: pytest.fail("no fingerprinting here"))
    summary = builder.build_inventory(**args)
    payload = (args["output"] / "inventory.json").read_bytes()
    data = json.loads(payload)
    assert set(data) == {"records", "metadata_bindings"}
    assert len(data["records"]) == summary["unique_original_sources"] == 2
    assert summary["verified_legacy_references"] == summary["expected_legacy_references"] == 3
    assert summary["unresolved"] == []
    assert summary["inventory_sha256"] == hashlib.sha256(payload).hexdigest()
    assert summary["status"] == "fingerprint_inventory_prepared"
    for field in ("original_pixels_fingerprinted", "training_authorized", "blind_test_authorized", "deployment_authorized"):
        assert summary[field] is False
    assert json.loads((args["output"] / "summary.json").read_bytes()) == summary
    assert {row["path"] for row in data["metadata_bindings"]} >= {str(args["link_report"]), str(args["protected_report"])}
    assert all(hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() == row["sha256"]
               for row in data["metadata_bindings"])
    assert {p.name for p in args["output"].iterdir()} == {"inventory.json", "summary.json"}


def test_same_original_sha_aliases_keep_one_path_and_union_actual_roles(evidence):
    save, _, _, originals = evidence
    args = save()
    builder.build_inventory(**args)
    rows = json.loads((args["output"] / "inventory.json").read_bytes())["records"]
    row = next(row for row in rows if row["sha256"] == originals[0]["source_sha256"])
    assert row["roles"] == ["capture", "qx3"]
    assert Path(row["path"]).name == "a.jpg"
    assert len({row["sha256"] for row in rows}) == len(rows) == 2


def test_unresolved_candidate_fields_are_not_added_to_inventory(evidence):
    save, links, _, originals = evidence
    links["records"][-1].update(status="unresolved", reason="legacy_jpeg_bytes_differ",
                                 source_path_b64=encoded("/not/an/existing/candidate.jpg"))
    links["status_counts"] = {"verified_source_link": 2, "unresolved": 1}
    args = save()
    summary = builder.build_inventory(**args)
    rows = json.loads((args["output"] / "inventory.json").read_bytes())["records"]
    assert {row["sha256"] for row in rows} == {originals[0]["source_sha256"]}
    assert summary["verified_legacy_references"] == 2 and summary["expected_legacy_references"] == 3
    assert len(summary["unresolved"]) == 1


@pytest.mark.parametrize("fault", ["missing", "partial", "wrong_reproduction", "link_sha", "protected_sha"])
def test_incomplete_coverage_or_changed_report_is_never_published(evidence, fault):
    save, links, _, _ = evidence
    if fault == "missing":
        links["records"].pop()
        links["status_counts"] = {"verified_source_link": 2}
    elif fault == "partial":
        links["max_per_kind"] = 3
    elif fault == "wrong_reproduction":
        links["records"][0]["regenerated_legacy_sha256"] = "0" * 64
    args = save()
    if fault in {"link_sha", "protected_sha"}:
        args["link_report" if fault == "link_sha" else "protected_report"].write_bytes(b"{}")
    with pytest.raises(ValueError):
        builder.build_inventory(**args)
    assert not args["output"].exists()


def test_existing_output_is_refused_without_reading_reports(evidence, monkeypatch):
    save, _, _, _ = evidence
    args = save()
    args["output"].mkdir()
    sentinel = args["output"] / "user.txt"
    sentinel.write_bytes(b"preserve")
    monkeypatch.setattr(builder.planner, "load_pinned", lambda *_: pytest.fail("preflight must refuse first"))
    with pytest.raises(ValueError, match="fresh output"):
        builder.build_inventory(**args)
    assert sentinel.read_bytes() == b"preserve"
    assert list(args["output"].iterdir()) == [sentinel]


@pytest.mark.parametrize("when", ["before_publication", "after_publication"])
def test_metadata_mutation_before_and_after_publication_fails(evidence, monkeypatch, when):
    save, _, _, _ = evidence
    args = save()
    if when == "before_publication":
        actual = builder.planner.validate_legacy_link_report
        def validate(**kwargs):
            value = actual(**kwargs)
            args["link_report"].write_bytes(b"{}")
            return value
        monkeypatch.setattr(builder.planner, "validate_legacy_link_report", validate)
    else:
        actual = builder.planner.recheck_legacy_bindings
        def recheck(bindings, reports):
            if (args["output"] / "inventory.json").exists():
                args["link_report"].write_bytes(b"{}")
            return actual(bindings, reports)
        monkeypatch.setattr(builder.planner, "recheck_legacy_bindings", recheck)
    with pytest.raises(ValueError):
        builder.build_inventory(**args)
    if when == "before_publication":
        assert not args["output"].exists()
    else:
        assert json.loads((args["output"] / "failed.json").read_bytes())["status"] == "failed"


@pytest.mark.parametrize("roles", [{"qx3": True}, ["qx3", "qx3"], [], ["training"], "qx3"])
def test_protected_roles_must_be_a_valid_unique_list(evidence, roles):
    save, _, protected, _ = evidence
    protected["records"][0]["roles"] = roles
    args = save()
    with pytest.raises(ValueError):
        builder.build_inventory(**args)
    assert not args["output"].exists()


def test_published_summary_mutation_is_not_reported_as_success(evidence, monkeypatch):
    save, _, _, _ = evidence
    args = save()
    actual = builder.planner.recheck_legacy_bindings
    def recheck(bindings, reports):
        actual(bindings, reports)
        summary = args["output"] / "summary.json"
        if summary.exists():
            summary.write_bytes(b'{"training_authorized":true}')
    monkeypatch.setattr(builder.planner, "recheck_legacy_bindings", recheck)
    with pytest.raises(ValueError):
        builder.build_inventory(**args)
    assert (args["output"] / "failed.json").exists()
