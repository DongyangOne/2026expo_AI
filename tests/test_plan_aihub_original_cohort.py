"""Metadata cohort planning only: protect holdouts without publishing training data."""

import base64
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import pytest

from scripts import plan_aihub_original_cohort as planner
from test_materialize_audited_aihub_sources import fixture as materializer_inputs


def original(number, *, split="training", phash=None, source_sha=None):
    source_id = f"{number:020x}"
    return {
        "status": "verified_pair", "source_id": source_id, "split": split,
        "declared_class": "can", "class_id": 0, "class_name": "can",
        "source_sha256": source_sha or hashlib.sha256(f"source:{number}".encode()).hexdigest(),
        "source_phash64": phash or hashlib.sha256(f"phash:{number}".encode()).hexdigest()[:16],
        "label_sha256": hashlib.sha256(f"label:{number}".encode()).hexdigest(),
        "source_path_b64": base64.urlsafe_b64encode(f"/app/aihub/{source_id}.jpg".encode()).decode(),
        "label_path_b64": base64.urlsafe_b64encode(f"/app/aihub/{source_id}.json".encode()).decode(),
        "source_bytes": 1024, "image_width": 64, "image_height": 48,
        "bbox_xywh": [8.0, 6.0, 24.0, 18.0], "annotation_dent": 0,
        "conditions": {"dent": -1, "label": -1, "foreign_material": -1},
    }


def accepted_ids(plan):
    return {row["source_id"] for row in plan["records"]}


def exclusions(plan):
    rows = plan["exclusions"]
    assert len({row["source_id"] for row in rows}) == len(rows)
    assert all(isinstance(row["reasons"], list) and row["reasons"] for row in rows)
    return {row["source_id"]: row["reasons"] for row in rows}


def unrelated_protected():
    return [{"source_sha256": hashlib.sha256(b"unrelated-protected").hexdigest(),
             "source_phash64": "5555555555555555"}]


def test_hamming_index_matches_bruteforce_including_distance_boundary():
    rng = random.Random(20260904)
    values = [rng.getrandbits(64) for _ in range(80)]
    index = planner.HammingIndex(distance=4)
    for key, value in enumerate(values):
        index.add(value, key)
    queries = [rng.getrandbits(64) for _ in range(20)]
    queries += [value ^ ((1 << distance) - 1) for value in values[:8] for distance in range(7)]
    for query in queries:
        expected = {key for key, value in enumerate(values) if (value ^ query).bit_count() <= 4}
        assert index.matches(query) == expected
        assert index.has_match(query) is bool(expected)


def test_hamming_index_keeps_all_keys_for_identical_values_and_high_bit():
    index = planner.HammingIndex(distance=4)
    index.add(1 << 63, "a")
    index.add(1 << 63, "b")
    assert index.matches((1 << 63) | 15) == {"a", "b"}
    assert index.matches((1 << 63) | 31) == set()


@pytest.mark.parametrize("invalid", [-1, 1 << 64, True, 1.5, "123"])
def test_hamming_index_rejects_non_uint64_values(invalid):
    index = planner.HammingIndex(distance=4)
    with pytest.raises(planner.PlanError):
        index.add(invalid, "invalid")
    with pytest.raises(planner.PlanError):
        index.matches(invalid)


@pytest.mark.parametrize("distance", [-1, 8, True, 1.5, "4"])
def test_hamming_index_rejects_invalid_distance(distance):
    with pytest.raises(planner.PlanError):
        planner.HammingIndex(distance=distance)


@pytest.mark.parametrize("distance", [0, 7])
def test_hamming_index_supported_distance_extremes(distance):
    index = planner.HammingIndex(distance=distance)
    index.add((1 << 64) - 1, "max-uint64")
    assert index.matches(((1 << 64) - 1) ^ ((1 << distance) - 1)) == {"max-uint64"}
    assert index.matches(((1 << 64) - 1) ^ ((1 << (distance + 1)) - 1)) == set()


def test_unmatched_sources_keep_unknown_states_without_changing_inputs():
    rows = [original(1), original(2, split="validation")]
    snapshot = copy.deepcopy(rows)
    result = planner.plan_records(rows, unrelated_protected(), set())
    assert accepted_ids(result) == {row["source_id"] for row in rows}
    assert result["exclusions"] == []
    assert all(row["conditions"] == {"dent": -1, "label": -1, "foreign_material": -1}
               for row in result["records"])
    assert rows == snapshot


@pytest.mark.parametrize("match_kind", ["sha", "source_id", "phash"])
def test_protected_evidence_excludes_matching_original(match_kind):
    row = original(3, phash="0000000000000000")
    protected = [{"source_sha256": "f" * 64, "source_phash64": "ffffffffffffffff"}]
    protected_ids = set()
    if match_kind == "sha":
        protected[0]["source_sha256"] = row["source_sha256"]
    elif match_kind == "source_id":
        protected_ids.add(row["source_id"])
    else:
        protected[0]["source_phash64"] = "000000000000000f"  # exactly four bits
    result = planner.plan_records([row], protected, protected_ids)
    assert result["records"] == []
    expected_reason = {"sha": "protected_exact_sha256", "source_id": "protected_source_id",
                       "phash": "protected_near_phash"}[match_kind]
    assert exclusions(result) == {row["source_id"]: [expected_reason]}


def test_protected_phash_five_bit_difference_is_not_a_near_match():
    row = original(4, phash="0000000000000000")
    protected = [{"source_sha256": "f" * 64, "source_phash64": "000000000000001f"}]
    result = planner.plan_records([row], protected, set())
    assert accepted_ids(result) == {row["source_id"]}
    assert result["exclusions"] == []


@pytest.mark.parametrize("match_kind", ["sha", "phash"])
def test_cross_split_duplicate_removes_both_sides(match_kind):
    train = original(5, phash="0000000000000000")
    validation = original(6, split="validation", phash="ffffffffffffffff")
    if match_kind == "sha":
        validation["source_sha256"] = train["source_sha256"]
        validation["source_phash64"] = train["source_phash64"]
    else:
        validation["source_phash64"] = "000000000000000f"
    result = planner.plan_records([train, validation], unrelated_protected(), set())
    assert result["records"] == []
    reasons = exclusions(result)
    assert set(reasons) == {train["source_id"], validation["source_id"]}
    assert all("cross_split_duplicate" in value for value in reasons.values())


def test_same_split_exact_duplicate_keeps_lexicographically_first_source_id():
    first = original(7)
    second = original(8, source_sha=first["source_sha256"], phash=first["source_phash64"])
    result = planner.plan_records([second, first], unrelated_protected(), set())
    assert accepted_ids(result) == {first["source_id"]}
    assert exclusions(result) == {second["source_id"]: ["same_split_duplicate"]}


def test_same_split_near_but_not_exact_sources_are_not_automatically_removed():
    rows = [original(9, phash="0000000000000000"), original(10, phash="0000000000000001")]
    result = planner.plan_records(rows, unrelated_protected(), set())
    assert accepted_ids(result) == {row["source_id"] for row in rows}
    assert result["exclusions"] == []


@pytest.mark.parametrize("conflict", ["class", "bbox", "phash"])
def test_same_sha_disagreement_is_not_resolved_by_source_id_order(conflict):
    first = original(11)
    second = original(12, source_sha=first["source_sha256"], phash=first["source_phash64"])
    if conflict == "class":
        second.update(class_id=3, class_name="plastic", declared_class="plastic")
    elif conflict == "phash":
        second["source_phash64"] = "f" * 16
    else:
        second["bbox_xywh"] = [9.0, 6.0, 24.0, 18.0]
    with pytest.raises(planner.PlanError):
        planner.plan_records([first, second], unrelated_protected(), set())


def test_duplicate_source_id_is_rejected_even_when_only_content_sha_is_tampered():
    first = original(13)
    second = copy.deepcopy(first)
    second["source_sha256"] = "f" * 64
    with pytest.raises(planner.PlanError):
        planner.plan_records([first, second], unrelated_protected(), set())


def test_conflicting_same_sha_fails_before_cross_split_exclusion_can_hide_it():
    first = original(15)
    second = original(16, split="validation", source_sha=first["source_sha256"],
                      phash=first["source_phash64"])
    second.update(class_id=3, class_name="plastic", declared_class="plastic")
    with pytest.raises(planner.PlanError):
        planner.plan_records([first, second], unrelated_protected(), set())


@pytest.mark.parametrize("kind", ["class", "bbox"])
@pytest.mark.parametrize("split", ["training", "validation"])
def test_opt_in_quarantines_every_annotation_conflict_member_with_exact_evidence(kind, split):
    first = original(21, phash="0000000000000000")
    second = original(22, split=split, source_sha=first["source_sha256"], phash=first["source_phash64"])
    if kind == "class":
        second.update(class_id=3, class_name="plastic")
    else:
        second["bbox_xywh"][0] += 1
    rows = [second, first, original(23, phash="ffffffffffffffff")]
    before = copy.deepcopy(rows)
    result = planner.plan_records(rows, [], set(), quarantine_annotation_conflicts=True)
    assert accepted_ids(result) == {rows[2]["source_id"]}
    for sid in (first["source_id"], second["source_id"]):
        assert "annotation_conflict_same_sha256" in exclusions(result)[sid]
        assert ("cross_split_duplicate" in exclusions(result)[sid]) is (split == "validation")
    evidence = result["annotation_conflict_quarantine"]
    assert evidence["enabled"] is True and evidence["group_count"] == 1 and evidence["source_count"] == 2
    group = evidence["groups"][0]
    assert group["source_sha256"] == first["source_sha256"]
    assert group["image_evidence"] == {k: first[k] for k in
                                       ("image_width", "image_height", "source_phash64", "source_bytes")}
    assert [member["source_id"] for member in group["members"]] == [first["source_id"], second["source_id"]]
    for member, expected in zip(group["members"], (first, second)):
        assert all(member[k] == expected[k] for k in member)
    assert result == planner.plan_records(list(reversed(rows)), [], set(), quarantine_annotation_conflicts=True)
    assert result["counts"]["originals"] == result["counts"]["accepted"] + result["counts"]["excluded"] == 3
    assert rows == before


@pytest.mark.parametrize("field,value", [("image_width", 65), ("image_height", 49),
                                         ("source_phash64", "f" * 16), ("source_bytes", 1025)])
def test_opt_in_never_quarantines_conflicting_image_evidence(field, value):
    first = original(24)
    second = original(25, source_sha=first["source_sha256"], phash=first["source_phash64"])
    second["bbox_xywh"][0] += 1
    second[field] = value
    with pytest.raises(planner.PlanError, match="conflicting image evidence"):
        planner.plan_records([first, second], [], set(), quarantine_annotation_conflicts=True)


@pytest.mark.parametrize("value", [None, False, 0, -1, 1.5, "1024"])
def test_source_byte_evidence_requires_exact_positive_integer(value):
    row = original(26)
    row["source_bytes"] = value
    with pytest.raises(planner.PlanError, match="source byte count"):
        planner.plan_records([row], [], set(), quarantine_annotation_conflicts=True)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_annotation_quarantine_requires_explicit_boolean(value):
    with pytest.raises(planner.PlanError, match="opt-in must be boolean"):
        planner.plan_records([original(27)], [], set(), quarantine_annotation_conflicts=value)


def test_quarantined_groups_remain_in_split_and_protected_near_indexes():
    first = original(28, phash="0000000000000000")
    second = original(29, source_sha=first["source_sha256"], phash=first["source_phash64"])
    second["bbox_xywh"][0] += 1
    near = original(30, split="validation", phash="000000000000000f")  # exactly distance four
    beyond = original(31, split="validation", phash="000000000000001f")
    protected = [{"source_sha256": first["source_sha256"], "source_phash64": first["source_phash64"]}]
    result = planner.plan_records([first, second, near, beyond], protected, {first["source_id"]},
                                  quarantine_annotation_conflicts=True)
    assert accepted_ids(result) == {beyond["source_id"]}
    reasons = exclusions(result)
    assert {"annotation_conflict_same_sha256", "cross_split_duplicate", "protected_exact_sha256",
            "protected_source_id", "protected_near_phash"} <= set(reasons[first["source_id"]])
    assert {"cross_split_duplicate", "protected_near_phash"} <= set(reasons[near["source_id"]])


def test_opt_in_preserves_nonconflicting_duplicate_and_default_report_behavior():
    first = original(32)
    second = original(33, source_sha=first["source_sha256"], phash=first["source_phash64"])
    default = planner.plan_records([second, first], [], set())
    enabled = planner.plan_records([second, first], [], set(), quarantine_annotation_conflicts=True)
    assert enabled.pop("annotation_conflict_quarantine") == {"enabled": True, "group_count": 0,
                                                            "source_count": 0, "groups": []}
    assert enabled == default
    assert set(default) == {"records", "exclusions", "counts"}


def test_quarantined_cohort_materializes_and_reader_preserves_exact_full_coverage(materializer_inputs):
    from scripts import audited_aihub_snapshot as reader
    from scripts import materialize_audited_aihub_sources as materializer

    data, save, output, sources, labels, _ = materializer_inputs
    rows = copy.deepcopy(data["records"])
    for row, path in zip(rows, sources):
        _, actual_sha, size, phash = planner.original.image_evidence(path)
        assert actual_sha == row["source_sha256"]
        row.update(source_bytes=size, source_phash64=phash)
    # A genuine same-image copy with its own conflicting JSON, not a fabricated winner.
    source, label = sources[0].with_name("b.png"), labels[0].with_name("b.json")
    source.write_bytes(sources[0].read_bytes())
    annotation = json.loads(labels[0].read_text(encoding="utf-8"))
    annotation["IMAGE_INFO"]["FILE_NAME"] = source.name
    annotation["ANNOTATION_INFO"][0]["POINTS"][0][0] += 1
    label.write_text(json.dumps(annotation, ensure_ascii=False), encoding="utf-8")
    duplicate = copy.deepcopy(rows[0])
    duplicate.update(source_id=hashlib.sha1(f"Training/{label.parent.name}/b.json".encode()).hexdigest()[:20],
                     source_path_b64=planner.original.encode_path(source), label_path_b64=planner.original.encode_path(label),
                     label_sha256=hashlib.sha256(label.read_bytes()).hexdigest())
    duplicate["bbox_xywh"][0] += 1
    planned = planner.plan_records([*rows, duplicate], [], set(), quarantine_annotation_conflicts=True)
    assert planned["counts"]["accepted"] == 1 and planned["counts"]["excluded"] == 2
    data.update(planned, full_cohort=True, pending_checks=["materialized_image_leakage", "independent_hardware_gate"])
    args = save()
    report = materializer.materialize_sources(**args)
    assert report["full_cohort"] is True
    assert report["cohort_records"] == report["verified_sources"] == report["materialized_sources"] == 1
    assert report["quality_excluded_sources"] == report["unprocessed_sources"] == 0
    assert report["training_authorized"] is report["deployment_authorized"] is False
    evidence = reader.load_audited_aihub_snapshot(output / "report.json",
                                                hashlib.sha256((output / "report.json").read_bytes()).hexdigest(),
                                                cohort_path=args["cohort"])
    lineage = json.loads((output / "lineage.jsonl").read_text(encoding="utf-8"))
    assert evidence.split_for(output / lineage["image_ref"]) == "validation"
    assert lineage["source_id"] == rows[1]["source_id"]
    evidence.recheck()


def test_pure_helper_can_plan_without_protected_references():
    row = original(18)
    result = planner.plan_records([row], [], set())
    assert accepted_ids(result) == {row["source_id"]}
    assert result["exclusions"] == []


@pytest.mark.parametrize("field,value", [
    ("source_sha256", None), ("source_sha256", "x" * 64),
    ("source_phash64", None), ("source_phash64", "g" * 16),
])
def test_invalid_protected_evidence_is_not_silently_ignored(field, value):
    protected = {"source_sha256": "f" * 64, "source_phash64": "ffffffffffffffff"}
    if value is None:
        del protected[field]
    else:
        protected[field] = value
    with pytest.raises(planner.PlanError):
        planner.plan_records([original(17)], [protected], set())


@pytest.mark.parametrize("field,value", [
    ("source_sha256", None), ("source_sha256", "not-a-sha"),
    ("label_sha256", None), ("label_sha256", "x" * 64),
    ("source_phash64", None), ("source_phash64", "0" * 15),
    ("class_id", True), ("class_id", 9), ("class_name", "plastic"),
    ("split", "blind_test"), ("split", "train"),
    ("conditions", {"dent": 1, "label": -1, "foreign_material": -1}),
])
def test_invalid_original_evidence_is_not_accepted(field, value):
    row = original(14)
    if value is None:
        del row[field]
    else:
        row[field] = value
    with pytest.raises(planner.PlanError):
        planner.plan_records([row], unrelated_protected(), set())


def _encoded(value):
    return base64.urlsafe_b64encode(str(value).encode()).decode()


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def legacy_evidence(tmp_path):
    """Synthetic pinned metadata contracts, not an actual image-linking audit."""
    source = original(31, phash="0000000000000000")
    alias = original(32, phash=source["source_phash64"], source_sha=source["source_sha256"])
    outside = original(33, split="validation", phash="ffffffffffffffff")
    near = original(34, phash="fffffffffffffff0")
    unrelated = original(35, split="validation", phash="9696969696969696")
    originals = [source, alias, near, unrelated]
    protected = []
    links = []
    consumed = []
    for kind, number, recovered in (("train", 1, source), ("train_r", 2, source), ("val", 3, outside)):
        prefix = "train_r" if kind == "train_r" else kind + "_"
        split = "val" if kind == "val" else "train"
        path = f"/app/yolo_dataset_9class_v2/{split}/images/{prefix}{number:07}.jpg"
        legacy_sha = hashlib.sha256(path.encode()).hexdigest()
        protected.append({"source_path_b64": _encoded(path), "source_sha256": legacy_sha,
                          "source_phash64": "5555555555555555", "roles": ["qx3"]})
        links.append({"kind": kind, "index": number, "legacy_sha256": legacy_sha, "legacy_path_b64": _encoded(path),
                      "status": "verified_source_link", "reason": "exact_legacy_jpeg_bytes",
                      "regenerated_legacy_sha256": legacy_sha, "source_path_b64": recovered["source_path_b64"],
                      "source_sha256": recovered["source_sha256"], "annotation_path_b64": recovered["label_path_b64"],
                      "annotation_sha256": recovered["label_sha256"], "membership": "outside_v3", "v3_source_ids": []})
        consumed.append({"path_b64": _encoded(path), "sha256": legacy_sha})
    consumed += [{"path_b64": row[field], "sha256": row[sha]} for row in (source, outside)
                 for field, sha in (("source_path_b64", "source_sha256"), ("label_path_b64", "label_sha256"))]
    protected_path, link_path, fp_path = (tmp_path / name for name in ("protected.json", "links.json", "fingerprints.json"))
    protected_doc = {"schema": "protected_image_fingerprint_snapshot.v1", "status": "snapshot_complete",
                     "snapshot_only": True, "training_authorized": False, "deployment_authorized": False,
                     "missing_sources": 0, "expected_sources": len(protected), "verified_sources": len(protected), "records": protected}
    protected_sha = _write_json(protected_path, protected_doc)
    consumed.append({"path_b64": _encoded(protected_path), "sha256": protected_sha})
    link_doc = {"schema": "legacy_aihub_source_link_probe.v1", "status": "probe_complete", "partial_selection": False,
                "max_per_kind": 0, "candidate_index_is_search_only": True,
                "protected_legacy_counts": {"train": 1, "train_r": 1, "val": 1},
                "status_counts": {"verified_source_link": 3}, "records": links, "metadata_and_consumed_inputs": consumed,
                **{field: False for field in ("training_authorized", "blind_test_authorized", "deployment_authorized",
                                              "complete_original_lineage", "original_alias_uniqueness_proven")}}
    auditor_sha = hashlib.sha256(Path(planner.original.__file__).read_bytes()).hexdigest()
    fp_doc = {"schema": "protected_image_fingerprint_snapshot.v1", "status": "snapshot_complete", "snapshot_only": True,
              "inventory_sha256": "a" * 64, "code_sha256": {"audit_aihub_original_annotations.py": auditor_sha},
              "expected_sources": 2, "verified_sources": 2, "missing_sources": 0,
              "records": [{**{key: row[key] for key in ("source_sha256", "source_path_b64", "source_phash64", "image_height", "image_width", "source_bytes")},
                           "roles": ["qx3"]} for row in (source, outside)],
              **{field: False for field in ("training_authorized", "deployment_authorized", "blind_test_authorized", "selection_authorized")}}
    def save():
        link_sha = _write_json(link_path, link_doc)
        fp_doc["metadata_bindings"] = [{"path_b64": _encoded(link_path), "sha256": link_sha},
                                       {"path_b64": _encoded(protected_path), "sha256": protected_sha}]
        fp_sha = _write_json(fp_path, fp_doc)
        return {"link_report": link_path, "link_sha256": link_sha, "fingerprint_report": fp_path,
                "fingerprint_sha256": fp_sha, "protected_report": protected_path, "protected_sha256": protected_sha,
                "protected": protected, "originals": originals, "original_auditor_sha256": auditor_sha}
    return save, link_doc, fp_doc, originals


def test_legacy_exact_alias_and_outside_pool_fingerprint_extend_existing_exclusions(legacy_evidence):
    save, _, _, rows = legacy_evidence
    args = save()
    ids, references, summary, bindings = planner.load_legacy_exclusions(**args)
    assert ids == {rows[0]["source_id"], rows[1]["source_id"]}
    assert len(references) == 5
    assert summary["recovered_paths_outside_original_pool"] == 1
    assert summary["verified_source_links"] == 3
    assert summary["unresolved_legacy_references"] == 0
    assert summary["complete_original_lineage"] is summary["training_authorized"] is False
    plan = planner.plan_records(rows, references, ids)
    assert accepted_ids(plan) == {rows[3]["source_id"]}
    assert "protected_near_phash" in exclusions(plan)[rows[2]["source_id"]]
    assert plan["records"][0]["split"] == "validation"
    assert {b["path"] for b in bindings} == {str(args[k]) for k in ("link_report", "fingerprint_report", "protected_report")}


def test_unresolved_candidate_fields_do_not_become_source_authority(legacy_evidence):
    save, links, fps, _ = legacy_evidence
    links["records"][-1].update(status="unresolved", reason="legacy_jpeg_bytes_differ")
    links["status_counts"] = {"verified_source_link": 2, "unresolved": 1}
    fps["records"] = fps["records"][:1]
    fps["expected_sources"] = fps["verified_sources"] = 1
    ids, references, summary, _ = planner.load_legacy_exclusions(**save())
    assert len(ids) == 2 and len(references) == 4
    assert summary["unresolved_legacy_references"] == 1
    assert summary["recovered_paths_outside_original_pool"] == 0
    assert summary["unresolved"][0]["reason"] == "legacy_jpeg_bytes_differ"


@pytest.mark.parametrize("fault", ["missing", "duplicate", "partial", "bool_limit", "wrong_index", "reproduction", "consumed", "status_count", "authority"])
def test_legacy_link_coverage_and_reproduction_fail_closed(legacy_evidence, fault):
    save, links, _, _ = legacy_evidence
    if fault == "missing": links["records"].pop()
    elif fault == "duplicate": links["records"][1] = copy.deepcopy(links["records"][0])
    elif fault == "partial": links["partial_selection"] = True
    elif fault == "bool_limit": links["max_per_kind"] = False
    elif fault == "wrong_index": links["records"][0]["index"] = 4
    elif fault == "reproduction": links["records"][0]["regenerated_legacy_sha256"] = "b" * 64
    elif fault == "consumed": links["metadata_and_consumed_inputs"][0]["sha256"] = "b" * 64
    elif fault == "status_count": links["status_counts"]["verified_source_link"] = True
    else: links["complete_original_lineage"] = True
    with pytest.raises(planner.PlanError): planner.load_legacy_exclusions(**save())


@pytest.mark.parametrize("fault", ["missing", "extra", "duplicate", "bool_count", "wrong_code", "authority", "same_sha_phash", "same_sha_shape"])
def test_recovered_fingerprints_must_cover_exact_verified_source_union(legacy_evidence, fault):
    save, _, fp, _ = legacy_evidence
    if fault == "missing": fp["records"].pop()
    elif fault == "extra": fp["records"].append(copy.deepcopy(fp["records"][0]) | {"source_sha256": "c" * 64})
    elif fault == "duplicate": fp["records"][1] = copy.deepcopy(fp["records"][0])
    elif fault == "bool_count": fp["missing_sources"] = False
    elif fault == "wrong_code": fp["code_sha256"]["audit_aihub_original_annotations.py"] = "c" * 64
    elif fault == "authority": fp["selection_authorized"] = True
    elif fault == "same_sha_phash": fp["records"][0]["source_phash64"] = "c" * 16
    else: fp["records"][0]["image_width"] += 1
    with pytest.raises(planner.PlanError): planner.load_legacy_exclusions(**save())


@pytest.mark.parametrize("fault", ["source_sha", "annotation_sha", "annotation_path", "official_split"])
def test_exact_path_join_cannot_hide_original_pair_or_split_mismatch(legacy_evidence, fault):
    save, _, _, rows = legacy_evidence
    args = save()
    if fault == "source_sha": rows[0]["source_sha256"] = "d" * 64
    elif fault == "annotation_sha": rows[0]["label_sha256"] = "d" * 64
    elif fault == "annotation_path": rows[0]["label_path_b64"] = _encoded("/app/wrong.json")
    else: rows[0]["split"] = "validation"
    with pytest.raises(planner.PlanError): planner.load_legacy_exclusions(**args)


def test_public_link_validator_is_reusable_before_fingerprint_creation(legacy_evidence):
    save, _, _, _ = legacy_evidence
    args = save()
    args["fingerprint_report"].unlink()
    evidence = planner.validate_legacy_link_report(**{key: args[key] for key in
        ("link_report", "link_sha256", "protected_report", "protected_sha256", "protected")})
    assert len(evidence["verified_records"]) == 3
    assert len(evidence["recovered"]) == 2
    assert len(evidence["bindings"]) == 2


@pytest.mark.parametrize("fault", ["link", "fingerprint", "failed", "late_pin"])
def test_legacy_metadata_pins_and_late_changes_fail(legacy_evidence, monkeypatch, fault):
    save, _, _, _ = legacy_evidence
    args = save()
    if fault in {"link", "fingerprint"}: args[f"{fault}_report"].write_bytes(b"{}");
    elif fault == "failed": (args["link_report"].parent / "failed.json").touch()
    else:
        actual = planner.load_pinned
        def mutate(path, sha):
            value = actual(path, sha)
            if path == args["fingerprint_report"]: args["link_report"].write_bytes(b"{}")
            return value
        monkeypatch.setattr(planner, "load_pinned", mutate)
    with pytest.raises(planner.PlanError): planner.load_legacy_exclusions(**args)


def test_optional_legacy_cli_arguments_are_all_or_none(monkeypatch, tmp_path):
    required = []
    for name in ("original-report", "protected-report", "selected-manifest", "original-auditor"):
        required += [f"--{name}", str(tmp_path / name), f"--{name}-sha256", "a" * 64]
    monkeypatch.setattr(sys, "argv", ["planner", *required, "--output", str(tmp_path / "out"),
                                   "--legacy-link-report", str(tmp_path / "links")])
    with pytest.raises(planner.PlanError, match="required together"): planner.main()


@pytest.mark.parametrize("fault", [None, "nested_output", "terminal_failure", "annotation_quarantine"])
def test_legacy_cli_publishes_only_a_pinned_non_authoritative_cohort(legacy_evidence, monkeypatch, tmp_path, fault):
    save, links, fp, originals = legacy_evidence
    args = save()
    source_report = tmp_path / "originals.json"
    selected = tmp_path / "selected.csv"
    selected.write_text("stem,source_id,category,class_id,source_path\n", encoding="utf-8")
    selected_sha = hashlib.sha256(selected.read_bytes()).hexdigest()
    all_originals = list(originals)
    for split_index, split in enumerate(("training", "validation")):
        for cls, name in enumerate(planner.original.CLASS_NAMES):
            row = original(100 + split_index * 9 + cls, split=split)
            row.update(class_id=cls, class_name=name, declared_class=name)
            all_originals.append(row)
    if fault == "annotation_quarantine":
        for number, split in ((900, "training"), (902, "validation")):
            first = original(number)
            second = original(number + 1, split=split, source_sha=first["source_sha256"], phash=first["source_phash64"])
            second["bbox_xywh"][0] += 1
            all_originals.extend((first, second))
    raw = {"schema": "aihub_original_annotation_audit_v1", "perceptual_hash": planner.PHASH_CONVENTION,
           "training_authorized": False, "deployment_authorized": False,
           "selected": len(all_originals), "verified": len(all_originals), "quarantined": 0,
           "manifest_counts": {"all": len(all_originals)}, "records": all_originals}
    original_sha = _write_json(source_report, raw)
    snapshot = json.loads(args["protected_report"].read_text())
    snapshot.update(metadata_bindings=[], code_sha256={"audit_aihub_original_annotations.py": args["original_auditor_sha256"]})
    args["protected_sha256"] = _write_json(args["protected_report"], snapshot)
    for binding in links["metadata_and_consumed_inputs"]:
        if binding["path_b64"] == _encoded(args["protected_report"]): binding["sha256"] = args["protected_sha256"]
    args["link_sha256"] = _write_json(args["link_report"], links)
    fp["metadata_bindings"] = [{"path_b64": _encoded(args["link_report"]), "sha256": args["link_sha256"]},
                               {"path_b64": _encoded(args["protected_report"]), "sha256": args["protected_sha256"]}]
    args["fingerprint_sha256"] = _write_json(args["fingerprint_report"], fp)
    output = (tmp_path if fault == "nested_output" else tmp_path.parent) / (tmp_path.name + "-cohort")
    argv = ["planner", "--output", str(output)]
    if fault == "annotation_quarantine":
        argv.append("--quarantine-annotation-conflicts")
    for name, path, sha in (("original-report", source_report, original_sha),
                           ("protected-report", args["protected_report"], args["protected_sha256"]),
                           ("selected-manifest", selected, selected_sha),
                           ("original-auditor", Path(planner.original.__file__), args["original_auditor_sha256"]),
                           ("legacy-link-report", args["link_report"], args["link_sha256"]),
                           ("legacy-original-fingerprint-report", args["fingerprint_report"], args["fingerprint_sha256"])):
        argv += [f"--{name}", str(path), f"--{name}-sha256", sha]
    monkeypatch.setattr(sys, "argv", argv)
    if fault == "terminal_failure":
        check = planner.recheck_legacy_bindings
        def fail_after_publish(bindings, reports):
            if (output / "cohort.json").exists(): raise planner.PlanError("simulated terminal input drift")
            return check(bindings, reports)
        monkeypatch.setattr(planner, "recheck_legacy_bindings", fail_after_publish)
    if fault in {"nested_output", "terminal_failure"}:
        with pytest.raises(planner.PlanError): planner.main()
        assert not (output / "cohort.json").exists()
        assert (output / "failed.json").exists() is (fault == "terminal_failure")
        return
    planner.main()
    report = json.loads((output / "cohort.json").read_text())
    assert report["training_authorized"] is report["deployment_authorized"] is False
    assert report["protected_identity_scope"]["complete_original_lineage"] is False
    assert report["protected_identity_scope"]["legacy_v2_references_without_original_id"] == 1
    assert report["legacy_exclusion_evidence"]["verified_source_links"] == 3
    assert "legacy_transform_aliases" in report["pending_checks"]
    assert len(report["counts"]["accepted_by_split_class"]) == 18
    assert hashlib.sha256(source_report.read_bytes()).hexdigest() == original_sha
    if fault == "annotation_quarantine":
        assert report["annotation_conflict_quarantine"]["group_count"] == 2
        assert report["annotation_conflict_quarantine"]["source_count"] == 4
        assert not {f"{n:020x}" for n in range(900, 904)} & accepted_ids(report)
        assert "quarantine every member" in report["policy"]
    else:
        assert "annotation_conflict_quarantine" not in report
