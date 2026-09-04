"""Metadata cohort planning only: protect holdouts without publishing training data."""

import base64
import copy
import hashlib
import random

import pytest

from scripts import plan_aihub_original_cohort as planner


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
