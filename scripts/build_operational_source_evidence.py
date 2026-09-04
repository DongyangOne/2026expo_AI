"""Seal revalidated operational pseudo-annotations, never runtime YOLO crops.

This standalone adapter does not change or regenerate the existing teacher or
quality artifacts. Absolute source/input paths are local-only; raw client and
device metadata are not copied. All downstream training and deployment gates
remain mandatory.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

try:
    from scripts import assemble_operational_quality_exclusions as assembler
    from scripts import build_operational_teacher_manifest as teacher
    from scripts import build_independent_localization_consensus as localization
    from scripts import operational_quality_assembly_contract as quality
except ModuleNotFoundError:
    import assemble_operational_quality_exclusions as assembler
    import build_operational_teacher_manifest as teacher
    import build_independent_localization_consensus as localization
    import operational_quality_assembly_contract as quality


SCHEMA = "operational_source_evidence_bundle.v1"
SOURCE_SCHEMA = "operational_source_evidence.v1"
ROLE = "source_evidence_only_not_training_authority"
SOURCE_ROLE = "vlm_teacher_pseudo_label_train_only"
INPUT_NAMES = (
    "teacher_queue", "teacher_labels", "capture_inventory", "known_audit",
    "provider_a_manifest", "provider_a_model", "provider_a_spec",
    "provider_b_manifest", "provider_b_model", "provider_b_spec",
)
MODEL_INPUTS = {"provider_a_model", "provider_b_model"}
FILES = {"index": "sources.jsonl", "receipt": "source_evidence_receipt.json", "marker": "source_evidence.sha256"}
AUTHORITY = {name: False for name in ("ground_truth", "training", "calibration", "blind_test", "deployment")}
RECEIPT_FIELDS = {
    "schema_version", "artifact_role", "status", "image_root", "inputs", "code_sha256",
    "source_count", "empty_scene_sources_not_emitted", "rejected_records", "known_audit_counts",
    "canonical_json", "teacher_validation", "quality_excluded_source_count", "privacy",
    "authority", "output_sha256",
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _file_sha(path: Path, description: str) -> tuple[Path, str]:
    return assembler._stable_file_sha256(path, description=description)


def _code_paths() -> dict[str, Path]:
    return {
        "source_adapter": Path(__file__), "teacher_builder": Path(teacher.__file__),
        "quality_assembler": Path(assembler.__file__), "quality_validator": Path(quality.__file__),
        "localization_contract": Path(localization.__file__),
        "teacher_contract": Path(__file__).parent / "operational_teacher_contract.py",
        "objective_queue_preparer": Path(__file__).parent / "prepare_operational_capture_queue.py",
    }


def _index_rows(content: bytes, description: str) -> dict[str, dict]:
    rows = assembler._load_jsonl_bytes(content, description=description)
    result = {}
    for row in rows:
        sha = assembler._require_sha(row.get("sha256"), description=description)
        if sha in result:
            raise ValueError(f"{description} contains duplicate source SHA")
        result[sha] = {key: value for key, value in row.items() if key != "_input_line"}
    return result


def _recomputed_localization(sha: str, providers: list[dict], evidence: list[dict]) -> dict:
    first, second = providers
    overlap = localization.bbox_iou(first["bbox_xyxy"], second["bbox_xyxy"])
    if overlap < localization.IOU_THRESHOLD:
        raise ValueError("accepted source has no independent localization consensus")
    contract = {
        "schema_version": localization.CONTRACT_SCHEMA_VERSION,
        "method": localization.AGGREGATE_METHOD, "iou_threshold": localization.IOU_THRESHOLD,
        "aggregate_tolerance": localization.AGGREGATE_TOLERANCE,
        "source_image_sha256": sha, "providers": evidence,
    }
    return {
        "schema_version": localization.LOCALIZATION_SCHEMA_VERSION, "source_image_sha256": sha,
        "bbox_xyxy": [(first["bbox_xyxy"][index] + second["bbox_xyxy"][index]) / 2.0 for index in range(4)],
        "providers": providers, "provider_iou": overlap, "contract": contract,
        "contract_sha256": _sha(localization.canonical_json(contract).encode("utf-8")),
        "deployed_prediction_used": False, "consensus": True,
    }


def _input_paths(*, teacher_output_dir: Path, quality_assembly_receipt: Path, inputs: dict) -> dict[str, Path]:
    result = dict(inputs)
    result.update({f"teacher_output_{name}": teacher_output_dir / filename for name, filename in teacher.ARTIFACT_NAMES.items()})
    result.update({f"quality_{name}": quality_assembly_receipt.parent / filename for name, filename in quality.QUALITY_ASSEMBLY_FILES.items()})
    if quality_assembly_receipt.name != quality.QUALITY_ASSEMBLY_FILES["receipt"]:
        raise ValueError("quality assembly receipt basename mismatch")
    return result


def _prepare(*, input_paths: dict[str, Path], image_root: Path, consumer_output: Path):
    image_root = assembler._stable_directory(image_root, description="source image root")
    code_sha = {name: _file_sha(path, f"source adapter code {name}")[1] for name, path in _code_paths().items()}
    contents, bindings = {}, {}
    for name, path in input_paths.items():
        if name in MODEL_INPUTS:
            resolved, digest = _file_sha(path, name)
        else:
            resolved, content = assembler._stable_regular_file(path, description=name)
            contents[name] = content
            digest = _sha(content)
        bindings[name] = {"path": resolved.as_posix(), "sha256": digest}
    teacher_root = Path(bindings["teacher_output_lineage"]["path"]).parent
    if {path.name for path in teacher_root.iterdir()} != set(teacher.ARTIFACT_NAMES.values()):
        raise ValueError("teacher output file set mismatch")
    quality_value = assembler._load_json_bytes(contents["quality_manifest"], description="quality manifest")
    quality_bundle = quality._validate_operational_quality_assembly(
        receipt_path=Path(bindings["quality_receipt"]["path"]),
        quality_path=Path(bindings["quality_manifest"]["path"]), quality_value=quality_value,
        quality_content=contents["quality_manifest"], output_dir=consumer_output,
    )
    exclusions = quality._validate_quality_manifest(quality_value, assembly_bundle=quality_bundle)
    assembly_receipt = assembler._load_json_bytes(contents["quality_receipt"], description="quality receipt")
    for name in (*INPUT_NAMES, *(f"teacher_output_{key}" for key in teacher.ARTIFACT_NAMES)):
        if assembly_receipt["input_sha256"][name] != bindings[name]["sha256"]:
            raise ValueError(f"source inputs differ from full quality assembly: {name}")
    queues = _index_rows(contents["teacher_queue"], "teacher queue")
    labels = _index_rows(contents["teacher_labels"], "teacher labels")
    known_value = assembler._load_json_bytes(contents["known_audit"], description="known audit")
    known = assembler._known_audit_shas(contents["known_audit"])
    # Validate all source paths before the older dry-run helper resolves them.
    source_bindings = {}
    for sha, row in queues.items():
        source = assembler._resolve_source(image_root, row["image_ref"], row_number=1)
        _, digest = _file_sha(source, "operational capture source")
        if digest != sha:
            raise ValueError("operational source image SHA mismatch")
        source_bindings[source] = digest
    output_contents = {name: contents[f"teacher_output_{name}"] for name in teacher.ARTIFACT_NAMES}
    rejection = assembler._load_json_bytes(output_contents["rejections"], description="teacher rejections")
    assembler._validate_rejection_report(rejection, queue_rows=len(queues))
    lineage = assembler._load_json_bytes(output_contents["lineage"], description="teacher lineage")
    assembler._validate_lineage(
        lineage, input_sha256={name: bindings[name]["sha256"] for name in INPUT_NAMES},
        teacher_output_contents=output_contents, rejection_report=rejection,
    )
    policy = lineage["policy"]
    if policy["role"] != "train":
        raise ValueError("operational source evidence is train-only")
    for name in ("minimum_confidence", "burst_gap_seconds", "minimum_bbox_area_ratio", "maximum_bbox_area_ratio"):
        assembler._require_finite_number(policy[name], description=f"teacher policy {name}")
    kwargs = {name: Path(bindings[name]["path"]) for name in INPUT_NAMES}
    kwargs.update({name: lineage["inputs"][name] for name in ("provider_a_name", "provider_b_name")})
    kwargs.update({name: policy[name] for name in ("role", "fold", "minimum_confidence", "burst_gap_seconds", "minimum_bbox_area_ratio", "maximum_bbox_area_ratio")})
    replay = teacher.build_operational_teacher_manifest(
        **kwargs, image_root=image_root, output_dir=teacher_root, dry_run=True,
    )
    if replay.get("dry_run") is not True or not quality._exact_json_equal(replay["output_digests"], lineage["output_digests"]):
        raise ValueError("teacher dry-run output digests differ from sealed outputs")
    for name in ("accepted", "crop_ready_manifest_rows", "empty_scene_inventory_rows", "rejected_records"):
        if type(replay.get(name)) is not int or replay[name] != rejection[name]:
            raise ValueError("teacher dry-run counts differ from sealed outputs")
    provider_rows, provider_evidence = [], []
    for prefix in ("a", "b"):
        rows, evidence = localization._provider_rows(
            kwargs[f"provider_{prefix}_manifest"], provider=kwargs[f"provider_{prefix}_name"],
            model_file=kwargs[f"provider_{prefix}_model"], spec_file=kwargs[f"provider_{prefix}_spec"],
        )
        provider_rows.append(rows)
        provider_evidence.append(evidence)
    accepted = assembler._load_jsonl_bytes(output_contents["jsonl"], description="accepted teacher sources")
    if not accepted:
        raise ValueError("source evidence requires at least one accepted positive source")
    payloads, records = {}, []
    for raw in accepted:
        row = {key: value for key, value in raw.items() if key != "_input_line"}
        if set(row) != set(teacher.MANIFEST_FIELDS):
            raise ValueError("accepted teacher source schema mismatch")
        sha = assembler._require_sha(row["source_sha256"], description="accepted source SHA")
        if sha in known or sha in exclusions or sha not in queues or sha not in labels:
            raise ValueError("accepted source is protected, excluded, or missing original evidence")
        source = assembler._resolve_source(image_root, queues[sha]["image_ref"], row_number=1)
        if row["filepath"] != source.as_posix() or os.fsdecode(base64.urlsafe_b64decode(row["source_path_b64"])) != str(source):
            raise ValueError("accepted source filepath binding mismatch")
        if row["dent"] != -1 or type(row["dent"]) is not int or row["label"] != -1 or type(row["label"]) is not int:
            raise ValueError("unknown dent and label states must remain integer -1")
        if type(row["material"]) is not int or row["material"] not in range(9):
            raise ValueError("source evidence accepts only nine positive materials")
        if type(row["foreign_material"]) is not int or row["foreign_material"] not in (0, 1):
            raise ValueError("foreign-material pseudo-label must be integer 0 or 1")
        if row["category"] != teacher.CLASS_NAMES[row["material"]]:
            raise ValueError("material/category mapping mismatch")
        for name in ("source_width", "source_height", "source_object_count"):
            if type(row[name]) is not int or row[name] <= 0:
                raise ValueError("source dimensions/object count must be positive integers")
        if row["source_object_count"] != 1 or row["bbox_source"] != "independent_localization_consensus":
            raise ValueError("positive source must have one independent pseudo-annotation")
        captured_at = assembler._timestamp(queues[sha]["timestamp"], description="source timestamp").isoformat().replace("+00:00", "Z")
        if captured_at != row["capture_timestamp"]:
            raise ValueError("capture timestamp differs from sealed teacher lineage")
        original = labels[sha]
        expected_contract, _ = teacher.build_teacher_contract(original["model"], original["model_digest"])
        if not quality._exact_json_equal(original["teacher_contract"], expected_contract):
            raise ValueError("teacher contract must match with exact JSON types")
        loc = _recomputed_localization(sha, [rows[sha] for rows in provider_rows], provider_evidence)
        if not quality._exact_json_equal(original.get("independent_localization"), loc):
            raise ValueError("original localization differs from recomputed provider evidence")
        record = {
            "source_id": sha, "source_sha256": sha, "source_filepath": source.as_posix(),
            "captured_at": captured_at, "object_group": row["object_group"], "capture_session": row["capture_session"],
            "origin": row["origin"], "role": "train", "material": row["material"], "category": row["category"],
            "dent": -1, "label": -1, "foreign_material": row["foreign_material"], "source_object_count": 1,
            "source_width": row["source_width"], "source_height": row["source_height"],
            "source_bbox_xyxy": loc["bbox_xyxy"], "bbox_source": "independent_localization_consensus",
            "annotation_authority": SOURCE_ROLE, "runtime_detector_executed": False, "training_crop_ready": False,
            "teacher_output_sha256": _sha(_json_bytes(original)), "localizer_output_sha256": _sha(_json_bytes(loc)),
        }
        evidence = {
            "schema_version": SOURCE_SCHEMA, "artifact_role": ROLE, "record": record,
            "input_sha256": {name: item["sha256"] for name, item in bindings.items()},
            "independent_localization": loc, "authority": AUTHORITY,
        }
        name = f"source_evidence/{sha}.json"
        content = _json_bytes(evidence)
        payloads[name] = content
        records.append(dict(record, source_evidence_ref=name, auditor_sha256=_sha(content)))
    records.sort(key=lambda row: row["source_sha256"])
    if len(records) != len({row["source_sha256"] for row in records}):
        raise ValueError("accepted sources contain duplicate SHA")
    payloads[FILES["index"]] = b"".join(_json_bytes(row) for row in records)
    receipt = {
        "schema_version": SCHEMA, "artifact_role": ROLE, "status": "source_evidence_ready",
        "image_root": image_root.as_posix(), "inputs": bindings, "code_sha256": code_sha,
        "source_count": len(records), "empty_scene_sources_not_emitted": replay["empty_scene_inventory_rows"],
        "rejected_records": replay["rejected_records"],
        "known_audit_counts": dict(sorted(Counter(row["split"] for row in known_value.values()).items())),
        "canonical_json": "utf8;sorted_keys;compact_separators;no_nan;one_LF;teacher_row_excludes_only_internal_input_line",
        "teacher_validation": "same_inputs_and_policy_dry_run_output_digests_exact",
        "quality_excluded_source_count": len(exclusions),
        "privacy": {"local_only_contains_absolute_paths": True, "raw_client_device_metadata_exported": False},
        "authority": AUTHORITY, "output_sha256": {name: _sha(content) for name, content in sorted(payloads.items())},
    }
    payloads[FILES["receipt"]] = _json_bytes(receipt)
    payloads[FILES["marker"]] = "".join(f"{_sha(content)}  {name}\n" for name, content in sorted(payloads.items())).encode("ascii")
    return records, payloads, receipt, (bindings, source_bindings, code_sha, quality_bundle, teacher_root)


def _rehash(state) -> None:
    bindings, sources, codes, quality_bundle, teacher_root = state
    for name, item in bindings.items():
        if _file_sha(Path(item["path"]), f"final {name}")[1] != item["sha256"]:
            raise RuntimeError(f"source evidence input changed: {name}")
    for path, expected in sources.items():
        if _file_sha(path, "final operational source")[1] != expected:
            raise RuntimeError("source evidence image changed")
    if {name: _file_sha(path, f"final code {name}")[1] for name, path in _code_paths().items()} != codes:
        raise RuntimeError("source evidence code changed")
    if {path.name for path in teacher_root.iterdir()} != set(teacher.ARTIFACT_NAMES.values()):
        raise RuntimeError("teacher output file set changed")
    quality._rehash_operational_quality_assembly(quality_bundle)


def _verify_outputs(root: Path, payloads: dict[str, bytes]) -> None:
    root = assembler._stable_directory(root, description="source evidence bundle")
    entries = list(root.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("source evidence bundle contains a symlink")
    actual = {path.relative_to(root).as_posix() for path in entries if path.is_file()}
    if actual != set(payloads) or {path.relative_to(root).as_posix() for path in entries if path.is_dir()} != {"source_evidence"}:
        raise ValueError("source evidence file set mismatch")
    for name, expected in payloads.items():
        if assembler._stable_bytes(root / name, description="source evidence output") != expected:
            raise ValueError(f"source evidence artifact mismatch: {name}")


def build_source_evidence(*, teacher_output_dir: Path, quality_assembly_receipt: Path,
                          image_root: Path, output_dir: Path, **inputs) -> dict:
    if set(inputs) != set(INPUT_NAMES):
        raise ValueError("source adapter requires the exact original teacher input set")
    output = quality._reject_symlink_components(output_dir, "source evidence output")
    if output.exists():
        raise FileExistsError("source evidence output must be new and immutable")
    parent = assembler._stable_directory(output.parent, description="source evidence output parent")
    paths = _input_paths(teacher_output_dir=teacher_output_dir, quality_assembly_receipt=quality_assembly_receipt, inputs=inputs)
    for protected in (image_root, teacher_output_dir, quality_assembly_receipt.parent):
        if output.is_relative_to(protected.resolve(strict=True)):
            raise ValueError("source evidence output must not be nested in input evidence")
    _, payloads, receipt, state = _prepare(input_paths=paths, image_root=image_root, consumer_output=output)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    try:
        for name, content in payloads.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        _verify_outputs(staging, payloads)
        _rehash(state)
        staged_identity = staging.stat()
        try:
            assembler._publish_directory_no_replace(staging, output)
            _verify_outputs(output, payloads)
            _rehash(state)
        except BaseException:
            # Preserve a concurrent publisher's directory. Mark failure only
            # if this exact staged directory was exposed, including a rename
            # helper that raises after the successful rename.
            if output.is_dir() and not output.is_symlink():
                actual = output.stat()
                if (actual.st_dev, actual.st_ino) == (staged_identity.st_dev, staged_identity.st_ino):
                    try:
                        with (output / "failed.json").open("xb") as handle:
                            handle.write(_json_bytes({"source_evidence_ready": False}))
                    except FileExistsError:
                        pass
            raise
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def validate_source_evidence_bundle(bundle_dir: Path) -> list[dict]:
    """Read-only source and input revalidation; return typed source annotations."""
    root = assembler._stable_directory(bundle_dir, description="source evidence bundle")
    if (root / "failed.json").exists():
        raise ValueError("source evidence bundle has a failure marker")
    receipt_path = root / FILES["receipt"]
    receipt = assembler._load_json_bytes(assembler._stable_bytes(receipt_path, description="source evidence receipt"), description="source evidence receipt")
    if type(receipt) is not dict or set(receipt) != RECEIPT_FIELDS or receipt.get("schema_version") != SCHEMA:
        raise ValueError("source evidence receipt schema mismatch")
    names = {*INPUT_NAMES, *(f"teacher_output_{name}" for name in teacher.ARTIFACT_NAMES), *(f"quality_{name}" for name in quality.QUALITY_ASSEMBLY_FILES)}
    if type(receipt.get("inputs")) is not dict or set(receipt["inputs"]) != names:
        raise ValueError("source evidence input binding schema mismatch")
    paths = {}
    for name, binding in receipt["inputs"].items():
        if type(binding) is not dict or set(binding) != {"path", "sha256"} or type(binding["path"]) is not str:
            raise ValueError("source evidence input binding is invalid")
        assembler._require_sha(binding["sha256"], description="source evidence input SHA")
        paths[name] = Path(binding["path"])
        if not paths[name].is_absolute():
            raise ValueError("source evidence input path must be absolute")
    if type(receipt.get("image_root")) is not str or not Path(receipt["image_root"]).is_absolute():
        raise ValueError("source evidence image root must be absolute")
    records, payloads, expected_receipt, state = _prepare(
        input_paths=paths, image_root=Path(receipt["image_root"]), consumer_output=root,
    )
    if not quality._exact_json_equal(receipt, expected_receipt):
        raise ValueError("source evidence receipt differs from actual revalidated inputs")
    _verify_outputs(root, payloads)
    _rehash(state)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*INPUT_NAMES, "teacher_output_dir", "quality_assembly_receipt", "image_root", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    receipt = build_source_evidence(**vars(parser.parse_args()))
    print(json.dumps({key: receipt[key] for key in ("status", "source_count", "rejected_records", "quality_excluded_source_count")}), flush=True)


if __name__ == "__main__":
    main()
