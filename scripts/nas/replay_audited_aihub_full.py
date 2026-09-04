"""One real full-cohort replay; no training, blind, or deployment authority.

Run in the launcher's pinned, network-none GPU image. Inputs/code are read-only;
only a fresh output directory is writable. The launcher checks Docker exit/OOM.
The existing validator alone loads the full audited-original reader, once. This
runner compares bound inventories, not original pixels or teacher predictions.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
import time


MAX_METADATA_BYTES = 512 * 1024**2
CODE_FILES = tuple("scripts/" + name for name in (
    "validate_v4_background_candidates.py", "prepare_proposal_verifier_dataset.py",
    "verifier_preprocessing_contract.py", "audited_aihub_snapshot.py",
    "audit_aihub_original_annotations.py", "materialize_audited_aihub_sources.py",
    "assemble_operational_quality_exclusions.py", "prepare_operational_capture_queue.py",
    "build_operational_teacher_manifest.py", "build_v4_quality_exclusion_manifest.py",
    "operational_teacher_contract.py", "build_independent_localization_consensus.py",
    "nas/run_v4_reproducible_generation.sh", "nas/replay_audited_aihub_full.py",
))
WORKSPACE_NAMES = ("manifest.csv", "dataset_info.json", "training", "validation")
AUTHORITY = {"training_authorized": False, "blind_test_authorized": False,
             "deployment_authorized": False, "promotion_authorized": False}
RAW_ROLE = "raw_v4_reproducible_generation_not_validation_or_promotion_authority"
REPLAY_ROLE = "v4_development_candidates_not_blind_or_deployment_authority"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha_argument(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA256 pin")
    return value


def regular(path, *, directory=False, exists=True):
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "absolute traversal-free path required")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink input/output forbidden")
    if exists:
        require(path.is_dir() if directory else path.is_file(), "required regular input missing")
    return path


def identity(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def consume(path, *, content=False, limit=None):
    path = regular(path)
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), "non-regular input")
    if limit is not None:
        require(before.st_size <= limit, "metadata exceeds byte limit")
    h, chunks, size = hashlib.sha256(), [], 0
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            require(limit is None or size <= limit, "metadata grew beyond byte limit")
            h.update(block)
            if content:
                chunks.append(block)
        closed = os.fstat(stream.fileno())
    after = regular(path).stat()
    key = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    # Windows path stat and fstat disagree about ctime semantics; compare each
    # kind with itself, while inode/mtime/size agree across all four observations.
    require(key(before) == key(opened) == key(closed) == key(after)
            and before.st_ctime_ns == after.st_ctime_ns and opened.st_ctime_ns == closed.st_ctime_ns
            and size == before.st_size, "input changed during read")
    return b"".join(chunks) if content else None, h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def parse_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def invalid(_):
        raise ValueError("non-finite JSON number")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def read_json(path):
    return parse_json(consume(path, content=True, limit=MAX_METADATA_BYTES)[0])


def relative(value):
    require(type(value) is str and "\\" not in value, "invalid relative inventory path")
    path = PurePosixPath(value)
    require(value and not path.is_absolute() and ".." not in path.parts
            and path.as_posix() == value and value != ".", "unsafe relative inventory path")
    return path


def decoded_path(value):
    require(type(value) is str, "missing original path")
    return regular(Path(os.fsdecode(base64.b64decode(value, altchars=b"-_", validate=True))))


def marker(raw):
    result = {}
    for line in raw.decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        require(match is not None, "invalid generation SHA marker")
        path = regular(Path(match[2]))
        require(path not in result, "duplicate generation marker path")
        result[path] = match[1]
    require(result, "empty generation marker")
    return result


def raw_tree(root):
    regular(root, directory=True)
    result = {}
    for p in sorted(root.rglob("*")):
        regular(p, directory=p.is_dir())
        if p.is_file():
            result[p.relative_to(root).as_posix()] = {"size": p.stat().st_size, "sha256": consume(p)[1]}
    return result


def publish(path, value, publications):
    raw = json_bytes(value)
    fd, name = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        publications[path] = (hashlib.sha256(raw).hexdigest(), identity(temp)[:2])
        os.link(temp, path)  # Exclusive; a foreign ready marker is never replaced.
    finally:
        temp.unlink(missing_ok=True)


def create_workspace(workspace, raw):
    workspace.mkdir(mode=0o700)
    for name in WORKSPACE_NAMES:
        source = regular(raw / name, directory=name in ("training", "validation"))
        os.symlink(os.path.relpath(source, workspace), workspace / name, target_is_directory=source.is_dir())


def load_validator(code_root):
    sys.path.insert(0, str(code_root))
    validator = importlib.import_module("scripts.validate_v4_background_candidates")
    for module_name in ("validate_v4_background_candidates", "prepare_proposal_verifier_dataset",
                        "verifier_preprocessing_contract"):
        module = importlib.import_module("scripts." + module_name)
        require(regular(module.__file__) == code_root / "scripts" / (module_name + ".py"),
                "imported code escaped frozen code root")
    return validator


def run(*, generation_dir: Path, code_root: Path, output_dir: Path, original_dataset_root: Path,
        detector_model: Path, detector_model_sha256: str, inference_spec: Path, inference_spec_sha256: str,
        audited_aihub_report: Path, audited_aihub_report_sha256: str,
        audited_aihub_cohort: Path, audited_aihub_cohort_sha256: str,
        generation_ready_sha256: str, manifest_sha256: str, dataset_info_sha256: str,
        code_pins: dict[str, str]):
    generation_dir, code_root, original_dataset_root = (regular(p, directory=True)
        for p in (generation_dir, code_root, original_dataset_root))
    output_dir = regular(output_dir, exists=False)
    detector_model, inference_spec, audited_aihub_report, audited_aihub_cohort = map(regular,
        (detector_model, inference_spec, audited_aihub_report, audited_aihub_cohort))
    require(audited_aihub_report.name == "report.json", "materializer report.json required")
    metadata_bytes, metadata_sha = consume(audited_aihub_report, content=True, limit=MAX_METADATA_BYTES)
    require(metadata_sha == sha_argument(audited_aihub_report_sha256), "materializer report pin mismatch")
    metadata = parse_json(metadata_bytes).get("metadata_bindings")
    require(type(metadata) is list and metadata, "materializer metadata bindings required")
    protected_roots = [generation_dir, code_root, original_dataset_root, detector_model.parent,
                       inference_spec.parent, audited_aihub_report.parent, audited_aihub_cohort.parent]
    protected_roots.extend(regular(item["path"]).parent for item in metadata)
    require(not output_dir.exists() and all(not output_dir.is_relative_to(p)
            and not p.is_relative_to(output_dir) for p in protected_roots), "output is not fresh/disjoint")
    require(type(code_pins) is dict and set(code_pins) == set(CODE_FILES), "exact frozen code pin set required")
    require(regular(__file__) == code_root / "scripts/nas/replay_audited_aihub_full.py", "runner is outside frozen code root")
    require(not (code_root / "scripts/__init__.py").exists(), "unexpected scripts package initializer")
    raw, control, data = generation_dir / "raw", generation_dir / "control", audited_aihub_report.parent
    regular(raw, directory=True); regular(control, directory=True)
    fixed = {control / "raw_generation_ready.json": generation_ready_sha256,
             raw / "manifest.csv": manifest_sha256, raw / "dataset_info.json": dataset_info_sha256,
             detector_model: detector_model_sha256, inference_spec: inference_spec_sha256,
             audited_aihub_report: audited_aihub_report_sha256, audited_aihub_cohort: audited_aihub_cohort_sha256}
    for value in fixed.values():
        sha_argument(value)
    pins, publications, source_identities = {}, {}, {}
    def pin(path, expected=None, *, content=False):
        blob, sha = consume(path, content=content, limit=MAX_METADATA_BYTES if content else None)
        require(expected is None or sha == sha_argument(expected), "frozen input/code hash mismatch")
        require(path not in pins or pins[path] == sha, "conflicting/changed input pin")
        pins[path] = sha
        return blob
    for name, expected in code_pins.items():
        pin(code_root / name, expected)
    # Large source/crop scans happen only after acquiring CUDA in this process.
    output_dir.mkdir(parents=True, exist_ok=False)
    owned = identity(output_dir)[:2]
    ready_path, workspace = output_dir / "replay_ready.json", output_dir / "replay"
    failure_paths = (control / "failed.txt", raw / "failed.json", data / "failed.json", audited_aihub_cohort.parent / "failed.json")
    stage, started = "preflight", time.monotonic()
    def owned_output():
        regular(output_dir, directory=True)
        require(identity(output_dir)[:2] == owned, "output directory ownership changed")
    def failures_absent():
        require(all(not p.exists() and not p.is_symlink() for p in failure_paths), "upstream failure marker present")
    try:
        failures_absent()
        validator = load_validator(code_root)
        guard = validator.eager_initialize_cuda_context("0")
        require(guard is not None, "same-process CUDA context required")
        print(json.dumps({"stage": "preflight", "status": "cuda_context_ready"}), flush=True)
        for p, expected in fixed.items():
            pin(p, expected)
        info, ready, materialized = (read_json(p) for p in
            (raw / "dataset_info.json", control / "raw_generation_ready.json", audited_aihub_report))
        expected_binding = {"report_path": audited_aihub_report.as_posix(), "report_sha256": audited_aihub_report_sha256,
                            "cohort_path": audited_aihub_cohort.as_posix(), "cohort_sha256": audited_aihub_cohort_sha256,
                            "require_full_cohort": True}
        require(info.get("audited_aihub_snapshot") == expected_binding
                and type(info["audited_aihub_snapshot"].get("require_full_cohort")) is bool,
                "full audited snapshot binding required")
        require("operational_source_evidence" not in info, "AIHub-only replay required")
        require(regular(info["model"]) == detector_model and regular(info["manifest"]) == raw / "manifest.csv"
                and regular(info["data"]) == data / "dataset.yaml" and regular(info["dataset_dir"], directory=True) == data,
                "generation input path mismatch")
        require(type(info["inference"].get("batch")) is int and info["inference"]["batch"] == 1
                and info["inference"].get("device") == "0", "frozen batch1 GPU generation required")
        require(ready.get("status") == "raw_generation_ready" and ready.get("artifact_role") == RAW_ROLE
                and type(ready.get("batch")) is int and ready["batch"] == 1, "invalid raw ready contract")
        require(all(ready.get(k) is False for k in ("validator_authority", "judge_authority", "training_authority",
                "blind_test_authority", "production_deployment_authorized")), "raw generation cannot grant authority")
        require(materialized.get("schema") == "audited_aihub_source_snapshot_v1"
                and materialized.get("status") == "snapshot_complete" and materialized.get("full_cohort") is True
                and type(materialized.get("requested_max_sources")) is int and materialized["requested_max_sources"] == 0
                and type(materialized.get("unprocessed_sources")) is int and materialized["unprocessed_sources"] == 0
                and materialized.get("cohort_sha256") == audited_aihub_cohort_sha256, "incomplete materialized snapshot")
        require(all(materialized.get(k) is False for k in ("training_authorized", "blind_test_authorized", "deployment_authorized")), "invalid snapshot authority")
        # Only metadata identities here, including quality-excluded originals.
        # Their image/JSON SHA and reproduction are checked by the one reader
        # invocation inside the validator; this closes the later ready boundary.
        cohort_metadata = parse_json(pin(audited_aihub_cohort, audited_aihub_cohort_sha256, content=True))
        cohort_rows = cohort_metadata.get("records")
        require(type(cohort_rows) is list and len(cohort_rows) == materialized.get("cohort_records"), "cohort metadata coverage mismatch")
        for row in cohort_rows:
            for field in ("source_path_b64", "label_path_b64"):
                p = decoded_path(row[field])
                require(p.is_relative_to(original_dataset_root), "original path outside declared read-only root")
                require(p not in source_identities or source_identities[p] == identity(p), "original identity changed")
                source_identities[p] = identity(p)
        del cohort_metadata, cohort_rows
        snapshot_ready = parse_json(pin(data / "snapshot_ready.json", content=True))
        require(snapshot_ready.get("status") == "snapshot_complete" and snapshot_ready.get("report_sha256") == audited_aihub_report_sha256, "snapshot ready binding mismatch")
        input_marker, output_marker = control / "inputs.sha256", control / "outputs.sha256"
        inp, out = marker(pin(input_marker, content=True)), marker(pin(output_marker, content=True))
        inventory_path, raw_inventory_path = control / "dataset_input_inventory.json", control / "raw_output_inventory.json"
        expected_inputs = {detector_model, data / "dataset.yaml", inventory_path, audited_aihub_report, audited_aihub_cohort}
        expected_inputs.update(code_root / name for name in ("scripts/prepare_proposal_verifier_dataset.py",
            "scripts/verifier_preprocessing_contract.py", "scripts/nas/run_v4_reproducible_generation.sh",
            "scripts/audited_aihub_snapshot.py", "scripts/audit_aihub_original_annotations.py", "scripts/materialize_audited_aihub_sources.py"))
        require(set(inp) == expected_inputs, "generation input marker set mismatch")
        require(set(out) == {raw / "manifest.csv", raw / "dataset_info.json", raw_inventory_path}, "generation output marker set mismatch")
        for p, expected in {**inp, **out}.items():
            pin(p, expected)
        require(ready.get("bindings") == {"input_marker_sha256": pins[input_marker], "output_marker_sha256": pins[output_marker],
                "manifest_sha256": manifest_sha256, "dataset_info_sha256": dataset_info_sha256}, "raw ready bindings mismatch")
        inventory, crop_inventory = read_json(inventory_path), read_json(raw_inventory_path)
        require(inventory.get("contract") == "resolved_yolo_train_val_sources_and_label_sidecars_sha256.v1"
                and inventory.get("data_path") == (data / "dataset.yaml").as_posix()
                and inventory.get("dataset_dir") == data.as_posix(), "source inventory root mismatch")
        expected_assets = {}
        lineage_raw = pin(data / "lineage.jsonl", materialized["lineage_sha256"], content=True)
        pin(data / "excluded.jsonl", materialized["excluded_sha256"])
        lineage_count = 0
        for line in lineage_raw.splitlines():
            require(bool(line), "empty materializer lineage row")
            row = parse_json(line); lineage_count += 1
            require(row.get("split") in ("training", "validation"), "invalid materializer split")
            for field in ("source_path_b64", "label_path_b64"):
                original_path = decoded_path(row[field])
                require(original_path.is_relative_to(original_dataset_root), "original path outside declared read-only root")
                require(source_identities.get(original_path) == identity(original_path), "original identity changed")
            for kind, ref_key, sha_key in (("source", "image_ref", "image_sha256"), ("label", "label_ref", "yolo_label_sha256")):
                p = data / relative(row[ref_key])
                require(p not in expected_assets, "duplicate materialized source/label path")
                expected_assets[p] = (kind, row["split"], sha_argument(row[sha_key]))
        del lineage_raw
        require(type(materialized.get("materialized_sources")) is int and lineage_count == materialized["materialized_sources"] > 0,
                "materializer lineage count mismatch")
        assets, seen = inventory.get("artifacts"), set()
        require(type(assets) is list and type(inventory.get("artifact_count")) is int
                and inventory["artifact_count"] == len(assets) == len(expected_assets), "source inventory coverage mismatch")
        for item in assets:
            p = regular(item["path"])
            require(p not in seen and expected_assets.get(p) == (item.get("kind"), item.get("split"), item.get("sha256"))
                    and item.get("exists") is True and type(item.get("size")) is int and p.stat().st_size == item["size"], "source inventory differs from materialized lineage")
            seen.add(p); source_identities[p] = identity(p)
        require(seen == set(expected_assets), "materialized source membership mismatch")
        del assets, expected_assets, inventory
        print(json.dumps({"stage": "preflight", "status": "source_inventory_bound",
                          "materialized_sources": lineage_count}), flush=True)
        expected_tree = {}
        require(crop_inventory.get("root") == raw.as_posix(), "raw inventory root mismatch")
        for item in crop_inventory.get("files", []):
            ref = relative(item["path"]).as_posix()
            require(ref not in expected_tree and type(item.get("size")) is int and item["size"] > 0, "invalid raw inventory entry")
            expected_tree[ref] = {"size": item["size"], "sha256": sha_argument(item["sha256"])}
        require(type(crop_inventory.get("file_count")) is int and crop_inventory["file_count"] == len(expected_tree) > 0, "raw inventory count mismatch")
        raw_rows = 0
        manifest_bytes = pin(raw / "manifest.csv", manifest_sha256, content=True)
        for row in csv.DictReader(io.StringIO(manifest_bytes.decode("utf-8-sig"), newline="")):
            require(row.get("origin") == "aihub_original_annotation_v1", "unexpected non-AIHub origin")
            raw_rows += 1
        require(raw_rows > 0, "empty raw manifest")
        del manifest_bytes
        def recheck():
            owned_output(); failures_absent()
            for p, expected in pins.items():
                require(consume(p)[1] == expected, "frozen input/code changed")
            for name in CODE_FILES:
                if name.endswith(".py"):
                    module = sys.modules.get(name[:-3].replace("/", "."))
                    if module is not None:
                        require(regular(module.__file__) == code_root / name, "imported code escaped frozen code root")
            require(raw_tree(raw) == expected_tree, "frozen raw inventory changed")
            for p, before in source_identities.items():
                regular(p)
                require(identity(p) == before, "original/materialized source or label changed")
            if workspace.exists():
                regular(workspace, directory=True)
                for name in WORKSPACE_NAMES:
                    p = workspace / name
                    require(p.is_symlink() and not os.path.isabs(os.readlink(p)) and p.resolve(strict=True) == raw / name, "workspace alias changed")
            for p, (sha, owned_file) in publications.items():
                regular(p)
                require(identity(p)[:2] == owned_file and consume(p)[1] == sha, "published artifact changed")
        recheck()
        publish(output_dir / "input_bindings.json", {"schema": "audited_aihub_full_replay_inputs.v1",
            "pins": {p.as_posix(): sha for p, sha in pins.items()}, "materialized_sources": lineage_count,
            "raw_rows": raw_rows, "audited_aihub_snapshot": expected_binding,
            "full_original_reader_invoked_by": "strict validator only", **AUTHORITY}, publications)
        create_workspace(workspace, raw)
        stage = "strict_replay"
        print(json.dumps({"stage": stage, "status": "started", "raw_rows": raw_rows}), flush=True)
        checked, report_path = workspace / "validated_manifest.csv", workspace / "validation_report.json"
        # No custom-provider or diagnostic escape hatch is exposed by this runner.
        report = validator.validate_manifest(input_manifest=workspace / "manifest.csv", dataset_info=workspace / "dataset_info.json",
            detector_model=detector_model, inference_spec=inference_spec, output_manifest=checked, output_report=report_path,
            prediction_provider=None, diagnostic_only=False, audited_aihub_report=audited_aihub_report,
            audited_aihub_report_sha256=audited_aihub_report_sha256, audited_aihub_cohort=audited_aihub_cohort)
        require(report == read_json(report_path), "validator return/publication mismatch")
        require(report.get("artifact_role") == REPLAY_ROLE and report.get("ready_for_lineage_upgrade") is True
                and type(report.get("rows")) is int and report["rows"] == raw_rows
                and report.get("blind_test_eligible") is False and report.get("production_deployment_authorized") is False,
                "strict validator report contract mismatch")
        expected_provenance = {"provider_kind": "frozen_yolo_runtime", "runtime_detector_executed": True,
            "runtime_top1_replayed": True, "provided_top1_predictions_matched": True, "proposal_class_confidence_bbox_matched": True,
            "confidence_abs_tolerance": 1e-6, "bbox_abs_tolerance": 1e-4, "production_or_blind_authority": False,
            "cuda_client_initialized_before_source_crop_scan": True, "detector_artifact_bytes_bound": True,
            "inference_spec_bytes_bound": True, "dataset_info_bytes_bound": True, "source_bbox_crop_bytes_recomputed": True}
        provenance = report.get("contract", {}).get("proposal_provenance", {})
        require(all(type(provenance.get(k)) is type(v) and provenance[k] == v for k, v in expected_provenance.items()), "strict runtime provenance mismatch")
        validated_sha = consume(checked)[1]
        require(report.get("bindings") == {"input_manifest_sha256": manifest_sha256, "dataset_info_sha256": dataset_info_sha256,
            "detector_model_sha256": detector_model_sha256, "inference_spec_sha256": inference_spec_sha256,
            "validated_manifest_sha256": validated_sha}, "validator output binding mismatch")
        for p in (checked, report_path):
            publications[p] = (consume(p)[1], identity(p)[:2])
        recheck()
        stage = "publication"
        summary = {"schema": "audited_aihub_full_replay.v1", "status": "strict_replay_complete",
            "ready_for_lineage_upgrade": True, "runtime_replay_count": 1, "raw_rows": raw_rows,
            "materialized_sources": lineage_count, "seconds": time.monotonic() - started,
            "audited_aihub_snapshot": expected_binding, "validation_report_sha256": publications[report_path][0],
            "validated_manifest_sha256": validated_sha, "input_bindings_sha256": publications[output_dir / "input_bindings.json"][0],
            "scope": "lineage input only; not training, blind, promotion, or deployment approval",
            "launcher_must_verify_container_exit_and_oom": True, **AUTHORITY}
        publish(ready_path, summary, publications)
        recheck()  # A publication-boundary failure dominates any exposed ready.
        return summary
    except BaseException as error:
        try:
            owned_output()
            try:
                owned_ready = publications.get(ready_path)
                if owned_ready and ready_path.is_file() and not ready_path.is_symlink() and identity(ready_path)[:2] == owned_ready[1] and consume(ready_path)[1] == owned_ready[0]:
                    ready_path.unlink()  # Only our exact newly published ready.
            except Exception:
                pass  # Still publish failure if a ready file races its cleanup.
            publish(output_dir / "failed.json", {"status": "failed", "stage": stage,
                "exception_type": type(error).__name__, "partial_outputs_preserved": True,
                "ready_for_lineage_upgrade": False, **AUTHORITY}, {})
        except Exception:
            pass  # Never alter a replaced output directory or foreign marker.
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("generation-dir", "code-root", "output-dir", "original-dataset-root", "detector-model",
                 "inference-spec", "audited-aihub-report", "audited-aihub-cohort"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("detector-model", "inference-spec", "audited-aihub-report", "audited-aihub-cohort",
                 "generation-ready", "manifest", "dataset-info"):
        parser.add_argument("--" + name + "-sha256", required=True, type=sha_argument)
    parser.add_argument("--code-pin", action="append", required=True, metavar="RELATIVE_PATH=SHA256")
    args = vars(parser.parse_args(argv)); pins = {}
    try:
        for value in args.pop("code_pin"):
            name, sha = value.split("=", 1)
            require(name not in pins, "duplicate code pin")
            pins[name] = sha_argument(sha)
        result = run(**args, code_pins=pins)
        print(json.dumps({"status": result["status"], "raw_rows": result["raw_rows"], **AUTHORITY}), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "exception_type": type(error).__name__, **AUTHORITY}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
