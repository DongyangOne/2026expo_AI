"""Observe at most 32 protected sources with frozen YOLO; never create targets.

CLI execution uses the real batch-one CUDA detector only. The optional Python
provider is for tests and its reports explicitly cannot attest runtime execution.
Returned proposal counts are AFTER the model's confidence filtering and NMS;
zero eligible proposals is not a claim that the image contains no object.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import platform
import tempfile
from pathlib import Path

import cv2
import numpy as np

try:
    from scripts import audit_proposal_crop_reuse as files
    from scripts import prepare_proposal_verifier_dataset as prepare
    from scripts import verifier_preprocessing_contract as preprocessing
except ModuleNotFoundError:
    import audit_proposal_crop_reuse as files
    import prepare_proposal_verifier_dataset as prepare
    import verifier_preprocessing_contract as preprocessing

MAX_SOURCES = 32
MODEL_LIMIT = 512 * 1024**2
AUTHORITY = {"training_authorized": False, "blind_test_authorized": False,
             "deployment_authorized": False, "formal_protected_coverage": False,
             "label_authority": False, "selection_authorized": False,
             "semantic_truth_established": False}
CONFIG = {"device": "0", "batch": 1, "imgsz": 640, "conf": 0.1, "nms_iou": 0.7,
          "selection": "highest_confidence_then_original_order", "crop_size": 320,
          "padding": 0.08, "letterbox_fill": 114, "jpeg_quality": 92}


class ObservationError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise ObservationError(message)


def code_paths():
    paths = [Path(module.__file__).absolute() for module in (files, prepare, preprocessing)]
    return {path.name: files.checked_path(path) for path in [Path(__file__).absolute(), *paths]}


def render(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode()


def parse_json(content):
    def bad_constant(_):
        raise ObservationError("nonfinite JSON constant")
    return json.loads(content, object_pairs_hook=files.unique_object, parse_constant=bad_constant)


def roles(value):
    require(type(value) is list and value and all(type(item) is str and item in {"qx3", "capture", "known_audit"} for item in value)
            and len(set(value)) == len(value), "invalid protected roles")
    return sorted(value)


def decoded_path(value):
    require(type(value) is str, "missing encoded protected source path")
    return files.checked_path(Path(os.fsdecode(base64.b64decode(value, altchars=b"-_", validate=True))))


def validate_spec(spec):
    require(type(spec) is dict, "invalid inference spec")
    contract = preprocessing.validate_crop_preprocessing_spec(spec)
    require((contract.size, contract.padding_ratio, contract.fill) == (320, 0.08, 114), "noncanonical crop contract")
    detector = spec.get("detector")
    require(type(detector) is dict, "missing detector spec")
    expected = {"task": "detect", "input_size": 640, "candidate_confidence": 0.1,
                "nms_iou": 0.7, "proposal_selection": "highest_confidence_then_original_order"}
    require(all(type(detector.get(k)) is type(v) and detector[k] == v for k, v in expected.items()), "noncanonical detector spec")
    require(type(spec["crop"].get("jpeg_quality")) is int and spec["crop"]["jpeg_quality"] == 92, "noncanonical JPEG quality")


def runtime_info(real):
    result = {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__,
              "opencv_build_sha256": hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
              "provider_kind": "frozen_yolo_runtime" if real else "custom_test_provider",
              "runtime_detector_executed": real, "requested_configuration": CONFIG,
              "returned_count_stage": "after model confidence filtering and NMS; not raw prefilter proposals"}
    if real:
        import torch
        import ultralytics
        result.update(torch=torch.__version__, ultralytics=ultralytics.__version__,
                      cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                      device_name=torch.cuda.get_device_name(0),
                      tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                      tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
                      deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()))
    return result


def strict_yolo_predictions(records, *, model_path):
    """Same frozen prepare invocation, with strict Results validation before zip.

    Do not reuse prepare's permissive None/zip conversion for absence evidence:
    malformed detector output must fail, not become an empty proposal tuple.
    """
    from ultralytics import YOLO
    detector = YOLO(str(model_path), task="detect")
    for record in records:
        decoded = prepare._read_image(record.path)
        require(decoded is not None, "detector snapshot decode failed")
        results = detector.predict(source=[str(record.path)], device="0", batch=1,
            imgsz=640, conf=0.1, iou=0.7, stream=True, save=False, verbose=False)
        seen = 0
        for result in results:
            require(seen == 0, "detector returned extra source results")
            seen += 1
            device = detector.predictor.model.device
            require(device.type == "cuda" and device.index == 0, "detector did not execute on CUDA device zero")
            require(isinstance(result.orig_img, np.ndarray) and result.orig_img.shape == decoded.shape
                    and np.array_equal(result.orig_img, decoded) and tuple(result.orig_shape) == decoded.shape[:2],
                    "detector result pixels/shape differ from source snapshot")
            require(result.boxes is not None, "detector omitted boxes result")
            boxes, classes, confidences = [value.detach().cpu().numpy() for value in
                                           (result.boxes.xyxy, result.boxes.cls, result.boxes.conf)]
            require(boxes.ndim == 2 and boxes.shape[1] == 4 and classes.ndim == confidences.ndim == 1
                    and len(boxes) == len(classes) == len(confidences), "detector result array shape/count mismatch")
            require(all(array.dtype.kind in "fiu" and np.isfinite(array).all() for array in (boxes, classes, confidences)),
                    "detector returned nonfinite or nonnumeric arrays")
            require(np.equal(classes, np.floor(classes)).all(), "detector returned nonintegral class")
            proposals = []
            for bbox, raw_class, confidence in zip(boxes, classes, confidences, strict=True):
                class_id = int(raw_class)
                require(0 <= class_id < 9, "detector class outside nine-material mapping")
                names = result.names
                name = names.get(class_id) if isinstance(names, dict) else names[class_id]
                require(name == prepare.CLASS_NAMES[class_id], "detector material mapping mismatch")
                proposals.append(prepare.Proposal(class_id, float(confidence), tuple(float(v) for v in bbox), name))
            yield prepare.PredictedFrame(record, decoded.shape[1], decoded.shape[0], tuple(proposals))
        require(seen == 1, "detector omitted a result")


def observe(*, inventory: Path, inventory_sha256: str, protected_report: Path,
            protected_report_sha256: str, model: Path, model_sha256: str,
            inference_spec: Path, inference_spec_sha256: str, code_pins: dict[str, str],
            output: Path, prediction_provider=None):
    """Publish observations only after every original input and crop is rechecked."""
    inputs = [(inventory, inventory_sha256, files.METADATA_LIMIT),
              (protected_report, protected_report_sha256, files.METADATA_LIMIT),
              (model, model_sha256, MODEL_LIMIT), (inference_spec, inference_spec_sha256, files.METADATA_LIMIT)]
    inputs = [(files.checked_path(p), files.sha256_value(sha), limit) for p, sha, limit in inputs]
    inventory, protected_report, model, inference_spec = [p for p, _, _ in inputs]
    require(model.suffix == ".pt" and len({p for p, _, _ in inputs}) == 4, "distinct inputs and a .pt detector required")
    output = files.checked_path(output, exists=False)
    code = code_paths()
    require(type(code_pins) is dict and set(code_pins) == set(code), "exact producer/helper code pins required")
    for name, path in code.items():
        expected = files.sha256_value(code_pins[name])
        require(files.read_file(path, files.METADATA_LIMIT)[0] == expected, "code pin mismatch")
        inputs.append((path, expected, files.METADATA_LIMIT))
    roots = {p.parent for p, _, _ in inputs}
    require(not output.exists() and not any(output.is_relative_to(p) or p.is_relative_to(output) for p in roots),
            "output must be fresh and disjoint from input trees")
    real = prediction_provider is None
    accelerator_guard = prepare.eager_initialize_cuda_context("0") if real else None
    require(not real or accelerator_guard is not None, "CUDA context was not established")
    # Reserve the same-process CUDA client BEFORE metadata/model/source bulk IO.
    documents = {}
    for p, expected, limit in inputs[:4]:
        if p == model:
            continue
        actual, _, content = files.read_file(p, limit, keep=True)
        require(actual == expected, "input metadata SHA mismatch")
        documents[p] = parse_json(content)
    validate_spec(documents[inference_spec])
    inv, protected = documents[inventory], documents[protected_report]
    require(type(inv) is dict and set(inv) == {"records", "metadata_bindings"}
            and type(inv["records"]) is list and 1 <= len(inv["records"]) <= MAX_SOURCES
            and type(inv["metadata_bindings"]) is list, "expected 1..32 source-only inventory records")
    require(type(protected) is dict and protected.get("schema") == "protected_image_fingerprint_snapshot.v1"
            and protected.get("status") == "snapshot_complete" and protected.get("snapshot_only") is True
            and protected.get("consumer_must_rehash_sources") is True and type(protected.get("records")) is list,
            "invalid protected fingerprint snapshot")
    for key in ("training_authorized", "deployment_authorized", "blind_test_authorized", "selection_authorized"):
        require(protected.get(key) is False, "fingerprint snapshot cannot grant authority")
    count = len(protected["records"])
    require(count > 0 and all(type(protected.get(k)) is int and protected[k] == count for k in ("expected_sources", "verified_sources"))
            and type(protected.get("missing_sources")) is int and protected["missing_sources"] == 0, "incomplete protected snapshot")
    by_sha = {}
    for row in protected["records"]:
        require(type(row) is dict, "invalid fingerprint record")
        sha = files.sha256_value(row.get("source_sha256"))
        require(sha not in by_sha, "duplicate protected source SHA")
        by_sha[sha] = row
    chosen, seen_paths, seen_shas = [], set(), set()
    for row in inv["records"]:
        require(type(row) is dict and set(row) == {"sha256", "path", "roles"}, "invalid inventory source fields")
        sha, path, source_roles = files.sha256_value(row["sha256"]), files.checked_path(row["path"]), roles(row["roles"])
        require(sha not in seen_shas and path not in seen_paths and sha in by_sha, "duplicate or non-member source")
        reference = by_sha[sha]
        require(decoded_path(reference.get("source_path_b64")) == path and roles(reference.get("roles")) == source_roles,
                "source path/roles differ from fingerprint membership")
        require(all(type(reference.get(k)) is int and reference[k] > 0 for k in ("source_bytes", "image_width", "image_height")), "invalid fingerprint dimensions/bytes")
        require(not output.is_relative_to(path.parent) and not path.is_relative_to(output), "output overlaps source tree")
        chosen.append((sha, path, source_roles, reference))
        seen_shas.add(sha); seen_paths.add(path)
        inputs.append((path, sha, files.CROP_LIMIT))
    for binding in inv["metadata_bindings"]:
        require(type(binding) is dict and set(binding) == {"path", "sha256"}, "invalid inventory binding")
        path, sha = files.checked_path(binding["path"]), files.sha256_value(binding["sha256"])
        require(not output.is_relative_to(path.parent) and not path.is_relative_to(output), "output overlaps bound metadata")
        require(files.read_file(path, files.METADATA_LIMIT)[0] == sha, "inventory metadata binding mismatch")
        inputs.append((path, sha, files.METADATA_LIMIT))
    require(not any((p.parent / "failed.json").exists() for p, _, _ in inputs), "input has a failure marker")
    output.mkdir(parents=True, exist_ok=False)
    owned = (output.stat().st_dev, output.stat().st_ino)
    published = None
    report_path = output / "report.json"
    output_pins = []
    def recheck():
        for path, sha, limit in inputs + output_pins:
            require(files.read_file(path, limit)[0] == sha, "input or crop changed during observation")
        require(not any((p.parent / "failed.json").exists() for p, _, _ in inputs), "input failure marker appeared")
        current = files.checked_path(output).stat()
        require((current.st_dev, current.st_ino) == owned, "output ownership changed")
    try:
        with tempfile.TemporaryDirectory(prefix=".frozen-observation-", dir=output) as temporary:
            temporary = Path(temporary)
            model_snapshot = temporary / "detector.pt"
            # Streaming copy avoids holding the detector weights in host RAM.
            require(files.read_file(model, MODEL_LIMIT)[0] == model_sha256, "detector SHA mismatch")
            with model.open("rb") as source, model_snapshot.open("xb") as dest:
                size = 0
                for chunk in iter(lambda: source.read(1024**2), b""):
                    size += len(chunk)
                    require(size <= MODEL_LIMIT, "detector snapshot exceeds byte limit")
                    dest.write(chunk)
            require(files.read_file(model_snapshot, MODEL_LIMIT)[0] == model_sha256, "detector snapshot mismatch")
            snapshots, lookup = [], {}
            for sha, path, source_roles, reference in chosen:
                actual, size, content = files.read_file(path, files.CROP_LIMIT, keep=True)
                require((actual, size) == (sha, reference["source_bytes"]), "source bytes differ from protected snapshot")
                image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
                require(image is not None and image.shape[:2] == (reference["image_height"], reference["image_width"])
                        and image.shape[0] * image.shape[1] <= 16_000_000, "source decode/dimensions mismatch")
                snap = temporary / (sha + path.suffix)
                with snap.open("xb") as handle: handle.write(content)
                require(files.read_file(snap, files.CROP_LIMIT)[0] == sha, "source snapshot mismatch")
                record = prepare.SourceRecord(snap, "protected", sha, None)
                snapshots.append(record)
                lookup[snap] = (sha, path, source_roles, reference)
            recheck()
            runtime = runtime_info(real)
            frames = (strict_yolo_predictions(snapshots, model_path=model_snapshot)
                      if real else prediction_provider(snapshots))
            observations, observed = [], set()
            for frame in frames:
                require(frame.source.path in lookup and frame.source.path not in observed, "unexpected or duplicate detector source")
                require(frame.source == snapshots[len(observed)], "detector source order changed")
                observed.add(frame.source.path)
                sha, path, source_roles, reference = lookup[frame.source.path]
                require((frame.width, frame.height) == (reference["image_width"], reference["image_height"]), "detector source shape mismatch")
                checked = []
                for index, proposal in enumerate(frame.proposals):
                    require(type(proposal.class_id) is int and 0 <= proposal.class_id < 9
                            and proposal.class_name == prepare.CLASS_NAMES[proposal.class_id], "invalid detector class")
                    require(not isinstance(proposal.confidence, (bool, np.bool_)) and math.isfinite(proposal.confidence)
                            and 0 <= proposal.confidence <= 1, "invalid detector confidence")
                    # Validate EVERY returned box, including below-floor boxes.
                    bounds = preprocessing.padded_clipped_bbox(proposal.bbox, width=frame.width, height=frame.height, padding=0.08)
                    checked.append((index, proposal, bounds))
                eligible = [item for item in checked if item[1].confidence >= 0.1]
                item = {"source_sha256": sha, "source_path_b64": base64.urlsafe_b64encode(os.fsencode(path)).decode(),
                        "source_bytes": reference["source_bytes"], "image_width": frame.width, "image_height": frame.height,
                        "roles": source_roles, "returned_proposals_after_model_confidence_nms": len(checked),
                        "eligible_proposals": len(eligible), "below_confidence_floor": len(checked) - len(eligible),
                        "observation_status": "crop_generated" if eligible else "no_eligible_proposal",
                        "selected_proposal": None, "crop": None, "object_absence_established": False}
                if eligible:
                    index, proposal, bounds = max(eligible, key=lambda entry: (entry[1].confidence, -entry[0]))
                    content = files.read_file(frame.source.path, files.CROP_LIMIT, keep=True)
                    require(content[0] == sha, "replay source snapshot changed")
                    image = cv2.imdecode(np.frombuffer(content[2], dtype=np.uint8), cv2.IMREAD_COLOR)
                    crop, actual_bounds = preprocessing.crop_and_letterbox_bgr(image, proposal.bbox, padding=0.08, size=320, fill=114)
                    require(actual_bounds == bounds, "crop geometry differs")
                    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    require(ok, "crop encoding failed")
                    crop_path = output / "crops" / (sha + ".jpg")
                    crop_path.parent.mkdir(exist_ok=True)
                    files.checked_path(crop_path, exists=False)
                    with crop_path.open("xb") as handle: handle.write(encoded.tobytes())
                    crop_sha, crop_size, _ = files.read_file(crop_path, files.CROP_LIMIT)
                    output_pins.append((crop_path, crop_sha, files.CROP_LIMIT))
                    item["selected_proposal"] = {"index": index, "predicted_class_id": proposal.class_id,
                        "predicted_class_name": proposal.class_name, "confidence": proposal.confidence, "bbox_xyxy": list(proposal.bbox)}
                    item["crop"] = {"path": crop_path.relative_to(output).as_posix(), "sha256": crop_sha, "bytes": crop_size,
                                    "bounds_xyxy": list(bounds), "width": 320, "height": 320, "provenance": "actual_yolo_runtime_top1" if real else "custom_test_provider"}
                observations.append(item)
                print(json.dumps({"observed": len(observations), "requested": len(chosen), "status": item["observation_status"]}), flush=True)
            require(len(observed) == len(chosen), "detector omitted a source result")
            for record in snapshots:
                require(files.read_file(record.path, files.CROP_LIMIT)[0] == record.source_id, "source snapshot changed during inference")
            require(files.read_file(model_snapshot, MODEL_LIMIT)[0] == model_sha256, "model snapshot changed during inference")
            recheck()
            require(runtime_info(real) == runtime, "runtime configuration changed")
            _ = accelerator_guard  # Keep the CUDA client alive through all observations and checks.
        report = {"schema": "protected_proposal_observation.v1", "status": "observation_complete",
                  "artifact_role": "protected_reference_observation_not_training_or_formal_inventory",
                  "requested_sources": len(chosen), "observed_sources": len(observations),
                  "crop_generated": sum(row["crop"] is not None for row in observations),
                  "no_eligible_proposal": sum(row["crop"] is None for row in observations),
                  "runtime": runtime, "records": observations,
                  "bindings": {"inventory_sha256": inventory_sha256, "protected_report_sha256": protected_report_sha256,
                               "model_sha256": model_sha256, "inference_spec_sha256": inference_spec_sha256,
                               "code_sha256": dict(sorted(code_pins.items())),
                               "input_files": [{"path_b64": base64.urlsafe_b64encode(os.fsencode(p)).decode(), "sha256": sha} for p, sha, _ in inputs]},
                  **AUTHORITY}
        recheck()
        published = render(report)
        with report_path.open("xb") as handle: handle.write(published)
        recheck()
        require(files.read_file(report_path, files.METADATA_LIMIT, keep=True)[2] == published, "published observation changed")
        return report
    except BaseException:
        current = files.checked_path(output).stat()
        if (current.st_dev, current.st_ino) == owned:
            if published is not None and report_path.is_file() and not report_path.is_symlink() and report_path.read_bytes() == published:
                report_path.unlink()
            with (output / "failed.json").open("xb") as handle:
                handle.write(render({"status": "failed", "partial_outputs_not_authoritative": True, **AUTHORITY}))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "protected-report", "model", "inference-spec", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("inventory-sha256", "protected-report-sha256", "model-sha256", "inference-spec-sha256"):
        parser.add_argument("--" + name, type=files.sha256_value, required=True)
    parser.add_argument("--code-pin", action="append", required=True, help="exact implementation basename=sha256")
    args = vars(parser.parse_args(argv))
    pins = {}
    for value in args.pop("code_pin"):
        name, separator, sha = value.partition("=")
        require(separator and name not in pins, "malformed or duplicate code pin")
        pins[name] = files.sha256_value(sha)
    report = observe(**args, code_pins=pins)
    print(json.dumps({k: report[k] for k in ("status", "requested_sources", "crop_generated", "no_eligible_proposal")} | AUTHORITY), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, **AUTHORITY}), flush=True)
        raise SystemExit(1) from None
