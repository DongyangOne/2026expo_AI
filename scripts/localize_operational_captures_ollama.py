"""Produce one immutable, raw-image-only Ollama localization provider run.

Run independently for each provider. A provider manifest is localization
evidence, not ground truth, training authorization, or deployment approval.
The model file must be the real GGUF FROM blob, not a model metadata JSON.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import struct
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

try:
    from scripts import label_operational_captures_ollama as teacher
    from scripts import prepare_operational_capture_queue as capture_queue
    from scripts.build_independent_localization_consensus import (
        canonical_json, provider_output_core,
    )
except ModuleNotFoundError:
    import label_operational_captures_ollama as teacher
    import prepare_operational_capture_queue as capture_queue
    from build_independent_localization_consensus import canonical_json, provider_output_core


SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["single", "empty", "multiple", "ambiguous"]},
        "bbox_norm": {"type": "array", "items": {"type": "number"}, "maxItems": 4},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["status", "bbox_norm", "confidence"],
    "additionalProperties": False,
}
MINIMUM_CONFIDENCE = 0.80
OPTIONS = {"temperature": 0, "seed": 20260904, "num_predict": 256, "num_ctx": 8192}
PROMPT = """Locate the one primary discarded item in this raw recycling-kiosk photograph.
Return only the JSON schema below. Do not classify its material. Do not include hands,
arms, bin walls, or background in the box. Small attached accessories belong to the
same item. A dented or crushed item is still a valid item. If there is no item use
empty; several separate items use multiple; an obscured/cut-off/unreadable item
boundary uses ambiguous. For those three statuses return bbox_norm=[].
For a clearly visible single item, return its tight bounding box as
bbox_norm=[left,top,right,bottom], with each axis normalized from 0 to 1000:
0 is the left/top image edge and 1000 is the right/bottom image edge.
Never invent a box for an uncertain boundary. confidence is between 0 and 1.
""" + canonical_json(SCHEMA)
FILES = {
    "manifest": "provider_manifest.jsonl", "spec": "inference_spec.json",
    "raw": "raw_replies.jsonl", "show": "model_show.json",
    "receipt": "localization_receipt.json", "marker": "localization.sha256",
}


def _json_bytes(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _json(content: bytes | str) -> object:
    return json.loads(content, object_pairs_hook=capture_queue._reject_duplicate_keys)


def _post(url: str, endpoint: str, payload: dict, timeout: int) -> bytes:
    request = urllib.request.Request(
        url.rstrip("/") + endpoint, data=_json_bytes(payload),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with teacher._open_no_redirect(request, timeout) as response:
        return response.read()


def _weight_sha(path: Path) -> str:
    path = teacher._checked_path(path, "model weight blob")
    if not path.is_file():
        raise ValueError("model-file must be a regular GGUF weight blob")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        header = handle.read(24)
        if len(header) != 24:
            raise ValueError("model-file is not a GGUF weight blob")
        magic, version, tensors, _ = struct.unpack("<4sIQQ", header)
        if magic != b"GGUF" or version not in (2, 3) or tensors <= 0 or before.st_size <= 24:
            raise ValueError("model-file is not a GGUF weight blob with tensors")
        digest.update(header)
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if identity(before) != identity(after):
        raise ValueError("model weight blob changed while hashing")
    return digest.hexdigest()


def _validate_show(content: bytes, weight_sha: str) -> None:
    value = _json(content)
    if (type(value) is not dict or type(value.get("capabilities")) is not list
            or "vision" not in value["capabilities"]):
        raise ValueError("Ollama model must advertise vision capability")
    if type(value.get("details")) is not dict or value["details"].get("format") != "gguf":
        raise ValueError("Ollama model format must be gguf")
    modelfile = value.get("modelfile")
    if not isinstance(modelfile, str):
        raise ValueError("Ollama show must expose its FROM blob")
    from_lines = re.findall(r"^\s*FROM\s+(.+?)\s*$", modelfile, re.IGNORECASE | re.MULTILINE)
    if len(from_lines) != 1:
        raise ValueError("Ollama show must expose exactly one FROM blob")
    blob_name = from_lines[0].strip().strip('"').replace("\\", "/").rsplit("/", 1)[-1]
    match = re.fullmatch(r"sha256[-:]([0-9a-f]{64})", blob_name)
    if match is None or match.group(1) != weight_sha:
        raise ValueError("model-file SHA does not match Ollama FROM weight blob")


def _box_reply(raw: bytes, *, model: str, width: int, height: int) -> tuple[list[float] | None, str]:
    body = _json(raw)
    if (
        not isinstance(body, dict) or body.get("model") != model
        or body.get("done") is not True or body.get("done_reason") != "stop"
    ):
        raise ValueError("incomplete_or_wrong_model_response")
    message = body.get("message")
    if not isinstance(message, dict):
        raise ValueError("missing_response_message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        # Same schema-only compatibility shape as the existing teacher client.
        content = message.get("thinking")
    if not isinstance(content, str):
        raise ValueError("missing_response_content")
    decision = _json(content)
    if type(decision) is not dict or set(decision) != set(SCHEMA["required"]):
        raise ValueError("localization_response_shape_mismatch")
    status, box, confidence = decision["status"], decision["bbox_norm"], decision["confidence"]
    if status not in SCHEMA["properties"]["status"]["enum"] or type(box) is not list:
        raise ValueError("localization_response_types_invalid")
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("localization_confidence_invalid")
    if status != "single":
        if box:
            raise ValueError("non_single_response_must_not_include_bbox")
        return None, status
    if len(box) != 4 or any(
        type(item) not in (int, float) or not math.isfinite(item) or not 0 <= item <= 1000
        for item in box
    ):
        raise ValueError("normalized_bbox_invalid")
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError("normalized_bbox_geometry_invalid")
    if confidence < MINIMUM_CONFIDENCE:
        return None, "low_confidence"
    return [float(x1 * width / 1000), float(y1 * height / 1000),
            float(x2 * width / 1000), float(y2 * height / 1000)], "accepted"


def localize_queue(
    *, queue_path: Path, image_root: Path, known_audit: Path, output_dir: Path,
    provider: str, model: str, model_digest: str, model_file: Path,
    url: str = "http://127.0.0.1:11434", timeout: int = 300,
) -> dict:
    teacher._validate_teacher_url(url, allow_external_url=False)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", provider):
        raise ValueError("provider name must be a simple non-empty identifier")
    if (type(model_digest) is not str or teacher._valid_sha256(model_digest) != model_digest
            or type(model) is not str or not model.strip()):
        raise ValueError("exact model name and lowercase package digest are required")
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("timeout must be a positive integer")
    image_root = teacher._checked_path(image_root, "image root").resolve(strict=True)
    output_dir = teacher._checked_path(output_dir, "localization output")
    if output_dir.exists() or output_dir.is_relative_to(image_root):
        raise ValueError("output must be new and outside the source image root")
    if not output_dir.parent.is_dir():
        raise ValueError("output parent must already exist")
    queue_path, queue_content = capture_queue._stable_regular_bytes(queue_path, description="teacher queue")
    known_audit, known_content = capture_queue._stable_regular_bytes(known_audit, description="known audit")
    _json(known_content)  # Reject duplicate keys before the existing audit validator.
    known = teacher.load_known_audit(known_audit)
    queue = []
    seen = set()
    for line in queue_content.splitlines():
        if not line.strip():
            continue
        row = _json(line)
        if type(row) is not dict or set(row) != {"sha256", "timestamp", "image_ref", "decision"}:
            raise ValueError("teacher queue shape must contain no predictions or private IDs")
        sha = row["sha256"]
        if type(sha) is not str or teacher._valid_sha256(sha) != sha or sha in seen or sha in known:
            raise ValueError("teacher queue SHA is invalid, duplicated, or protected")
        if row["decision"] != "teacher_required":
            raise ValueError("teacher queue decision must be teacher_required")
        if capture_queue._datetime(row["timestamp"]) < capture_queue.OPERATIONAL_CAPTURE_CUTOFF_KST:
            raise ValueError("capture is before the fixed 2026-08-01 cutoff")
        seen.add(sha)
        queue.append(row)
    if teacher._observe_model_digest(url, model, timeout) != model_digest:
        raise ValueError("model package digest mismatch")
    weight_sha = _weight_sha(model_file)
    show_content = _post(url, "/api/show", {"model": model}, timeout)
    _validate_show(show_content, weight_sha)
    spec = {
        "schema_version": "ollama_raw_image_localization_spec.v1", "provider": provider,
        "model": model, "model_package_sha256": model_digest, "model_weight_sha256": weight_sha,
        "model_show_sha256": teacher._sha256_bytes(show_content),
        "coordinate_system": "normalized_0_1000_xyxy_to_absolute_pixels",
        "conversion": "x_px=x_norm*source_width/1000;y_px=y_norm*source_height/1000;no_clamping",
        "minimum_confidence": MINIMUM_CONFIDENCE, "prompt": PROMPT, "response_schema": SCHEMA,
        "request": {"stream": False, "think": False, "keep_alive": "10m", "options": OPTIONS},
        "deployed_prediction_used": False, "material_hints_used": False,
    }
    spec_content = _json_bytes(spec)
    spec_sha = teacher._sha256_bytes(spec_content)
    output_dir.mkdir(mode=0o700)
    rows, decisions, source_bindings = [], [], {}
    try:
        teacher._immutable_write(output_dir / FILES["spec"], spec_content.decode())
        teacher._immutable_write(output_dir / FILES["show"], show_content.decode("utf-8"))
        with (output_dir / FILES["raw"]).open("xb") as raw_handle:
            for entry in sorted(queue, key=lambda item: item["sha256"]):
                sha = entry["sha256"]
                try:
                    if type(entry["image_ref"]) is not str:
                        raise ValueError("image_ref must be a relative string")
                    teacher._checked_path(image_root / entry["image_ref"], "source image")
                    image_path, _ = teacher._resolve_image_ref(image_root, entry["image_ref"])
                    quality, resolved, image = capture_queue._image_quality_assessment(image_path, sha)
                    if resolved is not None and image is not None:
                        source_bindings[resolved] = teacher._sha256_bytes(image)
                    if quality:
                        decisions.append({"source_image_sha256": sha, "accepted": False, "reason": ";".join(quality)})
                        print(json.dumps({"processed": len(decisions), "queued": len(queue), "accepted": len(rows), "reason": "capture_quality_excluded"}), flush=True)
                        continue
                except (OSError, ValueError) as error:
                    decisions.append({"source_image_sha256": sha, "accepted": False, "reason": type(error).__name__})
                    print(json.dumps({"processed": len(decisions), "queued": len(queue), "accepted": len(rows), "reason": "capture_unavailable"}), flush=True)
                    continue
                assert image is not None
                decoded = capture_queue.cv2.imdecode(capture_queue.np.frombuffer(image, dtype=capture_queue.np.uint8), capture_queue.cv2.IMREAD_COLOR)
                height, width = decoded.shape[:2]
                if teacher._observe_model_digest(url, model, timeout) != model_digest:
                    raise ValueError("model package digest changed before localization")
                payload = {"model": model, "messages": [{"role": "user", "content": PROMPT,
                    "images": [base64.b64encode(image).decode("ascii")]}], "format": SCHEMA, **spec["request"]}
                raw = b""
                transport_error = None
                try:
                    raw = _post(url, "/api/chat", payload, timeout)
                    box, reason = _box_reply(raw, model=model, width=width, height=height)
                except OSError as error:
                    transport_error = error
                    if isinstance(error, urllib.error.HTTPError):
                        raw = error.read()
                    box, reason = None, "provider_transport_failed"
                except (ValueError, TypeError, KeyError) as error:
                    box, reason = None, f"invalid_reply:{type(error).__name__}"
                raw_handle.write(_json_bytes({"source_image_sha256": sha,
                    "model_weight_sha256": weight_sha, "inference_spec_sha256": spec_sha,
                    "raw_response_sha256": teacher._sha256_bytes(raw),
                    "raw_response_b64": base64.b64encode(raw).decode("ascii")}))
                raw_handle.flush()
                if transport_error is not None:
                    raise RuntimeError("provider HTTP or transport failed; raw reply retained") from transport_error
                if teacher._observe_model_digest(url, model, timeout) != model_digest:
                    raise ValueError("model package digest changed during localization")
                if teacher._sha256_bytes(image_path.read_bytes()) != sha:
                    raise ValueError("source image changed during localization")
                decisions.append({"source_image_sha256": sha, "accepted": box is not None, "reason": reason})
                if box is not None:
                    core = provider_output_core(provider=provider, source_sha=sha, box=box, model_sha=weight_sha, spec_sha=spec_sha)
                    rows.append(dict(core, provider_output_sha256=teacher._sha256_bytes(canonical_json(core).encode())))
                print(json.dumps({"processed": len(decisions), "queued": len(queue), "accepted": len(rows), "reason": reason}), flush=True)
        if teacher._observe_model_digest(url, model, timeout) != model_digest:
            raise ValueError("model package digest changed before publication")
        current_show = _post(url, "/api/show", {"model": model}, timeout)
        if (_weight_sha(model_file) != weight_sha
                or canonical_json(_json(current_show)) != canonical_json(_json(show_content))):
            raise ValueError("model weights or show evidence changed before publication")
        for path, expected in source_bindings.items():
            if teacher._sha256_bytes(capture_queue._stable_regular_bytes(path, description="source final rehash")[1]) != expected:
                raise ValueError("source image changed before publication")
        for path, expected in ((queue_path, queue_content), (known_audit, known_content)):
            if capture_queue._stable_regular_bytes(path, description="input final rehash")[1] != expected:
                raise ValueError("queue or known audit changed before publication")
        manifest_content = b"".join(_json_bytes(row) for row in rows)
        teacher._immutable_write(output_dir / FILES["manifest"], manifest_content.decode())
        receipt = {
            "schema_version": "ollama_raw_image_localization_receipt.v1", "provider": provider,
            "status": "provider_evidence_ready", "queued": len(queue), "accepted": len(rows),
            "rejected": len(queue) - len(rows), "reason_counts": dict(sorted(Counter(row["reason"] for row in decisions).items())),
            "decisions": decisions, "queue_sha256": teacher._sha256_bytes(queue_content),
            "known_audit_sha256": teacher._sha256_bytes(known_content),
            "model_package_sha256": model_digest, "model_weight_sha256": weight_sha,
            "model_show_sha256": teacher._sha256_bytes(show_content), "inference_spec_sha256": spec_sha,
            "manifest_sha256": teacher._sha256_bytes(manifest_content),
            "raw_replies_sha256": teacher._sha256_bytes((output_dir / FILES["raw"]).read_bytes()),
            "deployed_prediction_used": False, "executed_code_cryptographically_attested": False,
            "authority": {name: False for name in ("training", "calibration", "blind_test", "deployment")},
        }
        teacher._immutable_write(output_dir / FILES["receipt"], _json_bytes(receipt).decode())
        marker = "".join(f"{teacher._sha256_bytes((output_dir / name).read_bytes())}  {name}\n"
                         for name in sorted(value for key, value in FILES.items() if key != "marker"))
        teacher._immutable_write(output_dir / FILES["marker"], marker)
        return receipt
    except BaseException as error:
        teacher._immutable_write(output_dir / "failed.json", _json_bytes({"error_type": type(error).__name__, "provider_evidence_ready": False}).decode())
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queue", "image-root", "known-audit", "output-dir", "model-file"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("provider", "model", "model-digest"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    summary = localize_queue(queue_path=args.queue, image_root=args.image_root,
        known_audit=args.known_audit, output_dir=args.output_dir, provider=args.provider,
        model=args.model, model_digest=args.model_digest, model_file=args.model_file,
        url=args.url, timeout=args.timeout)
    print(json.dumps({key: summary[key] for key in ("provider", "queued", "accepted", "rejected", "reason_counts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
