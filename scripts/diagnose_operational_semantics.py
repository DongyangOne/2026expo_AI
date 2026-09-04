"""Six raw-image-only semantic checks; never labels, training, or release authority.

Read a sealed source-evidence bundle and compare two installed local vision
models. No existing answer, detector output, expected label, or other photograph
is sent to either model. Existing semantic holds are never changed.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from scripts import build_operational_source_evidence as adapter
    from scripts import label_operational_captures_ollama as ollama
    from scripts.operational_teacher_contract import MATERIALS as TEACHER_MATERIALS, QUALITY_REASONS
except ModuleNotFoundError:
    import build_operational_source_evidence as adapter
    import label_operational_captures_ollama as ollama
    from operational_teacher_contract import MATERIALS as TEACHER_MATERIALS, QUALITY_REASONS


MODELS = ("qwen3-vl:8b", "qwen3.5:9b-q4_K_M")
URL = "http://127.0.0.1:11436"
MATERIALS = tuple(TEACHER_MATERIALS[:9]) + ("unknown",)
TRISTATES = ("unknown", "no", "yes")
CUES = (
    "metal_rim", "pull_tab", "metallic_surface", "bottle_neck",
    "transparent_rigid_body", "thin_flexible_film", "cellulose_fiber",
    "paper_edge", "foam_beads", "battery_terminals", "fluorescent_tube",
    "printed_graphics", "removable_label", "different_material_attachment",
    "same_material_accessory", "hand_contact", "background_object", "opaque_body",
)
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "material": {"type": "string", "enum": list(MATERIALS)},
        "target_bbox_xyxy": {"anyOf": [
            {"type": "null"}, {"type": "array", "minItems": 4, "maxItems": 4,
                              "items": {"type": "number", "minimum": 0, "maximum": 1000}},
        ]},
        "target_identifiable": {"type": "boolean"},
        "visible_cues": {"type": "array", "uniqueItems": True,
                         "items": {"type": "string", "enum": list(CUES)}},
        "foreign_material": {"type": "string", "enum": list(TRISTATES)},
        "label": {"type": "string", "enum": list(TRISTATES)},
        "quality": {"type": "string", "enum": list(QUALITY_REASONS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}
SCHEMA["required"] = list(SCHEMA["properties"])
OPTIONS = {"temperature": 0, "seed": 20260904, "num_ctx": 8192, "num_predict": 768}
PROMPT = """Inspect only this raw recycling-kiosk photograph. Identify the one primary
discarded item being presented in the foreground for insertion. Hands and arms
are not discarded items. Bin walls, furniture, fixed background objects, shadows,
and printed pictures of objects are not the foreground item. A recognizable
crushed/deformed item is still an item. Do not choose an object just because it is
largest, brightest, central, or has familiar printing. If multiple separate items
or occlusion prevent choosing one target reliably, do not invent a target.

Use the complete project taxonomy based on the actual foreground item's material:
can = metal beverage/food/aerosol can, including crushed metal cans;
pet = PET drink bottle, including a crushed bottle;
paper = paper, cardboard, carton, or a predominantly paper cup/container;
plastic = rigid or semi-rigid non-PET plastic item/container, including a plastic
takeaway cup; styrofoam = expanded foamed plastic with a foam/bead structure;
vinyl = thin flexible plastic film, bag, or wrapper, not a rigid plastic cup;
glass = glass item/container; battery = battery/cell;
fluorescent = fluorescent lamp/tube. If none fits or visual evidence is insufficient,
use unknown. Printed graphics alone do not make metal/plastic into paper. Do not
invent certainty about PET or glass solely from transparency.

Give a tight bounding box of the primary item, not the hand/background, as
target_bbox_xyxy=[left,top,right,bottom]. Coordinates are normalized to the whole
original image: left/top edge 0, right/bottom edge 1000. Use null when the target
cannot be identified; then target_identifiable=false, material=unknown, and both
foreign_material and label=unknown. Do not report a crop from a different object.

foreign_material concerns visible attached/mixed matter of a DIFFERENT material
from this target. A straw/cap/pull ring made of the same broad material is allowed
and not foreign material. A paper sleeve on a plastic takeaway cup is different
material. label means a visibly removable attached label, not ink printed directly
on the target. A removable product sleeve/sticker is handled by label; do not
count the product label itself as foreign material. A paper cup sleeve/band on
plastic is foreign material. Set both label=yes and foreign_material=yes only
when a removable product label and separate true foreign matter coexist.
Use yes/no/unknown; inability to see evidence is not proof of no.
visible_cues must contain only the listed enum codes, not text, brands, identities,
or reasoning. These cues are your visual claims, not independently verified facts.

quality=usable unless a severe frame crop removes essential target evidence,
person occlusion/dominance prevents reading its boundary, clutter/multiple objects
prevent selecting a target, or its boundary is unreadable; use the corresponding
schema enum. Crumpling/dents alone and light non-occluding hand contact are not
bad capture quality. confidence is 0..1 for the complete judgement. Return exactly
one JSON object satisfying the schema, without explanation or reasoning.
"""
PROMPT += "\nRequired JSON schema:\n" + json.dumps(SCHEMA, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
AUTHORITY = {name: False for name in (
    "ground_truth", "training", "calibration", "blind_test", "deployment",
    "semantic_hold_release", "automatic_relabeling", "majority_vote",
)}
FILES = {"contract": "diagnostic_contract.json", "requests": "diagnostic_requests.jsonl",
         "summary": "diagnostic_summary.json", "failure": "failed.json"}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _unique_json(content: str | bytes):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("duplicate JSON key")
            result[name] = value
        return result
    return json.loads(content, object_pairs_hook=unique)


def _digest(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected exact lowercase SHA-256")
    return value


def _request_json(url: str, endpoint: str, payload: dict | None = None, *, timeout: int):
    request = urllib.request.Request(
        url + endpoint, data=None if payload is None else _json_bytes(payload),
        headers={"Content-Type": "application/json"}, method="GET" if payload is None else "POST",
    )
    # Existing helper explicitly disables redirects; credentials/base64 may not
    # escape the fixed loopback endpoint through an HTTP redirect.
    with ollama._open_no_redirect(request, timeout) as response:
        return _unique_json(response.read())


def _check_models(url: str, expected: dict[str, str], timeout: int) -> dict:
    tags = _request_json(url, "/api/tags", timeout=timeout)
    if type(tags) is not dict or type(tags.get("models")) is not list:
        raise ValueError("invalid model inventory")
    result = {}
    for name in MODELS:
        matches = [entry for entry in tags["models"]
                   if type(entry) is dict and entry.get("name") == name]
        if len(matches) != 1 or matches[0].get("digest") != expected[name]:
            raise ValueError("installed model digest mismatch")
        show = _request_json(url, "/api/show", {"model": name}, timeout=timeout)
        if type(show) is not dict or type(show.get("capabilities")) is not list or "vision" not in show["capabilities"]:
            raise ValueError("model must advertise vision capability")
        result[name] = {"digest": expected[name], "vision": True,
                        "show_sha256": _sha(_json_bytes(show))}
    return result


def _validate_decision(value: object) -> dict:
    if type(value) is not dict or set(value) != set(SCHEMA["required"]):
        raise ValueError("invalid semantic response fields")
    for name, choices in (("material", MATERIALS), ("foreign_material", TRISTATES),
                          ("label", TRISTATES), ("quality", QUALITY_REASONS)):
        if type(value[name]) is not str or value[name] not in choices:
            raise ValueError("invalid semantic response enum")
    if type(value["target_identifiable"]) is not bool:
        raise ValueError("invalid target-identifiable type")
    cues = value["visible_cues"]
    if (type(cues) is not list or any(type(cue) is not str or cue not in CUES for cue in cues)
            or len(set(cues)) != len(cues)):
        raise ValueError("invalid visible cue enum list")
    confidence = value["confidence"]
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("invalid semantic confidence")
    box = value["target_bbox_xyxy"]
    if value["target_identifiable"]:
        if (type(box) is not list or len(box) != 4
                or any(type(n) not in (int, float) or not math.isfinite(n) or not 0 <= n <= 1000 for n in box)
                or not (box[0] < box[2] and box[1] < box[3])):
            raise ValueError("invalid normalized target bbox")
    elif box is not None or value["material"] != "unknown" or value["foreign_material"] != "unknown" or value["label"] != "unknown":
        raise ValueError("unidentified target must remain unknown")
    return value


def _parse_response(response: object, model: str) -> tuple[dict, dict]:
    if (type(response) is not dict or response.get("model") != model
            or response.get("done") is not True or response.get("done_reason") != "stop"):
        raise ValueError("incomplete or wrong-model response")
    message = response.get("message")
    if type(message) is not dict:
        raise ValueError("missing semantic response")
    content = message.get("content")
    field = "content"
    if not isinstance(content, str) or not content.strip():
        content, field = message.get("thinking"), "thinking_json_only"
    # json.loads accepts exactly one complete JSON value, not a substring from
    # prose, fences, or hidden chain-of-thought. Never store either raw field.
    if not isinstance(content, str):
        raise ValueError("missing structured semantic response")
    decision = _validate_decision(_unique_json(content))
    usage = {}
    for name in ("prompt_eval_count", "eval_count", "total_duration", "load_duration",
                 "prompt_eval_duration", "eval_duration"):
        number = response.get(name)
        if type(number) is not int or number < 0:
            raise ValueError("missing or invalid token/timing counters")
        usage[name] = number
    return decision, {"done": True, "done_reason": "stop", "structured_response_field": field, **usage}


def _safe_response_diagnostics(response: object) -> dict:
    """Safe failure evidence: no message, thinking, model text, or HTTP body."""
    result = {}
    try:
        result["response_sha256"] = _sha(_json_bytes(response))
    except (TypeError, ValueError):
        result["response_sha256"] = None
    if type(response) is dict:
        result["done"] = response.get("done") if type(response.get("done")) is bool else None
        reason = response.get("done_reason")
        result["done_reason"] = reason if type(reason) is str and reason in {"stop", "length", "load", "unload"} else "unknown"
        for name in ("prompt_eval_count", "eval_count", "total_duration", "load_duration",
                     "prompt_eval_duration", "eval_duration"):
            number = response.get(name)
            if type(number) is int and number >= 0:
                result[name] = number
    return result


def _overlap(first: list | None, second: list | None) -> float | None:
    if first is None or second is None:
        return None
    intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(0, min(first[3], second[3]) - max(first[1], second[1]))
    union = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - intersection
    return intersection / union


def _write_exclusive(path: Path, value: object) -> bytes:
    content = _json_bytes(value)
    with path.open("xb") as stream:
        stream.write(content)
    return content


def _bundle_snapshot(root: Path) -> dict[str, str | None]:
    root = adapter.assembler._stable_directory(root, description="diagnostic source bundle")
    snapshot = {}
    for path in sorted(root.rglob("*")):
        adapter.assembler._reject_symlink_components(path, description="diagnostic bundle member")
        if path.is_dir():
            snapshot[path.relative_to(root).as_posix()] = None
        else:
            _, sha = adapter.assembler._stable_file_sha256(path, description="diagnostic bundle member")
            snapshot[path.relative_to(root).as_posix()] = sha
    return snapshot


def diagnose_semantics(
    *, source_bundle_dir: Path, source_sha256s: list[str], expected_models: dict[str, str],
    output_dir: Path, url: str = URL, timeout: int = 300, semantic_hold_file: Path | None = None,
) -> dict:
    if url != URL:
        raise ValueError("only http://127.0.0.1:11436 is permitted")
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("timeout must be a positive integer")
    if set(expected_models) != set(MODELS):
        raise ValueError("exactly the two diagnostic model names are required")
    expected_models = {name: _digest(expected_models[name]) for name in MODELS}
    if len(set(expected_models.values())) != 2:
        raise ValueError("diagnostic model digests must differ")
    selected = [_digest(sha) for sha in source_sha256s]
    if len(selected) != 3 or len(set(selected)) != 3:
        raise ValueError("exactly three distinct full source SHA values are required")
    adapter.assembler._reject_symlink_components(output_dir, description="diagnostic output")
    if output_dir.exists():
        raise FileExistsError("diagnostic output must be fresh")
    bundle_snapshot = _bundle_snapshot(source_bundle_dir)
    rows = adapter.validate_source_evidence_bundle(source_bundle_dir)
    if _bundle_snapshot(source_bundle_dir) != bundle_snapshot:
        raise ValueError("source bundle changed while validating")
    by_sha = {row["source_sha256"]: row for row in rows}
    if any(sha not in by_sha for sha in selected):
        raise ValueError("requested source SHA is absent from validated bundle")
    chosen = [by_sha[sha] for sha in selected]
    receipt_path = source_bundle_dir / adapter.FILES["receipt"]
    _, receipt_content = adapter.assembler._stable_regular_file(receipt_path, description="diagnostic source receipt")
    receipt = _unique_json(receipt_content)
    if _sha(receipt_content) != bundle_snapshot[adapter.FILES["receipt"]]:
        raise ValueError("source receipt changed while selecting diagnostic inputs")
    protected = [source_bundle_dir, Path(receipt["image_root"]),
                 *(Path(binding["path"]).parent for name, binding in receipt["inputs"].items()
                   if name.startswith(("teacher_output_", "quality_")))]
    if any(output_dir.resolve(strict=False).is_relative_to(path.resolve(strict=True)) for path in protected):
        raise ValueError("diagnostic output cannot be nested in source evidence")
    index_path = source_bundle_dir / adapter.FILES["index"]
    _, index_sha = adapter.assembler._stable_file_sha256(index_path, description="source index")
    if index_sha != bundle_snapshot[adapter.FILES["index"]]:
        raise ValueError("source index changed while selecting diagnostic inputs")
    _, runner_sha = adapter.assembler._stable_file_sha256(Path(__file__), description="diagnostic runner")
    hold_sha = None
    if semantic_hold_file is not None:
        _, hold_sha = adapter.assembler._stable_file_sha256(semantic_hold_file, description="existing semantic hold")
    contract = {"schema_version": "operational_semantics_diagnostic.v1", "prompt": PROMPT,
                "response_schema": SCHEMA, "options": OPTIONS, "endpoint": "/api/chat",
                "stream": False, "think": False, "keep_alive": "0s",
                "image_transform": "original_bytes_no_crop_or_resize",
                "bbox_coordinates": "xyxy_normalized_0_to_1000_full_original_image",
                "authority": AUTHORITY, "runner_sha256": runner_sha}
    contract_sha = _sha(_json_bytes(contract))
    output_dir.mkdir(parents=True, exist_ok=False)
    owned_stat = output_dir.stat()
    identity = (owned_stat.st_dev, owned_stat.st_ino)
    stage, completed = "preflight", []
    last_chat_response = None
    preserved = {}
    try:
        preserved[FILES["contract"]] = _write_exclusive(output_dir / FILES["contract"], contract)
        before_models = _check_models(url, expected_models, timeout)
        with (output_dir / FILES["requests"]).open("xb") as stream:
            for model in MODELS:
                for sha in selected:
                    stage = "inference"
                    path = Path(by_sha[sha]["source_filepath"])
                    _, image = adapter.assembler._stable_regular_file(path, description="diagnostic source image")
                    if _sha(image) != sha:
                        raise ValueError("source image SHA mismatch")
                    payload = {"model": model, "messages": [{"role": "user", "content": PROMPT,
                                "images": [base64.b64encode(image).decode("ascii")]}],
                               "format": SCHEMA, "stream": False, "think": False,
                               "keep_alive": "0s", "options": OPTIONS}
                    started = time.monotonic()
                    response = _request_json(url, "/api/chat", payload, timeout=timeout)
                    last_chat_response = _safe_response_diagnostics(response)
                    decision, metrics = _parse_response(response, model)
                    _, current_sha = adapter.assembler._stable_file_sha256(path, description="diagnostic source post-request")
                    if current_sha != sha:
                        raise ValueError("source changed during diagnostic request")
                    entry = {"source_sha256": sha, "request_image_sha256": sha,
                             "prompt_contract_sha256": contract_sha, "model": model,
                             "model_digest": expected_models[model], "decision": decision,
                             "response": metrics, "wall_seconds": time.monotonic() - started,
                             "authority": AUTHORITY}
                    stream.write(_json_bytes(entry))
                    stream.flush()
                    completed.append(entry)
                    print(f"semantic diagnostic {len(completed)}/6 complete ({model})", flush=True)
        preserved[FILES["requests"]] = b"".join(_json_bytes(row) for row in completed)
        stage = "postflight"
        after_models = _check_models(url, expected_models, timeout)
        if before_models != after_models:
            raise ValueError("diagnostic model metadata changed")
        if (adapter.validate_source_evidence_bundle(source_bundle_dir) != rows
                or _bundle_snapshot(source_bundle_dir) != bundle_snapshot):
            raise ValueError("source evidence changed during diagnostic")
        _, final_index_sha = adapter.assembler._stable_file_sha256(index_path, description="source index postflight")
        if final_index_sha != index_sha:
            raise ValueError("source index changed during diagnostic")
        if semantic_hold_file is not None:
            _, final_hold_sha = adapter.assembler._stable_file_sha256(semantic_hold_file, description="semantic hold postflight")
            if final_hold_sha != hold_sha:
                raise ValueError("semantic hold changed during diagnostic")
        agreements = []
        for sha in selected:
            first, second = [row["decision"] for row in completed if row["source_sha256"] == sha]
            agreements.append({"source_sha256": sha,
                "material_agreement": first["material"] == second["material"],
                "target_identifiability_agreement": first["target_identifiable"] == second["target_identifiable"],
                "target_bbox_iou": _overlap(first["target_bbox_xyxy"], second["target_bbox_xyxy"]),
                "target_comparison": "overlap_only_no_acceptance_threshold",
                "authority": AUTHORITY})
        summary = {"schema_version": "operational_semantics_diagnostic_summary.v1",
                   "status": "diagnostic_complete", "requests_completed": len(completed),
                   "source_index_sha256": index_sha, "prompt_contract_sha256": contract_sha,
                   "models": before_models, "agreements": agreements, "authority": AUTHORITY,
                   "semantic_hold_action": "unchanged", "visual_cues_are_unverified_model_claims": True,
                   "semantic_hold_sha256": hold_sha,
                   "automatic_decision": None, "output_sha256": {name: _sha(content) for name, content in preserved.items()}}
        stage = "publication"
        preserved[FILES["summary"]] = _write_exclusive(output_dir / FILES["summary"], summary)
        for name, content in preserved.items():
            _, digest = adapter.assembler._stable_file_sha256(output_dir / name, description="diagnostic artifact post-publication")
            if digest != _sha(content):
                raise ValueError("diagnostic output changed during publication")
        _, final_runner_sha = adapter.assembler._stable_file_sha256(Path(__file__), description="diagnostic runner postflight")
        if final_runner_sha != runner_sha:
            raise ValueError("diagnostic runner changed")
        if _bundle_snapshot(source_bundle_dir) != bundle_snapshot:
            raise ValueError("source bundle changed during publication")
        for row in chosen:
            _, final_source_sha = adapter.assembler._stable_file_sha256(
                Path(row["source_filepath"]), description="diagnostic source publication")
            if final_source_sha != row["source_sha256"]:
                raise ValueError("source image changed during publication")
        if semantic_hold_file is not None:
            _, final_hold_sha = adapter.assembler._stable_file_sha256(semantic_hold_file, description="semantic hold publication")
            if final_hold_sha != hold_sha:
                raise ValueError("semantic hold changed during publication")
        return summary
    except Exception as error:
        # Error messages and HTTP bodies can contain raw responses or private
        # identifiers. Preserve only a stage/count code, never exception text.
        try:
            adapter.assembler._reject_symlink_components(output_dir, description="diagnostic failed output")
            current = output_dir.stat()
            if (current.st_dev, current.st_ino) == identity:
                _write_exclusive(output_dir / FILES["failure"], {
                    "status": "diagnostic_failed", "stage": stage,
                    "requests_completed": len(completed), "authority": AUTHORITY,
                    "semantic_hold_action": "unchanged",
                    "exception_type": ("HTTPError" if isinstance(error, urllib.error.HTTPError)
                                       else "ValueError" if isinstance(error, ValueError)
                                       else "OSError" if isinstance(error, OSError) else "unexpected_error"),
                    "http_status": error.code if isinstance(error, urllib.error.HTTPError) and type(error.code) is int else None,
                    "last_chat_response": last_chat_response,
                })
        except (OSError, ValueError):
            pass
        raise ValueError(f"semantic diagnostic failed during {stage}; outputs are not authority") from None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle-dir", type=Path, required=True)
    parser.add_argument("--source-sha256", action="append", required=True)
    parser.add_argument("--qwen3-vl-digest", required=True)
    parser.add_argument("--qwen35-digest", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--semantic-hold-file", type=Path)
    args = parser.parse_args()
    diagnose_semantics(source_bundle_dir=args.source_bundle_dir, source_sha256s=args.source_sha256,
                       expected_models=dict(zip(MODELS, (args.qwen3_vl_digest, args.qwen35_digest))),
                       output_dir=args.output_dir, timeout=args.timeout, semantic_hold_file=args.semantic_hold_file)


if __name__ == "__main__":
    main()
