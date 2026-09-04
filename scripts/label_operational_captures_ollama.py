"""Blind-label new operational captures with two structured Qwen-VL passes."""

from __future__ import annotations

import argparse
import base64
import ipaddress
import json
import math
import os
import tempfile
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

try:
    from scripts.operational_teacher_contract import (
        ADJUDICATION_PROMPT, DECISION_SCHEMA as SCHEMA, MATERIALS, PROMPTS,
        QUALITY_REASONS, REQUEST_CONTRACT, REQUEST_OPTIONS,
        TEACHER_LABEL_BASE_FIELDS, TEACHER_LABEL_SCHEMA_VERSION,
        build_teacher_contract, load_known_audit,
        canonical_json as _canonical_json, render_prompt,
        sha256_bytes as _sha256_bytes, valid_sha256 as _valid_sha256,
    )
except ModuleNotFoundError:  # direct ``python scripts/...py`` execution
    from operational_teacher_contract import (  # type: ignore[no-redef]
        ADJUDICATION_PROMPT, DECISION_SCHEMA as SCHEMA, MATERIALS, PROMPTS,
        QUALITY_REASONS, REQUEST_CONTRACT, REQUEST_OPTIONS,
        TEACHER_LABEL_BASE_FIELDS, TEACHER_LABEL_SCHEMA_VERSION,
        build_teacher_contract, load_known_audit,
        canonical_json as _canonical_json, render_prompt,
        sha256_bytes as _sha256_bytes, valid_sha256 as _valid_sha256,
    )


def _confidence(result: dict) -> float:
    value = result.get("confidence")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid confidence")
    confidence = float(value)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("invalid confidence")
    return confidence


def _decision_tuple(result: dict) -> tuple[str, bool, bool, bool, str]:
    if not isinstance(result, dict):
        raise ValueError("decision must be an object")
    if set(result) != set(SCHEMA["required"]):
        raise ValueError("decision fields must exactly match the trusted schema")
    if result.get("material") not in MATERIALS:
        raise ValueError("invalid material")
    if not isinstance(result.get("single_object"), bool):
        raise ValueError("invalid single_object")
    if not isinstance(result.get("foreign_material"), bool):
        raise ValueError("invalid foreign_material")
    if not isinstance(result.get("training_usable"), bool):
        raise ValueError("invalid training_usable")
    if result.get("quality_reason") not in QUALITY_REASONS:
        raise ValueError("invalid quality_reason")
    if result["training_usable"] != (result["quality_reason"] == "usable"):
        raise ValueError("training_usable and quality_reason disagree")
    return (
        str(result["material"]),
        result["single_object"],
        result["foreign_material"],
        result["training_usable"],
        str(result["quality_reason"]),
    )


def _consensus_summary(passes: list[dict]) -> tuple[bool, dict | None, float]:
    """Return an exact-tuple 2-vote consensus from two or three blind passes."""
    if len(passes) < 2:
        return False, None, 0.0
    for item in passes:
        _confidence(item)
    counts = Counter(_decision_tuple(item) for item in passes)
    decision, votes = counts.most_common(1)[0]
    if votes < 2:
        return False, None, 0.0
    supporting = [
        _confidence(item)
        for item in passes
        if _decision_tuple(item) == decision
    ]
    material, single_object, foreign_material, training_usable, quality_reason = decision
    return (
        True,
        {
            "material": material,
            "single_object": single_object,
            "foreign_material": foreign_material,
            "training_usable": training_usable,
            "quality_reason": quality_reason,
            "votes": votes,
            "pass_count": len(passes),
        },
        min(supporting),
    )


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError(f"teacher URL redirect refused: HTTP {code}")


def _open_no_redirect(request: urllib.request.Request, timeout: int):
    return urllib.request.build_opener(_RejectRedirects()).open(request, timeout=timeout)


def _request(url: str, model: str, image: bytes, prompt: str, timeout: int) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": render_prompt(prompt),
                "images": [base64.b64encode(image).decode("ascii")],
            }
        ],
        "format": SCHEMA,
        "stream": REQUEST_CONTRACT["stream"],
        "think": REQUEST_CONTRACT["think"],
        "keep_alive": REQUEST_CONTRACT["keep_alive"],
        "options": REQUEST_OPTIONS,
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _open_no_redirect(request, timeout) as response:
        body = json.load(response)
    if not isinstance(body, dict) or body.get("model") != model:
        raise ValueError("Ollama response model does not match requested model")
    message = body.get("message") or {}
    content = message.get("content") or ""
    # Recent Ollama/Qwen3-VL combinations can place a schema-constrained final
    # JSON object in ``message.thinking`` even with ``think=false``. Accept that
    # compatibility shape only after a normal stop and only when the entire
    # field is one JSON object. Truncated/internal reasoning remains an error.
    if not content.strip() and body.get("done_reason") == "stop":
        thinking = (message.get("thinking") or "").strip()
        if thinking.startswith("{") and thinking.endswith("}"):
            content = thinking
    if not content.strip():
        raise ValueError(
            "empty model content "
            f"(thinking_chars={len(message.get('thinking') or '')}, "
            f"done_reason={body.get('done_reason')})"
        )
    result = json.loads(content)
    _decision_tuple(result)
    _confidence(result)
    return result


def _validate_teacher_url(url: str, *, allow_external_url: bool) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("teacher URL must be an http(s) URL with a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("teacher URL must not contain credentials")
    if allow_external_url:
        return
    hostname = parsed.hostname.casefold()
    if hostname == "localhost":
        return
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError as error:
        raise ValueError(
            "teacher URL host must be loopback/private unless --allow-external-url is set"
        ) from error
    if not (address.is_loopback or address.is_private):
        raise ValueError(
            "teacher URL host must be loopback/private unless --allow-external-url is set"
        )


def _normalize_ollama_digest(value: object) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized.startswith("sha256:"):
        normalized = normalized[7:]
    return _valid_sha256(normalized)


def _observe_model_digest(url: str, model: str, timeout: int) -> str:
    request = urllib.request.Request(
        url.rstrip("/") + "/api/tags", method="GET"
    )
    with _open_no_redirect(request, timeout) as response:
        body = json.load(response)
    rows = body.get("models") if isinstance(body, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Ollama /api/tags response has no models array")
    matches = [
        row for row in rows
        if isinstance(row, dict) and (row.get("name") == model or row.get("model") == model)
    ]
    if len(matches) != 1:
        raise ValueError("requested model is missing or ambiguous in Ollama /api/tags")
    digest = _normalize_ollama_digest(matches[0].get("digest"))
    if digest is None:
        raise ValueError("Ollama model digest is invalid")
    return digest


def _resolve_image_ref(image_root: Path, value: object) -> tuple[Path, str]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("image_ref must be a non-empty relative string")
    relative = Path(value.strip())
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise ValueError("image_ref must stay relative to image_root")
    resolved = (image_root / relative).resolve(strict=False)
    try:
        portable = resolved.relative_to(image_root)
    except ValueError as error:
        raise ValueError("image_ref escapes image_root") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"image_ref does not resolve to a file: {value}")
    return resolved, portable.as_posix()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    os.replace(temporary, path)


def _checked_path(path: Path, description: str) -> Path:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{description} must not contain symlink components")
    return absolute


def _immutable_write(path: Path, content: str, *, allow_identical: bool = False) -> None:
    """Publish shared cache/output bytes without replacing another run's file."""
    path = _checked_path(path, "immutable teacher artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    _checked_path(path, "immutable teacher artifact")
    encoded = content.encode("utf-8")
    descriptor, raw = tempfile.mkstemp(prefix=".teacher-", suffix=".tmp", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            # Cache entries are content addressed; an identical concurrent
            # publication is harmless, but a replacement is never allowed.
            if not allow_identical:
                raise
            _checked_path(path, "immutable teacher artifact")
            if not path.is_file() or path.read_bytes() != encoded:
                raise ValueError("immutable teacher artifact already exists with different bytes")
    finally:
        temporary.unlink(missing_ok=True)


def _shared_checkpoint_candidates(directory: Path) -> list[dict]:
    directory = _checked_path(directory, "shared checkpoint directory")
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise ValueError("shared checkpoint directory must be a directory")
    rows = []
    for path in sorted(directory.iterdir()):
        _checked_path(path, "shared checkpoint file")
        # A concurrent atomic publication can have an incomplete private file.
        if path.name.startswith(".teacher-") and path.name.endswith(".tmp"):
            continue
        if path.suffix != ".json" or _valid_sha256(path.stem) != path.stem or not path.is_file():
            raise ValueError("unexpected shared checkpoint file")
        content = path.read_bytes()
        if _sha256_bytes(content) != path.stem:
            raise ValueError("shared checkpoint content digest mismatch")
        try:
            row = json.loads(content)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("shared checkpoint is not valid JSON") from error
        if not isinstance(row, dict):
            raise ValueError("shared checkpoint must be an object")
        rows.append(row)
    return rows


def _save_shared_checkpoint(directory: Path, row: dict) -> None:
    content = _canonical_json(row) + "\n"
    digest = _sha256_bytes(content.encode("utf-8"))
    _immutable_write(directory / f"{digest}.json", content, allow_identical=True)


def _checkpoint_path(checkpoint_dir: Path, sha256: str) -> Path:
    return checkpoint_dir / f"{sha256}.json"


def _load_checkpoint(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _current_completed_label(
    row: dict,
    *,
    teacher_contract: dict,
    teacher_contract_sha256: str,
    input_image_sha256: str,
    image_ref: str,
) -> bool:
    """Accept a checkpoint only when every derived field recomputes exactly."""
    if set(row) != TEACHER_LABEL_BASE_FIELDS:
        return False
    if row.get("schema_version") != TEACHER_LABEL_SCHEMA_VERSION:
        return False
    if row.get("teacher_contract") != teacher_contract:
        return False
    if row.get("teacher_contract_sha256") != teacher_contract_sha256:
        return False
    if row.get("model") != teacher_contract.get("model_identifier"):
        return False
    if row.get("model_digest") != teacher_contract.get("model_digest"):
        return False
    if row.get("image_ref") != image_ref:
        return False
    if row.get("input_image_sha256") != input_image_sha256:
        return False
    if row.get("sha256") != input_image_sha256:
        return False
    if not isinstance(row.get("errors"), list) or row["errors"]:
        return False
    passes = row.get("passes")
    if not isinstance(passes, list) or len(passes) not in {
        len(PROMPTS), len(PROMPTS) + 1
    }:
        return False
    try:
        consensus, decision, minimum = _consensus_summary(passes)
        initial_consensus, _, _ = _consensus_summary(passes[: len(PROMPTS)])
    except (KeyError, TypeError, ValueError):
        return False
    if len(passes) == len(PROMPTS) + 1 and initial_consensus:
        return False
    if row.get("consensus") is not consensus:
        return False
    if row.get("consensus_decision") != decision:
        return False
    reported_minimum = row.get("minimum_confidence")
    if (
        isinstance(reported_minimum, bool)
        or not isinstance(reported_minimum, (int, float))
        or not math.isfinite(float(reported_minimum))
        or float(reported_minimum) != minimum
    ):
        return False
    if consensus:
        return len(passes) >= len(PROMPTS)
    return len(passes) == len(PROMPTS) + 1


def label_queue(
    queue_path: Path,
    output_path: Path,
    *,
    image_root: Path,
    known_audit: Path,
    url: str,
    model: str,
    model_digest: str,
    timeout: int,
    retries: int,
    allow_external_url: bool = False,
    checkpoint_dir: Path | None = None,
) -> dict:
    _validate_teacher_url(url, allow_external_url=allow_external_url)
    image_root = image_root.resolve(strict=True)
    if not image_root.is_dir():
        raise NotADirectoryError(f"image_root is not a directory: {image_root}")
    if not model.strip():
        raise ValueError("model must not be empty")
    if retries < 1:
        raise ValueError("retries must be at least 1")
    declared_digest = _valid_sha256(model_digest)
    if declared_digest is None:
        raise ValueError("model_digest must be exactly 64 hexadecimal characters")
    observed_digest = _observe_model_digest(url, model, timeout)
    if observed_digest != declared_digest:
        raise ValueError("declared model_digest does not match Ollama /api/tags")
    known = load_known_audit(known_audit)
    queue = [
        json.loads(line)
        for line in queue_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    normalized_queue: dict[str, dict] = {}
    for row in queue:
        if not isinstance(row, dict):
            raise ValueError("teacher queue rows must be objects")
        sha256 = _valid_sha256(row.get("sha256"))
        if sha256 is None:
            raise ValueError("teacher queue contains an invalid sha256")
        if sha256 in normalized_queue:
            raise ValueError(f"teacher queue contains duplicate sha256: {sha256}")
        if set(row) != {"sha256", "timestamp", "image_ref", "decision"}:
            raise ValueError("teacher queue row shape is not exact")
        if row.get("decision") != "teacher_required":
            raise ValueError("teacher queue decision must be teacher_required")
        if sha256 in known:
            raise ValueError("known train/validation SHA cannot enter teacher labeling")
        forbidden = {
            "client_id", "device_id", "deployed", "verifier", "image_path", "filepath"
        }.intersection(row)
        if forbidden:
            raise ValueError(
                "teacher queue contains forbidden private/prediction/path fields: "
                + ", ".join(sorted(forbidden))
            )
        normalized_queue[sha256] = dict(row, sha256=sha256)

    contract, contract_sha256 = build_teacher_contract(model, observed_digest)
    shared_cache = checkpoint_dir is not None
    if shared_cache:
        assert checkpoint_dir is not None
        cache_root = _checked_path(checkpoint_dir, "shared checkpoint directory")
        output_path = _checked_path(output_path, "teacher output")
        if output_path.is_relative_to(cache_root) or cache_root.is_relative_to(output_path):
            raise ValueError("teacher output and shared checkpoint directory must not overlap")
        if output_path.exists():
            raise FileExistsError("shared-cache labeling requires a new immutable output file")
        for source_path in (queue_path, known_audit):
            if _checked_path(source_path, "teacher input").is_relative_to(cache_root):
                raise ValueError("shared checkpoint directory must not contain teacher inputs")
        checkpoint_dir = cache_root / contract_sha256
    else:
        checkpoint_dir = output_path.parent / f"{output_path.name}.checkpoints"
    checkpoint_dir = _checked_path(checkpoint_dir, "checkpoint directory")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _checked_path(checkpoint_dir, "checkpoint directory")
    completed: dict[str, dict] = {}
    source_bindings: dict[Path, str] = {}
    # newly_labeled counts completed decisions produced now, including successful
    # retries. retried counts images with an earlier incomplete/incompatible attempt.
    counters = {"reused": 0, "newly_labeled": 0, "retried": 0, "image_error": 0}
    ordered_queue = [normalized_queue[sha] for sha in sorted(normalized_queue)]
    for index, row in enumerate(ordered_queue, start=1):
        sha256 = row["sha256"]
        checkpoint = _checkpoint_path(checkpoint_dir, sha256)
        shared_source_dir = checkpoint_dir / sha256
        if _observe_model_digest(url, model, timeout) != observed_digest:
            raise ValueError("Ollama model digest changed before image labeling")
        try:
            image_path, image_ref = _resolve_image_ref(image_root, row.get("image_ref"))
            if shared_cache and image_path.is_relative_to(cache_root):
                raise ValueError("image must not be inside shared checkpoint directory")
            image = image_path.read_bytes()
            input_image_sha256 = _sha256_bytes(image)
            if input_image_sha256 != sha256:
                raise ValueError("image_sha256_mismatch")
            source_bindings[image_path] = input_image_sha256
        except (OSError, ValueError) as error:
            result = {
                "schema_version": TEACHER_LABEL_SCHEMA_VERSION,
                "sha256": sha256,
                "image_ref": "",
                "input_image_sha256": None,
                "teacher_contract": contract,
                "teacher_contract_sha256": contract_sha256,
                "model": model,
                "model_digest": contract["model_digest"],
                "passes": [],
                "errors": [f"{type(error).__name__}: {error}"],
                "consensus": False,
                "consensus_decision": None,
                "minimum_confidence": 0.0,
            }
            if shared_cache:
                _save_shared_checkpoint(shared_source_dir, result)
            else:
                _atomic_write(checkpoint, _canonical_json(result) + "\n")
            counters["image_error"] += 1
            completed[sha256] = result
            print(f"[{index}/{len(queue)}] {sha256[:12]} image_error", flush=True)
            continue

        if shared_cache:
            previous_rows = _shared_checkpoint_candidates(shared_source_dir)
        else:
            _checked_path(checkpoint, "checkpoint file")
            previous = _load_checkpoint(checkpoint)
            previous_rows = [previous] if previous is not None else []
        # A row with two valid passes is a completed teacher decision even when
        # the passes disagree.  Transport/truncation errors must be retried;
        # treating them as completed made resumable NAS jobs permanently skip
        # the failed images.
        previous = next((candidate for candidate in previous_rows if _current_completed_label(
            candidate, teacher_contract=contract, teacher_contract_sha256=contract_sha256,
            input_image_sha256=input_image_sha256, image_ref=image_ref,
        )), None)
        if previous is not None:
            completed[sha256] = previous
            counters["reused"] += 1
            continue
        if previous_rows:
            counters["retried"] += 1
        passes = []
        errors = []
        for prompt in PROMPTS:
            for attempt in range(1, retries + 1):
                try:
                    passes.append(_request(url, model, image, prompt, timeout))
                    break
                except Exception as exc:
                    if attempt == retries:
                        errors.append(f"{type(exc).__name__}: {exc}")
                        break
                    time.sleep(attempt * 2)
            if errors:
                break
        consensus, consensus_decision, minimum_confidence = _consensus_summary(passes)
        # Fast path: two agreeing passes stop.  Only disagreements pay for a
        # third blind adjudication pass, which gives an exact-tuple majority.
        if not errors and len(passes) == len(PROMPTS) and not consensus:
            for attempt in range(1, retries + 1):
                try:
                    passes.append(
                        _request(url, model, image, ADJUDICATION_PROMPT, timeout)
                    )
                    break
                except Exception as exc:
                    if attempt == retries:
                        errors.append(f"{type(exc).__name__}: {exc}")
                        break
                    time.sleep(attempt * 2)
            consensus, consensus_decision, minimum_confidence = _consensus_summary(passes)
        result = {
            "schema_version": TEACHER_LABEL_SCHEMA_VERSION,
            "sha256": sha256,
            "image_ref": image_ref,
            "input_image_sha256": input_image_sha256,
            "teacher_contract": contract,
            "teacher_contract_sha256": contract_sha256,
            "model": model,
            "model_digest": contract["model_digest"],
            "passes": passes,
            "errors": errors,
            "consensus": consensus,
            "consensus_decision": consensus_decision,
            "minimum_confidence": minimum_confidence,
        }
        if _observe_model_digest(url, model, timeout) != observed_digest:
            raise ValueError("Ollama model digest changed during image labeling")
        if _sha256_bytes(image_path.read_bytes()) != input_image_sha256:
            raise ValueError("source image changed during teacher labeling")
        completed[sha256] = result
        if shared_cache:
            _save_shared_checkpoint(shared_source_dir, result)
        else:
            _atomic_write(checkpoint, _canonical_json(result) + "\n")
        if _current_completed_label(
            result, teacher_contract=contract, teacher_contract_sha256=contract_sha256,
            input_image_sha256=input_image_sha256, image_ref=image_ref,
        ):
            counters["newly_labeled"] += 1
        print(f"[{index}/{len(queue)}] {sha256[:12]} consensus={consensus}", flush=True)

    final_digest = _observe_model_digest(url, model, timeout)
    if final_digest != observed_digest:
        raise ValueError("Ollama model digest changed during teacher labeling")
    for source_path, expected_sha in source_bindings.items():
        if _sha256_bytes(source_path.read_bytes()) != expected_sha:
            raise ValueError("source image changed before teacher output publication")
    rows = [completed[sha] for sha in sorted(completed)]
    # The portable final artifact is emitted once, atomically, from the
    # per-image checkpoints.  A killed process can resume without either an
    # O(N^2) rewrite or duplicate/stale rows.
    output_content = "".join(_canonical_json(row) + "\n" for row in rows)
    if shared_cache:
        _immutable_write(output_path, output_content)
    else:
        _atomic_write(output_path, output_content)
    return {
        "schema_version": TEACHER_LABEL_SCHEMA_VERSION,
        "teacher_contract_sha256": contract_sha256,
        "queued": len(queue),
        "completed": len(rows),
        **counters,
        "consensus": sum(row["consensus"] for row in rows),
        "high_confidence_consensus": sum(
            row["consensus"] and row["minimum_confidence"] >= 0.8 for row in rows
        ),
        "high_confidence_training_usable_consensus": sum(
            row["consensus"]
            and row["minimum_confidence"] >= 0.8
            and (row.get("consensus_decision") or {}).get("training_usable") is True
            for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--known-audit", required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3-vl:8b")
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--allow-external-url", action="store_true")
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        help="Optional shared cache root; immutable entries are isolated by teacher contract SHA",
    )
    args = parser.parse_args()
    summary = label_queue(
        args.queue,
        args.output,
        image_root=args.image_root,
        known_audit=args.known_audit,
        url=args.url,
        model=args.model,
        model_digest=args.model_digest,
        timeout=args.timeout,
        retries=args.retries,
        allow_external_url=args.allow_external_url,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
