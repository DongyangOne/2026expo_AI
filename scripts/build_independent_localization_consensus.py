"""Build frozen two-provider localization consensus without deployed evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Sequence

try:
    from scripts.operational_teacher_contract import (
        TEACHER_LABEL_BASE_FIELDS, TEACHER_LABEL_SCHEMA_VERSION,
        build_teacher_contract, valid_sha256,
    )
except ModuleNotFoundError:
    from operational_teacher_contract import (  # type: ignore
        TEACHER_LABEL_BASE_FIELDS, TEACHER_LABEL_SCHEMA_VERSION,
        build_teacher_contract, valid_sha256,
    )


PROVIDER_SCHEMA_VERSION = "independent_localization_provider.v1"
LOCALIZATION_SCHEMA_VERSION = "independent_localization.v2"
CONTRACT_SCHEMA_VERSION = "independent_localization_consensus_contract.v1"
IOU_THRESHOLD = 0.75
AGGREGATE_METHOD = "coordinate_mean"
AGGREGATE_TOLERANCE = 1e-9


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_sha(value: object) -> str | None:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64:
        return None
    try:
        int(normalized, 16)
    except ValueError:
        return None
    return normalized


def bbox(value: object) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise ValueError("provider bbox_xyxy must contain four numeric values")
    result = [float(item) for item in value]
    x1, y1, x2, y2 = result
    if not all(math.isfinite(item) for item in result) or min(x1, y1) < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("provider bbox_xyxy has invalid geometry")
    return result


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ix1, iy1 = max(first[0], second[0]), max(first[1], second[1])
    ix2, iy2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def provider_output_core(
    *, provider: str, source_sha: str, box: list[float], model_sha: str, spec_sha: str
) -> dict:
    return {
        "schema_version": PROVIDER_SCHEMA_VERSION,
        "provider": provider,
        "source_image_sha256": source_sha,
        "bbox_xyxy": box,
        "model_sha256": model_sha,
        "inference_spec_sha256": spec_sha,
        "deployed_prediction_used": False,
    }


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"row must be an object at {path}:{number}")
        rows.append(value)
    return rows


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    os.replace(temporary, path)


def _provider_rows(
    manifest: Path, *, provider: str, model_file: Path, spec_file: Path
) -> tuple[dict[str, dict], dict]:
    if not provider.strip():
        raise ValueError("provider name must not be empty")
    model_sha, spec_sha = sha256_file(model_file), sha256_file(spec_file)
    rows = {}
    for row in _load_jsonl(manifest):
        if {"deployed", "verifier", "result"}.intersection(row):
            raise ValueError("provider manifest contains deployed prediction fields")
        source_sha = valid_sha(row.get("source_image_sha256"))
        if source_sha is None or source_sha in rows:
            raise ValueError("provider manifest source SHA is invalid or duplicated")
        box = bbox(row.get("bbox_xyxy"))
        core = provider_output_core(
            provider=provider,
            source_sha=source_sha,
            box=box,
            model_sha=model_sha,
            spec_sha=spec_sha,
        )
        if set(row) != {*core, "provider_output_sha256"}:
            raise ValueError("provider row shape is not exact")
        if any(row.get(key) != value for key, value in core.items()):
            raise ValueError("provider row does not match frozen provider files")
        output_sha = sha256_bytes(canonical_json(core).encode("utf-8"))
        if row.get("provider_output_sha256") != output_sha:
            raise ValueError("provider output hash mismatch")
        rows[source_sha] = dict(core, provider_output_sha256=output_sha)
    evidence = {
        "provider": provider,
        "manifest_sha256": sha256_file(manifest),
        "model_sha256": model_sha,
        "inference_spec_sha256": spec_sha,
    }
    return rows, evidence


def build_consensus(
    *, teacher_labels: Path, provider_a_manifest: Path, provider_a_name: str,
    provider_a_model: Path, provider_a_spec: Path, provider_b_manifest: Path,
    provider_b_name: str, provider_b_model: Path, provider_b_spec: Path,
    output: Path,
) -> dict:
    input_paths = {
        "teacher_labels": teacher_labels,
        "provider_a_manifest": provider_a_manifest,
        "provider_a_model": provider_a_model, "provider_a_spec": provider_a_spec,
        "provider_b_manifest": provider_b_manifest,
        "provider_b_model": provider_b_model, "provider_b_spec": provider_b_spec,
    }
    input_snapshot = {name: sha256_file(path) for name, path in input_paths.items()}
    if provider_a_name.strip().casefold() == provider_b_name.strip().casefold():
        raise ValueError("provider names must be distinct")
    first, first_evidence = _provider_rows(
        provider_a_manifest, provider=provider_a_name,
        model_file=provider_a_model, spec_file=provider_a_spec,
    )
    second, second_evidence = _provider_rows(
        provider_b_manifest, provider=provider_b_name,
        model_file=provider_b_model, spec_file=provider_b_spec,
    )
    if first_evidence["model_sha256"] == second_evidence["model_sha256"]:
        raise ValueError("provider model SHA values must be distinct")
    teacher_rows = _load_jsonl(teacher_labels)
    enriched = []
    accepted = 0
    for row in sorted(teacher_rows, key=lambda item: str(item.get("sha256") or "")):
        if set(row) != set(TEACHER_LABEL_BASE_FIELDS):
            raise ValueError("teacher label row shape is not exact")
        if row.get("schema_version") != TEACHER_LABEL_SCHEMA_VERSION:
            raise ValueError("teacher label schema version is not current")
        source_sha = valid_sha256(row.get("sha256"))
        model = row.get("model")
        digest = row.get("model_digest")
        if source_sha is None or not isinstance(model, str) or not model:
            raise ValueError("teacher label identity is invalid")
        expected_contract, expected_contract_sha = build_teacher_contract(model, digest)
        if (
            row.get("teacher_contract") != expected_contract
            or row.get("teacher_contract_sha256") != expected_contract_sha
            or row.get("input_image_sha256") != source_sha
        ):
            raise ValueError("teacher label trusted contract binding is invalid")
        if {"deployed", "verifier", "bbox"}.intersection(row):
            raise ValueError("teacher labels contain deployed prediction fields")
        updated = dict(row)
        if source_sha in first and source_sha in second:
            a, b = first[source_sha], second[source_sha]
            if a["provider_output_sha256"] == b["provider_output_sha256"]:
                raise ValueError("provider output evidence must be distinct")
            overlap = bbox_iou(a["bbox_xyxy"], b["bbox_xyxy"])
            if overlap >= IOU_THRESHOLD:
                aggregate = [
                    (a["bbox_xyxy"][index] + b["bbox_xyxy"][index]) / 2.0
                    for index in range(4)
                ]
                contract = {
                    "schema_version": CONTRACT_SCHEMA_VERSION,
                    "method": AGGREGATE_METHOD,
                    "iou_threshold": IOU_THRESHOLD,
                    "aggregate_tolerance": AGGREGATE_TOLERANCE,
                    "source_image_sha256": source_sha,
                    "providers": [first_evidence, second_evidence],
                }
                contract_sha = sha256_bytes(canonical_json(contract).encode("utf-8"))
                updated["independent_localization"] = {
                    "schema_version": LOCALIZATION_SCHEMA_VERSION,
                    "source_image_sha256": source_sha,
                    "bbox_xyxy": aggregate,
                    "providers": [a, b],
                    "provider_iou": overlap,
                    "contract": contract,
                    "contract_sha256": contract_sha,
                    "deployed_prediction_used": False,
                    "consensus": True,
                }
                accepted += 1
        enriched.append(updated)
    if {name: sha256_file(path) for name, path in input_paths.items()} != input_snapshot:
        raise ValueError("localization producer input changed before final publish")
    _atomic_write(output, "".join(canonical_json(row) + "\n" for row in enriched))
    return {"rows": len(enriched), "localized": accepted, "iou_threshold": IOU_THRESHOLD}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-labels", required=True, type=Path)
    for prefix in ("a", "b"):
        parser.add_argument(f"--provider-{prefix}-manifest", required=True, type=Path)
        parser.add_argument(f"--provider-{prefix}-name", required=True)
        parser.add_argument(f"--provider-{prefix}-model", required=True, type=Path)
        parser.add_argument(f"--provider-{prefix}-spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_consensus(
        teacher_labels=args.teacher_labels,
        provider_a_manifest=args.provider_a_manifest,
        provider_a_name=args.provider_a_name,
        provider_a_model=args.provider_a_model,
        provider_a_spec=args.provider_a_spec,
        provider_b_manifest=args.provider_b_manifest,
        provider_b_name=args.provider_b_name,
        provider_b_model=args.provider_b_model,
        provider_b_spec=args.provider_b_spec,
        output=args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
