"""Read-only aggregation for the operations monitor."""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


_CAPTURE_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _safe_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _image_path(root: Path, metadata: dict[str, Any]) -> Path | None:
    relative = metadata.get("image", {}).get("path")
    if not isinstance(relative, str):
        return None
    root_resolved = root.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root_resolved) or not candidate.is_file():
        return None
    return candidate


def _image_size(path: Path | None) -> tuple[int, int] | None:
    if path is None:
        return None
    try:
        with Image.open(path) as image:
            return image.size
    except (OSError, ValueError):
        return None


def _bbox_quality(
    bbox: list[float] | None, image_size: tuple[int, int] | None
) -> dict[str, Any] | None:
    if not bbox or len(bbox) != 4 or not image_size:
        return None
    width, height = image_size
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = (float(value) for value in bbox)
    area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height)
    edge_touches = sum(
        (
            x1 <= width * 0.01,
            y1 <= height * 0.01,
            x2 >= width * 0.99,
            y2 >= height * 0.99,
        )
    )
    return {
        "area_ratio": round(area_ratio, 4),
        "edge_touches": edge_touches,
        "suspicious": area_ratio >= 0.5 or edge_touches >= 2,
    }


def _load_audit(path: Path | None) -> dict[str, list[tuple[datetime, dict[str, Any]]]]:
    audit: dict[str, list[tuple[datetime, dict[str, Any]]]] = defaultdict(list)
    if path is None or not path.is_file():
        return audit
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return audit
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        client_id = value.get("client_id")
        timestamp = _parse_timestamp(value.get("timestamp"))
        if isinstance(client_id, str) and timestamp is not None:
            audit[client_id].append((timestamp, value))
    return audit


def _nearest_audit(
    audit: dict[str, list[tuple[datetime, dict[str, Any]]]],
    client_id: str | None,
    timestamp: datetime | None,
) -> dict[str, Any] | None:
    if not client_id or timestamp is None:
        return None
    candidates = audit.get(client_id, [])
    if not candidates:
        return None
    nearest_time, nearest = min(
        candidates, key=lambda item: abs((item[0] - timestamp).total_seconds())
    )
    if abs((nearest_time - timestamp).total_seconds()) > 120:
        return None
    return nearest


def _percentile_nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 1)


def build_monitor_snapshot(
    capture_root: Path,
    *,
    limit: int = 100,
    audit_log_path: Path | None = None,
) -> dict[str, Any]:
    """Aggregate recent captures without changing inference or review files."""
    root = capture_root.resolve()
    audit = _load_audit(audit_log_path)
    captures: list[dict[str, Any]] = []

    if root.is_dir():
        metadata_paths = sorted(
            root.rglob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True
        )[: max(1, min(limit, 500))]
    else:
        metadata_paths = []

    for metadata_path in metadata_paths:
        metadata = _safe_json(metadata_path)
        if metadata is None:
            continue
        capture_id = metadata.get("capture_id") or metadata_path.stem
        if not isinstance(capture_id, str) or not _CAPTURE_ID.fullmatch(capture_id):
            continue
        result = metadata.get("result") or {}
        request = metadata.get("request") or {}
        image_path = _image_path(root, metadata)
        size = _image_size(image_path)
        timestamp_text = metadata.get("timestamp")
        timestamp = _parse_timestamp(timestamp_text)
        audit_value = _nearest_audit(audit, request.get("client_id"), timestamp)
        bbox = result.get("bbox")
        quality = _bbox_quality(bbox, size)
        classification = result.get("classification") or None
        guidance = result.get("guidance") or []
        captures.append(
            {
                "capture_id": capture_id,
                "timestamp": timestamp_text,
                "client_id": request.get("client_id"),
                "weight_g": request.get("weight_g"),
                "status": result.get("status"),
                "classification": classification,
                "conditions": result.get("conditions") or {},
                "guidance": guidance,
                "rejection": result.get("rejection"),
                "general": result.get("general"),
                "bbox": bbox,
                "bbox_quality": quality,
                "image": {
                    "available": image_path is not None,
                    "width": size[0] if size else None,
                    "height": size[1] if size else None,
                    "url": f"/api/v1/monitor/captures/{capture_id}/image",
                },
                "process_ms": (metadata.get("metrics") or {}).get("process_ms"),
                "review": metadata.get("review") or {},
                "audit": {
                    "yolo": audit_value.get("yolo") if audit_value else None,
                    "local_llm": audit_value.get("local_llm") if audit_value else None,
                    "mode": audit_value.get("mode") if audit_value else None,
                    "reason": audit_value.get("reason") if audit_value else None,
                },
            }
        )

    total = len(captures)
    with_bbox = sum(bool(item["bbox"]) for item in captures)
    not_detected = sum(item["status"] == "NOT_DETECTED" for item in captures)
    suspicious = sum(
        bool(item["bbox_quality"] and item["bbox_quality"]["suspicious"])
        for item in captures
    )
    latencies = [
        float(item["process_ms"])
        for item in captures
        if isinstance(item["process_ms"], (int, float))
    ]
    reviewed = [
        item["review"].get("is_correct")
        for item in captures
        if isinstance(item["review"].get("is_correct"), bool)
    ]
    class_counts = Counter(
        item["classification"]["class_name"]
        for item in captures
        if item["classification"] and item["classification"].get("class_name")
    )
    status_counts = Counter(item["status"] for item in captures if item["status"])
    guidance_counts = Counter(
        entry.get("code")
        for item in captures
        for entry in item["guidance"]
        if entry.get("code")
    )

    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "window": {"requested_limit": limit, "returned": total},
        "summary": {
            "total_requests": total,
            "bbox_detection_rate": round(with_bbox / total * 100, 1) if total else None,
            "not_detected_count": not_detected,
            "suspicious_bbox_count": suspicious,
            "suspicious_bbox_rate": round(suspicious / total * 100, 1) if total else None,
            "latency_ms": {
                "samples": len(latencies),
                "median": round(statistics.median(latencies), 1) if latencies else None,
                "p95": _percentile_nearest_rank(latencies, 0.95),
            },
            "reviewed_accuracy": {
                "samples": len(reviewed),
                "percent": round(sum(reviewed) / len(reviewed) * 100, 1)
                if reviewed
                else None,
            },
        },
        "distributions": {
            "status": dict(status_counts),
            "class": dict(class_counts),
            "guidance": dict(guidance_counts),
        },
        "captures": captures,
    }


def find_capture_image(capture_root: Path, capture_id: str) -> Path | None:
    """Resolve an indexed capture image while rejecting traversal and wildcards."""
    if not _CAPTURE_ID.fullmatch(capture_id):
        return None
    root = capture_root.resolve()
    if not root.is_dir():
        return None
    for metadata_path in root.rglob(f"{capture_id}.json"):
        metadata = _safe_json(metadata_path)
        if metadata and metadata.get("capture_id", metadata_path.stem) == capture_id:
            return _image_path(root, metadata)
    return None
