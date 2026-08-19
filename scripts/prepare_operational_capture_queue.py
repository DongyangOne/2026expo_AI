"""Inventory Pi request captures and create a privacy-safe VLM teacher queue.

Existing audited train/validation assignments are retained by SHA-256.  New
images are never assigned the deployed prediction as ground truth; they are
sent to a separate teacher queue without client_id values.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


KST = timezone(timedelta(hours=9))


def _datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _nearest_shadow(
    metadata: dict, shadows_by_client: dict[str, list[dict]], max_seconds: float = 30
) -> dict | None:
    client_id = metadata.get("request", {}).get("client_id")
    timestamp = _datetime(metadata["timestamp"])
    candidates = shadows_by_client.get(client_id, [])
    if not candidates:
        return None
    best = min(candidates, key=lambda row: abs((_datetime(row["timestamp"]) - timestamp).total_seconds()))
    delta = abs((_datetime(best["timestamp"]) - timestamp).total_seconds())
    return best if delta <= max_seconds else None


def prepare_queue(
    *,
    captures_dir: Path,
    shadow_log: Path,
    known_audit: Path,
    output_dir: Path,
    start_kst: datetime,
    image_path_prefix: str | None = None,
) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    latest_by_sha: dict[str, tuple[dict, Path]] = {}
    rows_after_cutoff = 0
    for metadata_path in sorted(captures_dir.rglob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if _datetime(metadata["timestamp"]).astimezone(KST) < start_kst:
            continue
        rows_after_cutoff += 1
        sha256 = metadata["image"]["sha256"]
        previous = latest_by_sha.get(sha256)
        if previous is None or metadata["timestamp"] > previous[0]["timestamp"]:
            latest_by_sha[sha256] = (metadata, metadata_path)

    shadows_by_client: dict[str, list[dict]] = defaultdict(list)
    if shadow_log.is_file():
        for line in shadow_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                shadows_by_client[row.get("client_id")].append(row)
    known = json.loads(known_audit.read_text(encoding="utf-8"))

    inventory = []
    teacher_queue = []
    decisions = Counter()
    for sha256, (metadata, metadata_path) in sorted(
        latest_by_sha.items(), key=lambda item: item[1][0]["timestamp"]
    ):
        known_row = known.get(sha256)
        result = metadata.get("result", {})
        classification = result.get("classification") or {}
        shadow = _nearest_shadow(metadata, shadows_by_client)
        verifier = ((shadow or {}).get("verifier") or {}).get("material") or {}
        source_image_path = captures_dir / metadata["image"]["path"]
        image_path = (
            Path(image_path_prefix) / metadata["image"]["path"]
            if image_path_prefix
            else source_image_path.resolve()
        )

        if known_row:
            decision = (
                "known_train" if known_row.get("split") == "train" else "protected_validation"
            )
            expected = known_row.get("label")
        else:
            decision = "teacher_required"
            expected = None
        decisions[decision] += 1
        row = {
            "sha256": sha256,
            "timestamp": metadata["timestamp"],
            "image_path": image_path.as_posix(),
            "decision": decision,
            "known_label": expected,
            "deployed": {
                "status": result.get("status"),
                "class_name": classification.get("class_name"),
                "confidence": classification.get("confidence"),
                "bbox": result.get("bbox"),
            },
            "verifier": {
                "class_name": verifier.get("class_name"),
                "confidence": verifier.get("confidence"),
                "agreement": (shadow or {}).get("material_agreement"),
            },
        }
        inventory.append(row)
        if decision == "teacher_required":
            teacher_queue.append(row)

    (output_dir / "capture_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "teacher_queue.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in teacher_queue),
        encoding="utf-8",
    )
    summary = {
        "start_kst": start_kst.isoformat(),
        "capture_rows_after_cutoff": rows_after_cutoff,
        "unique_images": len(latest_by_sha),
        "decisions": dict(decisions),
        "teacher_queue": len(teacher_queue),
        "client_ids_exported": False,
    }
    (output_dir / "queue_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures-dir", required=True, type=Path)
    parser.add_argument("--shadow-log", required=True, type=Path)
    parser.add_argument("--known-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-kst", default="2026-08-01T00:00:00+09:00")
    parser.add_argument(
        "--image-path-prefix",
        help="Optional runtime prefix used in the exported teacher queue",
    )
    args = parser.parse_args()
    start = _datetime(args.start_kst).astimezone(KST)
    prepare_queue(
        captures_dir=args.captures_dir,
        shadow_log=args.shadow_log,
        known_audit=args.known_audit,
        output_dir=args.output_dir,
        start_kst=start,
        image_path_prefix=args.image_path_prefix,
    )


if __name__ == "__main__":
    main()
