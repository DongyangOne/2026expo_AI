"""검수가 끝난 운영 캡처를 crop 검증기 manifest로 변환한다."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from extract_verifier_crops import CLASS_NAMES, letterbox
except ImportError:  # `python -m scripts.import_reviewed_captures` 및 테스트 지원
    from scripts.extract_verifier_crops import CLASS_NAMES, letterbox

CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}


def _optional_binary(review: dict, key: str) -> int:
    value = review.get(key)
    return int(value) if isinstance(value, bool) else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--split", choices=("training", "validation"), default="training")
    args = parser.parse_args()

    captures_dir = Path(args.captures_dir)
    output_dir = Path(args.output_dir)
    rows = []
    skipped = 0

    for metadata_path in sorted(captures_dir.rglob("*.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            review = metadata.get("review", {})
            result = metadata.get("result", {})
            classification = result.get("classification") or {}
            expected = review.get("expected_class")
            if not expected and review.get("is_correct") is True:
                expected = classification.get("class_name")
            if expected not in CLASS_IDS:
                skipped += 1
                continue
            bbox = result.get("bbox")
            if not bbox or len(bbox) != 4:
                skipped += 1
                continue
            image_path = captures_dir / metadata["image"]["path"]
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        height, width = image.shape[:2]
        x1, y1, x2, y2 = (float(value) for value in bbox)
        box_w, box_h = x2 - x1, y2 - y1
        left = max(0, int(x1 - box_w * args.padding))
        top = max(0, int(y1 - box_h * args.padding))
        right = min(width, int(x2 + box_w * args.padding))
        bottom = min(height, int(y2 + box_h * args.padding))
        if right <= left or bottom <= top:
            skipped += 1
            continue

        capture_id = metadata["capture_id"]
        relative_path = Path(args.split) / expected / f"capture_{capture_id}.jpg"
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        crop = letterbox(image[top:bottom, left:right], args.size)
        ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            skipped += 1
            continue
        encoded.tofile(destination)

        rows.append(
            (
                relative_path.as_posix(), args.split, capture_id, CLASS_IDS[expected], expected,
                _optional_binary(review, "is_dented"),
                _optional_binary(review, "has_label"),
                _optional_binary(review, "has_foreign_material"),
                -1, "reviewed_capture",
            )
        )

    manifest_path = output_dir / "reviewed_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "filepath", "split", "source_id", "material", "category",
                "dent", "label", "foreign_material", "label_proxy", "raw_dirtiness",
            ]
        )
        writer.writerows(rows)
    print(f"완료: reviewed={len(rows):,}, skipped={skipped:,} → {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
