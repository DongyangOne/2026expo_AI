"""Evaluate a YOLO checkpoint on the fixed original validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def evaluate(
    model_path: Path,
    data_path: Path,
    output_path: Path,
    *,
    device: str,
    batch: int,
    workers: int,
    imgsz: int,
    project: Path,
    name: str,
) -> dict:
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_path),
        split="val",
        device=device,
        batch=batch,
        workers=workers,
        imgsz=imgsz,
        project=str(project),
        name=name,
        exist_ok=False,
        plots=False,
        verbose=True,
    )
    names = getattr(metrics, "names", model.names)
    per_class = {
        str(names[index]): float(value)
        for index, value in enumerate(metrics.box.maps)
    }
    report = {
        "model": str(model_path),
        "data": str(data_path),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "map50_95_by_class": per_class,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=28)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    evaluate(
        args.model,
        args.data,
        args.output,
        device=args.device,
        batch=args.batch,
        workers=args.workers,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
