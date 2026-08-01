"""노트북에서 운영 하드웨어 캡처용 YOLO 후보를 별도 학습한다.

운영 weights를 덮어쓰지 않으며 결과는 지정한 runs 디렉터리에만 저장한다.
원본 데이터가 합쳐지기 전의 소규모 적응 학습이므로 backbone 대부분을 동결하고
낮은 학습률을 사용한다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--freeze", type=int, default=20)
    parser.add_argument("--lr0", type=float, default=0.0002)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device != "cpu":
        raise RuntimeError("CUDA를 사용할 수 없습니다. CPU 전용 환경에서 후보 학습을 중단합니다.")
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    if not args.data.exists():
        raise FileNotFoundError(args.data)

    args.model = args.model.resolve()
    args.data = args.data.resolve()
    args.project = args.project.resolve()
    run_dir = args.project / args.name
    if run_dir.exists():
        raise FileExistsError(f"run already exists: {run_dir}")
    args.project.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.project / f"{args.name}_launch.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    model = YOLO(str(args.model))
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        freeze=args.freeze,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.05,
        weight_decay=0.0005,
        warmup_epochs=2.0,
        patience=args.patience,
        amp=True,
        cache="ram",
        seed=20260801,
        deterministic=True,
        mosaic=0.5,
        close_mosaic=5,
        mixup=0.0,
        copy_paste=0.0,
        degrees=5.0,
        translate=0.05,
        scale=0.30,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.01,
        hsv_s=0.35,
        hsv_v=0.25,
        project=str(args.project),
        name=args.name,
        exist_ok=False,
        save=True,
        save_period=5,
        plots=True,
        verbose=True,
    )
    summary = {
        key: float(value) if hasattr(value, "__float__") else value
        for key, value in results.results_dict.items()
    }
    (run_dir / "candidate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
