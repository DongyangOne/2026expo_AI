"""객체 crop 기반 9종 품목 + 상태 멀티태스크 검증기 학습.

기본 백본은 배포 호환성이 검증된 MobileNetV3-Small이다. 최신 경량 후보는
`--backbone mobilenetv4_conv_small.e2400_r224_in1k`처럼 timm 이름을 넘겨
동일 데이터에서 비교한다. 실제 Pi5 ONNX/NCNN 지연시간을 측정한 뒤 채택한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CLASS_NAMES = [
    "can", "pet", "paper", "plastic", "styrofoam",
    "vinyl", "glass", "battery", "fluorescent",
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TASK_NAMES = ("material", "dent", "label", "foreign_material")
TASK_CLASSES = {"material": set(range(9)), "dent": {0, 1}, "label": {0, 1}, "foreign_material": {0, 1}}


def read_manifest(
    paths: list[str],
    use_label_proxy: bool,
    proxy_weight: float,
    oversample_paths: list[str] | None = None,
    oversample_repeats: int = 1,
    oversample_specs: list[tuple[str, int]] | None = None,
):
    rows = []
    sources = [(path, 1) for path in paths]
    sources.extend((path, oversample_repeats) for path in (oversample_paths or []))
    sources.extend(oversample_specs or [])
    for manifest_path, training_repeats in sources:
        root = Path(manifest_path).parent
        with open(manifest_path, encoding="utf-8") as file:
            for row in csv.DictReader(file):
                label = int(row.get("label", -1))
                label_weight = 1.0
                if label < 0 and use_label_proxy:
                    label = int(row.get("label_proxy", -1))
                    label_weight = proxy_weight if label >= 0 else 1.0
                parsed = {
                    "path": str(root / row["filepath"]),
                    "split": row["split"].lower(),
                    "material": int(row["material"]),
                    "dent": int(row.get("dent", -1)),
                    "label": label,
                    "foreign_material": int(row.get("foreign_material", -1)),
                    "label_weight": label_weight,
                }
                repeats = training_repeats if parsed["split"] == "training" else 1
                rows.extend(dict(parsed) for _ in range(repeats))
    return rows


def enabled_tasks_for(rows) -> list[str]:
    enabled = []
    for task in TASK_NAMES:
        required = TASK_CLASSES[task]
        train_values = {
            row[task] for row in rows
            if row["split"] == "training" and row[task] >= 0
        }
        val_values = {
            row[task] for row in rows
            if row["split"] == "validation" and row[task] >= 0
        }
        if train_values == required and val_values == required:
            enabled.append(task)
    return enabled


class VerifierDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return (
            tensor,
            row["material"], row["dent"], row["label"], row["foreign_material"],
            row["label_weight"],
        )


def _build_backbone(name: str, pretrained: bool):
    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        features = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        return backbone, features

    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("최신 백본 사용 시 `pip install timm`이 필요합니다.") from exc
    backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
    return backbone, backbone.num_features


class CropVerifier(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True):
        super().__init__()
        self.backbone, features = _build_backbone(backbone_name, pretrained)
        self.material_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(features, len(CLASS_NAMES)))
        self.dent_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(features, 2))
        self.label_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(features, 2))
        self.foreign_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(features, 2))

    def forward(self, image):
        features = self.backbone(image)
        return (
            self.material_head(features),
            self.dent_head(features),
            self.label_head(features),
            self.foreign_head(features),
        )


def class_weights(values, classes: int):
    counts = Counter(value for value in values if value >= 0)
    total = sum(counts.values())
    if total == 0:
        return None
    return torch.tensor(
        [total / (classes * max(1, counts.get(index, 0))) for index in range(classes)],
        dtype=torch.float32,
    )


def _masked_loss(logits, target, criterion, sample_weight=None):
    mask = target >= 0
    if not mask.any():
        return None, 0, 0
    losses = criterion(logits[mask], target[mask])
    if sample_weight is not None:
        losses = losses * sample_weight[mask]
    correct = (logits[mask].argmax(1) == target[mask]).sum().item()
    return losses.mean(), correct, mask.sum().item()


def run_epoch(model, loader, criteria, task_weights, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = {name: 0 for name in TASK_NAMES}
    counts = {name: 0 for name in TASK_NAMES}
    material_correct = [0] * len(CLASS_NAMES)
    material_counts = [0] * len(CLASS_NAMES)

    for image, material, dent, label, foreign, label_weight in loader:
        image = image.to(device)
        targets = [material.to(device), dent.to(device), label.to(device), foreign.to(device)]
        label_weight = label_weight.to(device=device, dtype=torch.float32)
        with torch.set_grad_enabled(training):
            outputs = model(image)
            material_mask = targets[0] >= 0
            if material_mask.any():
                material_truth = targets[0][material_mask]
                material_guess = outputs[0][material_mask].argmax(1)
                batch_counts = torch.bincount(
                    material_truth, minlength=len(CLASS_NAMES),
                ).detach().cpu().tolist()
                batch_correct = torch.bincount(
                    material_truth[material_guess == material_truth],
                    minlength=len(CLASS_NAMES),
                ).detach().cpu().tolist()
                material_counts = [
                    total + value for total, value in zip(material_counts, batch_counts)
                ]
                material_correct = [
                    total + value for total, value in zip(material_correct, batch_correct)
                ]
            loss = None
            for index, task in enumerate(TASK_NAMES):
                sample_weight = label_weight if task == "label" else None
                task_loss, task_correct, task_count = _masked_loss(
                    outputs[index], targets[index], criteria[task], sample_weight,
                )
                correct[task] += task_correct
                counts[task] += task_count
                if task_loss is not None:
                    weighted = task_loss * task_weights[task]
                    loss = weighted if loss is None else loss + weighted
            if training and loss is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if loss is not None:
                total_loss += float(loss.detach())

    accuracy = {
        task: (correct[task] / counts[task] if counts[task] else None)
        for task in TASK_NAMES
    }
    material_class_accuracy = {
        CLASS_NAMES[index]: (
            material_correct[index] / material_counts[index]
            if material_counts[index] else None
        )
        for index in range(len(CLASS_NAMES))
    }
    return total_loss / max(1, len(loader)), accuracy, counts, material_class_accuracy


def _parse_oversample_spec(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--oversample-spec은 PATH=REPEATS 형식이어야 합니다.")
    path, raw_repeats = value.rsplit("=", 1)
    try:
        repeats = int(raw_repeats)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("REPEATS는 정수여야 합니다.") from exc
    if not path.strip() or repeats < 1:
        raise argparse.ArgumentTypeError("PATH는 비어 있지 않고 REPEATS는 양수여야 합니다.")
    return path.strip(), repeats


def _format_metrics(metrics):
    return " ".join(
        f"{task}={value:.3f}" if value is not None else f"{task}=NA"
        for task, value in metrics.items()
    )


def export_onnx(model, path: Path, size: int):
    model.eval().cpu()
    dummy = torch.randn(1, 3, size, size)
    kwargs = dict(
        input_names=["img"],
        output_names=["material", "dent", "label", "foreign_material"],
        opset_version=17,
        dynamic_axes={"img": {0: "batch"}},
    )
    try:
        torch.onnx.export(model, dummy, path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, path, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument(
        "--oversample-manifest", action="append", default=[],
        help="validation은 한 번, training 행만 --oversample-repeats만큼 추가합니다.",
    )
    parser.add_argument("--oversample-repeats", type=int, default=5)
    parser.add_argument(
        "--oversample-spec", action="append", default=[], type=_parse_oversample_spec,
        metavar="PATH=REPEATS",
        help="manifest마다 서로 다른 training 반복 수를 지정합니다.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone", default="mobilenet_v3_small")
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--init-checkpoint")
    parser.add_argument(
        "--camera-augmentation", action="store_true",
        help="하드웨어 카메라의 원근·약한 초점 흐림을 모사합니다.",
    )
    parser.add_argument("--use-label-proxy", action="store_true")
    parser.add_argument("--label-proxy-weight", type=float, default=0.25)
    parser.add_argument("--material-weight", type=float, default=1.0)
    parser.add_argument("--dent-weight", type=float, default=0.5)
    parser.add_argument("--label-weight", type=float, default=0.5)
    parser.add_argument("--foreign-weight", type=float, default=1.0)
    parser.add_argument("--selection-material-weight", type=float, default=1.0)
    parser.add_argument("--selection-dent-weight", type=float, default=1.0)
    parser.add_argument("--selection-label-weight", type=float, default=1.0)
    parser.add_argument("--selection-foreign-weight", type=float, default=1.0)
    parser.add_argument(
        "--selection-material-target", action="append", default=[], choices=CLASS_NAMES,
        help="체크포인트 선택에서 별도 평균 재현율을 계산할 품목입니다.",
    )
    parser.add_argument("--selection-material-target-weight", type=float, default=0.0)
    args = parser.parse_args()
    if args.oversample_repeats < 1:
        parser.error("--oversample-repeats must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(
        args.manifest,
        args.use_label_proxy,
        args.label_proxy_weight,
        args.oversample_manifest,
        args.oversample_repeats,
        args.oversample_spec,
    )
    train_rows = [row for row in rows if row["split"] == "training"]
    val_rows = [row for row in rows if row["split"] == "validation"]
    if not train_rows or not val_rows:
        raise SystemExit("[ERROR] manifest에 training/validation 데이터가 모두 필요합니다.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} train={len(train_rows):,} val={len(val_rows):,}", flush=True)
    label_counts = {
        task: sum(row[task] >= 0 for row in train_rows)
        for task in TASK_NAMES
    }
    enabled_tasks = enabled_tasks_for(rows)
    print(f"labeled counts={label_counts}", flush=True)
    print(f"enabled tasks={enabled_tasks}", flush=True)

    train_steps = [
        transforms.Resize((args.size, args.size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.04, 0.04), scale=(0.92, 1.08)),
    ]
    if args.camera_augmentation:
        train_steps.extend([
            transforms.RandomPerspective(distortion_scale=0.20, p=0.25),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.15,
            ),
        ])
    train_steps.extend([
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.04),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1, scale=(0.01, 0.05)),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_transform = transforms.Compose(train_steps)
    val_transform = transforms.Compose(
        [
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_loader = DataLoader(
        VerifierDataset(train_rows, train_transform), batch_size=args.batch,
        shuffle=True, num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        VerifierDataset(val_rows, val_transform), batch_size=args.batch,
        shuffle=False, num_workers=args.workers, pin_memory=True,
    )

    model = CropVerifier(args.backbone, pretrained=not bool(args.init_checkpoint))
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if initial.get("backbone") != args.backbone:
            raise RuntimeError(
                f"초기 체크포인트 backbone={initial.get('backbone')!r}, "
                f"요청 backbone={args.backbone!r} 불일치"
            )
        if int(initial.get("input_size", args.size)) != args.size:
            raise RuntimeError("초기 체크포인트 input_size가 요청 size와 다릅니다.")
        if initial.get("classes", CLASS_NAMES) != CLASS_NAMES:
            raise RuntimeError("초기 체크포인트 class 순서가 현재 계약과 다릅니다.")
        model.load_state_dict(initial["state_dict"])
        print(f"initial checkpoint={args.init_checkpoint}", flush=True)
    model = model.to(device)
    weight_values = {
        "material": class_weights([row["material"] for row in train_rows], len(CLASS_NAMES)),
        "dent": class_weights([row["dent"] for row in train_rows], 2),
        "label": class_weights([row["label"] for row in train_rows], 2),
        "foreign_material": class_weights([row["foreign_material"] for row in train_rows], 2),
    }
    criteria = {
        task: nn.CrossEntropyLoss(
            weight=(weights.to(device) if weights is not None else None), reduction="none",
        )
        for task, weights in weight_values.items()
    }
    task_weights = {
        "material": args.material_weight,
        "dent": args.dent_weight,
        "label": args.label_weight,
        "foreign_material": args.foreign_weight,
    }
    task_weights = {
        task: weight if task in enabled_tasks else 0.0
        for task, weight in task_weights.items()
    }
    selection_weights = {
        "material": args.selection_material_weight,
        "dent": args.selection_dent_weight,
        "label": args.selection_label_weight,
        "foreign_material": args.selection_foreign_weight,
        "material_target": args.selection_material_target_weight,
    }
    if any(weight < 0 for weight in selection_weights.values()):
        parser.error("selection weights must be non-negative")
    if args.selection_material_target_weight > 0 and not args.selection_material_target:
        parser.error("target weight 사용 시 --selection-material-target이 필요합니다.")
    if not any(selection_weights[task] > 0 for task in enabled_tasks) and not (
        args.selection_material_target and args.selection_material_target_weight > 0
    ):
        parser.error("at least one enabled selection weight must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    checkpoint_path = output_dir / "best_verifier.pt"
    best_score = -1.0
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _, train_material_classes = run_epoch(
            model, train_loader, criteria, task_weights, device, optimizer,
        )
        val_loss, val_metrics, val_counts, val_material_classes = run_epoch(
            model, val_loader, criteria, task_weights, device,
        )
        scheduler.step()
        available = {
            task: val_metrics[task]
            for task in enabled_tasks
            if val_metrics[task] is not None and selection_weights[task] > 0
        }
        target_values = [
            val_material_classes[name]
            for name in args.selection_material_target
            if val_material_classes[name] is not None
        ]
        if target_values and args.selection_material_target_weight > 0:
            available["material_target"] = sum(target_values) / len(target_values)
        if not available:
            raise RuntimeError("train/validation에 완전한 정답을 가진 활성 task가 없습니다.")
        score_weight = sum(selection_weights[task] for task in available)
        score = sum(
            value * selection_weights[task] for task, value in available.items()
        ) / score_weight
        print(
            f"[{epoch:02d}/{args.epochs}] loss={train_loss:.4f}/{val_loss:.4f} "
            f"train {_format_metrics(train_metrics)} | val {_format_metrics(val_metrics)}"
            + (
                " | target " + " ".join(
                    f"{name}={val_material_classes[name]:.3f}"
                    for name in args.selection_material_target
                    if val_material_classes[name] is not None
                )
                if args.selection_material_target else ""
            ),
            flush=True,
        )
        if score > best_score:
            best_score = score
            no_improve = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "input_size": args.size,
                    "classes": CLASS_NAMES,
                    "epoch": epoch,
                    "selection_score": score,
                    "selection_weights": selection_weights,
                    "val_metrics": val_metrics,
                    "val_counts": val_counts,
                    "val_material_class_accuracy": val_material_classes,
                    "selection_material_targets": args.selection_material_target,
                },
                checkpoint_path,
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"조기 종료: {args.patience} epochs no improvement", flush=True)
                break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    export_model = CropVerifier(checkpoint["backbone"], pretrained=False)
    export_model.load_state_dict(checkpoint["state_dict"])
    onnx_path = output_dir / "verifier.onnx"
    export_onnx(export_model, onnx_path, checkpoint["input_size"])

    enabled_outputs = enabled_tasks
    task_class_counts = {
        split: {
            task: dict(sorted(Counter(
                row[task] for row in rows
                if row["split"] == split and row[task] >= 0
            ).items()))
            for task in TASK_NAMES
        }
        for split in ("training", "validation")
    }
    metadata = {
        "backbone": args.backbone,
        "input_size": args.size,
        "classes": CLASS_NAMES,
        "outputs": list(TASK_NAMES),
        "enabled_outputs": enabled_outputs,
        "training_label_counts": label_counts,
        "task_class_counts": task_class_counts,
        "uses_label_proxy": args.use_label_proxy,
        "initial_checkpoint": args.init_checkpoint,
        "camera_augmentation": args.camera_augmentation,
        "best_epoch": checkpoint.get("epoch"),
        "best_selection_score": checkpoint.get("selection_score"),
        "best_val_metrics": checkpoint.get("val_metrics"),
        "best_val_material_class_accuracy": checkpoint.get("val_material_class_accuracy"),
        "selection_weights": checkpoint.get("selection_weights", selection_weights),
        "selection_material_targets": checkpoint.get(
            "selection_material_targets", args.selection_material_target,
        ),
        "warning": "Do not consume outputs absent from enabled_outputs.",
    }
    (output_dir / "verifier_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"ONNX: {onnx_path}", flush=True)
    print(f"enabled outputs: {enabled_outputs}", flush=True)


if __name__ == "__main__":
    main()
