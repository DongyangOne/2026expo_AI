"""객체 crop 기반 9종 품목 + 상태 멀티태스크 검증기 학습.

기본 백본은 배포 호환성이 검증된 MobileNetV3-Small이다. 최신 경량 후보는
`--backbone mobilenetv4_conv_small.e2400_r224_in1k`처럼 timm 이름을 넘겨
동일 데이터에서 비교한다. 실제 Pi5 ONNX/NCNN 지연시간을 측정한 뒤 채택한다.

기본 material 계약은 기존 9종을 그대로 유지한다. 명시적으로
`--include-background`를 지정한 학습에서만 열 번째 ``background`` 클래스를
추가한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
BACKGROUND_CLASS_NAME = "background"
BACKGROUND_CLASS_NAMES = [*CLASS_NAMES, BACKGROUND_CLASS_NAME]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TASK_NAMES = ("material", "dent", "label", "foreign_material")
TASK_CLASSES = {"material": set(range(9)), "dent": {0, 1}, "label": {0, 1}, "foreign_material": {0, 1}}
CLASS_WEIGHT_MODES = ("inverse", "none", "effective-number")
DEFAULT_CLASS_WEIGHT_BETA = 0.9999


def material_class_names(include_background: bool = False) -> list[str]:
    """Return a fresh material-class contract for one training run."""
    return list(BACKGROUND_CLASS_NAMES if include_background else CLASS_NAMES)


def task_classes_for(classes: list[str] | tuple[str, ...]) -> dict[str, set[int]]:
    return {
        "material": set(range(len(classes))),
        "dent": {0, 1},
        "label": {0, 1},
        "foreign_material": {0, 1},
    }


def validate_task_labels(rows, classes: list[str] | tuple[str, ...]) -> None:
    """Fail early when a manifest contains a label outside the active contract."""
    allowed_by_task = task_classes_for(classes)
    for row_index, row in enumerate(rows, start=1):
        for task, allowed in allowed_by_task.items():
            value = row[task]
            if value >= 0 and value not in allowed:
                raise ValueError(
                    f"manifest row {row_index}: {task}={value} is outside "
                    f"the active classes {sorted(allowed)}"
                )


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


def enabled_tasks_for(
    rows,
    classes: list[str] | tuple[str, ...] = CLASS_NAMES,
) -> list[str]:
    enabled = []
    task_classes = task_classes_for(classes)
    for task in TASK_NAMES:
        required = task_classes[task]
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


class MotionBlur(nn.Module):
    """투입 순간의 손떨림·이동 흔들림을 모사한다. torchvision에 없어 직접 구현한다."""

    def __init__(self, kernel: int = 7, p: float = 0.15):
        super().__init__()
        self.kernel = kernel if kernel % 2 else kernel + 1
        self.p = p

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return image
        weight = torch.zeros(self.kernel, self.kernel)
        if torch.rand(1).item() < 0.5:
            weight[self.kernel // 2, :] = 1.0          # 수평 흔들림
        else:
            weight[:, self.kernel // 2] = 1.0          # 수직 흔들림
        weight = (weight / weight.sum()).expand(3, 1, self.kernel, self.kernel)
        blurred = torch.nn.functional.conv2d(
            image.unsqueeze(0), weight, padding=self.kernel // 2, groups=3,
        )
        return blurred.squeeze(0)


class GaussianNoise(nn.Module):
    """저조도 키오스크 카메라의 센서 노이즈를 모사한다."""

    def __init__(self, sigma: float = 0.04, p: float = 0.20):
        super().__init__()
        self.sigma = sigma
        self.p = p

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return image
        return (image + torch.randn_like(image) * self.sigma).clamp(0.0, 1.0)


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
    # timm의 num_features는 백본군에 따라 head 확장 전 채널을 보고하기도 한다
    # (예: mobilenetv4_conv_small은 960을 보고하지만 실제 pooled 출력은 1280이다).
    # 신고값을 믿지 않고 더미 forward로 실제 채널 수를 확인한다.
    backbone.eval()
    with torch.no_grad():
        probe = backbone(torch.zeros(1, 3, 224, 224))
    features = int(probe.shape[-1])
    if features != backbone.num_features:
        print(
            f"[경고] {name}: timm num_features={backbone.num_features}이지만 "
            f"실제 pooled 출력={features}. 실측값을 사용합니다.", flush=True,
        )
    return backbone, features


class CropVerifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool = True,
        material_classes: list[str] | tuple[str, ...] = CLASS_NAMES,
    ):
        super().__init__()
        if not material_classes or len(set(material_classes)) != len(material_classes):
            raise ValueError("material_classes must be non-empty and unique")
        self.material_classes = tuple(material_classes)
        self.backbone, features = _build_backbone(backbone_name, pretrained)
        self.material_head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(features, len(self.material_classes)),
        )
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


def class_weights(
    values,
    classes: int,
    mode: str = "inverse",
    beta: float = DEFAULT_CLASS_WEIGHT_BETA,
):
    """Build class weights while preserving the legacy inverse default.

    Effective-number weights follow ``(1 - beta) / (1 - beta**count)`` and
    are normalized to a mean of one so that changing the mode does not also
    change the overall loss scale. Missing classes use a count of one, matching
    the finite fallback used by the legacy inverse implementation.
    """
    if classes < 1:
        raise ValueError("classes must be positive")
    if mode not in CLASS_WEIGHT_MODES:
        raise ValueError(f"unsupported class weight mode: {mode}")
    if not math.isfinite(beta) or not 0 <= beta < 1:
        raise ValueError("class weight beta must be finite and in [0, 1)")
    if mode == "none":
        return None

    counts = Counter(value for value in values if value >= 0)
    total = sum(counts.values())
    if total == 0:
        return None
    finite_counts = [max(1, counts.get(index, 0)) for index in range(classes)]
    if mode == "inverse":
        return torch.tensor(
            [total / (classes * count) for count in finite_counts],
            dtype=torch.float32,
        )

    count_tensor = torch.tensor(finite_counts, dtype=torch.float64)
    if beta == 0:
        weights = torch.ones(classes, dtype=torch.float64)
    else:
        denominator = -torch.expm1(count_tensor * math.log(beta))
        weights = (1 - beta) / denominator
    weights = weights / weights.mean()
    if not torch.isfinite(weights).all():
        raise ValueError("effective-number class weights must be finite")
    return weights.to(dtype=torch.float32)


def build_criteria(weight_values, device, label_smoothing: float = 0.0):
    if not math.isfinite(label_smoothing) or not 0 <= label_smoothing < 1:
        raise ValueError("label smoothing must be finite and in [0, 1)")
    return {
        task: nn.CrossEntropyLoss(
            weight=(weights.to(device) if weights is not None else None),
            reduction="none",
            label_smoothing=label_smoothing,
        )
        for task, weights in weight_values.items()
    }


def resolve_learning_rates(
    lr: float,
    backbone_lr: float | None = None,
    head_lr: float | None = None,
) -> tuple[float, float]:
    values = {
        "lr": lr,
        "backbone_lr": lr if backbone_lr is None else backbone_lr,
        "head_lr": lr if head_lr is None else head_lr,
    }
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    return values["backbone_lr"], values["head_lr"]


def build_optimizer(
    model: CropVerifier,
    lr: float,
    backbone_lr: float | None = None,
    head_lr: float | None = None,
):
    actual_backbone_lr, actual_head_lr = resolve_learning_rates(
        lr, backbone_lr, head_lr,
    )
    backbone_parameters = list(model.backbone.parameters())
    backbone_parameter_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [
        parameter for parameter in model.parameters()
        if id(parameter) not in backbone_parameter_ids
    ]
    if not backbone_parameters or not head_parameters:
        raise ValueError("optimizer requires non-empty backbone and head parameters")
    return torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": actual_backbone_lr, "name": "backbone"},
            {"params": head_parameters, "lr": actual_head_lr, "name": "heads"},
        ],
        lr=lr,
        weight_decay=1e-4,
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


def run_epoch(
    model,
    loader,
    criteria,
    task_weights,
    device,
    optimizer=None,
    material_classes: list[str] | tuple[str, ...] = CLASS_NAMES,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = {name: 0 for name in TASK_NAMES}
    counts = {name: 0 for name in TASK_NAMES}
    material_correct = [0] * len(material_classes)
    material_counts = [0] * len(material_classes)

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
                    material_truth, minlength=len(material_classes),
                ).detach().cpu().tolist()
                batch_correct = torch.bincount(
                    material_truth[material_guess == material_truth],
                    minlength=len(material_classes),
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
        material_classes[index]: (
            material_correct[index] / material_counts[index]
            if material_counts[index] else None
        )
        for index in range(len(material_classes))
    }
    return total_loss / max(1, len(loader)), accuracy, counts, material_class_accuracy


def load_initial_checkpoint_state(
    model: CropVerifier,
    checkpoint: dict,
    target_classes: list[str] | tuple[str, ...],
) -> dict:
    """Load an exact checkpoint or expand the legacy 9-class material head.

    For the opt-in 10-class model, every compatible tensor is transferred from
    the 9-class checkpoint. The first nine rows of the material classifier are
    copied and the new background row retains the model's normal initialization.
    """
    source_classes = list(checkpoint.get("classes", CLASS_NAMES))
    target_classes = list(target_classes)
    source_state = checkpoint.get("state_dict")
    if not isinstance(source_state, dict):
        raise RuntimeError("초기 체크포인트에 state_dict가 없습니다.")

    if source_classes == target_classes:
        model.load_state_dict(source_state)
        return {"mode": "exact", "source_classes": source_classes}

    expanding_background = (
        source_classes == CLASS_NAMES
        and target_classes == BACKGROUND_CLASS_NAMES
    )
    if not expanding_background:
        raise RuntimeError("초기 체크포인트 class 순서가 현재 계약과 다릅니다.")

    target_state = model.state_dict()
    missing = sorted(set(target_state) - set(source_state))
    unexpected = sorted(set(source_state) - set(target_state))
    if missing or unexpected:
        raise RuntimeError(
            f"초기 체크포인트 state_dict 키 불일치: missing={missing}, "
            f"unexpected={unexpected}"
        )

    material_output = model.material_head[-1]
    material_output_name = next(
        name for name, module in model.named_modules() if module is material_output
    )
    expanded_keys = {f"{material_output_name}.weight"}
    if material_output.bias is not None:
        expanded_keys.add(f"{material_output_name}.bias")

    expanded_state = {}
    source_count = len(source_classes)
    target_count = len(target_classes)
    for key, target_tensor in target_state.items():
        source_tensor = source_state[key]
        if key in expanded_keys:
            if (
                source_tensor.shape[0] != source_count
                or target_tensor.shape[0] != target_count
                or source_tensor.shape[1:] != target_tensor.shape[1:]
            ):
                raise RuntimeError(
                    f"초기 체크포인트 material head shape 불일치: "
                    f"{key}={tuple(source_tensor.shape)} -> {tuple(target_tensor.shape)}"
                )
            expanded = target_tensor.clone()
            expanded[:source_count].copy_(
                source_tensor.to(device=expanded.device, dtype=expanded.dtype)
            )
            expanded_state[key] = expanded
        else:
            if source_tensor.shape != target_tensor.shape:
                raise RuntimeError(
                    f"초기 체크포인트 tensor shape 불일치: "
                    f"{key}={tuple(source_tensor.shape)} -> {tuple(target_tensor.shape)}"
                )
            expanded_state[key] = source_tensor

    model.load_state_dict(expanded_state)
    return {"mode": "expanded_background", "source_classes": source_classes}


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
    parser.add_argument(
        "--backbone-lr", type=float,
        help="백본 학습률입니다. 생략하면 --lr을 사용합니다.",
    )
    parser.add_argument(
        "--head-lr", type=float,
        help="분류/상태 head 학습률입니다. 생략하면 --lr을 사용합니다.",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--class-weight-mode", choices=CLASS_WEIGHT_MODES, default="inverse",
        help="클래스 불균형 가중치 방식입니다. 기본 inverse는 기존 동작입니다.",
    )
    parser.add_argument(
        "--class-weight-beta", type=float, default=DEFAULT_CLASS_WEIGHT_BETA,
        help="effective-number 모드의 beta 값([0, 1))입니다.",
    )
    parser.add_argument("--init-checkpoint")
    parser.add_argument(
        "--include-background", action="store_true",
        help="material head에 열 번째 background 클래스를 명시적으로 추가합니다.",
    )
    parser.add_argument(
        "--camera-augmentation", action="store_true",
        help="하드웨어 카메라의 원근·약한 초점 흐림을 모사합니다.",
    )
    parser.add_argument(
        "--kiosk-augmentation", action="store_true",
        help="camera 증강에 국소대비·강한 색조·모션블러·센서노이즈를 더합니다. "
             "실제 키오스크 조명이 학습 데이터와 다를 때의 오분류를 겨냥합니다.",
    )
    parser.add_argument(
        "--augmix", action="store_true",
        help="AugMix(arXiv:1912.02781)로 미지의 손상에 대한 강건성을 높입니다.",
    )
    parser.add_argument("--augmix-severity", type=int, default=3)
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
        "--selection-material-target", action="append", default=[],
        choices=BACKGROUND_CLASS_NAMES,
        help="체크포인트 선택에서 별도 평균 재현율을 계산할 품목입니다.",
    )
    parser.add_argument("--selection-material-target-weight", type=float, default=0.0)
    args = parser.parse_args()
    if args.oversample_repeats < 1:
        parser.error("--oversample-repeats must be positive")
    try:
        actual_backbone_lr, actual_head_lr = resolve_learning_rates(
            args.lr, args.backbone_lr, args.head_lr,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not math.isfinite(args.label_smoothing) or not 0 <= args.label_smoothing < 1:
        parser.error("--label-smoothing must be finite and in [0, 1)")
    if not math.isfinite(args.class_weight_beta) or not 0 <= args.class_weight_beta < 1:
        parser.error("--class-weight-beta must be finite and in [0, 1)")
    active_classes = material_class_names(args.include_background)
    if (
        BACKGROUND_CLASS_NAME in args.selection_material_target
        and not args.include_background
    ):
        parser.error("background target 사용 시 --include-background가 필요합니다.")

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
    try:
        validate_task_labels(rows, active_classes)
    except ValueError as exc:
        parser.error(str(exc))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} train={len(train_rows):,} val={len(val_rows):,}", flush=True)
    label_counts = {
        task: sum(row[task] >= 0 for row in train_rows)
        for task in TASK_NAMES
    }
    enabled_tasks = enabled_tasks_for(rows, active_classes)
    print(f"labeled counts={label_counts}", flush=True)
    print(f"enabled tasks={enabled_tasks}", flush=True)

    train_steps = [
        transforms.Resize((args.size, args.size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.04, 0.04), scale=(0.92, 1.08)),
    ]
    if args.camera_augmentation or args.kiosk_augmentation:
        train_steps.extend([
            transforms.RandomPerspective(distortion_scale=0.20, p=0.25),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.15,
            ),
        ])
    if args.kiosk_augmentation:
        # 조명·화이트밸런스가 학습 데이터와 다른 실제 키오스크 입력을 겨냥한다.
        train_steps.extend([
            transforms.RandomAutocontrast(p=0.25),
            transforms.RandomEqualize(p=0.10),
        ])
    if args.augmix:
        # AugMix는 여러 증강 체인을 섞어 원본에 합성한다. CIFAR-10-C에서
        # RandAugment(19.65%)보다 낮은 오류율(15.24%)을 보고한 손상 강건화 기법이다.
        # arXiv:1912.02781. uint8 PIL 입력이 필요하므로 ToTensor 앞에 둔다.
        train_steps.append(transforms.AugMix(severity=args.augmix_severity))
    train_steps.append(
        transforms.ColorJitter(0.35, 0.35, 0.35, 0.08)
        if args.kiosk_augmentation
        else transforms.ColorJitter(0.2, 0.2, 0.2, 0.04)
    )
    train_steps.append(transforms.ToTensor())
    if args.kiosk_augmentation:
        train_steps.extend([MotionBlur(), GaussianNoise()])
    train_steps.extend([
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

    model = CropVerifier(
        args.backbone,
        pretrained=not bool(args.init_checkpoint),
        material_classes=active_classes,
    )
    initial_checkpoint_info = None
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if initial.get("backbone") != args.backbone:
            raise RuntimeError(
                f"초기 체크포인트 backbone={initial.get('backbone')!r}, "
                f"요청 backbone={args.backbone!r} 불일치"
            )
        if int(initial.get("input_size", args.size)) != args.size:
            raise RuntimeError("초기 체크포인트 input_size가 요청 size와 다릅니다.")
        initial_checkpoint_info = load_initial_checkpoint_state(
            model, initial, active_classes,
        )
        print(
            f"initial checkpoint={args.init_checkpoint} "
            f"mode={initial_checkpoint_info['mode']}",
            flush=True,
        )
    model = model.to(device)
    weight_values = {
        "material": class_weights(
            [row["material"] for row in train_rows], len(active_classes),
            args.class_weight_mode, args.class_weight_beta,
        ),
        "dent": class_weights(
            [row["dent"] for row in train_rows], 2,
            args.class_weight_mode, args.class_weight_beta,
        ),
        "label": class_weights(
            [row["label"] for row in train_rows], 2,
            args.class_weight_mode, args.class_weight_beta,
        ),
        "foreign_material": class_weights(
            [row["foreign_material"] for row in train_rows], 2,
            args.class_weight_mode, args.class_weight_beta,
        ),
    }
    criteria = build_criteria(weight_values, device, args.label_smoothing)
    training_config = {
        "label_smoothing": args.label_smoothing,
        "learning_rates": {
            "base": args.lr,
            "backbone": actual_backbone_lr,
            "heads": actual_head_lr,
        },
        "class_weights": {
            "mode": args.class_weight_mode,
            "beta": args.class_weight_beta,
            "values": {
                task: (weights.tolist() if weights is not None else None)
                for task, weights in weight_values.items()
            },
        },
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
    optimizer = build_optimizer(
        model, args.lr, args.backbone_lr, args.head_lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    checkpoint_path = output_dir / "best_verifier.pt"
    best_score = -1.0
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _, train_material_classes = run_epoch(
            model, train_loader, criteria, task_weights, device, optimizer,
            active_classes,
        )
        val_loss, val_metrics, val_counts, val_material_classes = run_epoch(
            model, val_loader, criteria, task_weights, device,
            material_classes=active_classes,
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
                    "classes": active_classes,
                    "include_background": args.include_background,
                    "background_class_id": (
                        active_classes.index(BACKGROUND_CLASS_NAME)
                        if args.include_background else None
                    ),
                    "epoch": epoch,
                    "selection_score": score,
                    "selection_weights": selection_weights,
                    "val_metrics": val_metrics,
                    "val_counts": val_counts,
                    "val_material_class_accuracy": val_material_classes,
                    "selection_material_targets": args.selection_material_target,
                    "training_config": training_config,
                },
                checkpoint_path,
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"조기 종료: {args.patience} epochs no improvement", flush=True)
                break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_classes = list(checkpoint.get("classes", CLASS_NAMES))
    export_model = CropVerifier(
        checkpoint["backbone"], pretrained=False,
        material_classes=checkpoint_classes,
    )
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
        "classes": checkpoint_classes,
        "material_class_count": len(checkpoint_classes),
        "include_background": BACKGROUND_CLASS_NAME in checkpoint_classes,
        "background_class_id": (
            checkpoint_classes.index(BACKGROUND_CLASS_NAME)
            if BACKGROUND_CLASS_NAME in checkpoint_classes else None
        ),
        "outputs": list(TASK_NAMES),
        "enabled_outputs": enabled_outputs,
        "training_label_counts": label_counts,
        "task_class_counts": task_class_counts,
        "uses_label_proxy": args.use_label_proxy,
        "initial_checkpoint": args.init_checkpoint,
        "initial_checkpoint_transfer": initial_checkpoint_info,
        "camera_augmentation": args.camera_augmentation,
        "kiosk_augmentation": args.kiosk_augmentation,
        "augmix": args.augmix,
        "augmix_severity": args.augmix_severity if args.augmix else None,
        "training_config": checkpoint.get("training_config", training_config),
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
