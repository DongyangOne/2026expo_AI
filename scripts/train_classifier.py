"""
상태 멀티헤드 분류기 학습 (MobileNetV3-Small 백본 + dent 헤드 + label 헤드)

- 입력: extract_crops.py가 만든 crops_state_v1/manifest.csv (filepath, category, dent, label)
- dent  : 원형=0 / 변형=1  (페트+캔 공통 학습)
- label : 라벨없음=0 / 라벨있음=1  (페트병만; 캔·내부오염은 -1로 라벨헤드 loss에서 마스킹)
- YOLO가 페트병/캔을 이미 구분하므로 추론 시 페트병=두 헤드, 캔=dent만 읽으면 됨 → 모델 1개로 충분.
- 클래스 불균형은 CrossEntropy class-weight로 보정.
- 출력: best.pt + multihead.onnx (이후 onnx2ncnn으로 NCNN 변환)

실행 (NAS Docker, GPU):
  docker run -d --name train_cls --gpus all -v /share/Container:/app ultralytics/ultralytics:latest \
    python /app/train_classifier.py \
      --crops_dir /app/crops_state_v1 --epochs 25 --batch 128
"""

import argparse
import csv
import os
import random
from collections import Counter

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def read_manifest(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["filepath"], r["category"], int(r["dent"]), int(r["label"])))
    return rows


class CropDataset(Dataset):
    def __init__(self, rows, root, tf):
        self.rows, self.root, self.tf = rows, root, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        fp, _cat, dent, label = self.rows[i]
        img = Image.open(os.path.join(self.root, fp)).convert("RGB")
        return self.tf(img), dent, label


class MultiHead(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        bb = models.mobilenet_v3_small(weights=w)
        feat = bb.classifier[0].in_features      # 576
        bb.classifier = nn.Identity()
        self.backbone = bb
        self.dent_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(feat, 2))
        self.label_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(feat, 2))

    def forward(self, x):
        f = self.backbone(x)
        return self.dent_head(f), self.label_head(f)


def class_weights(values, n_cls=2):
    c = Counter(v for v in values if v >= 0)
    total = sum(c.values())
    if total == 0:
        return None
    w = [total / (n_cls * max(1, c.get(i, 0))) for i in range(n_cls)]
    return torch.tensor(w, dtype=torch.float32)


def run_epoch(model, loader, dent_ce, label_ce, device, opt=None):
    train = opt is not None
    model.train(train)
    dent_ok, dent_n, lab_ok, lab_n = 0, 0, 0, 0
    for img, dent, label in loader:
        img, dent, label = img.to(device), dent.to(device), label.to(device)
        with torch.set_grad_enabled(train):
            d_out, l_out = model(img)
            loss = 0.0
            dmask = dent >= 0   # 플라스틱 등은 dent=-1 → 마스킹
            if dmask.any():
                loss = loss + dent_ce(d_out[dmask], dent[dmask])
            lmask = label >= 0  # 캔 등은 label=-1 → 마스킹
            if lmask.any():
                loss = loss + label_ce(l_out[lmask], label[lmask])
            if train and not isinstance(loss, float):
                opt.zero_grad()
                loss.backward()
                opt.step()
        if dmask.any():
            dent_ok += (d_out[dmask].argmax(1) == dent[dmask]).sum().item()
            dent_n += dmask.sum().item()
        if lmask.any():
            lab_ok += (l_out[lmask].argmax(1) == label[lmask]).sum().item()
            lab_n += lmask.sum().item()
    return dent_ok / max(1, dent_n), lab_ok / max(1, lab_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops_dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15, help="val 개선 없이 N epoch 지나면 조기 종료")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    rows = read_manifest(os.path.join(args.crops_dir, "manifest.csv"))
    random.seed(42)
    random.shuffle(rows)
    n_val = int(len(rows) * args.val_ratio)
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    print(f"train={len(train_rows)} val={len(val_rows)}", flush=True)

    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.RandomRotation(12),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tf_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = CropDataset(train_rows, args.crops_dir, tf_train)
    val_ds = CropDataset(val_rows, args.crops_dir, tf_val)
    train_ld = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_ld = DataLoader(val_ds, args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = MultiHead(pretrained=True).to(device)
    dent_ce = nn.CrossEntropyLoss(weight=class_weights([r[2] for r in train_rows]).to(device))
    label_ce = nn.CrossEntropyLoss(weight=class_weights([r[3] for r in train_rows]).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best, no_improve = 0.0, 0
    for ep in range(1, args.epochs + 1):
        tr_d, tr_l = run_epoch(model, train_ld, dent_ce, label_ce, device, opt)
        va_d, va_l = run_epoch(model, val_ld, dent_ce, label_ce, device)
        sched.step()
        score = va_d + va_l
        print(f"[{ep:02d}/{args.epochs}] train dent={tr_d:.3f} label={tr_l:.3f} | "
              f"val dent={va_d:.3f} label={va_l:.3f}", flush=True)
        if score > best:
            best = score
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.crops_dir, "best.pt"))
            print(f"    ✓ best 저장 (val dent={va_d:.3f} label={va_l:.3f})", flush=True)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n조기 종료: val {args.patience} epoch 개선 없음 (best score={best:.4f})", flush=True)
                break

    # ONNX export (NCNN 변환용)
    model.load_state_dict(torch.load(os.path.join(args.crops_dir, "best.pt")))
    model.eval().cpu()
    dummy = torch.randn(1, 3, 224, 224)
    onnx_path = os.path.join(args.crops_dir, "multihead.onnx")
    torch.onnx.export(model, dummy, onnx_path, input_names=["img"],
                      output_names=["dent", "label"], opset_version=12,
                      dynamic_axes={"img": {0: "batch"}})
    print(f"\nONNX: {onnx_path}", flush=True)


if __name__ == "__main__":
    main()
