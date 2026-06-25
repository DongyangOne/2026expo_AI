"""
best.pt → multihead.onnx 재변환 (학습 컨테이너의 onnxscript 누락 우회).

최신 torch는 export 기본이 dynamo(onnxscript 의존) → dynamo=False 로 레거시
TorchScript exporter 강제 (onnxscript 불필요). 구버전 torch는 해당 인자 없으므로 폴백.

실행 (NAS Docker):
  docker run --rm -v /share/Container:/app ultralytics/ultralytics:latest \
    python /app/export_onnx.py
"""

import sys

sys.path.insert(0, "/app")

import torch
from train_classifier import MultiHead

CKPT = "/app/crops_state_v1/best.pt"
OUT = "/app/crops_state_v1/multihead.onnx"

model = MultiHead(pretrained=False)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()

dummy = torch.randn(1, 3, 224, 224)
kwargs = dict(
    input_names=["img"],
    output_names=["dent", "label"],
    opset_version=12,
    dynamic_axes={"img": {0: "batch"}},
)

try:
    torch.onnx.export(model, dummy, OUT, dynamo=False, **kwargs)
except TypeError:
    torch.onnx.export(model, dummy, OUT, **kwargs)

print(f"ONNX 생성 완료: {OUT}", flush=True)
