"""multihead.onnx 검증 — 입출력 이름/shape + 더미 추론 (pipeline 호환 확인)."""

import numpy as np
import onnxruntime as ort

s = ort.InferenceSession("/app/crops_state_v1/multihead.onnx", providers=["CPUExecutionProvider"])
print("INPUTS :", [(i.name, i.shape) for i in s.get_inputs()], flush=True)
print("OUTPUTS:", [(o.name, o.shape) for o in s.get_outputs()], flush=True)

x = np.random.randn(1, 3, 224, 224).astype("float32")
dent, label = s.run(["dent", "label"], {"img": x})
print(f"dent shape={dent.shape} argmax={int(dent.argmax())}", flush=True)
print(f"label shape={label.shape} argmax={int(label.argmax())}", flush=True)
print("PIPELINE 호환 OK (img → dent/label)", flush=True)
