"""학습 시작 전 CUDA 가용성 어설션 (컨테이너 내부 실행).

NVRM 커널 메모리 부족(NV_ERR_NO_MEMORY)으로 cuInit이 일시 실패할 수 있어
재시도 루프를 둔다 — 변환 작업 직후 커널 메모리 회수에 시간이 걸리는 경우 대비.
"""
import sys
import time

import torch

ATTEMPTS = 10
WAIT_SEC = 30

for i in range(1, ATTEMPTS + 1):
    if torch.cuda.is_available():
        print(f"[gpu_assert] CUDA OK (attempt {i}): {torch.cuda.get_device_name(0)}", flush=True)
        sys.exit(0)
    print(f"[gpu_assert] CUDA unavailable (attempt {i}/{ATTEMPTS}) — {WAIT_SEC}s 후 재시도", flush=True)
    time.sleep(WAIT_SEC)

print("[gpu_assert] FAILED: CUDA를 잡지 못했습니다. 호스트 dmesg의 NVRM 메모리 오류 확인 필요.", flush=True)
sys.exit(1)
