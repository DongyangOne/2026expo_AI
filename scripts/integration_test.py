"""통합 테스트 — 실제 모델 로드 후 /detect 엔드포인트에 이미지+무게 호출."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("API_KEY", "test-key")

from fastapi.testclient import TestClient

from app.main import app

H = {"X-API-Key": "test-key"}

# (이미지, 무게g, 설명) — _probe 의 실제 AI Hub 이미지
CASES = [
    ("_probe/clean_1.jpg", 25.0,  "페트병 · 라벨제거 · 무게정상"),
    ("_probe/ext_2.jpg",   30.0,  "페트병 · 라벨부착"),
    ("_probe/clean_3.jpg", 600.0, "페트병 · 무게이상(내용물 의심)"),
    ("_probe/pe_ext_1.jpg", 50.0, "플라스틱 · 라벨부착"),
    ("_probe/pe_clean_1.jpg", 40.0, "플라스틱 · 라벨제거"),
]


def brief(resp: dict) -> str:
    cls = (resp.get("classification") or {}).get("class_name")
    conf = (resp.get("classification") or {}).get("confidence")
    cond = resp.get("conditions") or {}
    w = resp.get("weight") or {}
    g = [x["code"] for x in resp.get("guidance", [])]
    rej = (resp.get("rejection") or {}).get("code")
    gen = (resp.get("general") or {}).get("code")
    return (f"status={resp['status']} class={cls}({conf}) "
            f"dent={cond.get('is_dented')} label={cond.get('has_label')} "
            f"anomaly={w.get('anomaly')} guidance={g} rejection={rej} general={gen}")


with TestClient(app) as client:
    print("HEALTH:", client.get("/health").json())
    print("=" * 70)
    for path, w, desc in CASES:
        if not os.path.exists(path):
            print(f"[SKIP] {desc} — {path} 없음")
            continue
        with open(path, "rb") as f:
            r = client.post(
                "/api/v1/detect",
                files={"image": (os.path.basename(path), f, "image/jpeg")},
                data={"weight_g": str(w)},
                headers=H,
            )
        print(f"[{desc}]  HTTP {r.status_code}")
        print("  ", brief(r.json()))
        print("-" * 70)

    # 인증 실패 케이스
    with open(CASES[0][0], "rb") as f:
        r = client.post("/api/v1/detect", files={"image": ("a.jpg", f, "image/jpeg")},
                        data={"weight_g": "10"}, headers={"X-API-Key": "wrong"})
    print(f"[잘못된 API키]  HTTP {r.status_code}  {r.json()}")
