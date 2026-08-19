"""Blind-label new operational captures with two structured Qwen-VL passes."""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path


MATERIALS = [
    "can", "pet", "paper", "plastic", "styrofoam", "vinyl", "glass",
    "battery", "fluorescent", "negative", "exclude",
]
SCHEMA = {
    "type": "object",
    "properties": {
        "material": {"type": "string", "enum": MATERIALS},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "single_object": {"type": "boolean"},
        "foreign_material": {"type": "boolean"},
        "description": {"type": "string", "maxLength": 160},
    },
    "required": [
        "material", "confidence", "single_object", "foreign_material", "description"
    ],
}

PROMPTS = (
    """사진 중앙 투입구에 들어온 쓰레기의 주재질 하나를 분류하세요. 가능한 값은
can(금속 캔), pet(PET 음료병), paper(종이/종이상자), plastic(PET 아닌 플라스틱),
styrofoam(스티로폼), vinyl(비닐/봉투), glass, battery, fluorescent,
negative(쓰레기 없음), exclude(판별 불가/여러 쓰레기 혼합)입니다. 빨대처럼 같은
재질의 작은 부속품은 허용하지만 종이컵 띠지처럼 다른 재질이 붙었으면
foreign_material=true로 표시하세요. 모델의 기존 예측은 참고하지 말고 사진만 보세요.""",
    """재활용 키오스크 카메라 사진을 독립적으로 재검토하세요. 화면의 주된 물체가
정확히 하나인지, 그 물체의 주재질이 9종 중 무엇인지 판단하세요. 물체가 없으면
negative, 여러 물체/너무 불명확하면 exclude를 사용하세요. 주재질과 다른 부착물이나
혼합물이 보이면 foreign_material=true입니다. 자신 없으면 confidence를 낮추세요.""",
)


def _request(url: str, model: str, image: bytes, prompt: str, timeout: int) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt + " JSON 스키마에 맞는 결과만 반환하세요.",
                "images": [base64.b64encode(image).decode("ascii")],
            }
        ],
        "format": SCHEMA,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0,
            "seed": 20260819,
            # The teacher only needs a compact JSON decision.  A hard ceiling
            # prevents a malformed/free-form response from monopolising the
            # NAS GPU while preserving every structured field above.
            "num_predict": 160,
        },
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    result = json.loads(body["message"]["content"])
    if result.get("material") not in MATERIALS:
        raise ValueError(f"invalid material: {result.get('material')}")
    if not 0 <= float(result.get("confidence", -1)) <= 1:
        raise ValueError("invalid confidence")
    return result


def label_queue(
    queue_path: Path,
    output_path: Path,
    *,
    url: str,
    model: str,
    timeout: int,
    retries: int,
) -> dict:
    queue = [
        json.loads(line)
        for line in queue_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    completed: dict[str, dict] = {}
    if output_path.is_file():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["sha256"]] = row

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output:
        for index, row in enumerate(queue, start=1):
            if row["sha256"] in completed:
                continue
            image = Path(row["image_path"]).read_bytes()
            passes = []
            for prompt in PROMPTS:
                for attempt in range(1, retries + 1):
                    try:
                        passes.append(_request(url, model, image, prompt, timeout))
                        break
                    except Exception:
                        if attempt == retries:
                            raise
                        time.sleep(attempt * 2)
            consensus = (
                passes[0]["material"] == passes[1]["material"]
                and passes[0]["single_object"] == passes[1]["single_object"]
                and passes[0]["foreign_material"] == passes[1]["foreign_material"]
            )
            result = {
                "sha256": row["sha256"],
                "image_path": row["image_path"],
                "model": model,
                "passes": passes,
                "consensus": consensus,
                "minimum_confidence": min(float(item["confidence"]) for item in passes),
                "deployed": row.get("deployed"),
                "verifier": row.get("verifier"),
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(f"[{index}/{len(queue)}] {row['sha256'][:12]} consensus={consensus}", flush=True)

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "queued": len(queue),
        "completed": len(rows),
        "consensus": sum(row["consensus"] for row in rows),
        "high_confidence_consensus": sum(
            row["consensus"] and row["minimum_confidence"] >= 0.8 for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3-vl:8b")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    summary = label_queue(
        args.queue,
        args.output,
        url=args.url,
        model=args.model,
        timeout=args.timeout,
        retries=args.retries,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
