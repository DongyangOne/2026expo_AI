"""고해상도 원본 crop을 로컬 Qwen 멀티모달 모델로 검수해 상태 pseudo-label을 만든다.

원본 이미지나 고해상도 crop을 복제하지 않는다. ``manifest.csv``의
``source_path_b64``와 원본 bbox를 읽어 두 가지 crop을 메모리에서만 만든다.
판정 원문은 JSONL에 이어 쓰고, 확신도 기준을 통과한 값만 별도 manifest에 병합한다.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import cv2
import numpy as np

try:
    from extract_verifier_crops import decode_source_path
except ImportError:  # ``python -m scripts.pseudo_label_status_qwen`` 지원
    from scripts.extract_verifier_crops import decode_source_path


TRANSFORMERS_MODEL_DEFAULT = "Qwen/Qwen3-VL-4B-Instruct"
OLLAMA_MODEL_DEFAULT = "qwen3.5:9b-q4_K_M"
OLLAMA_NUM_CTX_DEFAULT = 16384
OLLAMA_IMAGE_MAX_SIDE_DEFAULT = 640
STATUS_CATEGORIES = {"can", "pet", "paper", "plastic", "vinyl"}
LABEL_CATEGORIES = {"pet", "plastic"}
DECISIONS = {"neither", "label_only", "foreign_only", "both", "exclude", "ambiguous"}
EVIDENCE_CODES = {
    "none", "label", "different_material", "contamination",
    "same_material_accessory", "multiple_items", "unclear",
}

OLLAMA_FORMAT = {
    "type": "object",
    "properties": {
        "d": {"type": "string", "enum": sorted(DECISIONS)},
        "s": {"type": "boolean"},
        "c": {"type": "number", "minimum": 0, "maximum": 1},
        "e": {"type": "string", "enum": sorted(EVIDENCE_CODES)},
    },
    "required": ["d", "s", "c", "e"],
    "additionalProperties": False,
}

PROMPT = """You are cleaning a recycling-kiosk image dataset.
The supplied image(s) show the SAME annotated object. View(s): {view_order}.
Known material category: {category}
Legacy DIRTINESS value (candidate hint only, NEVER ground truth): {raw_dirtiness}

Decide only from visible evidence using these exact rules:
1. A removable label is a detachable PET sleeve or sticker. Printed can graphics, ink, and logos are NOT labels.
2. True foreign material is a clearly different material attached to or mixed with the main item, or real food/soil/fabric/metal/paper contamination.
3. A straw, cap, lid, pull ring, or similar non-label accessory made of the same broad recyclable material as the main item is ALLOWED and is NOT true foreign material. Set same_material_accessory_only=true, but still use neither or label_only and keep the sample eligible.
4. Material must be compared with the main item. For example, a paper sleeve/band on a plastic or PET takeaway cup is true foreign material and must be foreign_only or both. A primary recyclable item with a visibly attached different material can still be a single primary item.
5. A removable product sleeve or sticker is always handled by has_removable_label, even when its material is similar to the main item.
6. If multiple separate waste objects are present, the crop is unclear, or evidence is insufficient, use ambiguous or exclude. Never guess.
7. decision fully determines the label/foreign flags; do not add redundant fields.

Return ONLY one compact JSON object. Keys: d=decision, s=single primary item,
c=confidence, e=visible evidence code. e must be one of none, label,
different_material, contamination, same_material_accessory, multiple_items, unclear.
{{"d":"neither|label_only|foreign_only|both|exclude|ambiguous","s":true,"c":0.99,"e":"none"}}
"""


def _imread_unicode(path: Path):
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def _crop(image: np.ndarray, row: dict, padding: float) -> np.ndarray | None:
    height, width = image.shape[:2]
    x = float(row["source_bbox_x"])
    y = float(row["source_bbox_y"])
    box_w = float(row["source_bbox_w"])
    box_h = float(row["source_bbox_h"])
    x1 = max(0, int(x - box_w * padding))
    y1 = max(0, int(y - box_h * padding))
    x2 = min(width, int(x + box_w * (1 + padding)))
    y2 = min(height, int(y + box_h * (1 + padding)))
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)


def parse_teacher_output(text: str) -> dict:
    """모델 출력을 검증하고 결정과 불리언 조합이 일치할 때만 반환한다."""
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1]).strip()
    result = json.loads(value)
    if set(result) == {"d", "s", "c", "e"}:
        if result["e"] not in EVIDENCE_CODES:
            raise ValueError(f'invalid evidence code: {result["e"]}')
        label_flag, foreign_flag = {
            "neither": (False, False),
            "label_only": (True, False),
            "foreign_only": (False, True),
            "both": (True, True),
            "exclude": (None, None),
            "ambiguous": (None, None),
        }[result["d"]]
        result = {
            "decision": result["d"],
            "has_removable_label": label_flag,
            "has_true_foreign_material": foreign_flag,
            "same_material_accessory_only": result["e"] == "same_material_accessory",
            "is_single_primary_item": result["s"],
            "confidence": result["c"],
            "reason": result["e"],
        }
    if set(result) == {"d", "l", "f", "a", "s", "c", "e"}:
        if result["e"] not in EVIDENCE_CODES:
            raise ValueError(f'invalid evidence code: {result["e"]}')
        label_flag = result["l"]
        foreign_flag = result["f"]
        if label_flag is None or foreign_flag is None:
            if not (
                label_flag is None
                and foreign_flag is None
                and result["d"] in {"exclude", "ambiguous"}
            ):
                raise ValueError("compact null flags require exclude or ambiguous")
            decision = result["d"]
        else:
            if not isinstance(label_flag, bool) or not isinstance(foreign_flag, bool):
                raise ValueError("compact label/foreign flags must be bool or null")
            decision = {
                (False, False): "neither",
                (True, False): "label_only",
                (False, True): "foreign_only",
                (True, True): "both",
            }[(label_flag, foreign_flag)]
        result = {
            "decision": decision,
            "has_removable_label": label_flag,
            "has_true_foreign_material": foreign_flag,
            "same_material_accessory_only": result["a"],
            "is_single_primary_item": result["s"],
            "confidence": result["c"],
            "reason": result["e"],
        }
    required = {
        "decision", "has_removable_label", "has_true_foreign_material",
        "same_material_accessory_only", "is_single_primary_item", "confidence", "reason",
    }
    if set(result) != required:
        raise ValueError(f"unexpected fields: {sorted(set(result) ^ required)}")
    decision = result["decision"]
    if decision not in DECISIONS:
        raise ValueError(f"invalid decision: {decision}")
    if not isinstance(result["same_material_accessory_only"], bool):
        raise ValueError("same_material_accessory_only must be bool")
    if not isinstance(result["is_single_primary_item"], bool):
        raise ValueError("is_single_primary_item must be bool")
    if not isinstance(result["reason"], str):
        raise ValueError("reason must be str")
    confidence = result["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be numeric")
    if not 0 <= float(confidence) <= 1:
        raise ValueError("confidence outside 0..1")

    expected = {
        "neither": (False, False),
        "label_only": (True, False),
        "foreign_only": (False, True),
        "both": (True, True),
        "exclude": (None, None),
        "ambiguous": (None, None),
    }[decision]
    actual = (result["has_removable_label"], result["has_true_foreign_material"])
    if actual != expected:
        raise ValueError(f"decision/flags mismatch: {decision} vs {actual}")
    return result


def accepted_status(result: dict, category: str, min_confidence: float) -> dict:
    """학습에 넣을 수 있는 상태만 0/1로 바꾸고 나머지는 -1로 마스킹한다."""
    accepted = (
        result["decision"] not in {"exclude", "ambiguous"}
        and result["is_single_primary_item"]
        and float(result["confidence"]) >= min_confidence
    )
    if not accepted:
        return {"label": -1, "foreign_material": -1, "status_eligible": 0}
    label = int(result["has_removable_label"]) if category in LABEL_CATEGORIES else -1
    foreign = int(result["has_true_foreign_material"])
    return {"label": label, "foreign_material": foreign, "status_eligible": 1}


def consensus_teacher(results: list[dict]) -> dict:
    """여러 시야 판정이 완전히 일치할 때만 하나의 자동 teacher 정답으로 만든다."""
    if not results:
        raise ValueError("no teacher results")
    keys = (
        "decision", "has_removable_label", "has_true_foreign_material",
        "is_single_primary_item",
    )
    signature = tuple(results[0][key] for key in keys)
    if any(tuple(result[key] for key in keys) != signature for result in results[1:]):
        return {
            "decision": "ambiguous",
            "has_removable_label": None,
            "has_true_foreign_material": None,
            "same_material_accessory_only": False,
            "is_single_primary_item": all(
                result["is_single_primary_item"] for result in results
            ),
            "confidence": min(float(result["confidence"]) for result in results),
            "reason": "automatic view consensus failed",
        }
    merged = dict(results[0])
    merged["same_material_accessory_only"] = any(
        result["same_material_accessory_only"] for result in results
    )
    merged["confidence"] = min(float(result["confidence"]) for result in results)
    merged["reason"] = " | ".join(result["reason"] for result in results)
    return merged


def _load_results(path: Path) -> dict[tuple[str, str], dict]:
    results = {}
    if not path.exists():
        return results
    with open(path, encoding="utf-8") as file:
        for line in file:
            try:
                item = json.loads(line)
                results[(item["source_id"], item["filepath"])] = item
            except (json.JSONDecodeError, KeyError):
                continue
    return results


def _select_candidates(
    base_rows: list[dict],
    previous: dict[tuple[str, str], dict],
    split: str = "all",
    retry_errors_only: bool = False,
) -> list[dict]:
    candidates = []
    for row in base_rows:
        if row["category"] not in STATUS_CATEGORIES:
            continue
        if split != "all" and row["split"] != split:
            continue
        key = (row["source_id"], row["filepath"])
        prior = previous.get(key)
        if retry_errors_only:
            if prior and prior.get("error"):
                candidates.append(row)
        elif prior is None:
            candidates.append(row)
    return _round_robin(candidates)


def _round_robin(rows: list[dict]) -> list[dict]:
    """작은 prototype에서도 split/품목/오염 힌트를 균형 있게 섞는다."""
    groups: dict[tuple[str, str, str], deque[dict]] = defaultdict(deque)
    for row in rows:
        groups[(row["split"], row["category"], row.get("raw_dirtiness", ""))].append(row)
    ordered = []
    base_keys = sorted({(split, category) for split, category, _ in groups})
    dirtiness_orders = {}
    cursors = {}
    for index, base in enumerate(base_keys):
        values = sorted(key[2] for key in groups if key[:2] == base)
        offset = index % len(values)
        dirtiness_orders[base] = values[offset:] + values[:offset]
        cursors[base] = 0

    remaining = sum(len(value) for value in groups.values())
    while remaining:
        progressed = False
        for base in base_keys:
            values = dirtiness_orders[base]
            for _ in values:
                cursor = cursors[base] % len(values)
                cursors[base] += 1
                key = (*base, values[cursor])
                if groups[key]:
                    ordered.append(groups[key].popleft())
                    remaining -= 1
                    progressed = True
                    break
        if not progressed:
            raise RuntimeError("round-robin candidate accounting mismatch")
    return ordered


def merge_manifest(
    base_rows: list[dict], results: dict[tuple[str, str], dict], output_path: Path
) -> None:
    extra_fields = [
        "status_eligible", "teacher_status", "teacher_confidence", "teacher_reason",
        "teacher_model", "teacher_rejected",
    ]
    fieldnames = list(base_rows[0]) + [name for name in extra_fields if name not in base_rows[0]]
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for base in base_rows:
            row = dict(base)
            result = results.get((base["source_id"], base["filepath"]))
            if result:
                row["label"] = result["accepted"]["label"]
                row["foreign_material"] = result["accepted"]["foreign_material"]
                row["status_eligible"] = result["accepted"]["status_eligible"]
                teacher = result.get("teacher") or {}
                row["teacher_status"] = teacher.get("decision", "error")
                row["teacher_confidence"] = teacher.get("confidence", "")
                row["teacher_reason"] = teacher.get("reason", result.get("error", ""))
                row["teacher_model"] = result["model"]
                row["teacher_rejected"] = int(
                    result["accepted"]["status_eligible"] != 1
                )
            else:
                row["status_eligible"] = ""
                row["teacher_rejected"] = ""
            writer.writerow(row)


def _load_model(model_name: str):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, dtype="auto", device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    return model, processor, torch


def _infer_transformers(
    model, processor, torch, tight: np.ndarray, context: np.ndarray, prompt: str
) -> str:
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.fromarray(tight)},
                {"type": "image", "image": Image.fromarray(context)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=220, do_sample=False)
    trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def _jpeg_b64(image: np.ndarray) -> str:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("in-memory image encoding failed")
    return base64.b64encode(encoded).decode("ascii")


def _resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    """시각 증거는 보존하면서 VLM 이미지 토큰과 처리 시간을 제한한다."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _infer_ollama(
    base_url: str,
    model_name: str,
    first: np.ndarray,
    second: np.ndarray | None,
    prompt: str,
    timeout: int,
    num_ctx: int,
    image_max_side: int,
    request_retries: int,
) -> str:
    first = _resize_max_side(first, image_max_side)
    images = [_jpeg_b64(first)]
    if second is not None:
        second = _resize_max_side(second, image_max_side)
        images.append(_jpeg_b64(second))
    payload = {
        "model": model_name,
        "stream": False,
        "think": False,
        "format": OLLAMA_FORMAT,
        "keep_alive": "30m",
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": images,
            }
        ],
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": num_ctx,
            "num_predict": 256,
        },
    }
    result = None
    for attempt in range(request_retries + 1):
        request = urllib.request.Request(
            base_url.rstrip("/") + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code < 500 or attempt >= request_retries:
                raise RuntimeError(f"Ollama HTTP {error.code}: {body[:2000]}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt >= request_retries:
                raise RuntimeError(f"Ollama request failed: {error}") from error
        time.sleep(min(8, 2 ** attempt))
    if result is None:
        raise RuntimeError("Ollama request produced no result")
    message = result["message"]
    content = message.get("content", "")
    if not content.strip():
        thinking = message.get("thinking", "")
        raise ValueError(
            "empty Ollama content; "
            f"done_reason={result.get('done_reason', '')}; "
            f"thinking={thinking[:1200]!r}"
        )
    return content


def needs_adaptive_second_pass(
    teacher: dict,
    row: dict,
    confidence_threshold: float,
    negative_audit_rate: float,
) -> bool:
    """양성·불확실 샘플은 재검증하고, 확실한 정상은 감사분만 재검증한다."""
    decision = teacher["decision"]
    label_irrelevant = (
        decision == "label_only" and row.get("category") not in LABEL_CATEGORIES
    )
    if decision != "neither" and not label_irrelevant:
        return True
    if not teacher["is_single_primary_item"]:
        return True
    if float(teacher["confidence"]) < confidence_threshold:
        return True
    if negative_audit_rate <= 0:
        return False
    key = f'{row["source_id"]}|{row["filepath"]}'.encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    return bucket < negative_audit_rate


def ollama_url_for_row(ollama_urls: str, row: dict) -> str:
    """같은 샘플의 모든 pass를 한 Ollama 인스턴스에 고정해 분산한다."""
    urls = [url.strip().rstrip("/") for url in ollama_urls.split(",") if url.strip()]
    if not urls:
        raise ValueError("at least one Ollama URL is required")
    key = f'{row["source_id"]}|{row["filepath"]}'.encode("utf-8")
    index = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(urls)
    return urls[index]


def _process_candidate(
    row: dict,
    args,
    model=None,
    processor=None,
    torch=None,
    ollama_url_override: str | None = None,
) -> dict:
    record = {
        "source_id": row["source_id"],
        "filepath": row["filepath"],
        "split": row["split"],
        "category": row["category"],
        "model": args.model,
    }
    raw_outputs = []
    try:
        image_path = decode_source_path(row["source_path_b64"])
        image = _imread_unicode(image_path)
        if image is None:
            raise ValueError("source image unreadable")
        tight = _crop(image, row, padding=0.08)
        context = _crop(image, row, padding=0.30)
        if tight is None or context is None:
            raise ValueError("invalid source bbox")
        if args.adaptive_consensus:
            view_orders = [
                (tight, None, "one tight crop for the fast first pass"),
                (context, None, "one wider context crop for the verification pass"),
            ]
        else:
            view_orders = [
                (tight, context, "tight crop first, wider context crop second"),
                (context, tight, "wider context crop first, tight crop second"),
            ]
        teacher_passes = []
        second_pass = args.consensus_passes == 2 and not args.adaptive_consensus
        ollama_url = (
            ollama_url_override or ollama_url_for_row(args.ollama_url, row)
            if args.backend == "ollama" else None
        )
        for pass_index, (first, second, view_order) in enumerate(view_orders):
            if pass_index == 1 and not second_pass:
                break
            prompt = PROMPT.format(
                view_order=view_order,
                category=row["category"],
                raw_dirtiness=row.get("raw_dirtiness", ""),
            )
            if args.backend == "ollama":
                max_side = (
                    args.adaptive_first_image_max_side
                    if args.adaptive_consensus and pass_index == 0
                    else args.image_max_side
                )
                raw = _infer_ollama(
                    ollama_url, args.model, first, second,
                    prompt, args.request_timeout, args.num_ctx,
                    max_side, args.request_retries,
                )
            else:
                raw = _infer_transformers(
                    model, processor, torch, first, second, prompt
                )
            raw_outputs.append(raw)
            teacher_passes.append(parse_teacher_output(raw))
            if pass_index == 0 and args.adaptive_consensus and args.consensus_passes == 2:
                second_pass = needs_adaptive_second_pass(
                    teacher_passes[0], row, args.adaptive_confidence,
                    args.adaptive_negative_audit_rate,
                )
        record["raw_outputs"] = raw_outputs
        teacher = consensus_teacher(teacher_passes)
        record["teacher_passes"] = teacher_passes
        record["teacher"] = teacher
        record["adaptive"] = {
            "enabled": args.adaptive_consensus,
            "second_pass": len(teacher_passes) == 2,
            "first_image_max_side": (
                args.adaptive_first_image_max_side
                if args.adaptive_consensus else args.image_max_side
            ),
            "second_image_max_side": args.image_max_side,
        }
        record["accepted"] = accepted_status(
            teacher, row["category"], args.min_confidence
        )
    except Exception as error:
        record["raw_outputs"] = raw_outputs
        record["teacher"] = None
        record["accepted"] = {
            "label": -1, "foreign_material": -1, "status_eligible": 0,
        }
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def _processed_records(candidates, args, model=None, processor=None, torch=None):
    if args.workers == 1:
        for row in candidates:
            yield _process_candidate(row, args, model, processor, torch)
        return

    iterator = iter(candidates)
    ollama_urls = [
        url.strip().rstrip("/")
        for url in args.ollama_url.split(",")
        if url.strip()
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {}
        for worker_index in range(args.workers):
            try:
                row = next(iterator)
            except StopIteration:
                break
            ollama_url = (
                ollama_urls[worker_index % len(ollama_urls)]
                if args.backend == "ollama" else None
            )
            pending[executor.submit(
                _process_candidate, row, args, model, processor, torch,
                ollama_url,
            )] = ollama_url
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                ollama_url = pending.pop(future)
                yield future.result()
                try:
                    row = next(iterator)
                except StopIteration:
                    continue
                pending[executor.submit(
                    _process_candidate, row, args, model, processor, torch,
                    ollama_url,
                )] = ollama_url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--backend", choices=("ollama", "transformers"), default="ollama")
    parser.add_argument("--model")
    parser.add_argument(
        "--ollama-url", default="http://127.0.0.1:11434",
        help="Ollama URL. 여러 인스턴스는 쉼표로 구분하며 샘플 단위로 분산합니다.",
    )
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--num-ctx", type=int, default=OLLAMA_NUM_CTX_DEFAULT)
    parser.add_argument(
        "--image-max-side", type=int, default=OLLAMA_IMAGE_MAX_SIDE_DEFAULT,
        help="Ollama로 보내기 전 tight/context 각각의 최대 변 길이.",
    )
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-jsonl")
    parser.add_argument("--merged-manifest")
    parser.add_argument("--min-confidence", type=float, default=0.90)
    parser.add_argument(
        "--consensus-passes", type=int, choices=(1, 2), default=2,
        help="1=한 번, 2=tight/context 순서를 바꿔 두 판정이 일치할 때만 채택.",
    )
    parser.add_argument(
        "--adaptive-consensus", action="store_true",
        help="384px 1차 후 양성·불확실 샘플과 음성 감사분만 640px 2차 합의를 수행합니다.",
    )
    parser.add_argument("--adaptive-confidence", type=float, default=0.90)
    parser.add_argument("--adaptive-first-image-max-side", type=int, default=384)
    parser.add_argument("--adaptive-negative-audit-rate", type=float, default=0.10)
    parser.add_argument("--limit", type=int, default=0, help="새로 판정할 최대 행 수. 0이면 전체.")
    parser.add_argument("--split", choices=("all", "training", "validation"), default="all")
    parser.add_argument(
        "--retry-errors-only", action="store_true",
        help="기존 JSONL에서 error로 끝난 행만 다시 판정하고 최신 결과를 덧붙인다.",
    )
    args = parser.parse_args()
    if args.num_ctx < 8192:
        raise SystemExit("[ERROR] --num-ctx must be at least 8192 for two images")
    if args.image_max_side < 320:
        raise SystemExit("[ERROR] --image-max-side must be at least 320")
    if args.request_retries < 0:
        raise SystemExit("[ERROR] --request-retries must be non-negative")
    if args.workers < 1:
        raise SystemExit("[ERROR] --workers must be at least 1")
    if args.backend == "transformers" and args.workers != 1:
        raise SystemExit("[ERROR] transformers backend requires --workers 1")
    if args.backend == "transformers" and args.adaptive_consensus:
        raise SystemExit("[ERROR] adaptive consensus currently requires Ollama")
    if not 0 <= args.adaptive_confidence <= 1:
        raise SystemExit("[ERROR] --adaptive-confidence must be between 0 and 1")
    if args.adaptive_first_image_max_side < 320:
        raise SystemExit("[ERROR] --adaptive-first-image-max-side must be at least 320")
    if not 0 <= args.adaptive_negative_audit_rate <= 1:
        raise SystemExit("[ERROR] --adaptive-negative-audit-rate must be between 0 and 1")
    if args.model is None:
        args.model = (
            OLLAMA_MODEL_DEFAULT
            if args.backend == "ollama"
            else TRANSFORMERS_MODEL_DEFAULT
        )

    manifest_path = Path(args.manifest)
    output_jsonl = Path(args.output_jsonl or manifest_path.parent / "pseudo_status.jsonl")
    merged_manifest = Path(
        args.merged_manifest or manifest_path.parent / "manifest_with_pseudo_status.csv"
    )
    with open(manifest_path, encoding="utf-8") as file:
        base_rows = list(csv.DictReader(file))
    if not base_rows:
        raise SystemExit("[ERROR] empty manifest")
    required = {
        "source_path_b64", "source_bbox_x", "source_bbox_y",
        "source_bbox_w", "source_bbox_h",
    }
    missing = required - set(base_rows[0])
    if missing:
        raise SystemExit(f"[ERROR] v3 source reference fields missing: {sorted(missing)}")

    previous = _load_results(output_jsonl)
    candidates = _select_candidates(
        base_rows, previous, args.split, args.retry_errors_only
    )
    if args.limit > 0:
        candidates = candidates[:args.limit]
    print(
        f"대상: new={len(candidates):,}, resumed={len(previous):,}, "
        f"model={args.model}, min_confidence={args.min_confidence:.2f}",
        flush=True,
    )
    if not candidates:
        merge_manifest(base_rows, previous, merged_manifest)
        print(f"병합 완료 → {merged_manifest}", flush=True)
        return

    if args.backend == "transformers":
        model, processor, torch = _load_model(args.model)
    else:
        model = processor = torch = None
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_jsonl, "a", encoding="utf-8") as output:
            records = _processed_records(
                candidates, args, model, processor, torch
            )
            for index, record in enumerate(records, 1):
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                previous[(record["source_id"], record["filepath"])] = record
                if index % 10 == 0 or index == len(candidates):
                    print(f"  진행 {index:,}/{len(candidates):,}", flush=True)
    finally:
        merge_manifest(base_rows, previous, merged_manifest)
        print(f"결과: {output_jsonl}\n병합: {merged_manifest}", flush=True)


if __name__ == "__main__":
    main()
