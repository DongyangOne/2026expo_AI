"""임시 crop 검증기를 비동기로 실행하고 YOLO와의 비교 결과만 JSONL에 기록한다."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import numpy as np

from app.core.config import settings
from app.services import inference

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="verifier-shadow")
_write_lock = Lock()


def _run_and_log(
    session,
    img: np.ndarray,
    bbox: list[float],
    yolo_class_id: int,
    yolo_confidence: float,
    client_id: str,
) -> None:
    try:
        prediction = inference.run_verifier(session, img, bbox)
        if prediction is None:
            return
        material = prediction["material"]
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # 하드웨어 식별자는 생성·변환하지 않고 요청 값을 그대로 보존한다.
            "client_id": client_id,
            "bbox": [round(float(value), 1) for value in bbox],
            "yolo": {
                "class_id": yolo_class_id,
                "class_name": inference.VERIFIER_CLASS_NAMES[yolo_class_id]
                if 0 <= yolo_class_id < len(inference.VERIFIER_CLASS_NAMES) else None,
                "confidence": round(float(yolo_confidence), 6),
            },
            "verifier": prediction,
            "material_agreement": material["class_id"] == yolo_class_id,
            "mode": "shadow",
        }
        log_path = Path(settings.VERIFIER_SHADOW_LOG_PATH)
        with _write_lock:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # shadow 실패는 API 응답과 Spring 콜백에 절대 영향을 주지 않는다.
        logger.exception("crop verifier shadow 추론 실패")


def submit(
    session,
    img: np.ndarray,
    bbox: list[float],
    yolo_class_id: int,
    yolo_confidence: float,
    client_id: str,
) -> None:
    if not settings.VERIFIER_SHADOW_ENABLED or session is None:
        return
    try:
        _executor.submit(
            _run_and_log, session, img, bbox, yolo_class_id, yolo_confidence, client_id
        )
    except RuntimeError:
        logger.warning("종료 중이라 crop verifier shadow 작업을 생략합니다")


def shutdown() -> None:
    _executor.shutdown(wait=True)
