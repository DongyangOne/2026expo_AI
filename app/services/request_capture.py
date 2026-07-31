"""재학습·오인식 검수용 요청 이미지/판정 결과 저장."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.core.config import settings
from app.schemas.response import DetectResponse

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_last_prune_at = 0.0
_PRUNE_INTERVAL_SEC = 10 * 60


def _image_extension(image_bytes: bytes, content_type: str | None) -> str:
    """파일명 대신 magic bytes를 우선해 안전한 이미지 확장자를 선택한다."""
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if content_type == "image/jpeg":
        return ".jpg"
    if content_type == "image/png":
        return ".png"
    return ".img"


def _safe_original_filename(filename: str | None) -> str | None:
    if not filename:
        return None
    # Windows/POSIX 경로 구분자를 모두 제거하고 메타데이터 길이도 제한한다.
    return filename.replace("\\", "/").rsplit("/", 1)[-1][:255]


def _atomic_write(path: Path, content: bytes) -> None:
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with open(temp_path, "xb") as file:
            file.write(content)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _capture_groups(root: Path) -> list[tuple[float, int, list[Path]]]:
    """동일 stem의 이미지/JSON을 한 묶음으로 반환한다."""
    groups: dict[Path, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and not path.name.startswith("."):
            groups.setdefault(path.with_suffix(""), []).append(path)

    captures: list[tuple[float, int, list[Path]]] = []
    for paths in groups.values():
        try:
            stats = [path.stat() for path in paths]
        except FileNotFoundError:
            continue
        captures.append(
            (
                min(stat.st_mtime for stat in stats),
                sum(stat.st_size for stat in stats),
                paths,
            )
        )
    return captures


def _prune_if_due(root: Path) -> None:
    """보존 기간과 최대 용량을 넘긴 오래된 캡처 묶음을 제거한다."""
    global _last_prune_at

    now_mono = time.monotonic()
    if now_mono - _last_prune_at < _PRUNE_INTERVAL_SEC:
        return
    _last_prune_at = now_mono

    captures = _capture_groups(root)
    retention_days = settings.CAPTURE_RETENTION_DAYS
    if retention_days > 0:
        cutoff = time.time() - retention_days * 24 * 60 * 60
        for mtime, _, paths in captures:
            if mtime < cutoff:
                for path in paths:
                    path.unlink(missing_ok=True)
        captures = [capture for capture in captures if capture[0] >= cutoff]

    max_bytes = settings.CAPTURE_MAX_STORAGE_MB * 1024 * 1024
    if max_bytes <= 0:
        return

    total_bytes = sum(size for _, size, _ in captures)
    for _, size, paths in sorted(captures, key=lambda item: item[0]):
        if total_bytes <= max_bytes:
            break
        for path in paths:
            path.unlink(missing_ok=True)
        total_bytes -= size


def save_capture(
    *,
    image_bytes: bytes,
    original_filename: str | None,
    content_type: str | None,
    client_id: str,
    weight_g: float | None,
    result: DetectResponse,
) -> None:
    """원본 이미지와 판정 JSON을 같은 capture_id로 저장한다.

    백그라운드 작업에서 호출되며, 저장 실패는 API 응답이나 Spring 콜백에
    영향을 주지 않는다. API 키와 요청 헤더는 기록하지 않는다.
    """
    try:
        if not image_bytes:
            logger.warning("요청 캡처 생략: 빈 이미지")
            return
        if len(image_bytes) > settings.CAPTURE_MAX_IMAGE_BYTES:
            logger.warning(
                "요청 캡처 생략: 이미지가 최대 크기를 초과함 (%d > %d bytes)",
                len(image_bytes),
                settings.CAPTURE_MAX_IMAGE_BYTES,
            )
            return

        captured_at = datetime.now(timezone.utc)
        capture_id = f"{captured_at.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:12]}"
        root = Path(settings.CAPTURE_DIR)
        day_dir = root / captured_at.date().isoformat()
        extension = _image_extension(image_bytes, content_type)
        image_path = day_dir / f"{capture_id}{extension}"
        metadata_path = day_dir / f"{capture_id}.json"

        metadata = {
            "capture_id": capture_id,
            "timestamp": captured_at.isoformat(),
            "image": {
                "path": image_path.relative_to(root).as_posix(),
                "original_filename": _safe_original_filename(original_filename),
                "content_type": content_type,
                "size_bytes": len(image_bytes),
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
            },
            "request": {
                # 하드웨어가 전달한 값은 응답/콜백과 마찬가지로 그대로 보존한다.
                "client_id": client_id,
                "weight_g": weight_g,
            },
            "result": result.model_dump(mode="json"),
            "review": {
                "is_correct": None,
                "expected_class": None,
                "notes": None,
            },
        }
        metadata_bytes = json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8")

        with _lock:
            day_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write(image_path, image_bytes)
            try:
                _atomic_write(metadata_path, metadata_bytes)
            except Exception:
                image_path.unlink(missing_ok=True)
                raise
            _prune_if_due(root)
    except Exception:
        logger.exception("요청 이미지/판정 캡처 기록 실패")
