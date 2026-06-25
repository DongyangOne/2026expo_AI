import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    모델 보관소.
    lifespan 에서 인스턴스 생성 → app.state.registry 에 저장 → Depends()로 주입.
      - main : YOLO 9-class 감지
      - state: 상태 멀티헤드(dent/label) ONNX. 미탑재 시 conditions=null (무게만 검사)
    """

    def __init__(self, main_path: str, state_path: Optional[str] = None) -> None:
        from ultralytics import YOLO

        _check_path(main_path, required=True)
        self._main = YOLO(main_path)
        logger.info("주 모델 로드: %s", main_path)

        self._state = None
        if state_path:
            if _check_path(state_path, required=False):
                import onnxruntime as ort
                self._state = ort.InferenceSession(state_path, providers=["CPUExecutionProvider"])
                logger.info("상태 멀티헤드 로드: %s", state_path)
            else:
                logger.warning("상태 멀티헤드 파일 없음 (경로: %s) — conditions=null 로 동작", state_path)
        else:
            logger.info("상태 멀티헤드 미설정 — conditions=null 로 동작")

    def main(self):
        if self._main is None:
            raise RuntimeError("주 모델이 로드되지 않았습니다. 서버 로그를 확인하세요.")
        return self._main

    def state(self):
        """상태 멀티헤드 ONNX 세션 (미탑재 시 None)."""
        return self._state

    def status(self) -> dict:
        return {"main": self._main is not None, "state": self._state is not None}


def _check_path(path: str, *, required: bool) -> bool:
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(f"모델 경로를 찾을 수 없습니다: {path}")
        return False
    return True
