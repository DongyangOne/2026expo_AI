import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_VERIFIER_OUTPUT_NAMES = frozenset(
    {"material", "dent", "label", "foreign_material"}
)
_VERIFIER_LEGACY_CLASSES = (
    "can", "pet", "paper", "plastic", "styrofoam",
    "vinyl", "glass", "battery", "fluorescent",
)
_VERIFIER_ALLOWED_CLASS_CONTRACTS = frozenset(
    {_VERIFIER_LEGACY_CLASSES, (*_VERIFIER_LEGACY_CLASSES, "background")}
)


@dataclass(frozen=True)
class VerifierRuntime:
    """ONNX 세션과 학습 산출물 metadata를 함께 보존한다."""

    session: object
    class_names: tuple[str, ...]
    enabled_outputs: frozenset[str]
    metadata_path: Path


class ModelRegistry:
    """
    모델 보관소.
    lifespan 에서 인스턴스 생성 → app.state.registry 에 저장 → Depends()로 주입.
      - main : YOLO 9-class 감지
      - state   : 기존 상태 멀티헤드(dent/label) ONNX
      - verifier: YOLO bbox의 9종+상태를 재검증하는 320px ONNX (초기 shadow)
    """

    def __init__(
        self,
        main_path: str,
        state_path: Optional[str] = None,
        verifier_path: Optional[str] = None,
    ) -> None:
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

        self._verifier = None
        if verifier_path:
            if _check_path(verifier_path, required=False):
                try:
                    metadata = _load_verifier_metadata(verifier_path)
                    import onnxruntime as ort
                    session = ort.InferenceSession(
                        verifier_path, providers=["CPUExecutionProvider"]
                    )
                    if metadata is None:
                        # sidecar가 없는 기존 9-class 모델은 원래의 4-head
                        # 런타임 계약을 그대로 사용한다.
                        self._verifier = session
                    else:
                        class_names, enabled_outputs, metadata_path = metadata
                        available = {output.name for output in session.get_outputs()}
                        missing = enabled_outputs - available
                        if missing:
                            raise ValueError(
                                "verifier metadata enabled_outputs와 ONNX 출력 불일치: "
                                f"{sorted(missing)}"
                            )
                        self._verifier = VerifierRuntime(
                            session=session,
                            class_names=class_names,
                            enabled_outputs=enabled_outputs,
                            metadata_path=metadata_path,
                        )
                    logger.info("crop 검증기 로드(shadow): %s", verifier_path)
                except Exception as exc:
                    # 선택적 shadow 후보의 metadata가 손상됐을 때 API 시작이나
                    # 운영 응답에 영향을 주지 않고 후보만 비활성화한다.
                    logger.error(
                        "crop 검증기 metadata 오류로 shadow 비활성: %s", exc
                    )
            else:
                logger.warning("crop 검증기 파일 없음 (경로: %s) — shadow 비활성", verifier_path)

    def main(self):
        if self._main is None:
            raise RuntimeError("주 모델이 로드되지 않았습니다. 서버 로그를 확인하세요.")
        return self._main

    def state(self):
        """상태 멀티헤드 ONNX 세션 (미탑재 시 None)."""
        return self._state

    def verifier(self):
        """YOLO 뒤에서 비동기로 실행하는 crop 검증기 ONNX 세션."""
        return self._verifier

    def status(self) -> dict:
        return {
            "main": self._main is not None,
            "state": self._state is not None,
            "verifier": self._verifier is not None,
        }


def _check_path(path: str, *, required: bool) -> bool:
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(f"모델 경로를 찾을 수 없습니다: {path}")
        return False
    return True


def _load_verifier_metadata(
    verifier_path: str,
) -> tuple[tuple[str, ...], frozenset[str], Path] | None:
    """학습기가 ONNX 옆에 기록한 metadata를 검증해 읽는다.

    sidecar가 없으면 기존 9-class/4-head 모델로 간주하기 위해 ``None``을
    반환한다. sidecar가 존재하지만 계약이 잘못된 경우에는 안전하게 후보를
    비활성화할 수 있도록 예외를 발생시킨다.
    """
    metadata_path = Path(verifier_path).with_name("verifier_metadata.json")
    if not metadata_path.is_file():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"verifier metadata 최상위 형식 오류: {metadata_path}")
    classes = metadata.get("classes")
    enabled_outputs = metadata.get("enabled_outputs")
    if (
        not isinstance(classes, list)
        or not classes
        or any(not isinstance(name, str) or not name for name in classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError(f"verifier classes metadata 오류: {metadata_path}")
    class_contract = tuple(classes)
    if class_contract not in _VERIFIER_ALLOWED_CLASS_CONTRACTS:
        raise ValueError(f"verifier class 순서/계약 오류: {metadata_path}")
    if (
        not isinstance(enabled_outputs, list)
        or "material" not in enabled_outputs
        or len(set(enabled_outputs)) != len(enabled_outputs)
        or any(name not in _VERIFIER_OUTPUT_NAMES for name in enabled_outputs)
    ):
        raise ValueError(f"verifier enabled_outputs metadata 오류: {metadata_path}")

    declared_count = metadata.get("material_class_count")
    if declared_count is not None and declared_count != len(classes):
        raise ValueError(f"verifier material_class_count metadata 오류: {metadata_path}")

    return class_contract, frozenset(enabled_outputs), metadata_path
