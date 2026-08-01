"""
응답 계약(Contract)에 쓰이는 모든 열거형.

raw string 대신 enum 을 써서
  1) 정의되지 않은 값이 응답에 새어나가는 것을 차단
  2) Spring 이 코드로 분기 가능 (한국어 텍스트에 비종속)
  3) /docs 스키마에 가능한 값이 전부 노출
"""

from enum import Enum


class WasteClass(str, Enum):
    """주 모델 9-class. PET 감지는 외부 응답에서 PLASTIC으로 정규화한다."""
    CAN         = "can"
    PET         = "pet"
    PAPER       = "paper"
    PLASTIC     = "plastic"
    STYROFOAM   = "styrofoam"
    VINYL       = "vinyl"
    GLASS       = "glass"
    BATTERY     = "battery"
    FLUORESCENT = "fluorescent"


class DetectionStatus(str, Enum):
    """
    Spring 이 가장 먼저 분기하는 판별자(discriminator).
    이 값 하나로 후속 처리 분기가 결정된다.
    하드웨어 함: [플라스틱, 캔, 종이, 일반] + 수거거부.
    """
    ALLOWED       = "ALLOWED"        # 지정 함 투입 허용 (plastic/can/paper/vinyl) + 상태조건 충족
    GENERAL_WASTE = "GENERAL_WASTE"  # 저신뢰 / 미분류
    REJECTED      = "REJECTED"       # 거부: 조건불충족(재처리 guidance) 또는 완전거부(유리 등 rejection)
    NOT_DETECTED  = "NOT_DETECTED"   # 감지 실패 → 재시도 안내


class GuidanceCode(str, Enum):
    """허용 품목의 조건 불충족 시 재처리 안내 코드."""
    EMPTY_CONTENTS  = "EMPTY_CONTENTS"   # 내용물 비우기 (페트·플라스틱·캔 무게 이상)
    WEIGHT_ANOMALY  = "WEIGHT_ANOMALY"   # 무게 이상 (종이·비닐)
    FOREIGN_MATERIAL = "FOREIGN_MATERIAL"  # 외부 이물질 제거
    REMOVE_LABEL    = "REMOVE_LABEL"     # 라벨 제거 (페트·플라스틱)
    COMPRESS        = "COMPRESS"         # 압착 (페트·캔 미압착)


class RejectionCode(str, Enum):
    """수거 거부(REJECTED) 사유 코드."""
    GLASS       = "GLASS"
    BATTERY     = "BATTERY"
    FLUORESCENT = "FLUORESCENT"
    STYROFOAM   = "STYROFOAM"


class GeneralWasteCode(str, Enum):
    """일반쓰레기(GENERAL_WASTE) 사유 코드."""
    VINYL       = "VINYL"            # 하위 호환용 (정상 비닐은 ALLOWED)
    LOW_CONFIDENCE = "LOW_CONFIDENCE"  # 신뢰도 미달
    UNCLASSIFIED   = "UNCLASSIFIED"    # 기타 미분류
