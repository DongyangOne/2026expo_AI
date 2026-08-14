"""
도메인 규칙: 분류(허용/일반/거부) 판정 · 안내 문구 생성.

파이프라인(흐름 제어)과 분리하여 "쓰레기 처리 규칙"만 모은다.
문구 수정 시 이 파일만 건드리면 된다.

하드웨어 함: [플라스틱, 캔, 종이, 일반] + 수거거부.
  - ALLOWED       : plastic(PET 포함)/can/paper + 상태 조건 충족 → 재활용 함
  - REJECTED(재처리): plastic(PET 포함)/can/paper + 조건 불충족 → guidance 안내 후 재투입
  - REJECTED(거부) : glass/battery/fluorescent/styrofoam → 수거 불가
  - ALLOWED(비닐)  : vinyl + 상태 조건 충족 → 비닐함
  - GENERAL_WASTE  : 저신뢰 / 미분류 → 분류 불가
"""

from app.schemas.enums import (
    GeneralWasteCode,
    GuidanceCode,
    RejectionCode,
    WasteClass,
)
from app.schemas.response import Conditions, GeneralWaste, Guidance, Rejection

# ── 품목 분류 ────────────────────────────────────────────────────────────────────
ALLOWED_CLASSES: set[WasteClass] = {
    WasteClass.PET, WasteClass.PLASTIC, WasteClass.CAN, WasteClass.PAPER,
}

REJECTED_CLASSES: set[WasteClass] = {
    WasteClass.GLASS, WasteClass.BATTERY,
    WasteClass.FLUORESCENT, WasteClass.STYROFOAM,
}

VINYL_CLASSES: set[WasteClass] = {
    WasteClass.VINYL,
}

# ── 안내 문구 사전 ────────────────────────────────────────────────────────────────
# 조건 불충족 시 재처리 안내 (있으면 REJECTED, 없으면 ALLOWED)
_GUIDANCE_TEXT: dict[GuidanceCode, str] = {
    GuidanceCode.EMPTY_CONTENTS:  "내용물이 남아 있거나 무게가 정상 범위를 벗어났어요. 내용물을 비우고 다시 넣어 주세요.",
    GuidanceCode.WEIGHT_ANOMALY:  "무게가 정상 범위를 벗어났어요. 확인하고 다시 넣어 주세요.",
    GuidanceCode.FOREIGN_MATERIAL: "외부 이물질을 제거하고 다시 넣어 주세요.",
    GuidanceCode.REMOVE_LABEL:    "라벨을 제거하고 다시 넣어 주세요.",
    GuidanceCode.COMPRESS:        "플라스틱 병·캔은 납작하게 압착해서 다시 넣어 주세요.",
}

_EMPTY_CONTENTS_CLASSES = {WasteClass.PET, WasteClass.PLASTIC, WasteClass.CAN}
_WEIGHT_ANOMALY_CLASSES = {WasteClass.PAPER, WasteClass.VINYL}

# 완전 수거 거부
_REJECTION: dict[WasteClass, tuple[RejectionCode, str]] = {
    WasteClass.GLASS:       (RejectionCode.GLASS,       "유리병은 깨질 위험이 있어요. 유리병 전용 분리수거함에 배출해 주세요."),
    WasteClass.BATTERY:     (RejectionCode.BATTERY,     "건전지는 일반 분리수거가 어려워요. 마트나 편의점의 건전지 수거함을 이용해 주세요."),
    WasteClass.FLUORESCENT: (RejectionCode.FLUORESCENT, "형광등은 특수 처리가 필요해요. 주민센터나 형광등 수거함에 배출해 주세요."),
    WasteClass.STYROFOAM:   (RejectionCode.STYROFOAM,   "스티로폼은 이 기기에서 처리하기 어려워요. 대형 스티로폼 전용 분리수거함을 이용해 주세요."),
}

# 일반쓰레기
_GENERAL_TEXT: dict[GeneralWasteCode, str] = {
    GeneralWasteCode.VINYL:          "비닐은 일반쓰레기로 배출해 주세요.",
    GeneralWasteCode.LOW_CONFIDENCE: "정확히 분류하기 어려워요. 일반쓰레기로 배출해 주세요.",
    GeneralWasteCode.UNCLASSIFIED:   "재활용 대상이 아니에요. 일반쓰레기로 배출해 주세요.",
}


# ── 판정 함수 ────────────────────────────────────────────────────────────────────
def is_allowed(cls: WasteClass) -> bool:
    return cls in ALLOWED_CLASSES


def is_rejected(cls: WasteClass) -> bool:
    return cls in REJECTED_CLASSES


def is_vinyl(cls: WasteClass) -> bool:
    return cls in VINYL_CLASSES


# ── 안내 생성 ────────────────────────────────────────────────────────────────────
def build_guidance(
    cls: WasteClass,
    conditions: Conditions,
    weight_anomaly: bool,
    has_foreign_material: bool | None = None,
) -> list[Guidance]:
    """
    허용 품목의 조건 불충족 항목을 재처리 안내로 생성.
    빈 리스트  → 모든 조건 충족 → ALLOWED
    비어있지않음 → 조건 불충족   → REJECTED (재처리 후 재투입)
    헤드 미대상/미탑재(None)은 검사하지 않음.
    우선순위: 무게 → 외부 이물질 → 라벨 → 압착.
    """
    codes: list[GuidanceCode] = []

    if weight_anomaly and cls in _EMPTY_CONTENTS_CLASSES:
        codes.append(GuidanceCode.EMPTY_CONTENTS)
    elif weight_anomaly and cls in _WEIGHT_ANOMALY_CLASSES:
        codes.append(GuidanceCode.WEIGHT_ANOMALY)
    if has_foreign_material is True:
        codes.append(GuidanceCode.FOREIGN_MATERIAL)
    if conditions.has_label is True:                          # 페트·플라스틱 라벨 부착
        codes.append(GuidanceCode.REMOVE_LABEL)
    if cls in (WasteClass.PET, WasteClass.CAN) and conditions.is_dented is False:  # 미압착
        codes.append(GuidanceCode.COMPRESS)

    return [Guidance(code=c, message=_GUIDANCE_TEXT[c]) for c in codes]


def build_rejection(cls: WasteClass) -> Rejection:
    """완전 수거 거부 품목(유리/건전지/형광등/스티로폼) 안내."""
    code, msg = _REJECTION[cls]
    return Rejection(code=code, message=msg)


def build_general(code: GeneralWasteCode) -> GeneralWaste:
    """일반쓰레기 안내. VINYL 코드는 기존 계약 호환을 위해 유지한다."""
    return GeneralWaste(code=code, message=_GENERAL_TEXT[code])
