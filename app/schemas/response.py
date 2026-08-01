"""
응답 DTO (Spring 과의 계약서).

구조:
  DetectResponse
  ├─ client_id       : str               ← 하드웨어가 보낸 사용자/피드백 구분 ID
  ├─ status          : DetectionStatus   ← Spring 의 1차 분기 판별자
  ├─ classification  : Classification?   ← 분류 결과 (NOT_DETECTED 시 null)
  ├─ conditions      : Conditions        ← 상태 감지 (is_dented/has_label/has_foreign_material)
  ├─ weight          : WeightInfo        ← 무게 + 이상 여부
  ├─ guidance        : Guidance[]        ← 조건 불충족 시 재처리 안내 (압착/라벨/비우기). 충족 시 빈 배열
  ├─ rejection       : Rejection?        ← 완전 거부 사유 (유리/건전지 등)
  ├─ general         : GeneralWaste?     ← 일반쓰레기 사유 (저신뢰/미분류)
  └─ bbox            : float[]?          ← 감지 영역 [x1,y1,x2,y2]

status 별 채워지는 필드:
  ALLOWED        → classification, conditions, weight, guidance(빈 배열)
  REJECTED(재처리) → classification, conditions, weight, guidance(압착/라벨/비우기)
  REJECTED(완전거부) → classification, rejection
  GENERAL_WASTE  → classification, general
  NOT_DETECTED   → (없음)
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.enums import (
    DetectionStatus,
    GeneralWasteCode,
    GuidanceCode,
    RejectionCode,
    WasteClass,
)


class Classification(BaseModel):
    """외부 분류 결과. 모델 PET 출력은 PLASTIC(class_id=3)으로 통합한다."""
    class_id:   int        = Field(..., description="외부 클래스 ID (PET은 PLASTIC의 3으로 정규화)", ge=0, le=8)
    class_name: WasteClass = Field(..., description="주 분류 (PET은 plastic으로 정규화)")
    confidence: float      = Field(..., description="주 분류 신뢰도", ge=0.0, le=1.0)


class Conditions(BaseModel):
    """객체 상태 (멀티헤드). 모델 미탑재 또는 비대상 헤드는 null."""
    has_label: Optional[bool]            = Field(None, description="라벨 부착 여부 (페트·플라스틱)")
    is_dented: Optional[bool]            = Field(None, description="압착(찌그러짐) 여부 (페트·캔)")
    has_foreign_material: Optional[bool] = Field(None, description="외부 이물질 여부 (지원 모델 탑재 시)")


class WeightInfo(BaseModel):
    """무게 센서 값과 이상 여부."""
    value_g: Optional[float] = Field(None, description="입력 무게 (g). 미입력 시 null")
    anomaly: bool            = Field(False, description="정상 범위 초과 여부 (내용물/이물질 간주)")


class Guidance(BaseModel):
    """조건 불충족 시 재처리 안내. code 로 분기, message 로 표시."""
    code:    GuidanceCode = Field(..., description="처리 안내 코드 (분기용)")
    message: str          = Field(..., description="사용자 표시 문구")


class Rejection(BaseModel):
    """완전 수거 거부 안내 (유리/건전지/형광등/스티로폼)."""
    code:    RejectionCode = Field(..., description="거부 사유 코드")
    message: str           = Field(..., description="올바른 배출 방법 안내")


class GeneralWaste(BaseModel):
    """일반쓰레기 안내. VINYL 코드는 기존 계약 호환을 위해 유지한다."""
    code:    GeneralWasteCode = Field(..., description="일반쓰레기 사유 코드")
    message: str              = Field(..., description="안내 문구")


class DetectResponse(BaseModel):
    client_id:       str                      = Field(..., description="요청에서 받은 사용자/피드백 구분 ID")
    status:         DetectionStatus          = Field(..., description="처리 판별자 (Spring 1차 분기)")
    classification: Optional[Classification]  = Field(None, description="분류 결과 (감지 실패 시 null)")
    conditions:     Conditions               = Field(default_factory=Conditions)
    weight:         WeightInfo               = Field(default_factory=WeightInfo)
    guidance:       list[Guidance]           = Field(default_factory=list)
    rejection:      Optional[Rejection]      = Field(None, description="완전 거부 사유 (REJECTED 완전거부)")
    general:        Optional[GeneralWaste]   = Field(None, description="일반쓰레기 사유 (GENERAL_WASTE)")
    bbox:           Optional[list[float]]    = Field(None, description="감지 영역 [x1, y1, x2, y2] 픽셀")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "client_id": "hardware-user-001",
                    "status": "ALLOWED",
                    "classification": {"class_id": 3, "class_name": "plastic", "confidence": 0.94},
                    "conditions": {"has_label": False, "is_dented": True, "has_foreign_material": None},
                    "weight": {"value_g": 28.0, "anomaly": False},
                    "guidance": [],
                    "rejection": None,
                    "general": None,
                    "bbox": [120.0, 80.0, 410.0, 560.0],
                },
                {
                    "client_id": "hardware-user-001",
                    "status": "REJECTED",
                    "classification": {"class_id": 3, "class_name": "plastic", "confidence": 0.91},
                    "conditions": {"has_label": True, "is_dented": False, "has_foreign_material": True},
                    "weight": {"value_g": 540.0, "anomaly": True},
                    "guidance": [
                        {"code": "EMPTY_CONTENTS", "message": "내용물이 남아 있는 것 같아요. 비우고 다시 넣어 주세요."},
                        {"code": "FOREIGN_MATERIAL", "message": "외부 이물질을 제거하고 다시 넣어 주세요."},
                        {"code": "REMOVE_LABEL", "message": "라벨을 제거해 주세요."},
                        {"code": "COMPRESS", "message": "플라스틱 병·캔은 납작하게 압착해서 넣어 주세요."}
                    ],
                    "rejection": None,
                    "general": None,
                    "bbox": [120.0, 80.0, 410.0, 560.0],
                },
                {
                    "client_id": "hardware-user-001",
                    "status": "REJECTED",
                    "classification": {"class_id": 6, "class_name": "glass", "confidence": 0.88},
                    "weight": {"value_g": 320.0, "anomaly": False},
                    "guidance": [],
                    "rejection": {"code": "GLASS", "message": "유리병은 깨질 위험이 있어요. 유리병 전용 분리수거함에 배출해 주세요."},
                    "general": None,
                    "bbox": [90.0, 60.0, 500.0, 720.0],
                },
                {
                    "client_id": "hardware-user-001",
                    "status": "ALLOWED",
                    "classification": {"class_id": 5, "class_name": "vinyl", "confidence": 0.90},
                    "conditions": {"has_label": None, "is_dented": None, "has_foreign_material": None},
                    "weight": {"value_g": 12.0, "anomaly": False},
                    "guidance": [],
                    "rejection": None,
                    "general": None,
                    "bbox": [100.0, 110.0, 480.0, 470.0],
                },
                {
                    "client_id": "hardware-user-001",
                    "status": "NOT_DETECTED",
                    "classification": None,
                    "weight": {"value_g": None, "anomaly": False},
                    "guidance": [],
                    "rejection": None,
                    "general": None,
                    "bbox": None,
                },
            ]
        }
    )


# ── 에러 응답 ────────────────────────────────────────────────────────────────────
class ErrorDetail(BaseModel):
    code:    str = Field(..., description="에러 코드 (분기용)")
    message: str = Field(..., description="에러 설명")


class ErrorResponse(BaseModel):
    error: ErrorDetail

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"error": {"code": "INVALID_IMAGE", "message": "이미지 디코딩에 실패했습니다."}}]
        }
    )
