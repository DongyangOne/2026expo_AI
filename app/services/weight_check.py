"""
무게 이상 감지.

bbox 크기로 용량을 추정해 '빈 용기 표준무게 + margin' 과 비교 → 초과 시 내용물/이물질 간주.
bbox 미제공 시 품목별 단일 상한(_MAX_FALLBACK_G)으로 폴백.

⚠️ 캘리브레이션 필요 (키오스크 셋업 후) — docs/WEIGHT_KIOSK_PARAMS.md:
   - 카메라-물체 거리 고정 가정 → bbox 면적비가 실제 크기에 비례
   - 대표 샘플로 _EMPTY_WEIGHT_G · _SIZE_*_MAX 실측 보정
   - 저울 정밀도 확정 후 _MARGIN_G 조정
"""

from typing import Optional

# 빈 용기 표준무게 (g) — 크기 버킷(S/M/L)별. 키오스크 대표샘플 실측으로 보정.
_EMPTY_WEIGHT_G: dict[str, dict[str, float]] = {
    "pet":     {"S": 15.0, "M": 22.0, "L": 45.0},   # 350ml / 500ml / 1.5~2L
    "can":     {"S": 14.0, "M": 20.0, "L": 40.0},   # 알루미늄 / 중형 / 철캔
    "plastic": {"S": 10.0, "M": 30.0, "L": 80.0},   # 용기류
    "paper":   {"S": 5.0,  "M": 40.0, "L": 200.0},  # 낱장 / 박스류
}

# bbox 면적의 이미지 대비 비율 → 크기 버킷 경계. 카메라 고정 가정, 캘리브레이션 필요.
_SIZE_S_MAX = 0.10   # 면적비 < 0.10 → S
_SIZE_M_MAX = 0.28   # < 0.28 → M, 그 이상 → L

# margin: 저울오차 + 라벨/뚜껑 + 잔여물 허용 (g). 저울 정밀도 확정 후 조정.
_MARGIN_G = 15.0

# bbox 미제공 시 폴백 상한 (g) — 크기 무관 보수적 단일 임계.
_MAX_FALLBACK_G: dict[str, float] = {
    "pet": 300.0, "can": 200.0, "plastic": 500.0, "paper": 2000.0,
    "styrofoam": 500.0, "vinyl": 400.0, "glass": 1000.0,
    "battery": 200.0, "fluorescent": 500.0,
}


def _size_bucket(bbox: list[float], img_area: float) -> str:
    """bbox 면적의 이미지 대비 비율 → S/M/L."""
    x1, y1, x2, y2 = bbox
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ratio = area / img_area if img_area > 0 else 0.0
    if ratio < _SIZE_S_MAX:
        return "S"
    if ratio < _SIZE_M_MAX:
        return "M"
    return "L"


def expected_empty_weight(class_name: str, bbox: list[float], img_area: float) -> Optional[float]:
    """bbox 크기로 추정한 빈 용기 표준무게 (g). 표 없으면 None."""
    table = _EMPTY_WEIGHT_G.get(class_name)
    if table is None:
        return None
    return table[_size_bucket(bbox, img_area)]


def is_anomaly(
    class_name: str,
    weight_g: float,
    bbox: Optional[list[float]] = None,
    img_area: Optional[float] = None,
) -> bool:
    """
    True → 무게가 정상 범위 초과 (내용물 있음 / 이물질 포함 간주).
    bbox+img_area 제공 시 크기보정, 아니면 품목별 단일 상한으로 폴백.
    """
    if bbox is not None and img_area:
        expected = expected_empty_weight(class_name, bbox, img_area)
        if expected is not None:
            return weight_g > expected + _MARGIN_G

    max_g = _MAX_FALLBACK_G.get(class_name)
    if max_g is None:
        return False
    return weight_g > max_g
