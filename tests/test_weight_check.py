"""weight_check 단위 테스트 — bbox 크기보정, 폴백 단일 임계."""

from app.services import weight_check as wc

_IMG = 1000.0 * 1000.0  # 1000x1000 기준 이미지 면적


# ── 크기 버킷 ───────────────────────────────────────────────────────────────────
class TestSizeBucket:
    def test_small(self):
        # 100x100 / 1e6 = 0.01 < 0.10 → S
        assert wc._size_bucket([0, 0, 100, 100], _IMG) == "S"

    def test_medium(self):
        # 400x400 / 1e6 = 0.16 → M (0.10~0.28)
        assert wc._size_bucket([0, 0, 400, 400], _IMG) == "M"

    def test_large(self):
        # 900x900 / 1e6 = 0.81 → L
        assert wc._size_bucket([0, 0, 900, 900], _IMG) == "L"


# ── 크기보정 무게 이상 판정 ─────────────────────────────────────────────────────
class TestAnomalySizeAware:
    def test_작은_빈페트_정상(self):
        # S 빈병 15g + margin 15 = 30 이하 → 정상
        assert not wc.is_anomaly("pet", 25.0, bbox=[0, 0, 100, 100], img_area=_IMG)

    def test_작은_페트_내용물_이상(self):
        # 200g >> 30 → 이상
        assert wc.is_anomaly("pet", 200.0, bbox=[0, 0, 100, 100], img_area=_IMG)

    def test_큰_빈페트_정상(self):
        # L 빈병(2L) 45g + margin 15 = 60 이하 → 55g 정상 (단일임계였으면 오판 안 함)
        assert not wc.is_anomaly("pet", 55.0, bbox=[0, 0, 900, 900], img_area=_IMG)

    def test_캔_내용물_이상(self):
        # S 캔 14g + 15 = 29, 150g → 이상
        assert wc.is_anomaly("can", 150.0, bbox=[0, 0, 100, 100], img_area=_IMG)


# ── 폴백 (bbox 미제공) ──────────────────────────────────────────────────────────
class TestAnomalyFallback:
    def test_pet_폴백_정상(self):
        assert not wc.is_anomaly("pet", 250.0)  # < 300

    def test_pet_폴백_이상(self):
        assert wc.is_anomaly("pet", 350.0)      # > 300

    def test_미등록품목_항상_정상(self):
        assert not wc.is_anomaly("unknown", 99999.0)

    def test_img_area_0이면_폴백(self):
        # img_area=0 → 크기보정 불가 → 폴백
        assert wc.is_anomaly("pet", 350.0, bbox=[0, 0, 100, 100], img_area=0.0)


# ── 표준무게 추정 ───────────────────────────────────────────────────────────────
class TestExpectedWeight:
    def test_pet_크기별(self):
        assert wc.expected_empty_weight("pet", [0, 0, 100, 100], _IMG) == 15.0
        assert wc.expected_empty_weight("pet", [0, 0, 900, 900], _IMG) == 45.0

    def test_미등록품목_None(self):
        assert wc.expected_empty_weight("glass", [0, 0, 100, 100], _IMG) is None
