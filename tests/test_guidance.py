"""guidance 도메인 규칙 단위 테스트 — 조건 충족/불충족 분기, 품목 판정, 안내 생성."""

from app.schemas.enums import GeneralWasteCode, GuidanceCode, RejectionCode, WasteClass
from app.schemas.response import Conditions
from app.services import guidance


def _codes(g):
    return [x.code for x in g]


# ── build_guidance: 조건 충족 → 빈 배열(ALLOWED) ────────────────────────────────
class TestConditionsMet:
    def test_pet_충족(self):
        # 라벨X·압착O·무게정상 → 안내 없음
        g = guidance.build_guidance(WasteClass.PET, Conditions(has_label=False, is_dented=True), False)
        assert g == []

    def test_can_충족(self):
        g = guidance.build_guidance(WasteClass.CAN, Conditions(is_dented=True), False)
        assert g == []

    def test_plastic_충족(self):
        g = guidance.build_guidance(WasteClass.PLASTIC, Conditions(has_label=False), False)
        assert g == []

    def test_paper_충족(self):
        g = guidance.build_guidance(WasteClass.PAPER, Conditions(), False)
        assert g == []

    def test_모델_미탑재_무게정상이면_충족(self):
        # 멀티헤드 없음 → conditions 전부 None → 무게만 보고 충족
        g = guidance.build_guidance(WasteClass.PET, Conditions(), False)
        assert g == []


# ── build_guidance: 조건 불충족 → 재처리 안내(REJECTED) ─────────────────────────
class TestConditionsUnmet:
    def test_pet_세조건_불충족_우선순위(self):
        # 무게 → 라벨 → 압착 순
        g = guidance.build_guidance(WasteClass.PET, Conditions(has_label=True, is_dented=False), True)
        assert _codes(g) == [GuidanceCode.EMPTY_CONTENTS, GuidanceCode.REMOVE_LABEL, GuidanceCode.COMPRESS]

    def test_pet_외부이물질까지_불충족_우선순위(self):
        g = guidance.build_guidance(
            WasteClass.PET,
            Conditions(has_foreign_material=True, has_label=True, is_dented=False),
            True,
        )
        assert _codes(g) == [
            GuidanceCode.EMPTY_CONTENTS,
            GuidanceCode.FOREIGN_MATERIAL,
            GuidanceCode.REMOVE_LABEL,
            GuidanceCode.COMPRESS,
        ]

    def test_can_미압착만(self):
        g = guidance.build_guidance(WasteClass.CAN, Conditions(is_dented=False), False)
        assert _codes(g) == [GuidanceCode.COMPRESS]

    def test_plastic_라벨만(self):
        g = guidance.build_guidance(WasteClass.PLASTIC, Conditions(has_label=True), False)
        assert _codes(g) == [GuidanceCode.REMOVE_LABEL]

    def test_paper_무게만(self):
        g = guidance.build_guidance(WasteClass.PAPER, Conditions(), True)
        assert _codes(g) == [GuidanceCode.WEIGHT_ANOMALY]

    def test_vinyl_무게만(self):
        g = guidance.build_guidance(WasteClass.VINYL, Conditions(), True)
        assert _codes(g) == [GuidanceCode.WEIGHT_ANOMALY]

    def test_plastic_무게만(self):
        g = guidance.build_guidance(WasteClass.PLASTIC, Conditions(), True)
        assert _codes(g) == [GuidanceCode.EMPTY_CONTENTS]

    def test_can_무게만(self):
        g = guidance.build_guidance(WasteClass.CAN, Conditions(), True)
        assert _codes(g) == [GuidanceCode.EMPTY_CONTENTS]

    def test_외부이물질만(self):
        g = guidance.build_guidance(
            WasteClass.PAPER,
            Conditions(has_foreign_material=True),
            False,
        )
        assert _codes(g) == [GuidanceCode.FOREIGN_MATERIAL]


# ── 품목별 헤드 비대상은 검사 안 함 (None 무시) ─────────────────────────────────
class TestHeadScope:
    def test_can_라벨헤드_없음(self):
        # 캔은 has_label 대상 아님 → True여도(있을 리 없지만) None이면 REMOVE_LABEL 안 생성
        g = guidance.build_guidance(WasteClass.CAN, Conditions(has_label=None, is_dented=True), False)
        assert g == []

    def test_plastic_압착헤드_없음(self):
        # 플라스틱은 is_dented 대상 아님 → None이면 COMPRESS 안 생성
        g = guidance.build_guidance(WasteClass.PLASTIC, Conditions(has_label=False, is_dented=None), False)
        assert g == []


# ── 품목 분류 판정 ──────────────────────────────────────────────────────────────
class TestClassification:
    def test_allowed(self):
        for c in (WasteClass.PET, WasteClass.PLASTIC, WasteClass.CAN, WasteClass.PAPER):
            assert guidance.is_allowed(c)

    def test_rejected(self):
        for c in (
            WasteClass.GLASS,
            WasteClass.BATTERY,
            WasteClass.FLUORESCENT,
            WasteClass.STYROFOAM,
        ):
            assert guidance.is_rejected(c)

    def test_general(self):
        assert guidance.is_vinyl(WasteClass.VINYL)

    def test_상호배타(self):
        # 9개 모델 클래스는 허용/거부/일반 중 정확히 하나에 속함
        for c in WasteClass:
            n = sum((guidance.is_allowed(c), guidance.is_rejected(c), guidance.is_vinyl(c)))
            assert n == 1, f"{c} 의 처리 분류가 없거나 중복됨"


# ── 안내 메시지 생성 ────────────────────────────────────────────────────────────
class TestMessages:
    def test_rejection_전품목(self):
        for c in (
            WasteClass.GLASS,
            WasteClass.BATTERY,
            WasteClass.FLUORESCENT,
            WasteClass.STYROFOAM,
        ):
            r = guidance.build_rejection(c)
            assert r.code == RejectionCode(c.value.upper()) and r.message

    def test_general_메시지(self):
        g = guidance.build_general(GeneralWasteCode.VINYL)
        assert g.code == GeneralWasteCode.VINYL
        assert "일반쓰레기" in g.message

    def test_guidance_메시지_매핑(self):
        # 생성된 모든 guidance 는 코드+메시지를 가짐
        g = guidance.build_guidance(WasteClass.PET, Conditions(has_label=True, is_dented=False), True)
        for item in g:
            assert item.code in GuidanceCode and item.message
