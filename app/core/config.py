from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # ── 인증 ────────────────────────────────────────────────────────────────────
    API_KEY: str

    # ── 모델 경로 ────────────────────────────────────────────────────────────────
    # 메인 9-class YOLO. NCNN: 디렉터리 경로 / .pt: 파일 경로
    MAIN_MODEL_PATH: str = "weights/yolo26m_best_ncnn_model"
    # 상태 멀티헤드(dent/label) ONNX 경로. None = 미탑재 → conditions=null (무게만 검사)
    STATE_MODEL_PATH: str | None = "weights/multihead.onnx"
    # YOLO bbox를 320px crop으로 다시 확인하는 임시 9종+상태 검증기.
    VERIFIER_MODEL_PATH: str | None = "weights/verifier_qwen35_mnv3_v1.onnx"
    # 초기에는 응답/콜백을 바꾸지 않고 YOLO와의 비교 결과만 별도 JSONL에 기록한다.
    VERIFIER_SHADOW_ENABLED: bool = True
    VERIFIER_SHADOW_LOG_PATH: str = "logs/verifier_shadow.jsonl"

    # 검증기 추론을 여러 뷰(원본·좌우반전·중앙확대)로 돌려 확률을 평균낸다.
    # 학습 미사용 하드웨어 35건: 26/35(74.3%) → 28/35(80.0%). 기존 정답을 깨뜨린
    # 건은 0건이었지만 McNemar p=0.500이라 이 표본 크기로는 개선이 입증되지 않았다.
    # 뷰 수만큼 forward가 늘어 Pi5 지연시간이 커지므로 기본값은 off로 둔다.
    VERIFIER_TTA_ENABLED: bool = False

    # 압착 판정을 검증기 dent 헤드로 만들고, '압착됨'으로 통과시킬 때만 확신을 요구한다.
    # 실제 키오스크 crop 31건(전부 미압착이 정답)에서 '이미 압착됨' 오탐:
    #   state 5/31(16.1%), 검증기 1/31(3.2%), 검증기+0.90 요구 0/31(0%).
    # 오탐은 안 구긴 캔·페트병을 그대로 통과시키므로 확신이 없으면 압착을 요구한다.
    # 반대 방향(이미 압착된 것을 또 요구)은 하드웨어 정답에 압착 양성이 0건이라
    # 측정하지 못했다. 재투입 안내가 한 번 더 나가는 쪽이 안전하다고 보고 감수한다.
    VERIFIER_DENT_HEAD_ENABLED: bool = True
    VERIFIER_DENT_CONF: float = 0.90

    # conditions.has_label을 구형 state 모델 대신 crop 검증기 label 헤드로 만든다.
    # 학습에 쓰이지 않은 하드웨어 crop 19건(라벨있음 13·없음 6) 실측:
    #   state 16/19(84.2%), 검증기 19/19(100%).
    #   state의 오류 3건 중 2건이 "라벨이 있는데 없다고 함"이라 미제거 라벨을 통과시킨다.
    # dent와 foreign_material은 하드웨어 정답에 양성 표본이 없어 근거가 없으므로 바꾸지 않는다.
    VERIFIER_LABEL_HEAD_ENABLED: bool = True

    # YOLO가 TRUST_CONF에 못 미쳐 일반쓰레기로 보낼 건을 crop 검증기가 확신하면 되살린다.
    # 학습 미사용 하드웨어 35건 실측(현재 동작 → 구제 0.60):
    #   정답 20→27, 보류 14→4, 오답 1→4 (McNemar p=0.016, 현재만 맞은 건 0건).
    # 특히 플라스틱은 7건 전부 보류되던 것이 5건 정답으로 바뀐다.
    VERIFIER_RESCUE_ENABLED: bool = True
    VERIFIER_RESCUE_CONF: float = 0.60

    # YOLO와 crop 검증기가 응답 클래스에서 불일치하면 확정하지 않고 일반쓰레기로 보낸다.
    # 학습에 쓰이지 않은 하드웨어 감사 35건에서 두 모델이 합의한 21건은 전부 정답이었고,
    # 불일치한 14건은 YOLO 4건·검증기 5건만 맞아 어느 쪽도 근거가 되지 못했다.
    # 켜면 확정 오답이 7→0건으로 줄지만 보류가 3→14건으로 늘어 처리량을 포기한다.
    VERIFIER_AGREEMENT_GATE_ENABLED: bool = False

    # 저신뢰 PET/PLASTIC과 VINYL이 같은 bbox에서 경쟁할 때만 crop 검증기로 교정한다.
    VINYL_CORRECTION_ENABLED: bool = True
    VINYL_CANDIDATE_CONF: float = 0.10
    VINYL_CANDIDATE_IOU: float = 0.70
    VINYL_CANDIDATE_RATIO: float = 0.40
    VINYL_VERIFIER_CONF: float = 0.65
    VINYL_VERIFIER_MARGIN: float = 0.25

    # ── 추론 설정 ────────────────────────────────────────────────────────────────
    # 2단계 임계값:
    #   DETECT_CONF 미만 → 박스 없음 → NOT_DETECTED
    #   DETECT_CONF 이상 ~ TRUST_CONF 미만 → LOW_CONFIDENCE (일반쓰레기)
    #   TRUST_CONF 이상 → ALLOWED / REJECTED 정상 판정
    DETECT_CONF: float = 0.25       # YOLO 박스 생성 최소 신뢰도
    DETECT_IOU: float = 0.70        # NMS IoU (Ultralytics 기본값을 명시적으로 고정)
    TRUST_CONF: float = 0.55        # 신뢰 판정 임계값
    IMG_SIZE: int = 640
    SUB_IMG_SIZE: int = 224
    DEVICE: str = "cpu"              # Pi5 CPU, GPU 환경이면 "0"

    # ── 무게 이상 감지 ────────────────────────────────────────────────────────────
    WEIGHT_ANOMALY_ENABLED: bool = True

    # ── Spring 콜백 ──────────────────────────────────────────────────────────────
    # None 이면 콜백 비활성 (Spring URL 미설정 시)
    SPRING_CALLBACK_URL: str | None = None
    SPRING_TIMEOUT_SEC: float = 3.0
    # 순간적인 타임아웃/연결 오류/5xx 응답만 재시도한다. 4xx는 계약 오류이므로 즉시 종료한다.
    SPRING_MAX_ATTEMPTS: int = 3
    SPRING_RETRY_BACKOFF_SEC: float = 0.5

    # ── 결과 로깅 ────────────────────────────────────────────────────────────────
    # True: 판정 결과를 logs/results.jsonl 에 항상 기록 (Spring URL 없어도 동작)
    LOG_RESULTS: bool = True
    LOG_DIR: str = "logs"

    # ── 재학습/오인식 검수용 요청 캡처 ───────────────────────────────────────────
    # 원본 이미지와 판정 JSON을 동일한 capture_id로 저장한다.
    CAPTURE_REQUESTS: bool = True
    CAPTURE_DIR: str = "logs/captures"
    CAPTURE_MAX_IMAGE_BYTES: int = 20 * 1024 * 1024
    CAPTURE_RETENTION_DAYS: int = 90
    CAPTURE_MAX_STORAGE_MB: int = 10 * 1024


settings = Settings()
