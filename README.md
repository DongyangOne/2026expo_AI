# 2026 동양미래 EXPO — 재활용품 AI 분류 서버

라즈베리파이 5에서 동작하는 FastAPI 기반 AI 분류 서버.  
하드웨어(카메라 + 무게센서)로부터 이미지와 무게를 받아 9-class YOLO + 멀티헤드 상태분류기로 판정 후 Spring 서버로 결과를 전송한다.

---

## 수신 형식 (하드웨어 → AI 서버)

**`POST /api/v1/detect`**  
`Content-Type: multipart/form-data`

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `image` | File (jpg/png) | ✅ | 카메라 촬영 이미지 |
| `client_id` | string | ✅ | 사용자/피드백 구분 ID. AI 응답과 Spring 콜백에 그대로 반환 |
| `weight_g` | float | ❌ | 무게센서 값 (g). 미입력 시 무게 이상감지 생략 |

**헤더**

| 헤더 | 설명 |
|------|------|
| `X-API-Key` | 서버 인증 키 (`.env`의 `API_KEY`와 일치해야 함) |

**요청 예시**

```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -H "X-API-Key: 인증키" \
  -F "image=@sample.jpg" \
  -F "client_id=hardware-user-001" \
  -F "weight_g=28.0"
```

---

## 응답 형식 (AI 서버 → 하드웨어 / Spring)

```json
{
  "client_id": "hardware-user-001",
  "status": "ALLOWED",
  "classification": {"class_id": 1, "class_name": "pet", "confidence": 0.94},
  "conditions": {"has_label": false, "is_dented": true, "has_foreign_material": null},
  "weight": {"value_g": 28.0, "anomaly": false},
  "guidance": [],
  "rejection": null,
  "general": null,
  "bbox": [120.0, 80.0, 410.0, 560.0]
}
```

### `status` 판별자

| 값 | 의미 | 채워지는 필드 |
|----|------|--------------|
| `ALLOWED` | 재활용 허용 | `classification`, `conditions`, `weight`, `guidance`(빈 배열) |
| `REJECTED` | 거부 | 조건불충족: `guidance` / 완전거부(유리 등): `rejection` |
| `GENERAL_WASTE` | 일반쓰레기 | `general` |
| `NOT_DETECTED` | 감지 실패 | (없음) |

### `guidance` 코드 (REJECTED 재처리 안내)

| code | 의미 |
|------|------|
| `EMPTY_CONTENTS` | 내용물 비우기 (페트·플라스틱·캔 무게 이상) |
| `WEIGHT_ANOMALY` | 무게 이상 확인 (종이·비닐) |
| `FOREIGN_MATERIAL` | 외부 이물질 제거 |
| `REMOVE_LABEL` | 라벨 제거 (페트·플라스틱) |
| `COMPRESS` | 압착 (페트·캔) |

> 현재 배포된 상태 모델은 `dent`/`label` 2헤드이므로 `has_foreign_material`은 `null`이다. `foreign_material` 헤드가 포함된 모델을 탑재하면 `FOREIGN_MATERIAL` 안내가 자동 활성화된다.

### `rejection` 코드 (완전 거부)

| code | 의미 |
|------|------|
| `GLASS` | 유리 |
| `BATTERY` | 건전지 |
| `FLUORESCENT` | 형광등 |
| `STYROFOAM` | 스티로폼 |

### `general` 코드 (일반쓰레기)

| code | 의미 |
|------|------|
| `VINYL` | 비닐 |
| `LOW_CONFIDENCE` | 신뢰도 미달 |
| `UNCLASSIFIED` | 미분류 |

---

## Spring 콜백

하드웨어 응답 직후 백그라운드로 Spring 서버에 `client_id`를 포함한 동일한 JSON을 POST한다.
`.env`의 `SPRING_CALLBACK_URL` 미설정 시 전송하지 않는다.

콜백 URL: `https://oneexpo.kro.kr/api/v1/feedbackDetail/results`

---

## 환경 설정 (`.env`)

```env
API_KEY=인증키
SPRING_CALLBACK_URL=https://oneexpo.kro.kr/api/v1/feedbackDetail/results
# 선택
MAIN_MODEL_PATH=weights/yolo26m_best_ncnn_model
STATE_MODEL_PATH=weights/multihead.onnx
VERIFIER_MODEL_PATH=weights/verifier_qwen35_mnv3_v1.onnx
VERIFIER_SHADOW_ENABLED=true
VERIFIER_SHADOW_LOG_PATH=logs/verifier_shadow.jsonl
DETECT_CONF=0.25
TRUST_CONF=0.55
WEIGHT_ANOMALY_ENABLED=true
# 결과 로깅 (Spring 없이도 logs/results.jsonl 에 저장)
LOG_RESULTS=true
LOG_DIR=logs
# 재학습/오인식 검수용 원본 이미지 + 판정 JSON 저장
CAPTURE_REQUESTS=true
CAPTURE_DIR=logs/captures
CAPTURE_RETENTION_DAYS=90
CAPTURE_MAX_STORAGE_MB=10240
```

### 결과 로그 (`logs/results.jsonl`)

`LOG_RESULTS=true`(기본값)이면 Spring 콜백 여부와 무관하게 판정 결과를 JSONL 형식으로 기록한다.  
`SPRING_CALLBACK_URL` 미설정 시 로그만 저장하므로 Spring 서버 없이도 결과를 확인할 수 있다.

```jsonl
{"timestamp":"2026-07-03T10:00:00+00:00","client_id":"hardware-user-001","status":"ALLOWED","classification":{"class_id":1,...},...}
```

### 요청 이미지와 판정 캡처 (`logs/captures/`)

`CAPTURE_REQUESTS=true`이면 정상적으로 판정된 요청마다 원본 이미지와 판정 JSON을 같은
`capture_id`로 저장한다. Docker Compose에서는 `./logs:/app/logs` 볼륨을 사용하므로
컨테이너를 다시 만들어도 파일이 유지된다.

```text
logs/captures/2026-07-31/
  20260731T012345123456Z_a1b2c3d4e5f6.jpg
  20260731T012345123456Z_a1b2c3d4e5f6.json
```

JSON에는 요청의 `client_id`와 무게, 예측 클래스·신뢰도·bbox·상태, 이미지 SHA-256과
아래 검수 필드가 포함된다. API 키와 요청 헤더는 저장하지 않는다.

```json
{
  "review": {
    "is_correct": null,
    "expected_class": null,
    "is_single_object": null,
    "is_dented": null,
    "has_label": null,
    "has_foreign_material": null,
    "notes": null
  }
}
```

기본 보존 기간은 90일, 최대 용량은 10GB이며 초과 시 오래된 이미지/JSON 쌍부터 제거한다.
운영 캡처는 자동 teacher의 tight/context 합의와 확신도 기준을 통과한 경우에만 hard
sample로 재학습 데이터에 추가한다. 합의 실패·저신뢰·다중 객체는 `-1`로 마스킹해
학습에서 제외하며 사람 검토를 전제로 하지 않는다.

### 임시 crop 검증기 shadow 로그

`VERIFIER_SHADOW_ENABLED=true`이면 기존 YOLO가 만든 bbox를 임시 320px 검증기로
비동기 재검증하고 `logs/verifier_shadow.jsonl`에 YOLO/검증기 품목 일치 여부와
압착·라벨·외부 이물질 출력을 기록한다. 이 결과는 초기에는 API 응답, guidance,
Spring 콜백을 변경하지 않는다. 임시 모델의 운영 분포 정확도를 확인한 뒤에만 판정에
사용한다.

crop 검증기 학습에는 원본 정답이 단일 객체이고 자동 teacher도 단일 주 객체로
판정한 이미지만 사용한다. `label`과 `foreign_material`은 서로 독립된 정답이며,
네 조합(둘 다 없음/라벨만/외부 이물질만/둘 다 있음)을 그대로 기록한다.

객체 bbox를 crop한 뒤 9종 품목과 상태를 다시 확인하는 검증기의 확정 구조, 라벨 정책,
NAS 실행 명령은 [`docs/CROP_VERIFIER_PLAN.md`](docs/CROP_VERIFIER_PLAN.md)에 정리했다.
1일차 prototype은 고해상도 원본을 복제하지 않고 경로+bbox만 참조하며, 320px crop 생성 →
기존 `naco-ollama`의 `qwen3.5:9b-q4_K_M`로 tight/context 자동 합의 상태 pseudo-label을
만드는 순서로 진행한다. 사람 검토는 두지 않는다. NAS 여유 공간 500GB와
새 crop 20GB 상한을 통과해야 전체 정제를 계속한다.
Ollama는 연속 이미지 prompt cache와 16K context를 지원하는 `0.32.0`을 사용한다.

---

## 실행

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API 문서: `http://localhost:8000/docs`

### 배포

`main`에 푸시하면 GitHub Actions에서 테스트를 실행한다. Pi5의 `ai-autodeploy.timer`가
5분마다 `origin/main`의 새 커밋을 확인하고 `docker compose up -d --build`로 자동 배포한다.
