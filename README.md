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
  "conditions": {"has_label": false, "is_dented": true},
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
| `EMPTY_CONTENTS` | 내용물 비우기 |
| `REMOVE_LABEL` | 라벨 제거 (페트·플라스틱) |
| `COMPRESS` | 압착 (페트·캔) |

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
DETECT_CONF=0.25
TRUST_CONF=0.55
WEIGHT_ANOMALY_ENABLED=true
# 결과 로깅 (Spring 없이도 logs/results.jsonl 에 저장)
LOG_RESULTS=true
LOG_DIR=logs
```

### 결과 로그 (`logs/results.jsonl`)

`LOG_RESULTS=true`(기본값)이면 Spring 콜백 여부와 무관하게 판정 결과를 JSONL 형식으로 기록한다.  
`SPRING_CALLBACK_URL` 미설정 시 로그만 저장하므로 Spring 서버 없이도 결과를 확인할 수 있다.

```jsonl
{"timestamp":"2026-07-03T10:00:00+00:00","client_id":"hardware-user-001","status":"ALLOWED","classification":{"class_id":1,...},...}
```

---

## 실행

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API 문서: `http://localhost:8000/docs`
