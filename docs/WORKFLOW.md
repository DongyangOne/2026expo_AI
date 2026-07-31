# 2026 동양미래 EXPO — 재활용 분류 AI 워크플로우

> 키오스크 재활용품 자동 분류 시스템. 사진 + 무게센서 + `client_id` 입력 → 분류·상태 판정 → Spring 서버 전송.
> 최종 갱신: 2026-07-20

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 목적 | 키오스크에 투입된 재활용품을 사진+무게로 분류하고 상태(라벨/찌그러짐/내용물)를 판정 |
| 추론 환경 | Raspberry Pi 5 CPU (메인 NCNN + 상태 ONNX Runtime) |
| 입력 | 카메라 사진 + 무게센서값 + 사용자/피드백 구분용 `client_id` (**투입부/센서는 타팀 하드웨어**) |
| 출력 | 하드웨어에 `DetectResponse` 즉시 반환 + 동일 JSON을 Spring 서버로 백그라운드 POST (**자체 DB 없음**) |
| 담당 범위 | AI 처리 + 결과 API 전송 (내가 담당) / 사진·무게 수집 (타팀) |

---

## 2. 시스템 아키텍처

```
[타팀 하드웨어]                    [내 담당: FastAPI on Pi5]                         [Spring]
 카메라 사진  ─┐
 무게센서값   ─┼─→ POST /api/v1/detect ─→ 메인 YOLO26m (9클래스)
 client_id    ─┘                         ├─→ 상태 멀티헤드 (라벨/압착)
                                         ├─→ 무게 이상 로직
                                         ├─→ DetectResponse(client_id 유지) ─→ 하드웨어 즉시 응답
                                         └─→ 동일 JSON 백그라운드 POST ─────→ 콜백 API
```

- `client_id`는 1~128자의 필수 문자열이며 AI 서버에서 생성하거나 변경하지 않는다.
- Spring 콜백 URL: `https://oneexpo.kro.kr/api/v1/feedbackDetail/results`

---

## 3. 분류 체계

### 9클래스 (메인 모델)
`can · pet · paper · plastic · styrofoam · vinyl · glass · battery · fluorescent`

| 구분 | 클래스 | status | 후속 |
|------|--------|--------|------|
| **재활용 허용** | 페트·플라스틱·캔·종이 | ALLOWED(조건충족) / REJECTED(불충족) | 조건검사 후 재활용 함 |
| **일반쓰레기** | 비닐 · 저신뢰 · 미분류 | 비닐 정상: GENERAL_WASTE / 비닐 무게·이물질 이상: REJECTED | 일반함 또는 재처리 |
| **수거 거부** | 유리·건전지·형광등·스티로폼 | REJECTED | 분리 불가 안내 |
| **미감지** | — | NOT_DETECTED | 재시도 |

### 상태 조건 (허용 품목) — 충족=ALLOWED, 불충족=REJECTED(재처리)
| 품목 | 라벨 떼기 | 압착 | 무게정상 |
|------|:---:|:---:|:---:|
| 페트병 | ✅ | ✅ | ✅ |
| 플라스틱 | ✅ | — | ✅ |
| 캔 | — | ✅ | ✅ |
| 종이 | — | — | ✅ |

> 조건 하나라도 불충족 → `REJECTED` + guidance(`EMPTY_CONTENTS`/`WEIGHT_ANOMALY`/`FOREIGN_MATERIAL`/`REMOVE_LABEL`/`COMPRESS`) 재처리 안내 → 사용자 처리 후 재투입.
> 비닐은 무게·외부 이물질 이상이 없으면 일반쓰레기(`GENERAL_WASTE`)이고, 이상이 있으면 재처리(`REJECTED`)한다. 유리·건전지·형광등·스티로폼은 완전 수거거부(`REJECTED`+`rejection`).

### GuidanceCode 매핑

| 조건 | 코드 |
|------|------|
| 페트·플라스틱·캔 무게 이상/내용물 존재 | `EMPTY_CONTENTS` |
| 종이·비닐 무게 이상 | `WEIGHT_ANOMALY` |
| 외부 이물질 | `FOREIGN_MATERIAL` |
| 라벨 미제거 | `REMOVE_LABEL` |
| 페트·캔 미압착 | `COMPRESS` |

> 현재 배포된 상태 모델은 `dent`/`label` 2헤드다. 계약과 런타임은 선택적 `foreign_material` 출력을 지원하지만, 해당 헤드가 없는 현 모델에서는 `conditions.has_foreign_material=null`이며 `FOREIGN_MATERIAL`을 자동 생성하지 않는다.

---

## 4. 런타임 추론 파이프라인

`app/services/pipeline.py` 의 `run()`:

```
1. 필수 `client_id` 수신 + 이미지 디코드
2. 메인 YOLO 감지 (conf=DETECT_CONF 0.25)
     └ 박스 없음 → NOT_DETECTED
3. 최고신뢰 박스 → class_id, confidence, bbox
4. 신뢰도 판정
     └ confidence < TRUST_CONF(0.55) → LOW_CONFIDENCE (일반쓰레기)
5. [허용: 페트/플라스틱/캔/종이]
     ├ 상태 멀티헤드(crop ONNX) → conditions.is_dented / has_label / has_foreign_material(선택)
     ├ 무게 → weight.anomaly
     └ build_guidance() → 불충족 안내. 비면 ALLOWED, 있으면 REJECTED(재처리)
6. [거부: 유리/건전지/형광등/스티로폼] → REJECTED + rejection
7. [비닐] → 무게·외부 이물질 이상이면 REJECTED + guidance, 정상이면 GENERAL_WASTE + general
8. `client_id`를 포함한 DetectResponse 조립 → 하드웨어 응답 + Spring 콜백(fire-and-forget)
```

**2단계 신뢰도 게이트:**
- `DETECT_CONF=0.25`: 박스 생성 최소선 (미만 = 미감지)
- `TRUST_CONF=0.55`: 분류 신뢰선 (미만 = 일반쓰레기)

---

## 5. 모델 구성 (3-tier)

| # | 모델 | 입력 | 출력 | 백본 | 포맷 | 상태 |
|---|------|------|------|------|------|------|
| 1 | 메인 감지 | 640px | 9클래스 bbox | YOLO26m | NCNN | Pi5 배포·로드 정상 |
| 2 | 상태 멀티헤드 | 224px crop | dent(2) + label(2), foreign_material(선택) | MobileNetV3-Small | ONNX | Pi5 배포 모델은 2헤드 |
| (3) | 색/재질 SubClass | 224px crop | 무색/유색, 철/알루미늄 | — | — | **보류** (상태 우선) |

> 모델 3은 이전 설계의 색·재질 세부분류. 이번 우선순위는 **상태(라벨/찌그러짐)**라 보류.
> 단 캔 철/알루미늄은 무게가 달라(철>알루미늄) 무게로직 정밀도에 도움 → 추후 재검토.

### 멀티헤드 설계 (모델 2)
- **공유 백본 1개** + dent 헤드 + label 헤드. 추론 1회로 두 출력.
- 런타임에서는 페트병은 두 헤드, 캔은 dent, 플라스틱은 label 결과를 사용한다.
- dent는 페트+캔 공통 학습, label은 페트+플라스틱 공통 학습하며 비대상 헤드는 loss에서 마스킹한다.

---

## 6. 학습 파이프라인

### 6-1. 메인 모델 (YOLO26m)
```
AI Hub 원본 JSON+이미지 (2TB)
  └→ scripts/convert_v2.py
       · POLYGON 라벨 복원 (v1은 20~50% 누락 버그)
       · 640px 리사이즈 (4MB→~50KB, I/O 70배↓)
       · 폴더당 15000 stride 샘플링
  └→ /share/Container/yolo_dataset_9class_v2/ (train/val + dataset.yaml)
  └→ YOLO26m 학습 → best.pt → NCNN export
```

### 6-2. 상태 멀티헤드 분류기
```
AI Hub 원본 JSON+이미지 (2TB, 직접촬영)
  └→ scripts/extract_crops.py
       · bbox crop + 12% 패딩 → 224px letterbox
       · DAMAGE → dent (원형0 / 찌그러짐·완전압착1)
       · DIRTINESS → label (오염없음0 / 이물질외부1, 내부·전체는 -1 마스킹)
       · ※ 원본 사용 필수 (640변환본은 파일명 리네임돼 JSON 매칭 불가)
  └→ /share/Container/crops_state_v1/ (pet/ + can/ + manifest.csv)
  └→ scripts/train_classifier.py
       · MobileNetV3-Small + dent/label 헤드
       · 클래스 불균형 → CrossEntropy class-weight
  └→ best.pt → multihead.onnx → onnx2ncnn
```

---

## 7. 데이터셋 (AI Hub dataSetSn=71362)

- **재활용품 분류 및 선별 데이터**, 약 100만 장, JSON 라벨 (BBOX 70% / POLYGON 30%)
- 촬영: 직접촬영 70.2% (키오스크 환경 유사) + 선별영상추출 29.8% (컨베이어)
- **객체 속성 5개**: `DAMAGE` `DIRTINESS` `COVER` `TRANSPARENCY` `SHAPE`
- **핵심 발견**: `DIRTINESS=이물질(외부)` ≈ **라벨 부착** (육안검증). `오염없음`=라벨 뗀 맨 용기.
- 상세: `memory/dataset_attribute_labels.md`, 데이터 점검: `docs/DATA_AUDIT.md`

### 실측 상태 분포 (직접촬영)
| 품목 | 찌그러짐+완전압착 | 라벨있음/없음 |
|------|------|------|
| 페트병 무색 | 29.7% (21,659) | 71.7% / 20% |
| 페트병 유색 | 19.3% (11,889) | 75% / 23% |
| 철캔 | 18.8% (10,616) | 50% / 46% |
| 플라스틱 PE | 4.5% (2,166) ⚠️데이터빈약 | 87% / 12% |

---

## 8. 배포 (Raspberry Pi 5)

### 런타임 구성

- 메인 모델: `MAIN_MODEL_PATH=weights/yolo26m_best_ncnn_model` (NCNN, ARM CPU 추론)
- 상태 모델: `STATE_MODEL_PATH=weights/multihead.onnx` (ONNX Runtime)
- 실제 환경 파일: `/home/one/2026expo_AI/.env`
- Spring 콜백: `SPRING_CALLBACK_URL=https://oneexpo.kro.kr/api/v1/feedbackDetail/results`
- 결과 로그: `LOG_RESULTS=true`이면 `logs/results.jsonl`에 항상 기록
- 재학습 캡처: `CAPTURE_REQUESTS=true`이면 `logs/captures/YYYY-MM-DD/`에 원본 이미지와 판정 JSON 쌍 저장
- 로그 영속화: Docker Compose의 `./logs:/app/logs` 볼륨 사용
- 콜백은 fire-and-forget 방식이며 연결 실패·타임아웃이 AI 응답에 영향을 주지 않는다.

### 오인식 개선 루프

1. 실제 키오스크 요청의 원본 이미지와 판정 결과를 같은 `capture_id`로 수집한다.
2. JSON의 `review.is_correct`, `review.expected_class`, `review.notes`를 검수자가 기록한다.
3. 콜라 캔→스티로폼/종이처럼 틀린 표본과 유사 정상 표본을 함께 hard sample로 구성한다.
4. 실제 키오스크 배경·조명·거리 분포를 유지해 메인 YOLO를 파인튜닝한다.
5. 기존 검증셋과 별도의 키오스크 hard-case 검증셋에서 클래스별 혼동행렬을 비교한다.

> 현재 모델은 재활용 데이터셋의 촬영 분포를 학습했기 때문에 흰 배경 상품 이미지,
> 렌더링 이미지, 인쇄 무늬처럼 운영 카메라와 다른 입력에서 재질 대신 배경·윤곽·색상에
> 치우쳐 오분류할 수 있다. 단순 임계값 상향만으로는 고신뢰 오분류를 해결할 수 없으므로
> 운영 이미지 수집과 재학습을 우선한다.

### 배포 흐름

1. 별도 기능 브랜치 없이 `main`에 직접 반영한다.
2. GitHub Actions는 `main` 푸시 시 단위 테스트를 실행한다.
3. Pi5의 `ai-autodeploy.timer`가 5분마다 `origin/main`의 새 커밋을 확인한다.
4. 새 커밋이 있으면 `/home/one/auto-deploy.sh`가 `git pull --ff-only` 후 `docker compose up -d --build`를 실행한다.
5. 시스템 nginx가 `https://ai.oneexpo.kro.kr` 요청을 FastAPI `localhost:8000`으로 프록시한다.

운영 확인:

```bash
systemctl status ai-autodeploy.timer
docker ps --filter name=ai-server
curl http://127.0.0.1:8000/health
```

정상 헬스 응답:

```json
{"status":"ok","models":{"main":true,"state":true}}
```

---

## 9. 인프라 (NAS / 학습 환경)

| 항목 | 값 |
|------|-----|
| NAS | QNAP, 192.168.0.110, Ryzen 1700 + RTX 2000 Ada 16GB |
| Docker | `/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker` (PATH에 없음, sudo 필요) |
| 학습 이미지 | `ultralytics/ultralytics:latest` |
| SMB | `\\192.168.0.110\Container` → 컨테이너 `/app` (`/share/Container`) |
| 접속 | plink `-hostkey ssh-ed25519 255 SHA256:HLPv3nWI2ozOxgy2AAzjguq1rrON+KOuWhh3zscXpy0` |
| 한글 출력 | plink가 한글 깨뜨림 → `base64 -w0` 우회 후 로컬 디코드 |
| 원본 데이터 | `/share/Container/ai_dataset/학습용_데이터/01-1.정식개방데이터/Training/` |

> 상세 접속법: `memory/nas-dataset-workflow.md`

---

## 10. 현재 운영 상태 (2026-07-20)

| 단계 | 상태 |
|------|------|
| 데이터 변환 (convert_v2) | ✅ 완료 (train 라벨 518,728 / val 68,147) |
| 메인 YOLO26m NCNN | ✅ Pi5 배포 및 로드 정상 |
| 상태 멀티헤드 ONNX | ✅ Pi5 배포 및 로드 정상 |
| `/api/v1/detect` + `client_id` 계약 | ✅ 구현·테스트·배포 완료 |
| Spring 콜백 URL | ✅ AI 서버 환경 및 컨테이너 반영 완료 |
| Spring 콜백 수신 테스트 | ⏳ Spring 측 엔드포인트 배포 후 확인 |
| Pi 자동 배포 배치 | ✅ `ai-autodeploy.timer` 활성, 5분 간격 |
| 공개 헬스 체크 | ✅ `https://ai.oneexpo.kro.kr/health` 정상 |

---

## 11. 코드 맵

```
app/
  core/config.py          설정 (임계값, 모델경로, Spring URL)
  models/registry.py      모델 보관소 (graceful degradation)
  services/request_capture.py  재학습용 원본 이미지/판정 JSON 쌍 저장
  schemas/
    enums.py              WasteClass, DetectionStatus, GuidanceCode, RejectionCode, GeneralWasteCode
    response.py           DetectResponse DTO (client_id/status/classification/conditions/weight/guidance/rejection/general)
    request.py            multipart 요청 (image/client_id/weight_g)
  services/
    pipeline.py           흐름 제어만 (run)
    inference.py          모델 추론 (전처리 + YOLO·멀티헤드 세션 호출)
    guidance.py           허용/거부/일반 판정 + 안내 (도메인 규칙)
    weight_check.py       무게 이상 (bbox 크기보정 + 단일임계 폴백)
    spring_client.py      Spring 콜백 (fire-and-forget)
  api/v1/detect.py        엔드포인트
  main.py                 앱 + 통합 에러 핸들러

tests/
  test_api_contract.py     client_id 필수 요청·응답 계약 및 콜백 전달 테스트
  test_guidance.py        도메인 규칙 단위테스트
  test_spring_client.py    Spring 콜백 URL·client_id 전달 테스트
  test_weight_check.py    무게 판정 단위테스트

scripts/
  convert_v2.py           메인 학습용 변환 (폴리곤복원 + 640리사이즈)
  extract_crops.py        상태분류기 crop 추출 (bbox + DAMAGE/DIRTINESS)
  train_classifier.py     멀티헤드 학습 (MobileNetV3 + dent/label)

docs/
  WORKFLOW.md             (이 문서)
  WEIGHT_KIOSK_PARAMS.md  무게 로직 + 키오스크 파라미터 체크리스트
  DATA_AUDIT.md           변환 데이터 정합성 점검
```
