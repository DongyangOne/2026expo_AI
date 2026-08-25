# 2026 동양미래 EXPO — 재활용 분류 AI 워크플로우

> 키오스크 재활용품 자동 분류 시스템. 사진 + 무게센서 + `client_id` 입력 → 분류·상태 판정 → Spring 서버 전송.
> 최종 갱신: 2026-07-31

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
- Spring 콜백 URL: `https://oneexpo.kro.kr/api/v1/feedback-detail/result`

---

## 3. 분류 체계

### 9클래스 (메인 모델)
`can · pet · paper · plastic · styrofoam · vinyl · glass · battery · fluorescent`

| 구분 | 클래스 | status | 후속 |
|------|--------|--------|------|
| **지정 함 허용** | 플라스틱(PET 포함)·캔·종이·비닐 | ALLOWED(조건충족) / REJECTED(불충족) | 조건검사 후 해당 함 |
| **분류 불가** | 저신뢰 · 미분류 | GENERAL_WASTE | 일반함 |
| **수거 거부** | 유리·건전지·형광등·스티로폼 | REJECTED | 수거 불가 안내 |
| **미감지** | — | NOT_DETECTED | 재시도 |

> 모델은 학습 정확도를 위해 PET와 기타 플라스틱을 별도 클래스로 유지한다. 운영 응답과 Spring 콜백에서는 PET도 `class_id=3`, `class_name=plastic`으로 정규화한다.

### 상태 조건 (허용 품목) — 충족=ALLOWED, 불충족=REJECTED(재처리)
| 품목 | 라벨 떼기 | 압착 | 무게정상 |
|------|:---:|:---:|:---:|
| 플라스틱 병(모델 PET) | ✅ | ✅ | ✅ |
| 기타 플라스틱 | ✅ | — | ✅ |
| 캔 | — | ✅ | ✅ |
| 종이 | — | — | ✅ |

> 조건 하나라도 불충족 → `REJECTED` + guidance(`EMPTY_CONTENTS`/`WEIGHT_ANOMALY`/`FOREIGN_MATERIAL`/`REMOVE_LABEL`/`COMPRESS`) 재처리 안내 → 사용자 처리 후 재투입.
> 비닐은 무게·외부 이물질 이상이 없을 때만 비닐함 투입(`ALLOWED`)을 허용하고, 이상이 있으면 재처리(`REJECTED`)한다. 저신뢰·미분류는 `GENERAL_WASTE`로 유지한다. 유리·건전지·형광등·스티로폼은 완전 수거거부(`REJECTED`+`rejection`).

### GuidanceCode 매핑

| 조건 | 코드 |
|------|------|
| 플라스틱(PET 포함)·캔 무게 이상/내용물 존재 | `EMPTY_CONTENTS` |
| 종이·비닐 무게 이상 | `WEIGHT_ANOMALY` |
| 외부 이물질 | `FOREIGN_MATERIAL` |
| 라벨 미제거 | `REMOVE_LABEL` |
| 페트·캔 미압착 | `COMPRESS` |

> 현재 배포된 상태 모델은 `dent`/`label` 2헤드다. 외부 이물질 필드는 응답 조건값으로 보내지 않으며, 향후 해당 헤드가 추가되면 `FOREIGN_MATERIAL` guidance만 생성한다.

---

## 4. 런타임 추론 파이프라인

`app/services/pipeline.py` 의 `run()`:

```
1. 필수 `client_id` 수신 + 이미지 디코드
2. 메인 YOLO 감지 (최종 최소선=DETECT_CONF 0.25, 비닐 보조 후보 최소선=0.10)
     └ 0.25 이상 박스 없음 → NOT_DETECTED
3. 최고신뢰 박스 → 모델 class_id, confidence, bbox (PET은 외부 응답에서 PLASTIC으로 정규화)
4. 저신뢰 PET/PLASTIC이 같은 bbox의 VINYL 후보와 경쟁하면 crop 검증기 교차 확인
     └ bbox IoU·후보 비율·검증기 신뢰도·신뢰도 차이를 모두 만족할 때만 VINYL로 교정
5. 신뢰도 판정
     └ confidence < TRUST_CONF(0.55) → LOW_CONFIDENCE (일반쓰레기)
6. [허용: 플라스틱(PET 포함)/캔/종이]
     ├ 상태 멀티헤드(crop ONNX) → conditions.is_dented / has_label + 내부 foreign_material(선택)
     ├ 무게 → weight.anomaly
     └ build_guidance() → 불충족 안내. 비면 ALLOWED, 있으면 REJECTED(재처리)
7. [거부: 유리/건전지/형광등/스티로폼] → REJECTED + rejection
8. [비닐] → 무게·외부 이물질 이상이면 REJECTED + guidance, 정상이면 ALLOWED(비닐함)
9. `client_id`를 포함한 DetectResponse 조립 → 하드웨어 응답 + Spring 콜백(fire-and-forget)
```

**2단계 신뢰도 게이트:**
- `DETECT_CONF=0.25`: 박스 생성 최소선 (미만 = 미감지)
- `TRUST_CONF=0.55`: 분류 신뢰선 (미만 = 일반쓰레기)

---

## 5. 모델 구성 (2-stage + 상태 헤드)

| # | 모델 | 입력 | 출력 | 백본 | 포맷 | 상태 |
|---|------|------|------|------|------|------|
| 1 | 메인 감지 | 640px | 9클래스 bbox + 1차 품목 | YOLO26m epoch 40 | NCNN | 체크포인트 고정, Pi5 배포 기준선 |
| 2 | crop 검증기 | 320px crop | material(9) + dent(2) + label(2) + foreign_material(2) | MobileNetV3-Small 기준선 | ONNX | 기본 shadow, 저신뢰 PET/PLASTIC↔VINYL 교차 신호에만 제한적 교정 |

> 최신 경량 후보 MobileNetV4 Conv Small과 RepViT는 동일 데이터로 학습한 뒤 Pi5의
> 실제 ONNX 정확도·p50/p95 지연시간을 비교해 기준선보다 좋을 때만 교체한다.

### 멀티헤드 설계 (모델 2)
- **공유 백본 1개** + material/dent/label/foreign_material 헤드. 추론 1회로 품목과 상태를 재검증한다.
- 학습 원본은 라벨링된 쓰레기 객체가 정확히 하나인 이미지만 허용한다.
- dent는 캔+페트에 사용하고 비대상 클래스는 loss에서 마스킹한다.
- 공식 데이터의 `DIRTINESS=이물질(외부)`는 외부 오염과 라벨 부착을 하나로 합친 값이므로 두 헤드는 Qwen3-VL 두 시야 자동 합의 정답이 train/validation에 모두 모일 때까지 마스킹한다.
- 초기에는 검증 결과를 로그에만 남기는 shadow mode로 적용해 기존 응답과 Spring 콜백 계약에 영향을 주지 않는다.
- 상세 확정안: [`CROP_VERIFIER_PLAN.md`](CROP_VERIFIER_PLAN.md)

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

### 6-2. 객체 crop 멀티태스크 검증기
```
AI Hub 원본 JSON+이미지 (2TB, 직접촬영)
  └→ scripts/extract_verifier_crops.py
       · 공식 Training/Validation 분리 유지
       · ANNOTATION_INFO가 정확히 1개인 원본만 사용
       · bbox crop + 8% 패딩 → 320px letterbox
       · 9종 material 정답
       · DAMAGE → dent (원형0 / 찌그러짐·완전압착1)
       · label/foreign_material → -1 마스킹
       · DIRTINESS 변환값은 teacher 힌트용 label_proxy에만 저장
       · 고해상도 원본 복제 없이 source_path_b64+bbox로 참조
       · ※ 원본 사용 필수 (640변환본은 파일명 리네임돼 JSON 매칭 불가)
  └→ /share/Container/crops_verifier_single_v3/ (training/ + validation/ + manifest.csv)
  └→ scripts/pseudo_label_status_qwen.py
       · 같은 모델 볼륨을 공유하는 qwen3.5:9b-q4_K_M Ollama 2개/worker 2개
       · 384px tight 1장 1차 후 양성·불확실·정상 10%만 640px wider-context 1장 2차 합의
       · 2차 대상은 tight/context 두 시야가 일치하고 확신도 0.90 이상일 때만 자동 수용
       · 라벨/진짜 이물질/같은 재질 부속품 허용 규칙을 독립 판정
  └→ scripts/audit_pseudo_status.py (처리 완료·일관성·헤드 분포 자동 검사)
  └→ scripts/select_curated_verifier_manifest.py
       · 품목별 최대 10,000 training / 2,000 validation 균형 선별
       · 상태 네 조합을 round-robin해 validation 양·음성 보존
  └→ scripts/prepare_hardware_capture_dataset.py
       · 실제 키오스크 crop manifest 추가
       · training만 5배 oversampling, validation은 정확히 한 번 유지
  └→ scripts/train_verifier.py
       · MobileNetV3-Small + material/dent/label/foreign_material 헤드
       · 미라벨 상태는 masked loss
  └→ best_verifier.pt + verifier.onnx + verifier_metadata.json
```

---

## 7. 데이터셋 (AI Hub dataSetSn=71362)

- **재활용품 분류 및 선별 데이터**, 약 100만 장, JSON 라벨 (BBOX 70% / POLYGON 30%)
- 촬영: 직접촬영 70.2% (키오스크 환경 유사) + 선별영상추출 29.8% (컨베이어)
- **객체 속성 5개**: `DAMAGE` `DIRTINESS` `COVER` `TRANSPARENCY` `SHAPE`
- **핵심 제약**: `DIRTINESS=이물질(외부)`에는 상품 라벨과 실제 외부 이물질이 섞일 수 있어 두 정답으로 직접 사용하지 않는다.
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
- 임시 검증기: `VERIFIER_MODEL_PATH=weights/verifier_qwen35_mnv3_v1.onnx`
- 검증기 적용: 기본 shadow이며, `VINYL_CORRECTION_ENABLED=true`일 때 같은 bbox의
  저신뢰 PET/PLASTIC·VINYL 경쟁 사례만 제한적으로 교정
- shadow/교정 로그: `VERIFIER_SHADOW_ENABLED=true`, `logs/verifier_shadow.jsonl`
- 실제 환경 파일: `/home/one/2026expo_AI/.env`
- Spring 콜백: `SPRING_CALLBACK_URL=https://oneexpo.kro.kr/api/v1/feedback-detail/result`
- 결과 로그: `LOG_RESULTS=true`이면 `logs/results.jsonl`에 항상 기록
- 콜백 결과 로그: `logs/callbacks.jsonl`에 성공/재시도/최종 실패와 HTTP 상태 기록
- 재학습 캡처: `CAPTURE_REQUESTS=true`이면 `logs/captures/YYYY-MM-DD/`에 원본 이미지와 판정 JSON 쌍 저장
- 로그 영속화: Docker Compose의 `./logs:/app/logs` 볼륨 사용
- 콜백은 fire-and-forget 방식이며 연결 실패·타임아웃·재시도가 AI 응답에 영향을 주지 않는다.
- 타임아웃·연결 오류·HTTP 408/425/429/5xx만 최대 3회 재시도하고 HTTP 4xx는 즉시 실패 처리한다.

### 오인식 개선 루프

1. 실제 키오스크 요청의 원본 이미지와 판정 결과를 같은 `capture_id`로 수집한다.
2. YOLO가 선택한 bbox를 crop하고 Qwen3-VL teacher의 tight/context 두 판정을 수행한다.
3. 두 판정이 일치하고 확신도 0.90 이상인 단일 객체만 hard sample로 자동 수용한다. 나머지는 `-1`로 마스킹해 학습에서 제외한다.
4. 수용된 상태 표본과 유사 정상 표본을 함께 검증기 hard sample로 구성한다.
5. 실제 키오스크 배경·조명·거리 분포를 유지하고, 기존 검증셋과 별도의 키오스크 hard-case 검증셋에서 혼동행렬을 비교한다.

> 현재 모델은 재활용 데이터셋의 촬영 분포를 학습했기 때문에 흰 배경 상품 이미지,
> 렌더링 이미지, 인쇄 무늬처럼 운영 카메라와 다른 입력에서 재질 대신 배경·윤곽·색상에
> 치우쳐 오분류할 수 있다. 단순 임계값 상향만으로는 고신뢰 오분류를 해결할 수 없으므로
> 운영 이미지 수집과 재학습을 우선한다.

### 하드웨어 캡처 자동 정제 및 노트북 후보 학습

NAS의 teacher 작업과 충돌하지 않게 운영 캡처 적응 후보는 별도 노트북 환경에서
만든다. 운영 모델과 원본 캡처는 덮어쓰지 않으며 모든 산출물은 `runs/` 아래에 둔다.

1. Pi5의 `logs/captures` 이미지/JSON 쌍을 로컬로 복사한다.
2. 이미지 SHA-256으로 재요청 중복을 제거한다.
3. `audit spec`의 snapshot 개수와 SHA anchor를 확인해 다른 캡처 순서에 잘못된
   라벨이 적용되지 않게 한다.
4. 단일 물체만 YOLO 재료 학습에 사용하고, 혼합 이물질은 detector에서 제외한 뒤
   verifier 상태 manifest에만 `foreign_material=1`로 남긴다.
5. 빈 장비·사람·장비 밖의 물체는 빈 YOLO 라벨의 negative 이미지로 포함한다.
6. 동일 실물의 연속 촬영본은 하나의 `object_group`으로 묶어 train/val 누수를 막는다.
7. 기존 모델과 후보를 동일 holdout, 동일 `DETECT_CONF=0.25` / `TRUST_CONF=0.55`로
   비교한다. 실제 물체 정분류와 빈 장면 specificity가 모두 확인되기 전에는 export나
   Pi5 배포를 하지 않는다.

```powershell
python scripts/prepare_hardware_capture_dataset.py `
  --captures-dir runs/hardware_capture_prep/raw/captures `
  --audit-spec runs/hardware_capture_prep/audit_spec.json `
  --candidates runs/hardware_capture_prep/low_conf_candidates.json `
  --output-dir runs/hardware_capture_prep/dataset `
  --render-overlays

python scripts/train_hardware_candidate.py `
  --model weights/yolo26m_best.pt `
  --data runs/hardware_capture_prep/dataset/yolo/dataset.yaml `
  --project runs/hardware_capture_prep/training `
  --name candidate_freeze20 --epochs 25 --batch 8 --freeze 20

python scripts/evaluate_hardware_detector.py `
  --model runs/hardware_capture_prep/training/candidate_freeze20/weights/best.pt `
  --dataset-dir runs/hardware_capture_prep/dataset/yolo `
  --output runs/hardware_capture_prep/candidate_metrics.json `
  --thresholds 0.25 0.55
```

소규모 운영 캡처만으로 만든 모델은 hardware adapter 후보일 뿐 최종 모델이 아니다.
최종 학습은 NAS의 기존 9종 원본 train 데이터와 정제된 hardware hard sample을 함께
사용해 glass/battery/fluorescent 등 이번 캡처에 없는 클래스의 망각을 방지한다.

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

임시 검증기까지 로드된 정상 헬스 응답:

```json
{"status":"ok","models":{"main":true,"state":true,"verifier":true}}
```

---

## 9. 인프라 (NAS / 학습 환경)

| 항목 | 값 |
|------|-----|
| NAS | QNAP, 192.168.0.110, Ryzen 5 PRO 1600 (12 logical CPU) + RTX 2000 Ada 16GB |
| Docker | `/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker` (PATH에 없음, sudo 필요) |
| 학습 이미지 | `ultralytics/ultralytics:latest` |
| SMB | `\\192.168.0.110\Container` → 컨테이너 `/app` (`/share/Container`) |
| 접속 | plink `-hostkey ssh-ed25519 255 SHA256:HLPv3nWI2ozOxgy2AAzjguq1rrON+KOuWhh3zscXpy0` |
| 한글 출력 | plink가 한글 깨뜨림 → `base64 -w0` 우회 후 로컬 디코드 |
| 원본 데이터 | `/share/Container/ai_dataset/학습용_데이터/01-1.정식개방데이터/Training/` |

> 상세 접속법: `memory/nas-dataset-workflow.md`

---

## 10. 현재 운영 상태 (2026-08-03)

| 단계 | 상태 |
|------|------|
| 데이터 변환 (convert_v2) | ✅ 완료 (train 라벨 518,728 / val 68,147) |
| 메인 YOLO26m NCNN | ✅ Pi5 배포 및 로드 정상 |
| NAS 메인 학습 | ✅ epoch 40 체크포인트 보존 후 중지 (학습 프로세스 0) |
| 상태 멀티헤드 ONNX | ✅ Pi5 배포 및 로드 정상 |
| 9종 crop 검증기 | ✅ v7 선별+하드웨어 보정 MobileNetV3/ONNX 완료, 배포 파일 교체 |
| 최대 데이터 v4 추출 | ✅ 단일 객체 337,760장, 약 8.5GB 추출 완료 |
| Qwen 상태 teacher 확장 | ✅ v7 50,000건 완료, 46,913건 수용(93.826%), 오류 46건(0.092%) |
| 최종 선별 데이터 | ✅ 9종 90,274장 + 하드웨어 verifier 103장, audit 통과 |
| 최종 검증기 학습 | ✅ hard100 보정 20 epoch, best epoch 17, material val 0.95281 |
| 하드웨어 verifier holdout | ✅ 외부 계약 정확도 42.86%→74.29%, macro-F1 0.598→0.679 |
| proposal background 후보 | 🔄 실제 YOLO bbox 기반 9종+background 데이터 생성·학습, 자동 배포 금지 |
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
  services/verifier_shadow.py  YOLO/crop 검증기 비교 비동기 JSONL 로그
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
  extract_verifier_crops.py  9종 crop + 안전한 상태 manifest 생성
  audit_verifier_dataset.py  학습 전 9종·분할·파일·마스킹 무결성 검사
  audit_pseudo_status.py     자동 상태 라벨의 처리 완료·일관성·헤드 분포 검사
  merge_pseudo_status_manifest.py 동일 표본끼리 pseudo-label을 안전하게 병합
  select_curated_verifier_manifest.py 9종/상태 조합을 균형 선별한 최종 manifest 생성
  select_curated_yolo_dataset.py 고해상도 원본 기반 선택 YOLO 데이터 생성(유휴 NAS용)
  import_reviewed_captures.py 운영 캡처의 선택 정답을 crop manifest로 변환(자동 학습 기본 경로에서는 미사용)
  prepare_hardware_capture_dataset.py SHA 중복 제거 + YOLO negative/crop 상태 manifest 생성
  train_hardware_candidate.py 노트북 GPU용 보수적 하드웨어 적응 후보 학습
  evaluate_hardware_detector.py 고정 holdout의 운영 임계값별 후보 비교
  evaluate_hybrid_policy.py 운영 NCNN 선택 bbox와 negative를 포함한 교정 승격 게이트
  prepare_proposal_verifier_dataset.py 실제 YOLO bbox의 9종+background crop 데이터 생성
  evaluate_verifier.py    같은 crop holdout에서 기존/후보 ONNX와 PET 통합 외부 계약 비교
  train_verifier.py       9종+상태 학습/하드웨어 oversampling/checkpoint 이어학습/가중 선택/ONNX export
  nas/watch_evaluate_verifier.sh 학습 종료 후 고정 holdout 비교를 자동 실행
  nas/watch_proposal_verifier_pipeline.sh background 후보 생성·학습 체인(자동 배포 없음)

requirements-training.txt NAS 학습의 ONNX export 및 최신 경량 백본 의존성

docs/
  WORKFLOW.md             (이 문서)
  HARDWARE_CAPTURE_PROTOTYPE_20260801.md 노트북 적응 후보 데이터·지표·NAS 인계 게이트
  CROP_VERIFIER_PLAN.md   객체 crop 검증기 확정 구조·라벨 정책·실행 절차
  WEIGHT_KIOSK_PARAMS.md  무게 로직 + 키오스크 파라미터 체크리스트
  DATA_AUDIT.md           변환 데이터 정합성 점검
```
