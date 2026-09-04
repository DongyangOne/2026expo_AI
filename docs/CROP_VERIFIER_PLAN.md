# 객체 crop 검증기 확정안

> 확정일: 2026-07-31
> 목표: 전체 화면의 배경·색상에 끌려가는 9종 오분류를 줄이고, 같은 객체 crop에서 품목과 상태를 함께 검증한다.

## 1. 최종 구조

```text
입력 원본
  -> YOLO epoch 40: 객체 위치와 1차 품목 후보 검출
  -> 선택 bbox를 고해상도 원본에서 crop + padding + 320px letterbox
  -> 경량 멀티태스크 검증기
       - material: 9종
       - dent: 찌그러짐/압착
       - label: 라벨 존재
       - foreign_material: 외부 이물질
  -> YOLO와 material 결과 비교
       - 일치·고신뢰: 기존 판정 진행
       - 불일치·저신뢰: shadow 로그 수집 후 재촬영 정책 검토
```

- YOLO는 전체 화면에서 객체를 찾는 역할로 유지한다. epoch 40 체크포인트를 기준선으로 고정하고 당장 다시 학습하지 않는다.
- 학습에는 원본 JSON의 `ANNOTATION_INFO`가 정확히 1개이고 자동 teacher도 단일 주 객체로 판정한 이미지만 사용한다.
- 검증기는 하나의 공유 백본과 4개 헤드를 사용한다. 한 이미지에 품목·찌그러짐·라벨·이물질 속성이 동시에 존재하므로 단일 라벨 분류기로 합치지 않는다.
- crop은 비율을 왜곡하거나 물체 끝을 자르지 않도록 `letterbox`로 만든다.
- 고해상도 원본은 복제하지 않는다. manifest에 원본 경로와 bbox만 기록하고, 상태 teacher가 실행될 때 메모리에서만 tight/context crop을 만든다.
- 첫 배포는 **shadow mode**로 운영한다. 검증 결과를 로그에만 남기고 기존 응답과 Spring 콜백 JSON은 바꾸지 않는다.

## 2. 기술 선택 근거

| 후보 | 판단 |
|------|------|
| MobileNetV3-Small | ONNX 변환과 Pi5 운영 기준선. 먼저 전체 파이프라인을 검증한다. |
| MobileNetV4 Conv Small | 최신 경량 후보. 동일 데이터·동일 입력 크기로 정확도/지연시간을 비교한다. |
| RepViT | 경량 CNN 후보. Pi5에서 ONNX 지연시간과 정확도가 더 좋을 때만 교체한다. |
| YOLO classification | 기본 단일 클래스 출력은 동시에 존재하는 상태 속성에 맞지 않아 최종 구조로 사용하지 않는다. |
| DINOv2 | 운영 모델보다는 적은 라벨 데이터의 teacher/특징 추출 실험 후보로 둔다. |
| EfficientAD/Dinomaly | 미지의 이상을 찾는 보조 후보. 정상 상품 라벨까지 이상으로 볼 수 있어 초기 외부 이물질 판정기로 바로 쓰지 않는다. |

참고 자료:

- MobileNetV4: <https://arxiv.org/abs/2404.10518>
- RepViT: <https://openaccess.thecvf.com/content/CVPR2024/html/Wang_RepViT_Revisiting_Mobile_CNN_From_ViT_Perspective_CVPR_2024_paper.html>
- Ultralytics classification: <https://docs.ultralytics.com/tasks/classify/>
- DINOv2: <https://arxiv.org/abs/2304.07193>
- EfficientAD: <https://openaccess.thecvf.com/content/WACV2024/papers/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.pdf>
- Dinomaly: <https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_Dinomaly_The_Less_Is_More_Philosophy_in_Multi-Class_Unsupervised_Anomaly_CVPR_2025_paper.pdf>

## 3. 데이터 라벨 정책

| 출력 | 초기 학습 데이터 | 초기 활성화 |
|------|------------------|:-----------:|
| `material` | AI Hub 직접촬영 9종의 bbox crop | O |
| `dent` | 캔·페트의 `DAMAGE` | O |
| `label` | Qwen3-VL 두 시야 자동 합의 pseudo-label | 조건부 |
| `foreign_material` | Qwen3-VL 두 시야 자동 합의 pseudo-label | 조건부 |

- 공식 가이드의 `DIRTINESS=이물질(외부)` 정의는 **외부 오염 또는 라벨 부착**을 하나로 묶는다. 스티커와 라벨도 외부 이물질로 기록하도록 되어 있어 기존 JSON만으로 두 상태를 분리할 수 없다.
- 따라서 `label`과 `foreign_material`은 기본값 `-1`로 두어 loss에서 마스킹한다.
- `DIRTINESS` 변환값은 `label_proxy` 열에만 보존하며, 명시적인 실험 옵션 없이는 학습하지 않는다.
- 로컬 Qwen3-VL이 tight/context 두 시야에서 같은 결론을 내리고 확신도 0.90 이상일 때만 두 상태 헤드의 자동 정답으로 가져온다. `label`과 `foreign_material`은 train/validation에 `0`과 `1`이 모두 있을 때만 활성화된다.
- 자동 teacher는 `라벨만`, `외부 이물질만`, `둘 다 있음`, `둘 다 없음` 네 조합을 서로 다른 정답으로 기록한다.
- 상태 pseudo-label 대상은 실제 키오스크 투입 후보인 `can`, `pet`, `paper`, `plastic`, `vinyl`로 제한한다. 9종 품목 재검증 데이터는 그대로 유지한다.
- 분리 가능한 PET 슬리브·스티커만 `label`이다. 캔에 인쇄된 로고와 그래픽은 라벨이 아니다.
- 본체와 다른 재질이 부착·혼합됐거나 음식물·흙·천·금속·종이 등 실제 오염이 보일 때만 `foreign_material=1`이다.
- 본체와 같은 넓은 재활용 재질의 빨대·뚜껑·링은 허용한다. `foreign_material=0`인 음성 샘플로 사용한다.
- 플라스틱/PET 테이크아웃 컵의 종이 띠지처럼 본체와 다른 재질이 부착된 경우는 `foreign_material=1`이다.
- 로컬 VLM teacher의 확신도 0.90 이상만 0/1로 수용한다. 애매함·두 시야 불일치·다중 객체·파싱 실패는 모두 `-1`로 마스킹해 자동 제외한다.

```json
{
  "review": {
    "is_correct": true,
    "expected_class": "can",
    "is_single_object": true,
    "is_dented": true,
    "has_label": false,
    "has_foreign_material": false,
    "notes": null
  }
}
```

## 4. NAS 실행 절차

기본 `ultralytics/ultralytics:latest` 이미지에는 ONNX exporter 패키지가 없을 수 있다.
NAS에서는 `Dockerfile.training`으로 `expo-verifier-train:20260731` 이미지를 한 번 만들고
재사용한다. `timm`은 MobileNetV4/RepViT 비교에 사용하고, `onnx`는 최종 모델
내보내기에 필요하다.

```bash
docker build -f Dockerfile.training -t expo-verifier-train:20260731 .
```

### 1일차 prototype: crop 생성

```bash
python /app/extract_verifier_crops.py \
  --dataset-dir /app/ai_dataset/학습용_데이터 \
  --output-dir /app/crops_verifier_single_v3_smoke_20260731 \
  --size 320 --workers 8 \
  --max-per-folder 20 --val-max-per-folder 10 \
  --min-free-gb 500 --max-output-gb 2
```

공식 Training/Validation 분리를 그대로 유지하고 직접촬영 데이터만 사용한다. 출력은
`manifest.csv`와 320px crop뿐이다. 고해상도 원본 crop은 저장하지 않으며
`source_path_b64`와 원본 bbox로 참조한다. 기본값으로 다중 객체 이미지를 제외하며,
운영용 데이터 생성에서 `--allow-multiple-objects`는 사용하지 않는다.

추출기는 기본적으로 NAS 여유 공간 500GB를 남기고 이번 crop 출력이 20GB를 넘으면
중단한다. 1일차 smoke는 더 작은 2GB 상한을 사용한다.

```bash
python /app/scripts/audit_verifier_dataset.py \
  --manifest /app/crops_verifier_single_v3_smoke_20260731/manifest.csv \
  --require-single-object --require-masked-status
```

### 1일차 prototype: 로컬 상태 teacher

외부 API로 이미지를 보내지 않고 NAS의 기존 `naco-ollama`와 RTX 2000 Ada에서
`qwen3.5:9b-q4_K_M`를 실행한다. 이 모델은 2026년 공개된 Qwen3.5 계열의 이미지 입력
지원 모델이며, 구조화 JSON 생성을 위해 thinking은 끈다. 모델은 기존
`/share/Container/naco_ai/ollama` 볼륨에 한 번만 저장한다.
연속 멀티모달 요청의 prompt-cache 수정이 포함된 `ollama/ollama:0.32.0`을 고정해
사용한다. 5만건 확장 작업은 16GB VRAM에 9B Q4 모델 두 개를 동시에 적재할 수 있도록
인스턴스당 `num_ctx=8192`를 사용한다. teacher 입력은 384px tight 한 장의 1차 판정 후
양성·불확실 샘플과 결정론적으로 선택한 정상 10%만 640px wider-context 한 장으로
2차 합의를 수행한다. tight와 wider-context라는 서로 다른 시야의 판정을 비교하면서
첫 tight 이미지는 다시 보내지 않는다. 확실한 정상 샘플의 불필요한 context 인코딩과
2차 판정을 생략하되 원본 bbox 증거는 유지한다.
원본 `DIRTINESS`는 Qwen prompt 힌트로는 유지하지만 공식 값이 라벨과 외부 이물질을
구분하지 못하므로 2차 강제 조건으로 사용하지 않는다.
`label` 학습 대상이 아닌 can/paper/vinyl의 `label_only`는 정상 음성과 동일하게 10%
감사만 수행한다. PET/plastic의 라벨 양성과 모든 `foreign_only`/`both`는 640px 2차를
반드시 수행한다.
모델 응답은 자유문장 reason과 decision에서 결정 가능한 중복 boolean을 생략하고,
`decision`/`single`/`confidence`/근거 enum의 compact wire key만 생성한다. parser에서
기존 긴 필드 구조로 즉시 복원하므로 JSONL·manifest 계약은 유지하면서 생성 토큰
병목만 줄인다.
작은 prototype limit은 split, 품목,
기존 `DIRTINESS` 힌트를 round-robin해 깨끗함/내부/외부/전체 오염 후보가 한쪽으로
쏠리지 않게 한다.
기존 Compose는
`/share/Container/container-station-data/application/naco/docker-compose.yml`과
`docker-compose.resource.yml`을 그대로 사용하며, teacher는 `naco_naco-internal`
네트워크에서만 Ollama를 호출한다.

QNAP의 자동 NVIDIA runtime 주입은 모델 로딩 시 CUDA fault-buffer 초기화가
실패할 수 있어, 기존 YOLO 학습 컨테이너와 같은 direct-driver 방식을
사용한다. `naco-ollama`는 `runtime: runc`, NVIDIA device node 직접 연결,
다음 QPKG 라이브러리 read-only bind를 유지한다.

- `/share/CACHEDEV1_DATA/.qpkg/NVIDIA_GPU_DRV/usr/lib` → `/usr/local/nvidia/lib64`
- `/share/CACHEDEV1_DATA/.qpkg/NVIDIA_GPU_DRV/cuda-12.9/lib64` → `/usr/local/cuda/lib64`
- `LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/lib/ollama/cuda_v12`

```bash
cd /share/Container/container-station-data/application/naco
docker compose -f docker-compose.yml -f docker-compose.resource.yml up -d naco-ollama
docker exec naco-ollama ollama pull qwen3.5:9b-q4_K_M

docker run --rm --network naco_naco-internal \
  -v /share/Container:/app \
  -v /share/Container/extract_verifier_crops_v3.py:/app/extract_verifier_crops.py:ro \
  expo-verifier-train:20260731 \
  python /app/pseudo_label_status_qwen.py \
  --manifest /app/crops_verifier_single_v3_smoke_20260731/manifest.csv \
  --backend ollama \
  --ollama-url http://naco-ollama:11434,http://naco-ollama-worker2:11434 \
  --model qwen3.5:9b-q4_K_M \
  --num-ctx 8192 \
  --workers 2 \
  --adaptive-consensus \
  --adaptive-confidence 0.90 \
  --adaptive-first-image-max-side 384 \
  --adaptive-negative-audit-rate 0.10 \
  --image-max-side 640 \
  --limit 50 --min-confidence 0.90
```

teacher 실행은 이미 만들어 둔 `expo-verifier-train:20260731` 이미지를
`naco_naco-internal` 네트워크에 잠시 연결한다. `naco-ollama`와
`naco-ollama-worker2`는 같은 모델 볼륨을 읽고 각각 worker 하나를 처리한다. Qwen3.5는
Ollama 0.32에서 한 서버의 parallel slot을 지원하지 않으므로 서버를 둘로 분리했으며,
NAS 호스트에 11434 포트를 새로 공개하지 않는다.

#### 2026-07-31 teacher 모델 선택

같은 8B급 이미지 모델 중 `Qwen3-VL-8B-Instruct`, `MiniCPM-V 4.5`,
`Qwen3.5-9B`를 비교했다. 정확히 8B인 `MiniCPM-V 4.5`는 모바일·영상 효율이
강점이지만 Qwen3-8B 기반이고, 이 파이프라인은 영상보다 작은 라벨/이물질의 정지
이미지 판별과 엄격한 JSON 지시 준수가 중요하다. Qwen3.5-9B는 더 최신 세대이고
공식 시각·공간·지시 준수 지표가 높으며 Ollama의 이미지 입력 및 thinking 비활성화를
지원한다. 따라서 16GB VRAM에 맞는 6.6GB Q4_K_M 태그를 고정한다.

- 1순위: `qwen3.5:9b-q4_K_M`
- 대체 후보: `minicpm-v4.5:8b` (Qwen3.5 실데이터 smoke 실패 시에만 사용)
- Qwen3.6은 공식 소형 8B/9B가 없고 27B부터라 이 NAS의 16GB VRAM 범위에서 제외

산출물은 `pseudo_status.jsonl`과 `manifest_with_pseudo_status.csv`이다. JSONL은 매
이미지 직후 이어 쓰므로 중단 후 재실행해도 이미 판정한 행은 건너뛴다. 실제 학습은
두 시야 자동 합의와 데이터 감사가 `라벨/진짜 이물질/같은 재질 부속품 허용` 규칙을
통과한 뒤 자동으로 시작한다. 사람 검토 단계는 두지 않는다.

```bash
python /app/scripts/audit_pseudo_status.py \
  --manifest /app/crops_verifier_single_v3_smoke_20260731/manifest_with_pseudo_status.csv \
  --min-coverage 0.50
```

전체 학습에서는 `--require-complete --require-ready-heads`를 추가한다.
`label`과 `foreign_material` 각각 Training/Validation에 0과 1이 모두 없으면 해당
상태 헤드의 학습 진입을 자동 차단한다. prototype의 50건 제한 실행은
미처리 행을 허용하고, 실제 처리한 행 중 수용률만 `--min-coverage`로 검사한다.

### prototype 통과 후 전체 정제

```bash
python /app/extract_verifier_crops.py \
  --dataset-dir /app/ai_dataset/학습용_데이터 \
  --output-dir /app/crops_verifier_single_v3 \
  --size 320 --workers 8 \
  --max-per-folder 10000 --val-max-per-folder 2000 \
  --min-free-gb 500 --max-output-gb 20
```

### 기준선 학습

`dataset_info.json` 생성 후 먼저 manifest를 감사한다. 9종·두 분할·이미지 파일·분할
누수·상태 라벨 마스킹을 모두 통과해야 학습을 시작한다.

```bash
python /app/scripts/audit_verifier_dataset.py \
  --manifest /app/crops_verifier_single_v3/manifest.csv \
  --require-single-object \
  --require-masked-status
```

```bash
python /app/train_verifier.py \
  --manifest /app/crops_verifier_single_v3/manifest_with_pseudo_status.csv \
  --output-dir /app/runs/verifier_single_mnv3_v2 \
  --backbone mobilenet_v3_small --size 320
```

라벨 또는 외부 이물질 자동 정답이 한쪽 값만 있거나 validation에 없으면
해당 헤드는 자동 비활성화된다.

### 최신 백본 비교

```bash
python /app/train_verifier.py \
  --manifest /app/crops_verifier_single_v3/manifest_with_pseudo_status.csv \
  --output-dir /app/runs/verifier_mnv4_v1 \
  --backbone mobilenetv4_conv_small.e2400_r224_in1k --size 320
```

MobileNetV4/RepViT 채택은 validation macro-F1, 오분류 혼동행렬, ONNX 크기, Pi5 p50/p95 지연시간을 모두 비교한 뒤 결정한다.

### 4.1 MobileNetV4-Conv-Small 비교 실측 (2026-09-05, 종결)

위 계획을 8월 1일부터 방치하다 2026-09-05에 실제로 실행했다. NAS GPU가 유휴라
`verifier_curated_v7_mnv3_20260803`과 동일 데이터·하이퍼파라미터로 세 후보를
학습해 하드웨어 홀드아웃 35건(`hardware_capture_prep_20260803/dataset_v2/verifier/`,
학습에 전혀 쓰이지 않은 validation split)으로 배포본과 비교했다.

| 모델 | material(응답기준) | dent | label | McNemar p (vs 배포본) |
|---|---:|---:|---:|---:|
| 배포본(hard100, MobileNetV3-Small) | 74.3% | 92.3% | 100% | — |
| MobileNetV3-Small + kiosk-augmentation | 62.9% | 92.3% | 94.7% | 0.219 |
| MobileNetV4-Conv-Small + kiosk-augmentation | 71.4% | 100% | 94.7% | **1.000** |
| MobileNetV4-Conv-Small (증강 없음, 순수 백본 비교) | 62.9% | 100% | 94.7% | 0.289 |

**결론: 세 후보 모두 배포본을 통계적으로 이기지 못했다(전부 p>0.05).** 특히
MobileNetV4 대 배포본은 p=1.0으로 완전한 동률이다. `--kiosk-augmentation`도
material 정확도를 개선하지 못했고 MobileNetV3에서는 오히려 악화시켰다.
n=35라는 검증셋 크기 자체가 이 정도 차이를 구분하지 못한다 — 백본이나 증강
선택의 문제가 아니라 **검증셋 크기가 병목**이라는 뜻이다. arXiv:2607.01984의
"MobileNetV3-Small은 심한 자원 제약 하에서 가장 강한 후보"라는 결론과 부합한다.

**결정: MobileNetV3-Small(배포본)을 유지한다.** 백본 재탐색은 검증셋이 커지기
전까지 우선순위가 아니다. 다운로드한 네 ONNX와 원본 로그는
`weights/verifier_qwen35_mnv3_v1.onnx`(배포본)만 남기고 나머지는 저장소에
커밋하지 않는다. 재현이 필요하면 이 절의 학습 커맨드와 NAS
`runs/verifier_kiosk_aug_20260904/`, `runs/verifier_mnv4_kiosk_aug_20260904/`,
`runs/verifier_mnv4_noaug_20260905/`를 참조한다.

## 5. 현재 체크포인트

- 기존 NAS 학습은 epoch 41 진행 중 중지했고, **epoch 40 결과를 최종 기준선**으로 보존했다.
- 보존 위치: `/share/Container/runs/trash_v2_full-2/weights/final_epoch40_20260731/`
- 파일: `best_epoch40.pt`, `last_epoch40.pt`, `results_through_epoch40.csv`
- epoch 40 지표: precision `0.88443`, recall `0.84564`, mAP50 `0.91759`, mAP50-95 `0.88858`
- 중지 후 YOLO 학습 프로세스 수: `0`

### 임시 crop 검증기 (2026-08-01)

- Qwen teacher 2,000건 중 1,819건을 자동 수용하고 37건의 파싱 오류는 `-1`로
  마스킹해 학습에서 제외했다.
- 단일 객체 crop 162,305장으로 MobileNetV3-Small 50 epoch 학습과 ONNX export를
  완료했다. 최적 validation 정확도는 material `0.9540`, dent `0.9221`,
  label `0.8770`, foreign_material `0.9802`다.
- 외부 이물질 validation 양성은 11장뿐이므로 해당 정확도만으로 실판정을 활성화하지
  않는다. 임시 ONNX는 기존 YOLO 뒤 shadow mode로 실행해 응답/콜백과 분리한다.
- 상품 이미지 펩시 캔 smoke에서는 임시 검증기도 `plastic`으로 오판했다. 이는 운영
  분포에서 바로 판정을 덮어쓰지 않고 YOLO/검증기 불일치 로그를 먼저 모아야 한다는
  근거다.

### 최대 데이터 확장 및 v7 완료 (2026-08-03)

- `extract_verifier_single_v4_max50k_20260801`: 다중 객체를 제외한 뒤 training은
  품목당 최대 50,000장, validation은 최대 10,000장으로 추출한다.
- 실제 원본 상한은 캔·페트·플라스틱·유리 50,000장, 종이 약 39,700장,
  스티로폼 약 28,000장, 비닐 약 16,800장, 건전지 3,272장, 형광등 2,403장이다.
  부족한 두 품목은 중복 파일로 수량을 부풀리지 않고 전량 사용+학습 증강으로 보완한다.
- `pseudo_teacher_qwen35_50k_v2_20260801`: 기존 2,000건을 이어받아 라벨/외부
  이물질 자동 teacher를 총 50,000건까지 확장했다.
- `pseudo_teacher_qwen35_50k_adaptive_dual_v7_20260801`: 9B 모델 두 인스턴스를 RTX 2000
  Ada에 동시에 적재하고 worker를 각 서버에 고정한다. 384px tight 단일 이미지 1차와
  조건부 640px wider-context 단일 이미지 2차 합의로 기존 전 샘플 640px 2-pass 대비 처리 시간을
  줄였다. JSONL은 매 샘플 flush하므로 전환 전 결과와 오류도 그대로 이어받았다.

#### v7 최종 결과

- 모델 warm 이후 5.35분 동안 110건을 추가해 `20.55건/분`을 기록했다. 기존 9B
  순차 640px 2-pass의 약 `6.5건/분` 대비 `3.16배`다.
- 110건 중 조건부 2차 판정은 38건(`34.5%`), 신규 오류는 0건이었다.
- `pseudo_teacher_qwen35_50k_adaptive_dual_v7_20260801`은 2026-08-03 00:00:10 KST에
  exit code 0으로 종료했다. 고유 50,000건 중 46,913건(`93.826%`)을 자동 수용했고,
  teacher 오류는 46건(`0.092%`), 무효 행은 0건이다.
- 50,000건은 계획한 상태 teacher 상한이다. 전체 manifest 162,305건 중 나머지
  57,945개 상태 대상 행은 누락이 아니라 의도적으로 미처리 상태를 유지한다.
- pseudo-label audit은 coverage `0.93826`, 최대 오류율 `0.01`, 두 상태 head의
  train/validation 양·음성 존재 조건을 모두 통과했다.

#### 최종 선별 학습 및 하드웨어 보정 (2026-08-03 완료)

- 전체 데이터를 다시 무차별 학습하지 않고
  `manifest_curated_v7_balanced_20260803.csv` 90,274장을 사용한다. training은
  캔/PET/플라스틱/유리 각 10,000장, 종이 9,982장, 스티로폼 10,000장, 비닐
  9,995장, 건전지 3,225장, 형광등 2,379장이고 validation은 별도로 유지한다.
- 실제 키오스크 crop 103장은 `hardware_capture_prep_20260803/dataset_v2`에서 합쳤다.
  validation 35장은 항상 한 번만 유지하고 training 68장만 증강 oversampling한다.
- 1차 `train_verifier_curated_v7_mnv3_20260803`은 18 epoch에서 조기 종료됐다. 하드웨어
  material 정확도는 기존 `31.43%`에서 `51.43%`로 올랐지만 macro-F1이
  `0.480`에서 `0.453`으로 내려가 전 품목 배포 게이트는 통과하지 못했다.
- 원인은 5배 oversampling이 전체 75,921 training 행 중 340행(`0.45%`)에 불과해
  영수증·납작한 트레이·흰 포장 스티로폼 같은 실기기 분포를 충분히 반영하지 못한 것이다.
- 1차 최고 체크포인트에서 이어서 하드웨어 training만 100배로 높이고, 기존 9종
  75,581장을 계속 replay한 `train_verifier_curated_v7_hard100_mnv3_20260803`을
  20 epoch 학습했다. 학습률은 `3e-4`, material loss weight는 `2.0`, 체크포인트
  선택 가중치는 material/dent/label/foreign=`4/1/2/1`이다.
- 최종 최고는 epoch 17이며 원본 validation 정확도는 material `0.95281`, dent
  `0.88587`, label `0.88050`, foreign_material `0.99708`이다.
- 하드웨어 holdout 35장의 내부 9종 정확도는 배포 모델 `31.43%` → 최종 후보
  `71.43%`, macro-F1은 `0.480` → `0.676`이다. 외부 계약에서 PET를 plastic으로
  합치면 정확도 `42.86%` → `74.29%`, macro-F1 `0.598` → `0.679`다.
- 하드웨어 label은 `31.58%` → `100%`, dent는 `92.31%` 유지, 외부 이물질 음성
  오탐은 0건이다. 다만 하드웨어 외부 이물질 양성은 0장이므로 해당 출력의 운영
  활성화 근거로 사용하지 않는다.
- 최종 ONNX는 기존 추적 경로 `weights/verifier_qwen35_mnv3_v1.onnx`를 교체한다.
  런타임은 기존처럼 shadow와 제한적 PET/plastic↔vinyl 교정만 사용하며 Spring JSON
  계약은 바뀌지 않는다.
- 고해상도 2TB 원본을 다시 읽는 YOLO 선별 생성은 NAS 부하가 커 현재 실행 경로에서
  제외했다. 필요하면 `select_curated_yolo_dataset.py`를 유휴 시간에 worker 2 이하로
  실행한다. 중간 생성물만 제거했으며 원본 데이터는 건드리지 않았다.

### 실제 YOLO proposal + background 후보 (2026-08-25)

기존 9종 검증기는 정답 bbox crop에서는 강하지만, 실제 YOLO가 배경이나 물체 일부를
잘못 잡은 crop에도 반드시 9종 중 하나를 출력한다. 운영 NCNN과 실제 선택 bbox로
positive 35장·negative 6장을 다시 평가했을 때 범용 교정 정책은 승격 게이트를 통과하지
못했다. 따라서 기존 YOLO와 현재 제한적 교정은 유지하고 다음 후보를 별도로 학습한다.

- 입력은 정답 crop이 아니라 운영 YOLO가 실제 생성한 proposal bbox만 사용한다.
- 원본 정답이 0개 또는 1개인 이미지만 허용하고 다중 객체 이미지는 제외한다.
- proposal IoU가 `0.50` 이상이면 정답 품목, `0.10` 이하면 열 번째 내부 클래스
  `background`, 그 사이는 애매한 표본으로 제외한다.
- `background`는 학습·검증기 내부 전용이며 API와 Spring callback에는 절대 반환하지
  않는다. background 판정으로 기존 저신뢰 결과를 `ALLOWED`로 승격하지 않는다.
- 9종 체크포인트의 backbone·상태 head·material 첫 9행을 이어받고 background 행만
  새로 초기화한다. 상태 head 출력 계약은 그대로 유지한다.
- NAS 자동 체인은 `scripts/nas/watch_proposal_verifier_pipeline.sh`로 proposal 생성과
  후보 학습까지만 수행한다. 운영 가중치를 자동 교체하지 않는다.

승격 평가는 `scripts/evaluate_hardware_detector.py`가 기록한 운영 NCNN 선택 bbox와
`scripts/evaluate_hybrid_policy.py`를 사용한다. 정답 bbox crop만으로 만든 결과는 배포
근거로 인정하지 않는다. 실기기 외부 정확도 `+5%p`, macro-F1 비하락, 품목별 recall
하락 `1%p` 이내, negative specificity 비하락, harmful correction 0건, negative의
허용 계열 승격 0건을 모두 만족해야 한다.

### 1차 proposal 후보 평가와 clean v2 재학습 (2026-08-26)

- 1차 10-class 후보는 proposal validation 정확도 `85.663%`, macro-F1 `88.803%`였지만
  고정 하드웨어 41장에서는 기존 모델과 동일하게 전체 정확도 `60.98%`, positive 정확도
  `65.71%`로 개선 폭이 `0%p`였다. 배포 게이트에서 거부했고 운영 모델은 변경하지 않았다.
- 원인은 학습 background 20,000장 중 19,985장이 GT가 있는 프레임의 low-IoU
  proposal이어서, 실제 다른 재활용품까지 background로 잘못 가르친 데이터 오염이었다.
  또한 모든 proposal과 `conf=0.05`를 사용해 운영의 최고 신뢰 bbox 하나와 분포가 달랐다.
- clean 정책은 `--proposal-selection runtime-top1`, `--conf 0.25`,
  `--background-policy no-ground-truth-only`다. GT가 있는 프레임의 low-IoU proposal은
  자동 background 라벨로 쓰지 않는다.
- 기존 clean manifest에서 상용 선별 폴더에 섞인 하드웨어 복제본 84행도 제거했다.
  재학습 본체는 AI Hub 9종 단일 객체 crop 89,779장(training 76,463 / validation
  13,316)이며, 하드웨어는 단일 객체 원본 60장과 실제 runtime proposal hard sample만
  별도 가중한다. source object가 2개인 비닐 8장은 제외했다.
- 내부 background는 하드웨어 training 원본 3장을 source 기준으로 2 train / 1
  validation으로 분리한다. 고정 하드웨어 holdout 41장은 checkpoint 선택과 gradient에
  모두 사용하지 않는다.
- 실제 loader는 training 82,315 / validation 13,317이며, 반복 적용된 하드웨어 행은
  약 `7.1%`다. 소수 이미지를 수백 배로 과적합시키던 초안 비율은 사용하지 않는다.
- background veto는 실제 positive를 억제하면 기존 YOLO도 오답이었는지와 무관하게
  항상 harmful로 센다. `confidence=0.90`, verifier-over-YOLO `margin=0.30`을 미리
  고정하고 harmful 0건, wrong-to-wrong 0건과 기존 배포 지표를 모두 요구한다.
- 통과해도 `offline_candidate_ready.txt`만 기록한다. 10-class metadata와 선택 정책을
  함께 보존하고, 실제 런타임 정책 구현 및 독립 end-to-end 하드웨어 검증 전에는 Pi나
  production 모델을 교체하지 않는다.

현재 AI Hub clean 목록은 이전 cap 적용 proposal manifest 안에서 source별 top-1을 다시
선택한 빠른 보정본이다. 원본 이미지에 대해 YOLO 전체 후보를 다시 추론한 완전한 runtime
top-1 목록은 아니므로, 이 후보가 고정 게이트를 통과하지 못하면 원본 source 재추론으로
다음 데이터 버전을 만든다.

새 운영 쓰레기는 이미지 SHA-256 중복 제거, 동일 실물 `object_group` 분할, tight/context
teacher 합의와 confidence 기준을 통과한 단일 객체만 다음 hard sample에 추가한다. 이 구조는
기존 9종의 새 모양·브랜드·조명 변화에는 대응하지만, 처음 보는 새 재질을 즉시 자동 학습해
보장하는 구조는 아니다. 새 클래스와 저신뢰·미분류는 안전한 비허용 경로로 유지하고 충분한
고유 표본과 독립 검증셋이 쌓인 뒤에만 학습 계약을 확장한다.

### v3 저신뢰 proposal·엄격 lineage 개선 루프 (2026-08-27)

v2 후보가 배포 게이트에서 거부된 뒤에도 개선을 중단하지 않고, 검출 recall과 crop
검증을 서로 분리해 다시 진행한다. 고정 하드웨어 41장은 현재 진단용으로만 사용하며
threshold 선택과 최종 배포 승인용 blind test로 간주하지 않는다.

- 같은 PyTorch YOLO와 NMS IoU `0.70`에서 candidate confidence를 `0.25`에서
  `0.10`으로 낮추면 진단 정확도가 `25/41 (60.98%)`에서 `28/41 (68.29%)`로
  올랐다. negative specificity는 `3/6 (50%)`로 같았다. 이 수치는 v3 개발 근거일
  뿐 독립 검증 결과가 아니다.
- `configs/detector_inference_v3.json`에 입력 크기, 후보 confidence, NMS, bbox
  선택 순서, crop padding·letterbox를 고정하고 파일 SHA-256을 calibration policy와
  blind gate에 함께 묶는다.
- 단일 객체 22만여 source를 YOLO로 다시 추론한다. 같은 바이트의 복제본은 SHA-256으로
  한 번만 처리하고, 동일 SHA에 서로 다른 정답이 붙은 데이터는 어느 한쪽을 임의로
  택하지 않고 전부 격리한다.
- 학습 crop은 `runtime-top1`, `conf=0.10`, `NMS IoU=0.70`, padding `0.08`,
  320px letterbox로 다시 만든다. GT가 있는 프레임의 low-IoU proposal은 background로
  사용하지 않고, GT가 없는 source만 background 자동 정답으로 허용한다.
- v3 검증기는 background를 10번째 material class로 넣지 않는다. 모든 proposal을
  보는 binary `objectness(material/background)` head와, positive crop만 loss에
  참여하는 9-class `material` head를 별도로 학습한다.
- 기존 proposal manifest는 source와 crop 바이트를 다시 해시한 strict schema로
  변환한다. `sample_id`, `source_sha256`, `image_sha256`, `object_group`,
  `capture_session`, `role`, `fold`, `origin` 중 하나라도 train/model-validation을
  가로지르면 학습을 차단한다.
- 8월 운영 캡처 teacher queue 20건 중 19건이 다중 판정 합의에 도달했고, 보수적
  기준으로 15건·12개 object group만 train 전용으로 수용했다. 구성은 plastic 14,
  paper 1로 편향되어 있으므로 AI Hub 9종 replay를 대체하지 않고 hardware adaptation
  표본으로만 사용한다. 실제 운영 bbox와 같은 방식으로 materialize한 15 crop은
  rejection 없이 strict component audit를 통과했다.
- manifest 행을 복제하지 않고 `origin` 기반 deterministic weighted sampler로 실제
  하드웨어와 운영 crop의 기대 표본 비중을 합계 3~8%로 제한한다. validation은 가중하지
  않는다.
- calibration은 calibration role만 사용해 temperature와 임계값을 고정한다. blind
  evaluator에는 threshold CLI가 없으며, calibration과 source/object group이 겹치면
  즉시 실패한다. offline blind gate를 통과해도 production 교체 권한은 생기지 않는다.
- 독립적인 신규 하드웨어 blind set과 end-to-end 응답/Spring callback/Pi 지연시간
  게이트까지 통과하기 전에는 현재 Pi 모델과 production 파일을 교체하지 않는다.
- calibration 또는 독립 blind 지표가 거부되면
  `scripts/plan_verifier_next_iteration.py`가 sample 수가 아닌 `object_group` 단위로
  검출 미탐·false positive·background 오판·품목 혼동·harmful correction을 집계한다.
  입력 report와 학습 metadata SHA-256에 묶인 `next_offline_iteration_plan.json`에는
  새롭고 서로 분리된 hard positive/negative 수집량, 전체 3~8% 범위의 adaptation
  sampler 목표, 필요할 때만 1회의 stronger-backbone ablation을 기록한다.
- 계획 생성은 재학습이나 배포가 아니다. 기존 파일을 덮어쓰지 않고 plan 하나당 base
  candidate 1회와 선택적 backbone ablation 1회만 허용한다. 거부된 blind 표본은
  postmortem 전용이며 다음 candidate의 학습·calibration·architecture 선택에 넣지 않는다.
  threshold와 gate 기준은 blind 결과로 바꾸지 않고 새 calibration partition에서만 다시
  고정하며, 다음 평가는 새로 수집한 독립 blind set을 요구한다.

## 6. 단계별 적용 기준

1. 1일차에는 소량 v3 crop과 Qwen teacher 50건으로 끝까지 흐르는 prototype을 만든다.
2. tight/context 두 판정의 자동 합의율과 품목별 양성·음성 분포를 감사한다.
3. 전체 v3 crop을 만들되 500GB 여유 공간과 20GB 출력 상한을 계속 적용한다.
4. 수용된 pseudo-label의 품목별 균형 표본과 하드웨어 hard sample로
   `material`/`dent`/`label`/`foreign_material`을 학습한다.
5. 실제 키오스크 캡처로 9종 혼동행렬과 YOLO/검증기 불일치율을 측정한다.
6. shadow mode의 고신뢰 오탐·미탐 후보를 자동 재분류해 주기적으로 개선한다.
7. 두 상태 헤드가 별도 검증셋 기준을 통과하면 런타임에서 활성화한다.
8. 검증기 불일치 시 재촬영 응답을 추가하려면 그때 Spring/하드웨어 계약을 함께 변경한다.
