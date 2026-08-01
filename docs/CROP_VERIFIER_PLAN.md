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
지원 모델이며, 구조화 JSON 생성을 위해 thinking은 끈다. 모델은 기존 `/share/Container/naco_ai/ollama` 볼륨에
한 번만 저장하며 별도 Ollama 컨테이너나 모델 볼륨을 만들지 않는다.
연속 멀티모달 요청의 prompt-cache 수정이 포함된 `ollama/ollama:0.32.0`을 고정해
사용하고, 고해상도 tight/context 두 장이 4K를 넘을 수 있으므로 `num_ctx=16384`를
사용한다. teacher 입력은 각 crop의 최대 변만 640px로 축소해 원본 bbox 증거를
보존하면서 이미지 토큰과 처리 시간을 제한한다. 작은 prototype limit은 split, 품목,
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
  --ollama-url http://naco-ollama:11434 \
  --model qwen3.5:9b-q4_K_M \
  --num-ctx 16384 \
  --image-max-side 640 \
  --limit 50 --min-confidence 0.90
```

teacher 실행은 이미 만들어 둔 `expo-verifier-train:20260731` 이미지를
`naco_naco-internal` 네트워크에 잠시 연결한다. Ollama 컨테이너와 모델 볼륨은
기존 `naco-ollama`를 그대로 사용하고 NAS 호스트에 11434 포트를 새로 공개하지 않는다.

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

### 최대 데이터 확장 작업 (2026-08-01 시작)

- `extract_verifier_single_v4_max50k_20260801`: 다중 객체를 제외한 뒤 training은
  품목당 최대 50,000장, validation은 최대 10,000장으로 추출한다.
- 실제 원본 상한은 캔·페트·플라스틱·유리 50,000장, 종이 약 39,700장,
  스티로폼 약 28,000장, 비닐 약 16,800장, 건전지 3,272장, 형광등 2,403장이다.
  부족한 두 품목은 중복 파일로 수량을 부풀리지 않고 전량 사용+학습 증강으로 보완한다.
- `pseudo_teacher_qwen35_50k_v2_20260801`: 기존 2,000건을 이어받아 라벨/외부
  이물질 자동 teacher를 총 50,000건까지 확장한다.

## 6. 단계별 적용 기준

1. 1일차에는 소량 v3 crop과 Qwen teacher 50건으로 끝까지 흐르는 prototype을 만든다.
2. tight/context 두 판정의 자동 합의율과 품목별 양성·음성 분포를 감사한다.
3. 전체 v3 crop을 만들되 500GB 여유 공간과 20GB 출력 상한을 계속 적용한다.
4. 수용된 pseudo-label로 `material`/`dent`/`label`/`foreign_material`을 학습한다.
5. 실제 키오스크 캡처로 9종 혼동행렬과 YOLO/검증기 불일치율을 측정한다.
6. shadow mode의 고신뢰 오탐·미탐 후보를 자동 재분류해 주기적으로 개선한다.
7. 두 상태 헤드가 별도 검증셋 기준을 통과하면 런타임에서 활성화한다.
8. 검증기 불일치 시 재촬영 응답을 추가하려면 그때 Spring/하드웨어 계약을 함께 변경한다.
