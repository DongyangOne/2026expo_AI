# 객체 crop 검증기 확정안

> 확정일: 2026-07-31
> 목표: 전체 화면의 배경·색상에 끌려가는 9종 오분류를 줄이고, 같은 객체 crop에서 품목과 상태를 함께 검증한다.

## 1. 최종 구조

```text
입력 원본
  -> YOLO epoch 40: 객체 위치와 1차 품목 후보 검출
  -> 선택 bbox crop + padding + 320px letterbox
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
- 검증기는 하나의 공유 백본과 4개 헤드를 사용한다. 한 이미지에 품목·찌그러짐·라벨·이물질 속성이 동시에 존재하므로 단일 라벨 분류기로 합치지 않는다.
- crop은 비율을 왜곡하거나 물체 끝을 자르지 않도록 `letterbox`로 만든다.
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
| `label` | 검수자가 확정한 `review.has_label` | X |
| `foreign_material` | 검수자가 확정한 `review.has_foreign_material` | X |

- 기존 `DIRTINESS=이물질(외부)`는 실제 쓰레기 이물질과 제품 라벨을 확실히 구분하지 못한다.
- 따라서 `label`과 `foreign_material`은 기본값 `-1`로 두어 loss에서 마스킹한다.
- `DIRTINESS` 변환값은 `label_proxy` 열에만 보존하며, 명시적인 실험 옵션 없이는 학습하지 않는다.
- 운영 캡처 JSON에서 사람이 아래 필드를 검수한 데이터만 두 상태 헤드의 정답으로 가져온다.

```json
{
  "review": {
    "is_correct": true,
    "expected_class": "can",
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

### crop 생성

```bash
python /app/extract_verifier_crops.py \
  --dataset-dir /app/ai_dataset/학습용_데이터 \
  --output-dir /app/crops_verifier_v1 \
  --size 320 --workers 2 \
  --max-per-folder 10000 --val-max-per-folder 2000
```

공식 Training/Validation 분리를 그대로 유지하고 직접촬영 데이터만 사용한다. 출력은 `manifest.csv`와 320px crop이다.

### 검수 캡처 추가

```bash
python /app/import_reviewed_captures.py \
  --capture-dir /app/logs/captures \
  --output-dir /app/crops_verifier_reviewed_v1 \
  --size 320
```

### 기준선 학습

`dataset_info.json` 생성 후 먼저 manifest를 감사한다. 9종·두 분할·이미지 파일·분할
누수·상태 라벨 마스킹을 모두 통과해야 학습을 시작한다.

```bash
python /app/scripts/audit_verifier_dataset.py \
  --manifest /app/crops_verifier_v1/manifest.csv \
  --require-masked-status
```

```bash
python /app/train_verifier.py \
  --manifest /app/crops_verifier_v1/manifest.csv \
  --manifest /app/crops_verifier_reviewed_v1/reviewed_manifest.csv \
  --output-dir /app/runs/verifier_mnv3_v1 \
  --backbone mobilenet_v3_small --size 320
```

검수 manifest가 아직 없으면 두 번째 `--manifest`를 생략한다. 이 경우 metadata의 활성 출력은 `material`, `dent`뿐이어야 한다.

### 최신 백본 비교

```bash
python /app/train_verifier.py \
  --manifest /app/crops_verifier_v1/manifest.csv \
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

## 6. 단계별 적용 기준

1. AI Hub crop으로 `material` + `dent` 기준선을 학습한다.
2. 실제 키오스크 캡처로 9종 혼동행렬과 YOLO/검증기 불일치율을 측정한다.
3. shadow mode에서 오탐·미탐을 검수하고 `label`/`foreign_material` 정답을 축적한다.
4. 두 상태 헤드가 별도 검증셋 기준을 통과하면 런타임에서 활성화한다.
5. 검증기 불일치 시 재촬영 응답을 추가하려면 그때 Spring/하드웨어 계약을 함께 변경한다.
