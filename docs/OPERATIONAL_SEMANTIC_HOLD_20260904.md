# 운영 후보 9장 의미 검증 보류 — 2026-09-04

## 현재 결정

`operational_refresh_80bf78a_20260904_101000`의 accepted 운영 후보 **9장 전부를
semantic hold로 유지한다.** 일부 사진의 잘못된 재질 합의와 다른 배경 물체를 가리킨
reference bbox가 확인됐다. 이 후보들을 맞는 정답으로 간주해 학습하거나,
현재 운영/Pi 모델을 교체하지 않는다. 원본 label·provider·봉인 contract는 보존한다.

이번에는 실제 YOLO 9장 관측과 저장된 teacher 결정/모델 metadata를 확인했다.
**schema·SHA·source 연결 검증 통과는 semantic accuracy 통과가 아니다.**
실제 학습은 시작하지 않았으며, 새 blind 평가나 production 배포를 승인하지 않았다.

## 1. 실제 증거

작업 루트: `/share/Container/operational_refresh_80bf78a_20260904_101000`.
원본 teacher JSONL은 SMB 목록 조회가 가능했지만 내용 읽기 권한이 없어 권한을
바꾸지 않았다. 기존 승인된 NAS 실행 경로로 필요한 세 사례의 결정 필드만 추출한
소형 보고서를 읽고 SHA를 확인했다. 이미지/base64·client ID·인증정보는 이 문서에 넣지 않는다.

| 작업 루트 기준 증거 | SHA256 |
|---|---|
| `work/ctx8192/yolo_observation_20260904_retry01/observations.json` | `0e007baa29d16f42f9def94bc6283b7fdf3bfd8e157b6ab1ffb3276843263f60` |
| `teacher_semantics_inspect_20260904.json` | `2adac42f1cfceb9bf53261fb2a50fd97bdbbfa54c69ff03113f65f423ad006e8` |
| `teacher_vision_capabilities_20260904.json` | `acac8ceaaa7d8773140108eb041940b4530492e4021c4f520487e2731f483e00` |

좁은 조회가 대상으로 삼은 원본 `work/ctx8192/teacher_labels.jsonl` SHA는
`dd2bf293bb354f1cf809f38d61cf53ec27996dc9af6f00f6888ae7ad48ac8aea`다.
이는 localization을 덧붙인 `teacher_localized_ac.jsonl`과 다른 파일이다.

세 사례의 teacher는 모두 `qwen3.5:9b-q4_K_M`, model digest
`6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`, contract SHA
`d64bc6b9ce4b3950f81e8c6c533cd8047751b736bd22e4d208482ecec182fb89`다.
각 record의 source SHA와 input-image SHA가 일치한다.

실제 YOLO 관측 모델 SHA는
`7b849c25c3983a54b4b6c922e425798f89326b2da21e862b90d2ee0c6a181f69`이며,
imgsz=640, confidence=0.10, NMS IoU=0.70으로 실행됐다. 9장 모두 검출됐지만
teacher pseudo-label과 raw 9-class가 같은 것은 2장, reference IoU≥0.50은 5장이다.
**2/9는 모델 정답률이 아니라 두 예측 체계의 일치 수**다. 관측 report 자체도
`ground_truth_accuracy_measured=false`, `training_started=false`,
`production_modified=false`, `deployment_authorized=false`로 명시한다.

## 2. 세 사례: 합의가 실제 정답을 보증하지 않았다

상위 작업에서 이미지를 직접 확인한 진단과 실제 저장된 pass를 나란히 기록한다.
이 세 사례는 오류 진단용이며 독립 blind 세트가 아니다. 전체 9장의 정답을 새로
확정하거나 기존 라벨을 수정한 것은 아니다.

| 진단 사례 | teacher pass 순서 | 선택된 합의 | 실제 YOLO 관측 | YOLO/reference IoU |
|---|---|---|---|---:|
| 비닐봉투 | plastic → plastic | plastic, 2/2 | vinyl, confidence≈0.3370 | 0.9034 |
| 찌그러진 Fanta 금속 캔 | can → plastic → plastic | plastic, 2/3 | can, confidence≈0.7951 | 0.8912 |
| 앞쪽 PET 음료병 | pet → plastic → plastic | plastic, 2/3 | pet, confidence≈0.9296 | 0.0000 |

세 record 모두 errors=[]이고 모든 pass confidence=0.95,
single_object=true, training_usable=true, quality_reason=usable,
foreign_material=false다. 캔과 PET는 **세 번 모두 동일한 판정이 아니었다.**
첫 pass는 각각 can/PET였지만 같은 모델의 후속 두 표가 plastic으로 뒤집었다.

PET 사례는 재질만의 문제가 아니다. 직접 사진 확인에서 localizer reference가
앞쪽 주대상이 아닌 오른쪽 배경 병을 가리킨 것으로 확인됐다. reference와 실제
YOLO bbox의 IoU도 0이다. 두 localizer가 같은 잘못된 물체를 고르면 provider 간
IoU 합의가 높아도 주대상 annotation이 맞는 것은 아니다.

## 3. 코드에서 확인한 것과 확인하지 못한 것

### 이미지 전달과 모델

- `scripts/label_operational_captures_ollama.py`는 원본 파일 bytes를 읽고 queue의
  SHA와 대조한다. `_request`는 같은 bytes를 base64로 `messages[0].images`의
  **단일 이미지**에 넣는다. 이 경로에서 별도 resize·crop은 하지 않는다.
- 같은 사진의 모든 pass는 동일 image 변수와 동일 model을 쓴다. 파일 SHA와
  `/api/tags` model digest는 처리 전후 및 최종 게시 전 재확인한다.
- 실제 request는 `/api/chat`, stream=false, think=false, num_ctx=8192,
  num_predict=1024, temperature=0, seed=20260819다. prompt에는 `/no_think`와
  JSON-only 요구가 붙는다.
- 실제 `/api/show` 요약에는 completion/vision/tools/thinking capability,
  qwen35 family, 9.7B, Q4_K_M, vision block 27, patch size 16 등이 있다.
  따라서 text-only 모델을 사용했다고 판단할 근거는 없다.
- 다만 모델 metadata는 **이 요청의 vision encoder 처리와 올바른 시각적 grounding**을
  증명하지 않는다. 현재 teacher는 정규화된 JSON pass만 저장하며 원본 HTTP body,
  token count, 응답 종료 상태 및 서버 내부 영상 처리 telemetry를 별도 증거로
  보존하지 않는다. metadata 수치만으로 resize/vision 고장을 단정하지 않는다.

### 합의와 재질 매핑

- `_decision_tuple`과 `_consensus_summary`는 모델이 반환한 material을 그대로
  사용한다. 운영 API의 PET→plastic 통합이 이 teacher 결정 경로에 적용된 것은 아니다.
  can/vinyl을 plastic으로 치환하는 코드 매핑 버그는 이번 검토에서 찾지 못했다.
- 앞선 답변을 다음 요청에 보여주지는 않지만, 동일 가중치·동일 이미지·동일 seed의
  prompt 변형은 서로 독립적인 모델 판정이 아니다. 상관된 오답이 반복될 수 있다.
- 두 pass가 같은 tuple이면 종료하고, 다르면 세 번째 pass의 다수결을 쓴다.
  self-reported confidence 0.95는 측정된 95% 정답률이 아니다.
- JSON schema는 material enum, boolean/type, quality 일관성을 검사한다.
  금속 테두리·비닐의 얇고 유연한 형태·PET 병 형태 등의 관측 근거를 검증하지 않는다.
  형식적으로 유효한 plastic 답변도 의미상 틀릴 수 있다.

## 4. 원인 가설 — 아직 증명된 인과관계가 아니다

1. **pass 사이 분류 기준의 불균일.** `operational_teacher_contract.py`의 첫 prompt는
   can(금속 캔), pet(PET 음료병), plastic(PET 아닌 플라스틱), vinyl(비닐/봉투) 등
   전체 taxonomy를 설명하지만 두 번째와 adjudication prompt는 같은 상세 표를
   반복하지 않는다. 캔/PET의 첫 판정이 후속 plastic 두 표로 뒤집힌 실제 양상과
   관련 있을 수 있다. 그러나 비닐은 첫 pass부터 틀렸으므로 이것만이 원인이라고
   할 수 없다. 모든 prompt의 taxonomy를 같게 한 통제 비교가 필요하다.
2. **같은 모델 다수결의 상관 오류.** 세 번 실행해도 서로 다른 시각 모델의 교차검증이
   아니다. 반복 투표 횟수 또는 confidence 문턱만 올려 해결됐다고 주장할 수 없다.
3. **주대상 identity의 미결박.** localizer prompt의 one primary item/배경 제외 지시만으로
   앞쪽 투입 대상과 배경 병이 항상 구분되지는 않는다. geometry agreement와
   주대상 semantic agreement는 다른 검사다.
4. **시각 모델/전처리/장면 특성의 영향.** 구겨진 금속, 반사, 얇은 비닐, 여러 병이 보이는
   장면에서 모델이 잘못된 시각 특징을 사용했을 가능성이 있다. 다만 서버 영상 처리의
   실제 실패, 특정 quantization의 원인, 8B 모델의 우위를 현재 근거만으로 확정하지 않는다.

## 5. 다음 단계: 작은 신규 계약 비교, 기존 hold 해제 아님

다음은 실행 계획이다. 현재 봉인 contract·원본 출력·모델 파일을 변경하지 않고
새 contract/version과 불변 run 경로에서 진행한다.

1. **세 사례를 고정된 진단 세트로 유지한다.** 기존 9장은 계속 hold하며 이 세 사례를
   학습 정답이나 신규 독립 blind로 전환하지 않는다. 실패·불일치·원래 pass를 모두 보존한다.
2. **taxonomy를 모든 prompt에 동일하게 제공한다.** can/PET/plastic/vinyl 등 9종의
   구분과 동일 재질 부속품/다른 재질 부착물 규칙을 첫·둘째·adjudication에 동일하게
   명시한다. 예전 contract를 수정하지 않고 새 SHA/cache namespace로 비교한다.
3. **기존 설치된 8B vision 모델을 독립 semantic judge 후보로 비교한다.**
   `qwen3-vl:8b`를 현재 9B와 다른 모델의 후보로 사용하되, 실행 전 실제 capability,
   model digest 및 가중치 identity를 재확인한다. 다른 이름/크기만으로 독립 정답을
   보장하지 않는다. 이전 모델 답을 보여주지 않고 동일한 사진·taxonomy로 판단하게 한다.
   새 다운로드·서비스 변경·학습 실행은 이 문서가 승인하지 않는다.
4. **전경 대상과 분류/crop을 하나의 대상에 결박한다.** 주대상 위치·관측 가능한 형태를
   독립적으로 확인하고 localizer bbox가 바로 그 물체를 둘러싸는지 검사한다.
   필요하면 하드웨어의 명시된 투입 영역과 원본 위치의 접합 검증을 새 계약에 포함하되,
   배경 병으로의 대상 전환을 허용하지 않는다. 기존 deployed YOLO bbox를 독립 GT의
   fallback으로 사용하거나 잘못된 reference를 조용히 교체하지 않는다.
5. **재질·대상 모두 합의하지 않으면 미확정으로 둔다.** 동일 모델 2/3 다수결이 다른
   모델의 semantic 불일치를 자동으로 덮어쓰지 못하게 한다. 구조화된 시각 근거도 모델의
   또 다른 주장일 수 있으므로 형식 검증만으로 승격하지 않는다. 실제 final crop의
   동일 대상 여부와 material을 함께 판단한다.
6. **소규모 비교 결과를 별도로 기록한다.** 기존 9B/균일 taxonomy 9B/독립 8B의
   판정과 대상 bbox, 불일치, 오류, 요청 이미지 SHA, 모델/contract SHA와 runtime
   관측을 나란히 남긴다. 세 진단 사례의 수정은 보편적 정확도 증거가 아니다.
   추가 독립 데이터에서 재확인한 뒤에만 이후 정제·재학습 후보 수용을 판단한다.

현재의 다음 진전은 **teacher/대상 의미 검증의 소규모 비교**다. 데이터 무결성 검사를
약화하거나 기존 9장의 hold를 삭제해서 학습 단계로 넘기는 것은 다음 단계가 아니다.
