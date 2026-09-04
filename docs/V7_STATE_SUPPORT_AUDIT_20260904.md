# V7 상태 라벨 지원·근거 감사 — 2026-09-04

## 결론과 범위

V7에는 상태 라벨이 있지만 `foreign_material=1`의 AIHub 원본 manifest 지원은
**train 34행 / validation 21행**이다. 이는 AIHub의 공식 이물질 정답 55장이 아니라
**Qwen teacher가 만든 pseudo-label 55행**이다. 실제 JSONL 연결과 AIHub 소형 JSON
sidecar를 확인했으며 이미지는 열지 않았다. 라벨의 시각적 정확도, 현재 하드웨어
성능, 신규 crop으로의 전이, 보호 데이터 누수, 배포 승인은 확인하지 않았다.

모두 동일 teacher의 두 pass 합의·0.90 이상 confidence를 갖추었지만,
**4행에 `same_material_accessory_only=true`와 이물질 양성이 함께 존재한다.**
기존 parser/수용 함수에는 이 모순을 거부하는 검사가 없었다. 두 pass의 합의는
의미가 맞다는 보증이 아니다. 원본 CSV/JSONL과 모델은 수정하지 않았다.

## 실제 실행 증거

NAS 작업 루트는 `/share/Container/operational_refresh_80bf78a_20260904_101000`이다.
동일 마운트는 실행 컨테이너에서 `/app`으로 보인다. 다음 두 CPU 읽기 전용 집계는
실제 종료 코드 0으로 완료됐다. 큰 CSV/JSONL은 NAS 로컬에서 streaming으로 읽었고,
입력 파일 초기/최종 SHA와 stat을 확인했다. 출력은 표본 ID·원본 이미지·인증정보를
포함하지 않는 집계 JSON이다.

| 증거 파일 — 작업 루트 기준 | SHA256 |
|---|---|
| `label_support_20260904.json` | `4578ec905fa460515a888a577dc77675a7797e7fb606bbc686d1cc806561542a` |
| `trace_v7_foreign_20260904.json` | `f3defaf118a7c97d66d416456ee15d76283089f5493e22ebcae82f4f52467393` |
| `count_manifest_label_support_20260904.py` — 실제 실행본 | `9e6eeb2ff4903cd1003b636bb59125ffdc43154db37e926dad4f45c9c3c76594` |
| `trace_v7_foreign_positive_support_20260904.py` — 실제 실행본 | `13148d6fdfe5b600aced8fa00474f4425d0698ca1a6f50ae66eea73472c3cc27` |

첫 집계 실행본은 로컬 helper의 origin-family 집계 추가 **이전** 버전이다.
그 이후 helper의 SHA로 이 결과를 표시하면 안 된다.

| 실제 입력 — `/share/Container` 기준 | SHA256 |
|---|---|
| `crops_verifier_single_v3/manifest_curated_v7_balanced_20260803.csv` | `be8b1dc56cc325b7cb43926d8d9ce46fb839eaa4cfeed65c61f43116821aa90a` |
| `verifier_teacher_qwen35_50k_v2_20260801/pseudo_status.jsonl` | `6b709d2a2464a2b3c455d05e71fcae9aa2477cc7a2dc96b83f2019ab11bd55bc` |
| `proposal_verifier_multitask_v4_bgfix_20260831/manifest.csv` | `5c9e0e933e75cd7318a5ca9b3f5baf4460fee02466eb0a572cdd1aec5dda2dbb` |
| `proposal_verifier_multitask_v3_lowconf_20260827_strict/proposal/proposal_strict_sanitized_v1_20260827.csv` | `267232f92b7472a1a2ee52c2c9e0580074a61333f807fb53718eefe1b3239630` |

NAS에 남은 `pseudo_label_status_qwen.py` 소형 소스의 현재 SHA는
`f70bd9a7fffdebd669a48fac908a9aa5bef10d3808d5997b5279f426f78daeb0`이다.
당시 teacher JSONL은 실행 코드/prompt/model **digest를 봉인하지 않았으므로** 이 소스가
모든 과거 행을 생성한 바로 그 버전이라는 것까지 증명하지는 못한다.

## 1. 원본 행 수와 반복 샘플링 수를 구분한다

V7 CSV는 90,274행 = train 75,581행 + validation 14,693행이다.
다음은 고유 이미지 수가 아닌 **CSV 원본 행 수**이며 교차 manifest 중복 제거는 하지 않았다.

| split | 상태 | -1: 미확정/대상 아님 | 0 | 1 |
|---|---|---:|---:|---:|
| train | dent | 55,581 | 10,070 | 9,930 |
| train | label | 63,906 | 6,065 | 5,610 |
| train | foreign_material | 45,441 | 30,106 | 34 |
| validation | dent | 10,693 | 2,454 | 1,546 |
| validation | label | 11,574 | 1,803 | 1,316 |
| validation | foreign_material | 5,839 | 8,833 | 21 |

V4 bgfix CSV는 91,938행(train 75,459 / validation 16,479)이며 9종+background가
양쪽에 있지만 **세 상태 모두 전 행 `-1`**이다. V3 strict proposal 83,296행도
세 상태가 전부 `-1`이다. 이 runtime-crop manifest만 바로 학습해 상태 개선을
기대할 수 없다. 기존 V3 학습 메타의 출력도 objectness/material뿐이다.

별도로 실제 읽은
`hardware_capture_prep_20260803/dataset_v2/verifier/hardware_manifest.csv`는
103행(train 68 / validation 35), SHA
`7d1e22f4815bd91269f59620a1f97559b82be318fe72485aba451b1d1dde897d`이다.
foreign은 train 0=60/1=8, validation 0=35/1=0이다.
따라서 hard100 메타의 train 이물질 양성 834는 **34 + 8 × 100**이다.
834개의 서로 다른 AIHub 양성이나 하드웨어 평가 양성이 아니다. 이 8행 역시
보호 데이터와의 재감사 없이 신규 학습 투입 승인으로 간주하지 않는다.

## 2. 상태 정답의 출처

현재 확인한 코드 경로:

- `scripts/extract_verifier_crops.py`: AIHub `ANNOTATION_INFO.DAMAGE`를 can/PET에만
  적용한다. `원형 → dent=0`, `찌그러짐/완전압착 → dent=1`; 그 외는 `-1`이다.
- 같은 extractor는 `label`과 `foreign_material`을 **처음에는 모두 `-1`**로 저장한다.
  AIHub `DIRTINESS`는 `raw_dirtiness`에 보존하고, 별도의 `label_proxy`만 만든다.
  `DIRTINESS`만으로 라벨과 실제 다른 재질 이물질을 구분할 수 없기 때문이다.
- `scripts/pseudo_label_status_qwen.py`: 원본 경로와 annotation bbox에서 tight
  padding 0.08 / context padding 0.30 view를 만들고 teacher가 상태를 추정한다.
  `foreign_only/both`를 foreign=1로 변환한다. label은 PET/plastic에만 0/1을 준다.
  단일 주대상, 비모호 결정, confidence 기준을 통과하지 못하면 상태는 `-1`이다.
- `merge_manifest` 및 `scripts/merge_pseudo_status_manifest.py`가 결과를 연결하고,
  `scripts/select_curated_verifier_manifest.py`가 최종 균형 manifest를 선별했다.
  이 연결은 과거 문자열 식별자 중심이며 오늘의 content-SHA/provenance 계약과 다르다.

55행 모두 정확한 teacher record 하나에 연결됐다. 해당 JSONL 전체 parse 오류 0,
선택된 key 중 중복 teacher record 0, 각 행 2 pass, 두 pass와 최종 결정 일치 55,
각 pass confidence ≥0.90 55, 최종 confidence=min(pass confidence) 55였다.
모델 표기는 모두 `qwen3.5:9b-q4_K_M`이지만 model digest·prompt SHA·원본 이미지 SHA
필드가 해당 legacy record에 없다. 문자열 모델 이름을 현재 설치 모델의 불변 identity로
승격하면 안 된다.

55개의 정확한 AIHub JSON sidecar만 읽었다. 모두 단일 annotation이며, 저장된
`raw_dirtiness`가 annotation hint와 일치하고 dent도 위 매핑과 일치했다.
이는 **데이터 연결 확인**이지 사진에 이물질이 실제 보인다는 검증이 아니다.

## 3. 이물질 양성 55행의 분포와 모순

| split | can | PET | paper | plastic | vinyl | 합계 |
|---|---:|---:|---:|---:|---:|---:|
| train | 22 | 0 | 6 | 4 | 2 | 34 |
| validation | 9 | 1 | 1 | 9 | 1 | 21 |

train 결정은 foreign_only 30 / both 4, validation은 foreign_only 21이다.
label target은 train -1=30/0=4, validation -1=11/0=10이며, label=1은 없다.
`both`는 PET/plastic 밖에서는 label=-1로 마스킹될 수 있으므로 `both` 문자열만으로
label 양성이 실제 학습에 쓰였다고 계산하면 안 된다.

AIHub 오염 hint는 다음과 같다.

| split | 오염없음 | 이물질(내부) | 이물질(외부) | 이물질(전체) |
|---|---:|---:|---:|---:|
| train | 1 | 17 | 14 | 2 |
| validation | 0 | 4 | 13 | 4 |

내부 hint 21행을 사용자 계약의 외부 이물질 양성으로 그대로 볼 수 없다.
다만 hint 자체가 정답은 아니므로 이 21행을 자동 음성으로 바꾸는 것도 금지한다.
캔/플라스틱의 내용물은 `EMPTY_CONTENTS`, 실제 다른 재질 부착·혼합은
`FOREIGN_MATERIAL`, 제거할 상품 라벨은 `REMOVE_LABEL`이라는 판정 의미를
실제 최종 crop에서 다시 구분해야 한다.

두 pass 합계 110회의 사유 코드는 contamination 17, different_material 9,
same_material_accessory 8, 기타 자연어 사유 76이다. 기타 76개를 이물질 종류로
추정하지 않았다. **4개의 양성 행에는 same-material-accessory-only flag가
동시에 존재한다.** 4행과 내부 hint 21행의 중첩은 이번 집계에서 계산하지 않았으므로
단순히 55−4−21을 신뢰 가능한 양성 수로 보고해서는 안 된다.

### legacy guard가 모순을 허용하는 이유

`parse_teacher_output`은 decision과 label/foreign flag의 조합과 각 boolean 타입을
검사하지만, `same_material_accessory_only=true`와 foreign=true의 충돌을 거부하지
않는다. `consensus_teacher`의 합의 key에서도 accessory flag는 제외되며 마지막에
`any()`로 합쳐진다. `accepted_status` 역시 accessory 모순을 검사하지 않는다.

보강 전 해당 함수들만 CPU에서 추출해 실행한 **합성 입력 재현**:

```json
{"d":"foreign_only","s":true,"c":0.99,"e":"same_material_accessory"}
```

동일 입력 두 pass를 합의시키면 accessory-only=true인 상태로
`{"label":0,"foreign_material":1,"status_eligible":1}`이 수용된다.
합성 재현은 실제 4장의 이미지 내용이나 정확한 raw 응답을 대체하지 않는다.
실제 4행의 모순 flag 존재와, 코드의 재현 가능한 누락을 각각 확인한 것이다.

현재 운영 teacher는 `operational_teacher_contract.py` /
`label_operational_captures_ollama.py` / `build_operational_teacher_manifest.py`의
별도 schema·digest·consensus 계약을 사용한다. legacy 행을 현재 계약을 통과한
것처럼 바꾸거나 현재 계약의 revision 없이 임의로 재사용하지 않는다.

## 4. 약 99.7%라는 기존 지표의 한계

`runs/verifier_curated_v7_hard100_mnv3_20260803/verifier_metadata.json`에 저장된
foreign validation support는 0=8,868 / 1=21, best_val_metrics accuracy는
0.997075036562043(약 99.7075%)이다. 같은 분포에서 전부 0만 답하는 기준선도
8,868/8,889 = **99.7638%**다. 기존 accuracy는 이 단순 기준선보다 낮다.

이 수치는 저장된 과거 평가 메타의 해석이지 이번에 모델을 재실행한 결과가 아니다.
confusion matrix·양성 recall/precision·클래스별 지원·오탐률 및 독립 하드웨어
이물질 양성이 없으면 실제 이물질 검출 성능을 증명할 수 없다. 또한 평가 표적 자체가
teacher pseudo-label이므로 물리적 정답과 일치하는지는 별도 검증 대상이다.

## 5. 다음 mining과 실제 YOLO crop 상태 연결 조건

이번 확인으로 legacy CSV를 자동 수정하거나 새 학습을 승인하지 않는다.
다음 단계를 새 불변 경로에서 진행할 때 적용할 조건이다.

1. **후보와 정답을 구분한다.** 55행은 재검증할 hard-positive seed다. 모순 4행은
   신규 state-label 수용 대상에서 격리하되 원본과 거부 사유를 보존한다. 내부 hint와
   기타 사유는 의미 재검증 후보이지 자동 양성/음성 판정 근거가 아니다.
2. **이름으로 join하지 않는다.** legacy `source_id`는 split/folder/filename에서
   만든 SHA1 앞 20자리이며 현재 proposal의 이미지 byte SHA256과 다르다.
   `source_path_b64`를 제한된 원본 root 안에서 복원하고 실제 원본 SHA, annotation
   sidecar SHA, 단일 GT bbox·크기·재질·split을 결박한 source evidence를 새로 만든다.
   basename이나 서로 다른 두 `source_id`를 비교해 같은 사진이라고 판단하지 않는다.
3. **실제 추론 crop을 검증한다.** 고정 YOLO의 실제 top1 class/confidence/bbox,
   padding/letterbox/crop byte SHA와 독립 annotation을 분리한다. teacher GT bbox
   crop을 runtime YOLO crop이라고 표시하지 않는다. 미검출·부분 잘림·객체 불일치를
   억지 상태 양성으로 만들지 않는다.
4. **보이는 속성만 옮긴다.** can/PET dent는 동일 대상의 형태가 충분히 보이는지
   확인한다. label/foreign는 최종 YOLO crop에 단서가 남아 있는지 별도로 다시
   검증한다. 더 넓은 context에만 있던 이물질을 작은 crop의 양성으로 복사하지 않는다.
   미확정은 `-1`이며 특히 부분 잘림에서 0을 추정하지 않는다.
5. **상태 의미와 provenance를 따로 봉인한다.** 동일 재질 빨대·뚜껑은 허용,
   캔 인쇄/잉크는 removable label 아님, 상품 sleeve/sticker와 컵의 다른 재질 띠지,
   내용물과 외부 혼합을 구분한다. 실제 crop·모델 digest·prompt/contract SHA·개별
   pass·합의·거부 기록을 보존하고 원본 재질 annotation의 권한을 상태 pseudo-label에
   전이하지 않는다. legacy model tag만으로 현재 모델 digest를 소급 부여하지 않는다.
6. **분리와 평가를 지킨다.** 기존 train/validation을 섞지 않고 운영 teacher는
   train-only 권한을 유지한다. 고정 하드웨어 41장과 QX3 3,500장 등 보호 자료의
   source/crop SHA·object-group/session·근접중복 감사는 별도로 수행한다.
   조건별 0/1 존재만 통과해도 높은 정확도를 보장하지 않는다. PET 이물질 train 0,
   paper/vinyl validation 각 1처럼 극소수인 범주는 추가 mining이 필요하다.

현재 가능한 안전한 진전은 **근거가 결박된 실제 crop 재검증과 신규 상태 후보 정제**다.
이번 결과를 배포 승인이나 보호 누수 감사 완료로 간주하지 않는다.

## 6. 로컬 future-use 방어 보강

2026-09-04 추가 요청에 따라 `scripts/pseudo_label_status_qwen.py`의 재수용 경로를
보강했다. NAS teacher 재실행·원본 CSV/JSONL 수정·모델 학습/배포는 수행하지 않았다.

- parser, 직접 `consensus_teacher`, 직접 `accepted_status` 모두
  accessory-only=true와 foreign 양성의 모순을 `ValueError`로 거부한다.
  compact 응답의 `e=same_material_accessory`도 검사하여 보조 flag만 false로 바꿔
  우회할 수 없게 한다. 정상 neither/label_only와 동일 재질 부속품 허용은 유지한다.
- 기존 `_process_candidate`의 error 경로는 해당 실패를 `label=-1`,
  `foreign_material=-1`, `status_eligible=0`으로 기록한다. foreign=0으로 임의 변경하지 않는다.
- JSONL 재개 시 실제 적용할 마지막 record의 teacher/pass 근거를 확인하고, 수용된
  양성·음성 모두 raw 응답 모순을 검사한다. accepted 양성은 각 근거와도 교차 검사한다.
  모순 winner를 조용히 건너뛰지 않고 중단한다. 명시적 후속 재판정으로
  이미 저장된 정상 마지막 record가 있으면 원래 last-record-wins 동작을 유지한다.
- `merge_manifest` 직접 호출도 목적 파일을 열기 전에 동일 검사를 한다.
  실패하면 기존 결과 파일과 입력 mapping/cache bytes를 유지한다.
- 이미 거부된 error record의 잘못된 raw 응답은 오류 근거로 보존할 수 있다.
  이것을 accepted 양성으로 사용하는 것은 차단한다.

PROMPT, 응답 schema, 모델 metadata, 최신 `operational_teacher_contract.py`는
변경하지 않았다. 기존 캐시에 새 model/prompt digest를 소급 부여하지 않는다.
이 수정은 **새로운 재수용 방어**일 뿐 과거 라벨 정제 완료나 모델 정확도 개선 결과가 아니다.
별도의 legacy CSV-only merge 도구로 옛 라벨을 복사하는 행위까지 기존 라벨을
수정하여 막은 것은 아니므로, 원본 V7 CSV는 계속 이 문서의 mining seed 취급을 유지한다.

`tests/test_pseudo_status_accessory_guard.py`에서 compact/full/direct 함수, cached
teacher/pass/raw 교차 우회, last-record retry, main 재개, 병합 전 파일 보존,
실제 처리 함수의 unknown error 반환을 검증한다. 기존 verifier/curated manifest
회귀도 함께 실행한다. 테스트는 CPU fixture이며 NAS·이미지 정답 평가가 아니다.
