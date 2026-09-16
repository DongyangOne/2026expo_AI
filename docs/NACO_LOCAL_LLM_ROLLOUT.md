# NAS 로컬 Vision LLM 운영 규약

## 현재 역할

운영 Pi는 `LOCAL_LLM_MODE=primary`로 동작한다.

1. YOLO는 물체의 bbox와 `NOT_DETECTED`만 결정한다.
2. bbox crop을 NAS의 API-key 보호 Vision LLM에 보낸다.
3. LLM은 9개 품목, `has_label`, `is_dented`, `has_foreign_material`, 단일 주 물체 여부를
   엄격한 JSON으로 반환한다.
4. AI 서버는 무게와 조건을 조합해 `ALLOWED`·`REJECTED`·`GENERAL_WASTE`를 결정하고,
   즉시 응답과 Spring 콜백에 같은 JSON을 보낸다.

PET는 외부 계약에서 항상 `plastic / class_id=3`으로 정규화한다. 정상 비닐은
`vinyl / class_id=5 / ALLOWED`다.

## Fail-closed 규칙

LLM 호출 실패, JSON 계약 불일치, `LOCAL_LLM_MIN_CONFIDENCE` 미만, 또는
`is_single_primary_item=false`이면 기존 YOLO 품목으로 통과시키지 않는다. 기본 설정
`LOCAL_LLM_PRIMARY_FALLBACK_TO_YOLO=false`에서는 다음처럼 보류한다.

```json
{
  "status": "GENERAL_WASTE",
  "general": {"code": "LOW_CONFIDENCE"}
}
```

`classification`을 가진 `GENERAL_WASTE`를 가정해 Spring/하드웨어를 구현하면 안 된다.
이 상태는 통 선택을 확정하지 못했다는 뜻이다. `client_id`는 모든 응답과 콜백에서 원본 그대로 유지된다.

## 상태와 안내 코드

`conditions`에는 Spring 계약상 `has_label`, `is_dented`만 포함한다. 외부 이물질은 별도
필드가 아니라 `guidance[].code`로 전달한다.

| 조건 | guidance code |
|---|---|
| 플라스틱(PET 포함)·캔 무게 이상 또는 내용물 존재 추정 | `EMPTY_CONTENTS` |
| 종이·비닐 무게 이상 | `WEIGHT_ANOMALY` |
| 다른 재질의 부착물·혼합 이물질 | `FOREIGN_MATERIAL` |
| 플라스틱(PET 포함) 라벨 미제거 | `REMOVE_LABEL` |
| PET병·캔 미압착 | `COMPRESS` |

같은 재질 부속품(예: 플라스틱 빨대)은 `FOREIGN_MATERIAL`로 보지 않는다. 서로 다른 재질의
테이크아웃 컵 종이 슬리브 같은 부착물은 이물질이다. 여러 위반이면 guidance 배열에 함께 넣는다.

유리·건전지·형광등·스티로폼은 재처리 안내가 아니라 `REJECTED`와 각각
`GLASS`·`BATTERY`·`FLUORESCENT`·`STYROFOAM` rejection code로 반환한다.

## 보안 및 검증 경계

LLM gateway URL과 API key는 실제 Pi `.env`에만 둔다. 공개 인터넷에 Ollama 포트를 직접
노출하지 않는다. 이미지/결과 캡처는 재학습 후보로 쓸 수 있지만, 자동 정답이나 자동 배포 근거가
아니다.

2026-09-16 기준, 실제 하드웨어 capture 한 장을 이 경로로 내부 실행해 `plastic`과
`FOREIGN_MATERIAL`을 포함한 `REJECTED` 응답을 확인했다. 이는 연결·계약 smoke test이며,
독립 하드웨어 정답셋 전체에 대한 정확도 보증은 아니다.
