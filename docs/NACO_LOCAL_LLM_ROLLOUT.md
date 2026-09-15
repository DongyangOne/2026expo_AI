# NAS naco 로컬 Vision LLM 전환 계획

## 목표와 경계

YOLO는 계속 사용한다. YOLO가 `bbox`와 `NOT_DETECTED`를 결정하고, 검출된 단일 crop의
9개 품목·라벨·압착·외부 이물질은 NAS의 `qwen3-vl:8b`가 재판정한다. 무게 검사,
`ALLOWED`/`REJECTED`/`GENERAL_WASTE`, guidance, Spring callback DTO는 기존 규칙 엔진을
그대로 사용한다.

## 단계

1. NAS `naco-ollama`의 `qwen3-vl:8b`와 `/api/chat`을 확인한다.
2. Pi에서만 접근 가능한 TLS/API-key gateway를 만들고, 키는 Pi의 실제 `.env`에만 저장한다.
3. `LOCAL_LLM_MODE=shadow`로 YOLO와 LLM의 품목·상태 결과/지연 시간을 로그로 비교한다.
4. 고정 하드웨어 사진과 새 독립 사진에서 품목·상태·무게 guidance 계약을 검증한다.
5. 기준을 만족할 때만 `primary`를 켠다. 호출 실패, JSON 불일치, 저신뢰(<`LOCAL_LLM_MIN_CONFIDENCE`)는
   기존 YOLO/상태 모델로 즉시 fallback 한다.

## 운영 설정

`LOCAL_LLM_BASE_URL`, `LOCAL_LLM_API_KEY`는 Git에 넣지 않는다. NAS Ollama는 기본적으로
API-key 인증을 강제하지 않으므로, Pi↔NAS 경로에는 별도 gateway가 필요하다. 공개 인터넷에
Ollama 11434를 직접 노출하지 않는다.

## 승인 전 금지

- LLM의 자기신뢰도만으로 무게·상태 기준을 완화하지 않는다.
- 로그를 자동 학습 데이터나 자동 배포 근거로 사용하지 않는다.
- 독립 하드웨어 E2E와 Spring callback 검증 전 production 응답 계약을 변경하지 않는다.
