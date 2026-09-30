# AIHub 9종 270장 재평가

기준일: 2026-09-30

Vision LLM: NAS Ollama `minicpm-v4.5:8b`

## 측정 조건

- AIHub 9종 클래스별 30장, 총 270장
- 기존 평가와 동일한 manifest 사용
- Pi 운영 컨테이너의 `pipeline.run`에서 NCNN YOLO와 NAS Vision LLM까지 실행
- 플라스틱 재질 전용 2차 판정 포함
- 입력 무게 20g 고정
- Spring callback과 HTTP 직렬화 제외

## 결과

| 지표 | 결과 |
|---|---:|
| 전체 정확도 | **259/270, 95.93%** |
| balanced accuracy | **95.93%** |
| 위험 거부품 recall | **114/120, 95.00%** |
| 미감지 | **0/270** |
| 실행 오류 | **0/270** |
| 평균 응답 시간 | **3.56초** |
| 중앙값 | **3.17초** |
| p95 | **6.19초** |
| 최대 | **7.60초** |
| 전체 소요 | **16분 2.58초** |

## 클래스별 결과

| 클래스 | 정답/전체 | recall |
|---|---:|---:|
| can | 30/30 | 100.00% |
| pet → plastic | 30/30 | 100.00% |
| paper | 30/30 | 100.00% |
| plastic | **27/30** | **90.00%** |
| styrofoam | 27/30 | 90.00% |
| vinyl | 28/30 | 93.33% |
| glass | 28/30 | 93.33% |
| battery | 30/30 | 100.00% |
| fluorescent | 29/30 | 96.67% |

플라스틱 오분류는 `paper` 1건, `styrofoam` 2건이다. 이전 재측정의 23/30에서
27/30으로 4건 증가했다. 다른 클래스 결과는 이전 재측정과 같다.

## 결과 파일

- manifest: [`artifacts/aihub_270_2832a2c/manifest.csv`](artifacts/aihub_270_2832a2c/manifest.csv)
- 결과: [`artifacts/aihub_270_2832a2c/results_current_20260930.json`](artifacts/aihub_270_2832a2c/results_current_20260930.json)
- manifest SHA-256: `30311baed22c9ee190ca4361027430bbe8fc785c5883c5c33d9cc39b746605e8`
- 결과 SHA-256: `2299bdeb42a5a60811cc89397678792254a21e91ced20f2eb17e101ad9a40405`
- Pi 보존본: `/home/one/aihub270_eval_20260929/results_current_20260930.json`
