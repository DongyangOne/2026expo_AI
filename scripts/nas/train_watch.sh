#!/bin/bash
# 변환(convert_remainder) 종료 대기 → 성공 시 학습 자동 기동.
# root + setsid 로 실행 (SSH 끊겨도 유지). 로그: /share/Container/train_chain.log
D=/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker

echo "[watch] $(date '+%F %T') convert_remainder 종료 대기 시작"
code=$($D wait convert_remainder 2>&1)
echo "[watch] $(date '+%F %T') convert_remainder 종료 — exit code: $code"

if [ "$code" != "0" ]; then
  echo "[watch] 변환 비정상 종료 — 학습 시작 중단. docker logs convert_remainder 확인 필요."
  exit 1
fi

echo "[watch] 워커 메모리 회수 대기 30s"
sleep 30

bash /share/Container/start_train.sh
echo "[watch] $(date '+%F %T') 체인 완료"
