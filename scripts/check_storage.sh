#!/bin/bash
echo "=== 디스크 종류 (rota: 0=SSD, 1=HDD) ==="
for d in /sys/block/sd*; do
  n=${d##*/}
  rota=$(cat "$d/queue/rotational" 2>/dev/null)
  sz=$(cat "$d/size" 2>/dev/null)
  gb=$(( sz / 2 / 1024 / 1024 ))
  model=$(cat "$d/device/model" 2>/dev/null)
  echo "$n  rota=$rota  ${gb}GB  $model"
done
echo "=== QNAP 스토리지/SSD캐시 ==="
/sbin/qcli_storage 2>/dev/null | head -30 || echo "qcli_storage 없음"
echo "=== dm 매핑 (cache 결합 확인) ==="
/sbin/dmsetup ls 2>/dev/null | head -15 || dmsetup ls 2>/dev/null | head -15
echo "=== /share/Container 마운트 ==="
df -h /share/Container 2>/dev/null | tail -1
