#!/bin/bash
# 인자로 받은 라벨폴더/원천glob에서 오염없음 vs 이물질(외부) 실제 이미지 경로 추출
BASE="/share/Container/ai_dataset/학습용_데이터/01-1.정식개방데이터/Training"
LBL="$BASE/02.라벨링데이터/$1"
SRC_GLOB="$BASE/01.원천데이터/$2"

find_img() {
  find $SRC_GLOB -name "$1.jpg" 2>/dev/null | head -1
}

{
  echo "### CLEAN (오염없음)"
  grep -rl '"DIRTINESS": "오염없음"' "$LBL" 2>/dev/null | head -4 | while read f; do
    bn=$(basename "$f" .json); find_img "$bn"
  done
  echo "### EXTERNAL (이물질외부)"
  grep -rl '"DIRTINESS": "이물질(외부)"' "$LBL" 2>/dev/null | head -4 | while read f; do
    bn=$(basename "$f" .json); find_img "$bn"
  done
} | base64 -w0
