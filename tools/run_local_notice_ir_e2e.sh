#!/usr/bin/env bash

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
TOKEN="${TOKEN:-test-token}"
PITCH_ID="${PITCH_ID:-pitch-123}"
NOTICE_PDF="${NOTICE_PDF:-data/input/sample_notice.pdf}"
IR_PDF="${IR_PDF:-data/input/sample_irdeck.pdf}"
OUT_DIR="${OUT_DIR:-/tmp/poki_e2e}"
NOTICE_POLL_SEC="${NOTICE_POLL_SEC:-3}"
IR_POLL_SEC="${IR_POLL_SEC:-5}"
SKIP_NOTICE="${SKIP_NOTICE:-0}"

mkdir -p "$OUT_DIR"

echo "BASE_URL=$BASE_URL"
echo "PITCH_ID=$PITCH_ID"
echo "NOTICE_PDF=$NOTICE_PDF"
echo "IR_PDF=$IR_PDF"
echo "OUT_DIR=$OUT_DIR"

if [[ ! -f "$NOTICE_PDF" ]]; then
  echo "ERROR: notice pdf not found: $NOTICE_PDF"
  exit 1
fi

if [[ ! -f "$IR_PDF" ]]; then
  echo "ERROR: ir pdf not found: $IR_PDF"
  exit 1
fi

echo "[1/5] Notice upload + analyze"
if [[ "$SKIP_NOTICE" == "1" ]]; then
  echo "SKIP_NOTICE=1 -> notice 단계를 건너뜁니다."
else
  curl -s -X POST "$BASE_URL/pitches/$PITCH_ID/notice" \
    -H "Authorization: Bearer $TOKEN" \
    -F "file=@$NOTICE_PDF" | tee "$OUT_DIR/notice_start.json"

  NOTICE_ID=$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("notice_id",""))' "$OUT_DIR/notice_start.json")
  if [[ -z "$NOTICE_ID" ]]; then
    echo "ERROR: notice start failed"
    cat "$OUT_DIR/notice_start.json"
    exit 1
  fi
  echo "NOTICE_ID=$NOTICE_ID"

  echo "[2/5] Notice polling"
  while true; do
    curl -s "$BASE_URL/notices/$NOTICE_ID" \
      -H "Authorization: Bearer $TOKEN" | tee "$OUT_DIR/notice_result.json"
    STATUS=$(python -c 'import json,sys;print(json.load(open(sys.argv[1])).get("analysis_status",""))' "$OUT_DIR/notice_result.json")
    echo "notice status=$STATUS"
    if [[ "$STATUS" == "COMPLETED" ]]; then
      break
    fi
    if [[ "$STATUS" == "FAILED" ]]; then
      echo "WARNING: notice 분석 실패. IR 단계는 계속 진행합니다."
      break
    fi
    sleep "$NOTICE_POLL_SEC"
  done
fi

echo "[3/5] IR Deck upload + analyze"
curl -s -X POST "$BASE_URL/api/pitches/$PITCH_ID/ir-decks/analyze" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@$IR_PDF" | tee "$OUT_DIR/ir_start.json"

DECK_ID=$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("ir_deck_id",""))' "$OUT_DIR/ir_start.json")
if [[ -z "$DECK_ID" ]]; then
  echo "ERROR: IR start failed"
  cat "$OUT_DIR/ir_start.json"
  exit 1
fi
echo "DECK_ID=$DECK_ID"

echo "[4/5] IR summary polling"
while true; do
  curl -s "$BASE_URL/api/ir-decks/$DECK_ID" \
    -H "Authorization: Bearer $TOKEN" | tee "$OUT_DIR/ir_summary.json"
  STATUS=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["analysis_status"])' "$OUT_DIR/ir_summary.json")
  echo "ir status=$STATUS"
  if [[ "$STATUS" == "COMPLETED" || "$STATUS" == "FAILED" ]]; then
    break
  fi
  sleep "$IR_POLL_SEC"
done

echo "[5/5] IR slides fetch"
curl -s "$BASE_URL/api/ir-decks/$DECK_ID/slides" \
  -H "Authorization: Bearer $TOKEN" | tee "$OUT_DIR/ir_slides.json"

echo "DONE"
echo "notice_start:  $OUT_DIR/notice_start.json"
echo "notice_result: $OUT_DIR/notice_result.json"
echo "ir_start:      $OUT_DIR/ir_start.json"
echo "ir_summary:    $OUT_DIR/ir_summary.json"
echo "ir_slides:     $OUT_DIR/ir_slides.json"
