#!/usr/bin/env bash
# Shell integration examples. Run from the repo root with the venv active.
set -euo pipefail
MODEL=${1:-models/rul_gru.pt}

# 1. Latest RUL of every stack as CSV
pemfc-rul predict --model "$MODEL" --input data/test.csv --format csv

# 2. Pipe: filter one stack's rows with any tool, send via stdin
{ head -1 data/test.csv; grep '^103,' data/test.csv; } | pemfc-rul predict --model "$MODEL"

# 3. Use the JSON in a script (python used as a portable JSON parser here; jq works too:
#    ... | jq '.[] | select(.rul_h < 150) | .unit')
pemfc-rul predict --model "$MODEL" --input data/test.csv \
  | python -c 'import json,sys; [print("ALERT unit", r["unit"], round(r["rul_h"])) for r in json.load(sys.stdin) if r["rul_h"] < 150]'

# 4. Exit codes for error handling
code=0
pemfc-rul predict --model "$MODEL" --input does_not_exist.csv 2>/dev/null || code=$?
echo "missing input -> exit code $code (expected 1)"

# 5. Persistent server: one JSON request per line
printf '%s\n' '{"cmd":"info"}' '{"id":"t1","csv":"data/test.csv"}' '{"cmd":"quit"}' \
  | pemfc-rul serve --model "$MODEL" | cut -c1-160
