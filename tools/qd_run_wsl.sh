#!/usr/bin/env bash
# Run one QD exploration in the FOREGROUND inside WSL, appending to its log, at low CPU priority.
# Normally started hidden by tools/qd_background.ps1 (a live wsl.exe keeps the WSL VM up; a nohup'd
# child of a closed shell did not reliably survive here). Control the run with
#   python tools/qd_explore.py status|pause|resume|stop sweep_runs/QD_<ID>
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="$1"; shift || true
ID=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['id'].upper().replace('-','_'))" "$CFG")
OUT="sweep_runs/QD_${ID}"
mkdir -p "$OUT"
if pgrep -f "qd_explore.py run $CFG" >/dev/null; then
  echo "$(date '+%F %T') already running: $CFG" >> "$OUT/log.txt"; exit 1
fi
echo "$(date '+%F %T') start $CFG $*" >> "$OUT/log.txt"
exec nice -n 10 "$HOME/jax_irer/bin/python" -u tools/qd_explore.py run "$CFG" "$@" >> "$OUT/log.txt" 2>&1 < /dev/null
