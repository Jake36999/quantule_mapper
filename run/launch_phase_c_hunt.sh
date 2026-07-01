#!/usr/bin/env bash
set -euo pipefail

cd /mnt/f/quantule_mapper
source ~/jax_irer/bin/activate

ts="$(date +%Y%m%d_%H%M%S)"
log="runtime_logs/core_sat_hunt_${ts}.log"
echo "Launching Phase C K-varied hunt at $(date -Is)" > "$log"
echo "Command: python jax_scout/core_saturation_search.py --hours 6 --batch 64 --ic-counts 1,2,3,4,6" >> "$log"
stdbuf -oL -eL python jax_scout/core_saturation_search.py --hours 6 --batch 64 --ic-counts "1,2,3,4,6" >> "$log" 2>&1
echo "HUNT EXIT $?" >> "$log"
