#!/usr/bin/env bash
# Full-replay re-validation after the ETDRK4 fix e270cdc (IMPLEMENTATION_PLAN_2026-10, Phase B4).
# Runs the ORIGINAL harnesses with their ORIGINAL arguments on the fixed solver, sequentially on the GPU,
# into sweep_runs/*_REVAL_e270cdc. Decision rules were fixed BEFORE launch in
# docs/instrument_integrity/REVALIDATION_E270CDC_RESULTS.md (section "Pre-registered decision rules").
# Each step is independent: a failure is logged and the next step still runs. All harnesses are resumable.
#
#   wsl:  setsid nohup bash tools/revalidation/reval_e270cdc.sh > sweep_runs/REVAL_e270cdc_driver.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
PY=~/jax_irer/bin/python
S=sweep_runs
STAMP=$(date +%Y%m%d_%H%M%S)
step () {  # name, command...
  local name=$1; shift
  echo "[$(date '+%F %T')] START $name :: $*"
  "$@" > "$S/REVAL_e270cdc_${name}.log" 2>&1
  echo "[$(date '+%F %T')] END   $name rc=$?"
}
echo "commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
# 1. C2.7 re-derivation (R0 CFL, R2 feb pure-NLS, R3 N=96 moving family)   ~0.8 h
step c27   $PY jax_scout/phase_d_c2_7_rederivation.py --out $S/C27_REDERIVE_REVAL_e270cdc
# 2. C1 dispersive transport probe, default grid                              ~1 h
step c1    $PY jax_scout/phase_d_c1_transport.py --out $S/PHASE_D_C1_TRANSPORT_REVAL_e270cdc
# 3. a* gain ladder, T=72000, 4 cells                                          ~4.3 h
step ladder $PY jax_scout/feb_gain_ladder_longt.py --out $S/FEB_GAIN_LADDER_LONGT_T72000_REVAL_e270cdc
# 4. NEW dt-convergence cell: x1.15 at dt/2, same physical time                ~2.2 h
step ladder_dt2 $PY jax_scout/feb_gain_ladder_longt.py --dt 0.0025 --cells a1.15_ladder \
     --out $S/FEB_GAIN_LADDER_LONGT_T72000_DT2_REVAL_e270cdc
# 5. a* confirm: longer-T, seeds, bracket (7 cells)                            ~9.6 h
step confirm $PY jax_scout/feb_astar_confirm.py --out $S/FEB_ASTAR_CONFIRM_REVAL_e270cdc
echo "[$(date '+%F %T')] ALL DONE ($STAMP)"
