#!/usr/bin/env bash
# GPU work queued behind the B4 replay (IMPLEMENTATION_PLAN_2026-10 Phases E/F).
# Waits for tools/revalidation/reval_e270cdc.sh to finish, then -- from the MAIN tree at a committed,
# detached HEAD so every run stamps a real commit -- runs:
#   1. the C2.7 R3 pilot spec            (spec layer vs the harness replay: same v/2Dk?)
#   2. the a* basin ensemble spec        (Phase F1: seeds 619/620/621 -> distinct basins?)
#   3. the a* continuation pilot         (Phase F3: relative equilibrium + Floquet across the bracket)
# Each step is independent; failures are logged and the next step runs.
#
#   wsl: setsid nohup bash tools/revalidation/post_replay_queue.sh <commit> > sweep_runs/POST_REPLAY_queue.log 2>&1 &
set -u
COMMIT=${1:?usage: post_replay_queue.sh <commit-to-run>}
MAIN=/mnt/f/quantule_mapper
PY=~/jax_irer/bin/python
while pgrep -f reval_e270cdc.sh > /dev/null; do sleep 60; done
echo "[$(date '+%F %T')] replay finished; switching main tree to $COMMIT"
cd "$MAIN"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then echo "main tree dirty -- abort"; exit 2; fi
git switch --detach "$COMMIT" || exit 2
echo "HEAD $(git rev-parse --short HEAD)"
step () { local n=$1; shift; echo "[$(date '+%F %T')] START $n"; "$@" > "sweep_runs/POST_REPLAY_$n.log" 2>&1; echo "[$(date '+%F %T')] END $n rc=$?"; }
step c27_pilot   $PY tools/run_spec.py specs/approved/c27-r3-soliton-transport-n1.json
step astar_basin $PY tools/run_spec.py specs/approved/astar-basin-ensemble.json
step astar_cont  $PY jax_scout/astar_continuation_pilot.py --N 32
echo "[$(date '+%F %T')] QUEUE DONE"
