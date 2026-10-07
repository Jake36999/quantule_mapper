#!/usr/bin/env python
"""Run queued "re-run with visuals" jobs, one at a time. A PERSON starts this; nothing else does.

    WSL ~/jax_irer:  python tools/render_queue_worker.py            # loop: run jobs as they arrive
                     python tools/render_queue_worker.py --once     # run what is queued, then exit

The viewer (tools/serve_viewer.py) only writes job files to specs/queue/. This worker moves each job
queued/ -> running/ -> done/ (or failed/), runs it with tools/run_spec.py into
sweep_runs/<SPEC_ID>_<stamp>/, and records the output directory in the job file. Keeping the runner a
separate, human-started process is deliberate: a fault in the page can never reach a running simulation.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUEUE = os.path.join(ROOT, "specs", "queue")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))


def run_job(path):
    import run_spec
    job = json.load(open(path, encoding="utf-8"))
    running = os.path.join(QUEUE, "running", os.path.basename(path))
    os.makedirs(os.path.dirname(running), exist_ok=True)
    shutil.move(path, running)
    sid = job["spec"]["id"]
    out = os.path.join(ROOT, "sweep_runs", "%s_%s" % (sid.upper().replace("-", "_").replace(".", "_"),
                                                     time.strftime("%Y%m%d_%H%M%S")))
    spec_file = os.path.join(QUEUE, "running", sid + ".spec.json")
    json.dump(job["spec"], open(spec_file, "w", encoding="utf-8"), indent=2)
    job.update(out=os.path.relpath(out, ROOT).replace(os.sep, "/"), started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print("[worker] %s -> %s" % (sid, job["out"]), flush=True)
    try:
        rc = run_spec.main([spec_file, "--out", out])
        state = "done" if rc == 0 else "failed"
        if rc:
            job["error"] = "run_spec exit code %s" % rc
    except Exception:                                      # keep the worker alive for the next job
        state, job["error"] = "failed", traceback.format_exc()[-2000:]
    job["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dst = os.path.join(QUEUE, state, os.path.basename(path))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(job, open(dst, "w", encoding="utf-8"), indent=2)
    os.remove(running)
    os.remove(spec_file)
    print("[worker] %s %s" % (sid, state.upper()), flush=True)
    return state


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--poll", type=float, default=15.0)
    a = ap.parse_args(argv)
    os.makedirs(QUEUE, exist_ok=True)
    print("[worker] watching %s (Ctrl-C to stop)" % QUEUE, flush=True)
    while True:
        jobs = sorted(glob.glob(os.path.join(QUEUE, "*.json")))
        for j in jobs:
            run_job(j)
        if a.once:
            return 0
        try:
            time.sleep(a.poll)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
