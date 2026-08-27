"""Run provenance stamp — one shared helper, so every harness records the same thing.

WHY. 128 of 183 catalogued runs recorded no git commit, so the index could not say which
version of the code produced them. The reason is structural: each of the ~126 harnesses writes
its own summary.json, and only the newest few thought to include a commit. A shared helper is
the fix — the alternative is editing 126 files and hoping the next one remembers.

WHAT TO RECORD, AND WHY EACH FIELD MATTERS

    commit        which code ran. Without it a result cannot be reproduced or superseded.
    dirty         whether the working tree had uncommitted changes. A dirty run is NOT
                  reproducible from the commit alone, and that must be visible rather than
                  assumed away.
    branch        which line of work it belonged to.
    started_utc   an unambiguous timestamp. Run-id timestamps are local-time strings, which
                  makes cross-machine ordering ambiguous.
    python/jax    the runtime that produced the numbers. The project has already been bitten
                  by a numpy-ABI mismatch between two local environments.

Usage in a harness, right before writing config.json or summary.json:

    from jax_scout.provenance import stamp
    write_json(out / "config.json", {"args": vars(args), "provenance": stamp(), ...})

Cheap (two short git calls), never raises, and degrades to nulls outside a repo.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=_REPO, capture_output=True,
                           text=True, timeout=30)
        if r.returncode != 0:
            return None
        return r.stdout.strip()          # "" is a valid answer (e.g. a clean tree)
    except Exception:
        return None


def stamp() -> dict:
    """Provenance for the run about to start. Never raises."""
    commit = _git("rev-parse", "HEAD")
    # -uno skips the untracked scan: with two venvs and sweep_runs/ in the tree a full
    # status can take longer than the timeout, which silently returned "unknown" before.
    # "Did the TRACKED code differ from the commit" is also the question that matters here.
    status = _git("status", "--porcelain", "--untracked-files=no")
    out = {
        "commit": commit,
        "commit_short": commit[:12] if commit else None,
        "dirty": (status != "") if status is not None else None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    try:
        import jax  # noqa: PLC0415
        out["jax"] = jax.__version__
        out["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception:
        pass
    try:
        import numpy  # noqa: PLC0415
        out["numpy"] = numpy.__version__
    except Exception:
        pass
    return out


def flat_stamp() -> dict:
    """Same fields, flattened with a `git_` prefix, for summaries that keep a flat config dict.

    `git_commit` is the key the results index reads, so a harness that writes this is
    automatically dated.
    """
    s = stamp()
    return {
        "git_commit": s["commit"],
        "git_dirty": s["dirty"],
        "git_branch": s["branch"],
        "started_utc": s["started_utc"],
        "python_version": s["python"],
        "jax_version": s.get("jax"),
        "numpy_version": s.get("numpy"),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(stamp(), indent=2))
