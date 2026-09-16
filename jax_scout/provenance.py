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
        # WHICH HARNESS RAN. Surveyed 2026-09-16: not one of the 279 run directories recorded
        # what produced it. The run id hints at it by convention, but a convention is not a
        # record -- several harnesses write the same prefix, and a renamed script leaves no
        # trace at all. Without this the index can say which COMMIT ran but not which ENTRY
        # POINT, which is the question you actually ask when reproducing a row.
        "harness": os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else None,
        "argv": sys.argv[1:] if len(sys.argv) > 1 else [],
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
        "harness": s.get("harness"),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(stamp(), indent=2))


# --------------------------------------------------------------- the shared JSON writer

#: Filenames that identify a run rather than describe one of its rows. Only these get stamped,
#: so a harness that writes many small JSONs does not pay for provenance 900 times.
RUN_METADATA_NAMES = frozenset({
    "config.json", "summary.json", "RUN_COMPLETE.json", "RUN_FAILED.json",
    "D4_SUMMARY.json", "D4_RUN_COMPLETE.json", "D5_SUMMARY.json",
    "run_config.json", "manifest.json", "environment_versions.json",
})


def write_json(path, payload, *, stamp_metadata=True):
    """Write JSON, stamping provenance when the file is the one that identifies the run.

    WHY THIS LIVES HERE. Twenty-three harnesses had defined their own identical one-line
    `write_json`, which is why 250 of 279 run directories carry no commit: provenance had no
    single place to be added. This is that place. A harness switches to it by importing instead
    of defining, and is dated from then on without any other change.

    The stamp is added only for `RUN_METADATA_NAMES`, only when the payload is a dict, and only
    when the caller has not already provided a `provenance` key -- so a harness that stamps
    deliberately keeps its own, and per-row files stay untouched.

    The caller's dict is never mutated. A harness that writes the same dict twice, or reuses it
    after writing, must see exactly what it passed in.

    The fallback encoder tries `float` BEFORE `str`, which is not cosmetic. The harnesses being
    replaced used a mix of `default=float` and `default=str`; a numpy scalar under `default=str`
    becomes the JSON *string* "1.23" instead of the number 1.23, silently changing the type of
    every numeric field for everything downstream. Numbers stay numbers; only genuinely
    non-numeric objects (Path, datetime) fall through to `str`. Neither raises, because a
    multi-hour GPU run must not die at the final write.
    """
    import json as _json

    def _fallback(o):
        try:
            return float(o)
        except (TypeError, ValueError):
            return str(o)

    name = os.path.basename(str(path))
    if stamp_metadata and name in RUN_METADATA_NAMES and isinstance(payload, dict) \
            and "provenance" not in payload:
        payload = {**payload, "provenance": stamp()}
    text = _json.dumps(payload, indent=2, sort_keys=True, default=_fallback)
    try:
        path.write_text(text, encoding="utf-8")     # pathlib.Path
    except AttributeError:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
