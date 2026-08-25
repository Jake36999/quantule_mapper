#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the Obsidian run catalogue from sweep_runs/.

Two-layer model, same as the evidence package: the notes and plots written here are
git-tracked and live in the vault (docs/); the raw run data in sweep_runs/ stays
gitignored and is referenced by path.

Scope: every substantive run - those carrying a summary.json, plus Codex-lane runs derived
from RUN_COMPLETE.json and their handoff reports. Thin/empty dirs go to a triage note.

Re-running is safe: anything below the REVIEW MARKER in an existing note is preserved,
so hand-written review notes survive a rebuild.

Usage:
    .venv/Scripts/python.exe tools/build_run_catalogue.py [--no-plots] [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import traceback
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(REPO, "sweep_runs")
VAULT = os.path.join(REPO, "docs")
OUT = os.path.join(VAULT, "runs")
PLOTS = os.path.join(OUT, "_plots")

REVIEW_MARKER = "## Review notes"

# ---------------------------------------------------------------- families

FAMILIES = [
    (r"^TG_B2", "TG-B2", "gravity", "Dual-substrate two-node force (A-well/A-hill)"),
    (r"^TG_B1S", "TG-B1S", "gravity", "Dual-substrate single-node / static baseline"),
    (r"^TG_S", "TG-S", "gravity", "Temporal-substrate scalar rung"),
    (r"^TG_", "TG-other", "gravity", "TG gravity-maturity ladder"),
    (r"^GRAVITY", "Gravity-D", "gravity", "Spatial-geometry gravity mirror (non-Newtonian)"),
    (r"^(CORE_SAT|ADAPTIVE_HUNT|STAGE|ATTRACTOR)", "Stability", "stability",
     "Phase C attractor / saturation hunts"),
    (r"^PHASE_D", "Phase-D", "transport", "Transport sector harness"),
    (r"^C3", "C3-KG", "transport", "Klein-Gordon Q-ball transport & collisions"),
    (r"^C2", "C2-NLS", "transport", "NLS soliton transport & collisions"),
    (r"^(VALIDATION|TRANSFER_DIAG|MIRROR)", "Validation", "infra", "Parity / validation / diagnostics"),
]


# Each sector is a branch in the sense of docs/DOCUMENTATION_METHODOLOGY.md: only the
# branch index links back to [[Main branch]], so run notes never wire straight into it.
SECTOR_INDEX = {
    "gravity": "Branch - Gravity - Index",
    "transport": "Branch - Transport - Index",
    "stability": "Branch - Stability - Index",
    "infra": "Branch - Validation - Index",
    "other": "Branch - Unsorted - Index",
}
SECTOR_TITLE = {
    "gravity": "Gravity / TG ladder", "transport": "Transport sector",
    "stability": "Stability sector", "infra": "Validation & diagnostics",
    "other": "Unsorted",
}
SECTOR_KIND = {
    "gravity": "main", "transport": "main", "stability": "main",
    "infra": "side", "other": "side",
}


def classify(run_id: str):
    for pat, fam, band, blurb in FAMILIES:
        if re.match(pat, run_id, re.I):
            return fam, band, blurb
    return "other", "other", "Unclassified run"


TS_RE = re.compile(r"_(\d{8})_(\d{6})$")


def run_date(run_id: str, path: str) -> str:
    m = TS_RE.search(run_id)
    if m:
        try:
            return dt.datetime.strptime(m.group(1), "%Y%m%d").date().isoformat()
        except ValueError:
            pass
    try:
        return dt.date.fromtimestamp(os.path.getmtime(path)).isoformat()
    except OSError:
        return "unknown"


# ---------------------------------------------------------------- extraction

SCALAR_KEYS = [
    ("N", ["N"]), ("L", ["L"]), ("T", ["T"]), ("dt", ["dt"]),
    ("elapsed_h", ["elapsed_hours", "elapsed_h"]),
    ("git_commit", ["git_commit"]), ("git_dirty", ["git_dirty"]),
    ("command", ["command"]), ("seed", ["parameter_rng_seed", "seed"]),
]


def dig(summary: dict, names):
    """Look for a key at top level, then inside config{}."""
    cfg = summary.get("config") if isinstance(summary.get("config"), dict) else {}
    for n in names:
        if n in summary and not isinstance(summary[n], (dict, list)):
            return summary[n]
        if n in cfg and not isinstance(cfg[n], (dict, list)):
            return cfg[n]
    return None


def fmt(v, prec=6):
    if isinstance(v, bool):
        return "`%s`" % ("true" if v else "false")
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) < 1e-3 or abs(v) >= 1e5:
            return "%.*e" % (prec - 2, v)
        return ("%.*f" % (prec, v)).rstrip("0").rstrip(".")
    if v is None:
        return ""
    s = str(v)
    return s if len(s) <= 90 else s[:87] + "..."


def rows_table(rows, max_rows=30, max_cols=14):
    """Flatten a list-of-dicts into a markdown table."""
    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        return None, 0
    cols, seen = [], set()
    for r in rows:
        for k, v in r.items():
            if k not in seen and not isinstance(v, (dict, list)):
                seen.add(k)
                cols.append(k)
    cols = cols[:max_cols]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows[:max_rows]:
        out.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(out), len(rows)


def yaml_scalar(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v).replace('"', "'")
    return '"%s"' % s


def extract(run_dir: str):
    run_id = os.path.basename(run_dir)
    sp = os.path.join(run_dir, "summary.json")
    with open(sp, encoding="utf-8") as fh:
        summary = json.load(fh)
    if not isinstance(summary, dict):
        summary = {"_raw": summary}

    fam, band, blurb = classify(run_id)
    rec = {
        "run_id": run_id,
        "run_dir": run_dir,
        "date": run_date(run_id, sp),
        "family": fam,
        "band": band,
        "blurb": blurb,
        "verdict": summary.get("verdict"),
        "summary": summary,
    }
    for key, names in SCALAR_KEYS:
        rec[key] = dig(summary, names)

    # completion sentinels. The RUN_COMPLETE.json convention is only used by the newer
    # TG harnesses, so an absent sentinel is only meaningful where it was expected:
    # either the family uses it, or the run wrote *_STARTED.json and never closed it.
    try:
        _f = os.listdir(run_dir)
    except OSError:
        _f = []
    rec["complete"] = "RUN_COMPLETE.json" in _f
    _started = any(x.endswith("_STARTED.json") for x in _f)
    _closed = any(x.endswith("_COMPLETE.json") for x in _f)
    rec["sentinel_expected"] = fam.startswith("TG") or _started
    rec["sentinel_missing"] = rec["sentinel_expected"] and not (rec["complete"] or
                                                               (_started and _closed))

    # artifacts
    try:
        files = sorted(os.listdir(run_dir))
    except OSError:
        files = []
    rec["csvs"] = [f for f in files if f.lower().endswith(".csv")]
    rec["imgs"] = [f for f in files if f.lower().endswith((".png", ".svg", ".jpg"))]
    rec["jsons"] = [f for f in files if f.lower().endswith(".json")]
    return rec


VERDICT_RE = re.compile(r"\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+){2,})\b")

# Codex-lane runs use a different convention: no summary.json, but a RUN_COMPLETE.json
# carrying `status`, a TECHNICAL_HANDOFF.md, preregistered matrices and model specs.
SPEC_FILES = ["model_spec.json", "source_spec.json", "preregistered_matrix.json",
              "preregistered_gates.json", "config.json", "sweep_meta.json",
              "contracts_summary.json", "metrics.json", "recommendation.json",
              "BRIDGE_SUMMARY.json", "source_matching.json"]
NARRATIVE_FILES = ["TECHNICAL_HANDOFF.md", "DISCREPANCY_REPORT.md", "OVERNIGHT_SUMMARY.md",
                   "POST_REVIEW_CORRECTION.md", "replication_report.md",
                   "REDUCED_MODEL_SIGN_ERROR_ANALYSIS.md", "OPEN_QUESTIONS.md",
                   "DOCUMENTATION_INPUTS.md"]


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _md_sections(path, wanted=("Finding", "Boundary", "Verdict", "Result", "Summary",
                               "Conclusion", "Recommendation")):
    """Pull named ## sections out of a handoff-style markdown report."""
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return {}, ""
    out, cur, buf = {}, None, []
    lead = []
    for line in text.splitlines():
        m = re.match(r"^#{2,3}\s+(.+?)\s*$", line)
        if m:
            if cur:
                out[cur] = "\n".join(buf).strip()
            head = m.group(1)
            cur = next((w for w in wanted if w.lower() in head.lower()), None)
            buf = []
        elif cur:
            buf.append(line)
        elif not line.startswith("#"):
            lead.append(line)
    if cur:
        out[cur] = "\n".join(buf).strip()
    return {k: v for k, v in out.items() if v}, "\n".join(lead).strip()


def extract_nosummary(run_dir: str):
    """Derive a record for a substantive run that carries no summary.json."""
    run_id = os.path.basename(run_dir)
    try:
        files = sorted(os.listdir(run_dir))
    except OSError:
        files = []
    fam, band, blurb = classify(run_id)

    rc = _load_json(os.path.join(run_dir, "RUN_COMPLETE.json")) or {}
    verdict = None
    if isinstance(rc, dict):
        verdict = rc.get("status") or rc.get("verdict")
    # fall back to a verdict-shaped token in the narrative reports
    narrative, provenance = {}, []
    for nf in NARRATIVE_FILES:
        p = os.path.join(run_dir, nf)
        if os.path.exists(p):
            secs, lead = _md_sections(p)
            if secs or lead:
                narrative[nf] = (secs, lead)
            provenance.append(nf)
    if not verdict:
        for nf, (secs, lead) in narrative.items():
            blob = " ".join(list(secs.values()) + [lead])
            m = VERDICT_RE.search(blob)
            if m:
                verdict = m.group(1)
                break

    specs = {}
    for sf in SPEC_FILES:
        p = os.path.join(run_dir, sf)
        if os.path.exists(p):
            d = _load_json(p)
            if isinstance(d, dict):
                specs[sf] = d
                provenance.append(sf)

    # a pseudo-summary so the shared note builder can work unchanged
    flat = {}
    for d in specs.values():
        for k, v in d.items():
            if not isinstance(v, (dict, list)) and k not in flat:
                flat[k] = v
    pseudo = dict(flat)
    if verdict:
        pseudo["verdict"] = verdict
    if isinstance(rc, dict):
        for k, v in rc.items():
            if not isinstance(v, (dict, list)):
                pseudo.setdefault(k, v)

    rec = {
        "run_id": run_id, "run_dir": run_dir, "date": run_date(run_id, run_dir),
        "family": fam, "band": band, "blurb": blurb, "verdict": verdict,
        "summary": pseudo, "derived": True, "narrative": narrative,
        "specs": specs, "provenance": provenance,
        "complete": "RUN_COMPLETE.json" in files,
    }
    for key, names in SCALAR_KEYS:
        rec[key] = dig(pseudo, names)
    rec["sentinel_missing"] = False
    rec["csvs"] = [f for f in files if f.lower().endswith(".csv")]
    rec["imgs"] = [f for f in files if f.lower().endswith((".png", ".svg", ".jpg"))]
    rec["jsons"] = [f for f in files if f.lower().endswith(".json")]
    return rec


# ---------------------------------------------------------------- plots

def _numeric_cols(path, limit_rows=200000):
    """Read a CSV with the stdlib; return (header, columns-as-float-lists, nrows)."""
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        rdr = csv.reader(fh)
        try:
            head = next(rdr)
        except StopIteration:
            return [], {}, 0
        cols = {h: [] for h in head}
        n = 0
        for row in rdr:
            if len(row) != len(head):
                continue
            for h, v in zip(head, row):
                try:
                    cols[h].append(float(v))
                except (ValueError, TypeError):
                    cols[h].append(float("nan"))
            n += 1
            if n >= limit_rows:
                break
    good = {h: v for h, v in cols.items()
            if v and sum(1 for x in v if x == x) > max(2, 0.5 * len(v))}
    return head, good, n


X_CANDIDATES = ("t", "time", "gen", "step", "iter", "sep", "n")


def make_plots(rec, plt):
    """Generate plots for one run. Returns list of vault-relative image paths."""
    run_dir, run_id = rec["run_dir"], rec["run_id"]
    made = []
    outdir = os.path.join(PLOTS, run_id)

    def save(fig, name):
        os.makedirs(outdir, exist_ok=True)
        p = os.path.join(outdir, name)
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        made.append("_plots/%s/%s" % (run_id, name))

    # --- special case: TG scalars_*_{well,hill,off}.csv -> one overlay of F_R vs t
    arms = defaultdict(dict)
    for f in rec["csvs"]:
        m = re.match(r"scalars_(.+)_(well|hill|off)\.csv$", f)
        if m:
            arms[m.group(1)][m.group(2)] = os.path.join(run_dir, f)
    for tag, byarm in sorted(arms.items()):
        try:
            fig, ax = plt.subplots(figsize=(8, 4.2))
            plotted = False
            for arm, colour in (("well", "#1f77b4"), ("hill", "#d62728"), ("off", "#7f7f7f")):
                if arm not in byarm:
                    continue
                _h, cols, _n = _numeric_cols(byarm[arm])
                if "t" not in cols or "F_R" not in cols:
                    continue
                ax.plot(cols["t"], cols["F_R"], label=arm, color=colour, lw=1.2)
                plotted = True
            if not plotted:
                plt.close(fig)
                continue
            ax.axhline(0, color="k", lw=0.6, ls=":")
            ax.set_xlabel("t")
            ax.set_ylabel(r"$F_R$   (<0 = attraction)")
            ax.set_title("%s - %s : F_R vs t by arm" % (run_id, tag))
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
            save(fig, "FR_%s.png" % re.sub(r"[^\w.-]", "_", tag))
        except Exception:
            plt.close("all")

    # --- generic: any other CSV with an x-like column
    handled = {f for byarm in arms.values() for p in byarm.values()
               for f in [os.path.basename(p)]}
    skip = ("artifact_hashes", "run_manifest")
    for f in rec["csvs"]:
        if f in handled or len(made) >= 5:
            continue
        if any(s in f for s in skip):
            continue
        path = os.path.join(run_dir, f)
        try:
            if os.path.getsize(path) > 40 * 1024 * 1024:
                continue
            head, cols, n = _numeric_cols(path)
            if n < 3 or not cols:
                continue
            xcol = next((c for c in X_CANDIDATES if c in cols), None)
            ycols = [c for c in head if c in cols and c != xcol][:6]
            if not ycols:
                continue
            fig, axes = plt.subplots(len(ycols), 1, figsize=(8, 1.5 * len(ycols) + 0.8),
                                     sharex=True, squeeze=False)
            for ax, c in zip(axes[:, 0], ycols):
                if xcol:
                    ax.plot(cols[xcol], cols[c], lw=1.0)
                else:
                    ax.plot(cols[c], lw=1.0)
                ax.set_ylabel(c, fontsize=8)
                ax.grid(alpha=0.25)
                ax.tick_params(labelsize=7)
            axes[-1, 0].set_xlabel(xcol or "index", fontsize=8)
            axes[0, 0].set_title("%s - %s (%d rows)" % (run_id, f, n), fontsize=9)
            save(fig, re.sub(r"[^\w.-]", "_", f).replace(".csv", "") + ".png")
        except Exception:
            plt.close("all")
    return made


# ------------------------------------------------- figure & render import

FIGURES = os.path.join(OUT, "_figures")
RENDERS = os.path.join(OUT, "_renders")
VIZ = os.path.join(REPO, "quantule_viz", "outputs")


def _copy_if_new(src, dst):
    try:
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src) \
                and os.path.getmtime(dst) >= os.path.getmtime(src):
            return False
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(src, "rb") as a, open(dst, "wb") as b:
            b.write(a.read())
        return True
    except OSError:
        return False


def import_figures(rec):
    """Copy a run's own rendered images into the vault so they can be embedded.

    sweep_runs/ is gitignored and outside docs/, so Obsidian cannot see or embed anything
    that lives there. These copies are the vault-visible half of the two-layer model.
    """
    run_id, run_dir = rec["run_id"], rec["run_dir"]
    made = []
    for rel in find_images(run_dir):
        src = os.path.join(run_dir, rel)
        flat = rel.replace("/", "__")
        _copy_if_new(src, os.path.join(FIGURES, run_id, flat))
        made.append(("_figures/%s/%s" % (run_id, flat), rel))
    return made


_VIZ_INDEX = None


def build_viz_index(run_ids):
    """Map quantule_viz render outputs back to the run they were rendered from.

    The render directory names embed the source run id, e.g.
    top1_diagnostic_sweep_runs_PHASE_C_VISUAL_ANALYSIS_20260624_161650_cases_k2_...
    """
    global _VIZ_INDEX
    if _VIZ_INDEX is not None:
        return _VIZ_INDEX
    idx = defaultdict(list)
    if not os.path.isdir(VIZ):
        _VIZ_INDEX = idx
        return idx
    ids = sorted(run_ids, key=len, reverse=True)     # longest match wins
    for root, _dirs, files in os.walk(VIZ):
        imgs = [f for f in files if f.lower().endswith(IMG_EXT)]
        if not imgs:
            continue
        key = root.replace("\\", "/")
        hit = next((r for r in ids if r in key), None)
        if hit:
            label = os.path.basename(root)
            for f in sorted(imgs):
                idx[hit].append((os.path.join(root, f), label, f))
    _VIZ_INDEX = idx
    return idx


def import_renders(rec, viz_index):
    run_id = rec["run_id"]
    made = []
    for src, label, fname in viz_index.get(run_id, []):
        flat = "%s__%s" % (label, fname)
        _copy_if_new(src, os.path.join(RENDERS, run_id, flat))
        made.append(("_renders/%s/%s" % (run_id, flat), "%s / %s" % (label, fname)))
    return made


# ---------------------------------------------------------------- note

def build_note(rec, plots):
    s = rec["summary"]
    verdict = rec["verdict"]
    fm = [
        "---",
        "run_id: %s" % yaml_scalar(rec["run_id"]),
        "date: %s" % rec["date"],
        "family: %s" % yaml_scalar(rec["family"]),
        "sector: %s" % yaml_scalar(rec["band"]),
        "verdict: %s" % yaml_scalar(verdict),
        "complete: %s" % ("true" if rec["complete"] else "false"),
        "source: %s" % ("derived" if rec.get("derived") else "summary.json"),
    ]
    for k in ("N", "L", "T", "dt", "elapsed_h", "seed", "git_commit"):
        if rec.get(k) is not None:
            fm.append("%s: %s" % (k, yaml_scalar(rec[k])))
    fm += [
        "n_csv: %d" % len(rec["csvs"]),
        "n_plots: %d" % len(plots),
        "tags: [run, %s, %s]" % (rec["band"], rec["family"].replace("-", "_")),
        "---",
        "",
    ]

    b = ["# %s" % rec["run_id"], ""]
    b.append("*%s* &middot; **%s** &middot; `%s`" % (rec["blurb"], rec["family"], rec["date"]))
    b.append("")

    if verdict:
        b += ["> [!abstract] Verdict", "> `%s`" % verdict, ""]
    else:
        b += ["> [!note] No verdict recorded", "> This run's summary carries no `verdict` key.", ""]

    if rec.get("derived"):
        b += ["> [!info] Derived note - no `summary.json`",
              "> This run predates or bypasses the `summary.json` convention. Fields below are"
              " reconstructed from: %s." % ", ".join("`%s`" % p for p in rec.get("provenance", []))
              or "the run directory.",
              "> The verdict shown is `RUN_COMPLETE.json.status` where present, otherwise the first"
              " verdict-shaped token found in the run's own report - **treat it as indicative and"
              " confirm against the source document.**", ""]

    # narrative sections lifted from the run's own reports
    for nf, (secs, lead) in (rec.get("narrative") or {}).items():
        if not secs and not lead:
            continue
        b += ["## From `%s`" % nf, ""]
        if lead:
            b += [lead[:900].strip(), ""]
        for head, body in secs.items():
            b += ["**%s**" % head, "", body[:1200].strip(), ""]

    if rec.get("sentinel_missing"):
        b += ["> [!warning] Completion sentinel missing",
              "> This run was expected to write a `RUN_COMPLETE.json` and did not - "
              "the summary may be partial.", ""]

    for key in ("note", "boundary"):
        if isinstance(s.get(key), str):
            b += ["**%s:** %s" % (key.capitalize(), s[key]), ""]

    # results table
    for key in ("rows", "results"):
        if isinstance(s.get(key), list) and s[key]:
            tbl, n = rows_table(s[key])
            if tbl:
                b += ["## Results (`%s`, %d rows)" % (key, n), "", tbl, ""]
                if n > 30:
                    b += ["*Table truncated to 30 of %d rows - full data in the run directory.*" % n, ""]
            break

    # notes list
    if isinstance(s.get("notes"), list) and s["notes"]:
        b += ["## Run notes", ""]
        b += ["- %s" % str(x) for x in s["notes"][:20]]
        b += [""]

    # remaining scalar / small-dict keys
    shown = {"verdict", "rows", "results", "notes", "config", "note", "boundary", "command"}
    extras = []
    for k, v in s.items():
        if k in shown:
            continue
        if isinstance(v, (int, float, bool, str)) and not isinstance(v, bool) or isinstance(v, bool):
            extras.append((k, fmt(v)))
        elif isinstance(v, dict) and v and len(v) <= 12 and all(
                not isinstance(x, (dict, list)) for x in v.values()):
            extras.append((k, ", ".join("`%s`=%s" % (a, fmt(bb)) for a, bb in v.items())))
        elif isinstance(v, list) and v and all(not isinstance(x, (dict, list)) for x in v):
            extras.append((k, ", ".join(fmt(x) for x in v[:12])))
    if extras:
        b += ["## Summary fields", "", "| field | value |", "|---|---|"]
        b += ["| `%s` | %s |" % (k, v) for k, v in extras[:40]]
        b += [""]

    # config
    cfg = s.get("config")
    if isinstance(cfg, dict) and cfg:
        items = [(k, v) for k, v in cfg.items() if not isinstance(v, (dict, list))]
        if items:
            b += ["## Configuration", "", "| parameter | value |", "|---|---|"]
            b += ["| `%s` | %s |" % (k, fmt(v)) for k, v in items]
            b += [""]

    if plots:
        b += ["## Plots", "",
              "*Generated from this run's CSVs by `tools/build_run_catalogue.py`. Regenerable — "
              "delete and rebuild to refresh.*", ""]
        for p in plots:
            b += ["![[%s]]" % p, "",
                  "*%s*" % os.path.basename(p).replace(".png", "").replace("_", " "), ""]

    figs = rec.get("figures") or []
    if figs:
        b += ["## Figures (produced by the run itself)", "",
              "*Copied from the run directory so Obsidian can display them. These are the run's own "
              "rendered output, not regenerated here.*", ""]
        for vaultpath, orig in figs:
            b += ["![[%s]]" % vaultpath, "", "*`%s`*" % orig, ""]

    rends = rec.get("renders") or []
    if rends:
        b += ["## Visualiser renders", "",
              "*From `quantule_viz/outputs/`, matched to this run by the render directory name.*", ""]
        for vaultpath, orig in rends:
            b += ["![[%s]]" % vaultpath, "", "*%s*" % orig, ""]

    b += ["## Artifacts", ""]
    b += ["- **Run directory** (gitignored, local only): `sweep_runs/%s/`" % rec["run_id"]]
    if rec["csvs"]:
        b += ["- **CSV data** (%d): %s" % (len(rec["csvs"]),
              ", ".join("`%s`" % c for c in rec["csvs"][:10])
              + (" ..." if len(rec["csvs"]) > 10 else ""))]
    if rec["imgs"]:
        b += ["- **Images in run dir** (%d): %s" % (len(rec["imgs"]),
              ", ".join("`%s`" % c for c in rec["imgs"][:8]))]
    if rec.get("command"):
        b += ["", "```bash", str(rec["command"])[:600], "```"]

    # ---- backlink block (categories per docs/DOCUMENTATION_METHODOLOGY.md) ----
    b += ["", "## Previous experiments", ""]
    prev = rec.get("prev") or []
    if prev:
        b += ["*Auto-derived: the preceding runs in the same family (`%s`). "
              "Correct by hand if the lineage is wrong — edits here survive rebuilds only if you "
              "move the link below the Review notes marker.*" % rec["family"], ""]
        b += ["- [[%s]] &middot; `%s`" % (p["run_id"], p["date"]) for p in prev]
    else:
        b += ["*None — this is the first catalogued run in the `%s` family.*" % rec["family"]]

    b += ["", "## Associated docs", ""]
    assoc = rec.get("assoc") or []
    if assoc:
        b += ["- [[%s]]" % a for a in assoc]
    else:
        b += ["*No prose document in `docs/` names this run id yet.*"]

    b += ["", "## Next experiment", ""]
    nxt = rec.get("next")
    if nxt:
        b += ["- [[%s]] &middot; `%s`" % (nxt["run_id"], nxt["date"])]
    else:
        b += ["*None yet — this is the most recent run in the `%s` family.*" % rec["family"]]

    b += ["", "## Branches", "",
          "- [[Main branch]] &larr; via [[%s|%s sector index]]" % (
              SECTOR_INDEX.get(rec["band"], "_INDEX"), rec["band"]),
          "- Sector: `%s` &middot; family: `%s`" % (rec["band"], rec["family"]),
          "", "---", ""]

    return "\n".join(fm) + "\n".join(b) + "\n"


JAKE_HDR = "### Reading - Jake"
CLAUDE_HDR = "### Reading - Claude"


def seed_review(rec, visual):
    """First-creation content for the preserved section.

    For runs that carry figures this seeds the paired-reading protocol
    (docs/DOCUMENTATION_METHODOLOGY.md §5): two independent reading blocks, so a
    disagreement between them is visible rather than averaged away.
    """
    s = [REVIEW_MARKER, "",
         "*(Preserved across catalogue rebuilds - everything above this line is regenerated.)*", ""]
    if not visual:
        return "\n".join(s) + "\n"
    s += ["> [!important] Paired reading - fill your block before reading the other one.",
          "> A disagreement here is the point, not a problem. Record the resolution; do not",
          "> overwrite the disagreement.", "",
          CLAUDE_HDR, "", "- **Observation:**", "- **Reading:**", "- **Confidence:**",
          "- **What would change my mind:**", "",
          JAKE_HDR, "", "- **Observation:**", "- **Reading:**", "- **Confidence:**",
          "- **What would change my mind:**", "",
          "### Comparison", "", "**Agree on:**", "", "**Disagree on:**", "",
          "**Resolution:**", ""]
    return "\n".join(s) + "\n"


def reading_state(preserved):
    """Has a human actually filled in a reading block, or is it still the skeleton?"""
    if JAKE_HDR not in preserved:
        return "n/a"
    tail = preserved.split(JAKE_HDR, 1)[1]
    tail = tail.split("### ", 1)[0]
    # a bullet counts as answered only if something follows the closing "**"
    for ln in tail.splitlines():
        t = ln.strip()
        if not t.startswith("- **"):
            continue
        after = t.split(":**", 1)[-1] if ":**" in t else t.split("**", 2)[-1]
        if after.strip():
            return "filled"
    return "empty"


def write_note(path, generated, rec=None, visual=False):
    """Write a note, preserving any hand-written review section."""
    preserved = ""
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                old = fh.read()
            idx = old.find(REVIEW_MARKER)
            if idx != -1:
                preserved = old[idx:]
        except OSError:
            pass
    if not preserved:
        preserved = seed_review(rec or {}, visual)
    elif visual and JAKE_HDR not in preserved:
        # note predates the paired-reading protocol, or figures arrived later: append the
        # skeleton without disturbing anything already written.
        preserved = preserved.rstrip() + "\n\n" + seed_review(rec or {}, True).split(
            "regenerated.)*", 1)[1].lstrip("\n")
    if rec is not None:
        rec["reading"] = reading_state(preserved)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(generated + preserved)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    plt = None
    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: F811
        except Exception as e:
            print("WARN matplotlib unavailable (%s); continuing without plots" % e)

    os.makedirs(OUT, exist_ok=True)
    all_dirs = sorted(d for d in os.listdir(SWEEP)
                      if os.path.isdir(os.path.join(SWEEP, d)))
    with_summary, without = [], []
    for d in all_dirs:
        p = os.path.join(SWEEP, d)
        (with_summary if os.path.exists(os.path.join(p, "summary.json")) else without).append(p)

    # round 2: promote the substantive no-summary runs (Codex-lane convention)
    promoted, remainder = [], []
    for p in without:
        (promoted if triage_class(walk_run(p)) == "substantive" else remainder).append(p)

    if args.limit:
        with_summary = with_summary[:args.limit]
        promoted = promoted[:args.limit]
    print("run dirs: %d | with summary.json: %d | promoted (derived): %d | remaining: %d"
          % (len(all_dirs), len(with_summary), len(promoted), len(remainder)))

    targets = [(rd, False) for rd in with_summary] + [(rd, True) for rd in promoted]

    # pass 1: extract every record so lineage and associations can be computed globally
    recs, failed = [], []
    for rd, derived in targets:
        try:
            recs.append(extract_nosummary(rd) if derived else extract(rd))
        except Exception as e:
            failed.append((os.path.basename(rd), repr(e)))

    # lineage: previous / next run within the same family, by date then run id
    byfam = defaultdict(list)
    for r in recs:
        byfam[r["family"]].append(r)
    for fam, rs in byfam.items():
        rs.sort(key=lambda x: (x["date"], x["run_id"]))
        for j, r in enumerate(rs):
            r["prev"] = rs[max(0, j - 3):j][::-1]
            r["next"] = rs[j + 1] if j + 1 < len(rs) else None

    # associations: prose documents in docs/ that name this run id
    doc_text = {}
    for root, dirs, files in os.walk(VAULT):
        dirs[:] = [d for d in dirs if d not in ("runs", ".obsidian", "_templates", ".trash")]
        for f in files:
            if f.endswith(".md"):
                p = os.path.join(root, f)
                try:
                    with open(p, encoding="utf-8", errors="replace") as fh:
                        doc_text[os.path.relpath(p, VAULT).replace("\\", "/")[:-3]] = fh.read()
                except OSError:
                    pass
    for r in recs:
        r["assoc"] = sorted(k for k, t in doc_text.items() if r["run_id"] in t)[:8]

    viz_index = build_viz_index([r["run_id"] for r in recs])

    n_total = len(recs)
    total_plots = total_figs = total_rends = 0
    for i, rec in enumerate(recs, 1):
        rd = rec["run_dir"]
        rec["figures"] = import_figures(rec)
        rec["renders"] = import_renders(rec, viz_index)
        total_figs += len(rec["figures"])
        total_rends += len(rec["renders"])
        plots = []
        if plt is not None:
            try:
                plots = make_plots(rec, plt)
            except Exception:
                traceback.print_exc(limit=1)
        else:
            # --no-plots must not silently strip embeds from already-built notes:
            # reuse whatever is already on disk for this run.
            pdir = os.path.join(PLOTS, rec["run_id"])
            if os.path.isdir(pdir):
                plots = ["_plots/%s/%s" % (rec["run_id"], f)
                         for f in sorted(os.listdir(pdir)) if f.endswith(".png")]
        total_plots += len(plots)
        rec["plots"] = plots
        visual = bool(plots or rec["figures"] or rec["renders"])
        write_note(os.path.join(OUT, "%s.md" % rec["run_id"]),
                   build_note(rec, plots), rec=rec, visual=visual)
        if i % 20 == 0:
            print("  ... %d/%d" % (i, n_total))

    write_index(recs, len(all_dirs), remainder, failed)
    write_branch_indexes(recs)
    write_tracker(recs)
    n_gal = write_viz_galleries()
    write_triage(remainder, promoted_ids={os.path.basename(p) for p in promoted})

    print("\nWROTE %d run notes -> docs/runs/" % len(recs))
    print("plots generated: %d | figures imported: %d | renders imported: %d | gallery images: %d"
          % (total_plots, total_figs, total_rends, n_gal))
    print("with verdict: %d | no verdict: %d"
          % (sum(1 for r in recs if r["verdict"]), sum(1 for r in recs if not r["verdict"])))
    if failed:
        print("FAILED to parse %d:" % len(failed))
        for n, e in failed[:10]:
            print("  ", n, e)


def write_index(recs, n_all, without, failed):
    byband = defaultdict(list)
    for r in recs:
        byband[r["band"]].append(r)
    fams = Counter(r["family"] for r in recs)
    verd = Counter()
    for r in recs:
        if isinstance(r["verdict"], str):
            verd[r["verdict"]] += 1

    L = ["---", "tags: [index, runs]", "---", "", "# Run Catalogue", "",
         "Every substantive simulation run under `sweep_runs/`, as one note each. Most come from a",
         "`summary.json`; runs on the Codex convention are **derived** from `RUN_COMPLETE.json` plus",
         "their own handoff reports and are marked `source: derived` in frontmatter.",
         "Generated by `tools/build_run_catalogue.py`; rebuild any time - hand-written",
         "**Review notes** sections are preserved.", "",
         "> [!info] Two-layer model",
         "> These notes and the plots under `_plots/` are git-tracked and live in the vault.",
         "> The raw run data in `sweep_runs/` is **gitignored and local-only** - notes reference it by path.",
         "> A note existing here is not a claim that its verdict is current; "
         "[[../IRER_MASTER_HYPOTHESIS_CATALOG|the master catalog]] is the authority on live status.", "",
         "## Coverage", "",
         "| | count |", "|---|---:|",
         "| run directories under `sweep_runs/` | %d |" % n_all,
         "| **catalogued here** | **%d** |" % len(recs),
         "| &nbsp;&nbsp;from `summary.json` | %d |" % sum(1 for r in recs if not r.get("derived")),
         "| &nbsp;&nbsp;derived (Codex-lane: `RUN_COMPLETE.json` + handoff reports) | %d |"
         % sum(1 for r in recs if r.get("derived")),
         "| of those, carrying a `verdict` | %d |" % sum(1 for r in recs if r["verdict"]),
         "| with a `RUN_COMPLETE.json` sentinel | %d |" % sum(1 for r in recs if r["complete"]),
         "| plots generated from CSVs | %d |" % sum(len(r.get("plots") or []) for r in recs),
         "| figures imported from runs | %d |" % sum(len(r.get("figures") or []) for r in recs),
         "| visualiser renders imported | %d |" % sum(len(r.get("renders") or []) for r in recs),
         "| runs with any visual | %d |" % sum(1 for r in recs if (r.get("plots") or r.get("figures") or r.get("renders"))),
         "| **awaiting a second reading** -> [[../EXPERIMENT_TRACKER]] | **%d** |"
         % sum(1 for r in recs if r.get("reading") == "empty"),
         "| not catalogued (thin/empty) -> [[_TRIAGE_NO_SUMMARY]] | %d |" % len(without),
         ""]
    if failed:
        L += ["| unparseable summaries | %d |" % len(failed), ""]

    L += ["## By family", "", "| family | runs |", "|---|---:|"]
    L += ["| %s | %d |" % (k, v) for k, v in fams.most_common()]
    L += ["", "## All runs", "",
          "```dataview", "TABLE date, family, verdict, N, elapsed_h AS \"hours\"",
          "FROM #run", "SORT date DESC", "```", "",
          "*(Requires the Dataview plugin. The static listing below always works.)*", ""]

    BANDS = [("gravity", "Gravity / TG ladder"), ("transport", "Transport sector"),
             ("stability", "Stability sector"), ("infra", "Validation & diagnostics"),
             ("other", "Other")]
    for band, label in BANDS:
        rs = byband.get(band)
        if not rs:
            continue
        L += ["### %s (%d)" % (label, len(rs)), "",
              "| run | date | family | verdict |", "|---|---|---|---|"]
        for r in sorted(rs, key=lambda x: (x["date"], x["run_id"]), reverse=True):
            v = r["verdict"] or "-"
            L.append("| [[%s]] | %s | %s | `%s` |" % (r["run_id"], r["date"], r["family"], v))
        L += [""]

    L += ["## Verdict frequency", "", "| verdict | n |", "|---|---:|"]
    L += ["| `%s` | %d |" % (k, v) for k, v in verd.most_common(40)]
    L += ["", "---", "",
          "*Standing posture: no matter claim, no gravity claim, no emergent-physics claim.*", ""]

    with open(os.path.join(OUT, "_INDEX.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


IMG_EXT = (".png", ".svg", ".jpg", ".jpeg", ".gif")


def walk_run(run_dir, max_depth=3):
    """Relative paths of every file in a run dir, recursing a few levels.

    Figures are often nested one level down (e.g. hifi_N128_L10_T1600/slices_stable.png),
    so a flat listdir misses them.
    """
    out = []
    base = os.path.abspath(run_dir)
    for root, dirs, files in os.walk(run_dir):
        depth = os.path.abspath(root)[len(base):].count(os.sep)
        if depth >= max_depth:
            dirs[:] = []
        for f in files:
            out.append(os.path.relpath(os.path.join(root, f), run_dir).replace("\\", "/"))
    return sorted(out)


def find_images(run_dir):
    return [f for f in walk_run(run_dir) if f.lower().endswith(IMG_EXT)]


def triage_class(files):
    """Substantive runs carry real output even without a summary.json.

    Rendered figures count: a run whose only artifact is a set of plots is exactly the
    kind of visual-analysis run this vault exists to review.
    """
    if not files:
        return "empty"
    csv_n = sum(1 for f in files if f.endswith(".csv"))
    md_n = sum(1 for f in files if f.endswith(".md"))
    img_n = sum(1 for f in files if f.lower().endswith(IMG_EXT))
    if "RUN_COMPLETE.json" in files or csv_n >= 2 or md_n or img_n:
        return "substantive"
    return "minor"


def write_branch_indexes(recs):
    """One index note per branch. Per docs/Side branch.md, ONLY these link to [[Main branch]],
    and each carries a chronological tracker of its own notes at the bottom."""
    byband = defaultdict(list)
    for r in recs:
        byband[r["band"]].append(r)
    for band, name in SECTOR_INDEX.items():
        rs = sorted(byband.get(band, []), key=lambda x: (x["date"], x["run_id"]))
        kind = SECTOR_KIND.get(band, "side")
        L = ["---", "tags: [branch, index, %s]" % band,
             "branch_kind: %s" % kind, "---", "",
             "# Branch — %s" % SECTOR_TITLE.get(band, band), "",
             "**Branch kind: %s.** " % kind.upper() +
             ("Works directly toward the goals in [[Main branch]]."
              if kind == "main" else
              "Exploratory / robustness / scoping. Does *not* directly advance the "
              "[[Main branch]] goals — kept separate on purpose."), "",
             "> [!info] Why this note exists",
             "> Per [[Side branch]], **only a branch index links back to [[Main branch]]**. Individual",
             "> run and experiment notes link here instead. That is what stops exploratory work",
             "> drifting into the main line uncontrolled and keeps graph view readable.", "",
             "- Parent: [[Main branch]]",
             "- Methodology: [[DOCUMENTATION_METHODOLOGY]]",
             "- All runs: [[runs/_INDEX|Run Catalogue]] &middot; tracker: [[EXPERIMENT_TRACKER]]", "",
             "## Chronology (%d runs)" % len(rs), "",
             "| date | run | family | verdict | figures | reading |",
             "|---|---|---|---|---:|---|"]
        for r in rs:
            nvis = len(r.get("plots") or []) + len(r.get("figures") or []) + len(r.get("renders") or [])
            rd = {"filled": "✅", "empty": "⬜", "n/a": "—"}.get(r.get("reading", "n/a"), "—")
            L.append("| %s | [[%s]] | %s | `%s` | %s | %s |"
                     % (r["date"], r["run_id"], r["family"], r["verdict"] or "-",
                        nvis or "", rd))
        L += ["", "## Open threads", "",
              "*Hand-maintained. What this branch is currently trying to settle.*", "",
              "- ", "", "---", ""]
        write_note(os.path.join(VAULT, "%s.md" % name), "\n".join(L) + "\n")


def write_tracker(recs):
    """Chronological experiment -> result tracker.

    Deliberately simple (Jake's spec): one row per experiment in time order, with a link to
    the figure-bearing note and whether a second reading exists. The paired readings themselves
    live below each run note's Review-notes marker so they survive rebuilds.
    """
    rs = sorted(recs, key=lambda x: (x["date"], x["run_id"]))
    vis = [r for r in rs if (r.get("plots") or r.get("figures") or r.get("renders"))]
    pending = [r for r in vis if r.get("reading") == "empty"]

    L = ["---", "tags: [index, tracker]", "---", "", "# Experiment Tracker", "",
         "Every catalogued run in **chronological order**: what was run, what came out, and whether a "
         "second reading of its visuals exists.", "",
         "> [!important] What this is for",
         "> Comparing **two independent readings of the same figure**. Each visual run's note carries a",
         "> paired-reading block below its Review-notes marker: one reading by Claude, one by Jake,",
         "> each written before seeing the other. A disagreement between them is the most useful signal",
         "> in the vault — record the resolution, do not overwrite the disagreement.",
         "> See [[DOCUMENTATION_METHODOLOGY]] §5.", "",
         "> [!warning] Verdicts here are historical",
         "> A row records what a run concluded at the time. Only [[IRER_MASTER_HYPOTHESIS_CATALOG]]",
         "> says what is currently live.", "",
         "## At a glance", "", "| | count |", "|---|---:|",
         "| experiments tracked | %d |" % len(rs),
         "| with visual output | %d |" % len(vis),
         "| **awaiting a second reading** | **%d** |" % len(pending),
         "| second reading recorded | %d |" % sum(1 for r in vis if r.get("reading") == "filled"),
         "", "## Visual review queue", "",
         "*Runs with figures whose paired-reading block is still empty — highest value first "
         "(most recent).*", ""]
    if pending:
        L += ["| date | run | family | verdict | figures |", "|---|---|---|---|---:|"]
        for r in sorted(pending, key=lambda x: x["date"], reverse=True)[:40]:
            n = len(r.get("plots") or []) + len(r.get("figures") or []) + len(r.get("renders") or [])
            L.append("| %s | [[%s]] | %s | `%s` | %d |"
                     % (r["date"], r["run_id"], r["family"], r["verdict"] or "-", n))
        if len(pending) > 40:
            L.append("")
            L.append("*...and %d more.*" % (len(pending) - 40))
    else:
        L += ["*Queue empty — every visual run has a second reading.*"]

    L += ["", "## Chronology", "", "| date | experiment | branch | result | figs | reading |",
          "|---|---|---|---|---:|---|"]
    for r in rs:
        n = len(r.get("plots") or []) + len(r.get("figures") or []) + len(r.get("renders") or [])
        rd = {"filled": "✅", "empty": "⬜", "n/a": "—"}.get(r.get("reading", "n/a"), "—")
        L.append("| %s | [[%s]] | [[%s\\|%s]] | `%s` | %s | %s |"
                 % (r["date"], r["run_id"], SECTOR_INDEX.get(r["band"], "_INDEX"),
                    r["band"], r["verdict"] or "-", n or "", rd))
    L += ["", "---", "",
          "*Standing posture: no matter claim, no gravity claim, no emergent-physics claim.*", ""]
    write_note(os.path.join(VAULT, "EXPERIMENT_TRACKER.md"), "\n".join(L) + "\n")


VIZ_BLURB = {
    "conservative_geometry_campaign": ("transport", "Conservative-geometry campaign renders "
                                       "(the C2 arc that the C2.6 geometry-off bug reshaped)."),
    "k6_mid_mass_true_emergence": ("stability", "k6 mid-mass emergence stills and montages."),
    "triangle_cupy_screen": ("stability", "Triangle-layout CuPy screening renders."),
    "triangle_layout_diagnostic": ("stability", "Triangle-layout diagnostic renders."),
    "candidate_2node_3node_search": ("stability", "Two/three-node candidate search — top "
                                     "diagnostics, matched to source runs where the render "
                                     "directory names them."),
    "gravity_geometry_characterization": ("gravity", "Gravity-geometry characterization renders."),
}
GALLERY = os.path.join(OUT, "_gallery")


def write_viz_galleries():
    """One gallery note per quantule_viz campaign folder.

    Most viz output belongs to a campaign rather than a single run, so it has no run note to
    live in. Without this those images stay invisible to the vault.
    """
    if not os.path.isdir(VIZ):
        return 0
    camps = defaultdict(list)
    for root, _d, files in os.walk(VIZ):
        imgs = sorted(f for f in files if f.lower().endswith(IMG_EXT))
        if not imgs:
            continue
        rel = os.path.relpath(root, VIZ).replace(os.sep, "/")
        top = rel.split("/")[0]
        for f in imgs:
            camps[top].append((os.path.join(root, f), rel, f))
    n = 0
    for camp, items in sorted(camps.items()):
        band, blurb = VIZ_BLURB.get(camp, ("other", "Visualiser output."))
        L = ["---", "tags: [gallery, viz, %s]" % band, "---", "",
             "# Gallery — %s" % camp.replace("_", " "), "",
             blurb, "",
             "> [!info] Source",
             "> `quantule_viz/outputs/%s/` — gitignored and outside the vault; these are copies so" % camp,
             "> Obsidian can display them. %d images." % len(items), "",
             "- Branch: [[%s]]" % SECTOR_INDEX.get(band, "Branch - Unsorted - Index"),
             "- Methodology: [[DOCUMENTATION_METHODOLOGY]] §5", "",
             "> [!warning] Every figure needs a sentence",
             "> These are imported unannotated. Add a description under any image you review —",
             "> below the Review-notes marker, or inline here if you prefer (inline edits above the",
             "> marker are lost on rebuild).", ""]
        bysub = defaultdict(list)
        for src, rel, f in items:
            bysub[rel].append((src, f))
        for rel in sorted(bysub):
            L += ["## `%s`" % rel, ""]
            for src, f in sorted(bysub[rel]):
                flat = (rel.replace("/", "__") + "__" + f)
                _copy_if_new(src, os.path.join(GALLERY, camp, flat))
                L += ["![[_gallery/%s/%s]]" % (camp, flat), "", "*%s*" % f, ""]
                n += 1
        L += ["---", ""]
        write_note(os.path.join(OUT, "Gallery - %s.md" % camp), "\n".join(L) + "\n")
    return n


def write_triage(without, promoted_ids=frozenset()):
    rows = []
    for p in without:
        rid = os.path.basename(p)
        files = walk_run(p)
        rows.append((triage_class(files), rid, run_date(rid, p), len(files),
                     sum(1 for f in files if f.endswith(".csv")),
                     ", ".join(sorted(files)[:6]) + (" ..." if len(files) > 6 else "")))
    rows.sort(key=lambda r: (r[2], r[1]), reverse=True)
    counts = Counter(r[0] for r in rows)

    L = ["---", "tags: [index, runs, triage]", "---", "", "# Triage - runs with no `summary.json`", "",
         "%d run directories under `sweep_runs/` carry no `summary.json` and are therefore **not**"
         % len(rows),
         "catalogued as notes.", "",
         "> [!success] The %d **substantive** no-summary runs have been promoted" % len(promoted_ids),
         "> Codex-lane runs use a different convention - `RUN_COMPLETE.json` with a `status` field,"
         " a `TECHNICAL_HANDOFF.md`, preregistered matrices and model specs - rather than a"
         " `summary.json`. They now have derived notes in [[_INDEX|the catalogue]], marked"
         " `source: derived`.",
         "> What remains below is genuinely thin: smoke tests, aborted launches, single-file dirs,"
         " and empty directories.", "",
         "| class | n | meaning |", "|---|---:|---|",
         "| `minor` | %d | smoke tests, aborted launches, single-file dirs |" % counts.get("minor", 0),
         "| `empty` | %d | no files at all; safe to delete |" % counts.get("empty", 0),
         "",
         "To promote any of these, give the run a `summary.json` (or write the note by hand) and rebuild.", "",
         "| class | run | date | files | csv | contents |", "|---|---|---|---:|---:|---|"]
    L += ["| `%s` | `%s` | %s | %d | %d | %s |" % r for r in rows]
    L += [""]
    with open(os.path.join(OUT, "_TRIAGE_NO_SUMMARY.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
