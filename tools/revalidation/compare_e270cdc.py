#!/usr/bin/env python
"""Apply the PRE-REGISTERED decision rules of docs/instrument_integrity/REVALIDATION_E270CDC_RESULTS.md
to the B4 replay, original vs fixed solver. Mechanical on purpose: the rules were committed before the
replay started (d3d87fd) and this script only evaluates them.

    python tools/revalidation/compare_e270cdc.py [--sweep sweep_runs] [--out docs/instrument_integrity/evidence/revalidation_e270cdc]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ORIG = {"ladder": "FEB_GAIN_LADDER_LONGT_T72000_20260701_175708", "confirm": "FEB_ASTAR_CONFIRM_20260702_003055",
        "c27": "C27_REDERIVE", "c1": "c1_longT_confirm.json"}
NEW = {"ladder": "FEB_GAIN_LADDER_LONGT_T72000_REVAL_e270cdc", "ladder_dt2": "FEB_GAIN_LADDER_LONGT_T72000_DT2_REVAL_e270cdc",
       "confirm": "FEB_ASTAR_CONFIRM_REVAL_e270cdc", "c27": "C27_REDERIVE_REVAL_e270cdc", "c1": "PHASE_D_C1_TRANSPORT_REVAL_e270cdc"}


def rows(path):
    p = glob.glob(os.path.join(path, "*_results.csv"))
    return {r["key"]: r for r in csv.DictReader(open(p[0], newline=""))} if p else {}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def slope_table(sweep):
    """{a_factor: (old_slope50, new_slope50, old_klass, new_klass, label)} across ladder + confirm."""
    out = {}
    for kind in ("ladder", "confirm"):
        o, n = rows(os.path.join(sweep, ORIG[kind])), rows(os.path.join(sweep, NEW[kind]))
        for key, r in n.items():
            ro = o.get(key, {})
            out[key] = {"a_factor": f(r.get("a_factor")), "T": r.get("T"), "seed": r.get("seed", "20260619"),
                        "old_slope50": f(ro.get("late_slope_50pct_per1k")), "new_slope50": f(r.get("late_slope_50pct_per1k")),
                        "old_klass": ro.get("klass"), "new_klass": r.get("klass"),
                        "old_n_fin": ro.get("n_fin"), "new_n_fin": r.get("n_fin")}
    return out


def astar_rule(tab, dt2):
    """Rule 1 (pre-registered)."""
    by = {k: v for k, v in tab.items()}
    pos = sorted((v["a_factor"], v["new_slope50"]) for k, v in by.items()
                 if str(v["seed"]) in ("20260619", "", "None") and v["T"] in ("72000", 72000))
    crossing = None
    for (a1, s1), (a2, s2) in zip(pos, pos[1:]):
        if s1 < 0 <= s2 or s1 <= 0 < s2:
            crossing = (a1, a2)
    checks = {}
    checks["crossing_bracket"] = crossing
    checks["crossing_in_1.15_1.16"] = bool(crossing and crossing[0] >= 1.15 - 1e-9 and crossing[1] <= 1.16 + 1e-9)
    lt = by.get("a1.15_longT", {}).get("new_slope50", float("nan"))
    checks["a1.15_T144k_slope"] = lt
    checks["a1.15_T144k_flat"] = abs(lt) < 1e-3
    seeds = [by.get(k, {}).get("new_slope50", float("nan")) for k in ("a1.15_seed620", "a1.15_seed621")]
    checks["seed_slopes"] = seeds
    checks["seeds_flat"] = all(abs(s) < 1e-3 for s in seeds)
    base = by.get("a1.15_ladder_T72000", {}).get("new_slope50", float("nan"))
    checks["dt2_slope"], checks["base_slope"] = dt2, base
    checks["dt_converged"] = (dt2 == dt2) and (base == base) and (dt2 * base >= 0 or abs(dt2 - base) < 5e-4) \
        and abs(dt2 - base) < 5e-4
    ok = all(checks[k] for k in ("crossing_in_1.15_1.16", "a1.15_T144k_flat", "seeds_flat", "dt_converged"))
    return ("A_STAR_UNCHANGED" if ok else "A_STAR_SHIFTED"), checks


def c27_rule(sweep):
    p = os.path.join(sweep, NEW["c27"], "r3_n96.json")
    if not os.path.exists(p):
        return "C27_PENDING", {}
    r3 = json.load(open(p))
    old = json.load(open(os.path.join(sweep, ORIG["c27"], "r3_n96.json")))
    ok = r3.get("verdict") == "CLEAN_TRANSPORT_N96_CONFIRMED" and all(
        0.999 <= b["v_frac"] <= 1.001 and b["mass_ret"] >= 0.999 for b in r3.get("boosts", []))
    return ("C2_7_UNCHANGED" if ok else "C2_7_SHIFTED"), {
        "new": [{k: b[k] for k in ("n", "v_frac", "mass_ret", "r2")} for b in r3.get("boosts", [])],
        "old": [{k: b[k] for k in ("n", "v_frac", "mass_ret", "r2")} for b in old.get("boosts", [])],
        "verdict_new": r3.get("verdict"), "verdict_old": old.get("verdict")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=os.path.join(ROOT, "sweep_runs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "instrument_integrity", "evidence", "revalidation_e270cdc"))
    a = ap.parse_args(argv)
    tab = slope_table(a.sweep)
    dt2r = rows(os.path.join(a.sweep, NEW["ladder_dt2"]))
    dt2 = f(next(iter(dt2r.values()), {}).get("late_slope_50pct_per1k")) if dt2r else float("nan")
    need = {"a1.15_ladder_T72000", "a1.15_longT", "a1.15_seed620", "a1.15_seed621", "a1.16"}
    if not need <= set(tab) or dt2 != dt2:
        va, ca = "A_STAR_PENDING", {"missing": sorted(need - set(tab)) + ([] if dt2 == dt2 else ["dt2 cell"])}
    else:
        va, ca = astar_rule(tab, dt2)
    vc, cc = c27_rule(a.sweep)
    res = {"a_star": {"verdict": va, "checks": ca}, "c27": {"verdict": vc, "checks": cc}, "cells": tab}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "decision_rules.json"), "w"), indent=2, default=str)
    print(json.dumps({"a_star": va, "c27": vc}, indent=2))
    for k, v in sorted(tab.items(), key=lambda kv: (kv[1]["a_factor"], kv[0])):
        print("%-24s a x%.3f  slope50/1k old %+.4f new %+.4f  klass %s -> %s  n_fin %s -> %s" % (
            k, v["a_factor"], v["old_slope50"], v["new_slope50"], v["old_klass"], v["new_klass"],
            v["old_n_fin"], v["new_n_fin"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
