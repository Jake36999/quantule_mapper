"""Quarantined Level-3 empirical comparator (OPT-IN).

Compares digitized experimental data (from published figures) against internal Quantule Mapper
results. Currently implements the V1 force-law comparison; V5 is scaffolded for later.

QUARANTINE (see docs/external_validation/empirical_comparison_branch/README.md):
  * This script is NOT part of `run_external_validation.py --metric all`; it must be invoked
    explicitly.
  * Match-level is fixed at CLOSE ANALOGUE. Nothing here promotes a verdict, gates production,
    or feeds the Hunter.
  * A dataset is refused unless it has a completed provenance record (all governance checkboxes
    ticked, match-level CLOSE ANALOGUE).
  * All output is PROVISIONAL_UNTIL_CLAUDE_REVIEW.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MATCH_LEVEL_CEILING = "CLOSE ANALOGUE"
INTERNAL_V1_RESULT = REPO / "docs/external_validation/generated_metrics/v1/result.json"
INTERNAL_V1_LAMBDA_FALLBACK = 1.9811  # C2 fit, canonical ~2A (see V1 result)


# --------------------------------------------------------------------------- guardrails
def check_provenance(provenance_path: Path) -> tuple[bool, list[str]]:
    """Refuse the dataset unless provenance is complete: no unchecked governance boxes and the
    CLOSE ANALOGUE ceiling is present."""
    problems: list[str] = []
    if not provenance_path.exists():
        return False, [f"Provenance record missing: {provenance_path}"]
    text = provenance_path.read_text(encoding="utf-8", errors="replace")
    if "- [ ]" in text:
        problems.append("Provenance has unchecked governance boxes ('- [ ]'); complete them first.")
    if MATCH_LEVEL_CEILING not in text:
        problems.append(f"Provenance must state match-level '{MATCH_LEVEL_CEILING}'.")
    if "TEMPLATE" in provenance_path.stem.upper():
        problems.append("Refusing to use the unfilled TEMPLATE as a provenance record.")
    return (not problems), problems


# --------------------------------------------------------------------------- internal ref
def internal_v1_lambda() -> tuple[float, str]:
    try:
        data = json.loads(INTERNAL_V1_RESULT.read_text(encoding="utf-8"))
        lam = data["fit_quality"]["c2_fit_quality"]["lambda"]
        if isinstance(lam, (int, float)) and math.isfinite(lam):
            return float(lam), "generated_metrics/v1 (C2 fit)"
    except Exception:
        pass
    return INTERNAL_V1_LAMBDA_FALLBACK, "fallback constant"


# --------------------------------------------------------------------------- data schema
# Digitized V1 dataset (CSV) required columns:
#   relative_phase        radians (Delta_phi)
#   q_half_separation     dimensionless (in soliton widths); half the pair separation
#   interaction_measure   a signed force proxy: >0 = repulsive (separation grows),
#                         <0 = attractive (separation shrinks). Any monotone force proxy is fine
#                         (initial acceleration, inverse squared oscillation period, etc.) as long
#                         as its SIGN encodes attract/repel and its MAGNITUDE scales with force.
def load_v1_dataset(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"relative_phase", "q_half_separation", "interaction_measure"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Digitized V1 dataset missing columns: {sorted(missing)}")
        for r in reader:
            try:
                rows.append(
                    {
                        "relative_phase": float(r["relative_phase"]),
                        "q_half_separation": float(r["q_half_separation"]),
                        "interaction_measure": float(r["interaction_measure"]),
                    }
                )
            except (TypeError, ValueError):
                continue
    if len(rows) < 3:
        raise ValueError("Need at least 3 usable digitized rows.")
    return rows


# --------------------------------------------------------------------------- V1 comparison
# A decay-rate fit is only trustworthy with enough distinct separations and a real linear trend.
DECAY_R2_MIN = 0.5
DECAY_MIN_DISTINCT_Q = 3


def compare_v1(rows: list[dict[str, float]]) -> dict[str, Any]:
    lam_int, lam_src = internal_v1_lambda()
    # Sign law: attractive (measure<0) when cos(dphi)>0, repulsive (measure>0) when cos(dphi)<0.
    neutral_band = 0.15
    sign_rows, sign_hits = 0, 0
    per_q: dict[float, list[int]] = {}  # q -> [hits, total]
    for r in rows:
        c = math.cos(r["relative_phase"])
        if abs(c) < neutral_band:
            continue
        expected = -1 if c > 0 else 1  # -1 attract, +1 repel
        m = r["interaction_measure"]
        if abs(m) < 1e-12:
            continue
        got = 1 if m > 0 else -1
        hit = int(expected == got)
        sign_rows += 1
        sign_hits += hit
        q = round(r["q_half_separation"], 4)
        bucket = per_q.setdefault(q, [0, 0])
        bucket[0] += hit
        bucket[1] += 1
    sign_accuracy = (sign_hits / sign_rows) if sign_rows else None
    sign_by_q = {
        str(q): {"hits": h, "n": t, "accuracy": (h / t if t else None)}
        for q, (h, t) in sorted(per_q.items())
    }

    # Decay rate: |measure / cos(dphi)| = C*exp(-lambda*q). Fit log-amplitude vs q.
    xs, ys = [], []
    for r in rows:
        c = math.cos(r["relative_phase"])
        if abs(c) < 1e-8:
            continue
        amp = abs(r["interaction_measure"] / c)
        if amp > 0:
            xs.append(r["q_half_separation"])
            ys.append(math.log(amp))
    lam_raw = None
    r_squared = None
    n_distinct_q = len({round(x, 4) for x in xs})
    if len(xs) >= 3 and n_distinct_q >= 2:
        x = np.array(xs)
        y = np.array(ys)
        slope, intercept = np.polyfit(x, y, 1)
        lam_raw = float(-slope)
        pred = intercept + slope * x
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = (1.0 - ss_res / ss_tot) if ss_tot > 0 else None

    # Gate the reported lambda: a meaningless fit (too few distinct q, or low R^2) must NOT be
    # presented as a comparable number. Only a reliable fit yields experimental_lambda / ratio.
    decay_reliable = (
        lam_raw is not None
        and r_squared is not None
        and r_squared >= DECAY_R2_MIN
        and n_distinct_q >= DECAY_MIN_DISTINCT_Q
    )
    lam_exp = lam_raw if decay_reliable else None
    lam_ratio = (lam_exp / lam_int) if (lam_exp and lam_int) else None
    if not decay_reliable:
        if n_distinct_q < DECAY_MIN_DISTINCT_Q:
            decay_reason = f"only {n_distinct_q} distinct q value(s); need >= {DECAY_MIN_DISTINCT_Q}"
        elif r_squared is None:
            decay_reason = "no fit"
        else:
            decay_reason = f"R^2 {r_squared:.3f} < {DECAY_R2_MIN}"
    else:
        decay_reason = "reliable"
    return {
        "metric": "v1",
        "internal_lambda": lam_int,
        "internal_lambda_source": lam_src,
        "experimental_lambda": lam_exp,
        "experimental_lambda_raw_unreliable": (lam_raw if not decay_reliable else None),
        "lambda_ratio_exp_over_internal": lam_ratio,
        "decay_fit_r_squared": r_squared,
        "decay_fit_reliable": decay_reliable,
        "decay_fit_reason": decay_reason,
        "n_distinct_q": n_distinct_q,
        "phase_sign_accuracy": sign_accuracy,
        "phase_sign_accuracy_by_q": sign_by_q,
        "n_sign_rows": sign_rows,
        "n_rows": len(rows),
        "match_level": MATCH_LEVEL_CEILING,
    }


# --------------------------------------------------------------------------- V5 collision
INTERNAL_V5_RESULT = REPO / "docs/external_validation/generated_metrics/v5/result.json"
V5_TRANSMITTED = {"PASS_THROUGH", "TRAJECTORY_JUMP", "NO_COLLAPSE", "BOUNCE"}
V5_NOT_TRANSMITTED = {"CAPTURE", "COLLAPSE", "MERGE"}
V5_AMBIGUOUS = {"INCONCLUSIVE", "UNKNOWN"}
V5_ALL_OUTCOMES = V5_TRANSMITTED | V5_NOT_TRANSMITTED | V5_AMBIGUOUS


def internal_v5_reference() -> dict[str, Any]:
    """Read the C3 phase×speed grid summary as the internal direction reference (data, not prose)."""
    try:
        data = json.loads(INTERNAL_V5_RESULT.read_text(encoding="utf-8"))
        fq = data["fit_quality"]
        return {
            "available": True,
            "source": "generated_metrics/v5",
            "non_antiphase_capture_fraction": fq.get("non_antiphase_capture_fraction"),
            "anti_phase_transmission_cells": fq.get("anti_phase_transmission_cells"),
            "outcome_counts": fq.get("outcome_counts"),
            # internal direction: transmission occurs ONLY at anti-phase (non-antiphase capture = 1.0)
            "internal_transmission_only_at_antiphase": (fq.get("non_antiphase_capture_fraction") == 1.0),
        }
    except Exception:
        return {"available": False, "internal_transmission_only_at_antiphase": True,
                "note": "internal C3 grid: transmission only at exact anti-phase; else capture (from memory of V5)."}


def phase_class(dphi: float) -> str:
    d = abs(math.atan2(math.sin(dphi), math.cos(dphi)))  # fold to [0, pi]
    if d <= 0.4:
        return "in_phase"
    if abs(d - math.pi) <= 0.4:
        return "anti_phase"
    return "off_phase"


def load_v5_dataset(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"relative_phase_rad", "observed_outcome"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Digitized V5 dataset missing columns: {sorted(missing)}")
        for r in reader:
            inc = str(r.get("include_in_comparison", "true")).strip().lower()
            if inc in {"false", "0", "no"}:
                continue
            outcome = str(r.get("observed_outcome", "")).strip().upper()
            if outcome not in V5_ALL_OUTCOMES:
                continue
            try:
                dphi = float(r["relative_phase_rad"])
            except (TypeError, ValueError):
                continue
            pc = str(r.get("phase_class", "")).strip().lower() or phase_class(dphi)
            speed = None
            try:
                sv = str(r.get("speed_value", "")).strip()
                speed = float(sv) if sv else None
            except (TypeError, ValueError):
                speed = None
            rows.append({"relative_phase_rad": dphi, "phase_class": pc,
                         "observed_outcome": outcome, "speed_value": speed})
    if not rows:
        raise ValueError("No usable V5 rows (check include_in_comparison / observed_outcome).")
    return rows


def compare_v5(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ref = internal_v5_reference()
    buckets: dict[str, list[int]] = {}  # phase_class -> [transmitted, total_decisive]
    n_ambiguous = 0
    contains_collapse = False
    for r in rows:
        oc = r["observed_outcome"]
        if oc in V5_AMBIGUOUS:
            n_ambiguous += 1
            continue
        if oc == "COLLAPSE":
            contains_collapse = True
        transmitted = int(oc in V5_TRANSMITTED)
        b = buckets.setdefault(r["phase_class"], [0, 0])
        b[0] += transmitted
        b[1] += 1
    frac = {pc: (t / n if n else None) for pc, (t, n) in buckets.items()}
    anti = frac.get("anti_phase")
    inph = frac.get("in_phase")
    offp = frac.get("off_phase")

    # Direction test: does the experiment concentrate transmission at anti-phase, like C3?
    if anti is None or inph is None:
        direction = "INSUFFICIENT"
        direction_consistent = None
    else:
        others = [x for x in (inph, offp) if x is not None]
        direction_consistent = bool(anti > max(others))
        direction = "DIRECTION_CONSISTENT" if direction_consistent else "DIRECTION_INCONSISTENT"

    # Coverage: what part of the C3 phase x speed grid this dataset can actually corroborate.
    # The distinctive C3 findings are (a) the NARROWNESS of the anti-phase channel (off-phase captures)
    # and (b) the speed dependence. A dataset with only in/anti-phase and no speed axis tests neither.
    has_off_phase = "off_phase" in buckets
    has_speed = any(r.get("speed_value") is not None for r in rows)
    n_distinct_phase = len({round(r["relative_phase_rad"], 2) for r in rows})
    if has_off_phase and has_speed:
        scope = "phase_and_speed"
    elif has_off_phase:
        scope = "phase_including_off_phase_no_speed"
    else:
        scope = "coarse_inphase_vs_antiphase_only"
    untested = []
    if not has_off_phase:
        untested.append("anti-phase channel NARROWNESS (no off-phase points; C3 finds off-phase captures)")
    if not has_speed:
        untested.append("speed dependence (no speed axis; C3 finds anti-phase transmits only below ~0.5c)")

    caveats = ["Qualitative phase-direction comparison only; no numeric, rate, or speed-boundary match."]
    if contains_collapse:
        caveats.append("IN-PHASE side is phenomenological only: COLLAPSE (density-spike destruction) is grouped as "
                       "not-transmitted but its mechanism differs from C3 CAPTURE (binding). The ANTI-PHASE side is "
                       "the stronger correspondence (destructive-interference node protects the solitons in both).")
    if untested:
        caveats.append("Corroborates coarse direction only; does NOT test: " + "; ".join(untested) + ".")
    return {
        "metric": "v5",
        "internal_reference": ref,
        "phase_class_transmission_fraction": frac,
        "phase_class_counts": {pc: {"transmitted": t, "decisive": n} for pc, (t, n) in buckets.items()},
        "n_ambiguous_dropped": n_ambiguous,
        "direction_verdict": direction,
        "direction_consistent_with_c3": direction_consistent,
        "n_rows_used": sum(n for _, n in buckets.values()),
        "coverage": {
            "scope": scope,
            "has_off_phase": has_off_phase,
            "has_speed_axis": has_speed,
            "n_distinct_phase_points": n_distinct_phase,
            "untested_c3_features": untested,
        },
        "match_level": MATCH_LEVEL_CEILING,
        "caveats": caveats,
    }


# --------------------------------------------------------------------------- reporting
def write_report(out_dir: Path, comparison: dict[str, Any], data_path: Path, provenance_path: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison_result.json").write_text(
        json.dumps(
            {
                "status": "PROVISIONAL_UNTIL_CLAUDE_REVIEW",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "data_source": str(data_path),
                "provenance": str(provenance_path),
                "comparison": comparison,
                "guardrails": {
                    "match_level_ceiling": MATCH_LEVEL_CEILING,
                    "promotes_verdict": False,
                    "gates_production": False,
                    "feeds_hunter": False,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if comparison.get("metric") == "v5":
        md = _v5_report_md(comparison, data_path, provenance_path)
    else:
        md = _v1_report_md(comparison, data_path, provenance_path)
    (out_dir / "comparison_report.md").write_text(md, encoding="utf-8")


def _v1_report_md(comparison: dict[str, Any], data_path: Path, provenance_path: Path) -> str:
    sign = comparison.get("phase_sign_accuracy")
    lam_i = comparison.get("internal_lambda")
    by_q = comparison.get("phase_sign_accuracy_by_q") or {}
    if comparison.get("decay_fit_reliable"):
        lam_line = (
            f"- Experimental decay rate lambda: `{comparison.get('experimental_lambda')}` "
            f"(r^2 = `{comparison.get('decay_fit_r_squared')}`); ratio exp/internal = "
            f"`{comparison.get('lambda_ratio_exp_over_internal')}`"
        )
    else:
        lam_line = (
            f"- Experimental decay rate lambda: **NOT REPORTED** — unreliable fit "
            f"({comparison.get('decay_fit_reason')}); raw value "
            f"`{comparison.get('experimental_lambda_raw_unreliable')}` is not a comparable measurement."
        )
    lines = [
        "# Empirical V1 Comparison (Level-3, CLOSE ANALOGUE)",
        "",
        "Status: `PROVISIONAL_UNTIL_CLAUDE_REVIEW`. Context overlay only — no verdict, gate, or",
        "IRER-framing change results from this comparison.",
        "",
        f"- Data source: `{data_path}`",
        f"- Provenance: `{provenance_path}`",
        f"- Phase sign-law accuracy (experiment, all rows): `{sign}`  over `{comparison.get('n_sign_rows')}` rows",
        "- Phase sign-law accuracy by half-separation q:",
    ]
    for q, d in by_q.items():
        lines.append(f"    - q={q}: `{d['accuracy']}`  ({d['hits']}/{d['n']})")
    lines += [
        lam_line,
        f"- Internal C2 lambda: `{lam_i}`  ({comparison.get('internal_lambda_source')})",
        "",
        "Interpretation is deferred to Claude review. Digitized-figure extraction uncertainty",
        "applies (see provenance). Match-level is capped at CLOSE ANALOGUE regardless of fit quality.",
    ]
    return "\n".join(lines) + "\n"


def _v5_report_md(comparison: dict[str, Any], data_path: Path, provenance_path: Path) -> str:
    frac = comparison.get("phase_class_transmission_fraction") or {}
    counts = comparison.get("phase_class_counts") or {}
    ref = comparison.get("internal_reference") or {}
    lines = [
        "# Empirical V5 Comparison (Level-3, CLOSE ANALOGUE) — collision phase×outcome",
        "",
        "Status: `PROVISIONAL_UNTIL_CLAUDE_REVIEW`. Qualitative direction overlay only — no verdict,",
        "gate, numeric match, or IRER-framing change.",
        "",
        f"- Data source: `{data_path}`",
        f"- Provenance: `{provenance_path}`",
        f"- Direction verdict: **`{comparison.get('direction_verdict')}`** "
        f"(consistent with C3 = `{comparison.get('direction_consistent_with_c3')}`)",
        f"- Internal C3 reference: transmission only at anti-phase = "
        f"`{ref.get('internal_transmission_only_at_antiphase')}` ({ref.get('source', 'memory')})",
        "- Experimental transmission (survived-as-two) fraction by phase class:",
    ]
    for pc in ("in_phase", "anti_phase", "off_phase"):
        if pc in frac:
            c = counts.get(pc, {})
            lines.append(f"    - {pc}: `{frac[pc]}`  ({c.get('transmitted')}/{c.get('decisive')} decisive)")
    cov = comparison.get("coverage") or {}
    lines += [
        f"- Ambiguous rows dropped: `{comparison.get('n_ambiguous_dropped')}`; decisive rows used: "
        f"`{comparison.get('n_rows_used')}`",
        f"- Coverage scope: **`{cov.get('scope')}`** (off-phase: `{cov.get('has_off_phase')}`, "
        f"speed axis: `{cov.get('has_speed_axis')}`, distinct phase points: `{cov.get('n_distinct_phase_points')}`)",
    ]
    if cov.get("untested_c3_features"):
        lines.append("- Untested C3 features (NOT corroborated by this dataset):")
        for u in cov["untested_c3_features"]:
            lines.append(f"    - {u}")
    lines += [
        "",
        "Caveats:",
    ]
    for c in comparison.get("caveats", []):
        lines.append(f"- {c}")
    lines += [
        "",
        "The test asks only whether transmission is concentrated at anti-phase (same *direction* as the",
        "C3 phase×speed grid), not whether speeds, boundaries, or rates match. Interpretation deferred to",
        "Claude review; match-level capped at CLOSE ANALOGUE.",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- self test
def self_test_v1() -> bool:
    rng = np.random.default_rng(0)
    lam_true = 2.0
    rows: list[dict[str, float]] = []
    for dphi in (0.0, math.pi / 4, 3 * math.pi / 4, math.pi):
        for q in (1.0, 1.5, 2.0, 2.5):
            measure = -math.exp(-lam_true * q) * math.cos(dphi)
            measure *= 1.0 + 0.02 * float(rng.standard_normal())  # 2% extraction-like noise
            rows.append({"relative_phase": dphi, "q_half_separation": q, "interaction_measure": measure})
    comp = compare_v1(rows)
    ok = True
    if comp["phase_sign_accuracy"] is None or comp["phase_sign_accuracy"] < 0.99:
        print(f"  V1 FAIL: sign accuracy {comp['phase_sign_accuracy']}"); ok = False
    if comp["experimental_lambda"] is None or abs(comp["experimental_lambda"] - lam_true) > 0.1:
        print(f"  V1 FAIL: lambda {comp['experimental_lambda']} vs {lam_true}"); ok = False
    if comp["match_level"] != MATCH_LEVEL_CEILING:
        print("  V1 FAIL: match-level ceiling not enforced"); ok = False
    print("  V1 %s: sign_acc=%.3f lambda=%.4f (true %.1f)"
          % ("PASS" if ok else "FAIL", comp["phase_sign_accuracy"], comp["experimental_lambda"], lam_true))
    return ok


def self_test_v5() -> bool:
    # Synthetic external pattern: in-phase collapses/captures, anti-phase passes through, off-phase captures.
    rows = []
    for dphi, outcome in ((0.0, "COLLAPSE"), (0.1, "CAPTURE"), (math.pi, "PASS_THROUGH"),
                          (math.pi - 0.1, "TRAJECTORY_JUMP"), (math.pi / 2, "CAPTURE"),
                          (3 * math.pi / 4, "CAPTURE"), (math.pi, "NO_COLLAPSE"), (0.0, "MERGE")):
        rows.append({"relative_phase_rad": dphi, "phase_class": phase_class(dphi), "observed_outcome": outcome})
    comp = compare_v5(rows)
    ok = True
    if comp["direction_verdict"] != "DIRECTION_CONSISTENT":
        print(f"  V5 FAIL: direction {comp['direction_verdict']}"); ok = False
    if comp["match_level"] != MATCH_LEVEL_CEILING:
        print("  V5 FAIL: match-level ceiling not enforced"); ok = False
    frac = comp["phase_class_transmission_fraction"]
    if not (frac.get("anti_phase", 0) > frac.get("in_phase", 1)):
        print(f"  V5 FAIL: anti-phase not > in-phase transmission {frac}"); ok = False
    print("  V5 %s: direction=%s frac=%s" % ("PASS" if ok else "FAIL", comp["direction_verdict"], frac))
    return ok


def self_test() -> int:
    print("SELF-TEST (synthetic, no external data):")
    ok = self_test_v1() and self_test_v5()
    print("SELF-TEST %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# --------------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Quarantined Level-3 empirical comparator (opt-in).")
    p.add_argument("--self-test", action="store_true", help="Run the synthetic self-test and exit.")
    p.add_argument("--metric", default="v1", choices=["v1", "v5"], help="Comparison metric.")
    p.add_argument("--data", help="Digitized dataset CSV (see the relevant DATA_SCHEMA).")
    p.add_argument("--provenance", help="Completed provenance record (.md).")
    p.add_argument("--out", default=None, help="Output dir (default: alongside the dataset).")
    args = p.parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.data or not args.provenance:
        p.error("--data and --provenance are required unless --self-test is used.")

    data_path = Path(args.data)
    provenance_path = Path(args.provenance)
    ok, problems = check_provenance(provenance_path)
    if not ok:
        print("REFUSED — provenance incomplete:")
        for pr in problems:
            print(f"  - {pr}")
        return 2

    if args.metric == "v5":
        comparison = compare_v5(load_v5_dataset(data_path))
        summary = (f"  direction={comparison['direction_verdict']} "
                   f"frac={comparison['phase_class_transmission_fraction']} match_level={comparison['match_level']}")
    else:
        comparison = compare_v1(load_v1_dataset(data_path))
        summary = (f"  sign_accuracy={comparison['phase_sign_accuracy']} "
                   f"exp_lambda={comparison['experimental_lambda']} match_level={comparison['match_level']}")
    out_dir = Path(args.out) if args.out else data_path.parent / f"{data_path.stem}_comparison"
    write_report(out_dir, comparison, data_path, provenance_path)
    print(f"empirical {args.metric} comparison: PROVISIONAL_UNTIL_CLAUDE_REVIEW -> {out_dir}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
