#!/usr/bin/env python3
"""IRER archive Stage 0 triage.

Enumerates the May-Aug 2025 per-chat transcript corpus, scores every
conversation with three term-group counters, cross-checks completeness
against the bank-1 merged monthlies, resolves v9's named-source citations
by quote-fragment search, and emits the coverage register + work queue.

Outputs (UTF-8):
  <archive>/00_CORPUS_COVERAGE_REGISTER.md
  <archive>/00_V9_CITATION_RESOLUTION.md
  <archive>/_work_queue.md
  <archive>/_triage.json          (machine-readable snapshot)
"""
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

TRANSCRIPTS = Path(r"F:\transcripts")
BANK = Path(r"D:\memory_bank\bank 1")
ARCHIVE = Path(r"F:\quantule_mapper\docs\theory_synthesis\irer_archive")

IN_SCOPE_MONTHS = ["2025_05_May", "2025_06_June", "2025_07_July", "2025_08_August"]
BANK_MONTHS = {
    "2025_05_May": "2025_May.txt",
    "2025_06_June": "2025_June.txt",
    "2025_07_July": "2025_July.txt",
    "2025_08_August": "2025_August.txt",
}
# citation search sweeps every per-chat month folder (incl. Jan-Apr), read-only
ALL_MONTH_RE = re.compile(r"^2025_\d{2}_")

# ---------------------------------------------------------------- term groups
STRONG = [
    r"\bIRER\b", r"\bquantules?\b", r"\bpayan\b", r"\bOIWs?\b",
    r"informational\s+indifference", r"potentiality\s+indifference",
    r"resolution\s+field", r"resonance\s+density", r"angular\s+deficits?",
    r"chronology\s+of\s+resolution", r"informational\s+par+al+el+s?",
    r"axis\s+of\s+least", r"path\s+of\s+least", r"angles?\s+of\s+the\s+manifold",
    r"\ba-?temporal\b", r"primordial\s+informational", r"informational\s+manifold",
    r"chorotic", r"gradient-?derived", r"prime[- ]harmonic",
    r"informational\s+resonance", r"ontological\s+informational",
    r"emergent\s+spacetime", r"c_?emergent", r"\bSPTT\b", r"chiral\s+pairs?",
    r"informational\s+collapse", r"manifoldic", r"informational\s+entropy",
    r"quantized\s+resonant", r"informational\s+event\s+horizon",
]
STRONG_CS = [r"\bPAS\b", r"\bPIF\b", r"\bFMIA\b", r"\bIQG\b", r"\bRFD\b", r"\bRD\b"]
PHYSICS = [
    r"phase[- ]field", r"free[- ]energy", r"soliton", r"entropy", r"gradient",
    r"resonance", r"manifold", r"collapse", r"\bspin\b", r"wavefunction",
    r"interference", r"decoherence", r"quantum", r"simulation", r"\bFFT\b",
    r"lagrangian", r"geodesic", r"eigen", r"harmonic",
]
ALETHEIA = [
    r"\baletheia\b", r"declaration\s+of\s+understanding", r"sentien\w*",
    r"conscious\w*", r"\bLLM\b", r"sycophan\w*", r"identity\s+matri(x|ces)",
    r"emergent\s+identity", r"\bASTE\b", r"\bSIE\b",
]

STRONG_RE = [re.compile(p, re.IGNORECASE) for p in STRONG]
STRONG_CS_RE = [re.compile(p) for p in STRONG_CS]
PHYSICS_RE = [re.compile(p, re.IGNORECASE) for p in PHYSICS]
ALETHEIA_RE = [re.compile(p, re.IGNORECASE) for p in ALETHEIA]

# ------------------------------------------------- v9 citation quote fragments
# Each: (v9 source name it evidences, label, regex). Spelling variants folded in.
FRAGMENTS = [
    ("sctriptpt2.txt/transcript.txt", "OU 'formalise this' intent",
     re.compile(r"whole\s+reason\s+i\s+want\s+to\s+study\s+physics", re.I)),
    ("sctriptpt2.txt/transcript.txt", "'potentially exists by chance'",
     re.compile(r"potentially\s+exists?\s+by\s+chance", re.I)),
    ("sctriptpt2.txt/transcript.txt", "'distribution curves of the intersecting waves'",
     re.compile(r"distribution\s+curves?\s+of\s+the\s+intersecting\s+waves?", re.I)),
    ("transcript.txt (phase-field discussion)", "'possibly also described better as'",
     re.compile(r"is\s+possibly\s+also\s+des?c?ribed\s+better\s+as", re.I)),
    ("transcript.txt (phase-field discussion)", "'coupled rotational information indifferences'",
     re.compile(r"coupled\s+rotational\s+information\s+indifferences?", re.I)),
    ("scriptpt4.txt (observer formalization)", "'informationally coherent resonance structure'",
     re.compile(r"informationally\s+coherent\s+resonance\s+structure", re.I)),
    ("transcript.txt (phase-field discussion)", "'forces of an IRER field'",
     re.compile(r"forces?\s+of\s+an\s+irer\s+field", re.I)),
    ("scriptpt3.txt (prime harmonics)", "'prime numbers, due to their indivisibility'",
     re.compile(r"prime\s+numbers?,?\s+due\s+to\s+their\s+indivisibility", re.I)),
    ("scriptpt3.txt (prime harmonics)", "prime-indexed frequency",
     re.compile(r"prime[- ]indexed", re.I)),
    ("scriptpt5.txt (analysis code)", "spatial_fft_analysis",
     re.compile(r"spatial_fft_analysis")),
    ("scriptpt5.txt (simulator)", "Rho1DSimulator",
     re.compile(r"Rho1DSimulator")),
    ("scriptpt5.txt (simulator)", "irer_simulator_fixed",
     re.compile(r"irer_simulator_fixed", re.I)),
    ("sctriptpt2.txt/transcript.txt", "'path of least activity'",
     re.compile(r"path\s+of\s+least\s+activity", re.I)),
    ("splash test.docx (Appendix D)", "'splash test'",
     re.compile(r"splash[\s_-]test", re.I)),
]


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def sha256_short(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def count_hits(text: str, regexes) -> int:
    return sum(len(r.findall(text)) for r in regexes)


def header_meta(text: str):
    title, updated = "", ""
    for line in text.splitlines()[:6]:
        if line.startswith("Title:"):
            title = line[len("Title:"):].strip()
        elif line.startswith("Last Updated:"):
            updated = line[len("Last Updated:"):].strip()
    return title, updated


def disposition(strong: int, physics: int, aletheia: int) -> str:
    if strong >= 8:
        return "EXTRACT"
    if strong >= 1 or aletheia >= 15 or physics >= 40:
        return "REVIEW"
    return "EXCLUDE_CANDIDATE"


def norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def main() -> int:
    today = date.today().isoformat()
    rows = []
    for month in IN_SCOPE_MONTHS:
        mdir = TRANSCRIPTS / month
        if not mdir.is_dir():
            print(f"WARNING: missing month folder {mdir}", file=sys.stderr)
            continue
        for p in sorted(mdir.glob("*.txt")):
            text = read_text(p)
            title, updated = header_meta(text)
            strong = count_hits(text, STRONG_RE) + count_hits(text, STRONG_CS_RE)
            physics = count_hits(text, PHYSICS_RE)
            aletheia = count_hits(text, ALETHEIA_RE)
            rows.append({
                "month": month,
                "file": p.name,
                "path": str(p),
                "title": title or p.stem,
                "updated": updated,
                "lines": text.count("\n") + 1,
                "kb": round(p.stat().st_size / 1024, 1),
                "sha256_16": sha256_short(p),
                "strong": strong,
                "physics": physics,
                "aletheia": aletheia,
                "disposition": disposition(strong, physics, aletheia),
                "status": "todo",
                "dossier": "",
            })

    # ------------------------------------------------ bank cross-check (May-Aug)
    bank_report = []
    perchat_titles = {m: set() for m in IN_SCOPE_MONTHS}
    for r in rows:
        perchat_titles[r["month"]].add(norm_title(r["title"]))
    for month, bank_name in BANK_MONTHS.items():
        bp = BANK / bank_name
        if not bp.is_file():
            bank_report.append((month, bank_name, "MISSING", []))
            continue
        btitles = [norm_title(m.group(1)) for m in
                   re.finditer(r"^TITLE:\s*(.+)$", read_text(bp), re.M)]
        missing = sorted(set(btitles) - perchat_titles[month])
        extra = sorted(perchat_titles[month] - set(btitles))
        bank_report.append((month, bank_name,
                            f"bank_blocks={len(btitles)} per_chat={len(perchat_titles[month])} "
                            f"bank_only={len(missing)} perchat_only={len(extra)}",
                            missing))

    # ------------------------------------------------ v9 citation resolution
    all_month_dirs = sorted(d for d in TRANSCRIPTS.iterdir()
                            if d.is_dir() and ALL_MONTH_RE.match(d.name))
    citation_hits = {i: [] for i in range(len(FRAGMENTS))}
    for d in all_month_dirs:
        for p in sorted(d.glob("*.txt")):
            text = read_text(p)
            lines = text.splitlines()
            for i, (_, _, rx) in enumerate(FRAGMENTS):
                if not rx.search(text):
                    continue
                locs = [n + 1 for n, ln in enumerate(lines) if rx.search(ln)]
                citation_hits[i].append((f"{d.name}/{p.name}", locs[:6]))

    # ------------------------------------------------------------- write files
    ARCHIVE.mkdir(parents=True, exist_ok=True)

    reg = ARCHIVE / "00_CORPUS_COVERAGE_REGISTER.md"
    lines_out = [
        "# Corpus Coverage Register — Round 1 (May–Aug 2025)",
        "",
        f"Generated by `tools/triage.py` on {today}. Corpus: `F:\\transcripts\\2025_05_May … 2025_08_August`.",
        "Every in-scope conversation appears here. `Status` is updated as extraction proceeds:",
        "`todo` → `dossier` (path filled in) | `excluded` (reason replaces dossier path) | `skimmed-excluded`.",
        "",
        "Term groups — strong: coined IRER vocabulary; physics: generic physics context; aletheia: AI-identity stream.",
        "Dispositions — EXTRACT (strong ≥ 8) · REVIEW (any strong hit, or heavy aletheia/physics) · EXCLUDE_CANDIDATE (no signal; still listed).",
        "",
    ]
    total = {"EXTRACT": 0, "REVIEW": 0, "EXCLUDE_CANDIDATE": 0}
    for month in IN_SCOPE_MONTHS:
        mrows = [r for r in rows if r["month"] == month]
        if not mrows:
            continue
        lines_out += [f"## {month} ({len(mrows)} conversations)", "",
                      "| File | Title | Date | Lines | KB | sha256/16 | strong | phys | aleth | Disposition | Status | Dossier/Reason |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(mrows, key=lambda x: (-x["strong"], x["file"])):
            total[r["disposition"]] += 1
            lines_out.append(
                f"| {r['file']} | {r['title']} | {r['updated'][:10]} | {r['lines']} | {r['kb']} "
                f"| `{r['sha256_16']}` | {r['strong']} | {r['physics']} | {r['aletheia']} "
                f"| {r['disposition']} | {r['status']} | {r['dossier']} |")
        lines_out.append("")
    lines_out += [
        "## Totals",
        "",
        f"- Conversations in scope: **{len(rows)}**",
        f"- EXTRACT: **{total['EXTRACT']}** · REVIEW: **{total['REVIEW']}** · EXCLUDE_CANDIDATE: **{total['EXCLUDE_CANDIDATE']}**",
        "",
        "## Bank-monthly completeness cross-check",
        "",
        "Per-chat archive is authoritative for May–Aug; the bank 1 merged monthlies are checked for conversations missing from it.",
        "",
    ]
    for month, bank_name, verdict, missing in bank_report:
        lines_out.append(f"- **{month}** vs `{bank_name}`: {verdict}")
        for t in missing:
            lines_out.append(f"  - bank-only title (recover from bank monthly): `{t}`")
    lines_out += ["", "## Out of scope this round (Round 2)", "",
                  "- `D:\\memory_bank\\bank 1\\2025_September.txt … 2026_January.txt` (application/iterative-design era)",
                  "- `Aleheia'sChat.txt` (compiled bridge record)",
                  "- `F:\\transcripts\\2025_01_January … 2025_04_April` (pre-May sweep; citation search below already covers quote lookups)",
                  "- `Category_A–D` compilations (dedup candidates)", ""]
    reg.write_text("\n".join(lines_out), encoding="utf-8")

    # citation resolution
    cit = ARCHIVE / "00_V9_CITATION_RESOLUTION.md"
    c = ["# v9 Citation Resolution — named sources → actual conversations",
         "",
         f"Generated by `tools/triage.py` on {today}. Jake confirmed the files v9 cites",
         "(`transcript.txt`, `sctriptpt2.txt`, `scriptpt3–5.txt`) are the same conversations found in",
         "`F:\\transcripts` under different names (copy-pasted from the website rather than data-exported).",
         "This table locates each v9-quoted excerpt in the surviving per-chat archive (all 2025 month folders searched, Jan–Aug).",
         "",
         "| v9 source name | Quoted fragment | Found in (file : lines) |",
         "|---|---|---|"]
    for i, (src, label, rx) in enumerate(FRAGMENTS):
        hits = citation_hits[i]
        if hits:
            loc = "<br>".join(f"`{f}` : {', '.join(map(str, ls))}" for f, ls in hits[:8])
            if len(hits) > 8:
                loc += f"<br>… +{len(hits) - 8} more files"
        else:
            loc = "**UNRESOLVED** (no match in per-chat archive)"
        c.append(f"| {src} | {label} | {loc} |")
    c += ["",
          "Reading: a fragment matching exactly one early file pins that v9 source name to that conversation.",
          "Fragments matching many files are recap-propagated phrases; the earliest dated hit is the origin candidate.",
          ""]
    cit.write_text("\n".join(c), encoding="utf-8")

    # work queue
    wq = ARCHIVE / "_work_queue.md"
    q = ["# Extraction work queue — Round 1",
         "",
         "Ordered by strong-term density within month (May first). Agents: claim a block, write the dossier,",
         "flip Status in 00_CORPUS_COVERAGE_REGISTER.md, tick here.", ""]
    for month in IN_SCOPE_MONTHS:
        q.append(f"## {month}")
        q.append("")
        mrows = [r for r in rows if r["month"] == month and r["disposition"] != "EXCLUDE_CANDIDATE"]
        for r in sorted(mrows, key=lambda x: -x["strong"]):
            q.append(f"- [ ] ({r['disposition']}, strong={r['strong']}, {r['kb']}KB) `{r['file']}`")
        q.append("")
    wq.write_text("\n".join(q), encoding="utf-8")

    (ARCHIVE / "_triage.json").write_text(
        json.dumps({"generated": today, "rows": rows}, indent=1), encoding="utf-8")

    print(f"register: {reg}")
    print(f"citations: {cit}")
    print(f"queue: {wq}")
    print(f"conversations: {len(rows)}  dispositions: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
