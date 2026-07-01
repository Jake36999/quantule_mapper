"""Command-line entry points for the centralized read-only visualization package."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io import IncompleteRunError, guard_outputs
from .plots import render_frame_pack
from .renderers import (
    core_basin,
    core_characterize,
    feb_bound_state,
    phase_c,
    phase_c_current_closure,
    phase_c_current_closure_dynamics,
    phase_c_option_b,
    phase_c_structured,
)


def _add_shared_run_options(parser: argparse.ArgumentParser, *, candidate: bool = False) -> None:
    parser.add_argument("run_dir", nargs="?", help="Run directory to analyze.")
    parser.add_argument("--outdir", help="Output directory. Defaults to the run directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated outputs.")
    parser.add_argument("--latest", action="store_true", help="Resolve the latest supported run automatically.")
    if candidate:
        parser.add_argument("--candidate", help="Optional candidate/config id to highlight.")


def run_phase_c(args: argparse.Namespace) -> int:
    outputs = phase_c.render(
        args.run_dir,
        outdir=args.outdir,
        overwrite=bool(args.overwrite),
        latest=bool(args.latest),
        candidate=args.candidate,
    )
    for output in outputs:
        print(output)
    return 0


def run_feb_bound_state(args: argparse.Namespace) -> int:
    outputs = feb_bound_state.render(
        args.run_dir,
        outdir=args.outdir,
        overwrite=bool(args.overwrite),
        latest=bool(args.latest),
    )
    for output in outputs:
        print(output)
    return 0


def run_core_basin(args: argparse.Namespace) -> int:
    outputs = core_basin.render(
        args.run_dir,
        outdir=args.outdir,
        overwrite=bool(args.overwrite),
        latest=bool(args.latest),
    )
    for output in outputs:
        print(output)
    return 0


def run_core_characterize(args: argparse.Namespace) -> int:
    outputs = core_characterize.render(
        args.run_dir,
        outdir=args.outdir,
        overwrite=bool(args.overwrite),
        latest=bool(args.latest),
        candidate=args.candidate,
    )
    for output in outputs:
        print(output)
    return 0


def run_frames(args: argparse.Namespace) -> int:
    npz_path = Path(args.npz_path)
    outdir = Path(args.outdir) if args.outdir else npz_path.parent
    guard_outputs(outdir, ("frame_density_slices.png", "frame_vector_preview.png"), bool(args.overwrite))
    outputs = render_frame_pack(npz_path, outdir)
    for output in outputs:
        print(output)
    return 0


def run_phase_c_structured(args: argparse.Namespace) -> int:
    outputs = phase_c_structured.render(
        args.summary_csv,
        outdir=args.outdir,
        overwrite=bool(args.overwrite),
        shortlist=args.shortlist,
        cases_root=args.cases_root,
    )
    for output in outputs:
        print(output)
    return 0


def run_phase_c_option_b(args: argparse.Namespace) -> int:
    outputs = phase_c_option_b.render(
        args.summary_csv,
        outdir=args.outdir,
        overwrite=bool(args.overwrite),
        shortlist=args.shortlist,
        cases_root=args.cases_root,
    )
    for output in outputs:
        print(output)
    return 0


def run_phase_c_current_closure(args: argparse.Namespace) -> int:
    outputs = phase_c_current_closure.render(
        args.stage1_root,
        outdir=args.outdir,
        overwrite=bool(args.overwrite),
        manifest=args.manifest,
    )
    for output in outputs:
        print(output)
    return 0


def run_phase_c_current_closure_dynamics(args: argparse.Namespace) -> int:
    outputs = phase_c_current_closure_dynamics.render(
        args.trace_root,
        outdir=args.outdir,
        overwrite=bool(args.overwrite),
    )
    for output in outputs:
        print(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantule_viz",
        description="Render standard figures from saved Quantule Mapper artifacts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_phase = sub.add_parser("phase-c", help="Render Phase C saved-result figures.")
    _add_shared_run_options(p_phase, candidate=True)
    p_phase.set_defaults(func=run_phase_c)

    p_feb = sub.add_parser("feb-bound-state", help="Render feb56dc7 bound-state figures.")
    _add_shared_run_options(p_feb)
    p_feb.set_defaults(func=run_feb_bound_state)

    p_basin = sub.add_parser("core-basin", help="Render core basin or refine run figures.")
    _add_shared_run_options(p_basin)
    p_basin.set_defaults(func=run_core_basin)

    p_characterize = sub.add_parser("core-characterize", help="Render long-time core characterization figures.")
    _add_shared_run_options(p_characterize, candidate=True)
    p_characterize.set_defaults(func=run_core_characterize)

    p_frames = sub.add_parser("frames", help="Render density/vector previews from an NPZ frame bundle.")
    p_frames.add_argument("npz_path", help="Path to an NPZ containing psi, frames, or a 3D/4D field array.")
    p_frames.add_argument("--outdir", help="Output directory. Defaults to the NPZ parent.")
    p_frames.add_argument("--overwrite", action="store_true", help="Accepted for CLI consistency.")
    p_frames.set_defaults(func=run_frames)

    p_structured = sub.add_parser(
        "phase-c-structured",
        help="Render the structured Phase C discovery visual-analysis pack.",
    )
    p_structured.add_argument(
        "summary_csv",
        nargs="?",
        help="Structured-discovery summary CSV. Defaults to docs/phase_c_structured_discovery_B_summary.csv.",
    )
    p_structured.add_argument(
        "--shortlist",
        help="Shortlist diagnostic metrics JSON. Defaults to runtime_logs/phase_c_structured_discovery_B_shortlist_metrics.json.",
    )
    p_structured.add_argument(
        "--cases-root",
        help="Directory containing per-case diagnostic bundles. Defaults to <outdir>/cases.",
    )
    p_structured.add_argument("--outdir", required=True, help="Output root for the visual-analysis pack.")
    p_structured.add_argument("--overwrite", action="store_true", help="Replace existing generated outputs.")
    p_structured.set_defaults(func=run_phase_c_structured)

    p_optb = sub.add_parser("phase-c-option-b", help="Render the Phase C Option B v2 morphology/comparison/inspection pack.")
    p_optb.add_argument("summary_csv", nargs="?", help="Structured-discovery summary CSV. Defaults to docs/phase_c_structured_discovery_B_summary.csv.")
    p_optb.add_argument("--shortlist", help="Shortlist diagnostic metrics JSON.")
    p_optb.add_argument("--cases-root", help="Directory with per-case diagnostic bundles (frames.npz + diagnostic_summary.json).")
    p_optb.add_argument("--outdir", required=True, help="Output root for the v2 visual-analysis pack.")
    p_optb.add_argument("--overwrite", action="store_true", help="Replace existing generated outputs.")
    p_optb.set_defaults(func=run_phase_c_option_b)

    p_close = sub.add_parser("phase-c-current-closure", help="Render the N96 current-closure / signed-vorticity static analysis.")
    p_close.add_argument("stage1_root", help="Stage 1 root with <case>/probe_data.npz (PHASE_C_OPTION_B_N96_STAGE1_*).")
    p_close.add_argument("--manifest", help="Stage 1 manifest JSON for class labels. Defaults to <stage1_root>/stage1_manifest.json.")
    p_close.add_argument("--outdir", required=True, help="Output root for the current-closure pack.")
    p_close.add_argument("--overwrite", action="store_true", help="Replace existing generated outputs.")
    p_close.set_defaults(func=run_phase_c_current_closure)

    p_dyn = sub.add_parser("phase-c-current-closure-dynamics", help="Render time-resolved current-closure from N96 trace bundles.")
    p_dyn.add_argument("trace_root", help="Trace root with <case>/frames.npz + diagnostic_summary.json (PHASE_C_OPTION_B_N96_TRACE_*).")
    p_dyn.add_argument("--outdir", required=True, help="Output root for the dynamics pack.")
    p_dyn.add_argument("--overwrite", action="store_true", help="Replace existing generated outputs.")
    p_dyn.set_defaults(func=run_phase_c_current_closure_dynamics)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileExistsError, IncompleteRunError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
