#!/usr/bin/env python3
"""
runners/run.py — Main entry point for the SFE instrument.

Accepts any supported file, detects format automatically, routes to the
correct domain analysis, saves outputs, and optionally calls the AI layer.

Usage
-----
    # Strain rosette CSV
    python runners/run.py sampledata.csv --domain strain

    # KW51 bridge modal data (.mat)
    python runners/run.py trackedmodes.mat --domain shm --W 24 \\
        --mat-key trackedmodes.fn --mat-columns 5 8 12 \\
        --mat-labels mode_6 mode_9 mode_13 \\
        --mat-timestamps trackedmodes.time

    # Finance price CSV
    python runners/run.py prices.csv --domain finance --W 20 \\
        --crash-start 2020-02-01 --crash-end 2020-04-30

    # EEG EDF (single subject)
    python runners/run.py S001R04.edf --domain eeg --W 160

    # Any domain + AI interpretation
    python runners/run.py data.csv --domain strain --ai

    # Skip confirmation prompt
    python runners/run.py data.csv --domain strain --auto
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sfe.formats import load
from sfe.outputs import save_run
from sfe.connect import print_summary


# ---------------------------------------------------------------------------
# Domain runners
# ---------------------------------------------------------------------------

def _run_strain(result, args, run_label):
    from sfe.analysis.strain import diurnal_breakdown, print_diurnal, strain_figures

    groups     = result.pair_groups or {"within": [], "cross": result.pairs}
    cross      = groups.get("cross", result.pairs)
    best_cross = max(cross, key=lambda p: p["rho_star"]) if cross else result.pairs[0]

    if hasattr(result, "timestamps") and result.timestamps is not None:
        try:
            breakdown = diurnal_breakdown(result, pair_idx=result.pairs.index(best_cross))
            print_diurnal(breakdown, pair_label=best_cross["label"])
        except Exception as e:
            print(f"  ⚠  Diurnal breakdown skipped: {e}")

    figs       = strain_figures(result, title_prefix=run_label)
    std_keys   = {"phase_portrait", "timeseries", "eigenspectrum"}
    extra_figs = {k: v for k, v in figs.items() if k not in std_keys}

    out_dir = save_run(
        result,
        domain="strain",
        label=run_label,
        figures=list(extra_figs.values()),
        figure_names=list(extra_figs.keys()),
        root=args.out,
    )
    return out_dir


def _run_shm(result, args, run_label):
    from sfe.analysis.shm import (
        slice_phase, detect_structural_transition,
        matlab_datenum_to_datetime,
    )

    # Convert MATLAB timestamps if present
    if hasattr(result, "timestamps") and result.timestamps is not None:
        try:
            result.timestamps = matlab_datenum_to_datetime(result.timestamps)
            print(f"  Timestamps converted: {result.timestamps[0]} → {result.timestamps[-1]}")
        except Exception as e:
            print(f"  ⚠  Timestamp conversion skipped: {e}")

    extra = {}

    # Phase boundary analysis if dates provided
    if args.phase_pre_end and args.phase_post_start:
        try:
            pre  = slice_phase(result, end_dt=args.phase_pre_end)
            post = slice_phase(result, start_dt=args.phase_post_start)
            transition = detect_structural_transition(pre, post)
            print(f"\n{transition}")
            extra["phase_pre_end"]    = args.phase_pre_end
            extra["phase_post_start"] = args.phase_post_start
            extra["transition"]       = transition.label
        except Exception as e:
            print(f"  ⚠  Phase transition analysis skipped: {e}")

    out_dir = save_run(
        result,
        domain="shm",
        label=run_label,
        extra=extra or None,
        root=args.out,
    )
    return out_dir


def _run_finance(result, args, run_label):
    from sfe.analysis.finance import slice_window, detect_regime

    extra = {}

    if args.crash_start and args.crash_end:
        try:
            crash   = slice_window(result, args.crash_start, args.crash_end)
            verdict = detect_regime(result, crash)
            print(f"\n{verdict}")
            extra["crash_window"]    = f"{args.crash_start} → {args.crash_end}"
            extra["regime"]          = verdict.label
            extra["band_gap_full"]   = f"{result.band_gap:.3f}×"
            extra["band_gap_crash"]  = f"{crash.band_gap:.3f}×"
            extra["bandgap_ratio"]   = f"{verdict.bandgap_ratio:.3f}×"
        except Exception as e:
            print(f"  ⚠  Crisis window analysis skipped: {e}")

    out_dir = save_run(
        result,
        domain="finance",
        label=run_label,
        extra=extra or None,
        root=args.out,
    )
    return out_dir


def _run_eeg(result, args, run_label):
    from sfe.analysis.eeg import event_locked_analysis

    extra = {}

    if hasattr(result, "events_df") and result.events_df is not None:
        try:
            ela = event_locked_analysis(result)
            if ela:
                print(f"\n  {ela.verdict}")
                extra["pct_drop"]  = f"{ela.pct_drop:.1f}%"
                extra["lead_s"]    = f"{ela.lead_s:+.2f}s"
                extra["direction"] = ela.direction
        except Exception as e:
            print(f"  ⚠  Event-locked analysis skipped: {e}")
    else:
        print("  No events found — skipping event-locked analysis.")

    out_dir = save_run(
        result,
        domain="eeg",
        label=run_label,
        extra=extra or None,
        root=args.out,
    )
    return out_dir


_DOMAIN_RUNNERS = {
    "strain":  _run_strain,
    "shm":     _run_shm,
    "finance": _run_finance,
    "eeg":     _run_eeg,
}


# ---------------------------------------------------------------------------
# AI layer
# ---------------------------------------------------------------------------

def _maybe_interpret(result, domain, out_dir, args):
    if not args.ai:
        return
    try:
        from sfe.ai import interpret, LLMConfig
        import os

        cfg = LLMConfig(
            api_key  = os.environ.get("SFE_LLM_API_KEY", ""),
            model    = args.ai_model,
            base_url = args.ai_url,
        )
        extra_ctx = None
        if hasattr(args, "crash_start") and args.crash_start:
            extra_ctx = f"Crash window: {args.crash_start} to {args.crash_end}."

        interp = interpret(result, domain=domain, config=cfg,
                           extra_context=extra_ctx, save_to=out_dir)
        print(f"\n{'='*62}")
        print(interp.interpretation)
        print(f"{'='*62}\n")

    except Exception as e:
        print(f"  ⚠  AI interpretation skipped: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="SFE instrument — main runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    p.add_argument("file",         help="Path to data file (.csv, .mat, .edf, .h5)")
    p.add_argument("--domain", "-d",
                   choices=["strain", "shm", "finance", "eeg"],
                   required=True,
                   help="Analysis domain")

    # Common
    p.add_argument("--W",    type=int,  default=None,
                   help="Window size in samples (default: auto per domain)")
    p.add_argument("--out",  type=str,  default="./sfe_runs",
                   help="Output folder root (default: ./sfe_runs)")
    p.add_argument("--auto", action="store_true",
                   help="Skip confirmation prompts (strain domain)")
    p.add_argument("--label", type=str, default=None,
                   help="Run label for output folder name (default: filename stem)")

    # .mat options (shm / any mat file)
    mat = p.add_argument_group(".mat options")
    mat.add_argument("--mat-key",        type=str, default=None,
                     help="Key inside .mat file, e.g. 'trackedmodes.fn'")
    mat.add_argument("--mat-columns",    type=int, nargs="+", default=None,
                     help="Column indices to select, e.g. 5 8 12")
    mat.add_argument("--mat-labels",     type=str, nargs="+", default=None,
                     help="Column labels, e.g. mode_6 mode_9 mode_13")
    mat.add_argument("--mat-timestamps", type=str, default=None,
                     help="Key for timestamps vector, e.g. 'trackedmodes.time'")
    mat.add_argument("--mat-normalize",  action="store_true",
                     help="Z-score normalize columns (recommended for modal data)")

    # SHM phase boundary
    shm = p.add_argument_group("SHM phase boundary options")
    shm.add_argument("--phase-pre-end",    type=str, default=None,
                     help="End of pre-transition phase, e.g. 2019-05-01")
    shm.add_argument("--phase-post-start", type=str, default=None,
                     help="Start of post-transition phase, e.g. 2019-09-01")

    # Finance crisis window
    fin = p.add_argument_group("Finance crisis window options")
    fin.add_argument("--crash-start", type=str, default=None,
                     help="Crisis window start date, e.g. 2020-02-01")
    fin.add_argument("--crash-end",   type=str, default=None,
                     help="Crisis window end date, e.g. 2020-04-30")

    # AI layer
    ai = p.add_argument_group("AI interpretation options")
    ai.add_argument("--ai",       action="store_true",
                    help="Run AI interpretation after instrument (requires SFE_LLM_API_KEY)")
    ai.add_argument("--ai-model", type=str,
                    default="claude-sonnet-4-6",
                    help="Model name (default: claude-sonnet-4-6)")
    ai.add_argument("--ai-url",   type=str,
                    default="https://api.anthropic.com/v1",
                    help="API base URL (default: Anthropic)")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    file_path = args.file
    domain    = args.domain
    run_label = args.label or Path(file_path).stem

    # ------------------------------------------------------------------ W
    # Default W per domain if not specified
    _DEFAULT_W = {"strain": None, "shm": 24, "finance": 20, "eeg": 160}
    W = args.W if args.W is not None else _DEFAULT_W[domain]

    # ------------------------------------------------------------------ load
    # Build kwargs for formats.load from CLI args
    load_kwargs = {}

    if args.auto:
        load_kwargs["auto"] = True

    # .mat options forwarded if present
    if Path(file_path).suffix.lower() == ".mat":
        if args.mat_key:        load_kwargs["key"]            = args.mat_key
        if args.mat_columns:    load_kwargs["columns"]        = args.mat_columns
        if args.mat_labels:     load_kwargs["labels"]         = args.mat_labels
        if args.mat_timestamps: load_kwargs["timestamps_key"] = args.mat_timestamps
        if args.mat_normalize:  load_kwargs["normalize"]      = True

    print(f"\n  Loading: {file_path}  (domain={domain}, W={W})")

    result = load(file_path, W=W, domain=domain, **load_kwargs)

    if result is None:
        # strain connector returns None on user abort
        sys.exit(0)

    print_summary(result)

    # ------------------------------------------------------------------ run
    runner  = _DOMAIN_RUNNERS[domain]
    out_dir = runner(result, args, run_label)

    # ------------------------------------------------------------------ ai
    _maybe_interpret(result, domain, out_dir, args)

    print(f"\n  Run saved to: {out_dir.path}\n")


if __name__ == "__main__":
    main()
