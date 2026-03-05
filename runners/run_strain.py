#!/usr/bin/env python3
"""
run_strain.py — Run SFE on a strain rosette CSV file.

Usage:
    python run_strain.py /path/to/sampledata.csv [--W 60] [--out ./results] [--auto]

    --W     Window size in samples. If omitted, auto-detected from sfreq.
    --out   Output folder root (default: ./sfe_runs).
    --auto  Skip confirmation prompt and run on detected/default settings.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sfe.connectors.strain import (
    from_strain_csv, diurnal_breakdown, print_diurnal, strain_figures
)
from sfe.outputs import save_run


def main():
    parser = argparse.ArgumentParser(description="SFE strain rosette run")
    parser.add_argument("csv", help="Path to strain CSV file (e.g. sampledata.csv)")
    parser.add_argument("--W",    type=int,  default=None,        help="Window size in samples (default: auto-detected from sfreq)")
    parser.add_argument("--out",  type=str,  default="./sfe_runs", help="Output folder root")
    parser.add_argument("--auto", action="store_true",             help="Skip confirmation prompt")
    args = parser.parse_args()

    # ------------------------------------------------------------------ load
    result = from_strain_csv(args.csv, W=args.W, auto=args.auto)

    if result is None:
        # User aborted at confirmation prompt
        sys.exit(0)

    # --------------------------------------------------------- pair groups
    groups = result.pair_groups
    within = groups["within"]
    cross  = groups["cross"]
    best_cross = max(cross, key=lambda p: p["rho_star"]) if cross else result.pairs[0]

    # ------------------------------------------------------------ diurnal
    if hasattr(result, "timestamps") and result.timestamps is not None:
        breakdown = diurnal_breakdown(result, pair_idx=result.pairs.index(best_cross))
        print_diurnal(breakdown, pair_label=best_cross["label"])

    # --------------------------------------------------------------- figures
    figs = strain_figures(result, title_prefix="Strain Rosette")

    # --------------------------------------------------------------- save
    std_keys   = {"phase_portrait", "timeseries", "eigenspectrum"}
    extra_figs = {k: v for k, v in figs.items() if k not in std_keys}
    out_dir = save_run(result, domain="strain", label="strain_rosette",
                       figures=list(extra_figs.values()),
                       figure_names=list(extra_figs.keys()),
                       root=args.out)
    print(f"\n  Run saved to: {out_dir}")


if __name__ == "__main__":
    main()
