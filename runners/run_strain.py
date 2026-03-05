#!/usr/bin/env python3
"""
run_strain.py — Run SFE on a strain rosette CSV file.

Usage:
    python run_strain.py /path/to/sampledata.csv [--W 60] [--out ./results]

Replicates the Colab strain.py output exactly:
  - Loads the CSV respecting DATA_START header
  - Runs all 36 pairs (9 channels)
  - Prints within/cross device summary
  - Prints eigenspectrum
  - Prints diurnal breakdown for best cross-device pair
  - Saves phase portrait, eigenspectrum, and diurnal figures
"""
import sys
import argparse
from pathlib import Path

# Allow running from project root: python run_strain.py ...
sys.path.insert(0, str(Path(__file__).parent.parent))

from sfe.connectors.strain import (
    from_strain_csv, diurnal_breakdown, print_diurnal, strain_figures
)
from sfe.outputs import save_run


def main():
    parser = argparse.ArgumentParser(description="SFE strain rosette run")
    parser.add_argument("csv", help="Path to strain CSV file (e.g. sampledata.csv)")
    parser.add_argument("--W",   type=int,  default=60,      help="Window size (default: 60)")
    parser.add_argument("--out", type=str,  default="./sfe_runs", help="Output folder root")
    args = parser.parse_args()

    # ------------------------------------------------------------------ load
    result = from_strain_csv(args.csv, W=args.W)

    # --------------------------------------------------------- eigenspectrum
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
    std_keys  = {"phase_portrait", "timeseries", "eigenspectrum"}
    extra_figs = {k: v for k, v in figs.items() if k not in std_keys}
    out_dir = save_run(result, domain="strain", label="strain_rosette",
                       figures=list(extra_figs.values()),
                       figure_names=list(extra_figs.keys()),
                       root=args.out)
    print(f"\n  Run saved to: {out_dir}")


if __name__ == "__main__":
    main()
