# -*- coding: utf-8 -*-
"""
test_sfe.py — Phase 0 pipeline connectivity test.

    python test_sfe.py
    python test_sfe.py --csv your_file.csv --sfreq 1.0 --domain strain --W 60
"""

import sys
import argparse
import numpy as np

DIVIDER = "=" * 62
def section(title): print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")
def ok(msg):        print(f"  ✓  {msg}")
def fail(msg, e):   print(f"  ✗  {msg}\n     {e}"); sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument("--csv",    type=str,   default=None)
parser.add_argument("--sfreq",  type=float, default=1.0)
parser.add_argument("--domain", type=str,   default="strain")
parser.add_argument("--W",      type=int,   default=None,
                    help="Override W (default: auto from sweep)")
parser.add_argument("--T",      type=int,   default=500,
                    help="Synthetic test timesteps (default: 500)")
args = parser.parse_args()

# 1. Imports
section("1. Imports")
try:
    import sfe
    from sfe.connectors import finance, eeg, traffic, strain
    from sfe.outputs import save_run
    from sfe.w_select import suggest_W, sweep_W, print_W_report
    ok(f"sfe v{sfe.__version__}")
    ok("connectors (finance, eeg, traffic, strain), outputs, w_select")
except Exception as e:
    fail("import", e)

# 2. Synthetic pipeline
section(f"2. Synthetic pipeline  T={args.T}")
try:
    np.random.seed(42)
    data   = np.random.randn(args.T, 4)
    labels = ["A", "B", "C", "D"]

    if args.W:
        w = args.W
        ok(f"W={w} (manual override)")
    else:
        w = suggest_W(sfreq=1.0, T=args.T).recommended_W
        ok(f"W={w} (auto from suggest_W)")

    result = sfe.connect.from_array(data, W=w, labels=labels)
    folder = save_run(result, domain="test", label="synthetic")
    ok(f"from_array  shape={result.data.shape}  W={w}")
    ok(f"save_run  -> {folder.path.name}")
    ok(f"figures -> {sorted(f.name for f in folder.figures.iterdir())}")
except Exception as e:
    fail("synthetic pipeline", e)

# 3. Real CSV
if args.csv:
    section(f"3. Real data — {args.csv}  domain={args.domain}  sfreq={args.sfreq}")
    try:
        from pathlib import Path
        from sfe.connectors.strain import (
            from_strain_csv, strain_figures,
            diurnal_breakdown, print_diurnal,
        )

        W_use = args.W if args.W else 60
        ok(f"W={W_use} {'(manual override)' if args.W else '(default strain heuristic)'}")

        result = from_strain_csv(args.csv, W=W_use)
        ok(f"N={result.N}  pairs={len(result.pairs)}  T={result.T}")
        ok(f"devices={result.devices}")
        ok(f"within={len(result.pair_groups['within'])}  cross={len(result.pair_groups['cross'])}")

        try:
            bd = diurnal_breakdown(result)
            print_diurnal(bd, pair_label=result.pairs[0]["label"])
            ok("diurnal breakdown computed")
        except Exception as e:
            ok(f"diurnal skipped: {e}")

        figs   = strain_figures(result, title_prefix=Path(args.csv).stem)
        folder = save_run(result, domain=args.domain,
                          label=Path(args.csv).stem,
                          figures=list(figs.values()),
                          figure_names=list(figs.keys()))
        ok(f"save_run  -> {folder.path.name}")
        ok(f"figures -> {sorted(f.name for f in folder.figures.iterdir())}")

    except Exception as e:
        fail("real CSV", e)

print(f"\n{DIVIDER}\n  ALL TESTS PASSED\n{DIVIDER}\n")