#!/usr/bin/env python3
"""
test_segments_finance.py — Phase segmentation on COVID crash (finance domain).

Real test of segments.py on continuous data.
Three phases: pre-crash, crash, recovery.

Run from project root:
    python test_segments_finance.py

Requires: yfinance
"""

import numpy as np
import matplotlib.pyplot as plt

from sfe.connectors.finance import from_yfinance
from sfe.analysis.segments import segment, compare_portraits, print_segment_summary
from sfe.outputs import save_run

# ── Tickers ────────────────────────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "JPM", "GS", "XOM", "LMT"]

# ── Load full period as log-returns ────────────────────────────────────────
print("Loading data via yfinance...")
full = from_yfinance(TICKERS, "2019-01-01", "2021-06-30", W=20)

# segments.py needs raw data + timestamps
# full.data is already log-returns (T, N), full.timestamps is DatetimeIndex
data       = full.data
timestamps = full.date  

print(f"Data shape: {data.shape}  ({data.shape[0]} trading days)")

# ── Phase boundaries ────────────────────────────────────────────────────────
phases = [
    ("pre-crash",  "2019-01-01", "2020-02-01"),   # background
    ("crash",      "2020-02-01", "2020-04-30"),   # acute event
    ("recovery",   "2020-04-30", "2021-06-30"),   # post-event
]

# ── Segmentation ────────────────────────────────────────────────────────────
print("\nRunning phase segmentation...")
results = segment(
    data       = data,
    timestamps = timestamps,
    phases     = phases,
    W          = 20,
    domain     = "finance",
    labels     = TICKERS,
)

print_segment_summary(results)

# ── Figures ─────────────────────────────────────────────────────────────────
print("Generating phase portrait comparison...")
fig = compare_portraits(
    results,
    title="COVID Crash — Phase Segmentation (AAPL, MSFT, JPM, GS, XOM, LMT)",
)
fig.savefig("covid_phases.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved: covid_phases.png")
plt.close(fig)

# ── Save individual phase runs ───────────────────────────────────────────────
print("\nSaving individual phase runs...")
for label, result in results:
    out_dir = save_run(
        result,
        domain = "finance",
        label  = f"COVID_phase_{label}",
        extra  = {
            "note":   f"COVID phase segmentation — {label}",
            "phase":  label,
            "tickers": ", ".join(TICKERS),
        },
        root = "./sfe_runs",
    )
    print(f"  {label} -> {out_dir.path}")

print("\nDone. Move covid_phases.png to images/ for SFE-Finance paper.")
print("Expected result:")
print("  pre-crash : scattered Active zone, moderate rho*, moderate drho")
print("  crash     : elevated rho*, elevated drho, moving toward Locked")
print("  recovery  : rho* dropping back, drho stabilizing")
