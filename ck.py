from sfe.connectors.finance import from_yfinance
from sfe.outputs import save_run

tickers = ["CL=F", "XOM", "CVX", "LMT", "GC=F", "USO"]

oil = from_yfinance(tickers, "2025-09-01", "2026-03-07", W=5)

out_dir = save_run(
    oil,
    domain="finance",
    label="crude_oil_geopolitical",
    extra={
        "note": "Crude oil geopolitical snapshot — Polymarket $110 by March 31 at 48%",
        "VIX":  "29.69 +25.01% (March 7 close)",
    },
    root="./sfe_runs",
)
print(f"Saved to: {out_dir.path}")