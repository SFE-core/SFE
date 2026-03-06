from sfe.connectors.finance import from_yfinance
from sfe.outputs import save_run

tickers = ["AAPL", "MSFT", "AMZN", "CSCO", "INTC", "ORCL"]

dotcom = from_yfinance(tickers, "2000-01-01", "2002-12-31", W=20)

out_dir = save_run(
    dotcom,
    domain="finance",
    label="dotcom_calibration",
    extra={"note": "Dotcom crash calibration — gradual correction baseline"},
    root="./sfe_runs",
)
print(f"Saved to: {out_dir.path}")