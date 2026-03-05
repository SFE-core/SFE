# -*- coding: utf-8 -*-
"""
sfe/connectors/finance.py — Finance domain connector.

Handles price data → stationary log-returns → SFEResult.
Supports full-period and windowed (crisis) analysis.

Preprocessing contract:
    Raw prices are I(1) — unit root by construction.
    Log-returns are the stationary transformation required by SFE core.
    Applying SFE to raw prices is physically meaningless.

Usage:
    from sfe.connectors.finance import from_yfinance, from_price_csv, slice_window

    result = from_yfinance(["AAPL", "MSFT", "GOOGL", "NVDA"],
                           start="2020-01-01", end="2024-01-01", W=20)

    crash = slice_window(result, "2020-02-01", "2020-04-30")
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from ..connect import from_dataframe, SFEResult

__all__ = [
    "from_yfinance",
    "from_price_csv",
    "from_price_dataframe",
    "slice_window",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_log_returns(prices):
    """
    Convert price DataFrame to log-returns and drop the first NaN row.
    Returns (returns_df, dates_index).
    """
    import pandas as _pd  # noqa: F401
    ret = np.log(prices / prices.shift(1)).dropna()
    return ret, ret.index


def _load_yfinance(tickers, start, end, csv_fallback=None):
    """Download via yfinance, optionally fall back to a local CSV."""
    import pandas as _pd

    if csv_fallback:
        for p in ([csv_fallback] if isinstance(csv_fallback, str) else csv_fallback):
            p = Path(p)
            if p.exists():
                df = _pd.read_csv(p, index_col=0, parse_dates=True)
                if isinstance(df.columns, _pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                ticker_cols = [c for c in df.columns if c in tickers]
                if ticker_cols:
                    df = df[ticker_cols]
                return df.dropna()

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance required. pip install yfinance")

    df = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False,
    )["Close"]

    if hasattr(df.columns, "get_level_values"):
        df.columns = df.columns.get_level_values(0)

    return df.dropna()


# ---------------------------------------------------------------------------
# Public connectors
# ---------------------------------------------------------------------------

def from_yfinance(
    tickers: list[str],
    start: str,
    end: str,
    W: int = 20,
    csv_fallback=None,
) -> SFEResult:
    """
    Download prices via yfinance, convert to log-returns, run SFE.

    Parameters
    ----------
    tickers      : list of str   e.g. ["AAPL", "MSFT", "GOOGL", "NVDA"]
    start        : str           "YYYY-MM-DD"
    end          : str           "YYYY-MM-DD"
    W            : int           rolling window in trading days (default 20 ≈ 1 month)
    csv_fallback : str or list   local CSV path(s) to try before downloading

    Returns
    -------
    SFEResult with .dates attribute set to the returns index.
    """
    prices = _load_yfinance(tickers, start, end, csv_fallback)
    return from_price_dataframe(prices, W=W)


def from_price_csv(
    path: str,
    W: int = 20,
    tickers: list[str] | None = None,
    delimiter: str = ",",
) -> SFEResult:
    """
    Load a price CSV (date index, ticker columns) and run SFE on log-returns.

    Parameters
    ----------
    path     : str         path to CSV with date index and price columns
    W        : int         rolling window in trading days
    tickers  : list, opt   subset of columns to use
    delimiter: str         default ','
    """
    import pandas as _pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = _pd.read_csv(path, index_col=0, parse_dates=True, sep=delimiter)

    if isinstance(df.columns, _pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if tickers is not None:
        available = [c for c in tickers if c in df.columns]
        if not available:
            raise ValueError(f"None of {tickers} found in columns {list(df.columns)}")
        df = df[available]

    df = df.select_dtypes(include="number").dropna()
    return from_price_dataframe(df, W=W)


def from_price_dataframe(prices, W: int = 20) -> SFEResult:
    """
    Run SFE on a price DataFrame (any index). Converts to log-returns internally.

    Parameters
    ----------
    prices : DataFrame   shape (T, N), numeric price columns
    W      : int         rolling window in trading days
    """
    import pandas as _pd  # noqa: F401

    prices = prices.select_dtypes(include="number").dropna()
    if prices.shape[1] < 2:
        raise ValueError(f"Need ≥2 price columns, got {prices.shape[1]}.")

    returns, dates = _to_log_returns(prices)

    result = from_dataframe(returns, W=W)

    # Attach the date index so callers can use slice_window
    result.dates = dates
    return result


# ---------------------------------------------------------------------------
# Window slicer — crisis / regime analysis
# ---------------------------------------------------------------------------

def slice_window(
    result: SFEResult,
    start: str,
    end: str,
    W: int | None = None,
) -> SFEResult:
    """
    Re-run SFE on a date-bounded window of a finance SFEResult.

    The result must have been produced by a finance connector
    (i.e. it has a .dates attribute).

    Parameters
    ----------
    result : SFEResult   original full-period result
    start  : str         "YYYY-MM-DD"  window start (inclusive)
    end    : str         "YYYY-MM-DD"  window end   (inclusive)
    W      : int, opt    override window size (default: same as original)

    Returns
    -------
    SFEResult for the sliced window, with .dates set.

    Example
    -------
    full  = from_yfinance(["AAPL","MSFT","GOOGL","NVDA"],
                          start="2019-01-01", end="2024-01-01", W=20)
    crash = slice_window(full, "2020-02-01", "2020-04-30")
    """
    import pandas as _pd

    if not hasattr(result, "dates"):
        raise AttributeError(
            "result.dates not found. Use a finance connector "
            "(from_yfinance / from_price_csv / from_price_dataframe)."
        )

    mask  = (result.dates >= start) & (result.dates <= end)
    dates = result.dates[mask]

    if mask.sum() == 0:
        raise ValueError(f"No data in window {start} → {end}.")

    W_use = W if W is not None else result.W

    # result.data is already log-returns (cleaned)
    data_window = result.data[mask.values]

    df_window = _pd.DataFrame(
        data_window,
        columns=result.labels,
        index=dates,
    )

    sub = from_dataframe(df_window, W=W_use)
    sub.dates = dates
    return sub
