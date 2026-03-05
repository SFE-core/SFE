# -*- coding: utf-8 -*-
"""
sfe/connectors/finance.py — Finance domain connector.

Handles price data → stationary log-returns → SFEResult.
Supports full-period and windowed (crisis) analysis, and regime detection.

Preprocessing contract:
    Raw prices are I(1) — unit root by construction.
    Log-returns are the stationary transformation required by SFE core.
    Applying SFE to raw prices is physically meaningless.

Usage:
    from sfe.connectors.finance import from_yfinance, from_price_csv, slice_window
    from sfe.connectors.finance import detect_regime, REGIME_THRESHOLDS

    result = from_yfinance(["AAPL", "MSFT", "GOOGL", "NVDA"],
                           start="2020-01-01", end="2024-01-01", W=20)

    crash = slice_window(result, "2020-02-01", "2020-04-30")

    verdict = detect_regime(result, crash)
    print(verdict.label)   # "CRISIS COUPLING — Branch A ✓"

Ticker presets:
    TICKERS_COVID  = ["AAPL", "MSFT", "GOOGL", "NVDA"]          (4 large-cap tech)
    TICKERS_2008   = ["AAPL", "MSFT", "GOOGL", "GS", "JPM", "C"]  (mixed tech+fin)
    TICKERS_DOTCOM = ["AAPL", "MSFT", "INTC", "CSCO", "ORCL", "AMZN"]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ..connect import from_dataframe, SFEResult

__all__ = [
    "from_yfinance",
    "from_price_csv",
    "from_price_dataframe",
    "slice_window",
    "detect_regime",
    "RegimeResult",
    "REGIME_THRESHOLDS",
    "TICKERS_COVID",
    "TICKERS_2008",
    "TICKERS_DOTCOM",
]

# ---------------------------------------------------------------------------
# Ticker presets
# ---------------------------------------------------------------------------

TICKERS_COVID  = ["AAPL", "MSFT", "GOOGL", "NVDA"]
TICKERS_2008   = ["AAPL", "MSFT", "GOOGL", "GS", "JPM", "C"]
TICKERS_DOTCOM = ["AAPL", "MSFT", "INTC", "CSCO", "ORCL", "AMZN"]

# ---------------------------------------------------------------------------
# Regime detection thresholds (Proposition 12, SFE-11)
# ---------------------------------------------------------------------------
# Calibrated on COVID crash (Branch A). Tested out-of-sample on 2008 (Branch B).
# DO NOT tune these without re-running the full empirical validation.

REGIME_THRESHOLDS = {
    "rho_delta_min":    0.10,   # minimum Δρ* to qualify as crisis
    "bandgap_mult_min": 1.50,   # Branch A: crash band gap ≥ this × full-period band gap
    "reff_delta_max":  -0.10,   # minimum r_eff collapse to qualify
    "pairs_pct_min":    0.50,   # Branch B: fraction of pairs showing elevation
}


# ---------------------------------------------------------------------------
# Regime result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RegimeResult:
    """
    Returned by detect_regime().

    Attributes
    ----------
    label       : str   human-readable verdict
    branch      : str   "A" | "B" | "none" | "silent"
    fired       : bool
    delta_rho   : float   mean Δρ* across pairs (crash - background)
    delta_reff  : float   Δr_eff corrected (crash - background)
    bandgap_ratio : float   crash band gap / background band gap
    pairs_elevated_pct : float   % pairs showing ρ* elevation
    drho_elevated_pct  : float   % pairs showing dρ elevation
    notes       : list of str    diagnostic messages
    """
    label               : str
    branch              : str
    fired               : bool
    delta_rho           : float
    delta_reff          : float
    bandgap_ratio       : float
    pairs_elevated_pct  : float
    drho_elevated_pct   : float
    notes               : list

    def __str__(self):
        lines = [
            f"Regime: {self.label}",
            f"  Branch        : {self.branch}",
            f"  Δρ* mean      : {self.delta_rho:+.4f}  "
            f"(threshold ≥+{REGIME_THRESHOLDS['rho_delta_min']:.2f})",
            f"  Δr_eff corr   : {self.delta_reff:+.4f}  "
            f"(threshold ≤{REGIME_THRESHOLDS['reff_delta_max']:+.2f})",
            f"  Band gap ratio: {self.bandgap_ratio:.3f}×  "
            f"(Branch A threshold ≥{REGIME_THRESHOLDS['bandgap_mult_min']:.2f}×)",
            f"  Pairs ρ* ↑    : {self.pairs_elevated_pct:.0f}%",
            f"  Pairs dρ ↑    : {self.drho_elevated_pct:.0f}%",
        ]
        for note in self.notes:
            lines.append(f"  Note: {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regime detector — Proposition 12 (SFE-11)
# ---------------------------------------------------------------------------

def detect_regime(
    background: SFEResult,
    crisis: SFEResult,
) -> RegimeResult:
    """
    Classify the crisis window against Proposition 12 Branch A / Branch B.

    Strictly implements the thresholds from SFE-11:

    Branch A (acute homogeneous crisis — e.g. COVID crash):
        (i)  ρ* > background + 0.10
        (ii) dρ > background dρ  (rises, NOT falls — this is a transition event)
        (iii) band gap ≥ 1.50× background band gap
        (iv) r_eff_corr < background − 0.10

    Branch B (acute heterogeneous contagion — e.g. 2008 Lehman):
        (i)  ρ* rises on > 50% of cross-pair combinations
        (ii) dρ rises on > 50% of pairs
        (iii) r_eff_corr < background − 0.10
        Band-gap condition absent — heterogeneous contagion diffuses
        variance across modes rather than concentrating it.

    Silent on gradual corrections (e.g. dot-com 2000-02):
        Neither branch fires when ρ* elevation is negligible and
        r_eff is unchanged.

    Parameters
    ----------
    background : SFEResult   full-period (or pre-crisis) result
    crisis     : SFEResult   crisis window result

    Returns
    -------
    RegimeResult
    """
    thr   = REGIME_THRESHOLDS
    notes = []

    # --- pair-level deltas ---
    bg_pairs     = {p["label"]: p for p in background.pairs}
    cr_pairs     = {p["label"]: p for p in crisis.pairs}
    shared_labels = [l for l in bg_pairs if l in cr_pairs]

    if not shared_labels:
        return RegimeResult(
            label="NO SHARED PAIRS — cannot assess",
            branch="none", fired=False,
            delta_rho=float("nan"), delta_reff=float("nan"),
            bandgap_ratio=float("nan"),
            pairs_elevated_pct=float("nan"),
            drho_elevated_pct=float("nan"),
            notes=["background and crisis results have no overlapping pair labels"],
        )

    delta_rhos  = [cr_pairs[l]["rho_star"]  - bg_pairs[l]["rho_star"]  for l in shared_labels]
    delta_drhos = [cr_pairs[l]["drho_mean"] - bg_pairs[l]["drho_mean"] for l in shared_labels]

    mean_delta_rho  = float(np.mean(delta_rhos))
    pairs_rho_up    = float(np.mean([d > 0 for d in delta_rhos]))   * 100
    pairs_drho_up   = float(np.mean([d > 0 for d in delta_drhos]))  * 100

    # --- global deltas ---
    delta_reff    = crisis.reff_corr  - background.reff_corr
    bandgap_ratio = (crisis.band_gap  / background.band_gap
                     if background.band_gap > 1e-9 else float("nan"))

    # --- Branch A evaluation ---
    branch_a_rho  = mean_delta_rho  >= thr["rho_delta_min"]
    branch_a_bg   = (not np.isnan(bandgap_ratio) and
                     bandgap_ratio  >= thr["bandgap_mult_min"])
    branch_a_reff = delta_reff      <= thr["reff_delta_max"]

    if branch_a_rho and branch_a_bg and branch_a_reff:
        label  = "CRISIS COUPLING — Branch A ✓  (acute homogeneous shock)"
        branch = "A"
        fired  = True
        notes.append(
            f"Single dominant factor loading all observers simultaneously. "
            f"Band gap explosion ({background.band_gap:.2f}× → {crisis.band_gap:.2f}× "
            f"= {bandgap_ratio:.2f}× ratio) is the distinguishing signature."
        )
        if not all(d > 0 for d in delta_drhos):
            pct_drho_up = float(np.mean([d > 0 for d in delta_drhos])) * 100
            notes.append(
                f"dρ rose on {pct_drho_up:.0f}% of pairs (expected: most pairs in Branch A)."
            )
        return RegimeResult(
            label=label, branch=branch, fired=fired,
            delta_rho=mean_delta_rho, delta_reff=delta_reff,
            bandgap_ratio=bandgap_ratio,
            pairs_elevated_pct=pairs_rho_up,
            drho_elevated_pct=pairs_drho_up,
            notes=notes,
        )

    # --- Branch B evaluation ---
    branch_b_rho  = pairs_rho_up  > thr["pairs_pct_min"] * 100
    branch_b_drho = pairs_drho_up > thr["pairs_pct_min"] * 100
    branch_b_reff = delta_reff    <= thr["reff_delta_max"]

    if branch_b_rho and branch_b_drho and branch_b_reff:
        label  = "CRISIS COUPLING — Branch B ✓  (heterogeneous contagion)"
        branch = "B"
        fired  = True
        notes.append(
            f"Contagion propagating across sectors with pre-existing multi-factor structure. "
            f"Band gap stable ({bandgap_ratio:.2f}×) — variance diffused across modes. "
            f"r_eff collapse ({delta_reff:+.3f}) and dρ elevation are the operative signatures."
        )
        if branch_a_rho and not branch_a_bg:
            notes.append(
                "ρ* threshold met but band gap did not explode — "
                "consistent with heterogeneous rather than homogeneous shock."
            )
        return RegimeResult(
            label=label, branch=branch, fired=fired,
            delta_rho=mean_delta_rho, delta_reff=delta_reff,
            bandgap_ratio=bandgap_ratio,
            pairs_elevated_pct=pairs_rho_up,
            drho_elevated_pct=pairs_drho_up,
            notes=notes,
        )

    # --- Silent (gradual correction or background coupling) ---
    if mean_delta_rho < thr["rho_delta_min"]:
        notes.append(
            f"ρ* elevation ({mean_delta_rho:+.4f}) is below threshold "
            f"({thr['rho_delta_min']:+.2f}) — consistent with gradual correction "
            f"or stable background coupling. Not a synchronization event."
        )
    elif not branch_a_bg and not branch_b_reff:
        notes.append(
            "ρ* elevated but neither band gap nor r_eff collapse threshold met."
        )
    elif not branch_b_reff:
        notes.append(
            f"r_eff collapse ({delta_reff:+.4f}) did not reach threshold "
            f"({thr['reff_delta_max']:+.2f})."
        )

    return RegimeResult(
        label="SILENT — no crisis coupling detected",
        branch="silent", fired=False,
        delta_rho=mean_delta_rho, delta_reff=delta_reff,
        bandgap_ratio=bandgap_ratio if not np.isnan(bandgap_ratio) else float("nan"),
        pairs_elevated_pct=pairs_rho_up,
        drho_elevated_pct=pairs_drho_up,
        notes=notes,
    )


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
                # Accept any columns that appear in the CSV, not just hardcoded tickers
                ticker_cols = [c for c in df.columns if c in tickers]
                if not ticker_cols:
                    ticker_cols = list(df.select_dtypes(include="number").columns)
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
    tickers      : list of str   e.g. TICKERS_COVID, TICKERS_2008, TICKERS_DOTCOM,
                                 or any custom list
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
    tickers  : list, opt   subset of columns to use (None = all numeric)
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
    full  = from_yfinance(TICKERS_COVID, start="2019-01-01", end="2024-01-01", W=20)
    crash = slice_window(full, "2020-02-01", "2020-04-30")
    regime = detect_regime(full, crash)
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
