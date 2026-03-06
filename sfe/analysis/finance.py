# -*- coding: utf-8 -*-
"""
sfe/analysis/finance.py — Finance domain analysis helpers.

Responsibilities
----------------
- run_crisis_analysis()  : full pipeline — background + slice + detect + figures
- finance_figures()      : standard figures + crisis overlay if crisis attached
- print_regime()         : console summary of a RegimeResult

What this module does NOT do
-----------------------------
- Phase portrait logic   → figures.phase_portrait() / crisis_overlay()
- Regime thresholds      → regimes.CRISIS_THRESHOLDS
- Domain routing         → result.domain (set by connector)
- Data loading           → connectors/finance.py

The finance connector sets result.domain = "finance" and result.crisis.
Everything downstream reads those fields.
"""

from __future__ import annotations

import numpy as np

from ..connect import SFEResult
from ..connectors.finance import (
    slice_window, detect_regime, RegimeResult,
    TICKERS_COVID, TICKERS_2008, TICKERS_DOTCOM,
)

__all__ = [
    "run_crisis_analysis",
    "finance_figures",
    "print_regime",
]


# ---------------------------------------------------------------------------
# Console printer
# ---------------------------------------------------------------------------

def print_regime(regime: RegimeResult) -> None:
    print()
    print("=" * 62)
    print("  REGIME DETECTION RESULT")
    print("=" * 62)
    print(str(regime))
    print("=" * 62)
    print()


# ---------------------------------------------------------------------------
# Full pipeline convenience wrapper
# ---------------------------------------------------------------------------

def run_crisis_analysis(
    background: SFEResult,
    crisis_start: str,
    crisis_end: str,
    W: int | None = None,
    verbose: bool = True,
) -> tuple[SFEResult, SFEResult, RegimeResult]:
    """
    Slice a crisis window, detect the regime, attach results.

    Parameters
    ----------
    background   : SFEResult   full-period result (from finance connector)
    crisis_start : str         "YYYY-MM-DD"
    crisis_end   : str         "YYYY-MM-DD"
    W            : int, opt    override window (default: same as background)
    verbose      : bool

    Returns
    -------
    (background, crisis, regime)
        background.crisis is set to the crisis SFEResult.
        crisis.domain     is set to "finance".

    Example
    -------
    from sfe.connectors.finance import from_yfinance, TICKERS_COVID
    from sfe.analysis.finance import run_crisis_analysis, print_regime

    bg = from_yfinance(TICKERS_COVID, "2019-01-01", "2021-01-01", W=20)
    bg, crash, regime = run_crisis_analysis(bg, "2020-02-01", "2020-04-30")
    print_regime(regime)
    """
    crisis = slice_window(background, crisis_start, crisis_end, W=W)
    regime = detect_regime(background, crisis)

    # Attach to background so figures can read it
    background.crisis = crisis

    if verbose:
        print_regime(regime)

    return background, crisis, regime


# ---------------------------------------------------------------------------
# Finance figures
# ---------------------------------------------------------------------------

def finance_figures(
    background: SFEResult,
    crisis: SFEResult | None = None,
    regime: RegimeResult | None = None,
    title_prefix: str = "",
) -> dict:
    """
    Generate all figures for a finance domain result.

    Standard figures (phase portrait, timeseries, eigenspectrum) come from
    figures.all_figures(background).

    If crisis is provided (or attached as background.crisis), also generates:
        crisis_overlay  — background vs crisis phase portrait, side by side,
                          with Branch A/B verdict stamped on

    Parameters
    ----------
    background   : SFEResult   full-period result
    crisis       : SFEResult   crisis window (or None)
    regime       : RegimeResult (or None — re-detected if crisis provided)
    title_prefix : str

    Returns
    -------
    dict with keys: "phase_portrait", "timeseries", "eigenspectrum",
                    "crisis_overlay" (if crisis available)
    """
    from ..figures import all_figures, crisis_overlay

    pfx    = title_prefix or "Finance"
    crisis = crisis or getattr(background, "crisis", None)

    figs = all_figures(background, title_prefix=pfx)

    if crisis is not None:
        if regime is None:
            regime = detect_regime(background, crisis)
        figs["crisis_overlay"] = crisis_overlay(
            background, crisis, regime,
            title_prefix=pfx,
        )

    return figs
