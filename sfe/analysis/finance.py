# -*- coding: utf-8 -*-
"""
sfe/analysis/finance.py — Analysis helpers for finance domain SFEResults.

Connectors produce SFEResults. This module operates on them.

    from sfe.analysis.finance import slice_window, detect_regime, RegimeResult
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from ..connect import SFEResult


__all__ = [
    "slice_window",
    "detect_regime",
    "RegimeResult",
    "REGIME_THRESHOLDS",
]


from .regimes import CRISIS_THRESHOLDS as REGIME_THRESHOLDS


@dataclass
class RegimeResult:
    """
    Returned by detect_regime().

    Attributes
    ----------
    label               : str
    branch              : str   "A" | "B" | "none" | "silent"
    fired               : bool
    delta_rho           : float
    delta_reff          : float
    bandgap_ratio       : float
    pairs_elevated_pct  : float
    drho_elevated_pct   : float
    notes               : list of str
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
# Window slicer
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
    result : SFEResult
    start  : str         "YYYY-MM-DD"
    end    : str         "YYYY-MM-DD"
    W      : int, opt    override window size (default: same as original)

    Returns
    -------
    SFEResult for the sliced window with .dates set.
    """
    import pandas as _pd
    from ..connect import from_dataframe

    if not hasattr(result, "dates"):
        raise AttributeError(
            "result.dates not found. Use a finance connector "
            "(from_yfinance / from_price_csv / from_price_dataframe)."
        )

    mask  = (result.dates >= start) & (result.dates <= end)
    dates = result.dates[mask]

    if mask.sum() == 0:
        raise ValueError(f"No data in window {start} → {end}.")

    W_use       = W if W is not None else result.W
    data_window = result.data[mask.values]
    df_window   = _pd.DataFrame(data_window, columns=result.labels, index=dates)

    sub       = from_dataframe(df_window, W=W_use)
    sub.dates = dates
    return sub


# ---------------------------------------------------------------------------
# Regime detector — Proposition 12 (SFE-11)
# ---------------------------------------------------------------------------

def detect_regime(
    background: SFEResult,
    crisis: SFEResult,
) -> RegimeResult:
    """
    Classify the crisis window against Proposition 12 Branch A / Branch B.

    Branch A (acute homogeneous crisis — e.g. COVID crash):
        ρ* > background + 0.10  AND  band gap ≥ 1.50×  AND  r_eff_corr < background − 0.10

    Branch B (acute heterogeneous contagion — e.g. 2008 Lehman):
        ρ* rises on > 50% of pairs  AND  dρ rises on > 50%  AND  r_eff_corr < background − 0.10
        Band gap does NOT explode — variance diffuses across modes.

    Silent on gradual corrections (e.g. dot-com 2000-02).
    """
    thr   = REGIME_THRESHOLDS
    notes = []

    bg_pairs      = {p["label"]: p for p in background.pairs}
    cr_pairs      = {p["label"]: p for p in crisis.pairs}
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

    mean_delta_rho = float(np.mean(delta_rhos))
    pairs_rho_up   = float(np.mean([d > 0 for d in delta_rhos]))  * 100
    pairs_drho_up  = float(np.mean([d > 0 for d in delta_drhos])) * 100

    delta_reff    = crisis.reff_corr - background.reff_corr
    bandgap_ratio = (crisis.band_gap / background.band_gap
                     if background.band_gap > 1e-9 else float("nan"))

    # Branch A
    branch_a_rho  = mean_delta_rho  >= thr["rho_delta_min"]
    branch_a_bg   = (not np.isnan(bandgap_ratio) and
                     bandgap_ratio  >= thr["bandgap_mult_min"])
    branch_a_reff = delta_reff      <= thr["reff_delta_max"]

    if branch_a_rho and branch_a_bg and branch_a_reff:
        notes.append(
            f"Band gap explosion ({background.band_gap:.2f}× → {crisis.band_gap:.2f}× "
            f"= {bandgap_ratio:.2f}× ratio) is the distinguishing signature."
        )
        return RegimeResult(
            label="CRISIS COUPLING — Branch A ✓  (acute homogeneous shock)",
            branch="A", fired=True,
            delta_rho=mean_delta_rho, delta_reff=delta_reff,
            bandgap_ratio=bandgap_ratio,
            pairs_elevated_pct=pairs_rho_up,
            drho_elevated_pct=pairs_drho_up,
            notes=notes,
        )

    # Branch B
    branch_b_rho  = pairs_rho_up  > thr["pairs_pct_min"] * 100
    branch_b_drho = pairs_drho_up > thr["pairs_pct_min"] * 100
    branch_b_reff = delta_reff    <= thr["reff_delta_max"]

    if branch_b_rho and branch_b_drho and branch_b_reff:
        notes.append(
            f"Band gap stable ({bandgap_ratio:.2f}×) — variance diffused across modes."
        )
        return RegimeResult(
            label="CRISIS COUPLING — Branch B ✓  (heterogeneous contagion)",
            branch="B", fired=True,
            delta_rho=mean_delta_rho, delta_reff=delta_reff,
            bandgap_ratio=bandgap_ratio,
            pairs_elevated_pct=pairs_rho_up,
            drho_elevated_pct=pairs_drho_up,
            notes=notes,
        )

    # Silent
    if mean_delta_rho < thr["rho_delta_min"]:
        notes.append(
            f"ρ* elevation ({mean_delta_rho:+.4f}) below threshold — "
            f"consistent with gradual correction or stable background coupling."
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
