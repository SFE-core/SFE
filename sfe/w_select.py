# -*- coding: utf-8 -*-
"""
sfe/w_select.py — W selection utilities for the SFE instrument.

Strategy 1 — Sample rate heuristic (runs automatically in connectors)
Strategy 2 — Multi-W stability sweep (this module)
Strategy 3 — dρ of dρ (reserved for SFE-12)

Usage
-----
    from sfe.w_select import suggest_W, sweep_W, print_W_report

    # Quick suggestion from sample rate alone
    suggestion = suggest_W(sfreq=1.0, T=3600, domain="strain")
    print_W_report(suggestion)

    # Full sweep — runs instrument at multiple W values
    from sfe.w_select import sweep_W
    sweep = sweep_W(data, labels=labels)
    print_W_report(sweep)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

__all__ = ["suggest_W", "sweep_W", "WSuggestion", "print_W_report"]

# ---------------------------------------------------------------------------
# Strategy 1 — heuristic table
# ---------------------------------------------------------------------------

# (sfreq_min, sfreq_max, W, label, reasoning)
_HEURISTIC_TABLE = [
    # sub-Hz — very slow processes (daily/hourly aggregates treated as timeseries)
    (0.0,    0.02,   20,  "~20 samples",    "slow process, use 20-sample window"),
    # ~1 Hz — structural / strain / slow sensors
    (0.02,   2.0,    60,  "1 minute",       "1Hz data: 1-minute coupling window"),
    # ~5–15 Hz — physiological slow band
    (2.0,    20.0,   60,  "~few seconds",   "low-freq bio/physical: 60-sample window"),
    # ~50–250 Hz — EEG / ECG / vibration
    (20.0,   300.0,  None, "1 second",      "use sfreq samples = 1-second window"),
    # >300 Hz — high-speed acquisition
    (300.0,  1e9,    None, "0.5 seconds",   "use sfreq//2 samples = 0.5-second window"),
]

# Daily / weekly financial heuristic (sfreq ~ 1 sample/day)
_FINANCE_W = 20    # 1 trading month
_TRAFFIC_W = 12    # 1 hour at 5-min intervals


def _heuristic_W(sfreq: float) -> tuple[int, str]:
    """Return (W, reasoning) from the heuristic table."""
    for lo, hi, w, label, reason in _HEURISTIC_TABLE:
        if lo <= sfreq < hi:
            if w is None:
                # Dynamic: compute from sfreq
                if sfreq >= 300:
                    w = max(int(sfreq // 2), 10)
                else:
                    w = max(int(round(sfreq)), 10)
            return w, f"{label} — {reason}"
    # Fallback
    return max(int(round(sfreq)), 10), "fallback: W = sfreq"


# ---------------------------------------------------------------------------
# WSuggestion dataclass
# ---------------------------------------------------------------------------

@dataclass
class WSuggestion:
    """
    Result of suggest_W() or sweep_W().

    Attributes
    ----------
    recommended_W   : int     the W to use
    reasoning       : str     human-readable explanation
    strategy        : str     "heuristic" | "sweep" | "manual"
    sfreq           : float
    T               : int
    domain          : str
    sweep_results   : list    populated by sweep_W(), empty for suggest_W()
    ai_note         : str     message to show the user about AI interpretation
    """
    recommended_W : int
    reasoning     : str
    strategy      : str
    sfreq         : float  = 1.0
    T             : int    = 0
    domain        : str    = "unknown"
    sweep_results : list   = field(default_factory=list)
    ai_note       : str    = ""

    def summary_for_prompt(self) -> str:
        """Compact string suitable for inclusion in an LLM prompt."""
        lines = [
            f"W selection — strategy: {self.strategy}",
            f"  sfreq={self.sfreq} Hz  T={self.T}  domain={self.domain}",
            f"  recommended W={self.recommended_W}  ({self.reasoning})",
        ]
        if self.sweep_results:
            lines.append("  Sweep stability scores (lower = more stable):")
            for r in self.sweep_results:
                marker = " ←" if r["W"] == self.recommended_W else ""
                lines.append(
                    f"    W={r['W']:>4}  stability={r['stability']:.5f}  "
                    f"rho*_mean={r['rho_star_mean']:.3f}{marker}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Strategy 1 — suggest_W (no data required)
# ---------------------------------------------------------------------------

def suggest_W(
    sfreq: float = 1.0,
    T: int = 0,
    domain: str = "unknown",
) -> WSuggestion:
    """
    Suggest W from sample rate and domain alone (Strategy 1).
    No data required — fast, runs before any instrument call.

    Parameters
    ----------
    sfreq  : float   sample rate in Hz
    T      : int     number of timesteps (used for sanity check only)
    domain : str     "finance" | "eeg" | "traffic" | "strain" | "unknown"

    Returns
    -------
    WSuggestion
    """
    domain_l = domain.lower()

    # Domain overrides
    if domain_l == "finance":
        W, reason = _FINANCE_W, "finance: 1 trading month (20 days)"
    elif domain_l == "traffic":
        W, reason = _TRAFFIC_W, "traffic: 1 hour at 5-min intervals (12 steps)"
    elif domain_l in ("eeg", "ecg", "neural"):
        W = max(int(round(sfreq)), 10)
        reason = f"neural: 1-second window ({W} samples at {sfreq} Hz)"
    elif domain_l in ("strain", "structural", "vibration"):
        W = 60
        reason = "structural/strain: 1-minute coupling window at 1 Hz"
    else:
        W, reason = _heuristic_W(sfreq)

    # Sanity — W must be < T // 3
    if T > 0:
        max_W = T // 3
        if W > max_W:
            W = max(max_W, 10)
            reason += f" (capped at T//3={max_W})"

    ai_note = (
        f"Manual W suggestion: W={W} ({reason})\n"
        f"For better W selection and interpretation, connect to an AI — "
        f"set SFE_LLM_API_KEY or pass a LLMConfig to sfe.ai.interpret()."
    )

    return WSuggestion(
        recommended_W = W,
        reasoning     = reason,
        strategy      = "heuristic",
        sfreq         = sfreq,
        T             = T,
        domain        = domain,
        ai_note       = ai_note,
    )


# ---------------------------------------------------------------------------
# Strategy 2 — sweep_W (runs instrument at multiple W values)
# ---------------------------------------------------------------------------

def sweep_W(
    data: np.ndarray,
    labels: list[str] | None = None,
    sfreq: float = 1.0,
    domain: str = "unknown",
    W_list: list[int] | None = None,
) -> WSuggestion:
    """
    Run SFE at multiple W values and select the most stable one (Strategy 2).

    Stability score = mean variance of ρ* across pairs over the three W runs.
    Lower score = more consistent reading = better W.

    Parameters
    ----------
    data    : ndarray (T, N)
    labels  : list of str
    sfreq   : float
    domain  : str
    W_list  : list of int   W values to sweep. Default: auto from heuristic.

    Returns
    -------
    WSuggestion with sweep_results populated.
    """
    from .connect import from_array  # local import — avoid circular

    data = np.asarray(data, dtype=float)
    T, N = data.shape

    # Build W_list from heuristic if not provided
    if W_list is None:
        base = suggest_W(sfreq=sfreq, T=T, domain=domain).recommended_W
        W_list = sorted(set([
            max(base // 2, 10),
            base,
            min(base * 2, T // 3),
        ]))

    sweep = []
    for W in W_list:
        if W < 2 or W >= T:
            continue
        try:
            result = from_array(data, W=W, labels=labels)
            rho_stars  = [p["rho_star"] for p in result.pairs]
            drho_means = [p["drho_mean"] for p in result.pairs]

            # Stability = mean of drho across pairs
            # Low drho = geometrically stable channel = good W
            stability = float(np.mean(drho_means)) if drho_means else float("nan")

            sweep.append({
                "W":            W,
                "stability":    stability,
                "rho_star_mean": float(np.mean(rho_stars)) if rho_stars else float("nan"),
                "n_reliable":   sum(p["zone"] == "reliable" for p in result.pairs),
                "n_flagged":    sum(p["nonstationary_pct"] > 40 for p in result.pairs),
            })
        except Exception:
            continue  # skip invalid W values silently

    if not sweep:
        # Fall back to heuristic
        return suggest_W(sfreq=sfreq, T=T, domain=domain)

    # Best W = lowest stability score (most geometrically stable)
    best = min(sweep, key=lambda r: r["stability"])
    W_best = best["W"]

    reasoning = (
        f"sweep over W={[r['W'] for r in sweep]} — "
        f"W={W_best} gives lowest mean dρ={best['stability']:.6f} "
        f"(most stable channel geometry)"
    )

    ai_note = (
        f"Manual W suggestion: W={W_best} ({reasoning})\n"
        f"For better W selection and interpretation, connect to an AI — "
        f"set SFE_LLM_API_KEY or pass a LLMConfig to sfe.ai.interpret()."
    )

    return WSuggestion(
        recommended_W = W_best,
        reasoning     = reasoning,
        strategy      = "sweep",
        sfreq         = sfreq,
        T             = T,
        domain        = domain,
        sweep_results = sweep,
        ai_note       = ai_note,
    )


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_W_report(suggestion: WSuggestion) -> None:
    """Print a human-readable W selection report to the console."""
    print()
    print("=" * 62)
    print("  W SELECTION REPORT")
    print("=" * 62)
    print(f"  Strategy      : {suggestion.strategy}")
    print(f"  Domain        : {suggestion.domain}")
    print(f"  Sample rate   : {suggestion.sfreq} Hz")
    print(f"  T (timesteps) : {suggestion.T}")
    print()
    print(f"  Recommended W : {suggestion.recommended_W}")
    print(f"  Reasoning     : {suggestion.reasoning}")

    if suggestion.sweep_results:
        print()
        print(f"  {'W':>6}  {'stability (dρ)':>16}  {'ρ* mean':>9}  "
              f"{'reliable':>8}  {'flagged':>7}")
        print("  " + "-" * 52)
        for r in suggestion.sweep_results:
            marker = " ←" if r["W"] == suggestion.recommended_W else ""
            print(f"  {r['W']:>6}  {r['stability']:>16.6f}  "
                  f"{r['rho_star_mean']:>9.3f}  "
                  f"{r['n_reliable']:>8}  {r['n_flagged']:>7}{marker}")

    print()
    print(f"  {suggestion.ai_note}")
    print("=" * 62)
    print()
