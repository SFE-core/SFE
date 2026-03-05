# -*- coding: utf-8 -*-
"""
sfe/figures.py — Standard figures for any SFEResult.

Three plots, domain-agnostic, called automatically by save_run():
    fig_phase_portrait  — ρ* vs dρ scatter, operating envelope zones
    fig_timeseries      — rolling ρ and dρ over time for each pair
    fig_eigenspectrum   — eigenvalue spectrum + r_eff joint trajectory

Usage
-----
    from sfe.figures import phase_portrait, timeseries, eigenspectrum

    fig1 = phase_portrait(result)
    fig2 = timeseries(result)
    fig3 = eigenspectrum(result)

    # Or all at once
    from sfe.figures import all_figures
    figs = all_figures(result)   # returns dict keyed by name
"""

from __future__ import annotations

import numpy as np

__all__ = ["phase_portrait", "timeseries", "eigenspectrum", "all_figures"]

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

_BG      = "#0d0d0d"
_PANEL   = "#111111"
_GRID    = "#1e1e1e"
_TEXT    = "#aaaaaa"
_WHITE   = "#ffffff"
_COLORS  = ["#ff6b35", "#00b4d8", "#ffd60a", "#c77dff",
            "#81c784", "#ff8a65", "#4fc3f7", "#aed581",
            "#f48fb1", "#80cbc4", "#ffcc02", "#ce93d8"]

def _style_ax(ax):
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_TEXT, labelsize=8)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)
    ax.grid(color=_GRID, lw=0.5, alpha=0.6)


def _fig(nrows, ncols, figsize, title):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=_BG)
    fig.suptitle(title, color=_WHITE, fontsize=10, y=0.98)
    return fig, axes


# ---------------------------------------------------------------------------
# Fig 1 — Phase portrait
# ---------------------------------------------------------------------------

def phase_portrait(result, title: str | None = None) -> "Figure":
    """
    ρ* vs dρ scatter for all pairs.
    Operating envelope zones drawn as background regions.

    Parameters
    ----------
    result : SFEResult
    title  : str, optional

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from .core import OPERATING_ENVELOPE

    env    = OPERATING_ENVELOPE
    rmin   = env["reliable_rho_min"]
    dmax   = env["degraded_rho_max"]

    title  = title or "SFE Phase Portrait — ρ* vs dρ"
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=_BG)
    _style_ax(ax)

    # Zone backgrounds
    ax.axvspan(rmin, 1.05, color="#1a3a1a", alpha=0.35, zorder=0)
    ax.axvspan(-0.05, dmax, color="#3a1a1a", alpha=0.25, zorder=0)
    ax.axvline(rmin, color="#38b000", lw=1.0, ls="--", alpha=0.6)
    ax.axvline(dmax, color="#ff4444", lw=1.0, ls="--", alpha=0.4)

    # Pairs
    for k, p in enumerate(result.pairs):
        color  = _COLORS[k % len(_COLORS)]
        marker = "o" if p["zone"] == "reliable" else \
                 "x" if p["zone"] == "degraded" else "^"
        ax.scatter(p["rho_star"], p["drho_mean"],
                   color=color, marker=marker, s=90,
                   zorder=3, edgecolors="#222", lw=0.5,
                   label=p["label"])
        ax.annotate(p["label"], (p["rho_star"], p["drho_mean"]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7, color=color, alpha=0.85)

    # Legend patches for zones
    patches = [
        mpatches.Patch(color="#1a3a1a", alpha=0.8, label=f"reliable (ρ*>{rmin})"),
        mpatches.Patch(color="#3a1a1a", alpha=0.8, label=f"degraded (ρ*<{dmax})"),
        mpatches.Patch(color="#2a2a1a", alpha=0.8, label="marginal"),
    ]
    ax.legend(handles=patches, fontsize=7, facecolor="#1a1a1a",
              labelcolor=_TEXT, loc="upper right")

    ax.set_xlabel("ρ* (mean absolute rolling correlation)")
    ax.set_ylabel("dρ (mean correlation variance)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(bottom=0)
    ax.set_title(title, color=_WHITE, fontsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 2 — Timeseries
# ---------------------------------------------------------------------------

def timeseries(result, title: str | None = None,
               max_pairs: int = 6) -> "Figure":
    """
    Rolling ρ and dρ over time for each pair (up to max_pairs).

    Parameters
    ----------
    result    : SFEResult
    title     : str, optional
    max_pairs : int   max pairs to plot (default 6, sorted by ρ* desc)

    Returns
    -------
    matplotlib Figure
    """
    pairs  = result.pairs[:max_pairs]
    n      = len(pairs)
    title  = title or "SFE Timeseries — Rolling ρ and dρ"

    fig, axes = _fig(n, 2, (14, 2.2 * n), title)
    if n == 1:
        axes = [axes]

    for row, p in enumerate(pairs):
        ax_rho, ax_drho = axes[row]
        color = _COLORS[row % len(_COLORS)]
        t     = np.arange(len(p["rho"]))

        _style_ax(ax_rho)
        ax_rho.plot(t, p["rho"], color=color, lw=0.8, alpha=0.9)
        ax_rho.axhline(0.45,  color="#38b000", lw=0.8, ls="--", alpha=0.5)
        ax_rho.axhline(0.0,   color=_GRID, lw=0.5)
        ax_rho.axhline(-0.45, color="#38b000", lw=0.8, ls="--", alpha=0.5)
        ax_rho.set_ylim(-1.05, 1.05)
        ax_rho.set_ylabel(f"{p['label']}\nρ", color=_TEXT, fontsize=8)

        _style_ax(ax_drho)
        ax_drho.plot(t, p["drho"], color="#ffd60a", lw=0.8, alpha=0.9)
        ax_drho.axhline(p["drho_mean"], color=_TEXT, lw=0.8,
                        ls="--", alpha=0.5,
                        label=f"mean={p['drho_mean']:.5f}")
        ax_drho.set_ylabel("dρ", color=_TEXT, fontsize=8)
        ax_drho.legend(fontsize=7, facecolor="#1a1a1a", labelcolor=_TEXT)

    axes[-1][0].set_xlabel("sample", color=_TEXT)
    axes[-1][1].set_xlabel("sample", color=_TEXT)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 3 — Eigenspectrum
# ---------------------------------------------------------------------------

def eigenspectrum(result, title: str | None = None) -> "Figure":
    """
    Eigenvalue spectrum of the global covariance + r_eff joint trajectory.

    Parameters
    ----------
    result : SFEResult
    title  : str, optional

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt

    title = title or "SFE Eigenspectrum & r_eff Joint"
    fig, (ax_spec, ax_reff) = plt.subplots(1, 2, figsize=(12, 5), facecolor=_BG)
    _style_ax(ax_spec)
    _style_ax(ax_reff)
    fig.suptitle(title, color=_WHITE, fontsize=10)

    # Eigenspectrum from global covariance
    cov  = np.cov(result.data.T)
    if cov.ndim < 2:
        cov = np.array([[float(cov)]])
    ev   = np.sort(np.linalg.eigvalsh(cov))[::-1]
    evn  = ev / ev.sum()
    N    = len(ev)

    colors_bar = ["#ff6b35"] + ["#00b4d8"] * (N - 1)
    ax_spec.bar(range(1, N + 1), evn * 100,
                color=colors_bar, edgecolor="#222", width=0.7)

    for k, v in enumerate(evn * 100):
        ax_spec.text(k + 1, v + 0.3, f"{v:.1f}%",
                     ha="center", color=_TEXT, fontsize=7)

    if N > 1:
        bg = ev[0] / ev[1] if ev[1] > 1e-12 else float("nan")
        ax_spec.text(1, evn[0] * 100 + 1.5,
                     f"λ₁/λ₂={bg:.2f}×",
                     color="#ff6b35", fontsize=9, ha="center")

    ax_spec.set_xlabel("Eigenvalue index")
    ax_spec.set_ylabel("Variance explained %")
    ax_spec.set_title("Global eigenspectrum", color=_WHITE, fontsize=9)

    # r_eff joint trajectory
    rj = result.reff_joint
    t  = np.arange(len(rj))
    ax_reff.plot(t, rj, color="#c77dff", lw=1.5)
    ax_reff.axhline(1.0, color="#38b000", lw=1, ls="--",
                    alpha=0.6, label="r_eff=1 (rank collapse)")
    ax_reff.axhline(float(N), color=_TEXT, lw=1, ls="--",
                    alpha=0.4, label=f"r_eff={N} (isotropic)")
    ax_reff.axhline(np.nanmean(rj), color="#ffd60a", lw=1, ls=":",
                    alpha=0.7, label=f"mean={np.nanmean(rj):.3f}")
    ax_reff.set_xlabel("window block")
    ax_reff.set_ylabel("r_eff joint")
    ax_reff.set_title("r_eff joint trajectory", color=_WHITE, fontsize=9)
    ax_reff.legend(fontsize=7, facecolor="#1a1a1a", labelcolor=_TEXT)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# All figures at once
# ---------------------------------------------------------------------------

def all_figures(result, title_prefix: str = "") -> dict:
    """
    Generate all three standard figures.

    Parameters
    ----------
    result       : SFEResult
    title_prefix : str   prepended to each figure title

    Returns
    -------
    dict with keys: "phase_portrait", "timeseries", "eigenspectrum"
    """
    pfx = f"{title_prefix} — " if title_prefix else ""
    return {
        "phase_portrait" : phase_portrait(result,   f"{pfx}Phase Portrait"),
        "timeseries"     : timeseries(result,        f"{pfx}Timeseries"),
        "eigenspectrum"  : eigenspectrum(result,     f"{pfx}Eigenspectrum"),
    }
