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

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from matplotlib.figure import Figure

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
# Cross-domain reference anchors — from regimes.py (single source of truth)
# Labels are "{domain} — {mechanism}" so they convey meaning regardless of
# the analyst's domain background.
# ---------------------------------------------------------------------------

from sfe.analysis.regimes import ref_points


# ---------------------------------------------------------------------------
# Fig 1 — Phase portrait
# ---------------------------------------------------------------------------

def phase_portrait(result, title: str | None = None) -> "Figure":
    """
    ρ* vs dρ scatter with four named coupling regions.

    Regions are painted first, then live data is plotted on top.
    Axis is scaled to live data only — reference stars are plotted
    after the axis is frozen so distant anchors don't compress the view.

    Four regions (substrate-independent):
        Locked       — ρ* > 0.70, dρ low   (strong, stable coupling)
        Transitioning — ρ* > 0.70, dρ high  (strong but unstable)
        Active       — ρ* 0.45–0.70         (real but not dominant)
        Decoupled    — ρ* < 0.45            (weak / instrument limit)

    The dρ boundary between Locked and Transitioning is relative:
    40% of the live data's max dρ, so it scales correctly across domains
    (strain at dρ≈0.0001 and finance at dρ≈0.030 get the same regions).
    """
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    from .core import OPERATING_ENVELOPE

    env  = OPERATING_ENVELOPE
    rmin = env["reliable_rho_min"]   # 0.45
    dmax = env["degraded_rho_max"]   # 0.20
    locked_rho = 0.70

    s      = result.summary_dict()
    bg_str = f"λ₁/λ₂={s['band_gap']:.2f}×   r_eff corr={s['reff_corr']:.3f}"
    title  = title or f"SFE — Phase Portrait   |   {bg_str}"

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=_BG)
    _style_ax(ax)

    # ── Step 1: plot live data, let matplotlib find natural bounds ──────
    for k, p in enumerate(result.pairs):
        ax.scatter(p["rho_star"], p["drho_mean"],
                   color=_COLORS[k % len(_COLORS)],
                   marker="o", s=60, zorder=5, alpha=0.90)

    ax.autoscale(enable=True)
    ax.set_ylim(bottom=0)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_pad = (x1 - x0) * 0.12
    y_pad = (y1 - y0) * 0.25     # extra headroom for region labels
    x0 = min(x0 - x_pad, -0.05)
    x1 = max(x1 + x_pad,  1.05)
    y1 = y1 + y_pad
    ax.set_xlim(x0, x1)
    ax.set_ylim(0,  y1)
    ax.autoscale(enable=False)   # freeze before drawing anything else

    # ── Step 2: dρ split threshold — relative to live data ─────────────
    # 40% of max observed dρ. Scales across domains automatically.
    live_drho_max = max(p["drho_mean"] for p in result.pairs) if result.pairs else y1
    drho_split    = live_drho_max * 0.40

    # ── Step 3: region backgrounds ──────────────────────────────────────
    # Decoupled  — ρ* < 0.45  (full height, drawn first)
    ax.axvspan(x0, dmax,       color="#3a1a1a", alpha=0.30, zorder=0)
    # Active     — 0.45 ≤ ρ* < 0.70
    ax.axvspan(dmax, locked_rho, color="#1a2a3a", alpha=0.30, zorder=0)
    # Locked     — ρ* ≥ 0.70, dρ ≤ drho_split
    ax.axvspan(locked_rho, x1,  color="#1a3a1a", alpha=0.30, zorder=0)

    # Transitioning overlay — ρ* ≥ 0.70, dρ > drho_split
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle(
        (locked_rho, drho_split), x1 - locked_rho, y1 - drho_split,
        facecolor="#3a2a0a", alpha=0.45, zorder=0,
    ))

    # Horizontal split line (Locked / Transitioning boundary)
    ax.axhline(drho_split, xmin=(locked_rho - x0) / (x1 - x0),
               color="#ffd54f", lw=0.7, ls=":", alpha=0.5, zorder=1)

    # Vertical boundary lines
    ax.axvline(dmax,       color="#555555", lw=0.8, ls="--", alpha=0.5, zorder=1)
    ax.axvline(rmin,       color="#ffd54f", lw=0.8, ls="--", alpha=0.6, zorder=1)
    ax.axvline(locked_rho, color="#888888", lw=0.8, ls="--", alpha=0.5, zorder=1)

    # ── Step 4: region labels ────────────────────────────────────────────
    # Placed at top of each region, inside the canvas
    label_y = y1 * 0.93
    _region_labels = [
        (dmax / 2 + x0 / 2,              label_y, "Decoupled",     "#aa4444"),
        ((dmax + locked_rho) / 2,         label_y, "Active",        "#4488aa"),
        ((locked_rho + x1) / 2,           label_y, "Locked",        "#44aa66"),
    ]
    for lx, ly, ltext, lcolor in _region_labels:
        ax.text(lx, ly, ltext,
                color=lcolor, fontsize=9, fontweight="bold",
                ha="center", va="top", alpha=0.85, zorder=2)

    # Transitioning label — top-right quadrant
    ax.text((locked_rho + x1) / 2, y1 * 0.70,
            "Transitioning",
            color="#cc8833", fontsize=9, fontweight="bold",
            ha="center", va="top", alpha=0.85, zorder=2)

    # ── Step 5: reference stars (after axis frozen) ─────────────────────
    legend_handles = []
    for rx, ry, rc, rl in ref_points():
        ax.scatter(rx, ry, marker="*", s=150, color=rc,
                   zorder=3, clip_on=False, alpha=0.90)
        legend_handles.append(
            mlines.Line2D([], [], marker="*", color=rc, lw=0,
                          markersize=9, label=rl)
        )

    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=7,
                  facecolor="#1a1a1a", labelcolor=_TEXT,
                  loc="upper left", title="Known regimes",
                  title_fontsize=7)

    ax.set_xlabel("ρ* (mean absolute rolling correlation)")
    ax.set_ylabel("dρ (mean correlation variance)")
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
    """
    pairs  = result.pairs[:max_pairs]
    n      = len(pairs)
    title  = title or "SFE Timeseries — Rolling ρ and dρ"

    fig, axes = _fig(n, 2, (14, 2.2 * n), title)
    if n == 1:
        axes = [axes]

    W = result.W
    for row, p in enumerate(pairs):
        ax_rho, ax_drho = axes[row]
        color = _COLORS[row % len(_COLORS)]
        rho_  = p["rho"][W:]
        drho_ = p["drho"][W:]
        t     = np.arange(len(rho_))

        _style_ax(ax_rho)
        ax_rho.plot(t, rho_, color=color, lw=0.8, alpha=0.9)
        ax_rho.axhline(0.45,  color="#38b000", lw=0.8, ls="--", alpha=0.5)
        ax_rho.axhline(0.0,   color=_GRID, lw=0.5)
        ax_rho.axhline(-0.45, color="#38b000", lw=0.8, ls="--", alpha=0.5)
        ax_rho.set_ylim(-1.05, 1.05)
        ax_rho.set_ylabel(f"{p['label']}\nρ", color=_TEXT, fontsize=8)

        _style_ax(ax_drho)
        ax_drho.plot(t, drho_, color="#ffd60a", lw=0.8, alpha=0.9)
        ax_drho.axhline(p["drho_mean"], color=_TEXT, lw=0.8,
                        ls="--", alpha=0.5,
                        label=f"mean={p['drho_mean']:.6f}")
        drho_p99 = float(np.nanpercentile(drho_, 99)) if len(drho_) else p["drho_mean"]
        pad = max(drho_p99 * 0.20, 1e-7)
        ax_drho.set_ylim(0, drho_p99 + pad)
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
    """
    import matplotlib.pyplot as plt

    s     = result.summary_dict()
    bg    = s["band_gap"]
    rc    = s["reff_corr"]
    title = title or f"SFE Eigenspectrum & r_eff Joint   |   λ₁/λ₂={bg:.2f}×   r_eff corr={rc:.3f}"

    fig, (ax_spec, ax_reff) = plt.subplots(1, 2, figsize=(12, 5), facecolor=_BG)
    _style_ax(ax_spec)
    _style_ax(ax_reff)
    fig.suptitle(title, color=_WHITE, fontsize=10)

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
        ax_spec.text(1, evn[0] * 100 + 1.5,
                     f"λ₁/λ₂={bg:.2f}×",
                     color="#ff6b35", fontsize=9, ha="center")

    ax_spec.set_xlabel("Eigenvalue index")
    ax_spec.set_ylabel("Variance explained %")
    ax_spec.set_title("Global eigenspectrum", color=_WHITE, fontsize=9)

    rj = result.reff_joint
    t  = np.arange(len(rj))
    ax_reff.plot(t, rj, color="#c77dff", lw=1.5)
    ax_reff.axhline(1.0, color="#38b000", lw=1, ls="--",
                    alpha=0.6, label="r_eff=1 (rank collapse)")
    ax_reff.axhline(float(N), color=_TEXT, lw=1, ls="--",
                    alpha=0.4, label=f"r_eff={N} (isotropic)")
    ax_reff.axhline(np.nanmean(rj), color="#ffd60a", lw=1, ls=":",
                    alpha=0.7, label=f"joint mean={np.nanmean(rj):.3f}")
    ax_reff.axhline(rc, color="#c77dff", lw=1, ls=":",
                    alpha=0.7, label=f"corrected={rc:.3f}")
    ax_reff.set_xlabel("window block")
    ax_reff.set_ylabel("r_eff joint")
    ax_reff.set_title("r_eff joint trajectory", color=_WHITE, fontsize=9)
    ax_reff.legend(fontsize=7, facecolor="#1a1a1a", labelcolor=_TEXT)

    fig.tight_layout()

    fig.sfe_band_gap  = bg
    fig.sfe_reff_corr = rc

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