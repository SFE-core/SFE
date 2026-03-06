# -*- coding: utf-8 -*-
"""
sfe/figures.py — Standard figures for any SFEResult.

Three plots, domain-agnostic in interface, domain-aware in content.
Called automatically by save_run().

    phase_portrait(result)   — ρ* vs dρ scatter, operating envelope zones
    timeseries(result)       — rolling ρ and dρ over time for each pair
    eigenspectrum(result)    — eigenvalue spectrum + r_eff joint trajectory

Domain identity
---------------
All figures read result.domain (set by the connector) and never accept
a domain parameter. ref_points(result.domain) is the single call that
decides what anchors appear on a phase portrait. figures.py contains
no domain logic of its own.

Strain-specific figures (within/cross split panel, heatmap, diurnal)
live in sfe/analysis/strain.py. They call phase_portrait(result) from
here rather than reimplementing it.

Usage
-----
    from sfe.figures import phase_portrait, timeseries, eigenspectrum, all_figures

    fig1 = phase_portrait(result)
    fig2 = timeseries(result)
    fig3 = eigenspectrum(result)

    figs = all_figures(result)   # dict keyed by name
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from matplotlib.figure import Figure

import numpy as np

__all__ = ["phase_portrait", "timeseries", "eigenspectrum",
           "crisis_overlay", "all_figures"]


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_BG     = "#0d0d0d"
_PANEL  = "#111111"
_GRID   = "#1e1e1e"
_TEXT   = "#aaaaaa"
_WHITE  = "#ffffff"
_COLORS = [
    "#ff6b35", "#00b4d8", "#ffd60a", "#c77dff",
    "#81c784", "#ff8a65", "#4fc3f7", "#aed581",
    "#f48fb1", "#80cbc4", "#ffcc02", "#ce93d8",
]


def _style_ax(ax):
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_TEXT, labelsize=8)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)
    ax.grid(color=_GRID, lw=0.5, alpha=0.6)


# ---------------------------------------------------------------------------
# Fig 1 — Phase portrait
# ---------------------------------------------------------------------------

def phase_portrait(result, title: str | None = None) -> "Figure":
    """
    ρ* vs dρ scatter with four named coupling regions.

    Reads result.domain to determine which reference anchors to show.
    No domain parameter — the result knows what it is.

    Regions (substrate-independent):
        Locked        — ρ* > 0.70, dρ low    (strong, stable coupling)
        Transitioning — ρ* > 0.70, dρ high   (strong but unstable)
        Active        — ρ* 0.45–0.70          (real but not dominant)
        Decoupled     — ρ* < 0.45             (weak / instrument limit)

    The dρ boundary between Locked and Transitioning scales to the live
    data's max dρ (40%), so strain at dρ≈0.0001 and finance at dρ≈0.030
    both get meaningful region boundaries.
    """
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    from .core import OPERATING_ENVELOPE
    from .analysis.regimes import ref_points

    env  = OPERATING_ENVELOPE
    rmin = env["reliable_rho_min"]   # 0.45
    dmax = env["degraded_rho_max"]   # 0.20
    locked_rho = 0.70

    s      = result.summary_dict()
    bg_str = f"λ₁/λ₂={s['band_gap']:.2f}×   r_eff corr={s['reff_corr']:.3f}"
    title  = title or f"SFE — Phase Portrait   |   {bg_str}"

    # Domain identity — read once, used for anchor selection
    domain = getattr(result, "domain", None)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=_BG)
    _style_ax(ax)

    # ── Step 1: plot live data, let matplotlib find natural bounds ──────
    for k, p in enumerate(result.pairs):
        ax.scatter(p["rho_star"], p["drho_mean"],
                   color=_COLORS[k % len(_COLORS)],
                   marker="o", s=60, zorder=5, alpha=0.90,
                   label=p["label"])

    ax.autoscale(enable=True)
    ax.set_ylim(bottom=0)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_pad = (x1 - x0) * 0.12
    y_pad = (y1 - y0) * 0.25
    x0 = min(x0 - x_pad, -0.05)
    x1 = max(x1 + x_pad,  1.05)
    y1 = y1 + y_pad
    ax.set_xlim(x0, x1)
    ax.set_ylim(0,  y1)
    ax.autoscale(enable=False)

    # ── Step 2: dρ split threshold — relative to live data ─────────────
    live_drho_max = max(p["drho_mean"] for p in result.pairs) if result.pairs else y1
    drho_split    = live_drho_max * 0.40

    # ── Step 3: region backgrounds ──────────────────────────────────────
    ax.axvspan(x0,         dmax,       color="#3a1a1a", alpha=0.30, zorder=0)
    ax.axvspan(dmax,       locked_rho, color="#1a2a3a", alpha=0.30, zorder=0)
    ax.axvspan(locked_rho, x1,         color="#1a3a1a", alpha=0.30, zorder=0)

    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle(
        (locked_rho, drho_split), x1 - locked_rho, y1 - drho_split,
        facecolor="#3a2a0a", alpha=0.45, zorder=0,
    ))

    # Horizontal split line
    ax.axhline(drho_split, xmin=(locked_rho - x0) / (x1 - x0),
               color="#ffd54f", lw=0.7, ls=":", alpha=0.5, zorder=1)

    # Vertical boundary lines
    ax.axvline(dmax,       color="#555555", lw=0.8, ls="--", alpha=0.5, zorder=1)
    ax.axvline(rmin,       color="#ffd54f", lw=0.8, ls="--", alpha=0.6, zorder=1)
    ax.axvline(locked_rho, color="#888888", lw=0.8, ls="--", alpha=0.5, zorder=1)

    # ── Step 4: region labels ────────────────────────────────────────────
    label_y = y1 * 0.93
    for lx, ltext, lcolor in [
        (dmax / 2 + x0 / 2,            "Decoupled",    "#aa4444"),
        ((dmax + locked_rho) / 2,       "Active",       "#4488aa"),
        ((locked_rho + x1) / 2,         "Locked",       "#44aa66"),
    ]:
        ax.text(lx, label_y, ltext,
                color=lcolor, fontsize=9, fontweight="bold",
                ha="center", va="top", alpha=0.85, zorder=2)

    ax.text((locked_rho + x1) / 2, y1 * 0.70,
            "Transitioning",
            color="#cc8833", fontsize=9, fontweight="bold",
            ha="center", va="top", alpha=0.85, zorder=2)

    # ── Step 5: reference anchors — domain-aware, single call ───────────
    # ref_points(domain) returns only what's appropriate for this domain.
    # strain/eeg/shm → empty. finance → internal regimes. traffic → OU mid-k.
    # unknown → all cross_domain_ref anchors.
    legend_handles = []
    anchors = ref_points(domain)
    for rx, ry, rc, rl in anchors:
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

    # Threshold line legend entries (always shown)
    ax.plot([], [], color="#ffd54f", lw=1, ls="--",
            label=f"reliable ρ*>{rmin}")
    ax.plot([], [], color="#555555", lw=1, ls="--",
            label=f"degraded ρ*<{dmax}")

    ax.set_xlabel("ρ* (mean absolute rolling correlation)")
    ax.set_ylabel("dρ (mean correlation variance)")
    ax.set_title(title, color=_WHITE, fontsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Strain split-panel phase portrait
# ---------------------------------------------------------------------------

def phase_portrait_split(result, title: str | None = None) -> "Figure":
    """
    Two-panel phase portrait: within-device (left) and cross-device (right).

    Only meaningful for strain domain results that have result.pair_groups set.
    Falls back to single-panel phase_portrait() if pair_groups is missing.

    Reads result.domain for anchor selection — same contract as phase_portrait().
    """
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from .core import OPERATING_ENVELOPE
    from .analysis.regimes import ref_points

    if not getattr(result, "pair_groups", None):
        return phase_portrait(result, title=title)

    env  = OPERATING_ENVELOPE
    rmin = env["reliable_rho_min"]
    dmax = env["degraded_rho_max"]
    locked_rho = 0.70

    s      = result.summary_dict()
    bg_str = f"λ₁/λ₂={s['band_gap']:.2f}×   r_eff corr={s['reff_corr']:.3f}"
    title  = title or f"SFE — Phase Portrait   |   {bg_str}"

    domain       = getattr(result, "domain", None)
    anchors      = ref_points(domain)
    within_pairs = result.pair_groups.get("within", [])
    cross_pairs  = result.pair_groups.get("cross",  result.pairs)

    fig, (ax_w, ax_c) = plt.subplots(
        1, 2, figsize=(16, 6), facecolor=_BG, sharey=True,
    )
    fig.suptitle(title, color=_WHITE, fontsize=10)

    for ax, group, panel_title in [
        (ax_w, within_pairs, "Within-device (same rosette)"),
        (ax_c, cross_pairs,  "Cross-device (structural coupling)"),
    ]:
        _style_ax(ax)

        # ── live data ──────────────────────────────────────────────────
        for k, p in enumerate(group):
            ax.scatter(p["rho_star"], p["drho_mean"],
                       color=_COLORS[k % len(_COLORS)],
                       marker="o", s=60, zorder=5, alpha=0.85)

        ax.autoscale(enable=True)
        ax.set_ylim(bottom=0)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        x_pad = (x1 - x0) * 0.12
        y_pad = (y1 - y0) * 0.25
        x0 = min(x0 - x_pad, -0.05)
        x1 = max(x1 + x_pad,  1.05)
        y1 = y1 + y_pad
        ax.set_xlim(x0, x1)
        ax.set_ylim(0,  y1)
        ax.autoscale(enable=False)

        # ── dρ split ───────────────────────────────────────────────────
        live_drho_max = max(p["drho_mean"] for p in group) if group else y1
        drho_split    = live_drho_max * 0.40

        # ── regions ────────────────────────────────────────────────────
        ax.axvspan(x0,         dmax,       color="#3a1a1a", alpha=0.30, zorder=0)
        ax.axvspan(dmax,       locked_rho, color="#1a2a3a", alpha=0.30, zorder=0)
        ax.axvspan(locked_rho, x1,         color="#1a3a1a", alpha=0.30, zorder=0)

        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle(
            (locked_rho, drho_split), x1 - locked_rho, y1 - drho_split,
            facecolor="#3a2a0a", alpha=0.45, zorder=0,
        ))

        ax.axhline(drho_split, xmin=(locked_rho - x0) / (x1 - x0),
                   color="#ffd54f", lw=0.7, ls=":", alpha=0.5, zorder=1)
        ax.axvline(dmax,       color="#555555", lw=0.8, ls="--", alpha=0.5, zorder=1)
        ax.axvline(rmin,       color="#ffd54f", lw=0.8, ls="--", alpha=0.6, zorder=1)
        ax.axvline(locked_rho, color="#888888", lw=0.8, ls="--", alpha=0.5, zorder=1)

        # ── region labels ──────────────────────────────────────────────
        label_y = y1 * 0.93
        for lx, ltext, lcolor in [
            (dmax / 2 + x0 / 2,        "Decoupled",  "#aa4444"),
            ((dmax + locked_rho) / 2,   "Active",     "#4488aa"),
            ((locked_rho + x1) / 2,     "Locked",     "#44aa66"),
        ]:
            ax.text(lx, label_y, ltext,
                    color=lcolor, fontsize=9, fontweight="bold",
                    ha="center", va="top", alpha=0.85, zorder=2)

        ax.text((locked_rho + x1) / 2, y1 * 0.70,
                "Transitioning",
                color="#cc8833", fontsize=9, fontweight="bold",
                ha="center", va="top", alpha=0.85, zorder=2)

        # ── anchors ────────────────────────────────────────────────────
        legend_handles = [
            mlines.Line2D([], [], color="#ffd54f", lw=1, ls="--",
                          label=f"reliable ρ*>{rmin}"),
            mlines.Line2D([], [], color="#555555", lw=1, ls="--",
                          label=f"degraded ρ*<{dmax}"),
        ]
        for rx, ry, rc, rl in anchors:
            ax.scatter(rx, ry, marker="*", s=130, color=rc, zorder=3,
                       clip_on=False, alpha=0.90)
            legend_handles.append(
                mlines.Line2D([], [], marker="*", color=rc, lw=0,
                              markersize=9, label=rl)
            )

        ax.legend(handles=legend_handles, fontsize=7,
                  facecolor="#1a1a1a", labelcolor=_TEXT,
                  loc="upper right")
        ax.set_xlabel("ρ* (mean absolute rolling correlation)")
        ax.set_title(panel_title, color=_WHITE, fontsize=9)

    ax_w.set_ylabel("dρ (mean correlation variance)")
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
    import matplotlib.pyplot as plt

    pairs = result.pairs[:max_pairs]
    n     = len(pairs)
    title = title or "SFE — Timeseries"

    fig, axes = plt.subplots(n, 2, figsize=(14, 2.2 * n), facecolor=_BG)
    fig.suptitle(title, color=_WHITE, fontsize=10, y=0.98)
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
        ax_rho.axhline( 0.45, color="#38b000", lw=0.8, ls="--", alpha=0.5)
        ax_rho.axhline( 0.0,  color=_GRID,     lw=0.5)
        ax_rho.axhline(-0.45, color="#38b000", lw=0.8, ls="--", alpha=0.5)
        ax_rho.set_ylim(-1.05, 1.05)
        ax_rho.set_ylabel(f"{p['label']}\nρ", color=_TEXT, fontsize=8)

        _style_ax(ax_drho)
        ax_drho.plot(t, drho_, color="#ffd60a", lw=0.8, alpha=0.9)
        ax_drho.axhline(p["drho_mean"], color=_TEXT, lw=0.8, ls="--", alpha=0.5,
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
    title = title or f"SFE — Eigenspectrum   |   λ₁/λ₂={bg:.2f}×   r_eff corr={rc:.3f}"

    fig, (ax_spec, ax_reff) = plt.subplots(1, 2, figsize=(12, 5), facecolor=_BG)
    fig.suptitle(title, color=_WHITE, fontsize=10)
    _style_ax(ax_spec)
    _style_ax(ax_reff)

    cov = np.cov(result.data.T)
    if cov.ndim < 2:
        cov = np.array([[float(cov)]])
    ev  = np.sort(np.linalg.eigvalsh(cov))[::-1]
    evn = ev / ev.sum()
    N   = len(ev)

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
    ax_reff.axhline(1.0,          color="#38b000", lw=1, ls="--", alpha=0.6,
                    label="r_eff=1 (rank collapse)")
    ax_reff.axhline(float(N),     color=_TEXT,     lw=1, ls="--", alpha=0.4,
                    label=f"r_eff={N} (isotropic)")
    ax_reff.axhline(np.nanmean(rj), color="#ffd60a", lw=1, ls=":", alpha=0.7,
                    label=f"joint mean={np.nanmean(rj):.3f}")
    ax_reff.axhline(rc,           color="#c77dff", lw=1, ls=":", alpha=0.7,
                    label=f"corrected={rc:.3f}")
    ax_reff.set_xlabel("window block")
    ax_reff.set_ylabel("r_eff joint")
    ax_reff.set_title("r_eff joint trajectory", color=_WHITE, fontsize=9)
    ax_reff.legend(fontsize=7, facecolor="#1a1a1a", labelcolor=_TEXT)

    fig.tight_layout()
    fig.sfe_band_gap  = bg
    fig.sfe_reff_corr = rc
    return fig


# ---------------------------------------------------------------------------
# Finance — Crisis overlay (background vs crisis, Branch A/B verdict)
# ---------------------------------------------------------------------------

def crisis_overlay(
    background,
    crisis,
    regime=None,
    title_prefix: str = "",
) -> "Figure":
    """
    Two-panel phase portrait: background (left) vs crisis window (right).

    Both panels share axes so the shift is visually immediate.
    Branch A/B verdict stamped on the crisis panel if regime provided.

    Parameters
    ----------
    background   : SFEResult   full-period result
    crisis       : SFEResult   crisis window result
    regime       : RegimeResult | None   from detect_regime()
    title_prefix : str
    """
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from matplotlib.patches import Rectangle
    from .core import OPERATING_ENVELOPE
    from .analysis.regimes import ref_points

    env        = OPERATING_ENVELOPE
    rmin       = env["reliable_rho_min"]
    dmax       = env["degraded_rho_max"]
    locked_rho = 0.70

    pfx   = f"{title_prefix} — " if title_prefix else ""
    label = getattr(regime, "label", "") if regime is not None else ""

    # ── shared axis limits ──────────────────────────────────────────────
    all_rho  = [p["rho_star"]  for p in background.pairs + crisis.pairs]
    all_drho = [p["drho_mean"] for p in background.pairs + crisis.pairs]

    x_pad = (max(all_rho)  - min(all_rho))  * 0.15 if all_rho  else 0.1
    y_pad = (max(all_drho) - min(all_drho)) * 0.25 if all_drho else 0.01

    x0 = min(min(all_rho)  - x_pad, -0.05)
    x1 = max(max(all_rho)  + x_pad,  1.05)
    y0 = 0.0
    y1 = max(all_drho) + y_pad if all_drho else 0.1

    fig, (ax_bg, ax_cr) = plt.subplots(
        1, 2, figsize=(14, 6), facecolor=_BG,
        sharey=True,
    )
    fig.suptitle(f"{pfx}Crisis Overlay", color=_WHITE, fontsize=10)

    def _draw_panel(ax, result, panel_title, annotate_regime=False, show_legend=True):
        _style_ax(ax)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

        live_drho_max = max(p["drho_mean"] for p in result.pairs) if result.pairs else y1
        drho_split    = live_drho_max * 0.40

        # Regions
        ax.axvspan(x0,         dmax,       color="#3a1a1a", alpha=0.30, zorder=0)
        ax.axvspan(dmax,       locked_rho, color="#1a2a3a", alpha=0.30, zorder=0)
        ax.axvspan(locked_rho, x1,         color="#1a3a1a", alpha=0.30, zorder=0)
        ax.add_patch(Rectangle(
            (locked_rho, drho_split), x1 - locked_rho, y1 - drho_split,
            facecolor="#3a2a0a", alpha=0.45, zorder=0,
        ))

        ax.axhline(drho_split, xmin=(locked_rho - x0) / (x1 - x0),
                   color="#ffd54f", lw=0.7, ls=":", alpha=0.5, zorder=1)
        ax.axvline(dmax,       color="#555555", lw=0.8, ls="--", alpha=0.5, zorder=1)
        ax.axvline(rmin,       color="#ffd54f", lw=0.8, ls="--", alpha=0.6, zorder=1)
        ax.axvline(locked_rho, color="#888888", lw=0.8, ls="--", alpha=0.5, zorder=1)

        # Region labels — placed in data coords to avoid clipping
        label_y = y1 * 0.93
        decoupled_x = (x0 + dmax) / 2
        for lx, ltext, lcolor in [
            (decoupled_x,                 "Decoupled",    "#aa4444"),
            ((dmax + locked_rho) / 2,     "Active",       "#4488aa"),
            ((locked_rho + x1) / 2,       "Locked",       "#44aa66"),
        ]:
            ax.text(lx, label_y, ltext, color=lcolor, fontsize=9,
                    fontweight="bold", ha="center", va="top",
                    alpha=0.85, zorder=2, clip_on=False)
        ax.text((locked_rho + x1) / 2, y1 * 0.70,
                "Transitioning", color="#cc8833", fontsize=9,
                fontweight="bold", ha="center", va="top",
                alpha=0.85, zorder=2)

        # Dots
        for k, p in enumerate(result.pairs):
            ax.scatter(p["rho_star"], p["drho_mean"],
                       color=_COLORS[k % len(_COLORS)],
                       s=55, alpha=0.88, zorder=5)

        # Reference anchors + legend — only on background panel
        domain  = getattr(result, "domain", None)
        anchors = ref_points(domain)
        for rx, ry, rc, rl in anchors:
            ax.scatter(rx, ry, marker="*", s=140, color=rc,
                       zorder=4, alpha=0.90, clip_on=False)

        if show_legend:
            legend_handles = [
                mlines.Line2D([], [], color="#ffd54f", lw=1, ls="--",
                              label=f"reliable ρ*>{rmin}"),
                mlines.Line2D([], [], color="#555555", lw=1, ls="--",
                              label=f"degraded ρ*<{dmax}"),
            ]
            for rx, ry, rc, rl in anchors:
                legend_handles.append(
                    mlines.Line2D([], [], marker="*", color=rc, lw=0,
                                  markersize=8, label=rl)
                )
            ax.legend(handles=legend_handles, fontsize=7,
                      facecolor="#1a1a1a", labelcolor=_TEXT,
                      loc="upper left", framealpha=0.7)

        # Stats
        rho_mean = np.mean([p["rho_star"] for p in result.pairs])
        n_rel    = sum(p["zone"] == "reliable" for p in result.pairs)
        s        = result.summary_dict()
        ax.text(0.97, 0.03,
                f"ρ* mean={rho_mean:.3f}\n"
                f"reliable={n_rel}/{len(result.pairs)}\n"
                f"λ₁/λ₂={s['band_gap']:.2f}×\n"
                f"r_eff={s['reff_corr']:.3f}",
                transform=ax.transAxes, color=_TEXT, fontsize=7,
                va="bottom", ha="right",
                bbox=dict(facecolor="#0d0d0d", alpha=0.7, pad=3))

        # Branch verdict stamp on crisis panel
        if annotate_regime and label:
            branch = getattr(regime, "branch", "")
            color  = ("#ff4444" if "A" in branch else
                      "#ff8c00" if "B" in branch else "#888888")
            short  = (f"Branch {branch} ✓" if branch in ("A", "B")
                      else "SILENT")
            ax.text(0.50, 0.97, short,
                    transform=ax.transAxes, color=color,
                    fontsize=11, fontweight="bold",
                    ha="center", va="top", zorder=10,
                    bbox=dict(facecolor="#0d0d0d", alpha=0.75,
                              edgecolor=color, pad=4))

        ax.set_xlabel("ρ* (mean absolute rolling correlation)",
                      color=_TEXT, fontsize=8)
        ax.set_title(panel_title, color=_WHITE, fontsize=9)

    # Date range labels if timestamps available
    def _date_label(result):
        ts = getattr(result, "timestamps", None)
        if ts is not None and hasattr(ts, "__len__") and len(ts) >= 2:
            try:
                return f"{str(ts[0])[:10]} → {str(ts[-1])[:10]}"
            except Exception:
                pass
        return ""

    bg_label = f"Background  {_date_label(background)}"
    cr_label = f"Crisis window  {_date_label(crisis)}"

    _draw_panel(ax_bg, background, bg_label, annotate_regime=False, show_legend=True)
    _draw_panel(ax_cr, crisis,     cr_label, annotate_regime=True,  show_legend=False)

    ax_bg.set_ylabel("dρ (mean correlation variance)", color=_TEXT, fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# All figures
# ---------------------------------------------------------------------------

def all_figures(result, title_prefix: str = "") -> dict:
    """
    Generate all three standard figures.

    For strain domain results with pair_groups set, phase_portrait uses
    the split-panel variant automatically.

    Parameters
    ----------
    result       : SFEResult
    title_prefix : str

    Returns
    -------
    dict with keys: "phase_portrait", "timeseries", "eigenspectrum"
    """
    pfx    = f"{title_prefix} — " if title_prefix else ""
    domain = getattr(result, "domain", None)

    # Strain gets the split panel; everything else gets the single panel
    if domain == "strain" and getattr(result, "pair_groups", None):
        pp = phase_portrait_split(result, title=f"{pfx}Phase Portrait")
    else:
        pp = phase_portrait(result, title=f"{pfx}Phase Portrait")

    figs = {
        "phase_portrait" : pp,
        "timeseries"     : timeseries(result,    title=f"{pfx}Timeseries"),
        "eigenspectrum"  : eigenspectrum(result, title=f"{pfx}Eigenspectrum"),
    }

    # Finance — crisis overlay if crisis attached
    if domain == "finance" and getattr(result, "crisis", None) is not None:
        from .connectors.finance import detect_regime
        regime = detect_regime(result, result.crisis)
        figs["crisis_overlay"] = crisis_overlay(
            result, result.crisis, regime,
            title_prefix=title_prefix,
        )

    return figs
