# -*- coding: utf-8 -*-
"""
sfe/analysis/strain.py — Analysis helpers for strain domain SFEResults.

Connectors produce SFEResults. This module operates on them.

    from sfe.analysis.strain import diurnal_breakdown, print_diurnal, strain_figures
"""

from __future__ import annotations

import numpy as np

from ..connect import SFEResult

__all__ = [
    "diurnal_breakdown",
    "print_diurnal",
    "strain_figures",
]

_SEPARATORS = [":", "_", "-"]


def _detect_label_format(labels: list[str]) -> tuple[str | None, str]:
    for sep in _SEPARATORS:
        if all(sep in lbl for lbl in labels):
            devices = {lbl.split(sep, 1)[0].strip() for lbl in labels}
            if len(devices) >= 2:
                return sep, f"device{sep}gauge  (separator='{sep}', devices={sorted(devices)})"
    return None, "no device grouping detected"


def _parse_label(col: str, sep: str | None) -> tuple[str, str]:
    if sep and sep in col:
        parts = col.split(sep, 1)
        return parts[0].strip(), parts[1].strip()
    return col, col


# ---------------------------------------------------------------------------
# Diurnal
# ---------------------------------------------------------------------------

def diurnal_breakdown(result: SFEResult, pair_idx: int = 0) -> list[dict]:
    """
    Compute hourly mean dρ and ρ* for a single pair.

    Requires result.timestamps to be parseable datetime values
    (set automatically by from_strain_csv).
    """
    if not hasattr(result, "timestamps") or result.timestamps is None:
        raise AttributeError(
            "result.timestamps not found. Timestamps must be parseable datetime "
            "values in the CSV index column."
        )
    p    = result.pairs[pair_idx]
    rho  = p["rho"]
    drho = p["drho"]
    ts   = result.timestamps

    if len(ts) != len(rho):
        raise ValueError(
            f"Timestamp length ({len(ts)}) != rho length ({len(rho)}). "
            f"Check that the CSV index was parsed as datetime."
        )

    hours = ts.hour if hasattr(ts, "hour") else np.array([t.hour for t in ts])
    breakdown = []
    for h in range(24):
        mask = hours == h
        if mask.sum() == 0:
            breakdown.append({
                "hour": h, "rho_star": float("nan"),
                "drho": float("nan"), "n_samples": 0,
            })
        else:
            breakdown.append({
                "hour":      h,
                "rho_star":  float(np.abs(rho[mask]).mean()),
                "drho":      float(drho[mask].mean()),
                "n_samples": int(mask.sum()),
            })
    return breakdown


def print_diurnal(breakdown: list[dict], pair_label: str = "") -> None:
    print(f"\n  -- DIURNAL{' -- ' + pair_label if pair_label else ''} --")
    for row in breakdown:
        if row["n_samples"] == 0:
            continue
        print(f"  {row['hour']:>02d}:00  drho={row['drho']:.6f}  "
              f"rho*={row['rho_star']:.4f}  n={row['n_samples']}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def strain_figures(result: SFEResult, title_prefix: str = "") -> dict:
    """
    Generate strain-specific figures: phase portrait (within/cross split),
    heatmap, and diurnal profile.

    Returns dict with keys: "phase_portrait", "timeseries",
    "eigenspectrum", "heatmap", "diurnal" (if timestamps available).
    """
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from ..figures import all_figures
    from ..core import OPERATING_ENVELOPE

    sep, _ = _detect_label_format(result.labels)
    pfx    = f"{title_prefix} -- " if title_prefix else "Strain Rosette -- "
    figs   = all_figures(result, title_prefix=title_prefix or "Strain Rosette")

    env  = OPERATING_ENVELOPE
    rmin = env["reliable_rho_min"]
    dmax = env["degraded_rho_max"]
    s    = result.summary_dict()
    bg_str = f"λ₁/λ₂={s['band_gap']:.2f}×   r_eff joint={s['reff_joint_mean']:.4f}"

    _CROSS_DOMAIN_REFS = [
        (0.963, 0.004, "#ff6b35", "ETT HUFL-MUFL"),
        (0.831, 0.004, "#c77dff", "EEG C3-C4"),
        (0.426, 0.047, "#00b4d8", "METR-LA"),
        (0.850, 0.005, "#00e676", "OU high-k"),
        (0.450, 0.020, "#ffd54f", "OU mid-k"),
        (0.180, 0.054, "#bdbdbd", "OU low-k"),
    ]
    _COLORS = ["#ff6b35", "#00b4d8", "#ffd60a", "#c77dff",
               "#81c784", "#ff8a65", "#4fc3f7", "#aed581",
               "#f48fb1", "#80cbc4"]

    within_pairs = result.pair_groups.get("within", []) if result.pair_groups else []
    cross_pairs  = result.pair_groups.get("cross",  []) if result.pair_groups else result.pairs

    # Phase portrait — within/cross split panels
    fig_pp, (ax_w, ax_c) = plt.subplots(
        1, 2, figsize=(16, 6), facecolor="#0d0d0d", sharey=True,
    )
    fig_pp.suptitle(
        f"{pfx}Phase Portrait   |   {bg_str}",
        color="#fff", fontsize=10,
    )
    for ax, group, title in [
        (ax_w, within_pairs, "Within-device (same rosette)"),
        (ax_c, cross_pairs,  "Cross-device (structural coupling)"),
    ]:
        ax.set_facecolor("#111111")
        ax.tick_params(colors="#aaa", labelsize=8)
        ax.xaxis.label.set_color("#aaa")
        ax.yaxis.label.set_color("#aaa")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e1e1e")
        ax.grid(color="#1e1e1e", lw=0.5, alpha=0.6)
        ax.axvspan(rmin, 1.05, color="#1a3a1a", alpha=0.35, zorder=0)
        ax.axvspan(-0.05, dmax, color="#3a1a1a", alpha=0.25, zorder=0)
        ax.axvline(rmin, color="#ffd54f", lw=0.8, ls="--", alpha=0.6)
        ax.axvline(dmax, color="#aaaaaa", lw=0.8, ls="--", alpha=0.4)

        legend_handles = [
            mlines.Line2D([], [], color="#ffd54f", lw=1.2, ls="--",
                          label=f"reliable ρ*>{rmin}"),
            mlines.Line2D([], [], color="#aaaaaa", lw=1.2, ls="--",
                          label=f"degraded ρ*<{dmax}"),
        ]
        for rx, ry, rc, rl in _CROSS_DOMAIN_REFS:
            ax.scatter(rx, ry, marker="*", s=130, color=rc, zorder=2)
            legend_handles.append(
                mlines.Line2D([], [], marker="*", color=rc, lw=0,
                              markersize=9, label=rl)
            )
        for k, p in enumerate(group):
            ax.scatter(p["rho_star"], p["drho_mean"],
                       color=_COLORS[k % len(_COLORS)],
                       marker="o", s=60, zorder=4, alpha=0.85)
        ax.legend(handles=legend_handles, fontsize=7,
                  facecolor="#1a1a1a", labelcolor="#aaa", loc="upper right")
        ax.autoscale(enable=True)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("ρ* (mean absolute rolling correlation)")
        ax.set_title(title, color="#fff", fontsize=9)

    ax_w.set_ylabel("dρ (mean correlation variance)")
    fig_pp.tight_layout()
    figs["phase_portrait"] = fig_pp

    # Heatmap
    N            = result.N
    labels       = result.labels
    matrix       = np.full((N, N), np.nan)
    np.fill_diagonal(matrix, 1.0)
    label_to_idx = {l: i for i, l in enumerate(labels)}

    for p in result.pairs:
        parts = p["label"].split("-", 1)
        if (len(parts) == 2
                and parts[0] in label_to_idx
                and parts[1] in label_to_idx):
            i = label_to_idx[parts[0]]
            j = label_to_idx[parts[1]]
            matrix[i, j] = matrix[j, i] = p["rho_star"]

    fig_hm, ax = plt.subplots(figsize=(8, 7), facecolor="#0d0d0d")
    ax.set_facecolor("#111")
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    cb = plt.colorbar(im, ax=ax, label="rho*")
    cb.ax.yaxis.label.set_color("#aaa")

    devices = getattr(result, "devices", None) or []
    for dev in devices:
        idxs = [i for i, l in enumerate(labels)
                if _parse_label(l, sep)[0] == dev]
        if len(idxs) >= 2:
            lo, hi = min(idxs) - 0.5, max(idxs) + 0.5
            rect = plt.Rectangle(
                (lo, lo), hi - lo, hi - lo,
                fill=False, edgecolor="#ffd60a", lw=1.5, linestyle="--",
            )
            ax.add_patch(rect)

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7, color="#aaa")
    ax.set_yticklabels(labels, fontsize=7, color="#aaa")
    ax.set_title(f"{pfx}ρ* Heatmap (dashed = within-device)",
                 color="#fff", fontsize=9)
    fig_hm.tight_layout()
    figs["heatmap"] = fig_hm

    # Diurnal (only if timestamps are present)
    try:
        breakdown  = diurnal_breakdown(result, pair_idx=0)
        top_label  = result.pairs[0]["label"]
        hours      = [r["hour"]     for r in breakdown if r["n_samples"] > 0]
        drho_vals  = [r["drho"]     for r in breakdown if r["n_samples"] > 0]
        rho_vals   = [r["rho_star"] for r in breakdown if r["n_samples"] > 0]

        fig_d, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 6), facecolor="#0d0d0d", sharex=True,
        )
        for ax in (ax1, ax2):
            ax.set_facecolor("#111")
            ax.tick_params(colors="#aaa")
            ax.grid(color="#1e1e1e", lw=0.5)
            for sp in ax.spines.values():
                sp.set_edgecolor("#1e1e1e")

        ax1.plot(hours, drho_vals, color="#ffd60a", lw=1.5, marker="o", ms=4)
        ax1.set_ylabel("dρ", color="#aaa")
        ax2.plot(hours, rho_vals,  color="#00b4d8", lw=1.5, marker="o", ms=4)
        ax2.set_ylabel("ρ*", color="#aaa")
        ax2.set_xlabel("Hour of recording (index, not UTC)", color="#aaa")
        fig_d.suptitle(f"{pfx}Diurnal -- {top_label}", color="#fff", fontsize=9)
        fig_d.tight_layout()
        figs["diurnal"] = fig_d
    except Exception:
        pass  # no timestamps — diurnal figure silently skipped

    return figs
