# -*- coding: utf-8 -*-
"""
sfe/analysis/strain.py — Analysis helpers for strain domain SFEResults.

Connectors produce SFEResults. This module operates on them.

Responsibilities
----------------
- diurnal_breakdown()  : hourly mean dρ and ρ* for a single pair
- print_diurnal()      : console summary of the diurnal breakdown
- strain_figures()     : heatmap + diurnal figures, plus standard figures
                         from figures.py (phase portrait already handles
                         within/cross split via result.domain + pair_groups)

What this module does NOT do
-----------------------------
- Phase portrait logic  → figures.phase_portrait_split()
- Anchor/ref selection  → regimes.ref_points()
- Domain routing        → result.domain (set by connector)

The strain connector sets result.domain = "strain" and result.pair_groups.
Everything downstream reads those fields — nothing is passed manually.
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
    Generate all figures for a strain domain result.

    Standard figures (phase portrait, timeseries, eigenspectrum) come from
    figures.all_figures(result). The phase portrait automatically uses the
    split-panel variant because result.domain == "strain" and pair_groups
    is set by the connector.

    Additional strain-specific figures added here:
        heatmap  — ρ* matrix across all 9 sensors, within-device blocks marked
        diurnal  — hourly dρ and ρ* for the top pair (requires timestamps)

    Returns
    -------
    dict with keys: "phase_portrait", "timeseries", "eigenspectrum",
                    "heatmap", "diurnal" (if timestamps available)
    """
    import matplotlib.pyplot as plt
    from ..figures import all_figures

    pfx  = f"{title_prefix} -- " if title_prefix else "Strain Rosette -- "
    sep, _ = _detect_label_format(result.labels)

    # Standard figures — phase portrait split is automatic via result.domain
    figs = all_figures(result, title_prefix=title_prefix or "Strain Rosette")

    # ── Heatmap ────────────────────────────────────────────────────────────
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

    # ── Diurnal ────────────────────────────────────────────────────────────
    # Uses the top pair by rho_star (index 0 after connector sorts descending)
    try:
        breakdown = diurnal_breakdown(result, pair_idx=0)
        top_label = result.pairs[0]["label"]
        hours     = [r["hour"]     for r in breakdown if r["n_samples"] > 0]
        drho_vals = [r["drho"]     for r in breakdown if r["n_samples"] > 0]
        rho_vals  = [r["rho_star"] for r in breakdown if r["n_samples"] > 0]

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

        ax2.plot(hours, rho_vals, color="#00b4d8", lw=1.5, marker="o", ms=4)
        ax2.set_ylabel("ρ*", color="#aaa")
        ax2.set_xlabel("Hour of recording (index, not UTC)", color="#aaa")

        fig_d.suptitle(f"{pfx}Diurnal -- {top_label}", color="#fff", fontsize=9)
        fig_d.tight_layout()
        figs["diurnal"] = fig_d

    except Exception:
        pass  # no timestamps — diurnal silently skipped

    return figs