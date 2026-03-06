# -*- coding: utf-8 -*-
"""
sfe/analysis/segments.py — Phase-segmented analysis for SFE.

Splits a dataset into named time phases and runs SFE independently
on each segment. Each segment returns a normal SFEResult — nothing
new to learn downstream.

Contract
--------
- Input  : raw data array + timestamps + phase boundary list
- Output : list of (label, SFEResult) tuples, one per phase
- Failures: segments with too few clean rows warn and are skipped,
            never crash the run
- Domain : inherited from the caller — set result.domain after
           calling segment() if needed, or pass domain= directly

Typical usage
-------------
    from sfe.analysis.segments import segment, compare_portraits

    results = segment(
        data       = f_array,          # (T, N) float64
        timestamps = sdn_converted,    # (T,) datetime array
        phases     = [
            ("pre",      "2018-10-17", "2019-02-01"),
            ("event",    "2019-02-01", "2019-04-01"),
            ("retrofit", "2019-04-01", "2019-07-01"),
            ("post",     "2019-07-01", "2020-01-12"),
        ],
        W      = 24,
        domain = "shm",
        labels = mode_labels,
    )

    fig = compare_portraits(results, title="KW51 — Phase Analysis")
    fig.savefig("kw51_phases.png", dpi=150, bbox_inches="tight")

What this module does NOT do
-----------------------------
- Modify the connector     → connectors set result.domain
- Own figure styling       → compare_portraits delegates to figures.py
- Store state              → every call is stateless
- Know about KW51          → domain-agnostic, works for any domain
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np

from ..connect import from_array, SFEResult

__all__ = [
    "segment",
    "compare_portraits",
    "print_segment_summary",
]

# Minimum rows required after NaN cleaning for a segment to be usable.
# W+1 is the hard floor from connect._clean; we add headroom for
# at least 3 window blocks so r_eff trajectory is meaningful.
_MIN_ROWS_MULTIPLIER = 3


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def _parse_date(d) -> np.datetime64:
    """Accept str, datetime, np.datetime64, or pandas Timestamp."""
    if isinstance(d, np.datetime64):
        return d
    if isinstance(d, datetime):
        return np.datetime64(d)
    if isinstance(d, str):
        return np.datetime64(d)
    # pandas Timestamp
    try:
        return np.datetime64(d)
    except Exception:
        raise TypeError(
            f"Cannot parse date: {d!r}. "
            f"Pass a string ('2019-04-01'), datetime, or np.datetime64."
        )


def _to_datetime64_array(timestamps) -> np.ndarray:
    """
    Convert timestamps to a numpy datetime64 array.
    Handles: datetime objects, MATLAB datenums (float), np.datetime64.
    """
    arr = np.asarray(timestamps)

    if arr.dtype.kind == 'f':
        # MATLAB serial datenum — convert via Python datetime
        from datetime import timedelta
        converted = np.array([
            np.datetime64(
                datetime.fromordinal(int(dn))
                + timedelta(days=float(dn) % 1)
                - timedelta(days=366)
            )
            for dn in arr.ravel()
        ])
        return converted

    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.ravel()

    # Try object array of datetimes
    try:
        return np.array([np.datetime64(t) for t in arr.ravel()])
    except Exception as e:
        raise TypeError(
            f"Cannot convert timestamps to datetime64: {e}\n"
            f"Provide datetime objects, ISO strings, or MATLAB datenums."
        )


# ---------------------------------------------------------------------------
# Core segmentation
# ---------------------------------------------------------------------------

def segment(
    data:       np.ndarray,
    timestamps: np.ndarray,
    phases:     Sequence[tuple],
    W:          int,
    domain:     str | None = None,
    labels:     list[str] | None = None,
    verbose:    bool = True,
) -> list[tuple[str, SFEResult]]:
    """
    Run SFE independently on each named phase of the data.

    Parameters
    ----------
    data        : ndarray (T, N)   Raw numeric array (NaNs allowed).
    timestamps  : array-like (T,)  One timestamp per row. Accepts datetime
                                   objects, ISO strings, or MATLAB datenums.
    phases      : list of (label, start, end)
                  Each entry defines one phase. start/end are inclusive/exclusive
                  date boundaries. Accepts str ('2019-04-01'), datetime,
                  or np.datetime64.
    W           : int              Rolling window size (same for all phases).
    domain      : str | None       Set result.domain on each segment result.
                                   If None, result.domain stays None.
    labels      : list of str      Column labels. If None, auto-generated.
    verbose     : bool             Print per-phase loading summary.

    Returns
    -------
    list of (label: str, result: SFEResult)
        Only phases with sufficient clean data are included.
        Phases that fail or are too small are skipped with a warning.

    Notes
    -----
    Each SFEResult is independent — it has its own pairs, band_gap,
    reff_corr, and quality report. Nothing is shared across segments.
    Downstream code (figures, regimes, AI layer) works unchanged.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D (T, N), got shape {data.shape}.")

    T = data.shape[0]
    col_labels = labels or [f"col{k}" for k in range(data.shape[1])]
    ts = _to_datetime64_array(timestamps)

    if len(ts) != T:
        raise ValueError(
            f"timestamps length ({len(ts)}) != data rows ({T})."
        )

    min_rows = W * _MIN_ROWS_MULTIPLIER + 1
    results  = []

    for entry in phases:
        if len(entry) != 3:
            raise ValueError(
                f"Each phase must be (label, start, end), got: {entry!r}"
            )
        label, start, end = entry
        t_start = _parse_date(start)
        t_end   = _parse_date(end)

        mask = (ts >= t_start) & (ts < t_end)
        n_rows = mask.sum()

        if verbose:
            print(f"  Phase '{label}': {t_start} → {t_end}  "
                  f"({n_rows} rows before cleaning)")

        if n_rows == 0:
            print(f"  ⚠  Phase '{label}': no rows in range — skipped.")
            continue

        if n_rows < min_rows:
            print(f"  ⚠  Phase '{label}': only {n_rows} rows "
                  f"(need ≥{min_rows} for W={W}, 3 blocks) — skipped.")
            continue

        phase_data = data[mask]

        try:
            result = from_array(phase_data, W=W, labels=col_labels)
        except ValueError as e:
            print(f"  ⚠  Phase '{label}': SFE failed — {e} — skipped.")
            continue

        if domain is not None:
            result.domain = domain

        # Attach phase metadata as timestamps slice
        result.timestamps = ts[mask]

        if verbose:
            q = result.quality
            print(f"         → cleaned: {q.cleaned_shape}  "
                  f"pairs: {len(result.pairs)}  "
                  f"reliable: {sum(p['zone']=='reliable' for p in result.pairs)}  "
                  f"rho*_mean: {np.mean([p['rho_star'] for p in result.pairs]):.3f}")

        results.append((label, result))

    if not results:
        print("  ⚠  No phases produced valid results.")

    return results


# ---------------------------------------------------------------------------
# Comparison figure
# ---------------------------------------------------------------------------

def compare_portraits(
    segments:     list[tuple[str, SFEResult]],
    title:        str = "",
    share_axes:   bool = True,
) -> "matplotlib.figure.Figure":
    """
    Side-by-side phase portraits for all segments on shared axes.

    Each panel is one phase. Axes are shared so the visual shift
    between phases is immediately readable — dots moving left means
    decorrelation, dots moving up means instability.

    Parameters
    ----------
    segments    : output of segment()
    title       : figure suptitle
    share_axes  : if True, all panels share x and y limits (default True)

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from ..analysis.regimes import ref_points, OPERATING_ENVELOPE

    if not segments:
        raise ValueError("No segments to plot.")

    n     = len(segments)
    env   = OPERATING_ENVELOPE
    rmin  = env["reliable_rho_min"]
    dmin  = env["degraded_rho_max"]

    fig = plt.figure(figsize=(6 * n, 6), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(1, n, figure=fig, wspace=0.08)

    axes = []
    all_rho  = []
    all_drho = []

    # Collect global range first if sharing axes
    if share_axes:
        for _, result in segments:
            for p in result.pairs:
                all_rho.append(p["rho_star"])
                all_drho.append(p["drho_mean"])
        rho_max  = max(all_rho)  if all_rho  else 1.0
        drho_max = max(all_drho) if all_drho else 0.1
        x_lim = (-0.05, min(rho_max  * 1.12, 1.05))
        y_lim = (-drho_max * 0.05, drho_max * 1.15)

    for idx, (label, result) in enumerate(segments):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor("#111")
        ax.tick_params(colors="#aaa", labelsize=7)
        ax.grid(color="#1e1e1e", lw=0.5)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e1e1e")

        # Threshold lines
        pairs     = result.pairs
        drho_vals = [p["drho_mean"] for p in pairs]
        drho_split = max(drho_vals) * 0.40 if drho_vals else 0.05

        ax.axvline(rmin, color="#ffd60a", lw=1.0, ls="--", alpha=0.7)
        ax.axvline(dmin, color="#888",    lw=0.8, ls="--", alpha=0.5)
        ax.axhline(drho_split, color="#ffd60a", lw=0.6, ls=":", alpha=0.4)

        # Region labels
        for txt, x, y, col in [
            ("Decoupled",    0.02, 0.92, "#e05252"),
            ("Active",       0.30, 0.92, "#5b9bd5"),
            ("Locked",       0.80, 0.92, "#6abf69"),
            ("Transitioning",0.80, 0.50, "#ffd060"),
        ]:
            ax.text(x, y, txt, transform=ax.transAxes,
                    color=col, fontsize=7, alpha=0.5, ha="center")

        # Dots — color by rho_star zone
        cmap = plt.cm.get_cmap("tab20", max(len(pairs), 1))
        for i, p in enumerate(pairs):
            ax.scatter(p["rho_star"], p["drho_mean"],
                       color=cmap(i), s=40, alpha=0.85, zorder=3)

        # Reference anchors (domain-aware, usually empty for shm)
        domain = getattr(result, "domain", None)
        for ref in ref_points(domain):
            ax.scatter(ref["rho_star"], ref["drho"],
                       marker="*", s=120,
                       color=ref.get("color", "#fff"),
                       zorder=5, alpha=0.8)

        # Stats annotation
        rho_mean = np.mean([p["rho_star"] for p in pairs]) if pairs else 0
        n_rel    = sum(p["zone"] == "reliable" for p in pairs)
        q        = result.quality
        ax.text(0.03, 0.03,
                f"ρ* mean={rho_mean:.3f}\n"
                f"reliable={n_rel}/{len(pairs)}\n"
                f"T={q.cleaned_shape[0] if q.cleaned_shape else '?'}",
                transform=ax.transAxes,
                color="#aaa", fontsize=6.5, va="bottom",
                bbox=dict(facecolor="#0d0d0d", alpha=0.6, pad=2))

        ax.set_xlabel("ρ* (mean absolute rolling correlation)",
                      color="#aaa", fontsize=8)
        if idx == 0:
            ax.set_ylabel("dρ (mean correlation variance)",
                          color="#aaa", fontsize=8)
        else:
            ax.set_yticklabels([])

        ax.set_title(label, color="#fff", fontsize=9, pad=6)

        if share_axes:
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)

        axes.append(ax)

    suptitle = title or "Phase Portrait Comparison"
    fig.suptitle(suptitle, color="#fff", fontsize=10, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_segment_summary(segments: list[tuple[str, SFEResult]]) -> None:
    """Print a compact comparison table across all segments."""
    if not segments:
        print("  (no segments)")
        return

    print()
    print("=" * 72)
    print("  PHASE SEGMENT SUMMARY")
    print("=" * 72)
    print(f"  {'Phase':<14} {'T':>6} {'Pairs':>6} {'Reliable':>9} "
          f"{'ρ* mean':>9} {'dρ mean':>10} {'band_gap':>10}")
    print(f"  {'-'*68}")

    for label, r in segments:
        rho_vals  = [p["rho_star"]  for p in r.pairs]
        drho_vals = [p["drho_mean"] for p in r.pairs]
        n_rel     = sum(p["zone"] == "reliable" for p in r.pairs)
        T         = r.quality.cleaned_shape[0] if r.quality.cleaned_shape else 0
        print(f"  {label:<14} {T:>6} {len(r.pairs):>6} {n_rel:>9} "
              f"{np.mean(rho_vals):>9.4f} {np.mean(drho_vals):>10.6f} "
              f"{r.band_gap:>9.2f}×")

    print()
