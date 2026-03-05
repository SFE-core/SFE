# -*- coding: utf-8 -*-
"""sfe/connectors/strain.py — Strain rosette domain connector.

Usage (interactive, default):
    result = from_strain_csv("mydata.csv")
    # Detects label format, suggests W from sfreq, asks for confirmation.

Usage (automated / scripted):
    result = from_strain_csv("mydata.csv", auto=True)
    # Prints detected settings and runs immediately without prompting.
    # Also triggered by env var SFE_AUTO=1.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..connect import from_array, SFEResult
from ..w_select import suggest_W

__all__ = ["from_strain_csv", "diurnal_breakdown", "print_diurnal", "strain_figures"]


# ---------------------------------------------------------------------------
# Label format detection
# ---------------------------------------------------------------------------

_SEPARATORS = [":", "_", "-"]


def _detect_label_format(labels: list[str]) -> tuple[str | None, str]:
    """
    Detect the separator used to split device from gauge in column labels.

    Tries ':', '_', '-' in order. A separator is accepted when it appears
    in *all* labels and produces at least two distinct device prefixes
    (otherwise there is no meaningful device grouping).

    Returns
    -------
    (separator, description)
        separator   : str | None   None = no device grouping detected
        description : str          human-readable explanation
    """
    for sep in _SEPARATORS:
        if all(sep in lbl for lbl in labels):
            devices = {lbl.split(sep, 1)[0].strip() for lbl in labels}
            if len(devices) >= 2:
                return sep, f"device{sep}gauge  (separator='{sep}', devices={sorted(devices)})"

    return None, "no device grouping detected — treating all channels as one device"


def _parse_label(col: str, sep: str | None) -> tuple[str, str]:
    """Split a label into (device, gauge) using the detected separator."""
    if sep and sep in col:
        parts = col.split(sep, 1)
        return parts[0].strip(), parts[1].strip()
    return col, col


def _group_pairs(pairs: list[dict], sep: str | None) -> dict:
    within, cross = [], []
    for p in pairs:
        label = p["label"]
        if "-" not in label:
            cross.append(p)
            continue
        left, right = label.split("-", 1)
        dev_l, _ = _parse_label(left, sep)
        dev_r, _ = _parse_label(right, sep)
        (within if dev_l == dev_r else cross).append(p)
    return {"within": within, "cross": cross}


# ---------------------------------------------------------------------------
# Pre-run summary printer
# ---------------------------------------------------------------------------

def _print_prerun(
    path: str,
    labels: list[str],
    sep: str | None,
    sep_desc: str,
    sfreq: float,
    T: int,
    W: int,
    W_reasoning: str,
    W_source: str,
    n_within: int,
    n_cross: int,
    auto: bool,
) -> None:
    mode = "auto=True  (running on detected/default settings)" if auto else "interactive"
    print()
    print("=" * 68)
    print("  SFE STRAIN — PRE-RUN SUMMARY")
    print("=" * 68)
    print(f"  File         : {Path(path).name}")
    print(f"  Mode         : {mode}")
    print()
    print(f"  Label format : {sep_desc}")
    print(f"  Channels     : {labels}")
    print(f"  sfreq        : {sfreq} Hz")
    print(f"  T            : {T} samples")
    print()
    print(f"  W ({W_source:<8}) : {W} samples  —  {W_reasoning}")
    print()
    print(f"  Pairs (est.) : {len(labels) * (len(labels)-1) // 2} total  "
          f"({n_within} within-device, {n_cross} cross-device)")
    print("=" * 68)


def _estimate_pair_groups(labels: list[str], sep: str | None) -> tuple[int, int]:
    """Estimate within/cross pair counts before running the instrument."""
    N = len(labels)
    n_total = N * (N - 1) // 2
    if sep is None:
        return 0, n_total

    within = 0
    for i in range(N):
        for j in range(i + 1, N):
            dev_i, _ = _parse_label(labels[i], sep)
            dev_j, _ = _parse_label(labels[j], sep)
            if dev_i == dev_j:
                within += 1
    return within, n_total - within


# ---------------------------------------------------------------------------
# Main connector
# ---------------------------------------------------------------------------

def from_strain_csv(
    path: str,
    W: int | None = None,
    delimiter: str = ",",
    auto: bool | None = None,
) -> SFEResult:
    """
    Load a strain rosette CSV and run SFE.

    Parameters
    ----------
    path      : str        Path to CSV file. Supports a DATA_START sentinel
                           and optional SampleRate header (see below).
    W         : int | None Rolling window in samples.
                           If None, auto-selected from sfreq via suggest_W().
    delimiter : str        Column delimiter (default ',').
    auto      : bool       If True, skip the confirmation prompt and run
                           immediately on detected/default settings.
                           Defaults to False (interactive).
                           Also set via env var SFE_AUTO=1.

    CSV format
    ----------
    The connector supports two layouts:

    1. Annotated header (your recording device format):
         SampleRate, 1Hz
         ...
         DATA_START
         timestamp, ch1, ch2, ...
         2024-07-14 05:00:00, 0.1, 0.2, ...

    2. Plain CSV (any other tool):
         timestamp, ch1, ch2, ...     ← or just ch1, ch2 with no timestamp
         2024-07-14 05:00:00, 0.1, ...

    Label format
    ------------
    The connector auto-detects the device/gauge separator in column names.
    Supported conventions (in priority order):
        device:gauge   →  50423:ch0, 50437:ch0
        device_gauge   →  DEV1_0deg, DEV1_45deg
        device-gauge   →  A-0, A-45, B-0

    If no consistent separator is found, all channels are treated as a
    single device (within-device pairs = 0, all pairs are cross-device).
    """
    import pandas as _pd

    # Resolve auto flag — env var overrides default, explicit arg overrides env var
    if auto is None:
        auto = os.environ.get("SFE_AUTO", "0").strip() in ("1", "true", "True", "yes")

    path = str(path)
    fpath = Path(path)
    if not fpath.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # ------------------------------------------------------------------ parse header
    skip = 0
    sfreq = 1.0
    with fpath.open(encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if stripped == "DATA_START":
                skip = i + 1
                break
            if "SampleRate" in line and "Hz" in line:
                try:
                    sfreq = float(stripped.split(",")[1].replace("Hz", "").strip())
                except Exception:
                    pass

    # ------------------------------------------------------------------ load data
    df = _pd.read_csv(
        fpath, skiprows=skip, index_col=0,
        encoding="utf-8", delimiter=delimiter,
    )
    try:
        df.index = _pd.to_datetime(df.index, format="mixed")
    except Exception:
        pass  # keep as-is if timestamps not parseable

    numeric = df.select_dtypes(include="number").dropna()
    if numeric.shape[1] < 2:
        raise ValueError(
            f"Need ≥2 numeric columns after loading, got {numeric.shape[1]}. "
            f"Columns found: {list(df.columns)}"
        )

    labels = list(numeric.columns)
    data = numeric.values.astype(float)
    T = data.shape[0]
    timestamps = numeric.index if hasattr(numeric.index, "hour") else None

    # ------------------------------------------------------------------ label detection
    sep, sep_desc = _detect_label_format(labels)
    n_within, n_cross = _estimate_pair_groups(labels, sep)

    # ------------------------------------------------------------------ W selection
    if W is not None:
        W_use = W
        W_source = "explicit"
        W_reasoning = "user-provided"
    else:
        suggestion = suggest_W(sfreq=sfreq, T=T, domain="strain")
        W_use = suggestion.recommended_W
        W_source = "auto"
        W_reasoning = suggestion.reasoning

    # ------------------------------------------------------------------ pre-run summary + confirm
    _print_prerun(
        path=path,
        labels=labels,
        sep=sep,
        sep_desc=sep_desc,
        sfreq=sfreq,
        T=T,
        W=W_use,
        W_reasoning=W_reasoning,
        W_source=W_source,
        n_within=n_within,
        n_cross=n_cross,
        auto=auto,
    )

    if not auto:
        print()
        answer = input("  Proceed with these settings? [y/n]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("  Aborted.")
            return None
        print()

    # ------------------------------------------------------------------ run
    print(f"  Running SFE on {fpath.name} ...")

    result = from_array(data, W=W_use, labels=labels)
    result.sfreq = sfreq
    result.timestamps = timestamps
    result.devices = sorted({_parse_label(l, sep)[0] for l in labels})
    result.pair_groups = _group_pairs(result.pairs, sep)

    # Note: if f(N) over-corrected (reff_corr < 1), core._run() already
    # fell back to the raw joint mean. result.reff_corr_fallback will be True.
    if getattr(result, "reff_corr_fallback", False):
        print(f"\n  ⚠  f(N) over-correction detected at N={result.N}: "
              f"raw joint mean ({result.reff_corr:.4f}) used instead of corrected value. "
              f"See SFE-11 Open Problem 3.")

    _print_strain_summary(result, sep=sep)
    return result


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_strain_summary(result: SFEResult, sep: str | None = None) -> None:
    groups = result.pair_groups
    within = groups["within"]
    cross = groups["cross"]

    def _mean(lst, key):
        vals = [p[key] for p in lst]
        return float(np.mean(vals)) if vals else float("nan")

    print()
    print("=" * 68)
    print("  STRAIN ROSETTE SFE SUMMARY")
    print("=" * 68)
    print(f"  Channels : {result.N}   Pairs : {len(result.pairs)}   "
          f"Devices : {result.devices}")
    print(f"  Within-device : {len(within)}   Cross-device : {len(cross)}")
    print(f"  band gap lambda1/lambda2 : {result.band_gap:.3f}x")
    print(f"  r_eff corrected          : {result.reff_corr:.4f}")
    print()

    for group_label, group in [("WITHIN-DEVICE", within), ("CROSS-DEVICE", cross)]:
        if not group:
            continue
        reliable = sum(1 for p in group if p["zone"] == "reliable")
        print(f"  -- {group_label} --")
        print(f"  Mean rho*  : {_mean(group, 'rho_star'):.4f}")
        print(f"  Mean drho  : {_mean(group, 'drho_mean'):.6f}")
        print(f"  Mean r_eff : {_mean(group, 'reff_mean'):.4f}")
        print(f"  Reliable   : {reliable}/{len(group)}")
        print()

    print(f"  -- TOP 5 PAIRS BY rho* --")
    print(f"  {'Pair':<36} {'rho*':>6} {'drho':>10} {'r_eff':>7}  Zone")
    print("  " + "-" * 64)
    for p in result.pairs[:5]:
        print(f"  {p['label']:<36} {p['rho_star']:>6.4f} "
              f"{p['drho_mean']:>10.6f} {p['reff_mean']:>7.4f}  {p['zone']}")

    print()
    print("  Experimental. Not financial, medical, or engineering advice.")
    print("=" * 68)
    print()


# ---------------------------------------------------------------------------
# Diurnal analysis
# ---------------------------------------------------------------------------

def diurnal_breakdown(result: SFEResult, pair_idx: int = 0) -> list[dict]:
    if not hasattr(result, "timestamps") or result.timestamps is None:
        raise AttributeError(
            "result.timestamps not found. Timestamps must be parseable datetime "
            "values in the CSV index column."
        )
    p = result.pairs[pair_idx]
    rho = p["rho"]
    drho = p["drho"]
    ts = result.timestamps

    if len(ts) != len(rho):
        raise ValueError(
            f"Timestamp length ({len(ts)}) != rho length ({len(rho)}). "
            f"This can happen if the CSV index was not parsed as datetime."
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
                "hour": h,
                "rho_star": float(np.abs(rho[mask]).mean()),
                "drho": float(drho[mask].mean()),
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
    import matplotlib.pyplot as plt
    from ..figures import all_figures

    # Re-detect separator from labels — avoids storing on __slots__
    sep, _ = _detect_label_format(result.labels)
    pfx = f"{title_prefix} -- " if title_prefix else "Strain Rosette -- "

    # Get timeseries and eigenspectrum from standard figures; override phase_portrait below
    figs = all_figures(result, title_prefix=title_prefix or "Strain Rosette")

    # Phase portrait — split within/cross panels matching Colab layout
    from ..core import OPERATING_ENVELOPE
    import matplotlib.lines as mlines

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

    within_pairs = result.pair_groups.get("within", [])
    cross_pairs  = result.pair_groups.get("cross",  [])

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

    N = result.N
    labels = result.labels
    matrix = np.full((N, N), np.nan)
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

    # Heatmap
    fig_hm, ax = plt.subplots(figsize=(8, 7), facecolor="#0d0d0d")
    ax.set_facecolor("#111")
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    cb = plt.colorbar(im, ax=ax, label="rho*")
    cb.ax.yaxis.label.set_color("#aaa")

    for dev in result.devices:
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
    ax.set_title(
        f"{pfx}rho* Heatmap (dashed = within-device)",
        color="#fff", fontsize=9,
    )
    fig_hm.tight_layout()
    figs["heatmap"] = fig_hm

    # Diurnal (best pair)
    try:
        breakdown = diurnal_breakdown(result, pair_idx=0)
        top_label = result.pairs[0]["label"]
        hours = [r["hour"] for r in breakdown if r["n_samples"] > 0]
        drho_vals = [r["drho"] for r in breakdown if r["n_samples"] > 0]
        rho_vals = [r["rho_star"] for r in breakdown if r["n_samples"] > 0]

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
        ax1.set_ylabel("drho", color="#aaa")
        ax2.plot(hours, rho_vals, color="#00b4d8", lw=1.5, marker="o", ms=4)
        ax2.set_ylabel("rho*", color="#aaa")
        ax2.set_xlabel("Hour of recording (index, not UTC)", color="#aaa")

        fig_d.suptitle(f"{pfx}Diurnal -- {top_label}", color="#fff", fontsize=9)
        fig_d.tight_layout()
        figs["diurnal"] = fig_d
    except Exception:
        pass

    return figs
