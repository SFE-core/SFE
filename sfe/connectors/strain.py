# -*- coding: utf-8 -*-
"""sfe/connectors/strain.py — Strain rosette domain connector."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..connect import from_array, SFEResult

__all__ = ["from_strain_csv", "diurnal_breakdown", "print_diurnal", "strain_figures"]

def _parse_label(col):
    if ":" in col:
        parts = col.split(":", 1)
        return parts[0].strip(), parts[1].strip()
    return col, col

def _group_pairs(pairs):
    within, cross = [], []
    for p in pairs:
        label = p["label"]
        if "-" not in label: cross.append(p); continue
        left, right = label.split("-", 1)
        dev_l, _ = _parse_label(left)
        dev_r, _ = _parse_label(right)
        (within if dev_l == dev_r else cross).append(p)
    return {"within": within, "cross": cross}

def from_strain_csv(path, W=60, delimiter=","):
    import pandas as _pd
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(f"File not found: {path}")
    skip = 0; sfreq = 1.0
    with path.open(encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if stripped == "DATA_START": skip = i + 1; break
            if "SampleRate" in line and "Hz" in line:
                try: sfreq = float(stripped.split(",")[1].replace("Hz","").strip())
                except: pass
    df = _pd.read_csv(path, skiprows=skip, index_col=0,
                      encoding="utf-8", delimiter=delimiter)
    try:
        df.index = _pd.to_datetime(df.index, format="mixed")
    except Exception:
        pass  # keep as-is if timestamps not parseable
    numeric = df.select_dtypes(include="number").dropna()
    labels = list(numeric.columns); data = numeric.values.astype(float)
    timestamps = numeric.index if hasattr(numeric.index, "hour") else None
    if data.shape[1] < 2: raise ValueError(f"Need >= 2 numeric columns, got {data.shape[1]}.")
    print(f"Loaded strain CSV: {path.name}")
    print(f"  Shape    : {data.shape[0]} rows x {data.shape[1]} channels")
    print(f"  Channels : {labels}")
    print(f"  sfreq    : {sfreq} Hz  |  W={W} samples")
    result = from_array(data, W=W, labels=labels)
    result.sfreq = sfreq; result.timestamps = timestamps
    result.devices = sorted(set(_parse_label(l)[0] for l in labels))
    result.pair_groups = _group_pairs(result.pairs)
    _print_strain_summary(result)
    return result

def _print_strain_summary(result):
    groups = result.pair_groups
    within = groups["within"]; cross = groups["cross"]
    def _mean(lst, key):
        vals = [p[key] for p in lst]
        return float(np.mean(vals)) if vals else float("nan")
    print(); print("=" * 68)
    print("  STRAIN ROSETTE SFE SUMMARY"); print("=" * 68)
    print(f"  Channels : {result.N}   Pairs : {len(result.pairs)}   Devices : {result.devices}")
    print(f"  Within-device: {len(within)}   Cross-device: {len(cross)}")
    print(f"  band gap lambda1/lambda2: {result.band_gap:.3f}x")
    print(f"  r_eff corrected:          {result.reff_corr:.4f}")
    print()
    for group_label, group in [("WITHIN-DEVICE", within), ("CROSS-DEVICE", cross)]:
        if not group: continue
        reliable = sum(1 for p in group if p["zone"] == "reliable")
        print(f"  -- {group_label} --")
        print(f"  Mean rho*  : {_mean(group, 'rho_star'):.4f}")
        print(f"  Mean drho  : {_mean(group, 'drho_mean'):.6f}")
        print(f"  Mean r_eff : {_mean(group, 'reff_mean'):.4f}")
        print(f"  Reliable   : {reliable}/{len(group)}"); print()
    cov = np.cov(result.data.T)
    if cov.ndim < 2: cov = np.array([[float(cov)]])
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]; evn = ev / ev.sum()
    print(f"  -- TOP 5 PAIRS BY rho* --")
    print(f"  {'Pair':<36} {'rho*':>6} {'drho':>10} {'r_eff':>7}  Zone")
    print("  " + "-" * 64)
    for p in result.pairs[:5]:
        print(f"  {p['label']:<36} {p['rho_star']:>6.4f} "
              f"{p['drho_mean']:>10.6f} {p['reff_mean']:>7.4f}  {p['zone']}")
    print(); print("  Experimental. Not financial, medical, or engineering advice.")
    print("=" * 68); print()

def diurnal_breakdown(result, pair_idx=0):
    if not hasattr(result, "timestamps") or result.timestamps is None:
        raise AttributeError("result.timestamps not found. Use from_strain_csv().")
    p = result.pairs[pair_idx]; rho = p["rho"]; drho = p["drho"]; ts = result.timestamps
    if len(ts) != len(rho): raise ValueError(f"Timestamp length ({len(ts)}) != rho length ({len(rho)}).")
    hours = ts.hour if hasattr(ts, "hour") else np.array([t.hour for t in ts])
    breakdown = []
    for h in range(24):
        mask = hours == h
        if mask.sum() == 0:
            breakdown.append({"hour": h, "rho_star": float("nan"), "drho": float("nan"), "n_samples": 0})
        else:
            breakdown.append({"hour": h, "rho_star": float(np.abs(rho[mask]).mean()),
                               "drho": float(drho[mask].mean()), "n_samples": int(mask.sum())})
    return breakdown

def print_diurnal(breakdown, pair_label=""):
    print(f"\n  -- DIURNAL{' -- ' + pair_label if pair_label else ''} --")
    for row in breakdown:
        if row["n_samples"] == 0: continue
        print(f"  {row['hour']:>02d}:00  drho={row['drho']:.6f}  rho*={row['rho_star']:.4f}  n={row['n_samples']}")

def strain_figures(result, title_prefix=""):
    import matplotlib.pyplot as plt
    from ..figures import all_figures
    pfx = f"{title_prefix} -- " if title_prefix else "Strain Rosette -- "
    figs = all_figures(result, title_prefix=title_prefix or "Strain Rosette")
    N = result.N; labels = result.labels
    matrix = np.full((N, N), np.nan); np.fill_diagonal(matrix, 1.0)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    for p in result.pairs:
        parts = p["label"].split("-", 1)
        if len(parts) == 2 and parts[0] in label_to_idx and parts[1] in label_to_idx:
            i = label_to_idx[parts[0]]; j = label_to_idx[parts[1]]
            matrix[i, j] = matrix[j, i] = p["rho_star"]
    fig_hm, ax = plt.subplots(figsize=(8, 7), facecolor="#0d0d0d")
    ax.set_facecolor("#111")
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="rho*").ax.yaxis.label.set_color("#aaa")
    for dev in result.devices:
        idxs = [i for i, l in enumerate(labels) if _parse_label(l)[0] == dev]
        if len(idxs) >= 2:
            lo, hi = min(idxs) - 0.5, max(idxs) + 0.5
            rect = plt.Rectangle((lo, lo), hi-lo, hi-lo, fill=False,
                                  edgecolor="#ffd60a", lw=1.5, linestyle="--")
            ax.add_patch(rect)
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7, color="#aaa")
    ax.set_yticklabels(labels, fontsize=7, color="#aaa")
    ax.set_title(f"{pfx}rho* Heatmap (dashed = within-device)", color="#fff", fontsize=9)
    fig_hm.tight_layout(); figs["heatmap"] = fig_hm
    try:
        breakdown = diurnal_breakdown(result, pair_idx=0)
        top_label = result.pairs[0]["label"]
        hours = [r["hour"] for r in breakdown if r["n_samples"] > 0]
        drho_vals = [r["drho"] for r in breakdown if r["n_samples"] > 0]
        rho_vals  = [r["rho_star"] for r in breakdown if r["n_samples"] > 0]
        fig_d, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), facecolor="#0d0d0d", sharex=True)
        for ax in (ax1, ax2):
            ax.set_facecolor("#111"); ax.tick_params(colors="#aaa"); ax.grid(color="#1e1e1e", lw=0.5)
            for sp in ax.spines.values(): sp.set_edgecolor("#1e1e1e")
        ax1.plot(hours, drho_vals, color="#ffd60a", lw=1.5, marker="o", ms=4)
        ax1.set_ylabel("drho", color="#aaa")
        ax2.plot(hours, rho_vals, color="#00b4d8", lw=1.5, marker="o", ms=4)
        ax2.set_ylabel("rho*", color="#aaa"); ax2.set_xlabel("Hour (UTC)", color="#aaa")
        fig_d.suptitle(f"{pfx}Diurnal -- {top_label}", color="#fff", fontsize=9)
        fig_d.tight_layout(); figs["diurnal"] = fig_d
    except Exception: pass
    return figs
