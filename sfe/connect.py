# -*- coding: utf-8 -*-
"""
sfe/connect.py — cleaning and connector layer.

[messy data] → _clean() → core → SFEResult → caller

Core never sees NaN, Inf, or wrong shapes. That is this module's job.
"""

from __future__ import annotations

import csv as _csv
from pathlib import Path

import numpy as np

from .core import pair_table, reff_joint, band_gap as _band_gap, reff_corrected as _reff_corrected, OPERATING_ENVELOPE

__all__ = [
    "from_dataframe", "from_array", "from_csv", "from_dict",
    "SFEResult", "print_summary",
]


# ---------------------------------------------------------------------------
# Data quality report
# ---------------------------------------------------------------------------

class DataQualityReport:
    """What _clean() found and what it did."""
    __slots__ = (
        "original_shape", "cleaned_shape",
        "columns_dropped", "rows_dropped", "inf_replaced", "nan_count",
        "warnings", "errors",
    )

    def __init__(self):
        self.original_shape  = None
        self.cleaned_shape   = None
        self.columns_dropped = []
        self.rows_dropped    = 0
        self.inf_replaced    = 0
        self.nan_count       = 0
        self.warnings        = []
        self.errors          = []

    def ok(self):
        return len(self.errors) == 0

    def __repr__(self):
        return (
            f"DataQualityReport(original={self.original_shape}, "
            f"cleaned={self.cleaned_shape}, rows_dropped={self.rows_dropped}, "
            f"cols_dropped={self.columns_dropped})"
        )


# ---------------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------------

def _clean(data: np.ndarray, labels: list[str],
           W: int) -> tuple[np.ndarray, list[str], DataQualityReport]:
    """
    Return a finite (T', N') float64 array safe to pass to core.

    Steps:
        1. Inf → NaN (vectorized).
        2. Drop all-NaN or constant columns (per-column pass, unavoidable).
        3. Drop rows with any NaN (vectorized row mask).
        4. Validate minimum shape.
    """
    q = DataQualityReport()
    q.original_shape = data.shape

    # 1. Inf → NaN
    inf_mask = np.isinf(data)
    q.inf_replaced = int(inf_mask.sum())
    if q.inf_replaced:
        data = data.copy()
        data[inf_mask] = np.nan
        q.warnings.append(f"{q.inf_replaced} Inf value(s) replaced with NaN.")

    q.nan_count = int(np.isnan(data).sum())

    # 2. Drop bad columns
    keep = []
    for k in range(data.shape[1]):
        valid = data[:, k][~np.isnan(data[:, k])]
        if len(valid) == 0:
            q.columns_dropped.append(labels[k])
            q.warnings.append(f"Column '{labels[k]}' is entirely NaN — dropped.")
        elif np.ptp(valid) < 1e-10:
            q.columns_dropped.append(labels[k])
            q.warnings.append(
                f"Column '{labels[k]}' is constant — dropped "
                f"(Pearson correlation undefined)."
            )
        else:
            keep.append(k)

    data   = data[:, keep]
    labels = [labels[k] for k in keep]

    # 3. Drop rows with any NaN
    row_ok = np.isfinite(data).all(axis=1)
    q.rows_dropped = int((~row_ok).sum())
    if q.rows_dropped:
        data = data[row_ok]
        q.warnings.append(
            f"{q.rows_dropped} row(s) with NaN/Inf removed "
            f"({len(data)} remain). "
            f"For gaps (e.g. market close), consider forward-filling first."
        )

    # 4. Validate
    if data.shape[1] < 2:
        q.errors.append(
            f"Need ≥2 valid columns after cleaning, got {data.shape[1]}. "
            f"Dropped: {q.columns_dropped}."
        )
    if data.shape[0] < W + 1:
        q.errors.append(
            f"Only {data.shape[0]} rows after cleaning; "
            f"W={W} needs ≥{W + 1}. Reduce W or provide more data."
        )

    q.cleaned_shape = data.shape
    return data, labels, q


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

class SFEResult:
    """
    Returned by every connector. Same structure regardless of input format.

    Attributes
    ----------
    pairs               list of dict   All N(N-1)/2 pairs, sorted by rho_star desc.
    reff_joint          ndarray        Joint effective rank, shape (T//W,).
    band_gap            float          Eigenspectrum band gap λ₁/λ₂ from global cov.
    reff_corr           float          reff_joint × f(N) correction, or raw joint mean
                                       if f(N) over-corrected (see reff_corr_fallback).
    reff_corr_fallback  bool           True if f(N) produced reff_corr < 1 (physically
                                       impossible) and the raw joint mean was used instead.
                                       Indicates the high band-gap / single-dominant-mode
                                       regime where f(N) does not apply.
                                       See SFE-11 Open Problem 3.
    W, N, T             int
    labels              list of str    Column labels after cleaning.
    data                ndarray        Cleaned data passed to core, shape (T, N).
    quality             DataQualityReport
    domain              str or None    Set by connectors (e.g. "strain"). Never set by core.
    """
    __slots__ = ("pairs", "reff_joint", "band_gap", "reff_corr",
                 "reff_corr_fallback",
                 "W", "N", "T", "labels", "data", "quality",
                 "domain",
                 "sfreq", "timestamps", "devices", "pair_groups",
                 "crisis")

    def __init__(self, pairs, reff_joint_series, band_gap_val,
                 reff_corr_val, reff_corr_fallback,
                 W, N, T, labels, data, quality):
        self.pairs              = pairs
        self.reff_joint         = reff_joint_series
        self.band_gap           = band_gap_val
        self.reff_corr          = reff_corr_val
        self.reff_corr_fallback = reff_corr_fallback
        self.W                  = W
        self.N                  = N
        self.T                  = T
        self.labels             = labels
        self.data               = data
        self.quality            = quality
        self.domain             = None   # set by connectors, never by core
        self.sfreq              = None
        self.timestamps         = None
        self.devices            = None
        self.pair_groups        = None
        self.crisis             = None   # set by finance connector after slice_window

    def reliable(self):
        """Pairs in the reliable zone (ρ* > 0.45)."""
        return [p for p in self.pairs if p["zone"] == "reliable"]

    def flagged(self):
        """Pairs with non-stationarity > 40%."""
        return [p for p in self.pairs if p["nonstationary_pct"] > 40.0]

    def summary_dict(self):
        rho_stars = [p["rho_star"] for p in self.pairs]
        drho_vals = [p["drho_mean"] for p in self.pairs]
        rj        = self.reff_joint
        return {
            "N":                   self.N,
            "T":                   self.T,
            "W":                   self.W,
            "n_pairs":             len(self.pairs),
            "n_reliable":          sum(p["zone"] == "reliable" for p in self.pairs),
            "n_flagged":           sum(p["nonstationary_pct"] > 40.0 for p in self.pairs),
            "rho_star_mean":       float(np.mean(rho_stars)) if rho_stars else float("nan"),
            "rho_star_max":        float(np.max(rho_stars))  if rho_stars else float("nan"),
            "drho_mean":           float(np.mean(drho_vals)) if drho_vals else float("nan"),
            "reff_joint_mean":     float(np.nanmean(rj)) if len(rj) else float("nan"),
            "band_gap":            self.band_gap,
            "reff_corr":           self.reff_corr,
            "reff_corr_fallback":  self.reff_corr_fallback,
            "rows_dropped":        self.quality.rows_dropped,
            "cols_dropped":        self.quality.columns_dropped,
            "warnings":            self.quality.warnings,
        }

    def __repr__(self):
        q = self.quality
        s = self.summary_dict()
        fallback_note = " [joint mean, f(N) suppressed]" if s["reff_corr_fallback"] else ""
        return (
            f"SFEResult(N={s['N']}, T={s['T']}, W={s['W']}, "
            f"pairs={s['n_pairs']}, reliable={s['n_reliable']}, "
            f"rho*_mean={s['rho_star_mean']:.3f}, "
            f"band_gap={s['band_gap']:.3f}, "
            f"reff_corr={s['reff_corr']:.3f}{fallback_note}, "
            f"rows_dropped={q.rows_dropped}, cols_dropped={q.columns_dropped})"
        )


# ---------------------------------------------------------------------------
# Internal runner
# ---------------------------------------------------------------------------

def _run(data, W, labels, quality):
    T, N  = data.shape
    pairs = pair_table(data, W=W, labels=labels)
    rj    = reff_joint(data, W=W)
    bg    = _band_gap(data)
    rc, fallback = _reff_corrected(data, W=W)
    return SFEResult(pairs, rj, bg, rc, fallback, W, N, T, labels, data, quality)


def _raise_if_bad(quality):
    if not quality.ok():
        raise ValueError(
            "Data could not be cleaned:\n" +
            "\n".join(f"  {e}" for e in quality.errors)
        )


# ---------------------------------------------------------------------------
# Connectors
# ---------------------------------------------------------------------------

def from_dataframe(df, W: int, columns=None) -> SFEResult:
    """
    Run SFE on a pandas DataFrame.

    Parameters
    ----------
    df      : DataFrame, shape (T, N)
    W       : int
    columns : list of str, optional   Subset of columns. Default: all numeric.
    """
    try:
        import pandas as _pd   # noqa: F401
    except ImportError:
        raise ImportError("pandas required. pip install pandas")

    df = df[columns] if columns is not None else df
    df = df.select_dtypes(include="number")
    if df.shape[1] < 2:
        raise ValueError(f"Need ≥2 numeric columns, got {df.shape[1]}.")

    data, labels = df.values.astype(float), list(df.columns)
    data, labels, q = _clean(data, labels, W)
    _raise_if_bad(q)
    return _run(data, W, labels, q)


def from_array(data, W: int, labels=None) -> SFEResult:
    """
    Run SFE on a numpy array or (x, y) tuple.

    Parameters
    ----------
    data   : array_like (T, N)  or  tuple (x, y)
    W      : int
    labels : list of str, optional
    """
    if isinstance(data, tuple):
        if len(data) != 2:
            raise ValueError("Tuple must be exactly (x, y).")
        data = np.column_stack([np.asarray(data[0]), np.asarray(data[1])])

    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        raise ValueError("1-D array: provide shape (T, N) or a tuple (x, y).")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Need 2-D array with ≥2 columns, got {data.shape}.")

    labels = labels or [str(k) for k in range(data.shape[1])]
    data, labels, q = _clean(data, labels, W)
    _raise_if_bad(q)
    return _run(data, W, labels, q)


def from_csv(path, W: int, columns=None,
             delimiter=",", skip_rows=0) -> SFEResult:
    """
    Run SFE on a CSV file. No pandas required.

    Parameters
    ----------
    path      : str or Path
    W         : int
    columns   : str | int | list of str | list of int | None
    delimiter : str    default ','
    skip_rows : int    rows before the header
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open(newline="") as f:
        reader = _csv.reader(f, delimiter=delimiter)
        for _ in range(skip_rows):
            next(reader)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty file: {path}")

    def _numeric(s):
        try: float(s); return True
        except ValueError: return False

    has_header = not all(_numeric(c) for c in rows[0])
    header     = rows[0] if has_header else [str(k) for k in range(len(rows[0]))]
    data_rows  = rows[1:] if has_header else rows

    parsed, bad = [], 0
    for row in data_rows:
        try:
            parsed.append([float(c) for c in row])
        except ValueError:
            bad += 1

    if not parsed:
        raise ValueError(f"No numeric rows in {path}.")

    arr = np.array(parsed, dtype=float)

    if columns is None:
        col_idx = list(range(arr.shape[1]))
    else:
        if isinstance(columns, (int, str)):
            columns = [columns]
        col_idx = [header.index(c) if isinstance(c, str) else c for c in columns]

    arr, labels = arr[:, col_idx], [header[k] for k in col_idx]
    if arr.shape[1] < 2:
        raise ValueError(f"Need ≥2 columns after selection, got {arr.shape[1]}.")

    arr, labels, q = _clean(arr, labels, W)
    if bad:
        q.warnings.append(f"{bad} non-numeric row(s) skipped during parse.")
    _raise_if_bad(q)
    return _run(arr, W, labels, q)


def from_dict(data: dict, W: int) -> SFEResult:
    """
    Run SFE on a dict of equal-length sequences.

    Parameters
    ----------
    data : dict   keys=labels, values=1-D sequences
    W    : int
    """
    if len(data) < 2:
        raise ValueError(f"Need ≥2 keys, got {len(data)}.")

    labels = list(data.keys())
    arrays = [np.asarray(v, dtype=float) for v in data.values()]
    lengths = {k: len(a) for k, a in zip(labels, arrays)}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"All sequences must be equal length. Got: {lengths}")

    arr = np.column_stack(arrays)
    arr, labels, q = _clean(arr, labels, W)
    _raise_if_bad(q)
    return _run(arr, W, labels, q)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(result: SFEResult, show_warnings: bool = True) -> None:
    """Print a plain-text summary of an SFEResult. No dependencies."""
    s   = result.summary_dict()
    q   = result.quality
    env = OPERATING_ENVELOPE

    fallback_note = (
        "  ← f(N) over-corrected (< 1), raw joint mean used — see SFE-11 Open Problem 3"
        if s["reff_corr_fallback"] else ""
    )

    print()
    print("=" * 62)
    print("  SFE RESULT SUMMARY")
    print("=" * 62)
    print(f"  Observers : {s['N']}   Timesteps : {s['T']}   Window : {s['W']}")
    print(f"  Pairs     : {s['n_pairs']}   "
          f"Reliable (ρ*>{env['reliable_rho_min']}) : {s['n_reliable']}   "
          f"Flagged NS>40% : {s['n_flagged']}")
    print(f"  ρ* mean   : {s['rho_star_mean']:.4f}   ρ* max : {s['rho_star_max']:.4f}")
    print(f"  dρ mean   : {s['drho_mean']:.6f}")
    print(f"  r_eff joint (mean) : {s['reff_joint_mean']:.4f}")
    print(f"  r_eff corrected    : {s['reff_corr']:.4f}   "
          f"band gap λ₁/λ₂ : {s['band_gap']:.3f}×{fallback_note}")

    if show_warnings and (q.rows_dropped or q.columns_dropped or q.warnings):
        print()
        print("  ── Data quality ──")
        for w in q.warnings:
            print(f"  ⚠  {w}")

    print()
    print(f"  {'Pair':<18} {'ρ*':>6} {'dρ':>9} {'r_eff':>7} {'NS%':>6}  Zone")
    print(f"  {'-'*58}")
    for p in result.pairs:
        print(f"  {p['label']:<25} {p['rho_star']:>8.4f} "
              f"{p['drho_mean']:>10.6f} {p['reff_mean']:>8.4f} "
              f"{p['nonstationary_pct']:>5.1f}%  {p['zone']}")

    if result.flagged():
        print()
        print("  ⚠  Non-stationary pairs (NS > 40%):")
        for p in result.flagged():
            print(f"     {p['label']}  NS={p['nonstationary_pct']:.1f}%")

    print()
    print("  Experimental results. Not financial, medical, or engineering advice.")
    print("=" * 62)
    print()
