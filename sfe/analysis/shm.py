# -*- coding: utf-8 -*-
"""
sfe/analysis/shm.py — Analysis helpers for Structural Health Monitoring SFEResults.

Connectors produce SFEResults. This module operates on them.

Handles phase-boundary analysis for datasets with known structural events
(retrofitting, damage introduction, environmental transitions).

KW51 has three structural phases:
    Phase 1: nominal operation  (before May 2019)
    Phase 2: retrofitting       (intentional structural intervention)
    Phase 3: post-retrofit      (resumed operation)

The instrument should remain silent during uniform phases and show dρ
elevation or ρ* drop at phase boundaries — the same logical structure as
Branch A/B in finance, but for structural transitions rather than market crises.

    from sfe.analysis.shm import (
        slice_phase, detect_structural_transition, PhaseTransitionResult,
        matlab_datenum_to_datetime,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from ..connect import SFEResult

__all__ = [
    "slice_phase",
    "detect_structural_transition",
    "PhaseTransitionResult",
    "matlab_datenum_to_datetime",
    "TRANSITION_THRESHOLDS",
]


from .regimes import TRANSITION_THRESHOLDS


# ---------------------------------------------------------------------------
# MATLAB datenum conversion
# ---------------------------------------------------------------------------

def matlab_datenum_to_datetime(datenums: np.ndarray) -> list[datetime]:
    """
    Convert MATLAB serial datenums to Python datetime objects.

    MATLAB epoch: January 0, year 0 = 0.
    Python offset: MATLAB datenum 719529 = 1970-01-01 (Unix epoch).

    Parameters
    ----------
    datenums : ndarray (T,)  MATLAB serial date numbers (float)

    Returns
    -------
    list of datetime objects, length T
    """
    # MATLAB datenum 1 = Jan 1, year 0001 in the proleptic Gregorian calendar
    # Offset to Python's datetime epoch (0001-01-01)
    matlab_epoch = datetime(1, 1, 1)
    return [
        matlab_epoch + timedelta(days=float(d) - 1)
        for d in datenums
    ]


# ---------------------------------------------------------------------------
# Phase slicer — index-based (works without timestamps)
# ---------------------------------------------------------------------------

def slice_phase(
    result: SFEResult,
    start: int | None = None,
    end:   int | None = None,
    start_dt: datetime | str | None = None,
    end_dt:   datetime | str | None = None,
    W: int | None = None,
) -> SFEResult:
    """
    Re-run SFE on a subset of a SFEResult's data.

    Two modes:
        - Index-based: pass start/end as sample indices.
        - Datetime-based: pass start_dt/end_dt; requires result.timestamps
          to be datetime objects (use matlab_datenum_to_datetime first).

    Parameters
    ----------
    result   : SFEResult
    start    : int | None   start sample index (inclusive)
    end      : int | None   end sample index (exclusive)
    start_dt : datetime | str | None   start datetime (for timestamp-indexed data)
    end_dt   : datetime | str | None   end datetime
    W        : int | None   override window size (default: same as original)

    Returns
    -------
    SFEResult for the sliced phase.
    """
    from ..connect import from_array

    W_use = W if W is not None else result.W

    # Datetime-based slicing
    if start_dt is not None or end_dt is not None:
        if not hasattr(result, "timestamps") or result.timestamps is None:
            raise AttributeError(
                "result.timestamps required for datetime-based slicing. "
                "Attach timestamps via from_mat(timestamps_key=...) and "
                "convert with matlab_datenum_to_datetime()."
            )
        ts = result.timestamps

        # Accept string dates
        if isinstance(start_dt, str):
            start_dt = datetime.fromisoformat(start_dt)
        if isinstance(end_dt, str):
            end_dt = datetime.fromisoformat(end_dt)

        mask = np.ones(len(ts), dtype=bool)
        if start_dt is not None:
            mask &= np.array([t >= start_dt for t in ts])
        if end_dt is not None:
            mask &= np.array([t <= end_dt for t in ts])

        if mask.sum() == 0:
            raise ValueError(
                f"No samples in window {start_dt} → {end_dt}."
            )

        data_slice = result.data[mask]
        ts_slice   = [ts[i] for i, m in enumerate(mask) if m]

    else:
        # Index-based slicing
        s = start if start is not None else 0
        e = end   if end   is not None else result.T
        if s >= e or s >= result.T:
            raise ValueError(f"Invalid slice [{s}:{e}] for T={result.T}.")
        data_slice = result.data[s:e]
        ts_slice   = None
        if hasattr(result, "timestamps") and result.timestamps is not None:
            ts_slice = result.timestamps[s:e] if hasattr(result.timestamps, '__getitem__') else None

    if data_slice.shape[0] <= W_use:
        raise ValueError(
            f"Slice has only {data_slice.shape[0]} samples; "
            f"W={W_use} requires more. Reduce W or widen the window."
        )

    sub            = from_array(data_slice, W=W_use, labels=result.labels)
    sub.timestamps = ts_slice
    sub.mat_key    = getattr(result, "mat_key", None)
    sub.mat_path   = getattr(result, "mat_path", None)
    return sub


# ---------------------------------------------------------------------------
# Structural transition detector
# ---------------------------------------------------------------------------

@dataclass
class PhaseTransitionResult:
    """
    Returned by detect_structural_transition().

    Attributes
    ----------
    label          : str    verdict string
    fired          : bool   True if a structural transition was detected
    delta_rho      : float  mean Δρ* (post − pre)
    drho_ratio     : float  post-transition dρ mean / pre-transition dρ mean
    delta_reff     : float  Δr_eff corrected (post − pre)
    bandgap_ratio  : float  post band_gap / pre band_gap
    notes          : list of str
    """
    label         : str
    fired         : bool
    delta_rho     : float
    drho_ratio    : float
    delta_reff    : float
    bandgap_ratio : float
    notes         : list = field(default_factory=list)

    def __str__(self):
        lines = [
            f"Structural transition: {self.label}",
            f"  Δρ* mean      : {self.delta_rho:+.4f}",
            f"  dρ ratio      : {self.drho_ratio:.3f}×  "
            f"(threshold ≥{TRANSITION_THRESHOLDS['drho_spike_mult']:.1f}×)",
            f"  Δr_eff corr   : {self.delta_reff:+.4f}",
            f"  Band gap ratio: {self.bandgap_ratio:.3f}×",
        ]
        for note in self.notes:
            lines.append(f"  Note: {note}")
        return "\n".join(lines)


def detect_structural_transition(
    pre: SFEResult,
    post: SFEResult,
) -> PhaseTransitionResult:
    """
    Compare pre- and post-transition SFEResults and assess whether the
    instrument detects a structural change.

    This is the SHM analogue of detect_regime() for finance. The detection
    logic is structurally identical — compare two windows, check thresholds —
    but the expected signatures differ:

        Structural transition (e.g. KW51 retrofitting):
            - ρ* may drop  (coupling loosens during intervention)
            - dρ rises     (channel destabilizes, less deterministic coupling)
            - r_eff rises  (more degrees of freedom — structure no longer monolithic)
            - band gap falls (dominant mode weakens relative to others)

        Silent during nominal phases (no intervention):
            - ρ* stable, dρ ≈ 0, r_eff ≈ 1, band gap stable

    Parameters
    ----------
    pre  : SFEResult   pre-transition phase
    post : SFEResult   post-transition or during-transition phase

    Returns
    -------
    PhaseTransitionResult

    Note
    ----
    TRANSITION_THRESHOLDS are initial values pending empirical calibration
    on KW51. They will be updated when the dataset is run.
    This follows the same pattern as the f(N) fallback: a new boundary
    condition is identified and patched; no existing theorem is modified.
    """
    thr   = TRANSITION_THRESHOLDS
    notes = []

    pre_pairs  = {p["label"]: p for p in pre.pairs}
    post_pairs = {p["label"]: p for p in post.pairs}
    shared     = [l for l in pre_pairs if l in post_pairs]

    if not shared:
        return PhaseTransitionResult(
            label="NO SHARED PAIRS — cannot assess",
            fired=False,
            delta_rho=float("nan"), drho_ratio=float("nan"),
            delta_reff=float("nan"), bandgap_ratio=float("nan"),
            notes=["pre and post results have no overlapping pair labels"],
        )

    delta_rhos = [post_pairs[l]["rho_star"]  - pre_pairs[l]["rho_star"]  for l in shared]
    pre_drhos  = [pre_pairs[l]["drho_mean"]  for l in shared]
    post_drhos = [post_pairs[l]["drho_mean"] for l in shared]

    mean_delta_rho = float(np.mean(delta_rhos))
    mean_pre_drho  = float(np.mean(pre_drhos))
    mean_post_drho = float(np.mean(post_drhos))
    drho_ratio     = mean_post_drho / (mean_pre_drho + 1e-12)

    delta_reff    = post.reff_corr - pre.reff_corr
    bandgap_ratio = (post.band_gap / pre.band_gap
                     if pre.band_gap > 1e-9 else float("nan"))

    # Detection conditions
    rho_dropped   = mean_delta_rho  <= thr["rho_delta_min"]
    drho_spiked   = drho_ratio      >= thr["drho_spike_mult"]
    reff_rose     = delta_reff      >= thr["reff_delta_min"]

    if drho_spiked and (rho_dropped or reff_rose):
        if rho_dropped:
            notes.append(
                f"ρ* dropped {mean_delta_rho:+.4f} — coupling loosened at transition."
            )
        if reff_rose:
            notes.append(
                f"r_eff rose {delta_reff:+.4f} — structure acquired new degrees of freedom."
            )
        notes.append(
            f"dρ ratio={drho_ratio:.2f}× — channel destabilized "
            f"(pre mean={mean_pre_drho:.6f}, post mean={mean_post_drho:.6f})."
        )
        if not np.isnan(bandgap_ratio):
            notes.append(f"Band gap ratio={bandgap_ratio:.3f}×.")

        notes.append(
            "BOUNDARY CONDITION NOTE: TRANSITION_THRESHOLDS are initial estimates. "
            "Calibrate against KW51 ground-truth phase labels before reporting."
        )

        return PhaseTransitionResult(
            label="STRUCTURAL TRANSITION DETECTED ✓",
            fired=True,
            delta_rho=mean_delta_rho, drho_ratio=drho_ratio,
            delta_reff=delta_reff, bandgap_ratio=bandgap_ratio,
            notes=notes,
        )

    # Silent
    notes.append(
        f"dρ ratio={drho_ratio:.2f}× (threshold {thr['drho_spike_mult']:.1f}×), "
        f"Δρ*={mean_delta_rho:+.4f} (threshold {thr['rho_delta_min']:+.2f}). "
        f"No structural transition detected."
    )

    return PhaseTransitionResult(
        label="SILENT — no structural transition detected",
        fired=False,
        delta_rho=mean_delta_rho, drho_ratio=drho_ratio,
        delta_reff=delta_reff,
        bandgap_ratio=bandgap_ratio if not np.isnan(bandgap_ratio) else float("nan"),
        notes=notes,
    )
