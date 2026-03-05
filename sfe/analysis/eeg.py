# -*- coding: utf-8 -*-
"""
sfe/analysis/eeg.py — Analysis helpers for EEG domain SFEResults.

Connectors produce SFEResults. This module operates on them.

    from sfe.analysis.eeg import event_locked_analysis, multi_subject_run
    from sfe.analysis.eeg import EventLockedResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..connect import SFEResult

__all__ = [
    "W_from_seconds",
    "epoch",
    "event_locked_analysis",
    "EventLockedResult",
    "multi_subject_run",
]

# PhysioNet eegmmidb event IDs
TASK_IDS = [2, 3]
REST_ID  = 1


def W_from_seconds(seconds: float, sfreq: float) -> int:
    """Convert a window duration in seconds to samples."""
    return int(round(seconds * sfreq))


def epoch(signal: np.ndarray, onsets: np.ndarray,
          pre: int, post: int) -> np.ndarray | None:
    """
    Stack signal epochs around event onsets.

    Parameters
    ----------
    signal : ndarray (T,)
    onsets : ndarray (E,)   sample indices
    pre    : int            samples before onset
    post   : int            samples after onset

    Returns
    -------
    ndarray (valid_epochs, pre+post) or None
    """
    epochs = []
    for s in onsets.astype(int):
        if s - pre >= 0 and s + post <= len(signal):
            epochs.append(signal[s - pre: s + post])
    return np.array(epochs) if epochs else None


@dataclass
class EventLockedResult:
    """
    Returned by event_locked_analysis().

    The two-phase EEG structure (SFE-11 Section 4.3):
        Phase 1: dρ spikes at stimulus onset (transition)
        Phase 2: dρ drops below baseline (re-lock)
        Lead time = peak_t - trough_t
        pct_drop > 20% = two-phase structure present
    """
    n_task_epochs    : int
    n_rest_epochs    : int
    baseline_drho    : float
    baseline_rho     : float
    drho_trough      : float
    rho_peak         : float
    trough_t         : float
    peak_t           : float
    pct_drop         : float
    lead_s           : float
    task_post_drho   : float
    rest_post_drho   : float
    direction        : str
    verdict          : str
    drho_task_mean   : np.ndarray
    rho_task_mean    : np.ndarray
    drho_rest_mean   : np.ndarray | None
    rho_rest_mean    : np.ndarray | None
    t_axis           : np.ndarray


def event_locked_analysis(
    result: SFEResult,
    pre_s:   float = 2.0,
    post_s:  float = 6.0,
    task_ids: list[int] = None,
    rest_id:  int = None,
) -> EventLockedResult | None:
    """
    Compute event-locked dρ and ρ* trajectories from a SFEResult.

    Requires result.events_df and result.sfreq (set by from_edf / from_eeg_csv).
    """
    task_ids = task_ids or TASK_IDS
    rest_id  = rest_id  or REST_ID

    if not hasattr(result, "events_df") or result.events_df is None:
        print("  ⚠  event_locked_analysis: no events_df attached to result.")
        return None
    if not hasattr(result, "sfreq") or result.sfreq is None:
        print("  ⚠  event_locked_analysis: no sfreq attached to result.")
        return None

    sfreq     = float(result.sfreq)
    pre_samp  = int(pre_s  * sfreq)
    post_samp = int(post_s * sfreq)
    win_len   = pre_samp + post_samp
    t_axis    = np.linspace(-pre_s, post_s, win_len)

    if not result.pairs:
        print("  ⚠  event_locked_analysis: no pairs in result.")
        return None

    pair     = result.pairs[0]
    rho_sig  = pair["rho"]
    drho_sig = pair["drho"]

    events_df    = result.events_df
    task_onsets  = events_df[events_df["event_id"].isin(task_ids)]["sample"].values
    rest_onsets  = events_df[events_df["event_id"] == rest_id]["sample"].values

    drho_task = epoch(drho_sig, task_onsets, pre_samp, post_samp)
    rho_task  = epoch(np.abs(rho_sig), task_onsets, pre_samp, post_samp)

    if drho_task is None or len(drho_task) < 1:
        print(f"  ⚠  event_locked_analysis: no valid task epochs.")
        return None

    drho_task_mean = np.nanmean(drho_task, axis=0)
    rho_task_mean  = np.nanmean(rho_task,  axis=0)

    post_start      = pre_samp
    trough_rel      = int(np.nanargmin(drho_task_mean[post_start:]))
    peak_rel        = int(np.nanargmax(rho_task_mean[post_start:]))
    trough_t        = float(t_axis[post_start + trough_rel])
    peak_t          = float(t_axis[post_start + peak_rel])
    drho_trough_val = float(drho_task_mean[post_start + trough_rel])
    rho_peak_val    = float(rho_task_mean[post_start + peak_rel])
    lead_s          = peak_t - trough_t

    baseline_drho  = float(np.mean(drho_task_mean[:pre_samp]))
    baseline_rho   = float(np.mean(rho_task_mean[:pre_samp]))
    pct_drop       = 100.0 * (baseline_drho - drho_trough_val) / (baseline_drho + 1e-12)
    task_post_drho = float(np.mean(drho_task_mean[post_start:]))

    drho_rest_mean = rho_rest_mean = None
    n_rest         = 0
    rest_post_drho = float("nan")

    drho_rest = epoch(drho_sig, rest_onsets, pre_samp, post_samp)
    rho_rest  = epoch(np.abs(rho_sig), rest_onsets, pre_samp, post_samp)

    if drho_rest is not None and len(drho_rest) > 0:
        n_rest         = len(drho_rest)
        drho_rest_mean = np.nanmean(drho_rest, axis=0)
        rho_rest_mean  = np.nanmean(rho_rest,  axis=0)
        rest_post_drho = float(np.mean(drho_rest_mean[post_start:]))

    direction = (
        "LOWER during task ✓  (lock-in)"
        if task_post_drho < rest_post_drho
        else "HIGHER during task  (transition mechanism)"
    ) if not np.isnan(rest_post_drho) else "N/A — no rest epochs"

    verdict = (
        f"Two-phase structure {'PRESENT ✓' if pct_drop > 20 else 'ABSENT — pct_drop < 20%'}. "
        f"dρ trough={drho_trough_val:.6f}  drop={pct_drop:.1f}%  "
        f"lead={lead_s:+.2f}s  epochs={len(drho_task)}."
    )

    return EventLockedResult(
        n_task_epochs  = len(drho_task),
        n_rest_epochs  = n_rest,
        baseline_drho  = baseline_drho,
        baseline_rho   = baseline_rho,
        drho_trough    = drho_trough_val,
        rho_peak       = rho_peak_val,
        trough_t       = trough_t,
        peak_t         = peak_t,
        pct_drop       = pct_drop,
        lead_s         = lead_s,
        task_post_drho = task_post_drho,
        rest_post_drho = rest_post_drho,
        direction      = direction,
        verdict        = verdict,
        drho_task_mean = drho_task_mean,
        rho_task_mean  = rho_task_mean,
        drho_rest_mean = drho_rest_mean,
        rho_rest_mean  = rho_rest_mean,
        t_axis         = t_axis,
    )


def multi_subject_run(
    edf_dir:   str,
    subjects:  list[str],
    runs:      list[str],
    channels:  list[str] | None = None,
    W:         int = 160,
    pre_s:     float = 2.0,
    post_s:    float = 6.0,
    min_epochs: int = 3,
    pass_threshold_pct: float = 20.0,
    pass_n_min: int = 8,
) -> dict:
    """
    Run event-locked SFE analysis across multiple subjects and runs.

    Imports from_edf lazily to avoid circular dependency.
    """
    from ..connectors.eeg import from_edf

    records = []

    for subj in subjects:
        for run in runs:
            edf_path = Path(edf_dir) / subj / f"{subj}{run}.edf"
            if not edf_path.exists():
                print(f"  {subj}/{run}: not found, skipping")
                continue
            try:
                result = from_edf(str(edf_path), channels=channels, W=W)

                if not hasattr(result, "events_df") or result.events_df is None:
                    print(f"  {subj}/{run}: no events, skipping")
                    continue

                ela = event_locked_analysis(result, pre_s=pre_s, post_s=post_s)

                if ela is None or ela.n_task_epochs < min_epochs:
                    print(f"  {subj}/{run}: only "
                          f"{ela.n_task_epochs if ela else 0} epochs, skipping")
                    continue

                records.append({
                    "subject":                 subj,
                    "run":                     run,
                    "n_epochs":                ela.n_task_epochs,
                    "rho_star":                result.summary_dict()["rho_star_mean"],
                    "baseline_drho":           ela.baseline_drho,
                    "drho_trough":             ela.drho_trough,
                    "pct_drop_below_baseline": ela.pct_drop,
                    "rho_peak_t":              ela.peak_t,
                    "drho_trough_t":           ela.trough_t,
                    "lead_s":                  ela.lead_s,
                })

                print(f"  {subj}/{run}: rho*={records[-1]['rho_star']:.3f}  "
                      f"drop={ela.pct_drop:.1f}%  lead={ela.lead_s:+.2f}s")

            except Exception as e:
                print(f"  {subj}/{run}: ERROR — {e}")

    if not records:
        return {"records": [], "summary": {}, "verdict": "NO DATA",
                "pass_n": 0, "pass_total": 0}

    summary = {}
    for col in ["rho_star", "drho_trough", "pct_drop_below_baseline",
                "rho_peak_t", "lead_s"]:
        vals = [r[col] for r in records]
        summary[col] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    r04_records = [r for r in records if r["run"] == "R04"]
    pass_n      = sum(r["pct_drop_below_baseline"] > pass_threshold_pct
                      for r in r04_records)
    pass_total  = len(r04_records)

    if pass_n >= pass_n_min:
        verdict = "TWO-PHASE STRUCTURE REPLICATED"
    elif pass_n >= pass_n_min // 2:
        verdict = "MIXED — check individual subjects"
    else:
        verdict = "NOT REPLICATED — single-subject artifact"

    print(f"\n  Pass condition (R04, pct_drop>{pass_threshold_pct:.0f}%): "
          f"{pass_n}/{pass_total}  → {verdict}")

    return {"records": records, "summary": summary, "verdict": verdict,
            "pass_n": pass_n, "pass_total": pass_total}
