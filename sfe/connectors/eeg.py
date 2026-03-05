# -*- coding: utf-8 -*-
"""
sfe/connectors/eeg.py — EEG domain connector.

Handles EDF / H5 / CSV biosignal files → channel extraction → SFEResult.
Supports PhysioNet-style EDF files (MNE backend) and pre-exported CSVs.

Preprocessing contract:
    Raw EEG is passed directly to SFE core — no filtering, no z-scoring.
    W is specified in samples; caller converts from seconds using sfreq.
    Event extraction and epoch stacking are handled by event_locked_analysis().

Usage:
    from sfe.connectors.eeg import from_edf, event_locked_analysis, multi_subject_run

    result = from_edf("S001R04.edf", channels=["C3", "C4"], W=160)

    ela = event_locked_analysis(result, pre_s=2.0, post_s=6.0)
    print(ela.pct_drop, ela.lead_s, ela.verdict)

    # Multi-subject replication
    summary = multi_subject_run(
        edf_dir="./eeg_data",
        subjects=["S001", "S002", "S003"],
        runs=["R04", "R08"],
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from ..connect import from_array, SFEResult

__all__ = [
    "from_edf",
    "from_eeg_csv",
    "from_h5",
    "pick_channels",
    "W_from_seconds",
    "epoch",
    "event_locked_analysis",
    "EventLockedResult",
    "multi_subject_run",
]

# ---------------------------------------------------------------------------
# Channel selection helpers
# ---------------------------------------------------------------------------

_MOTOR_CHANNEL_PRIORITY = [
    ["C3", "C4"],
    ["C3..", "C4.."],
    ["EEG C3", "EEG C4"],
]

# PhysioNet eegmmidb task event IDs
TASK_IDS = [2, 3]   # T1=left fist, T2=right fist
REST_ID  = 1        # T0=rest/baseline


def pick_channels(available: list[str], targets: list[str] | None = None) -> list[str]:
    """
    Select channels from available list.

    If targets is given, find the best prefix match for each target.
    If targets is None, try motor-cortex defaults then fall back to first two.
    """
    def _prefix_match(name, available):
        name = name.rstrip(".")
        matches = [c for c in available if c.startswith(name)]
        return matches[0] if matches else None

    if targets is not None:
        found = [_prefix_match(t, available) for t in targets]
        found = [c for c in found if c is not None]
        if len(found) >= 2:
            return found
        print(f"  ⚠  Only {len(found)}/{len(targets)} targets matched; "
              f"trying motor defaults.")

    for pair in _MOTOR_CHANNEL_PRIORITY:
        found = [_prefix_match(t, available) for t in pair]
        found = [c for c in found if c is not None]
        if len(found) >= 2:
            return found

    if len(available) >= 2:
        print("  ⚠  No motor channels found — using first two channels.")
        return available[:2]

    raise ValueError(
        f"Cannot select 2 channels from {len(available)} available: {available}"
    )


def W_from_seconds(seconds: float, sfreq: float) -> int:
    """Convert a window duration in seconds to samples."""
    return int(round(seconds * sfreq))


# ---------------------------------------------------------------------------
# Epoch helper — pure numpy, vectorized
# ---------------------------------------------------------------------------

def epoch(signal: np.ndarray, onsets: np.ndarray,
          pre: int, post: int) -> np.ndarray | None:
    """
    Stack signal epochs around event onsets.

    Parameters
    ----------
    signal : ndarray (T,)     1-D signal (rho, drho, etc.)
    onsets : ndarray (E,)     sample indices of event onsets
    pre    : int              samples before onset
    post   : int              samples after onset

    Returns
    -------
    ndarray (valid_epochs, pre+post)  or  None if no valid epochs
    """
    epochs = []
    for s in onsets.astype(int):
        if s - pre >= 0 and s + post <= len(signal):
            epochs.append(signal[s - pre: s + post])
    return np.array(epochs) if epochs else None


# ---------------------------------------------------------------------------
# Event-locked analysis result
# ---------------------------------------------------------------------------

@dataclass
class EventLockedResult:
    """
    Returned by event_locked_analysis().

    Attributes
    ----------
    n_task_epochs    : int
    n_rest_epochs    : int
    baseline_drho    : float   mean dρ in pre-onset window
    baseline_rho     : float   mean ρ* in pre-onset window
    drho_trough      : float   minimum post-onset dρ (averaged over epochs)
    rho_peak         : float   maximum post-onset ρ* (averaged over epochs)
    trough_t         : float   time of dρ trough in seconds post-onset
    peak_t           : float   time of ρ* peak in seconds post-onset
    pct_drop         : float   % drop of dρ below baseline (positive = drop)
    lead_s           : float   peak_t - trough_t  (negative = re-lock completes before ρ peak)
    task_post_drho   : float   mean dρ post-onset during task
    rest_post_drho   : float   mean dρ post-onset during rest
    direction        : str     "LOWER during task ✓" or "HIGHER during task"
    verdict          : str     plain-text summary
    drho_task_mean   : ndarray (pre+post,)  epoch-averaged dρ — task
    rho_task_mean    : ndarray (pre+post,)  epoch-averaged |ρ| — task
    drho_rest_mean   : ndarray (pre+post,) | None
    rho_rest_mean    : ndarray (pre+post,) | None
    t_axis           : ndarray (pre+post,)  time axis in seconds
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


# ---------------------------------------------------------------------------
# Event-locked analysis — main function
# ---------------------------------------------------------------------------

def event_locked_analysis(
    result: SFEResult,
    pre_s:  float = 2.0,
    post_s: float = 6.0,
    task_ids: list[int] = None,
    rest_id:  int = None,
) -> EventLockedResult | None:
    """
    Compute event-locked dρ and ρ* trajectories from a SFEResult.

    Requires result to have .events_df and .sfreq attached
    (set automatically by from_edf() and from_eeg_csv()).

    The two-phase EEG structure (SFE-11 Section 4.3):
        Phase 1: dρ spikes at stimulus onset (channel enters transition)
        Phase 2: dρ drops below baseline (channel re-locks, geometric stabilization)
        Lead time = peak_t - trough_t (negative = re-lock before ρ* maximum)
        pct_drop > 20% = two-phase structure present

    Parameters
    ----------
    result   : SFEResult with .events_df and .sfreq attached
    pre_s    : float   seconds before onset for baseline window (default 2.0)
    post_s   : float   seconds after onset for event window (default 6.0)
    task_ids : list    event IDs for task events (default TASK_IDS = [2, 3])
    rest_id  : int     event ID for rest (default REST_ID = 1)

    Returns
    -------
    EventLockedResult or None if no events found
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

    events_df = result.events_df

    # Use the first pair's rho / drho (C3-C4)
    if not result.pairs:
        print("  ⚠  event_locked_analysis: no pairs in result.")
        return None
    pair = result.pairs[0]
    rho_sig  = pair["rho"]
    drho_sig = pair["drho"]

    # Extract onsets
    task_onsets = events_df[events_df["event_id"].isin(task_ids)]["sample"].values
    rest_onsets = events_df[events_df["event_id"] == rest_id]["sample"].values

    # Task epochs
    drho_task = epoch(drho_sig, task_onsets, pre_samp, post_samp)
    rho_task  = epoch(np.abs(rho_sig), task_onsets, pre_samp, post_samp)

    if drho_task is None or len(drho_task) < 1:
        print(f"  ⚠  event_locked_analysis: no valid task epochs "
              f"(task_ids={task_ids}, found {len(task_onsets)} onsets).")
        return None

    drho_task_mean = np.nanmean(drho_task, axis=0)
    rho_task_mean  = np.nanmean(rho_task,  axis=0)

    # Trough and peak post-onset
    post_start      = pre_samp
    trough_rel      = int(np.nanargmin(drho_task_mean[post_start:]))
    peak_rel        = int(np.nanargmax(rho_task_mean[post_start:]))
    trough_t        = float(t_axis[post_start + trough_rel])
    peak_t          = float(t_axis[post_start + peak_rel])
    drho_trough_val = float(drho_task_mean[post_start + trough_rel])
    rho_peak_val    = float(rho_task_mean[post_start + peak_rel])
    lead_s          = peak_t - trough_t

    baseline_drho = float(np.mean(drho_task_mean[:pre_samp]))
    baseline_rho  = float(np.mean(rho_task_mean[:pre_samp]))
    pct_drop      = 100.0 * (baseline_drho - drho_trough_val) / (baseline_drho + 1e-12)

    # Rest epochs (optional)
    drho_rest_mean = rho_rest_mean = None
    n_rest = 0
    task_post_drho = float(np.mean(drho_task_mean[post_start:]))
    rest_post_drho = float("nan")

    drho_rest = epoch(drho_sig, rest_onsets, pre_samp, post_samp)
    rho_rest  = epoch(np.abs(rho_sig), rest_onsets, pre_samp, post_samp)

    if drho_rest is not None and len(drho_rest) > 0:
        n_rest         = len(drho_rest)
        drho_rest_mean = np.nanmean(drho_rest, axis=0)
        rho_rest_mean  = np.nanmean(rho_rest,  axis=0)
        rest_post_drho = float(np.mean(drho_rest_mean[post_start:]))

    direction = (
        "LOWER during task ✓  (lock-in, same direction as ETT)"
        if task_post_drho < rest_post_drho
        else "HIGHER during task  (transition mechanism, same direction as METR-LA)"
    ) if not np.isnan(rest_post_drho) else "N/A — no rest epochs"

    verdict = (
        f"Two-phase structure {'PRESENT ✓' if pct_drop > 20 else 'ABSENT — pct_drop < 20%'}. "
        f"dρ trough={drho_trough_val:.6f}  drop={pct_drop:.1f}%  lead={lead_s:+.2f}s  "
        f"epochs={len(drho_task)}."
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


# ---------------------------------------------------------------------------
# Multi-subject runner
# ---------------------------------------------------------------------------

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

    Replicates the multi-subject analysis from SFE-11 Section 4.3:
    Tests whether the two-phase event structure (dρ disruption → re-lock)
    found in Subject 1 Run 4 replicates across independent subjects.

    Parameters
    ----------
    edf_dir    : str              directory containing subject EDF subdirectories
    subjects   : list of str      e.g. ["S001", "S002", ..., "S010"]
    runs       : list of str      e.g. ["R04", "R08"]
    channels   : list of str      channel pair (default: auto motor cortex)
    W          : int              rolling window in samples (default 160 = 1s at 160Hz)
    pre_s      : float            pre-stimulus baseline window in seconds
    post_s     : float            post-stimulus window in seconds
    min_epochs : int              minimum valid epochs to include subject (default 3)
    pass_threshold_pct : float    pct_drop threshold for pass condition (default 20%)
    pass_n_min : int              minimum subjects passing for replication verdict (default 8)

    Returns
    -------
    dict with keys:
        "records"  : list of dicts, one per subject/run
        "summary"  : dict of aggregated mean ± std
        "verdict"  : str  "TWO-PHASE STRUCTURE REPLICATED" | "MIXED" | "NOT REPLICATED"
        "pass_n"   : int  number of subjects passing on R04
        "pass_total": int total subjects with R04 data
    """
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
                    print(f"  {subj}/{run}: only {ela.n_task_epochs if ela else 0} "
                          f"epochs (< {min_epochs}), skipping")
                    continue

                records.append({
                    "subject":                  subj,
                    "run":                      run,
                    "n_epochs":                 ela.n_task_epochs,
                    "rho_star":                 result.summary_dict()["rho_star_mean"],
                    "baseline_drho":            ela.baseline_drho,
                    "drho_trough":              ela.drho_trough,
                    "pct_drop_below_baseline":  ela.pct_drop,
                    "rho_peak_t":               ela.peak_t,
                    "drho_trough_t":            ela.trough_t,
                    "lead_s":                   ela.lead_s,
                })

                print(f"  {subj}/{run}: rho*={records[-1]['rho_star']:.3f}  "
                      f"drho_trough={ela.drho_trough:.6f}  "
                      f"drop={ela.pct_drop:.1f}%  "
                      f"lead={ela.lead_s:+.2f}s  "
                      f"epochs={ela.n_task_epochs}")

            except Exception as e:
                print(f"  {subj}/{run}: ERROR — {e}")

    if not records:
        return {
            "records": [], "summary": {}, "verdict": "NO DATA",
            "pass_n": 0, "pass_total": 0,
        }

    # Aggregate stats
    import numpy as _np
    summary = {}
    for col in ["rho_star", "drho_trough", "pct_drop_below_baseline",
                "rho_peak_t", "lead_s"]:
        vals = [r[col] for r in records]
        summary[col] = {"mean": float(_np.mean(vals)), "std": float(_np.std(vals))}

    # Pass condition: R04 pct_drop > pass_threshold_pct
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

    return {
        "records":     records,
        "summary":     summary,
        "verdict":     verdict,
        "pass_n":      pass_n,
        "pass_total":  pass_total,
    }


# ---------------------------------------------------------------------------
# EDF loader
# ---------------------------------------------------------------------------

def from_edf(
    path: str,
    channels: list[str] | None = None,
    W: int = 160,
    return_meta: bool = False,
) -> SFEResult | tuple[SFEResult, dict]:
    """
    Load an EDF file, pick channels, run SFE.

    Parameters
    ----------
    path        : str           path to .edf file
    channels    : list of str   channel names (prefix match). None = auto motor.
    W           : int           rolling window in samples (use W_from_seconds to convert)
    return_meta : bool          if True, also return a metadata dict

    Returns
    -------
    SFEResult  (or (SFEResult, meta) if return_meta=True)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EDF file not found: {path}")

    try:
        import mne
        mne.set_log_level("WARNING")

        raw      = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
        sfreq    = float(raw.info["sfreq"])
        avail    = raw.ch_names
        selected = pick_channels(avail, channels)

        picks       = mne.pick_channels(avail, include=selected)
        data, _     = raw[picks, :]
        data        = data.T   # (T, n_ch)

        try:
            events_mne, event_id = mne.events_from_annotations(raw, verbose=False)
            import pandas as _pd
            events_df = _pd.DataFrame(
                events_mne, columns=["sample", "prev", "event_id"]
            )
            events_df["event_label"] = events_df["event_id"].map(
                {v: k for k, v in event_id.items()}
            )
        except Exception:
            events_df = None

        print(f"  Loaded EDF via MNE: {path.name}")
        print(f"  Channels : {selected}")
        print(f"  sfreq    : {sfreq} Hz  |  {data.shape[0]} samples "
              f"({data.shape[0]/sfreq:.1f} s)")

    except ImportError:
        try:
            import pyedflib
        except ImportError:
            raise ImportError("mne or pyedflib required. pip install mne")

        f        = pyedflib.EdfReader(str(path))
        avail    = f.getSignalLabels()
        selected = pick_channels(list(avail), channels)
        idx      = [list(avail).index(c) for c in selected]
        arrays   = [f.readSignal(i).astype(float) for i in idx]
        sfreq    = float(f.getSampleFrequency(idx[0]))
        f._close()

        data      = np.column_stack(arrays)
        events_df = None

        print(f"  Loaded EDF via pyedflib: {path.name}")
        print(f"  Channels : {selected}  sfreq={sfreq} Hz")

    # Rescale W if sfreq differs from 160 Hz (PhysioNet default)
    W_use = max(int(round(W * sfreq / 160.0)), 2)
    if W_use != W:
        print(f"  W rescaled: {W} → {W_use} samples (sfreq={sfreq} Hz)")

    result = from_array(data, W=W_use, labels=selected)
    result.sfreq     = sfreq
    result.ch_names  = selected
    result.events_df = events_df

    if return_meta:
        meta = {
            "sfreq":     sfreq,
            "ch_names":  selected,
            "events_df": events_df,
            "n_samples": data.shape[0],
        }
        return result, meta

    return result


# ---------------------------------------------------------------------------
# CSV fallback
# ---------------------------------------------------------------------------

def from_eeg_csv(
    path: str,
    W: int = 160,
    channels: list[str] | None = None,
    events_path: str | None = None,
    sfreq: float = 160.0,
) -> SFEResult:
    """
    Load pre-exported EEG signals from a CSV file and run SFE.

    Parameters
    ----------
    path        : str           CSV with one column per channel
    W           : int           rolling window in samples
    channels    : list of str   column names to use (None = first two)
    events_path : str, opt      path to events CSV (columns: sample, event_id)
    sfreq       : float         sampling frequency (for metadata only)

    Returns
    -------
    SFEResult with .sfreq, .ch_names, .events_df attached.
    """
    import pandas as _pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EEG CSV not found: {path}")

    df = _pd.read_csv(path)
    numeric = df.select_dtypes(include="number")

    if channels is not None:
        available = [c for c in channels if c in numeric.columns]
        if len(available) < 2:
            raise ValueError(
                f"Requested channels {channels} not found in {list(numeric.columns)}"
            )
        numeric = numeric[available]
    else:
        if numeric.shape[1] < 2:
            raise ValueError(
                f"Need ≥2 numeric columns in {path}, got {numeric.shape[1]}."
            )
        numeric = numeric.iloc[:, :2]

    selected = list(numeric.columns)
    data     = numeric.values.astype(float)

    events_df = None
    if events_path is not None:
        ep = Path(events_path)
        if ep.exists():
            events_df = _pd.read_csv(ep)
        else:
            print(f"  ⚠  Events file not found: {ep}")

    print(f"  Loaded EEG CSV: {path.name}  channels={selected}  shape={data.shape}")

    result           = from_array(data, W=W, labels=selected)
    result.sfreq     = sfreq
    result.ch_names  = selected
    result.events_df = events_df
    return result


# ---------------------------------------------------------------------------
# H5 loader
# ---------------------------------------------------------------------------

def from_h5(
    path: str,
    W: int,
    key: str = "data",
    channels: list[str] | None = None,
    sfreq: float | None = None,
) -> SFEResult:
    """
    Load preprocessed EEG data from an HDF5 file and run SFE.

    Parameters
    ----------
    path     : str           path to .h5 file
    W        : int           rolling window in samples
    key      : str           dataset key inside the H5 file (default: "data")
    channels : list of str   channel labels (optional, for display)
    sfreq    : float, opt    sampling frequency (attached to result metadata)

    Returns
    -------
    SFEResult with .sfreq and .ch_names attached if provided.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required. pip install h5py")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"H5 file not found: {path}")

    with h5py.File(path, "r") as f:
        if key not in f:
            available = list(f.keys())
            raise KeyError(
                f"Key '{key}' not found in {path.name}. "
                f"Available keys: {available}"
            )
        data = f[key][()].astype(float)

    if data.ndim != 2:
        raise ValueError(
            f"H5 dataset '{key}' must be 2-D (T, N), got shape {data.shape}."
        )

    labels = channels or [str(k) for k in range(data.shape[1])]

    print(f"  Loaded H5: {path.name}  key='{key}'  shape={data.shape}")

    result          = from_array(data, W=W, labels=labels)
    result.sfreq    = sfreq
    result.ch_names = labels
    return result
