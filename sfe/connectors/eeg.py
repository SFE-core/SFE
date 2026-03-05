# -*- coding: utf-8 -*-
"""
sfe/connectors/eeg.py — EEG domain connector.

Handles EDF / H5 / CSV biosignal files → channel extraction → SFEResult.
Supports PhysioNet-style EDF files (MNE backend) and pre-exported CSVs.

Preprocessing contract:
    Raw EEG is passed directly to SFE core — no filtering, no z-scoring.
    W is specified in samples; caller converts from seconds using sfreq.
    Event extraction and epoch stacking remain in the stress-test layer.

Usage:
    from sfe.connectors.eeg import from_edf, from_eeg_csv, from_h5

    result = from_edf("S001R04.edf", channels=["C3", "C4"], W=160)
    result = from_edf("S001R04.edf", W=160)          # auto-picks motor channels
    result = from_eeg_csv("eeg_motor.csv", W=160)
    result = from_h5("epochs.h5", key="data", W=160)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from ..connect import from_array, SFEResult

__all__ = [
    "from_edf",
    "from_eeg_csv",
    "from_h5",
    "pick_channels",
    "W_from_seconds",
]

# ---------------------------------------------------------------------------
# Channel selection helpers
# ---------------------------------------------------------------------------

# Preferred motor-cortex channel pairs, tried in order
_MOTOR_CHANNEL_PRIORITY = [
    ["C3", "C4"],
    ["C3..", "C4.."],
    ["EEG C3", "EEG C4"],
]


def pick_channels(available: list[str], targets: list[str] | None = None) -> list[str]:
    """
    Select channels from available list.

    If targets is given, find the best prefix match for each target.
    If targets is None, try motor-cortex defaults then fall back to first two.

    Parameters
    ----------
    available : list of str   all channel names in the file
    targets   : list of str   desired channel names (prefix match)

    Returns
    -------
    list of str   matched channel names (from available)
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
        # Partial match — warn and continue to fallbacks
        print(f"  ⚠  Only {len(found)}/{len(targets)} targets matched; "
              f"trying motor defaults.")

    # Motor cortex defaults
    for pair in _MOTOR_CHANNEL_PRIORITY:
        found = [_prefix_match(t, available) for t in pair]
        found = [c for c in found if c is not None]
        if len(found) >= 2:
            return found

    # Last resort: first two channels in file
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
    return_meta : bool          if True, also return a metadata dict with
                                sfreq, ch_names, events_df, n_samples

    Returns
    -------
    SFEResult  (or (SFEResult, meta) if return_meta=True)

    Example
    -------
    result = from_edf("S001R04.edf", channels=["C3", "C4"], W=160)
    result, meta = from_edf("S001R04.edf", W=160, return_meta=True)
    events = meta["events_df"]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EDF file not found: {path}")

    # ── MNE backend ──
    try:
        import mne
        mne.set_log_level("WARNING")

        raw      = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
        sfreq    = float(raw.info["sfreq"])
        avail    = raw.ch_names
        selected = pick_channels(avail, channels)

        picks       = mne.pick_channels(avail, include=selected)
        data, _     = raw[picks, :]           # shape (n_ch, T)
        data        = data.T                  # → (T, n_ch)

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

        print(f"Loaded EDF via MNE: {path.name}")
        print(f"  Channels : {selected}")
        print(f"  sfreq    : {sfreq} Hz  |  {data.shape[0]} samples "
              f"({data.shape[0]/sfreq:.1f} s)")

    except ImportError:
        # ── pyedflib fallback ──
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

        data      = np.column_stack(arrays)   # (T, n_ch)
        events_df = None

        print(f"Loaded EDF via pyedflib: {path.name}")
        print(f"  Channels : {selected}  sfreq={sfreq} Hz")

    result = from_array(data, W=W, labels=selected)

    # Attach metadata for event-locked analysis in stress-test layer
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

    print(f"Loaded EEG CSV: {path.name}  channels={selected}  shape={data.shape}")

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

    Notes
    -----
    The H5 dataset at `key` must be shape (T, N) — timesteps × channels.
    If it's (N, T), transpose before passing or use numpy directly.
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

    print(f"Loaded H5: {path.name}  key='{key}'  shape={data.shape}")

    result          = from_array(data, W=W, labels=labels)
    result.sfreq    = sfreq
    result.ch_names = labels
    return result
