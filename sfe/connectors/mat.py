# -*- coding: utf-8 -*-
"""
sfe/connectors/mat.py — Generic MATLAB .mat connector for SFE.

Reads any .mat file, detects its structure, extracts a numeric time series,
and passes it through the standard cleaning + core pipeline.

No caller needs to know the data came from a .mat file.
The instrument is substrate-independent; this is just an ingestion path.

Supports:
    - MATLAB v5/v7 (.mat) via scipy.io.loadmat
    - MATLAB v7.3 HDF5-based .mat via h5py
    - KW51 trackedmodes format (struct with 'fn' field, shape (T, n_modes))
    - Any 2-D numeric array stored under any key
    - Struct arrays with numeric fields

Usage
-----
    from sfe.connectors.mat import from_mat

    # Auto-detect — let the connector find the right array
    result = from_mat("trackedmodes.mat", W=24)

    # Explicit key
    result = from_mat("trackedmodes.mat", key="fn", W=24)

    # Explicit columns (mode indices)
    result = from_mat("trackedmodes.mat", key="fn", columns=[0, 1, 2], W=24)

    # With timestamps for phase-boundary analysis
    result = from_mat("trackedmodes.mat", key="fn",
                      timestamps_key="time", W=24)

KW51 trackedmodes format
------------------------
    trackedmodes.mat contains a struct 'trackedmodes' with fields:
        fn      : (T, n_modes)  identified natural frequencies [Hz]
        zeta    : (T, n_modes)  damping ratios
        time    : (T, 1)        MATLAB serial datenums
        temp    : (T, 1)        temperature [°C]

    Modes 6, 9, 13 are the ones used in the SHM literature for this bridge.
    Pass columns=[5, 8, 12] (0-indexed) to select them explicitly, or leave
    columns=None to use all modes.

New boundary condition (SFE-11 Open Problem context)
-----------------------------------------------------
    The static-load strain result established: band_gap=61.66×, drho=0,
    rho*=0.936, regime="static lock".

    KW51 has three structural phases:
        Phase 1: nominal operation (before May 2019)
        Phase 2: retrofitting — intentional structural intervention
        Phase 3: post-retrofit operation

    If the instrument detects a dρ elevation or ρ* drop at the phase boundary,
    that is a new real-data regime: "structural transition under known intervention".
    This extends the SHM validation in the same self-consistent way as the f(N)
    fallback patch (Open Problem 3): a new boundary condition is identified and
    documented; no existing theorem is modified.

    The instrument should remain silent during nominal phases and fire at the
    transition — exactly the same logical structure as Branch A/B in finance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..connect import from_array, SFEResult

__all__ = ["from_mat", "mat_inspect", "load_mat_array"]


# ---------------------------------------------------------------------------
# Inspection utility — tell the caller what's in the file before loading
# ---------------------------------------------------------------------------

def mat_inspect(path: str) -> dict:
    """
    Inspect a .mat file and return a summary of its contents.

    Returns
    -------
    dict with keys:
        "format"   : str   "v5/v7" | "v7.3 (HDF5)"
        "keys"     : list  top-level variable names
        "shapes"   : dict  key → shape tuple (for numeric arrays)
        "dtypes"   : dict  key → dtype string
        "suggestion": str  best key to pass to from_mat()
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f".mat file not found: {path}")

    result = {"format": None, "keys": [], "shapes": {}, "dtypes": {}, "suggestion": None}

    # Try scipy first (v5/v7)
    try:
        import scipy.io as sio
        mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
        result["format"] = "v5/v7"
        for k, v in mat.items():
            if k.startswith("_"):
                continue
            result["keys"].append(k)
            if hasattr(v, "shape"):
                result["shapes"][k] = tuple(v.shape)
                result["dtypes"][k] = str(v.dtype)
        # Suggest the first 2D numeric key
        for k in result["keys"]:
            sh = result["shapes"].get(k)
            if sh and len(sh) == 2 and min(sh) >= 2:
                result["suggestion"] = k
                break
        return result
    except Exception as scipy_err:
        pass

    # Fall back to h5py (v7.3 HDF5)
    try:
        import h5py
        with h5py.File(path, "r") as f:
            result["format"] = "v7.3 (HDF5)"
            def _visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    result["keys"].append(name)
                    result["shapes"][name] = tuple(obj.shape)
                    result["dtypes"][name] = str(obj.dtype)
            f.visititems(_visit)
        for k in result["keys"]:
            sh = result["shapes"].get(k)
            if sh and len(sh) == 2 and min(sh) >= 2:
                result["suggestion"] = k
                break
        return result
    except Exception as h5_err:
        raise RuntimeError(
            f"Could not read {path.name} as v5/v7 or v7.3 .mat file.\n"
            f"  scipy error: {scipy_err}\n"
            f"  h5py  error: {h5_err}"
        )


def _print_inspect(info: dict, path: str) -> None:
    print()
    print(f"  .mat file : {Path(path).name}  (format: {info['format']})")
    print(f"  Keys      :")
    for k in info["keys"]:
        sh = info["shapes"].get(k, "?")
        dt = info["dtypes"].get(k, "?")
        marker = "  ← suggested" if k == info["suggestion"] else ""
        print(f"    {k:<30} shape={sh}  dtype={dt}{marker}")
    print()


# ---------------------------------------------------------------------------
# Array extractor — handles struct fields, plain arrays, HDF5 datasets
# ---------------------------------------------------------------------------

def load_mat_array(
    path: str,
    key: str | None = None,
    columns: list[int] | None = None,
    timestamps_key: str | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    """
    Load a numeric 2-D array from a .mat file.

    Parameters
    ----------
    path           : str
    key            : str | None   Variable name inside the .mat. None = auto-detect.
    columns        : list of int  Column (mode/channel) indices to keep. None = all.
    timestamps_key : str | None   Key for a timestamps vector (for phase analysis).
    verbose        : bool

    Returns
    -------
    (data, labels, timestamps)
        data       : ndarray (T, N)  float64, finite
        labels     : list of str
        timestamps : ndarray (T,) | None
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f".mat file not found: {path}")

    info = mat_inspect(str(path))
    if verbose:
        _print_inspect(info, str(path))

    # Resolve key
    if key is None:
        key = info["suggestion"]
        if key is None:
            raise ValueError(
                f"No 2-D numeric array found in {path.name}. "
                f"Available keys: {info['keys']}. Pass key= explicitly."
            )
        if verbose:
            print(f"  Auto-selected key: '{key}'")

    # Load raw value
    raw = _load_key(str(path), key, info["format"])
    data = _to_2d_array(raw, key)

    if verbose:
        print(f"  Loaded '{key}': shape={data.shape}  dtype={data.dtype}")

    # Column selection
    if columns is not None:
        if max(columns) >= data.shape[1]:
            raise ValueError(
                f"columns={columns} out of range for shape {data.shape}. "
                f"Max column index: {data.shape[1] - 1}."
            )
        data = data[:, columns]
        labels = [f"{key}_col{c}" for c in columns]
    else:
        labels = [f"{key}_col{c}" for c in range(data.shape[1])]

    # Try to extract mode labels from labels_m field (KW51 struct pattern)
    try:
        import scipy.io as sio
        mat = sio.loadmat(str(path), struct_as_record=False)
        top_key = key.split(".")[0]
        val = mat.get(top_key)
        if val is not None and hasattr(val, "shape") and val.shape == (1, 1):
            struct = val[0, 0]
            if hasattr(struct, "labels_m"):
                raw = struct.labels_m
                extracted = []
                for item in raw.ravel():
                    arr = np.asarray(item).ravel()
                    if arr.size > 0:
                        extracted.append(str(arr[0]))
                if len(extracted) == data.shape[1]:
                    labels = extracted
                    if verbose:
                        print(f"  Mode labels from labels_m: {labels}")
    except Exception:
        pass  # fallback to generic labels silently

    # Timestamps
    timestamps = None
    if timestamps_key is not None:
        try:
            ts_raw = _load_key(str(path), timestamps_key, info["format"])
            ts_arr = np.asarray(ts_raw, dtype=float).ravel()
            if len(ts_arr) == data.shape[0]:
                timestamps = ts_arr
                if verbose:
                    print(f"  Timestamps loaded from '{timestamps_key}': "
                          f"{len(timestamps)} values")
            else:
                print(f"  ⚠  Timestamp length ({len(ts_arr)}) ≠ data rows "
                      f"({data.shape[0]}); ignoring timestamps.")
        except Exception as e:
            print(f"  ⚠  Could not load timestamps from '{timestamps_key}': {e}")

    return data, labels, timestamps


def _load_key(path: str, key: str, fmt: str) -> Any:
    """Load a single key from a .mat file, handling struct fields and HDF5."""
    # Handle nested key (e.g. "modes.f")
    parts = key.split(".", 1)
    top_key = parts[0]
    sub_key = parts[1] if len(parts) > 1 else None

    if "HDF5" in fmt:
        import h5py
        with h5py.File(path, "r") as f:
            obj = f[top_key]
            if sub_key:
                obj = obj[sub_key]
            data = obj[()]
        return data
    else:
        import scipy.io as sio
        # Do NOT use squeeze_me — it mangles (1,1) struct arrays
        mat = sio.loadmat(path, struct_as_record=False)

        if top_key not in mat:
            raise KeyError(
                f"Key '{top_key}' not found. Available: "
                f"{[k for k in mat if not k.startswith('_')]}"
            )
        val = mat[top_key]

        # Unwrap MATLAB struct stored as (1,1) object array — standard MATLAB pattern
        # e.g. modes[0,0].f gives the (T, n_modes) frequency matrix
        if hasattr(val, "shape") and val.shape == (1, 1) and hasattr(val[0, 0], "_fieldnames"):
            val = val[0, 0]

        # MATLAB struct — access named field
        if sub_key:
            if not hasattr(val, sub_key):
                fields = [f for f in dir(val) if not f.startswith("_")]
                raise AttributeError(
                    f"Struct '{top_key}' has no field '{sub_key}'. "
                    f"Fields: {fields}"
                )
            return getattr(val, sub_key)

        # No subkey but it's a struct — try to extract numeric fields automatically
        if hasattr(val, "_fieldnames"):
            return _struct_to_array(val)

        return val


def _struct_to_array(struct) -> np.ndarray:
    """
    Convert a MATLAB struct with numeric fields to a 2D array.
    Each field becomes a column. Only float/int fields are included.
    """
    cols = []
    names = []
    for fname in struct._fieldnames:
        v = getattr(struct, fname)
        try:
            arr = np.asarray(v, dtype=float).ravel()
            cols.append(arr)
            names.append(fname)
        except (ValueError, TypeError):
            pass  # skip non-numeric fields

    if not cols:
        raise ValueError("Struct contains no numeric fields.")

    # Align lengths — use the minimum
    min_len = min(len(c) for c in cols)
    arr = np.column_stack([c[:min_len] for c in cols])
    return arr


def _to_2d_array(raw: Any, key: str) -> np.ndarray:
    """Convert whatever scipy/h5py returned into a finite float64 2-D array."""
    try:
        arr = np.asarray(raw, dtype=float)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Cannot convert '{key}' to numeric array: {e}\n"
            f"Value type: {type(raw)}. "
            f"If this is a struct, use key='structname.fieldname'."
        )

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if arr.ndim != 2:
        raise ValueError(
            f"'{key}' has shape {arr.shape}; need 2-D (T, N). "
            f"Reshape or select a subkey."
        )

    if arr.shape[1] < 2:
        raise ValueError(
            f"'{key}' has only {arr.shape[1]} column(s); SFE needs ≥2. "
            f"Use columns= to select multiple variables."
        )

    return arr


# ---------------------------------------------------------------------------
# Main connector
# ---------------------------------------------------------------------------

def from_mat(
    path: str,
    W: int,
    key: str | None = None,
    columns: list[int] | None = None,
    timestamps_key: str | None = None,
    labels: list[str] | None = None,
    normalize: bool = False,
    verbose: bool = True,
) -> SFEResult:
    """
    Load a .mat file and run SFE.

    This is a generic connector — it handles any .mat structure.
    The instrument is substrate-independent; this function is only
    responsible for getting a clean (T, N) float64 array into the pipeline.

    Parameters
    ----------
    path           : str
    W              : int           Rolling window in samples.
    key            : str | None    Variable name in the .mat (None = auto-detect).
                                   For struct fields: "structname.fieldname"
                                   e.g. "trackedmodes.fn" for KW51.
    columns        : list of int   Column indices to select (None = all).
                                   KW51 modes 6/9/13 → columns=[5, 8, 12]
    timestamps_key : str | None    Key for a timestamps vector.
                                   Used for phase-boundary analysis (attached
                                   to result.timestamps as raw values).
                                   KW51: timestamps_key="trackedmodes.time"
    labels         : list of str   Column labels. If None, auto-generated as
                                   "{key}_col0", "{key}_col1", ...
    normalize      : bool          Z-score normalize per column (default False).
                                   Set True for modal frequency data where
                                   absolute values differ across modes.
    verbose        : bool          Print loading summary (default True).

    Returns
    -------
    SFEResult with .timestamps, .mat_key, .mat_path attached.

    Examples
    --------
    # KW51 trackedmodes — all modes
    result = from_mat("trackedmodes.mat",
                      key="trackedmodes.fn",
                      timestamps_key="trackedmodes.time",
                      W=24, normalize=True)

    # KW51 — modes 6, 9, 13 only (literature standard)
    result = from_mat("trackedmodes.mat",
                      key="trackedmodes.fn",
                      columns=[5, 8, 12],
                      labels=["mode_6", "mode_9", "mode_13"],
                      timestamps_key="trackedmodes.time",
                      W=24, normalize=True)

    # Generic: any .mat with a 2-D array
    result = from_mat("mydata.mat", W=20)
    """
    data, auto_labels, timestamps = load_mat_array(
        path=path,
        key=key,
        columns=columns,
        timestamps_key=timestamps_key,
        verbose=verbose,
    )

    # Apply user labels if provided
    if labels is not None:
        if len(labels) != data.shape[1]:
            raise ValueError(
                f"labels has {len(labels)} entries but data has "
                f"{data.shape[1]} columns."
            )
        col_labels = labels
    else:
        col_labels = auto_labels

    # Optional z-score normalization
    # Modal frequencies differ in absolute value across modes (e.g. mode 6 ≈ 3 Hz,
    # mode 13 ≈ 8 Hz). Pearson correlation is scale-invariant, so normalization
    # does not affect ρ*, dρ, or r_eff. It only affects the raw data stored in
    # result.data. For modal data, normalize=True is recommended for interpretability.
    if normalize:
        mu  = data.mean(axis=0)
        std = data.std(axis=0)
        std[std < 1e-10] = 1.0
        data = (data - mu) / std
        if verbose:
            print(f"  Z-score normalized ({data.shape[1]} columns).")

    if verbose:
        print(f"  Running SFE: shape={data.shape}  W={W}  "
              f"labels={col_labels}")
        print()

    result = from_array(data, W=W, labels=col_labels)

    # Domain and metadata
    result.domain     = "shm"
    result.timestamps = timestamps

    # f(N) fallback reporting
    if getattr(result, "reff_corr_fallback", False):
        print(f"\n  ⚠  f(N) over-correction detected at N={result.N}: "
              f"raw joint mean ({result.reff_corr:.4f}) used. "
              f"See SFE-11 Open Problem 3.")

    return result