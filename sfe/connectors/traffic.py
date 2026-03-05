# -*- coding: utf-8 -*-
"""
sfe/connectors/traffic.py — Traffic / sensor-network domain connector.

Handles ETT, PEMS-BAY, and generic sensor CSV files → normalized signals → SFEResult.

Preprocessing contract:
    Sensor signals are z-score normalized per column before SFE.
    This is not parameter tuning — it puts heterogeneous physical units
    on a common scale so Pearson correlation is meaningful.
    Raw sensor data (e.g. transformer temperature, loop detector flow)
    is assumed stationary; no differencing is applied.

Usage:
    from sfe.connectors.traffic import from_ett, from_pems, from_sensor_csv

    result = from_ett("ETTh1", n_cols=4, W=20)
    result = from_pems("pems_bay.h5", sensor_ids=[0, 1, 2], W=60)
    result = from_sensor_csv("sensors.csv", W=20)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from ..connect import from_array, from_dataframe, SFEResult

__all__ = [
    "from_ett",
    "from_pems",
    "from_sensor_csv",
]

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _zscore(data: np.ndarray) -> np.ndarray:
    """Z-score normalize each column. Columns with zero std are left as-is."""
    mu  = data.mean(axis=0)
    std = data.std(axis=0)
    std[std < 1e-10] = 1.0
    return (data - mu) / std


# ---------------------------------------------------------------------------
# ETT (Electricity Transformer Temperature)
# ---------------------------------------------------------------------------

def from_ett(
    split: str = "ETTh1",
    n_cols: int = 4,
    W: int = 20,
    normalize: bool = True,
    local_path: str | None = None,
) -> SFEResult:
    """
    Load the ETT dataset and run SFE.

    Tries HuggingFace datasets first, then a local CSV fallback.

    Parameters
    ----------
    split      : str    "ETTh1" or "ETTh2" (default "ETTh1")
    n_cols     : int    number of columns to use (default 4)
    W          : int    rolling window in timesteps (hourly data, default 20 ≈ ~1 day)
    normalize  : bool   z-score normalize per column (default True)
    local_path : str    path to a local ETTh1.csv / ETTh2.csv to use instead

    Returns
    -------
    SFEResult
    """
    import pandas as _pd

    data, cols = None, None

    # Local CSV override
    if local_path:
        p = Path(local_path)
        if p.exists():
            df   = _pd.read_csv(p)
            num  = df.select_dtypes(include="number")
            cols = num.columns[:n_cols].tolist()
            data = num[cols].values.astype(float)
            print(f"Loaded ETT from local file: {p.name}  cols={cols}")

    # HuggingFace
    if data is None:
        try:
            from datasets import load_dataset
            print(f"Loading {split} from HuggingFace …")
            ds   = load_dataset("thuml/ETT-small", split)
            df   = ds["train"].to_pandas()
            num  = df.select_dtypes(include="number")
            cols = num.columns[:n_cols].tolist()
            data = num[cols].values.astype(float)
            print(f"  {len(df)} rows  cols={cols}")
        except Exception as e:
            print(f"  HuggingFace failed ({e}), trying local ETTh1.csv …")

    # Local fallback
    if data is None:
        for fname in [f"{split}.csv", f"{split.lower()}.csv"]:
            p = Path(fname)
            if p.exists():
                df   = _pd.read_csv(p)
                num  = df.select_dtypes(include="number")
                cols = num.columns[:n_cols].tolist()
                data = num[cols].values.astype(float)
                print(f"  Loaded from {fname}  cols={cols}")
                break

    if data is None:
        raise RuntimeError(
            f"Could not load {split}. Provide local_path= or install: "
            "pip install datasets"
        )

    if normalize:
        data = _zscore(data)

    print(f"  Shape: {data.shape}  W={W}  normalize={normalize}")
    result = from_array(data, W=W, labels=cols)
    result.dataset = split
    return result


# ---------------------------------------------------------------------------
# PEMS-BAY / METR-LA (H5 format)
# ---------------------------------------------------------------------------

def from_pems(
    path: str,
    W: int = 60,
    sensor_ids: list[int] | None = None,
    n_sensors: int | None = None,
    normalize: bool = True,
    data_key: str = "df",
) -> SFEResult:
    """
    Load PEMS-BAY or METR-LA traffic data from H5 and run SFE.

    Parameters
    ----------
    path       : str           path to .h5 file (pems_bay.h5 / metr_la.h5)
    W          : int           rolling window in timesteps (5-min intervals; 12 = 1 hour)
    sensor_ids : list of int   specific sensor column indices to use
    n_sensors  : int           use the first n_sensors columns (alternative to sensor_ids)
    normalize  : bool          z-score per column (default True)
    data_key   : str           HDF5 key (default "df" for standard PEMS files)

    Returns
    -------
    SFEResult
    """
    import pandas as _pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PEMS file not found: {path}")

    try:
        df = _pd.read_hdf(path, key=data_key)
    except Exception:
        # Some distributions use a different key
        try:
            import h5py
            with h5py.File(path, "r") as f:
                keys = list(f.keys())
            raise KeyError(
                f"Key '{data_key}' not found in {path.name}. "
                f"Available: {keys}. Pass data_key= to override."
            )
        except ImportError:
            raise

    # Select sensors
    if sensor_ids is not None:
        df = df.iloc[:, sensor_ids]
    elif n_sensors is not None:
        df = df.iloc[:, :n_sensors]

    df   = df.select_dtypes(include="number").dropna()
    cols = list(df.columns)
    data = df.values.astype(float)

    if normalize:
        data = _zscore(data)

    print(f"Loaded PEMS: {path.name}  sensors={len(cols)}  "
          f"shape={data.shape}  W={W}")

    result = from_array(data, W=W, labels=[str(c) for c in cols])
    result.dataset = path.name
    return result


# ---------------------------------------------------------------------------
# Generic sensor CSV
# ---------------------------------------------------------------------------

def from_sensor_csv(
    path: str,
    W: int = 20,
    columns: list[str] | list[int] | None = None,
    normalize: bool = True,
    delimiter: str = ",",
    skip_rows: int = 0,
) -> SFEResult:
    """
    Load any sensor/time-series CSV and run SFE.

    Parameters
    ----------
    path      : str                  path to CSV
    W         : int                  rolling window in timesteps
    columns   : list of str/int      column names or indices to use (None = all numeric)
    normalize : bool                 z-score per column (default True)
    delimiter : str                  default ','
    skip_rows : int                  rows to skip before the header

    Returns
    -------
    SFEResult
    """
    import pandas as _pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sensor CSV not found: {path}")

    df = _pd.read_csv(
        path, sep=delimiter, skiprows=skip_rows,
        index_col=False,
    )
    numeric = df.select_dtypes(include="number")

    if columns is not None:
        if isinstance(columns[0], int):
            numeric = numeric.iloc[:, columns]
        else:
            available = [c for c in columns if c in numeric.columns]
            if not available:
                raise ValueError(
                    f"None of {columns} found in {list(numeric.columns)}"
                )
            numeric = numeric[available]

    numeric = numeric.dropna()
    if numeric.shape[1] < 2:
        raise ValueError(
            f"Need ≥2 numeric columns after selection, got {numeric.shape[1]}."
        )

    cols = list(numeric.columns)
    data = numeric.values.astype(float)

    if normalize:
        data = _zscore(data)

    print(f"Loaded sensor CSV: {path.name}  cols={cols}  shape={data.shape}  W={W}")

    result = from_array(data, W=W, labels=cols)
    result.dataset = path.name
    return result
