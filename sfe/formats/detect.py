# -*- coding: utf-8 -*-
"""
sfe/formats/detect.py — Format detection and connector routing.

Single entry point: load(path, domain=None, W=..., **kwargs) → SFEResult

Format is detected from file extension.
Domain hint refines routing when the same extension maps to multiple connectors
(e.g. .h5 can be traffic/PEMS or EEG depending on domain).
All kwargs are forwarded to the underlying connector unchanged.

Format map
----------
Extension   Domain hint     Connector
---------   -----------     ---------
.csv        "strain"        connectors.strain.from_strain_csv
.csv        "finance"       connectors.finance.from_price_csv
.csv        "eeg"           connectors.eeg.from_eeg_csv
.csv        "traffic"       connectors.traffic.from_sensor_csv
.csv        None            connectors.strain.from_strain_csv  (default)
.mat        any             connectors.mat.from_mat
.edf        any             connectors.eeg.from_edf
.h5/.hdf5   "traffic"       connectors.traffic.from_pems
.h5/.hdf5   "eeg"           connectors.eeg.from_h5
.h5/.hdf5   None            connectors.traffic.from_pems  (default)
"""

from __future__ import annotations

from pathlib import Path

from ..connect import SFEResult

__all__ = ["load"]


def load(
    path: str,
    W: int,
    domain: str | None = None,
    **kwargs,
) -> SFEResult:
    """
    Load any supported file and run SFE.

    Format is detected from file extension.
    Domain hint refines connector selection when the extension is ambiguous.

    Parameters
    ----------
    path   : str           path to the data file
    W      : int           rolling window in samples (required)
    domain : str | None    optional hint: "strain" | "finance" | "eeg" |
                           "traffic" | "shm" — refines connector selection
                           for ambiguous extensions (.csv, .h5)
    **kwargs               forwarded verbatim to the underlying connector
                           e.g. key=, columns=, timestamps_key=, normalize=,
                           tickers=, channels=, sensor_ids=

    Returns
    -------
    SFEResult

    Examples
    --------
    from sfe.formats import load

    # KW51 bridge modal data
    result = load("trackedmodes.mat", W=24, domain="shm",
                  key="trackedmodes.fn",
                  columns=[5, 8, 12],
                  labels=["mode_6", "mode_9", "mode_13"],
                  timestamps_key="trackedmodes.time",
                  normalize=True)

    # Strain rosette CSV
    result = load("sampledata.csv", W=60, domain="strain", auto=True)

    # Finance price CSV
    result = load("prices.csv", W=20, domain="finance")

    # EEG EDF
    result = load("S001R04.edf", W=160, channels=["C3", "C4"])

    # PEMS traffic H5
    result = load("pems_bay.h5", W=12, domain="traffic", n_sensors=5)
    """
    p      = Path(path)
    ext    = p.suffix.lower()
    domain_l = domain.lower() if domain else None

    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # ------------------------------------------------------------------
    # .mat — always goes to the mat connector regardless of domain
    # ------------------------------------------------------------------
    if ext == ".mat":
        from ..connectors.mat import from_mat
        return from_mat(path, W=W, **kwargs)

    # ------------------------------------------------------------------
    # .edf — always EEG
    # ------------------------------------------------------------------
    if ext == ".edf":
        from ..connectors.eeg import from_edf
        return from_edf(path, W=W, **kwargs)

    # ------------------------------------------------------------------
    # .h5 / .hdf5 — traffic (default) or EEG
    # ------------------------------------------------------------------
    if ext in (".h5", ".hdf5"):
        if domain_l == "eeg":
            from ..connectors.eeg import from_h5
            return from_h5(path, W=W, **kwargs)
        else:
            # Default: PEMS-style traffic
            from ..connectors.traffic import from_pems
            return from_pems(path, W=W, **kwargs)

    # ------------------------------------------------------------------
    # .csv — domain-specific routing
    # ------------------------------------------------------------------
    if ext == ".csv":
        if domain_l == "finance":
            from ..connectors.finance import from_price_csv
            return from_price_csv(path, W=W, **kwargs)

        if domain_l == "eeg":
            from ..connectors.eeg import from_eeg_csv
            return from_eeg_csv(path, W=W, **kwargs)

        if domain_l == "traffic":
            from ..connectors.traffic import from_sensor_csv
            return from_sensor_csv(path, W=W, **kwargs)

        if domain_l in ("strain", "shm", None):
            # strain is the default CSV connector — handles DATA_START,
            # SampleRate header, device label detection
            from ..connectors.strain import from_strain_csv
            return from_strain_csv(path, W=W, **kwargs)

        # Unknown domain — fall back to strain CSV (most permissive parser)
        from ..connectors.strain import from_strain_csv
        return from_strain_csv(path, W=W, **kwargs)

    # ------------------------------------------------------------------
    # Unsupported extension
    # ------------------------------------------------------------------
    supported = ".mat, .edf, .h5, .hdf5, .csv"
    raise ValueError(
        f"Unsupported file extension '{ext}' for '{p.name}'. "
        f"Supported: {supported}. "
        f"To load a custom format, call the connector directly."
    )
