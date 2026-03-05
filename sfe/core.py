# sfe/core.py — numerical core for the SFE instrument.
# Requires finite float64 input (no NaN/Inf); data cleaning occurs in sfe/connect.py.
# Raises ValueError for structural violations.
# Experimental research code: https://doi.org/10.5281/zenodo.18869381
# Provided "as is". Use at your own risk.

from __future__ import annotations
import numpy as np
__all__ = [
    "rolling_corr", "rolling_drho", "reff", "reff_joint",
    "pair_table", "nonstationarity_flag", "f_N", "OPERATING_ENVELOPE",
]
OPERATING_ENVELOPE = {
    "reliable_rho_min": 0.45,
    "degraded_rho_max": 0.20,
    "bias_threshold":   0.04,
}
def _prefix(x: np.ndarray):
    return np.concatenate([[0.0], np.cumsum(x)])
def _wsum(pfx: np.ndarray, i: np.ndarray, W: int) -> np.ndarray:
    return pfx[i] - pfx[i - W]
def rolling_corr(x, y, W: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError(f"x and y must have equal length, got {len(x)} vs {len(y)}.")
    if W < 2 or W > len(x):
        raise ValueError(f"W must satisfy 2 <= W <= {len(x)}, got W={W}.")
    T   = len(x)
    i   = np.arange(W, T + 1)          
    Sx  = _wsum(_prefix(x),   i, W)
    Sy  = _wsum(_prefix(y),   i, W)
    Sx2 = _wsum(_prefix(x*x), i, W)
    Sy2 = _wsum(_prefix(y*y), i, W)
    Sxy = _wsum(_prefix(x*y), i, W)
    num = W * Sxy - Sx * Sy
    den = np.sqrt(np.maximum((W * Sx2 - Sx**2) * (W * Sy2 - Sy**2), 0.0))
    rho = np.where(den > 1e-9, num / den, 0.0)
    return np.concatenate([np.zeros(W - 1), rho])
def rolling_drho(rho, W: int) -> np.ndarray:
    rho = np.asarray(rho, dtype=float)
    if W < 2 or W > len(rho):
        raise ValueError(f"W must satisfy 2 <= W <= {len(rho)}, got W={W}.")
    T = len(rho)
    i = np.arange(W, T + 1)
    S1 = _wsum(_prefix(rho),     i, W)
    S2 = _wsum(_prefix(rho*rho), i, W)
    var = np.maximum(S2 / W - (S1 / W) ** 2, 0.0)
    return np.concatenate([np.zeros(W - 1), var])
def reff(rho, sigma1: float = 1.0, sigma2: float = 1.0):
    rho = np.asarray(rho, dtype=float)
    s1, s2 = sigma1 ** 2, sigma2 ** 2
    num = 2.0 * (s1 + s2) ** 2
    den = (s1 + s2) ** 2 + (s1 - s2) ** 2 + 4.0 * rho ** 2 * s1 * s2
    return num / den
def reff_joint(data, W: int) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D (T, N), got {data.shape}.")
    T, _ = data.shape
    out  = []
    for t in range(W, T + 1, W):
        cov = np.cov(data[t - W:t].T)
        if cov.ndim < 2:
            cov = np.array([[float(cov)]])
        ev = np.linalg.eigvalsh(cov)
        ev = ev[ev > 1e-10]
        p  = ev / ev.sum() if len(ev) else np.array([1.0])
        out.append(float(np.exp(-np.sum(p * np.log(p)))))
    return np.array(out)
def f_N(n) -> float | np.ndarray:
    return -0.106 * np.log(np.asarray(n, dtype=float)) + 1.070
def nonstationarity_flag(drho_val: float, rho_star: float, W: int) -> bool:
    return bool(drho_val > (1.0 - rho_star ** 2) ** 2 / W)
def pair_table(data, W: int, labels=None, skip=None) -> list[dict]:
    data   = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D (T, N), got {data.shape}.")
    T, N   = data.shape
    skip   = skip if skip is not None else W
    labels = labels or [str(k) for k in range(N)]
    i_idx = np.arange(W, T + 1)
    pfx_s  = {} 
    pfx_s2 = {} 
    col_S  = {}  
    col_S2 = {}
    for k in range(N):
        x = data[:, k]
        pfx_s[k]  = _prefix(x)
        pfx_s2[k] = _prefix(x * x)
        col_S[k]  = _wsum(pfx_s[k],  i_idx, W)
        col_S2[k] = _wsum(pfx_s2[k], i_idx, W)
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            Sxy = _wsum(_prefix(data[:, i] * data[:, j]), i_idx, W)
            num = W * Sxy - col_S[i] * col_S[j]
            den = np.sqrt(np.maximum(
                (W * col_S2[i] - col_S[i]**2) *
                (W * col_S2[j] - col_S[j]**2), 0.0
            ))
            rho_core = np.where(den > 1e-9, num / den, 0.0)
            rho_ts   = np.concatenate([np.zeros(W - 1), rho_core])
            drho_ts = rolling_drho(rho_ts, W)
            rho_s  = np.abs(rho_ts[skip:])
            drho_s = drho_ts[skip:]
            rho_star  = float(rho_s.mean())
            drho_mean = float(drho_s.mean())
            reff_mean = float(reff(rho_ts[skip:]).mean())
            zone = ("reliable" if rho_star > OPERATING_ENVELOPE["reliable_rho_min"]
                    else "degraded" if rho_star < OPERATING_ENVELOPE["degraded_rho_max"]
                    else "marginal")
            n_blk = len(rho_s) // W
            ns_pct = 0.0
            if n_blk > 1:
                blocks = rho_s[:n_blk * W].reshape(n_blk, W)
                bv     = blocks.var(axis=1)
                hi     = np.maximum(bv[:-1], bv[1:])
                lo     = np.minimum(bv[:-1], bv[1:])
                ns_pct = 100.0 * float(np.sum(hi / (lo + 1e-12) > 3.0)) / (n_blk - 1)
            pairs.append(dict(
                label=f"{labels[i]}-{labels[j]}", i=i, j=j,
                rho=rho_ts, drho=drho_ts,
                rho_star=rho_star, drho_mean=drho_mean, reff_mean=reff_mean,
                zone=zone, nonstationary_pct=ns_pct,
            ))
    pairs.sort(key=lambda p: -p["rho_star"])
    return pairs