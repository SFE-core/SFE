# -*- coding: utf-8 -*-
"""
sfe/analysis/absorption.py — Absorption signature statistic.

Computes cross-pair dispersion of drho (CV_drho) as defined in SFE-Finance.
Stripped from find.py visualization script.

Usage
-----
    from sfe.analysis.absorption import compute_absorption_signature
    sig = compute_absorption_signature(result.pairs)
    # sig.keys(): mu_drho, sigma_drho, cv_drho, rho_star_mean, n_pairs, classification
"""

from __future__ import annotations
import numpy as np

__all__ = ["compute_absorption_signature"]

# Thresholds from SFE-Finance calibration
TAU_CV     = 0.20   # below = homogeneous
TAU_R      = 3.0    # R_drho = mu / baseline above = elevated
BASELINE   = 0.01537  # Lehman mu(drho) — empirical baseline


def compute_absorption_signature(pairs: list[dict]) -> dict:
    """
    Compute cross-pair absorption signature from a list of pair dicts.

    Parameters
    ----------
    pairs : list of dicts with keys 'drho_mean' and 'rho_star'
            (standard SFEResult.pairs format)

    Returns
    -------
    dict with keys:
        mu_drho        : float  — mean of drho across pairs
        sigma_drho     : float  — std of drho across pairs
        cv_drho        : float  — coefficient of variation (sigma/mu)
        R_drho         : float  — mu / baseline (Lehman)
        rho_star_mean  : float  — mean rho* across pairs
        n_pairs        : int
        classification : str    — 'homogeneous' | 'dispersed'
        absorption     : bool   — True if CV < TAU_CV and R > TAU_R
    """
    if not pairs:
        return {
            "mu_drho": 0.0, "sigma_drho": 0.0, "cv_drho": 0.0,
            "R_drho": 0.0, "rho_star_mean": 0.0, "n_pairs": 0,
            "classification": "insufficient_data", "absorption": False,
        }

    drho_vals = np.array([p["drho_mean"] for p in pairs], dtype=float)
    rho_vals  = np.array([p["rho_star"]  for p in pairs], dtype=float)

    mu    = float(drho_vals.mean())
    sigma = float(drho_vals.std())
    cv    = sigma / mu if mu > 1e-10 else 0.0
    R     = mu / BASELINE if BASELINE > 0 else 0.0

    classification = "homogeneous" if cv < TAU_CV else "dispersed"
    absorption     = (cv < TAU_CV) and (R > TAU_R)

    return {
        "mu_drho":       mu,
        "sigma_drho":    sigma,
        "cv_drho":       cv,
        "R_drho":        R,
        "rho_star_mean": float(rho_vals.mean()),
        "n_pairs":       len(pairs),
        "classification": classification,
        "absorption":    absorption,
    }
