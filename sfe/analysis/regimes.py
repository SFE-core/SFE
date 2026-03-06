# -*- coding: utf-8 -*-
"""
sfe/analysis/regimes.py — Single source of truth for all validated coupling regimes.

This module mirrors the cross-domain summary table from SFE-11 and is the
reference point for:
    - The AI layer       (context for interpretation)
    - figures.py         (cross-domain reference anchors on phase portraits)
    - analysis modules   (detection thresholds)
    - classify_result()  (where does a new result land?)

Two registries:
    REGIMES          — validated domains: where the instrument works,
                       characterized by substrate, mechanism, and signature.
                       New domains are added here when validated, not before.

    KNOWN_LIMITS     — real-data boundary conditions of the validated regime.
                       Each entry is a characterized limit of the instrument,
                       identified during implementation or empirical validation.
                       Same class of self-detection as the NS% flag:
                       the instrument identifies its own boundary using only
                       quantities already computed.

The core operating envelope (OPERATING_ENVELOPE) is defined in core.py
and imported here. It is never redefined.

Detection thresholds (CRISIS_THRESHOLDS, TRANSITION_THRESHOLDS) live here
because they are empirical calibrations, not mathematical results.
They belong to the validated knowledge layer, not the core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

# OPERATING_ENVELOPE is a core result — import, never redefine
from ..core import OPERATING_ENVELOPE

__all__ = [
    "REGIMES",
    "KNOWN_LIMITS",
    "OPERATING_ENVELOPE",
    "CRISIS_THRESHOLDS",
    "TRANSITION_THRESHOLDS",
    "Regime",
    "KnownLimit",
    "get_regime",
    "ref_points",
    "domain_context",
    "classify_result",
    "ClassificationResult",
]


# ---------------------------------------------------------------------------
# Detection thresholds
# Empirical calibrations — single location, all analysis modules import here.
# ---------------------------------------------------------------------------

# Finance crisis detection (Proposition 12, SFE-11)
# Branch A calibrated on COVID-19 (1 event, 4 homogeneous assets).
# Branch B validated on 2008 Lehman (1 event, mixed sectors).
# Both require validation across additional crises — see Open Problem 6.
CRISIS_THRESHOLDS = {
    "rho_delta_min":    0.10,
    "bandgap_mult_min": 1.50,
    "reff_delta_max":  -0.10,
    "pairs_pct_min":    0.50,
}

# Structural transition detection (SHM)
# Uncalibrated — pending KW51 empirical run.
# Will be updated when KW51 phase boundary analysis produces ground truth.
TRANSITION_THRESHOLDS = {
    "rho_delta_min":   -0.05,
    "drho_spike_mult":  2.0,
    "reff_delta_min":   0.05,
}


# ---------------------------------------------------------------------------
# Regime dataclass
# ---------------------------------------------------------------------------

@dataclass
class Regime:
    """
    One row of the cross-domain validation table (SFE-11 Section 4).

    rho_star, band_gap, reff, drho are (min, max) observed ranges
    from the validated run. band_gap is None if not characterized.
    ref_point is (rho_star, drho) used as the phase portrait anchor.
    pending=True means the domain has not yet been validated —
    excluded from classification and ref_points.
    """
    domain     : str
    mechanism  : str
    substrate  : str
    timescale  : str
    rho_star   : Tuple[float, float]
    band_gap   : Optional[Tuple[float, float]]
    reff       : Tuple[float, float]
    drho       : Tuple[float, float]
    detection  : str
    ref_point  : Tuple[float, float]
    ref_color  : str
    notes            : str  = ""
    pending          : bool = False
    cross_domain_ref : bool = True   # show as anchor on ALL phase portraits


# ---------------------------------------------------------------------------
# Known limits dataclass
# ---------------------------------------------------------------------------

@dataclass
class KnownLimit:
    """
    A real-data boundary condition of the validated regime.

    The instrument identifies its own limits using only quantities already
    computed — same class of self-detection as the NS% flag.
    Each limit maps to an Open Problem in the paper.
    """
    id                  : str
    open_problem        : str
    description         : str
    discovered          : str
    status              : str   # "characterized" | "pending"
    instrument_response : str


# ---------------------------------------------------------------------------
# Validated regime registry
# ---------------------------------------------------------------------------

REGIMES: list[Regime] = [

    Regime(
        domain    = "ETTh1",
        mechanism = "Persistent lock",
        substrate = "Electrical infrastructure",
        timescale = "Hourly",
        rho_star  = (0.950, 0.975),
        band_gap  = (3.5, 5.0),
        reff      = (1.00, 1.10),
        drho      = (0.003, 0.005),
        detection = "reliable",
        ref_point = (0.963, 0.004),
        ref_color = "#ff6b35",
        notes     = "HUFL-MUFL pair. Persistent coupling from shared transformer load. "
                    "Granger causality precedes dρ lock-in by mean 4.0h.",
    ),

    Regime(
        domain    = "METR-LA",
        mechanism = "Sync transition",
        substrate = "Urban traffic network",
        timescale = "5-minute intervals",
        rho_star  = (0.40, 0.45),
        band_gap  = (4.5, 6.0),
        reff      = (2.0, 3.0),
        drho      = (0.040, 0.055),
        detection = "indicative",
        ref_point = (0.426, 0.047),
        ref_color = "#00b4d8",
        notes     = "207 sensors. ρ* in marginal zone — results indicative. "
                    "r_eff plateau at N≈50 reveals intrinsic network dimensionality. "
                    "See Open Problem 7.",
    ),

    Regime(
        domain    = "EEG motor cortex",
        mechanism = "Disruption/relock",
        substrate = "Neural tissue",
        timescale = "Seconds (160 Hz)",
        rho_star  = (0.72, 0.89),
        band_gap  = None,
        reff      = (1.10, 1.30),
        drho      = (0.003, 0.006),
        detection = "9/10 subjects replicated",
        ref_point = (0.831, 0.004),
        ref_color = "#c77dff",
        notes     = "Two-phase structure: dρ spike at onset, drop below baseline. "
                    "Lead time ~3.1s. pct_drop > 20% = structure present. "
                    "See Open Problem 8.",
    ),

    Regime(
        domain    = "Finance background",
        mechanism = "Sector coupling",
        substrate = "Financial markets",
        timescale = "Daily (20-day window)",
        rho_star  = (0.60, 0.70),
        band_gap  = (6.0, 8.0),
        reff      = (1.70, 2.10),
        drho      = (0.010, 0.030),
        detection = "reliable",
        ref_point = (0.654, 0.020),
        ref_color = "#ffd60a",
        notes     = "Full-period background. Baseline for crisis detection.",
    ),

    Regime(
        domain    = "Finance — COVID crash 2020",
        mechanism = "Acute homogeneous shock (Branch A)",
        substrate = "Financial markets",
        timescale = "Daily",
        rho_star  = (0.90, 0.93),
        band_gap  = (18.0, 23.0),
        reff      = (1.30, 1.50),
        drho      = (0.015, 0.035),
        detection = "Branch A ✓",
        ref_point = (0.915, 0.025),
        ref_color = "#ff4444",
        cross_domain_ref = False,
        notes     = "Band gap explosion ≥1.5× background is the discriminating signature. "
                    "CRISIS_THRESHOLDS Branch A calibrated on this event. "
                    "See Open Problem 6.",
    ),

    Regime(
        domain    = "Finance — Lehman 2008",
        mechanism = "Acute heterogeneous contagion (Branch B)",
        substrate = "Financial markets",
        timescale = "Daily",
        rho_star  = (0.68, 0.72),
        band_gap  = (5.50, 7.50),
        reff      = (1.80, 2.00),
        drho      = (0.020, 0.040),
        detection = "Branch B ✓",
        ref_point = (0.695, 0.030),
        ref_color = "#ff8800",
        cross_domain_ref = False,
        notes     = "Band gap stable — variance diffused across modes. "
                    "r_eff collapse + dρ elevation on >50% of pairs. "
                    "CRISIS_THRESHOLDS Branch B validated on this event.",
    ),

    Regime(
        domain    = "Finance — dot-com 2000-02",
        mechanism = "Gradual correction",
        substrate = "Financial markets",
        timescale = "Daily",
        rho_star  = (0.50, 0.56),
        band_gap  = (2.50, 3.20),
        reff      = (2.60, 2.90),
        drho      = (0.008, 0.020),
        detection = "correctly silent ✓",
        ref_point = (0.530, 0.015),
        ref_color = "#aaaaaa",
        cross_domain_ref = False,
        notes     = "Neither branch fires. Gradual valuation correction "
                    "is not a synchronization event.",
    ),

    Regime(
        domain    = "Strain rosette",
        mechanism = "Static lock",
        substrate = "Structural mechanics",
        timescale = "1 Hz (minutes to hours)",
        rho_star  = (0.920, 1.000),
        band_gap  = (55.0, 70.0),
        reff      = (1.05, 1.15),
        drho      = (0.0, 0.000005),
        detection = "36/36 pairs, dρ=0 for 23h",
        ref_point = (0.936, 0.000001),
        ref_color = "#00e676",
        notes     = "Highest band gap in cross-domain study (61.66×). "
                    "Upper bound of band-gap range; lower bound of dρ range. "
                    "f(N) over-correction at N=9 — see KNOWN_LIMITS KL-01.",
    ),

    # Pending — not yet validated
    Regime(
        domain    = "SHM — KW51 bridge",
        mechanism = "Structural transition (retrofitting)",
        substrate = "Structural mechanics",
        timescale = "Hourly (15-month campaign)",
        rho_star  = (0.0, 1.0),
        band_gap  = (0.0, 100.0),
        reff      = (1.0, 10.0),
        drho      = (0.0, 1.0),
        detection = "pending",
        ref_point = (0.0, 0.0),
        ref_color = "#888888",
        pending   = True,
        notes     = "Three phases: nominal / retrofitting / post-retrofit. "
                    "TRANSITION_THRESHOLDS uncalibrated — see KNOWN_LIMITS KL-02. "
                    "Will become a validated row when KW51 run completes.",
    ),

]


# ---------------------------------------------------------------------------
# Known limits registry
# ---------------------------------------------------------------------------

KNOWN_LIMITS: list[KnownLimit] = [

    KnownLimit(
        id           = "KL-01",
        open_problem = "Open Problem 3 — N-observer estimator boundary",
        description  = "f(N) correction over-corrects in the high band-gap, "
                       "single-dominant-mode regime (λ₁/λ₂ = 61.66×, N=9). "
                       "f(9) = 0.837 produces reff_corr = 0.916, violating the "
                       "physical bound reff ≥ 1. The high band-gap regime is the "
                       "boundary at which f(N) does not apply; the joint entropy "
                       "estimator tracks ground truth directly without correction.",
        discovered   = "Strain connector port, N=9, band_gap=61.66×",
        status       = "characterized",
        instrument_response = "reff_corr_fallback=True; raw joint mean returned. "
                              "Self-detected via physical bound reff ≥ 1 — "
                              "no external ground truth required.",
    ),

    KnownLimit(
        id           = "KL-02",
        open_problem = "New domain — SHM structural transition (KW51)",
        description  = "TRANSITION_THRESHOLDS are initial estimates for structural "
                       "transition detection. The regime signatures are hypothesized "
                       "from instrument theory but not yet empirically characterized "
                       "on KW51 ground-truth phase labels.",
        discovered   = "KW51 connector port — pending first run",
        status       = "pending",
        instrument_response = "detect_structural_transition() returns a calibration "
                              "warning until KW51 run completes and thresholds "
                              "are updated.",
    ),

]


# ---------------------------------------------------------------------------
# Lookup utilities
# ---------------------------------------------------------------------------

def get_regime(domain: str) -> Regime | None:
    """Return the Regime for a domain name (case-insensitive partial match)."""
    domain_l = domain.lower()
    for r in REGIMES:
        if domain_l in r.domain.lower():
            return r
    return None


def ref_points() -> list[tuple]:
    """
    Return (rho_star, drho, color, label) for all validated cross-domain anchors.
    Used by figures.py for phase portrait cross-domain anchors.

    Label format: "{domain} — {mechanism}"
    e.g. "ETTh1 — Persistent lock"
         "METR-LA — Sync transition"
         "EEG motor cortex — Disruption/relock"

    This format is substrate-independent: it tells the analyst what coupling
    geometry the anchor represents, not which system produced it.
    Only regimes with cross_domain_ref=True are included — domain-specific
    results (e.g. finance crash events) are excluded.
    """
    return [
        (
            r.ref_point[0],
            r.ref_point[1],
            r.ref_color,
            r.mechanism,
        )
        for r in REGIMES
        if not r.pending
        and r.cross_domain_ref
        and r.ref_point != (0.0, 0.0)
    ]


def domain_context(domain: str) -> str:
    """
    Return a plain-text domain context string for the AI layer.
    Replaces the hardcoded _DOMAIN_CONTEXT dict in ai.py.
    """
    r = get_regime(domain)
    if r is None:
        return (
            f"Data are multivariate time series from domain '{domain}'. "
            f"No validated regime entry found — interpret coupling structure "
            f"in general terms against the operating envelope."
        )
    return (
        f"Data are from the '{r.domain}' domain ({r.substrate}). "
        f"Characteristic timescale: {r.timescale}. "
        f"Known mechanism: {r.mechanism}. "
        f"Validated regime: ρ*∈{r.rho_star}, "
        f"band_gap∈{r.band_gap if r.band_gap else 'not characterized'}, "
        f"r_eff∈{r.reff}, dρ∈{r.drho}. "
        f"Detection: {r.detection}. "
        f"Notes: {r.notes}"
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Returned by classify_result()."""
    matched_regime : Regime | None
    in_envelope    : bool
    zone           : str        # "reliable" | "marginal" | "degraded"
    is_uncharted   : bool       # True = outside all validated regimes
    notes          : list[str] = field(default_factory=list)


def classify_result(result) -> ClassificationResult:
    """
    Compare a SFEResult against all validated regimes.

    If no regime matches, the result is outside the validated range.
    The instrument reports this explicitly — it does not fail silently.
    The result may be a candidate for a new validated regime row,
    or a new entry in KNOWN_LIMITS if it reveals an instrument boundary.

    Parameters
    ----------
    result : SFEResult

    Returns
    -------
    ClassificationResult
    """
    s     = result.summary_dict()
    rho   = s["rho_star_mean"]
    bg    = s["band_gap"]
    rc    = s["reff_corr"]
    drho  = s["drho_mean"]
    notes = []

    env = OPERATING_ENVELOPE
    if rho > env["reliable_rho_min"]:
        zone = "reliable"
    elif rho < env["degraded_rho_max"]:
        zone = "degraded"
    else:
        zone = "marginal"

    in_envelope = zone in ("reliable", "marginal")

    # Match against validated regimes only
    best_match = None
    for regime in REGIMES:
        if regime.pending:
            continue
        rho_ok  = regime.rho_star[0] <= rho <= regime.rho_star[1]
        reff_ok = regime.reff[0]     <= rc  <= regime.reff[1]
        bg_ok   = (regime.band_gap is None or
                   regime.band_gap[0] <= bg <= regime.band_gap[1])
        if rho_ok and reff_ok and bg_ok:
            best_match = regime
            break

    if best_match is not None:
        notes.append(
            f"Matches validated regime: {best_match.domain} "
            f"({best_match.mechanism})."
        )
        return ClassificationResult(
            matched_regime=best_match,
            in_envelope=in_envelope,
            zone=zone,
            is_uncharted=False,
            notes=notes,
        )

    # Outside all validated regimes
    notes.append(
        f"Outside all validated regimes: "
        f"ρ*={rho:.3f}, band_gap={bg:.2f}×, r_eff={rc:.3f}, dρ={drho:.6f}."
    )
    notes.append(
        "Instrument operating outside characterized range. "
        "Results are indicative. "
        "If reproducible, candidate for a new validated regime row in regimes.py."
    )
    if not in_envelope:
        notes.append(
            f"Zone: {zone}. "
            "Estimator bias may exceed 4% — see OPERATING_ENVELOPE in core.py."
        )

    return ClassificationResult(
        matched_regime=None,
        in_envelope=in_envelope,
        zone=zone,
        is_uncharted=True,
        notes=notes,
    )