# -*- coding: utf-8 -*-
"""
sfe/analysis/regimes.py — Single source of truth for all validated coupling regimes.

This module mirrors the cross-domain summary table from SFE-11 and is the
reference point for:
    - The AI layer       (context for interpretation)
    - figures.py         (domain-aware reference anchors on phase portraits)
    - analysis modules   (detection thresholds)
    - classify_result()  (where does a new result land?)

Two registries:
    REGIMES          — validated domains: where the instrument works,
                       characterized by substrate, mechanism, and signature.
                       New domains are added here when validated, not before.

    KNOWN_LIMITS     — real-data boundary conditions of the validated regime.
                       Each entry is a characterized limit of the instrument,
                       identified during implementation or empirical validation.

The core operating envelope (OPERATING_ENVELOPE) is defined in core.py
and imported here. It is never redefined.

Detection thresholds (CRISIS_THRESHOLDS, TRANSITION_THRESHOLDS) live here
because they are empirical calibrations, not mathematical results.
They belong to the validated knowledge layer, not the core.

Domain identity contract
------------------------
result.domain is set by the connector, carried by SFEResult, and read here.
ref_points(domain) returns only the anchors that help interpret that domain's
data — not a global list that every domain has to filter manually.

The rule per domain:
    strain   → no anchors (data exceeds all synthetic refs; threshold lines suffice)
    eeg      → no anchors (tightly locked, anchors add no interpretive value)
    shm      → no anchors until validated
    finance  → OU mid-k (background sector coupling sits near it)
    traffic  → OU mid-k (METR-LA marginal zone sits right next to it)
    unknown  → all validated cross_domain_ref anchors (safe fallback)

OU synthetic anchors are calibration references from the theoretical operating
envelope. They are domain-agnostic but only earn their place on a plot when
the live data actually sits near them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from ..core import OPERATING_ENVELOPE

__all__ = [
    "REGIMES",
    "KNOWN_LIMITS",
    "OPERATING_ENVELOPE",
    "CRISIS_THRESHOLDS",
    "TRANSITION_THRESHOLDS",
    "OU_ANCHORS",
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

CRISIS_THRESHOLDS = {
    "rho_delta_min":    0.10,
    "bandgap_mult_min": 1.50,
    "reff_delta_max":  -0.10,
    "pairs_pct_min":    0.50,
}

TRANSITION_THRESHOLDS = {
    "rho_delta_min":   -0.05,
    "drho_spike_mult":  2.0,
    "reff_delta_min":   0.05,
}


# ---------------------------------------------------------------------------
# OU synthetic anchors
# These are calibration references from the theoretical operating envelope.
# They are NOT domain-specific — they represent where OU processes with
# different confinement strengths sit in (rho*, drho) space.
# Only shown on domains where the live data sits near them.
# ---------------------------------------------------------------------------

OU_ANCHORS = {
    "high_k": {
        "rho_star": 0.963,
        "drho":     0.004,
        "color":    "#00e676",
        "label":    "OU high-k",
    },
    "mid_k": {
        "rho_star": 0.45,
        "drho":     0.020,
        "color":    "#ffd60a",
        "label":    "OU mid-k",
    },
    "low_k": {
        "rho_star": 0.20,
        "drho":     0.055,
        "color":    "#ffffff",
        "label":    "OU low-k",
    },
}


# ---------------------------------------------------------------------------
# Regime dataclass
# ---------------------------------------------------------------------------

@dataclass
class Regime:
    """
    One row of the cross-domain validation table (SFE-11 Section 4).

    domain_key is the canonical string that result.domain is set to
    by the connector. It is the join key between SFEResult and REGIMES.

    cross_domain_ref=True means this regime's ref_point is shown as an
    anchor on OTHER domains' phase portraits (for comparison context).
    cross_domain_ref=False means it is domain-internal (e.g. crisis events
    within the finance domain).
    """
    domain     : str
    domain_key : str                    # matches result.domain set by connector
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
    cross_domain_ref : bool = True


# ---------------------------------------------------------------------------
# Known limits dataclass
# ---------------------------------------------------------------------------

@dataclass
class KnownLimit:
    id                  : str
    open_problem        : str
    description         : str
    discovered          : str
    status              : str
    instrument_response : str


# ---------------------------------------------------------------------------
# Validated regime registry
# ---------------------------------------------------------------------------

REGIMES: list[Regime] = [

    Regime(
        domain     = "ETTh1",
        domain_key = "traffic",         # ETT goes through traffic connector
        mechanism  = "Persistent lock",
        substrate  = "Electrical infrastructure",
        timescale  = "Hourly",
        rho_star   = (0.950, 0.975),
        band_gap   = (3.5, 5.0),
        reff       = (1.00, 1.10),
        drho       = (0.003, 0.005),
        detection  = "reliable",
        ref_point  = (0.963, 0.004),
        ref_color  = "#ff6b35",
        notes      = "HUFL-MUFL pair. Persistent coupling from shared transformer load. "
                     "Granger causality precedes dρ lock-in by mean 4.0h.",
    ),

    Regime(
        domain     = "METR-LA",
        domain_key = "traffic",
        mechanism  = "Sync transition",
        substrate  = "Urban traffic network",
        timescale  = "5-minute intervals",
        rho_star   = (0.40, 0.45),
        band_gap   = (4.5, 6.0),
        reff       = (2.0, 3.0),
        drho       = (0.040, 0.055),
        detection  = "indicative",
        ref_point  = (0.426, 0.047),
        ref_color  = "#00b4d8",
        notes      = "207 sensors. ρ* in marginal zone — results indicative. "
                     "r_eff plateau at N≈50 reveals intrinsic network dimensionality. "
                     "See Open Problem 7.",
    ),

    Regime(
        domain     = "EEG motor cortex",
        domain_key = "eeg",
        mechanism  = "Disruption/relock",
        substrate  = "Neural tissue",
        timescale  = "Seconds (160 Hz)",
        rho_star   = (0.72, 0.89),
        band_gap   = None,
        reff       = (1.10, 1.30),
        drho       = (0.003, 0.006),
        detection  = "9/10 subjects replicated",
        ref_point  = (0.831, 0.004),
        ref_color  = "#c77dff",
        notes      = "Two-phase structure: dρ spike at onset, drop below baseline. "
                     "Lead time ~3.1s. pct_drop > 20% = structure present. "
                     "See Open Problem 8.",
    ),

    Regime(
        domain     = "Finance background",
        domain_key = "finance",
        mechanism  = "Sector coupling",
        substrate  = "Financial markets",
        timescale  = "Daily (20-day window)",
        rho_star   = (0.60, 0.70),
        band_gap   = (6.0, 8.0),
        reff       = (1.70, 2.10),
        drho       = (0.010, 0.030),
        detection  = "reliable",
        ref_point  = (0.654, 0.020),
        ref_color  = "#ffd60a",
        notes      = "Full-period background. Baseline for crisis detection.",
    ),

    Regime(
        domain     = "Finance — COVID crash 2020",
        domain_key = "finance",
        mechanism  = "Acute homogeneous shock (Branch A)",
        substrate  = "Financial markets",
        timescale  = "Daily",
        rho_star   = (0.90, 0.93),
        band_gap   = (18.0, 23.0),
        reff       = (1.30, 1.50),
        drho       = (0.015, 0.035),
        detection  = "Branch A ✓",
        ref_point  = (0.915, 0.025),
        ref_color  = "#ff4444",
        cross_domain_ref = False,
        notes      = "Band gap explosion ≥1.5× background is the discriminating signature. "
                     "CRISIS_THRESHOLDS Branch A calibrated on this event.",
    ),

    Regime(
        domain     = "Finance — Lehman 2008",
        domain_key = "finance",
        mechanism  = "Acute heterogeneous contagion (Branch B)",
        substrate  = "Financial markets",
        timescale  = "Daily",
        rho_star   = (0.68, 0.72),
        band_gap   = (5.50, 7.50),
        reff       = (1.80, 2.00),
        drho       = (0.020, 0.040),
        detection  = "Branch B ✓",
        ref_point  = (0.695, 0.030),
        ref_color  = "#ff8800",
        cross_domain_ref = False,
        notes      = "Band gap stable — variance diffused across modes. "
                     "r_eff collapse + dρ elevation on >50% of pairs.",
    ),

    Regime(
        domain     = "Finance — dot-com 2000-02",
        domain_key = "finance",
        mechanism  = "Gradual correction",
        substrate  = "Financial markets",
        timescale  = "Daily",
        rho_star   = (0.50, 0.56),
        band_gap   = (2.50, 3.20),
        reff       = (2.60, 2.90),
        drho       = (0.008, 0.020),
        detection  = "correctly silent ✓",
        ref_point  = (0.530, 0.015),
        ref_color  = "#aaaaaa",
        cross_domain_ref = False,
        notes      = "Neither branch fires. Gradual valuation correction "
                     "is not a synchronization event.",
    ),

    Regime(
        domain     = "Strain rosette",
        domain_key = "strain",
        mechanism  = "Static lock",
        substrate  = "Structural mechanics",
        timescale  = "1 Hz (minutes to hours)",
        rho_star   = (0.920, 1.000),
        band_gap   = (55.0, 70.0),
        reff       = (1.05, 1.15),
        drho       = (0.0, 0.000005),
        detection  = "36/36 pairs, dρ=0 for 23h",
        ref_point  = (0.936, 0.000001),
        ref_color  = "#00e676",
        notes      = "Highest band gap in cross-domain study (61.66×). "
                     "Upper bound of band-gap range; lower bound of dρ range. "
                     "f(N) over-correction at N=9 — see KNOWN_LIMITS KL-01.",
    ),

    Regime(
        domain     = "SHM — KW51 bridge",
        domain_key = "shm",
        mechanism  = "Structural transition (retrofitting)",
        substrate  = "Structural mechanics",
        timescale  = "Hourly (15-month campaign)",
        rho_star   = (0.0, 1.0),
        band_gap   = (0.0, 100.0),
        reff       = (1.0, 10.0),
        drho       = (0.0, 1.0),
        detection  = "pending",
        ref_point  = (0.0, 0.0),
        ref_color  = "#888888",
        pending    = True,
        notes      = "Three phases: nominal / retrofitting / post-retrofit. "
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
                       "physical bound reff ≥ 1.",
        discovered   = "Strain connector port, N=9, band_gap=61.66×",
        status       = "characterized",
        instrument_response = "reff_corr_fallback=True; raw joint mean returned.",
    ),

    KnownLimit(
        id           = "KL-02",
        open_problem = "New domain — SHM structural transition (KW51)",
        description  = "TRANSITION_THRESHOLDS are initial estimates for structural "
                       "transition detection. Not yet empirically characterized.",
        discovered   = "KW51 connector port — pending first run",
        status       = "pending",
        instrument_response = "detect_structural_transition() returns a calibration "
                              "warning until KW51 run completes.",
    ),

]


# ---------------------------------------------------------------------------
# Lookup utilities
# ---------------------------------------------------------------------------

def get_regime(domain: str) -> Regime | None:
    """Return the Regime for a domain name or domain_key (case-insensitive)."""
    domain_l = domain.lower()
    for r in REGIMES:
        if domain_l in r.domain.lower() or domain_l == r.domain_key.lower():
            return r
    return None


def ref_points(domain: str | None = None) -> list[tuple]:
    """
    Return (rho_star, drho, color, label) anchors appropriate for a given domain.

    This is the single place that decides what appears on a phase portrait.
    The decision is made once here; figures.py never needs domain logic.

    Domain rules
    ------------
    strain   → empty  — data exceeds all synthetic anchors; threshold lines suffice
    eeg      → empty  — tightly locked, anchors add no interpretive value
    shm      → empty  — pending validation, no characterized anchor yet
    finance  → OU mid-k only (background sector coupling sits near rho*≈0.45–0.65)
    traffic  → OU mid-k only (METR-LA marginal zone sits right next to it)
    None /
    unknown  → all cross_domain_ref anchors (safe fallback for uncharted domains)

    OU anchors are calibration references — they earn their place only when
    live data sits near them. They are never shown just for decoration.
    """
    if domain is None:
        domain_l = "unknown"
    else:
        domain_l = domain.lower()

    # Domains where anchors add no interpretive value
    if domain_l in ("strain", "eeg", "shm"):
        return []

    # Finance — show internal regime anchors (background + crises) for context
    # but not OU synthetic anchors (finance data doesn't sit near OU calibration points)
    if domain_l == "finance":
        return [
            (r.ref_point[0], r.ref_point[1], r.ref_color, r.mechanism)
            for r in REGIMES
            if r.domain_key == "finance"
            and not r.pending
            and r.ref_point != (0.0, 0.0)
        ]

    # Traffic — OU mid-k is meaningful (METR-LA sits right next to it)
    if domain_l == "traffic":
        mid = OU_ANCHORS["mid_k"]
        return [(mid["rho_star"], mid["drho"], mid["color"], mid["label"])]

    # Unknown / uncharted — show all validated cross_domain_ref anchors
    # Safe fallback: gives maximum context for domains not yet in REGIMES
    return [
        (r.ref_point[0], r.ref_point[1], r.ref_color, r.mechanism)
        for r in REGIMES
        if not r.pending
        and r.cross_domain_ref
        and r.ref_point != (0.0, 0.0)
    ]


def domain_context(domain: str) -> str:
    """
    Return a plain-text domain context string for the AI layer.
    Uses domain_key matching so result.domain maps correctly.
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
    matched_regime : Regime | None
    in_envelope    : bool
    zone           : str
    is_uncharted   : bool
    notes          : list[str] = field(default_factory=list)


def classify_result(result) -> ClassificationResult:
    """
    Compare a SFEResult against all validated regimes.

    Uses result.domain (set by connector) as the primary lookup key,
    then falls back to metric-range matching across all regimes.
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

    # Primary: match by domain_key if result carries domain identity
    result_domain = getattr(result, "domain", None)
    if result_domain is not None:
        for regime in REGIMES:
            if regime.pending:
                continue
            if regime.domain_key == result_domain:
                # Still verify metrics are in range
                rho_ok  = regime.rho_star[0] <= rho <= regime.rho_star[1]
                reff_ok = regime.reff[0]     <= rc  <= regime.reff[1]
                bg_ok   = (regime.band_gap is None or
                           regime.band_gap[0] <= bg <= regime.band_gap[1])
                if rho_ok and reff_ok and bg_ok:
                    notes.append(
                        f"Matches validated regime: {regime.domain} "
                        f"({regime.mechanism})."
                    )
                    return ClassificationResult(
                        matched_regime=regime,
                        in_envelope=in_envelope,
                        zone=zone,
                        is_uncharted=False,
                        notes=notes,
                    )

    # Fallback: metric-range matching across all regimes
    for regime in REGIMES:
        if regime.pending:
            continue
        rho_ok  = regime.rho_star[0] <= rho <= regime.rho_star[1]
        reff_ok = regime.reff[0]     <= rc  <= regime.reff[1]
        bg_ok   = (regime.band_gap is None or
                   regime.band_gap[0] <= bg <= regime.band_gap[1])
        if rho_ok and reff_ok and bg_ok:
            notes.append(
                f"Matches validated regime: {regime.domain} "
                f"({regime.mechanism}) — matched by metrics, not domain key."
            )
            return ClassificationResult(
                matched_regime=regime,
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