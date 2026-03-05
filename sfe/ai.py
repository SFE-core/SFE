# -*- coding: utf-8 -*-
"""
sfe/ai.py — LLM interpretation layer for the SFE instrument.

Builds a structured prompt from a SFEResult and sends it to any
OpenAI-compatible endpoint. Works out of the box with:
    - Anthropic (claude-*)
    - OpenAI (gpt-*)
    - Ollama (local models)
    - Any provider with an OpenAI-compatible /v1/chat/completions endpoint

The instrument measures. The LLM interprets.

Usage
-----
    from sfe.ai import interpret, LLMConfig

    # Anthropic
    result = interpret(sfe_result, domain="finance",
                       config=LLMConfig(
                           api_key="sk-ant-...",
                           model="claude-opus-4-6",
                           base_url="https://api.anthropic.com/v1",
                       ))

    # OpenAI
    result = interpret(sfe_result, domain="finance",
                       config=LLMConfig(
                           api_key="sk-...",
                           model="gpt-4o",
                       ))

    # Ollama (local)
    result = interpret(sfe_result, domain="eeg",
                       config=LLMConfig(
                           api_key="ollama",
                           model="llama3",
                           base_url="http://localhost:11434/v1",
                       ))

    # Print and save
    print(result.interpretation)
    result.save(run_folder)       # writes interpretation.txt alongside instrument files
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .connect import SFEResult
from .core import OPERATING_ENVELOPE

__all__ = ["LLMConfig", "InterpretationResult", "interpret", "build_prompt"]

# ---------------------------------------------------------------------------
# Domain context — one line fed to the LLM so it frames language correctly
# ---------------------------------------------------------------------------

_DOMAIN_CONTEXT = {
    "finance": (
        "Data are log-returns of financial instruments. "
        "Coupling reflects shared market structure. "
        "Non-stationarity often indicates regime transitions or crisis events."
    ),
    "eeg": (
        "Data are EEG biosignals from electrode pairs. "
        "Coupling reflects neural synchrony between brain regions. "
        "Event-locked dρ drops indicate task-driven channel stabilization."
    ),
    "traffic": (
        "Data are sensor readings from a physical network (traffic, power, temperature). "
        "Coupling reflects shared physical dynamics. "
        "High r_eff indicates the network has not collapsed to a single shared mode."
    ),
}

_DOMAIN_FALLBACK = (
    "Data are multivariate time series from an unspecified domain. "
    "Interpret coupling structure in general terms."
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """
    Provider configuration for the LLM interpretation layer.

    Parameters
    ----------
    api_key  : str   API key. Can also be set via SFE_LLM_API_KEY env var.
    model    : str   Model name. Default: claude-sonnet-4-6
    base_url : str   API base URL. Default: Anthropic endpoint.
    max_tokens : int Default: 1024
    temperature: float Default: 0.2 (low — we want precise technical language)
    extra_headers : dict  e.g. {"anthropic-version": "2023-06-01"}
    """
    api_key      : str   = ""
    model        : str   = "claude-sonnet-4-6"
    base_url     : str   = "https://api.anthropic.com/v1"
    max_tokens   : int   = 1024
    temperature  : float = 0.2
    extra_headers: dict  = field(default_factory=lambda: {
        "anthropic-version": "2023-06-01",
        "anthropic-beta":    "messages-2023-06-01",
    })

    def resolved_api_key(self) -> str:
        return self.api_key or os.environ.get("SFE_LLM_API_KEY", "")


# ---------------------------------------------------------------------------
# Prompt builder — pure function, no I/O
# ---------------------------------------------------------------------------

def build_prompt(result: SFEResult, domain: str = "unknown") -> str:
    """
    Build the interpretation prompt from a SFEResult.

    Pure function — no API calls. Useful for inspection or custom pipelines.

    Parameters
    ----------
    result : SFEResult
    domain : str   "finance" | "eeg" | "traffic" | any string

    Returns
    -------
    str   the full prompt text
    """
    s   = result.summary_dict()
    env = OPERATING_ENVELOPE
    ctx = _DOMAIN_CONTEXT.get(domain.lower(), _DOMAIN_FALLBACK)

    # Pair table
    header = f"  {'Pair':<20} {'ρ*':>6} {'dρ':>10} {'r_eff':>7} {'NS%':>6}"
    sep    = "  " + "-" * 52
    rows   = []
    for p in result.pairs:
        flag = "  ← flagged NS" if p["nonstationary_pct"] > 40 else ""
        rows.append(
            f"  {p['label']:<20} {p['rho_star']:>6.3f} "
            f"{p['drho_mean']:>10.6f} {p['reff_mean']:>7.3f} "
            f"{p['nonstationary_pct']:>5.1f}%{flag}"
        )
    pair_table = "\n".join([header, sep] + rows)

    # Quality notes
    quality_notes = ""
    if result.quality.rows_dropped or result.quality.columns_dropped:
        quality_notes = (
            f"\n  Data quality: "
            f"{result.quality.rows_dropped} rows dropped, "
            f"columns dropped: {result.quality.columns_dropped or 'none'}."
        )

    prompt = f"""You are interpreting output from the SFE instrument, which measures coupling structure in multivariate time series using rolling correlation geometry.

Domain context:
  {ctx}

Instrument definitions:
  ρ*     = mean absolute rolling correlation over the analysis period (0→1)
           higher = stronger, more persistent coupling between the pair
  dρ     = mean variance of ρ over the rolling window
           near 0 = geometrically stable channel; high = unstable or transitioning
  r_eff  = effective rank of the pair (1 = fully rank-collapsed, 2 = independent)
  NS%    = percentage of non-overlapping blocks where correlation variance
           is non-stationary (self-detected without external ground truth)
  r_eff joint = entropy-based effective dimensionality across all N observers

Operating envelope:
  Reliable zone : ρ* > {env['reliable_rho_min']} and dρ near 0
  Degraded zone : ρ* < {env['degraded_rho_max']}
  NS flag       : NS% > 40% indicates structural drift in that pair

Instrument output:
  Domain         : {domain}
  N (observers)  : {s['N']}
  T (timesteps)  : {s['T']}
  W (window)     : {s['W']}
  Pairs total    : {s['n_pairs']}
  ρ* mean        : {s['rho_star_mean']:.4f}
  ρ* max         : {s['rho_star_max']:.4f}
  dρ mean        : {s['drho_mean']:.6f}
  r_eff joint    : {s['reff_joint_mean']:.4f}{quality_notes}

Pair table:
{pair_table}

Answer the following. Be concise and technical. Do not restate definitions.

1. What is the dominant coupling regime across this dataset?
2. Which pairs are structurally reliable and which are unstable? Why?
3. Is W={s['W']} appropriate for this domain and T={s['T']}? Suggest an adjustment if needed.
4. What should the analyst investigate next based on these numbers?"""

    return prompt


# ---------------------------------------------------------------------------
# Interpretation result
# ---------------------------------------------------------------------------

@dataclass
class InterpretationResult:
    """
    Returned by interpret(). Holds the prompt, raw response, and metadata.

    Attributes
    ----------
    interpretation : str    LLM response text
    prompt         : str    the exact prompt sent
    model          : str    model used
    domain         : str
    timestamp      : str    ISO format
    """
    interpretation : str
    prompt         : str
    model          : str
    domain         : str
    timestamp      : str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def save(self, folder) -> Path:
        """
        Write interpretation.txt to a RunFolder or any Path-like location.

        Parameters
        ----------
        folder : RunFolder | str | Path

        Returns
        -------
        Path to written file
        """
        path = Path(folder.path) if hasattr(folder, "path") else Path(folder)
        out  = path / "interpretation.txt"

        content = "\n".join([
            "SFE LLM INTERPRETATION",
            "=" * 62,
            f"Timestamp : {self.timestamp}",
            f"Model     : {self.model}",
            f"Domain    : {self.domain}",
            "=" * 62,
            "",
            self.interpretation,
            "",
            "=" * 62,
            "PROMPT SENT",
            "=" * 62,
            "",
            self.prompt,
        ])

        out.write_text(content, encoding="utf-8")
        print(f"  → interpretation.txt")
        return out

    def __str__(self):
        return self.interpretation


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def interpret(
    result: SFEResult,
    domain: str = "unknown",
    config: LLMConfig | None = None,
    save_to=None,
) -> InterpretationResult:
    """
    Send an SFEResult to an LLM and return the interpretation.

    This function is entirely opt-in. The instrument (core + connectors)
    always runs locally. Call this only if you want LLM interpretation.

    Privacy note
    ------------
    Only derived metrics are sent to the API — ρ*, dρ, r_eff, NS%, and
    aggregate statistics. Raw data, prices, and returns never leave your
    machine. Column labels (ticker names, channel names) are included in
    the prompt; pass anonymous labels to the connector if that is a concern.

    To keep everything local, use Ollama:
        LLMConfig(api_key="ollama", model="llama3",
                  base_url="http://localhost:11434/v1", extra_headers={})

    Parameters
    ----------
    result  : SFEResult
    domain  : str          "finance" | "eeg" | "traffic" | any string
    config  : LLMConfig    provider config. Defaults to Anthropic claude-sonnet-4-6.
                           API key read from SFE_LLM_API_KEY if not set in config.
    save_to : RunFolder | Path | str | None
              if provided, saves interpretation.txt automatically

    Returns
    -------
    InterpretationResult

    Raises
    ------
    ImportError   if openai package is not installed
    ValueError    if no API key is found
    RuntimeError  if the API call fails
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package required for the AI layer.\n"
            "pip install openai\n"
            "(works with Anthropic, OpenAI, Ollama, and any compatible endpoint)"
        )

    cfg = config or LLMConfig()
    key = cfg.resolved_api_key()

    _is_remote = not any(
        local in cfg.base_url
        for local in ("localhost", "127.0.0.1", "0.0.0.0")
    )
    if _is_remote:
        print(
            f"\n  [sfe.ai] Sending derived metrics to {cfg.base_url} "
            f"(model: {cfg.model}).\n"
            f"  Only ρ*, dρ, r_eff, NS% and labels are included — "
            f"not raw data.\n"
            f"  For a fully local run: LLMConfig(api_key='ollama', "
            f"model='llama3', base_url='http://localhost:11434/v1', "
            f"extra_headers={{}})\n"
        )

    if not key:
        raise ValueError(
            "No API key found. Pass it via LLMConfig(api_key=...) "
            "or set the SFE_LLM_API_KEY environment variable."
        )

    prompt = build_prompt(result, domain=domain)

    client = OpenAI(
        api_key    = key,
        base_url   = cfg.base_url,
        default_headers = cfg.extra_headers,
    )

    try:
        response = client.chat.completions.create(
            model       = cfg.model,
            max_tokens  = cfg.max_tokens,
            temperature = cfg.temperature,
            messages    = [{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content.strip()

    except Exception as e:
        raise RuntimeError(
            f"LLM API call failed ({cfg.base_url}, model={cfg.model}):\n{e}"
        )

    interp = InterpretationResult(
        interpretation = text,
        prompt         = prompt,
        model          = cfg.model,
        domain         = domain,
    )

    if save_to is not None:
        interp.save(save_to)

    return interp
