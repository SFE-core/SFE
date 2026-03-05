# -*- coding: utf-8 -*-
"""
sfe/outputs.py — Dedicated output manager for the SFE instrument.

Creates timestamped, collision-free run folders under a project-level
outputs/ directory. Any connector can call save_run() to persist a
complete SFEResult with figures, CSVs, and a plain-text summary.

Folder structure
----------------
outputs/
├── finance/
│   ├── 2026-03-05_143201_covid_crash/
│   │   ├── summary.txt
│   │   ├── pairs.csv
│   │   ├── quality.txt
│   │   └── figures/
│   │       ├── fig_phase_portrait.png
│   │       └── ...
│   └── ...
├── eeg/
│   └── 2026-03-05_143205_S001_R04/
│       └── ...
└── traffic/
    └── ...

Usage
-----
    from sfe.outputs import RunFolder, save_run

    # Quick save — figures optional
    folder = save_run(result, domain="finance", label="covid_crash")
    print(folder.path)

    # Manual — full control
    run = RunFolder.create(domain="eeg", label="S001_R04")
    run.save_summary(result)
    run.save_pairs_csv(result)
    run.save_quality(result)
    run.save_figure(fig, "fig_event_locked")
    print(run.path)

Notes
-----
    - Folder names are YYYY-MM-DD_HHMMSS_<label> — no collisions, no renaming.
    - <label> is sanitized (spaces → underscores, special chars stripped).
    - outputs/ root defaults to the current working directory.
      Override with: RunFolder.create(..., root="path/to/outputs")
      or set SFE_OUTPUTS_ROOT environment variable.
    - Designed to be AI-readable: summary.txt and pairs.csv are
      structured plain text / CSV, safe to feed to a language model.
"""

from __future__ import annotations

import csv
import os
import re
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np

from .connect import SFEResult
from .core import OPERATING_ENVELOPE

__all__ = ["RunFolder", "save_run"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize(label: str) -> str:
    """Make label safe for use as a directory name component."""
    label = label.strip().replace(" ", "_")
    label = re.sub(r"[^\w\-]", "", label)   # keep word chars, dash
    return label[:64]                         # cap length


def _outputs_root(override: str | None) -> Path:
    """Resolve the outputs root directory."""
    if override:
        return Path(override)
    env = os.environ.get("SFE_OUTPUTS_ROOT")
    if env:
        return Path(env)
    return Path.cwd() / "outputs"


# ---------------------------------------------------------------------------
# RunFolder
# ---------------------------------------------------------------------------

class RunFolder:
    """
    Represents a single timestamped output directory for one SFE run.

    Create with RunFolder.create(), not directly.
    """

    def __init__(self, path: Path):
        self.path    = path
        self.figures = path / "figures"
        path.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(exist_ok=True)

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        domain: str,
        label: str,
        root: str | None = None,
    ) -> "RunFolder":
        """
        Create a new timestamped run folder.

        Parameters
        ----------
        domain : str   "finance" | "eeg" | "traffic" | any custom string
        label  : str   short descriptor, e.g. "covid_crash", "S001_R04"
        root   : str   override outputs root (default: cwd/outputs or SFE_OUTPUTS_ROOT)

        Returns
        -------
        RunFolder
        """
        ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        name    = f"{_sanitize(label)}_{ts}"
        path    = _outputs_root(root) / _sanitize(domain) / name
        return cls(path)

    # ── Writers ──────────────────────────────────────────────────────────────

    def save_summary(self, result: SFEResult, extra: dict | None = None) -> Path:
        """
        Write a human- and AI-readable plain-text summary.

        Parameters
        ----------
        result : SFEResult
        extra  : dict, optional   domain-specific key-value pairs appended
                                  at the end of the file (e.g. crash window dates,
                                  regime detection verdict, pass condition)

        Returns
        -------
        Path to summary.txt
        """
        s   = result.summary_dict()
        env = OPERATING_ENVELOPE
        out = self.path / "summary.txt"

        lines = [
            "=" * 62,
            "  SFE RUN SUMMARY",
            "=" * 62,
            f"  Run folder  : {self.path.name}",
            f"  Timestamp   : {datetime.now().isoformat(timespec='seconds')}",
            "",
            f"  N (observers)     : {s['N']}",
            f"  T (timesteps)     : {s['T']}",
            f"  W (window)        : {s['W']}",
            f"  Pairs total       : {s['n_pairs']}",
            f"  Reliable (ρ*>{env['reliable_rho_min']}) : {s['n_reliable']}",
            f"  Flagged NS>40%    : {s['n_flagged']}",
            "",
            f"  ρ* mean           : {s['rho_star_mean']:.4f}",
            f"  ρ* max            : {s['rho_star_max']:.4f}",
            f"  dρ mean           : {s['drho_mean']:.6f}",
            f"  r_eff joint mean  : {s['reff_joint_mean']:.4f}",
            "",
            "  ── Pairs ──",
            f"  {'Pair':<20} {'ρ*':>6} {'dρ':>10} {'r_eff':>7} {'NS%':>6}  Zone",
            "  " + "-" * 56,
        ]

        for p in result.pairs:
            lines.append(
                f"  {p['label']:<20} {p['rho_star']:>6.3f} "
                f"{p['drho_mean']:>10.6f} {p['reff_mean']:>7.3f} "
                f"{p['nonstationary_pct']:>5.1f}%  {p['zone']}"
            )

        if s["cols_dropped"] or s["warnings"]:
            lines += ["", "  ── Data quality ──"]
            for w in s["warnings"]:
                lines.append(f"  ⚠  {w}")

        if extra:
            lines += ["", "  ── Domain metadata ──"]
            for k, v in extra.items():
                lines.append(f"  {k:<22} : {v}")

        lines += [
            "",
            "  Experimental. Not financial, medical, or engineering advice.",
            "=" * 62,
        ]

        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"  → summary.txt")
        return out

    def save_pairs_csv(self, result: SFEResult) -> Path:
        """
        Write pairs table as CSV. Includes all scalar metrics per pair.
        Time-series arrays (rho, drho) are not written here — use save_arrays().

        Returns
        -------
        Path to pairs.csv
        """
        out = self.path / "pairs.csv"

        fieldnames = [
            "label", "i", "j",
            "rho_star", "drho_mean", "reff_mean",
            "zone", "nonstationary_pct",
        ]

        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for p in result.pairs:
                writer.writerow({k: p[k] for k in fieldnames})

        print(f"  → pairs.csv  ({len(result.pairs)} rows)")
        return out

    def save_quality(self, result: SFEResult) -> Path:
        """
        Write data quality report as plain text.

        Returns
        -------
        Path to quality.txt
        """
        q   = result.quality
        out = self.path / "quality.txt"

        lines = [
            "SFE DATA QUALITY REPORT",
            "-" * 40,
            f"original_shape  : {q.original_shape}",
            f"cleaned_shape   : {q.cleaned_shape}",
            f"rows_dropped    : {q.rows_dropped}",
            f"cols_dropped    : {q.columns_dropped}",
            f"inf_replaced    : {q.inf_replaced}",
            f"nan_count       : {q.nan_count}",
            "",
            "warnings:",
        ]
        for w in q.warnings:
            lines.append(f"  - {w}")
        if not q.warnings:
            lines.append("  (none)")

        lines += ["", "errors:"]
        for e in q.errors:
            lines.append(f"  - {e}")
        if not q.errors:
            lines.append("  (none)")

        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"  → quality.txt")
        return out

    def save_arrays(self, result: SFEResult) -> Path:
        """
        Save rho and drho time series for every pair as a compressed .npz.
        Useful for downstream analysis or feeding to an AI layer.

        Returns
        -------
        Path to arrays.npz
        """
        out     = self.path / "arrays.npz"
        payload = {}
        for p in result.pairs:
            key = _sanitize(p["label"])
            payload[f"{key}_rho"]  = p["rho"]
            payload[f"{key}_drho"] = p["drho"]
        payload["reff_joint"] = result.reff_joint
        np.savez_compressed(out, **payload)
        print(f"  → arrays.npz  ({len(result.pairs)} pairs)")
        return out

    def save_figure(self, fig, name: str, dpi: int = 150) -> Path:
        """
        Save a matplotlib figure to the figures/ subfolder.

        Parameters
        ----------
        fig  : matplotlib Figure
        name : str   filename without extension (e.g. "fig_phase_portrait")
        dpi  : int   default 150

        Returns
        -------
        Path to saved .png
        """
        out = self.figures / f"{name}.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  → figures/{name}.png")
        return out

    def write_text(self, filename: str, content: str) -> Path:
        """
        Write any additional plain-text file to the run folder.
        Useful for domain-specific metadata, AI prompts, verdict strings, etc.

        Parameters
        ----------
        filename : str   e.g. "regime_verdict.txt", "multisubject_log.txt"
        content  : str

        Returns
        -------
        Path to written file
        """
        out = self.path / filename
        out.write_text(content, encoding="utf-8")
        print(f"  → {filename}")
        return out

    def __repr__(self):
        return f"RunFolder('{self.path}')"


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def save_run(
    result: SFEResult,
    domain: str,
    label: str,
    figures: list | None = None,
    figure_names: list[str] | None = None,
    extra: dict | None = None,
    save_arrays: bool = False,
    root: str | None = None,
) -> RunFolder:
    """
    Create a run folder and save standard outputs in one call.

    Parameters
    ----------
    result       : SFEResult
    domain       : str              "finance" | "eeg" | "traffic" | custom
    label        : str              short run descriptor, e.g. "covid_crash"
    figures      : list of Figure   matplotlib figures to save (optional)
    figure_names : list of str      names for each figure (default: fig_0, fig_1 …)
    extra        : dict             domain metadata appended to summary.txt
                                    and useful for an AI layer later
    save_arrays  : bool             also save rho/drho arrays as .npz (default False)
    root         : str              override outputs root directory

    Returns
    -------
    RunFolder

    Example
    -------
    folder = save_run(result, domain="finance", label="covid_crash",
                      figures=[fig1, fig2],
                      figure_names=["phase_portrait", "timeseries"],
                      extra={"crash_window": "2020-02-01 → 2020-04-30",
                             "regime": "CRISIS COUPLING",
                             "tickers": "AAPL MSFT GOOGL NVDA"})
    print(folder.path)
    """
    run = RunFolder.create(domain=domain, label=label, root=root)

    print(f"\nSaving run → {run.path}")
    run.save_summary(result, extra=extra)
    run.save_pairs_csv(result)
    run.save_quality(result)

    if save_arrays:
        run.save_arrays(result)

    # Auto-generate standard figures
    try:
        from .figures import all_figures
        std_figs = all_figures(result, title_prefix=label)
        for name, fig in std_figs.items():
            run.save_figure(fig, name)
            import matplotlib.pyplot as plt
            plt.close(fig)
    except Exception as e:
        print(f"  ⚠  Standard figures skipped: {e}")

    # Any extra domain figures passed by the caller
    if figures:
        names = figure_names or [f"fig_{k}" for k in range(len(figures))]
        for fig, name in zip(figures, names):
            run.save_figure(fig, name)

    print(f"  Done.\n")
    return run
