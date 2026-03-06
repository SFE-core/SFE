# -*- coding: utf-8 -*-
"""
sfe/outputs.py — Dedicated output manager for the SFE instrument.
"""

from __future__ import annotations

import csv
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from .connect import SFEResult
from .core import OPERATING_ENVELOPE
from .analysis.absorption import compute_absorption_signature

__all__ = ["RunFolder", "save_run"]


def _sanitize(label: str) -> str:
    label = label.strip().replace(" ", "_")
    label = re.sub(r"[^\w\-]", "", label)
    return label[:64]


def _outputs_root(override: str | None) -> Path:
    if override:
        return Path(override)
    env = os.environ.get("SFE_OUTPUTS_ROOT")
    if env:
        return Path(env)
    return Path.cwd() / "outputs"


class RunFolder:
    """Timestamped output directory for one SFE run. Create via RunFolder.create()."""

    def __init__(self, path: Path):
        self.path    = path
        self.figures = path / "figures"
        path.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(exist_ok=True)

    @classmethod
    def create(cls, domain: str, label: str,
               root: str | None = None) -> "RunFolder":
        ts   = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        name = f"{_sanitize(label)}_{ts}"
        path = _outputs_root(root) / _sanitize(domain) / name
        return cls(path)

    def save_summary(self, result: SFEResult,
                     extra: dict | None = None) -> Path:
        """
        Write a human- and AI-readable plain-text summary.
        Includes band_gap and reff_corr per SFE-11.
        """
        s   = result.summary_dict()
        env = OPERATING_ENVELOPE
        out = self.path / "summary.txt"
        sig = compute_absorption_signature(result.pairs)

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
            f"  Reliable (rho*>{env['reliable_rho_min']}) : {s['n_reliable']}",
            f"  Flagged NS>40%    : {s['n_flagged']}",
            "",
            f"  rho* mean         : {s['rho_star_mean']:.4f}",
            f"  rho* max          : {s['rho_star_max']:.4f}",
            f"  drho mean         : {s['drho_mean']:.6f}",
            f"  r_eff joint mean  : {s['reff_joint_mean']:.4f}",
            f"  r_eff corrected   : {s['reff_corr']:.4f}   [f(N)=-0.106*ln(N)+1.070, N={s['N']}]",
            f"  band gap lam1/lam2: {s['band_gap']:.3f}x   [Branch A fires if crisis/bg >= 1.50x]",
            f"  CV(drho)          : {sig['cv_drho']:.4f}   [<0.20 homogeneous, >0.20 dispersed]",
            f"  R_drho            : {sig['R_drho']:.2f}x   [vs Lehman baseline 0.01537]",
            f"  absorption        : {'YES — active channel suppression' if sig['absorption'] else 'no'}   [{sig['classification']}]",
            "",
            "  -- Pairs --",
            f"  {'Pair':<25} {'rho*':>8} {'drho':>10} {'r_eff':>8} {'NS%':>6}  Zone",
            "  " + "-" * 56,
        ]

        for p in result.pairs:
            lines.append(
                f"  {p['label']:<25} {p['rho_star']:>8.4f} "
                f"{p['drho_mean']:>10.6f} {p['reff_mean']:>8.4f} "
                f"{p['nonstationary_pct']:>5.1f}%  {p['zone']}"
            )

        if s["cols_dropped"] or s["warnings"]:
            lines += ["", "  -- Data quality --"]
            for w in s["warnings"]:
                lines.append(f"  !  {w}")

        if extra:
            lines += ["", "  -- Domain metadata --"]
            for k, v in extra.items():
                lines.append(f"  {k:<22} : {v}")

        lines += [
            "",
            "  Experimental. Not financial, medical, or engineering advice.",
            "=" * 62,
        ]

        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"  -> summary.txt")
        return out

    def save_pairs_csv(self, result: SFEResult) -> Path:
        out = self.path / "pairs.csv"
        fieldnames = ["label", "i", "j", "rho_star", "drho_mean",
                      "reff_mean", "zone", "nonstationary_pct"]
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for p in result.pairs:
                writer.writerow({k: p[k] for k in fieldnames})
        print(f"  -> pairs.csv  ({len(result.pairs)} rows)")
        return out

    def save_quality(self, result: SFEResult) -> Path:
        q   = result.quality
        out = self.path / "quality.txt"
        lines = [
            "SFE DATA QUALITY REPORT", "-" * 40,
            f"original_shape  : {q.original_shape}",
            f"cleaned_shape   : {q.cleaned_shape}",
            f"rows_dropped    : {q.rows_dropped}",
            f"cols_dropped    : {q.columns_dropped}",
            f"inf_replaced    : {q.inf_replaced}",
            f"nan_count       : {q.nan_count}",
            "", "warnings:",
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
        print(f"  -> quality.txt")
        return out

    def save_arrays(self, result: SFEResult) -> Path:
        out     = self.path / "arrays.npz"
        payload = {}
        for p in result.pairs:
            key = _sanitize(p["label"])
            payload[f"{key}_rho"]  = p["rho"]
            payload[f"{key}_drho"] = p["drho"]
        payload["reff_joint"] = result.reff_joint
        np.savez_compressed(out, **payload)
        print(f"  -> arrays.npz  ({len(result.pairs)} pairs)")
        return out

    def save_figure(self, fig, name: str, dpi: int = 150) -> Path:
        out = self.figures / f"{name}.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  -> figures/{name}.png")
        return out

    def write_text(self, filename: str, content: str) -> Path:
        out = self.path / filename
        out.write_text(content, encoding="utf-8")
        print(f"  -> {filename}")
        return out

    def __repr__(self):
        return f"RunFolder('{self.path}')"


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
    domain       : str
    label        : str              e.g. "covid_crash"
    figures      : list of Figure   additional matplotlib figures
    figure_names : list of str
    extra        : dict             appended to summary.txt
                                    Good keys: "crash_window", "regime",
                                    "band_gap_full", "band_gap_crash"
    save_arrays  : bool             save rho/drho .npz (default False)
    root         : str              override outputs root

    Returns
    -------
    RunFolder
    """
    run = RunFolder.create(domain=domain, label=label, root=root)
    print(f"\nSaving run -> {run.path}")
    run.save_summary(result, extra=extra)
    run.save_pairs_csv(result)
    run.save_quality(result)

    if save_arrays:
        run.save_arrays(result)

    try:
        from .figures import all_figures
        std_figs = all_figures(result, title_prefix=label)
        for name, fig in std_figs.items():
            run.save_figure(fig, name)
            import matplotlib.pyplot as plt
            plt.close(fig)
    except Exception as e:
        print(f"  !  Standard figures skipped: {e}")

    if figures:
        names = figure_names or [f"fig_{k}" for k in range(len(figures))]
        for fig, name in zip(figures, names):
            run.save_figure(fig, name)

    print(f"  Done.\n")
    return run
