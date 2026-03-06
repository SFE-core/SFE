#!/usr/bin/env python3
"""
sfe12_3d.py — Absorption Signature Visualizer (SFE-12)

Genera las tres figuras para el paper:
    Fig 1: sfe12_calibration_scatter.png — espacio (mu_drho, CV_drho)
    Fig 2: sfe12_3d_portrait.png         — portrait 3D (rho*, mu_drho, CV_drho)
    Fig 3: sfe12_portrait_comparison.png — long vs short side-by-side

Usa los archivos de calibración en sfe_runs/ y los pares del VIX watch.
Corre desde el root del proyecto:
    python sfe12_3d.py

Requiere: matplotlib, numpy
No modifica nada del codebase existente.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import csv

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — ajusta estas rutas si tus runs están en otro lugar
# ─────────────────────────────────────────────────────────────────────────────
PAIRS_CSV = {
    "COVID (Branch A)":  "sfe_runs/finance/covid_calibration_*/pairs.csv",
    "Lehman (Branch B)": "sfe_runs/finance/lehman_calibration_*/pairs.csv",
    "Dot-com (silent)":  "sfe_runs/finance/dotcom_calibration_*/pairs.csv",
}

# Si prefieres pasar las rutas directas, úsalas aquí:
PAIRS_DIRECT = {
    "COVID (Branch A)":  None,   # e.g. "sfe_runs/finance/covid_.../pairs.csv"
    "Lehman (Branch B)": None,
    "Dot-com (silent)":  None,
}

# Datos del VIX watch 2026-03-06 (short window W=5) — del summary
VIX_WATCH = {
    "label":        "VIX Watch 2026",
    "rho_star_mean": 0.473,
    "mu_drho":       0.08745,
    "cv_drho":       0.182,
    "color":         "#ff4444",
    "marker":        "D",
}

# Colores y markers por evento
EVENT_STYLE = {
    "COVID (Branch A)":  {"color": "#ff6b35", "marker": "o"},
    "Lehman (Branch B)": {"color": "#ffd60a", "marker": "s"},
    "Dot-com (silent)":  {"color": "#888888", "marker": "^"},
    "VIX Watch 2026":    {"color": "#ff4444", "marker": "D"},
}

# Salida
OUT_DIR = Path(".")
DPI     = 150
BG      = "#0d0d0d"
PANEL   = "#111111"
GRID    = "#1e1e1e"
TEXT    = "#aaaaaa"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def style_ax(ax, is_3d=False):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    if not is_3d:
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(color=GRID, lw=0.5, alpha=0.6)
    else:
        ax.tick_params(colors=TEXT, labelsize=7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(GRID)
        ax.yaxis.pane.set_edgecolor(GRID)
        ax.zaxis.pane.set_edgecolor(GRID)
        ax.grid(color=GRID, lw=0.3, alpha=0.4)


def load_pairs_csv(path):
    """Carga pairs.csv y retorna lista de dicts."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "label":    row["label"],
                "rho_star": float(row["rho_star"]),
                "drho_mean": float(row["drho_mean"]),
            })
    return rows


def find_pairs_csv(glob_pattern):
    """Busca el primer archivo que coincide con el glob."""
    import glob as _glob
    matches = sorted(_glob.glob(glob_pattern))
    if matches:
        return matches[-1]   # más reciente
    return None


def compute_absorption_signature(pairs):
    """
    Calcula (mu_drho, sigma_drho, cv_drho, rho_star_mean) de una lista de pares.
    """
    drho_vals = np.array([p["drho_mean"] for p in pairs])
    rho_vals  = np.array([p["rho_star"]  for p in pairs])
    mu    = drho_vals.mean()
    sigma = drho_vals.std()
    cv    = sigma / mu if mu > 1e-10 else 0.0
    return {
        "mu_drho":       float(mu),
        "sigma_drho":    float(sigma),
        "cv_drho":       float(cv),
        "rho_star_mean": float(rho_vals.mean()),
        "n_pairs":       len(pairs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def load_all_events():
    """
    Carga los datos de calibración.
    Usa PAIRS_DIRECT si están definidos, si no busca con glob.
    Si no encuentra nada, usa los valores ya calculados del paper.
    """
    # Valores hardcoded del paper como fallback (del análisis del 2026-03-06)
    FALLBACK = {
        "COVID (Branch A)":  {"mu_drho": 0.01809, "cv_drho": 0.168, "rho_star_mean": 0.915, "n_pairs": 6},
        "Lehman (Branch B)": {"mu_drho": 0.01537, "cv_drho": 0.288, "rho_star_mean": 0.695, "n_pairs": 15},
        "Dot-com (silent)":  {"mu_drho": 0.01487, "cv_drho": 0.222, "rho_star_mean": 0.530, "n_pairs": 15},
    }

    events = {}
    for name, direct in PAIRS_DIRECT.items():
        path = direct
        if path is None:
            path = find_pairs_csv(PAIRS_CSV[name])

        if path and Path(path).exists():
            pairs = load_pairs_csv(path)
            sig   = compute_absorption_signature(pairs)
            events[name] = sig
            print(f"  Loaded {name}: {path}  ({sig['n_pairs']} pairs)")
            print(f"    mu_drho={sig['mu_drho']:.5f}  cv_drho={sig['cv_drho']:.3f}")
        else:
            events[name] = FALLBACK[name]
            print(f"  {name}: using paper values (no CSV found)")

    return events


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 1 — Calibration scatter (mu_drho vs CV_drho)
# ─────────────────────────────────────────────────────────────────────────────

def fig_calibration_scatter(events):
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
    style_ax(ax)

    # Umbrales
    ax.axhline(0.20, color="#ffd60a", lw=1.0, ls="--", alpha=0.7,
               label="CV threshold = 0.20")
    ax.axvline(events["Lehman (Branch B)"]["mu_drho"] * 3,
               color="#888888", lw=0.8, ls=":", alpha=0.5,
               label="3× Lehman baseline")

    # Regiones
    xlim = (0.0, 0.105)
    ax.axhspan(0.0,  0.20, xmin=0, xmax=1, color="#1a3a1a", alpha=0.18, zorder=0)
    ax.axhspan(0.20, 0.40, xmin=0, xmax=1, color="#1a2a3a", alpha=0.18, zorder=0)

    # Etiquetas de región
    ax.text(0.005, 0.19, "Homogeneous zone  (CV < 0.20)",
            color="#44aa66", fontsize=8, va="top", alpha=0.7)
    ax.text(0.005, 0.38, "Dispersed zone  (CV > 0.20)",
            color="#4488aa", fontsize=8, va="top", alpha=0.7)

    # Puntos históricos
    all_data = {**events, "VIX Watch 2026": {
        "mu_drho": VIX_WATCH["mu_drho"],
        "cv_drho": VIX_WATCH["cv_drho"],
        "rho_star_mean": VIX_WATCH["rho_star_mean"],
    }}

    for name, data in all_data.items():
        style = EVENT_STYLE.get(name, {"color": "#ffffff", "marker": "o"})
        size  = 220 if name == "VIX Watch 2026" else 160
        ax.scatter(data["mu_drho"], data["cv_drho"],
                   color=style["color"], marker=style["marker"],
                   s=size, zorder=5, edgecolors="#0d0d0d", linewidths=0.8,
                   label=name)

        # Anotación
        offset = (0.001, 0.008) if name != "VIX Watch 2026" else (0.001, -0.012)
        ax.annotate(
            name,
            xy=(data["mu_drho"], data["cv_drho"]),
            xytext=(data["mu_drho"] + offset[0], data["cv_drho"] + offset[1]),
            color=style["color"], fontsize=8, alpha=0.9,
        )

    ax.set_xlabel("μ(dρ)  —  mean cross-pair instability level", fontsize=9)
    ax.set_ylabel("CV(dρ)  =  σ(dρᵢⱼ) / μ(dρᵢⱼ)  —  cross-pair homogeneity", fontsize=9)
    ax.set_title("SFE-12 — Absorption Signature Space\n"
                 "Low CV + High μ = active channel suppression",
                 color="#ffffff", fontsize=10)
    ax.set_xlim(xlim)
    ax.set_ylim(0.0, 0.40)
    ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor=TEXT,
              loc="upper left", framealpha=0.8)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 2 — Portrait 3D (rho*, mu_drho, CV_drho)
# ─────────────────────────────────────────────────────────────────────────────

def fig_3d_portrait(events):
    fig = plt.figure(figsize=(11, 8), facecolor=BG)
    ax  = fig.add_subplot(111, projection="3d", facecolor=PANEL)
    style_ax(ax, is_3d=True)

    all_data = {**events, "VIX Watch 2026": {
        "mu_drho": VIX_WATCH["mu_drho"],
        "cv_drho": VIX_WATCH["cv_drho"],
        "rho_star_mean": VIX_WATCH["rho_star_mean"],
    }}

    for name, data in all_data.items():
        style = EVENT_STYLE.get(name, {"color": "#ffffff", "marker": "o"})
        size  = 300 if name == "VIX Watch 2026" else 180
        zorder = 10 if name == "VIX Watch 2026" else 5

        ax.scatter(
            data["rho_star_mean"],
            data["mu_drho"],
            data["cv_drho"],
            color=style["color"],
            marker=style["marker"],
            s=size,
            zorder=zorder,
            depthshade=False,
            label=name,
            edgecolors="#0d0d0d",
            linewidths=0.8,
        )

        # Proyecciones (sombras en los planos)
        ax.scatter(data["rho_star_mean"], data["mu_drho"], 0.0,
                   color=style["color"], marker="+", s=60, alpha=0.3, zorder=1)
        ax.scatter(data["rho_star_mean"], 0.0, data["cv_drho"],
                   color=style["color"], marker="+", s=60, alpha=0.3, zorder=1)
        ax.scatter(0.0, data["mu_drho"], data["cv_drho"],
                   color=style["color"], marker="+", s=60, alpha=0.3, zorder=1)

        # Líneas de caída al plano base
        ax.plot(
            [data["rho_star_mean"], data["rho_star_mean"]],
            [data["mu_drho"],       data["mu_drho"]],
            [0.0,                   data["cv_drho"]],
            color=style["color"], lw=0.5, alpha=0.25, ls=":",
        )

        # Etiqueta
        ax.text(
            data["rho_star_mean"] + 0.01,
            data["mu_drho"],
            data["cv_drho"] + 0.005,
            name, color=style["color"], fontsize=7, alpha=0.9,
        )

    # Plano umbral CV = 0.20
    rho_range = np.linspace(0.0, 1.0, 2)
    mu_range  = np.linspace(0.0, 0.10, 2)
    RHO, MU   = np.meshgrid(rho_range, mu_range)
    CV_PLANE  = np.full_like(RHO, 0.20)
    ax.plot_surface(RHO, MU, CV_PLANE,
                    alpha=0.08, color="#ffd60a", zorder=0)
    ax.text(0.5, 0.09, 0.20, "CV threshold = 0.20",
            color="#ffd60a", fontsize=7, alpha=0.6)

    ax.set_xlabel("ρ* mean", labelpad=8, color=TEXT, fontsize=8)
    ax.set_ylabel("μ(dρ)", labelpad=8, color=TEXT, fontsize=8)
    ax.set_zlabel("CV(dρ)", labelpad=8, color=TEXT, fontsize=8)
    ax.set_zlim(0.0, 0.40)
    ax.set_ylim(0.0, 0.10)
    ax.set_xlim(0.0, 1.0)

    ax.set_title("SFE-12 — Third Axis: Cross-Pair Dispersion\n"
                 "The 2026 observation occupies an uncharted region",
                 color="#ffffff", fontsize=10, pad=12)

    ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor=TEXT,
              loc="upper left", framealpha=0.8, bbox_to_anchor=(0.0, 0.95))

    # Ángulo de vista que muestra la separación máxima
    ax.view_init(elev=22, azim=-55)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 3 — Portrait comparison long vs short
# ─────────────────────────────────────────────────────────────────────────────

def fig_portrait_comparison():
    """
    Genera la comparación side-by-side de los portraits.
    Si tienes las figuras ya guardadas como PNG las combina,
    si no las tienes disponibles genera una versión esquemática.
    """
    
    long_png  = find_pairs_csv("sfe_runs/finance/VIX_watch_20260306_2026-03-06_012842/figures/phase_portrait.png")
    short_png = find_pairs_csv("sfe_runs/finance/VIX_watch_short_20260306_2026-03-06_014309/figures/phase_portrait.png")

    if long_png and short_png and Path(long_png).exists() and Path(short_png).exists():
        from PIL import Image
        img_long  = np.array(Image.open(long_png))
        img_short = np.array(Image.open(short_png))

        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG)
        ax_l.imshow(img_long)
        ax_r.imshow(img_short)
        ax_l.axis("off")
        ax_r.axis("off")
        ax_l.set_title("Long window  (W=20, 2 years)\n1/15 reliable pairs  •  CV=background",
                        color="#ffffff", fontsize=9)
        ax_r.set_title("Short window  (W=5, 6 months)\n11/15 reliable pairs  •  CV=0.182 ← ABSORPTION SIGNATURE",
                        color="#ff4444", fontsize=9)
        fig.suptitle("SFE-12 — Phase Portrait Comparison: Vertical Compression in Short Window",
                     color="#ffffff", fontsize=11, y=1.01)
        fig.tight_layout()
        return fig

    # Versión esquemática si no hay PNGs
    print("  PNGs de portraits no encontrados — generando versión esquemática")
    return _fig_portrait_schematic()


def _fig_portrait_schematic():
    """
    Versión esquemática del portrait comparison para cuando no hay PNGs.
    Muestra la diferencia de dispersión vertical entre long y short.
    """
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6),
                                      facecolor=BG, sharey=True)
    fig.suptitle("SFE-12 — Phase Portrait Comparison (schematic)\n"
                 "Vertical compression in short window = low CV(dρ) = absorption signature",
                 color="#ffffff", fontsize=10)

    # Long window: puntos dispersos verticalmente
    np.random.seed(42)
    rho_long  = np.random.uniform(0.15, 0.65, 15)
    drho_long = np.random.uniform(0.005, 0.035, 15)

    # Short window: comprimidos verticalmente (mismo rho*, drho uniforme alto)
    rho_short  = np.random.uniform(0.35, 0.62, 15)
    drho_short = np.random.uniform(0.072, 0.103, 15)  # rango estrecho — CV bajo

    colors = plt.cm.tab20(np.linspace(0, 1, 15))

    for ax, rho_vals, drho_vals, title, subtitle in [
        (ax_l, rho_long,  drho_long,
         "Long window  (W=20, 2 years)",
         "Dispersed vertically — organic structure\nCV(dρ) ~ background"),
        (ax_r, rho_short, drho_short,
         "Short window  (W=5, 6 months)",
         "Compressed vertically — homogeneous elevation\nCV(dρ) = 0.182 ← ABSORPTION SIGNATURE"),
    ]:
        style_ax(ax)
        for k, (r, d) in enumerate(zip(rho_vals, drho_vals)):
            ax.scatter(r, d, color=colors[k], s=60, alpha=0.88, zorder=5)

        ax.axvline(0.45, color="#ffd60a", lw=1.0, ls="--", alpha=0.6,
                   label="reliable ρ*>0.45")
        ax.axvline(0.20, color="#555555", lw=0.8, ls="--", alpha=0.5)

        # Bracket mostrando la compresión vertical
        ymin = drho_vals.min()
        ymax = drho_vals.max()
        spread = ymax - ymin
        ax.annotate("",
                     xy=(0.05, ymax), xytext=(0.05, ymin),
                     arrowprops=dict(arrowstyle="<->", color="#ffd60a", lw=1.2))
        ax.text(0.07, (ymax + ymin) / 2,
                f"σ(dρ)={spread:.4f}", color="#ffd60a", fontsize=7, va="center")

        ax.set_xlabel("ρ* (mean absolute rolling correlation)", color=TEXT, fontsize=8)
        ax.set_ylabel("dρ (mean correlation variance)", color=TEXT, fontsize=8)
        ax.set_title(title, color="#ffffff", fontsize=9)
        ax.text(0.02, 0.05, subtitle, transform=ax.transAxes,
                color=TEXT, fontsize=7.5, va="bottom",
                bbox=dict(facecolor="#0d0d0d", alpha=0.7, pad=3))
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.005, 0.12)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TABLA IMPRESA — resumen de calibración
# ─────────────────────────────────────────────────────────────────────────────

def print_calibration_table(events):
    baseline = events["Lehman (Branch B)"]["mu_drho"]
    print()
    print("=" * 72)
    print("  SFE-12 ABSORPTION SIGNATURE CALIBRATION TABLE")
    print("=" * 72)
    print(f"  {'Event':<25} {'μ(dρ)':>10} {'σ(dρ)':>10} {'CV(dρ)':>8} {'R_drho':>8}  Classification")
    print("  " + "-" * 70)

    all_data = {**events, "VIX Watch 2026": {
        "mu_drho": VIX_WATCH["mu_drho"],
        "cv_drho": VIX_WATCH["cv_drho"],
        "rho_star_mean": VIX_WATCH["rho_star_mean"],
        "sigma_drho": VIX_WATCH["mu_drho"] * VIX_WATCH["cv_drho"],
    }}

    for name, data in all_data.items():
        mu    = data["mu_drho"]
        cv    = data["cv_drho"]
        sigma = data.get("sigma_drho", mu * cv)
        R     = mu / baseline
        cls   = "homogeneous" if cv < 0.20 else "dispersed"
        if name == "VIX Watch 2026":
            cls = f"*** ABSORPTION ({cls})"
        print(f"  {name:<25} {mu:>10.5f} {sigma:>10.5f} {cv:>8.3f} {R:>8.2f}×  {cls}")

    print()
    print(f"  Threshold:  CV < 0.20  AND  R_drho > 3.0  →  active channel suppression")
    print(f"  Baseline:   Lehman μ(dρ) = {baseline:.5f}")
    print("=" * 72)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 62)
    print("  SFE-12 Absorption Signature Visualizer")
    print("=" * 62)
    print()

    # Carga datos
    events = load_all_events()
    print_calibration_table(events)

    # Figura 1
    print("Generating Fig 1: calibration scatter...")
    fig1 = fig_calibration_scatter(events)
    out1 = OUT_DIR / "sfe12_calibration_scatter.png"
    fig1.savefig(out1, dpi=DPI, bbox_inches="tight", facecolor=fig1.get_facecolor())
    print(f"  Saved: {out1}")
    plt.close(fig1)

    # Figura 2
    print("Generating Fig 2: 3D portrait...")
    fig2 = fig_3d_portrait(events)
    out2 = OUT_DIR / "sfe12_3d_portrait.png"
    fig2.savefig(out2, dpi=DPI, bbox_inches="tight", facecolor=fig2.get_facecolor())
    print(f"  Saved: {out2}")
    plt.close(fig2)

    # Figura 3
    print("Generating Fig 3: portrait comparison...")
    fig3 = fig_portrait_comparison()
    out3 = OUT_DIR / "sfe12_portrait_comparison.png"
    fig3.savefig(out3, dpi=DPI, bbox_inches="tight", facecolor=fig3.get_facecolor())
    print(f"  Saved: {out3}")
    plt.close(fig3)

    print()
    print("Done. Copy PNGs to your LaTeX folder and compile SFE-12v1.tex")
    print()
    print("Para el paper:")
    print("  sfe12_calibration_scatter.png  → Fig 1")
    print("  sfe12_3d_portrait.png          → Fig 2")
    print("  sfe12_portrait_comparison.png  → Fig 3")
    print()
    print("Si tienes los phase_portrait.png de tus runs en:")
    print("  sfe_runs/finance/VIX_watch_20260306_long_*/figures/phase_portrait.png")
    print("  sfe_runs/finance/VIX_watch_20260306_short_*/figures/phase_portrait.png")
    print("Fig 3 usará las imágenes reales en lugar del esquema.")
    print()


if __name__ == "__main__":
    main()