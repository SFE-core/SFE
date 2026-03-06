# SFE — Turn Your Black Box Into a Mirror Box
**An observability layer that reads coupling geometry from any multivariate time series.**

The instrument computes three quantities for every pair of observers over a rolling window:
- **ρ\*** — mean absolute rolling correlation (coupling level)
- **dρ** — variance of ρ over the window (geometric stability)
- **r_eff** — effective rank of the joint covariance (dimensionality)

Together with the **band gap** (λ₁/λ₂ of the global covariance eigenspectrum), these four numbers characterize the coupling regime of any multivariate time series — across physical substrates, timescales, and domains — feeding AI models with a structured geometric foundation instead of raw noise.

Install: `pip install .`
Development install: `pip install -e .`

---

## Theoretical basis

**SFE-11 — Rank Geometry of the Relational Channel**
Calderas Cervantes, J.D. (2026). *Finite-Sample Calibration, Operating Envelope, and Cross-Domain Validation as an Observability Layer.* Zenodo. https://doi.org/10.5281/zenodo.18869381

Three formal results underpin the instrument:

1. **Rank-Collapse Theorem** — dρ < δ implies r_eff → 1 + O(δ): near-zero dρ with |ρ\*| ≈ 1 is equivalent to the joint covariance being approximately rank 1.
2. **Operating envelope** — the rolling estimator is reliable (bias < 4%) for ρ\* > 0.45. Non-stationarity is self-detectable: empirical dρ > (1−ρ\*²)²/W flags structural drift without external ground truth.
3. **Series Closure** — dρ ≈ 0 ∧ |ρ\*| ≈ 1 ⟺ Σ̂ ≈ λ₁uu⊤: the instrument's failure modes are self-detectable using only quantities already computed.

**Crisis coupling (Proposition 12):**
- **Branch A** (acute homogeneous shock, e.g. COVID-19): band gap explodes ≥ 1.5× background, ρ\* rises, r_eff collapses.
- **Branch B** (heterogeneous contagion, e.g. 2008 Lehman): ρ\* rises on > 50% of pairs, r_eff collapses, band gap stable.
- **Silent** on gradual corrections (e.g. dot-com 2000–02): neither branch fires.

**f(N) correction — self-detection of over-correction:**
The finite-sample correction f(N) = −0.106·ln(N) + 1.070 is valid for N=2–10 on synthetic OU data (MAE < 0.04). On real data with a single dominant mode (high band gap), f(N) can over-correct, producing r_eff_corrected < 1 — a physically impossible value since r_eff ∈ [1, N]. The instrument detects this automatically: if the corrected value falls below 1, it falls back to the raw joint mean and sets `result.reff_corr_fallback = True`. Discovered during the strain connector port (N=9, band gap 61.66×) — the first real-data boundary condition for Open Problem 3 of SFE-11.

---

## Project status

**Phase 0**
Strain connector complete. 
Finance in calibrating (mapping all domains).
---

## Connector status

| Connector | Domain | Dataset | Status | Runner |
|---|---|---|---|---|
| `strain.py` | Structural Health Monitoring | Strain rosette CSV (any sensor, any Hz) | Calibrated | `run_strain.py` |
| `finance.py` | Financial returns | yfinance / price CSV | In progress | `run_finance.py` |
| `eeg.py` | Motor cortex EEG | PhysioNet eegmmidb (.edf) | Pending | `run_eeg.py` |
| `traffic.py` | Urban traffic / sensor networks | ETTh1, METR-LA / PEMS | Pending | `run_traffic.py` |

---

## Running a domain

```bash
# Strain rosette — interactive (asks for confirmation before running)
python runners/run_strain.py data/sampledata.csv

# Strain rosette — auto mode (skip confirmation, use detected settings)
python runners/run_strain.py data/sampledata.csv --auto

# Strain rosette — explicit W
python runners/run_strain.py data/sampledata.csv --W 60

# Finance (in progress)
python runners/run_finance.py --tickers AAPL MSFT GOOGL NVDA \
    --start 2020-01-01 --end 2020-06-01 --W 20
```

Output is saved to `sfe_runs/<domain>_<label>_<timestamp>/`.

### Strain connector — data format

The strain connector auto-detects label format and W from sfreq. It supports:

**Label conventions (auto-detected in priority order):**
| Format | Example | Detected as |
|---|---|---|
| `device:gauge` | `50423:ch1` | separator=`:` |
| `device_gauge` | `DEV1_0deg` | separator=`_` |
| `device-gauge` | `A-0` | separator=`-` |
| No grouping | `CH1`, `CH2` | single device, all pairs cross |

**CSV layouts supported:**
```
# Annotated (recording device format):
SampleRate, 1Hz
...
DATA_START
timestamp, 50423:ch1, 50423:ch2, ...

# Plain CSV (any tool):
timestamp, CH1, CH2, CH3, ...
```

**Sensor types accepted:**
- Strain gauges (foil, FBG, vibrating wire) — rosette or linear
- Accelerometers (MEMS, piezoelectric)
- Displacement sensors (LVDT, GNSS)
- Load cells, temperature sensors, acoustic emission sensors

**W selection:**
If `--W` is not passed, W is auto-selected from sfreq via `suggest_W()`. The pre-run summary always shows W and its source before asking for confirmation.

---

## Validated cross-domain results

| Domain | Mechanism | ρ\* | Band gap | r_eff | Detection |
|---|---|---|---|---|---|
| Strain rosette | Static lock | 0.936 | 61.66× | 1.09† | 36/36, dρ=0 for 23h |
| ETTh1 (electricity) | Persistent lock | 0.963 | 4.23× | 1.04 | reliable |
| METR-LA (traffic) | Sync transition | 0.426 | 5.18× | >2.0 | indicative |
| EEG (motor cortex) | Disruption/relock | 0.804 ± 0.086 | — | 1.19 | 9/10 subjects ✓ |
| Finance full | Sector coupling | 0.654 | 7.11× | 1.89 | reliable |
| COVID crash (2020) | Acute homogeneous | 0.915 | 20.77× | 1.39 | Branch A ✓ |
| 2008 Lehman | Acute heterogeneous | 0.695 | 6.47× | 1.91 | Branch B ✓ |
| Dot-com (2000–02) | Gradual correction | 0.530 | 2.81× | 2.74 | correctly silent ✓ |

† r_eff joint mean reported. f(N) correction suppressed at N=9 (over-correction detected, fallback fired). See `result.reff_corr_fallback`.

Six orders of magnitude in timescale (seconds to months), five physical substrates, six distinct coupling regimes — same instrument, no modification.

---

## Output files (all domains)

| File | Contents |
|---|---|
| `summary.txt` | Full run summary: N, T, W, all pairs with ρ\*, dρ, r_eff, NS%, zone |
| `pairs.csv` | Machine-readable pair table |
| `quality.txt` | Data quality report: rows/cols dropped, NaN/Inf counts, warnings |
| `phase_portrait.png` | ρ\* vs dρ — position in operating envelope with cross-domain reference anchors |
| `timeseries.png` | Rolling ρ and dρ over time for top 6 pairs |
| `eigenspectrum.png` | Global eigenspectrum (band gap λ₁/λ₂) + r_eff joint trajectory |
| `heatmap.png` | ρ\* matrix with within-device grouping boxes *(strain)* |
| `diurnal.png` | Hourly dρ/ρ\* breakdown for best pair *(strain)* |

---

## Key implementation notes

### Self-detection of f(N) over-correction
`core.reff_corrected()` returns `(value, fallback_bool)`. If f(N) produces a value below 1, the raw joint mean is returned and `result.reff_corr_fallback = True` is set. Surfaces in `summary.txt`, `print_summary()`, and `repr(result)`.

### Strain connector — mirror box design
`from_strain_csv()` is interactive by default. Before any computation it prints detected label format, device grouping, sfreq, W with reasoning, and estimated pair counts — then asks `Proceed? [y/n]`. Pass `auto=True` or set `SFE_AUTO=1` to skip in scripted runs.

### Operating envelope
| Zone | Condition | Meaning |
|---|---|---|
| Reliable | ρ\* > 0.45 | Estimator bias < 4%, theorem applies |
| Marginal | 0.20 ≤ ρ\* ≤ 0.45 | Use with caution |
| Degraded | ρ\* < 0.20 | Estimator in negative-bias regime |
| NS flagged | NS% > 40% | Structural drift detected |

---

## Module structure

```
sfe/
├── core.py          # Numerical primitives: rolling_corr, rolling_drho, reff,
│                    # reff_joint, pair_table, band_gap, reff_corrected, f_N
├── connect.py       # Data cleaning + SFEResult. All connectors go through here.
├── connectors/
│   ├── strain.py    # Strain rosette — auto label detection, interactive confirm
│   ├── finance.py   # Log-returns, crisis detection (Branch A/B), slice_window
│   ├── eeg.py       # EDF/CSV biosignals, event-locked analysis, multi-subject
│   └── traffic.py   # ETT, PEMS-BAY/METR-LA, generic sensor CSV
├── figures.py       # Standard figures (phase portrait, timeseries, eigenspectrum)
├── outputs.py       # RunFolder, save_run — timestamped output management
├── ai.py            # LLM interpretation layer (Anthropic, OpenAI, Ollama)
└── w_select.py      # W selection: heuristic (Strategy 1) + sweep (Strategy 2)

runners/
├── run_strain.py    # CLI runner for strain domain
├── run_finance.py   # CLI runner for finance domain (in progress)
├── run_eeg.py       # CLI runner for EEG domain (pending)
└── run_traffic.py   # CLI runner for traffic domain (pending)
```

---

## Privacy note (AI layer)

`sfe.ai.interpret()` sends only derived metrics to the LLM endpoint — ρ\*, dρ, r_eff, NS%, band_gap, reff_corr, and column labels. Raw data never leaves the machine. For fully local inference: `LLMConfig(api_key="ollama", model="llama3", base_url="http://localhost:11434/v1", extra_headers={})`.

---

## Vision

Our goal is to turn the **Black Box** into a **Mirror Box**, and then a **Symbiotic Mirror**.

---

*Currently in experimental phase. Not financial, medical, or engineering advice.*