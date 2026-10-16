# Factorized Multilinear Codebook Optimization (MPR)

Code, archived results and analysis scripts for

> J. Zhuang and Y. Chen, "Factorized Multilinear Codebook Optimization for Large Phased
> Arrays and Reconfigurable Intelligent Surfaces," submitted to *IEEE Access*, 2026.

MPR (multilinear polynomial relaxation) configures finite-state phased arrays and RISs
directly from their complex hardware state tables (ideal, model-based, measured,
element-dependent or multi-band). Each element's $K$-bit table is encoded by its
multilinear extension and kept factorized through the field computation, which gives
closed-form bit gradients for smooth power objectives and exact single-element state gains
for weighted quadratic objectives. A batched relax–round–refine solver evaluates both with
GPU matrix products (PyTorch, no custom kernels).

## Repository layout

```
mpr/            core library: encoding (Möbius transform, Gray map), objectives and
                closed-form gradients, solvers (MPR, SR, CQ, manifold, GA), exact
                single-receiver optimum, measured codebooks, BRISC loader
experiments/    one script per experiment (e1b … e12), the multi-GPU scheduler
                (run_campaign.py) and make_figures.py (figures, tables, in-text numbers)
analysis/       analyze_results.py: additional checks behind statements in the text
tests/          correctness tests (exact gradients/gains, brute-force optima, …)
examples/       quickstart.py: measured 3-bit table -> RIS configuration -> bias voltages
data/codebooks/ measured unit-cell table (Fara et al.) and digitized 4-bit table (Zhang et al.)
data/brisc/     download and preprocessing instructions for the BRISC channel dataset
results/        raw results used in the paper (CSV/JSON)
  main/         synthetic experiments and the 3-bit measured table
  measured/     4-bit measured table, 4-bit optimality, BRISC channels
  ablation/     Gray vs. natural-binary mapping and number of candidates
outputs/        regenerated from results/ by make_figures.sh: figs/, tables/
                (incl. numbers.tex), results_summary.md, analysis_checks.md
```

## Installation

Python ≥ 3.10.

```bash
pip install -r requirements.txt
```

The paper's experiments ran on NVIDIA RTX PRO 6000 GPUs; everything below also runs on a
CPU (slowly for the full campaign).

## Quick start

```bash
python examples/quickstart.py
```

loads the measured varactor table, selects 8 states without interpolation, optimizes a
32 × 32 RIS for 8 users, 8 leakage points and 16 subcarriers, and prints the bias voltage
of each element.

Basic use of the library:

```python
import torch
from mpr import practical_ris_codebook, codebook_to_coeffs, Objective, ris_problem, mpr

a, B, d, info = ris_problem(Nx=32, Ny=32, n_users=8, n_leak=8, n_freq=16, seed=0)
obj = Objective(a, B, codebook_to_coeffs(practical_ris_codebook(3)), d)  # min sum_i a_i |z_i|^2
res = mpr(obj, P=64, iters=300, refine_sweeps=10)    # refine_sweeps=0: relaxation only
print(res["y"], res["time"])                         # res["s"]: bit pattern per element
```

## Reproducing the paper

**Figures, tables and every number quoted in the text** (CPU, under a minute):

```bash
bash make_figures.sh
```

reads `results/` and writes `outputs/`. `outputs/tables/numbers.tex` defines one LaTeX
macro per number quoted in the manuscript; regenerating it from the archived results
reproduces the submitted values exactly.

**Tests:**

```bash
python -m pytest -q tests        # or: python tests/test_core.py, etc.
```

**Full experiment campaign** (about 5 h on 8 GPUs):

```bash
bash run_experiments.sh           # runs the tests, then all experiments in the background
bash run_experiments.sh status    # progress and ETA
MPR_RESULTS=$PWD/experiments/results_v2 bash make_figures.sh   # figures from the new run
```

Finished chunks are skipped on restart. The Gray/P ablation is run separately:
`python experiments/e4_ablation.py`. The BRISC experiment needs the dataset (about 19 GB),
see [`data/brisc/README.md`](data/brisc/README.md); without it, e12 is skipped.
For a CPU smoke test of the whole pipeline: `python experiments/run_campaign.py --quick --gpus a,b`.

## Experiments and paper items

| Experiment | Script | Result file(s) | Paper |
|---|---|---|---|
| Gradient cost and scaling | `e1b_scaling.py` | `main/e1b_scaling.csv` | Fig. 1 |
| Loss to the exact single-receiver optimum | `e11_optimality.py` | `main/e11_optimality.csv`, `measured/e11_measured4.csv` | Fig. 2, Sec. VI-C |
| Wall-clock-matched comparison | `e5_anytime.py` | `main/e5_anytime.csv` | Fig. 3 |
| Array and RIS benchmarks | `e2_array.py`, `e3_ris.py` | `main/e2_array.csv`, `main/e3_ris.csv` | Tables 2–3, Fig. 4 |
| Measured 3-bit unit-cell table | `e6_measured.py --lut …` | `main/e6_measured.csv`, `main/e6_codebooks.json` | Figs. 5–6 |
| Measured 4-bit table, dual band | `e6_measured.py --table …` | `measured/e6_measured_table.csv`, `e6_dualband_table.csv`, `e6_codebooks_table.json` | Figs. 7–8 |
| Measured RIS channels (BRISC) | `e12_brisc.py` | `measured/e12_brisc*.csv` | Fig. 9, Sec. VI-F |
| Element-dependent tables | `e7_variation.py` | `main/e7_variation.csv` | Fig. 10 |
| Max–min fairness | `e8_fairness.py` | `main/e8_fairness.csv` | Fig. 11 |
| Gray mapping and candidates | `e4_ablation.py` | `ablation/e4_ablation.csv` | Fig. 12 |
| Hyper-parameter sensitivity | `e9_sensitivity.py` | `main/e9_sensitivity.csv` | Fig. 13 |
| Beam pattern | `e10_pattern.py` | `main/e10_pattern.csv` | Fig. 14 |

Methods in the result files: `CQ`, `CQ+SR` (continuous-then-quantize), `MO`, `MO+SR`
(manifold optimization), `GA`, `SR` (successive refinement from random starts), `MPR`
(relaxation only), `MPR+SR` (full algorithm) and, in e12 only, `MPR+SR-dedup`
(`mpr(..., dedup=True)`, candidate diversification before refinement).
`jobs.tsv` lists the run time of each chunk of the original campaign.

## Data sources and licenses

| Data | Source | License |
|---|---|---|
| `data/codebooks/unit_cell_look_up_table.txt` | Fara et al., Zenodo, doi:10.5281/zenodo.7674708 | CC BY 4.0 |
| `data/codebooks/measured_16state.csv` | digitized from Fig. 6 of Zhang et al., *Adv. Electron. Mater.* 2025, doi:10.1002/aelm.202500446 (procedure in `measured_16state_SOURCE.md`) | values derived by the authors of this repository |
| BRISC channels (not included) | Piana et al., arXiv:2602.21102; Zenodo, doi:10.5281/zenodo.18714621 | CC BY 4.0 |

The code is released under the MIT License (`LICENSE`).

## Citation

See `CITATION.cff`. Until the paper is published, please cite the manuscript as above.

## Contact

Yaocheng Chen (corresponding author), Universidade de São Paulo — yaochengchen@usp.br
