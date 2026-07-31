# PAMS-ViT: Polarization-Aware Multi-Scale Vision Transformer for Significant Wave Height Estimation

Official implementation of:

> Md Istiak Ahammed, Fengying Dang. **Polarization-Aware Multi-Scale Vision
> Transformer for Significant Wave Height Estimation.** *Ocean Engineering*
> (under review).

This repository hosts the WavePerception lab's polarimetric wave-sensing
codebase, of which PAMS-ViT is the primary contribution described in the
paper above.

---

## Overview

PAMS-ViT estimates significant wave height (**H_s**) directly from raw
four-channel polarimetric images. It introduces two components:

- **Learnable Stokes Fusion (LSF)** — replaces fixed analytic Stokes
  preprocessing (intensity, DoLP, AoP) with a jointly optimized 1x1
  convolutional channel fusion, learned end-to-end with the rest of the
  network.
- **Multi-Scale Polarimetric Tokenization (MSPT)** — tokenizes the fused
  feature map at three patch scales (16, 32, 56 pixels) to jointly capture
  fine, intermediate, and coarse wave surface structure.

Under a leakage-free 12-fold leave-one-wave-train-out (LOWTO)
cross-validation protocol, PAMS-ViT achieves the lowest pooled RMSE
(0.134 cm, R² = 0.975) among seven evaluated models, using 2.49M
parameters — roughly 9x fewer than the strongest convolutional baseline
(InceptionV3).

---

## Repository structure

```
WavePerception/
├── configs/
│   ├── exp1_raw_input/        # Configs: raw polarimetric input [I0,I45,I90,I135]
│   └── exp2_manual_stokes/    # Configs: manually computed Stokes input [I,DoLP,sin(AoP),cos(AoP)]
├── scripts/
│   ├── exp1_raw_input/        # Shell scripts + result aggregation for the raw-input experiments
│   └── exp2_manual_stokes/    # Shell scripts + result aggregation for the manual-Stokes experiments
├── src/
│   ├── models/
│   │   ├── __init__.py        # Model registry (build_model, count_parameters)
│   │   └── pams_vit.py        # Proposed PAMS-ViT architecture (LSF + MSPT + transformer encoder)
│   ├── benchmarks/
│   │   ├── cnn.py             # CNN baseline
│   │   ├── mlp.py             # Shallow MLP baseline
│   │   ├── resnet.py          # ResNet34 baseline
│   │   ├── vit_baseline.py    # Vanilla ViT baseline
│   │   └── inception_v3.py    # InceptionV3 baseline
│   ├── experiments/
│   │   ├── lowto.py           # 12-fold LOWTO cross-validation protocol (core evaluation logic)
│   │   ├── single_split.py    # Single train/val/test split runner (quick local testing)
│   │   └── sensitivity.py     # Hyperparameter sensitivity sweep
│   ├── dataset_raw.py         # Dataset loader: raw polarimetric channels
│   ├── dataset_manual_stokes.py  # Dataset loader: manually computed Stokes-derived channels
│   ├── metrics.py             # RMSE, MAE, MSE, Bias, MAPE, pooled R²
│   ├── splits.py              # Wave-train-level disjointness / fold splitting
│   ├── stats.py                # Per-channel normalization statistics
│   ├── train_utils.py         # Training loop utilities
│   └── utils.py
├── main_raw.py                 # Entry point: train/evaluate on raw polarimetric input
├── main_manual_stokes.py       # Entry point: train/evaluate on manual Stokes-derived input
├── main_sensitivity.py         # Entry point: hyperparameter sensitivity sweep
├── results/                    # Output metrics, logs, checkpoints (populated after running)
├── requirements.txt
└── README.md
```

`src/models/` contains only the proposed PAMS-ViT architecture.
`src/benchmarks/` contains every baseline architecture it is compared
against. Both are accessed through the same registry
(`src/models/__init__.py`), so `build_model("pams_vit", params)` and
`build_model("resnet34", params)` are called identically.

---

## Installation

The code was developed and evaluated with:

- Python 3.10
- PyTorch 2.8.0
- CUDA 12.8

### 1. Clone the repository

```bash
git clone https://github.com/DANG-ResearchLab/WavePerception.git
cd WavePerception
```

### 2. Create an environment

Using conda (recommended):
```bash
conda create -n dl_wave python=3.10
conda activate dl_wave
pip install -r requirements.txt
```

Using venv:
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Install PyTorch separately if `requirements.txt` does not pin a
> CUDA-specific build for your system — see
> https://pytorch.org/get-started/locally/ for the correct command for
> your OS/CUDA version.

---

## Dataset

This work uses the publicly available polarimetric wave dataset:

> Ginio, N., Lindenbaum, M., Fishbain, B., Liberzon, D. (2025). Dataset of
> polarimetric images of mechanically generated water surface waves
> coupled with surface elevation records by wave gauges linear array.
> *Data in Brief*, 58, 111267.
> https://doi.org/10.1016/j.dib.2024.111267

**Download:** https://doi.org/10.57760/sciencedb.13968

After downloading, update the dataset path in the relevant config file(s)
under `configs/exp1_raw_input/` and `configs/exp2_manual_stokes/` to point
to your local copy (see the `data_root` or equivalent field in each YAML).

---

## Usage

All experiments are launched through one of the three entry points, each
pointed at a YAML config file.

### Train and evaluate a single model under LOWTO (raw input)

```bash
python main_raw.py --config configs/exp1_raw_input/lowto_pamsvit_fullLSF.yaml
```

Available configs in `configs/exp1_raw_input/`:

| Config file | Model |
|---|---|
| `lowto_mlp.yaml` | Shallow MLP |
| `lowto_cnn.yaml` | CNN |
| `lowto_vit.yaml` | Vanilla ViT |
| `lowto_resnet34.yaml` | ResNet34 |
| `lowto_inceptionv3.yaml` | InceptionV3 |
| `lowto_pamsvit.yaml` | PAMS-ViT without LSF (ablation) |
| `lowto_pamsvit_fullLSF.yaml` | PAMS-ViT (full model) |
| `single_split_cnn.yaml` | Quick single train/val/test split (no full 12-fold LOWTO), for local sanity checks |
| `sensitivity_pamsvit_fullLSF.yaml` | Hyperparameter sensitivity sweep config (Section 6.4) |

> **Note:** verify the `lowto_pamsvit.yaml` vs `lowto_pamsvit_fullLSF.yaml`
> mapping against your own config contents (`lsf: true/false` or
> equivalent field) before publishing — this table reflects the naming
> convention observed in the repository and should be double-checked.

### Train and evaluate under LOWTO (manually computed Stokes input)

```bash
python main_manual_stokes.py --config configs/exp2_manual_stokes/lowto_pams_vit_nolsf_exp2_manual_stokes.yaml
```

Configs in `configs/exp2_manual_stokes/` follow the same per-model naming
pattern, reproducing the input-representation ablation in Section 6.2.2
(Table 6). Note that PAMS-ViT with full LSF is not evaluated under manual
Stokes input, since LSF operates directly on raw channels (see the paper,
Section 6.2.2).

### Hyperparameter sensitivity sweep (Section 6.4, Table 7)

```bash
python main_sensitivity.py --config configs/exp1_raw_input/sensitivity_pamsvit_fullLSF.yaml
```

### Running on a SLURM cluster

Shell scripts for cluster submission are provided per model in
`scripts/exp1_raw_input/` and `scripts/exp2_manual_stokes/` (for example,
`run_pamsvit_fullLSF_lowto.sh`, `run_resnet34_lowto.sh`). Submit with:

```bash
sbatch scripts/exp1_raw_input/run_pamsvit_fullLSF_lowto.sh
```

Adjust the SLURM partition, GPU request, and environment activation line
at the top of each script to match your cluster.

For quick local runs without a scheduler, use:

```bash
bash scripts/exp1_raw_input/run_local.sh
```

### Aggregating results across folds

After all 12 LOWTO folds for a model have completed:

```bash
python scripts/exp1_raw_input/aggregate_results.py
```

This computes the pooled RMSE, MAE, MSE, Bias, MAPE, and pooled R² (Eqs.
16–24 in the paper) from the per-fold outputs in `results/`.

For the hyperparameter sensitivity sweep:
```bash
python scripts/exp1_raw_input/aggregate_sensitivity.py
```

### Measuring computational efficiency (Table 8)

```bash
python scripts/exp1_raw_input/measure_efficiency.py
```

Reports trainable parameters, FLOPs, peak GPU memory, and CPU inference
latency for a given model, matching the profiling protocol in Section 6.5.

---

## Reproducing paper results

| Paper item | How to reproduce |
|---|---|
| Table 4 (architectural comparison) | Run all `lowto_*.yaml` configs in `configs/exp1_raw_input/`, then `aggregate_results.py` |
| Table 5 (component ablation) | Compare `lowto_vit.yaml`, `lowto_pamsvit.yaml` (no LSF), `lowto_pamsvit_fullLSF.yaml` |
| Table 6 (input representation ablation) | Compare matching configs across `exp1_raw_input/` and `exp2_manual_stokes/` |
| Table 7 (hyperparameter sensitivity) | `main_sensitivity.py` + `aggregate_sensitivity.py` |
| Table 8 (computational efficiency) | `measure_efficiency.py` for each model |
| Figures 4–6 (per-fold error behavior) | Generated from the aggregated fold-level outputs in `results/` |

---

## Citation

If you use this code or the PAMS-ViT architecture, please cite:

```bibtex
@article{ahammed2026pamsvit,
  title   = {Polarization-Aware Multi-Scale Vision Transformer for Significant Wave Height Estimation},
  author  = {Ahammed, Md Istiak and Dang, Fengying},
  journal = {Ocean Engineering},
  year    = {2026},
  note    = {In press}
}
```

(Full citation details will be updated upon publication.)

Please also cite the dataset:
```bibtex
@article{ginio2025dataset,
  title   = {Dataset of polarimetric images of mechanically generated water surface waves coupled with surface elevation records by wave gauges linear array},
  author  = {Ginio, Noam and Lindenbaum, Michael and Fishbain, Barak and Liberzon, Dan},
  journal = {Data in Brief},
  volume  = {58},
  pages   = {111267},
  year    = {2025},
  doi     = {10.1016/j.dib.2024.111267}
}
```

---

## License

[Specify license — e.g., MIT. Add a `LICENSE` file to the repository root.]

## Contact

Md Istiak Ahammed — mahamm2@mtu.edu
Michigan Technological University, MEEM Department
