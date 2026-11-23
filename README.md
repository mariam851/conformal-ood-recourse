
# Conformalized Tree-Based Gradient Ensembles for Out-of-Distribution Detection and Plausible Recourse

Codebase, experiments, and manuscript for **Conformalized Tree-Based Gradient Ensembles for Out-of-Distribution Detection and Plausible Recourse**, studying conformalized out-of-distribution (OOD) gating coupled with actionability-constrained counterfactual recourse on gradient-boosted decision trees (GBDT).

## Overview

Optimizing counterfactual recourse solely against classifier confidence or narrow proxy scores on tabular tree ensembles can push candidates into sparse or out-of-distribution regions (an instance of Goodhart's law).

This repository implements:

- **Actionability-Constrained Recourse Search:** Evolutionary search enforcing feature immutability, monotonicity, and high-mode-share/zero-inflation protection (`capital-loss`, `capital-gain` on `Adult`), while tracking feasibility (`found`), achieved probability (`prob`), normalized `cost`, modified feature count (`n_chg`), and perturbation stability (`rob`).
- **Conformalized OOD Gating ($\delta = 0.05$):** Finite-sample conformal quantile thresholds calibrated on held-out calibration scores for base anomaly scorers (`IF`, `LOF`, `kNN`), evaluated individually (`gate_IF`, `gate_LOF`) and conjunctively (`gate_ALL`).
- **Held-Out Plausibility Auditing:** Post-hoc evaluation against held-out `GMM` and `OCSVM` detectors alongside matched-pair Wilcoxon signed-rank tests with Holm-Bonferroni correction (`p_holm`) and ablation studies (`gen10`, `gen100`, `relax`).

## Benchmarks & Key Findings

Evaluated on **UCI Adult** and **Give Me Some Credit (GMSC)**.

### Adult

**Matched Pairs (N = 541, `gate_ALL` vs. `none`):**

- Significant reduction in held-out `OCSVM` flags:
  - $\Delta \text{flag}_{\text{OCSVM}} = -0.0296$
  - $p_{\text{Holm}} = 4.0 \times 10^{-4}$
- Modest paired cost shift:
  - $\Delta \text{cost} = +0.0536$
- Lower conditional held-out flags:
  - `OCSVM`: 0.1625 → 0.0664
  - `GMM`: 0.0670 → 0.0203

### Give Me Some Credit (GMSC)

**Matched Pairs (N = 540, `gate_ALL` vs. `none`):**

- Significant reductions in held-out flags:
  - `OCSVM`: $\Delta \text{flag}_{\text{OCSVM}} = -0.0346$, $p = 1.0751 \times 10^{-4}$
  - `GMM`: $\Delta \text{flag}_{\text{GMM}} = -0.1663$, $p = 3.5679 \times 10^{-16}$
- Higher matched recourse cost:
  - $\Delta \text{cost} = +0.3035$
  - $p = 1.3365 \times 10^{-5}$
- Stricter feasibility:
  - `found`: 0.9240 → 0.3182

## Repository Structure

```text
├── configs/          # Experiment and calibration configurations
├── data/             # Dataset utilities / splits (check .gitignore for raw files)
├── docs/             # Project notes and documentation
├── notebooks/        # Exploratory, calibration, OOD signal, and analysis notebooks
├── paper_figures/    # Figures included in the manuscript
├── results/          # Saved benchmark and ablation outputs
├── scripts/          # Execution and evaluation scripts
├── src/              # Core modules (gating, recourse search, evaluation)
├── tests/            # Unit and integration tests
├── api.py            # API interface (if applicable)
├── OOD_detection_Paper.pdf
├── pyproject.toml
└── requirements.txt
```

## Setup & Usage

### 1. Create and Activate a Virtual Environment

**Python 3.12+ is required.**

```bash
python -m venv .venv
```

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Tests

```bash
pytest
```

## Author

**Mariam Zakaria Moussa Ali**

Department of Computer Science (AI & Data Science)  
Arab Open University, Egypt

Email: `mariamza200400@gmail.com`  
GitHub: [mariam851](https://github.com/mariam851)
