
# Conformalized Tree-Based Gradient Ensembles for Out-of-Distribution Detection and Plausible Recourse

A research codebase implementing conformalized out-of-distribution (OOD) gating and actionability-constrained counterfactual recourse for gradient-boosted decision trees (GBDTs).

This repository contains the experimental framework, evaluation results, visualizations, and research manuscript accompanying the study.

---

## Overview

Counterfactual recourse methods that optimize solely for classifier confidence or narrow proxy objectives may generate unrealistic recommendations in sparse or out-of-distribution regions. This phenomenon reflects a form of Goodhart's law, where optimizing a proxy objective can compromise the actual plausibility of the resulting solutions.

To address this issue, this project combines conformalized OOD detection with actionability-constrained evolutionary recourse search.

### Key Components

- **Actionability-Constrained Recourse Search:** Evolutionary optimization enforcing feature immutability, monotonicity, and high-mode-share/zero-inflation protection (`capital-loss`, `capital-gain` in the Adult dataset). Evaluation tracks feasibility (`found`), achieved probability (`prob`), normalized cost (`cost`), modified feature count (`n_chg`), and perturbation stability (`rob`).

- **Conformalized OOD Gating ($\delta = 0.05$):** Finite-sample conformal quantile thresholds calibrated using held-out calibration scores for three anomaly detectors: Isolation Forest (`IF`), Local Outlier Factor (`LOF`), and k-Nearest Neighbors (`kNN`). The framework evaluates individual gates (`gate_IF`, `gate_LOF`) and their conjunction (`gate_ALL`).

- **Held-Out Plausibility Auditing:** Independent post-hoc evaluation using Gaussian Mixture Models (`GMM`) and One-Class SVM (`OCSVM`), complemented by matched-pair Wilcoxon signed-rank tests, Holm-Bonferroni multiple-testing correction (`p_holm`), and ablation experiments (`gen10`, `gen100`, `relax`).

---

## Benchmarks & Experimental Results

The framework is evaluated on two tabular classification benchmarks:

- **UCI Adult**
- **Give Me Some Credit (GMSC)**

The primary comparison evaluates conformalized gating (`gate_ALL`) against the ungated baseline (`none`) using matched pairs.

### 1. UCI Adult

**Matched-pair evaluation (N = 541)**

The conformalized gating strategy demonstrates a significant reduction in held-out OCSVM flags:

| Metric | Result |
|---|---:|
| Change in OCSVM flag rate | -0.0296 |
| Holm-adjusted p-value | $4.0 \times 10^{-4}$ |
| Change in recourse cost | +0.0536 |
| Conditional OCSVM flags | 0.1625 → 0.0664 |
| Conditional GMM flags | 0.0670 → 0.0203 |

The results indicate improved held-out plausibility, accompanied by a modest increase in recourse cost.

---

### 2. Give Me Some Credit (GMSC)

**Matched-pair evaluation (N = 540)**

The conformalized gating strategy reduces held-out anomaly flags:

| Metric | Result |
|---|---:|
| Change in OCSVM flag rate | -0.0346 |
| OCSVM p-value | $1.0751 \times 10^{-4}$ |
| Change in GMM flag rate | -0.1663 |
| GMM p-value | $3.5679 \times 10^{-16}$ |
| Change in recourse cost | +0.3035 |
| Cost p-value | $1.3365 \times 10^{-5}$ |
| Feasibility (`found`) | 0.9240 → 0.3182 |

These results highlight the trade-off between improved held-out plausibility and recourse feasibility: stronger plausibility constraints can substantially reduce the proportion of successfully generated counterfactuals.

### Experimental Visualizations

The following figures illustrate the experimental results and analysis.

![Adult Experimental Results](https://github.com/user-attachments/assets/bf55980d-e8af-4f35-ae3b-c8790f82b781)

![Additional Experimental Analysis](https://github.com/user-attachments/assets/4db37811-af97-4faa-83d2-b66381836e67)

---

## Repository Structure

```text
├── configs/              # Experiment and calibration configurations
├── data/                 # Dataset utilities and data splits
├── docs/                 # Project documentation and notes
├── notebooks/            # Exploratory and experimental notebooks
├── paper_figures/        # Figures used in the research manuscript
├── results/              # Benchmark results and ablation outputs
├── scripts/              # Experiment execution and evaluation scripts
├── src/                  # Core implementation
│   ├── gating/           # Conformalized OOD gating
│   ├── recourse/         # Counterfactual recourse search
│   └── evaluation/       # Evaluation and statistical analysis
├── tests/                # Unit and integration tests
├── api.py                # API interface (if applicable)
├── OOD_detection_Paper.pdf
├── pyproject.toml
└── requirements.txt
```

---

## Installation

### Requirements

- Python 3.12+
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/mariam851/REPOSITORY_NAME.git
cd REPOSITORY_NAME
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it using:

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running Tests

Run the test suite using:

```bash
pytest
```

---

## Research Manuscript

The accompanying manuscript is available in:

[`OOD_detection_Paper.pdf`](OOD_detection_Paper.pdf)

---

## Author

**Mariam Zakaria Moussa Ali**

Department of Computer Science (AI & Data Science)  
Arab Open University, Egypt

- **Email:** mariamza200400@gmail.com
- **GitHub:** [mariam851](https://github.com/mariam851)
