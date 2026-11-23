# Literature notes

Status key: read = full text read · skim = abstract/intro only · notes = from my summary only (verify before citing)

| # | Work | What it does | Relation to us | Role in paper | Status |
|---|------|--------------|----------------|---------------|--------|
| 1 | TOOD, Shen et al., arXiv 2405.03060 | Tree embeddings + pairwise Hamming distance (APHD) over test samples | Starting point of C1 | Main method + baseline | skim |
| 2 | Novello, Dalmau, Andéol 2024 (OOD detection should use CP) | Corrects FPR/AUROC with calibration-conditional p-values; OOD scores as CP scores | Defines the conformal metrics we adopt | Metrics + related work | notes (verify arXiv id) |
| 3 | Bates et al., conformal p-values for outlier testing (workshop version: "Calibrated OOD Detection with Conformal P-values") | Conformal p-values from any OOD score; marginal validity, FDR control | Classical foundation of the p-value machinery | Related work | skim (find journal version) |
| 4 | CONFEX, Bilkhoo et al., arXiv 2510.19754 | CP + MILP counterfactuals, localised CP, tree partition; RF and MLP; Adult/German/GMSC/California | Closest to our recourse part | MUST be a baseline | read |
| 5 | ECCCo, Altmeyer et al., AAAI 2024 | CP set-size term in the counterfactual loss | Related | Related work / baseline | notes |
| 6 | CPICF, Adams et al., arXiv 2505.22326 | Individualised counterfactuals with CP intervals | Related | Related work | notes |
| 7 | Maalej et al. 2025 (PMLR v266) | Counterfactuals that explain conformal prediction sets | Related | Related work | skim |
| 8 | OCEAN, Parmentier & Vidal 2021 | Optimal MIP counterfactuals for tree ensembles + isolation-forest plausibility | Recourse baseline | Baseline | notes |
| 9 | FOCUS, Lucic et al., arXiv 1911.12199 | Differentiable relaxation for tree-ensemble counterfactuals | Recourse baseline | Baseline | notes |
| 10 | RobX, Dutta et al., ICML 2022 | Robust counterfactuals for tree ensembles; LOF realism | Source of stability metric | Metrics | notes |
| 11 | CPCF, arXiv 2605.06561 (May 2026) | Constraint programming, optimal CFs with plausibility | Recent, strong baseline candidate | Baseline if code exists | skim |
| 12 | LoBoost, arXiv 2602.22432 (Feb 2026) | Leaf-prefix partitions of boosted trees for local conformal regression | Leaf structure + CP, but regression | Related work | skim |
| 13 | Paper 1 (UQ and OOD for CE: pitfalls) | Trust score, MC-dropout, LOF for counterfactual quality | Motivation for the recourse gate | Related work | notes |
| 14 | NERO | Neuron-level relevance (LRP) for OOD, DNNs | Inspires the TreeSHAP relevance signal | Signals | notes |
| 15 | Medix | Median gradient statistics with wild data | Inspires the gradient-signature signal | Signals | notes |
| 16 | P²CE, arXiv 2606.18418 (Jun 2026) | Pareto-optimal counterfactuals with an auxiliary isolation forest inside the search; plausibility evaluated with a different detector (extended IF on test data) | Closest prior art for the IF gate AND for "evaluate with another detector" | MUST read; baseline candidate | skim (snippets only) |
| 17 | OCEAN, Parmentier & Vidal, ICML 2021 (PMLR v139) | IF plausibility constraints inside MIP counterfactuals for tree ensembles; LOF-based variant | Prior art for an IF gate on tree ensembles | Baseline (needs a MIP solver) | skim (snippets only) |
| 18 | "Don't Explain Noise", arXiv 2205.14116 | Robust counterfactuals for randomized ensembles; uses IF plausibility as a benchmark | Robustness + IF plausibility | Related work | skim (snippets only) |
| 19 | CEL benchmark, arXiv 2607.22045 | Counterfactual library and benchmark with LOF / IF outlier metrics | Evaluation practice; possible implementation source | Related work | skim (snippets only) |
| 20 | Guidotti, DAMI 2022 (CE review and benchmark) | Plausibility is not guaranteed except by endogenous methods; dace uses LOF in its loss | Context | Related work | skim (snippets only) |
| 21 | Native Guide, arXiv 2009.13211 | Time-series CE; plausibility checked with LOF / IF / OC-SVM; novelty detection called an imperfect proxy | Supports caution about detector-based plausibility | Related work | skim (snippets only) |
| 22 | PPCEF (normalizing flows), arXiv 2405.17640 | Plausible CE with flows; reports LOF, IF, probabilistic plausibility, log-density | Evaluation practice | Related work | skim (snippets only) |

### Questions to answer after reading (write the answers here)
1. Has anyone measured whether gating on one detector transfers to other detectors? (P²CE first.)
2. Is the margin vs robustness trade-off for tree-ensemble recourse already characterised (other-seed re-training and beyond)?
3. Do these papers use paired designs and calibrated references (real favourable rows) for plausibility?

## What we may claim today
- Conformal calibration of tree-embedding OOD scores on tabular data, with validity checks.
- Empirical comparison of GBDT-native signals under a unified conformal protocol.

## What we may NOT claim yet
- "First" anything, before the systematic search below is done.
- "Certified recourse" in the formal sense (CONFEX already addresses the exchangeability issue).
- Realism of counterfactuals: only measured empirically with detectors independent of our own.

## Systematic search protocol (to do)
1. Google Scholar "cited by" for #1, #2, #3. Record every hit.
2. arXiv listing (cs.LG, stat.ML), last 12 months: "tree embedding" + OOD, "conformal" + "tabular" + OOD, "gradient boosting" + OOD.
3. Check the workshops of NeurIPS/ICML on trustworthy ML and conformal prediction for 2025-2026.
4. Log each query, date, and result count here.
