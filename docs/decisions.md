# Design decisions log

Short, dated entries. Each one will become a sentence in the paper's method or limitations section.

## D0 — Scope after novelty check (2026-10-10)
- **Decision:** C1 (conformal-valid tree-embedding OOD detection) is the spine of the paper; C2 = GBDT-native signals with p-value fusion; C3 = OOD-gated recourse as an empirical study, not a "certified" claim.
- **Why:** CONFEX (arXiv 2510.19754) already covers conformal guarantees for counterfactuals, including the exchangeability problem. Conformal p-values for OOD are classical (Bates et al.).
- **Open questions:** (1) does TOOD's APHD, computed among test samples, break exchangeability? Read its Section 3 and write down the answer. (2) Is there prior work on conformalizing TOOD? Systematic search pending.

## D1 — Four-way split (train / calib_fit / calib / test)
- **Decision:** keep a held-out `calib_fit` reference sample, separate from `calib`.
- **Why:** signals such as APHD need an ID reference set. Using training rows biases scores
  (the trees were fit on them); reusing `calib` as reference would break exchangeability with test scores.
- **Cost:** ~10% less data for training. Revisit if a dataset is very small.

## D2 — Tabular data first
- **Decision:** Adult -> German Credit -> HELOC -> others. Vision/NLP only as an optional appendix.
- **Why:** recourse is only meaningful on actionable features, and trees are strongest on tables.

## D3 — Adult feature handling
- **Decision:** drop `fnlwgt` (census sampling weight) and `education` (duplicates ordinal `education-num`).
- **Immutable features for recourse:** age, sex, race, native-country.
- **Open question:** is `marital-status` / `relationship` actionable? Decide before the recourse stage.

## D4 — Encoding happens after splitting
- **Decision:** loaders do no encoding or scaling; any fitted preprocessing is fit on `train` only.
- **Why:** avoids leakage from calibration/test into the model.

## D5 — Early stopping uses a validation split carved from `train`
- **Decision:** `GBDTClassifier` holds out 10% of `train` for early stopping. Never `calib_fit`, `calib` or `test`.
- **Why:** selecting the model with the scores that conformal calibration relies on would void the guarantee.
- **Cost:** the final model sees 90% of `train`.

## D6 — OOD scenarios are paired, built from `test`, with statistics from `train`
- **Decision:** shift rows of `test`; per-feature std is measured on `train`; severity in units of std; no clipping.
- **Why:** paired design isolates the effect of the shift from sampling noise; `train` statistics avoid leaking test information.
- **Limitation:** shifted points can be implausible. Natural shifts (TableShift / folktables) are a separate family, to be added.

## D7 — Two artifacts found with the C2ST diagnostic and fixed
- **Fractional-part artifact:** shifting integer-valued features produced fractional values; tiny shifts looked separable
  (C2ST 0.989 without rounding vs 0.517 with rounding, on synthetic data). Fix: round integer-valued features back.
- **Spike features:** `capital-gain` / `capital-loss` are ~92-95% zeros; shifting them in std units creates impossible values
  and makes every scenario trivially separable. Fix: `dense_numeric` excludes features with mode share > 0.75
  (`max_mode_share`). `all_numeric` stays available to quantify the effect.
- **TODO on real Adult:** confirm which features the rule drops and that no dense feature sits near the 0.75 threshold.

## D8 — Scope of OOD scenarios so far
- Only synthetic covariate shift exists. Semantic shift (class removal), natural shift (TableShift/folktables)
  and cross-dataset shift are still to build. Reviewers will expect natural shift; do not skip it.

## D9 — APHD in leaf-mass form, and two opposite sign conventions
- **Identity:** APHD(x | R) = 1 - average leaf mass of x's leaves under R. Tested against the brute-force pairwise definition.
- **TOOD (batch, transductive):** score = leaf mass among the batch itself; OOD points cluster, so high mass = OOD.
- **Reference (inductive):** score = APHD to a fixed ID reference (calib_fit); fixed function of x, so split-conformal is valid.
- **Bug caught:** using the TOOD sign against an ID reference gave AUROC < 0.5 on synthetic scenarios.

## D10 — Embedder depth is a hyperparameter that changes the SIGN of the inductive signal
- **Observation (synthetic data, to confirm on real Adult):** with fully grown trees (min_samples_leaf=1) ref_aphd has AUROC well below 0.5
  (0.17 at mean_shift@4); with larger min_samples_leaf it moves above 0.5. TOOD batch also flips on gaussian_noise for msl >= 20.
- **Decision:** default min_samples_leaf=1 (sklearn default), fixed BEFORE looking at OOD results. Never choose it by test-scenario AUROC (OOD leakage).
- **Open:** (a) two-sided conformal p-value for the inductive signal; (b) reproduce TOOD's exact hyperparameters from its paper;
  (c) any tuning must use a held-out scenario family or nested seeds.

## D11 — Conformal validity: what holds and what does not (REVISED with real Adult results)
- **Reference APHD + split conformal:** valid by construction (fixed function of x). Conservative with tied scores
  (mean FPR 0.0467 at alpha=0.05 on real Adult, 5 seeds; 0.0457 on 250 synthetic replicates).
- **Naive conformalised TOOD (batch scores):** the earlier synthetic experiment suggested the FPR on ID rows inflates under contamination.
  **Real Adult does not support that**: FPR 0.0466 (pi=0), 0.0443/0.0356 (pi=0.1), 0.0382/0.0001 (pi=0.5; noise / mean_shift).
  The FPR falls with contamination. Hypothesis (untested): OOD rows dilute the leaf mass that ID rows share, so ID rows look
  more typical than the clean calibration batch. Consequence: no theoretical guarantee, p-values are not calibrated and move
  with the contamination level; validity is "conservative by accident", power depends strongly on pi (AUROC < 0.5 at pi=0.1, small shifts).
- **Single runs fluctuate** by ~sqrt(alpha(1-alpha)/n_cal) (Bates et al.): always report mean +- sd over the 10 seeds.
- **Retracted:** the synthetic-data observation that model confidence rises under shift. On real Adult, msp AUROC is > 0.5 on mean_shift
  and < 0.5 on gaussian_noise, so its direction depends on the shift kind.

## D12 — Real-data findings from notebook 04 (Adult, seed 0)
- ref_aphd: AUROC 0.55-0.80 on mean_shift, 0.44-0.50 on gaussian_noise at msl=1 (inverted), 0.66-0.70 on the same noise at msl>=5.
- TOOD batch at pi=0.5 is the strongest score on both kinds (ties Isolation Forest on noise, beats it on mean_shift);
  at pi=0.1 it is below 0.5 on small shifts. For msl >= 5 TOOD batch inverts on noise while ref_aphd turns positive.
- The per-sample, conformally valid variant (ref_aphd) is mostly weaker than Isolation Forest. The data do not currently support a
  claim that the tree-embedding score beats the baselines; do not write that claim.

## D13 — Tools for direction-agnostic, exact and calibration-conditional conformal OOD (stage 4)
- Two-sided p-values: no direction needed (validity by union bound at alpha/2 per tail).
- Smoothed p-values: exact alpha with ties (exactly Uniform(0,1) under exchangeability); never mix with the conditional threshold.
- FPR+ (Beta bound, exact for continuous scores, conservative with ties) and the largest rank k meeting the target with probability 1 - delta.
- To check on real data (notebook 06): conditional rule keeps the worst seed at or below alpha; power cost vs the marginal rule.

## D14 — Real-data results of notebook 06 (Adult, 10 seeds, alpha=0.05, delta=0.01, n_cal=9768)
- **Conditional rule works:** the two-sided calibration-conditional rule kept the worst seed at or below alpha for every method
  (max 0.0464), while the marginal rule reached 0.0579 (msp); FPR+ of the marginal rule 0.0552 vs 0.0499 for the conditional one.
  Power cost is small: Isolation Forest two-sided 0.056 -> 0.049 (mean_shift@1), 0.130 -> 0.114 (mean_shift@2), about -12%.
- **Smoothing** helps only where scores tie (ref_aphd 0.0478 -> 0.0490); Isolation Forest and msp have no ties.
- **Retracted:** "AUROC < 0.5 means informative but inverted, and a two-sided test recovers it". On real Adult the two-sided test
  does NOT rescue ref_aphd (TPR 0.001-0.008) or msp on noise (0.036): those scores move in the BULK of the ID distribution, not in its tails.
- **AUROC is the wrong metric for conformal flagging:** ref_aphd has AUROC 0.74 on mean_shift@2 yet TPR 0.014 at alpha=0.05.
  Report TPR at alpha (and AUROC only as a diagnostic).
- **Per-row valid detectors:** Isolation Forest dominates (TPR 0.100-0.288 one-sided). ref_aphd (TPR <= 0.022) is not competitive;
  do NOT claim the per-row tree-embedding score beats the baselines.

## D15 — Batch-level permutation test (stage 5)
- **Why:** TOOD is a batch method; its score is not a fixed function of a row, so a per-row conformal guarantee does not apply.
  A permutation test on the pooled reference + batch is exactly valid under H0 for any statistic.
- **Statistics:** TOOD within-batch collision (two-sided), RF-kernel unbiased MMD^2 vs the reference (upper), batch mean of IF / MSP scores (two-sided).
- **Scope:** tests the batch, not rows. Needs iid batch under H0 and scorers fit on data independent of the pool.
- **Novelty risk:** a forest-kernel two-sample statistic is standard. Required before any claim: C2ST and standard-kernel MMD baselines,
  batch-size sweep, systematic literature search (docs/literature_notes.md protocol).
- **Verified here only on synthetic data with a mini run; real-data results pending (notebook 06b).**

## D16 — Pivot of effort to the recourse study; actionability choices (TO BE CONFIRMED by the author)
- **Why the pivot:** by the rule fixed before seeing the numbers, the batch-level RF-kernel test did not clearly beat the batch mean of
  Isolation Forest (notebook 06b, real Adult, 3 seeds): valid under H0 (rejection 0.039-0.078, alpha 0.05), but
  mean_iforest 0.844 vs rf_mmd 0.289 on gaussian_noise@2 with pi=0.10, and the rf_mmd advantage on mean_shift@1 at low pi (0.200/0.400
  vs 0.111/0.289) is inside the noise (45 batches per cell). TOOD's own statistic is weak and non-monotone once made valid
  (0.778 at pi=0.25, 0.467 at pi=0.5). Evidence does not support a tree-embedding detection claim.
- **Recourse = empirical study** (as in D0): same evolutionary search with and without an OOD gate; gate = Isolation-Forest
  calibration-conditional rule; realism measured with an INDEPENDENT detector (LOF); robustness = other-seed models.
- **Actionability on Adult (author to confirm):** immutable = age, sex, race, native-country, marital-status, relationship;
  capital-gain / capital-loss not actionable (spike rule, D7); education-num may only increase. Mutable: education-num,
  hours-per-week, workclass, occupation. If success rate is low on real Adult, relax marital-status / relationship and say so in the paper.
- **Gate semantics:** PASS = "not rejected by a test with FPR <= alpha". No coverage claim for the counterfactual (selected against the gate).
- **Baselines still missing:** DiCE, OCEAN (needs a MIP solver), CONFEX (check code availability). Do not claim superiority before they exist.

## D17 — Reading of notebook 06b (real Adult)
- H0 rejection rates (180 clean batches): tood 0.0778, rf_mmd 0.0611, mean_iforest 0.0444, mean_msp 0.0389 (alpha 0.05, nominal se 0.016;
  batches within a seed share the pool and scorers, so the real se is larger). tood is 1.7 se above alpha: rerun with larger N_H0 before relying on it.
- mean_msp matches rf_mmd on mean_shift for pi >= 0.25. A trivial baseline is as good as the forest-kernel statistic there.
- Not done: C2ST and standard-kernel MMD baselines, batch-size sweep, literature check of forest-kernel two-sample tests.

## D18 — First real-data reading of notebook 08 (Adult, seed 0, 100 of 7762 unfavourable test rows)
- success: no_gate 0.65, gate 0.62. Paired cost +0.058 [0.007, 0.124], n_changed +0.081 [0.016, 0.177].
- Own detector (Isolation Forest) flag rate 0.138 -> 0.065. Independent detector (LOF) flag rate 0.092 -> 0.065
  (about 6 vs 4 flagged counterfactuals): inside the noise. References: real favourable test rows 0.039, original unfavourable rows 0.090.
- robust (3 re-trained models) 0.626 vs 0.640: counterfactuals sit on the decision boundary (target_prob = 0.5); the gate does not change that.
- own_flag of the gated set is 0.065, not 0. Hypothesis (to be verified with the band cell): the conditional rule is more permissive than the
  plain alpha test (k 439 vs 488) and the search pushes counterfactuals against the gate threshold.
- **Conclusion so far:** no evidence that the gate improves realism under an independent detector. Not a claim.
- **Next:** N_INSTANCES >= 500, 10 seeds, paired comparison of lof_flag / own_flag / robust, target_prob sweep (0.5, 0.6, 0.7),
  relax marital-status / relationship, separate "no feasible CF" from "search budget too small", add DiCE / OCEAN / CONFEX baselines.

## D19 — Reading of notebook 08 with 500 individuals (Adult, seed 0); supersedes D18's reading
- Paired (both found, gate - no_gate): cost +0.024 [0.009, 0.044]; n_changed +0.020 [0.003, 0.041]; own_flag (IF) -0.014 [-0.027, -0.003];
  lof_flag +0.003 [0.000, 0.010]; robust -0.002 [-0.010, 0.006]. success 0.614 -> 0.588.
- **No detectable realism gain on an independent detector (LOF).** The N=100 reading (LOF 9.2% -> 6.5%) was noise;
  the reference rate of the original individuals moved from 0.090 (n=100) to 0.048 (n=500).
- **Goodhart:** the gate lowers the flag rate of its own detector (7.8% -> 2.4%) and not of LOF.
- **Headline vs paired:** the headline own_flag drop (5.4 points) is much larger than the paired drop (1.4 points) because the two
  methods succeed on different individuals; part of the "improvement" is the gate refusing individuals. Always report paired numbers.
- **Boundary hugging confirmed:** 7 of 294 gated counterfactuals (2.4%) lie between the plain alpha-test threshold and the gate threshold,
  a band holding 0.5% of in-distribution calibration rows.
- **Counterfactuals look less typical than the individuals they start from** (LOF flag ~9% with or without the gate, vs 4.8% originals, 3.9% real favourable).
- **Robustness is the main practical weakness:** robust = 0.53 at target_prob 0.5 (counterfactuals on the decision boundary).
- Open: margin sweep (section 6), detector transfer across gates (section 7), cause of ~40% no-solution cases (action space vs search budget).
- **Possible framing (to be decided after sections 6-7 and a literature check):** an audit of plausibility gates for tabular recourse.
  Novelty is NOT established. Venue level realistically workshop / arXiv.
  
## D20 — Margin sweep and cross-detector gates (Adult, seed 0; N=150 and N=200)
- Margin: robust (share of re-trained models still favourable) 0.63 at target 0.5 -> 0.96 at 0.6 -> 1.00 at >= 0.7. Price of 0.6 vs 0.5 (no gate):
  success 0.593 -> 0.527, cost 2.353 -> 2.698 (+15%). At 0.8 success is 0.32. Robustness here is only w.r.t. re-training with other seeds
  on the same data (a weak model change). robust at 0.5 was 0.53 (N=500) and 0.63 (N=150): the exact value is not stable.
- lof_flag rises with the margin (no gate 0.079 -> 0.104): trend, not established (se ~ 0.03 per point).
- Cross gates (N=200, ~118 counterfactuals per method): gate_IF lowers flag_IF 0.109 -> 0.036 but not flag_LOF (0.084 -> 0.080);
  gate_LOF lowers flag_LOF 0.084 -> 0.042 but not flag_IF (0.109 -> 0.110); kNN drops for both (0.101 -> 0.062 / 0.076), within noise.
  => detector-specific improvement (Goodhart), IF <-> LOF.
- Reference for counterfactuals = real favourable rows (flag IF 0.077, LOF 0.039, kNN 0.071), not alpha. gate_IF over-corrects IF
  (0.036 < 0.077) while LOF stays at ~2x the reference.
- Literature (searches of 2026-10, two queries, NOT systematic): IF-based plausibility constraints in tree-ensemble recourse exist
  (OCEAN, ICML 2021; P2CE, arXiv 2606.18418, which also evaluates with a different detector). The IF gate is NOT novel.
  Not found: a paired test of transfer across detectors. Must read P2CE before claiming anything.
- Plan: multi-seed benchmark (scripts/run_recourse_benchmark.py + notebook 08b), literature notes, then decide between an audit paper
  and a constructive multi-detector gate evaluated on a held-out detector.

## D21 — Benchmark pilot (Adult, seed 0, 200 individuals, targets 0.5/0.6/0.7) and corrections to D19/D20
- Replicated: robust ~0.54 (target 0.5) -> ~0.93 (0.6) -> 1.00 (0.7); gates do not change it (paired diff <= +0.016).
- Replicated: each gate lowers its own detector in paired terms (gate_IF: flag_IF -0.032 / -0.044; gate_LOF: flag_LOF -0.064 / -0.034 / -0.032);
  no transfer IF <-> LOF (all off-diagonal paired diffs in [-0.008, 0]).
- NOT replicated, retracted: (a) gate_LOF partially lowering flag_kNN (paired 0.000 at all targets); gate_IF -> kNN is borderline (-0.024, -0.027);
  (b) "LOF flag rises with the margin" (here 0.125 -> 0.075 -> 0.053 without gate); (c) "counterfactuals are flagged ~2x more than the original individuals"
  (original-row LOF rate moved 0.025-0.090 across samples, counterfactual rate 0.053-0.125).
- At target 0.7 the IF gate almost never binds (paired diffs 0.000). gate_LOF costs less success than gate_IF at 0.6 (0.595 vs 0.565; none 0.600).
- Statistics: one seed (sd = NaN); percentile bootstrap is poor for sparse binary differences, use exact McNemar on discordant pairs
  (about 10-20 discordant pairs at ~125 pairs -> low power; need >= 5 seeds).
- **Decision rule fixed BEFORE the 5-seed results:** if IF <-> LOF off-diagonal effects stay ~0 (CI contains 0 and |effect| < 1 point) -> audit framing and test a
  multi-detector gate on a held-out detector (e.g. GMM density), plus a second dataset and baselines. If transfer appears consistently (<= -2 points across seeds)
  -> method framing, still requires baselines (P2CE, OCEAN, DiCE).
  
## D22 — Pre-registered decision rule for the held-out gate experiment (written BEFORE the data)
- Experiment: gates none / gate_IF / gate_LOF / gate_ALL (IF and LOF and kNN, conjunction), target 0.6, 5 seeds x 200 individuals, Adult.
  Evaluators: IF, LOF, kNN (gate pool) and GMM, OCSVM (held-out, never used inside a gate).
- Verdict per (gate, detector not in the gate), paired effect gate minus none: transfer = diff <= -0.02, CI below 0, Holm p < 0.05;
  none = |diff| < 0.01 and CI contains 0; inconclusive otherwise.
- Outcomes: (a) gate_ALL transfers to >= 1 held-out detector and single-detector gates do not -> multi-detector gating generalises
  (partially), report with cost. (b) gate_ALL is `none` on both held-out detectors -> Goodhart persists; the negative finding is complete,
  stop adding gate variants. (c) otherwise inconclusive: more seeds, no claim.
- Thresholds are not to be changed after seeing the numbers.

## D23 — Result of the held-out experiment (Adult, seeds 0–4, N=200, target=0.6)
- **Numbers (paired `gate - none`, Holm-adjusted over 15 flag comparisons):**
  - `gate_IF`: `flag_GMM` diff = `-0.0035` (`p_holm = 1.0000`, `none`), `flag_OCSVM` diff = `-0.0142` (`p_holm = 0.1934`, `inconclusive`).
  - `gate_LOF`: `flag_GMM` diff = `-0.0051` (`p_holm = 1.0000`, `none`), `flag_OCSVM` diff = `-0.0101` (`p_holm = 0.4922`, `inconclusive`).
  - `gate_ALL` (`IF ∧ LOF ∧ kNN`): `flag_GMM` diff = `-0.0092` (`[-0.0185, 0.0000]`, `p_holm = 0.6250`, `none`), `flag_OCSVM` diff = **`-0.0296`** (`[-0.0444, -0.0166]`, **`p_holm = 0.0004`**, 4/5 seeds $\ge 2$ pts drop, **`transfer`**). Raw `flag_OCSVM` drops from `0.161 ± 0.032` (`none`) to `0.066 ± 0.018` (`gate_ALL`), matching the real favourable reference (`0.074 ± 0.010`).
- **Outcome (D22 rule):** **Outcome (a) — Constructive partial generalisation.** Single-detector gates suffer from Goodhart's law (no transfer to held-out detectors), whereas combining diverse detectors (`gate_ALL`) transfers significantly to the held-out boundary detector (`OCSVM`).
- **Cost & Ablations:** `gate_ALL` reduces `success` from `0.597 ± 0.044` to `0.542 ± 0.029` (-5.5 pts), with paired `cost` `+0.0536` (`[0.0338, 0.0790]`), `n_changed` `+0.0573` (`[0.0314, 0.0850]`), and `robust` `0.915 ± 0.033` (vs `0.901` for `none`). Ablations (`gen10` vs `gen100` vs `relax`) confirm that the ~40% unfound rate is driven by ethical actionability constraints (`marital-status`, `relationship` — relaxing them raises `gate_IF` success from `0.493` to `0.723`), not evolutionary search budget.

## D24 — Corrections and wording
- The Holm column in the first 5-seed rule table was wrong (values sorted, not aligned with rows). Correct Holm (from the discordant counts):
  only the own-detector effects survive (gate_IF -> IF at targets 0.5 and 0.6; gate_LOF -> LOF at 0.5, 0.6, 0.7). The gate_IF -> kNN
  "transfer" (diff -0.012 / -0.014, verdict inconclusive) does NOT survive (Holm ~ 0.2). paper_artifacts.holm is tested.
- Wording: the IF <-> LOF result is "no detectable transfer" (verdict `none` in 6 of 6 cells, 477-633 pairs, 5 seeds, Adult, one model,
  CIs optimistic because individuals are pooled over seeds). It is not a proof.
- Second dataset: the protocol needs n_cal of a few thousand (conditional rank 169 of 200 at n_cal=4000, 4 of 10 at n_cal=200);
  German Credit (1000 rows) is too small. Dataset choice and its actionability decisions are the author's.