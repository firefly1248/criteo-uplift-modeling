# Criteo Uplift Dataset - Research Findings & Best Practices

## Dataset Overview

| Property | Value |
|----------|-------|
| Source | Criteo AI Lab |
| Rows | ~13.98M (v2.1) |
| Features | **12** anonymized (f0-f11), random projection of originals |
| Treatment | `treatment` - ad offer (instrument); `exposure` - ad view (endogenous) |
| Targets | `conversion` (primary), `visit` (secondary) |
| Available via | Hugging Face mirror `criteo/criteo-uplift` (sklift's S3 bucket is dead, 403), Kaggle |
| Randomization | Bernoulli, **~85% treated / 15% control** (treatment ratio ≈ 0.85) - unequal allocation, not 50/50 |

**Loading:**
```python
from criteo_data import fetch_criteo   # local drop-in; pulls from the HF mirror
dataset = fetch_criteo(target_col='conversion')
data, target, treatment = dataset.data, dataset.target, dataset.treatment
```

---

## Key Research Findings

### 1. Model Performance (external - "UpliftBench 2026"; verify before quoting)

These figures come from a cited large-scale comparison, **not** from this repo's runs. Their Qini scale is likely a different normalization than sklift's `qini_auc_score`, so do not expect these notebooks to reproduce them numerically. Reproduce on your own executed run and treat that as authoritative.

- Reported winner: S-Learner + LightGBM, Qini ≈ 0.376 - the intuition (gradient boosting regularizes effectively when the treatment effect is small) is sound even if the exact number is not comparable here
- Reported top-20% targeting captures a large share (~78%) of incremental conversions
- Causal Forest reportedly finds only a small fraction of confident persuadables / sleeping dogs (α=0.10)
- Theoretical ranking (DR > R > X > T > S) does not robustly hold with strong base learners - decide via bootstrap tiers (notebook 06), not assertion

### 2. Feature Importance

- f8 is reported as the dominant HTE driver (SHAP across multiple studies). Here it tops mean |SHAP|
  on the treated-arm outcome model, which is not the CATE (see `eda_shap.png`)
- Ranking by LightGBM gain on the outcome model is a different ordering (f11, f4, f8, f2) - gain ranks
  drivers of E[Y|X], not of tau(x)
- Features are randomly projected - no domain interpretation available
- Scales differ by two orders of magnitude (sd from 0.023 for f11 to 7.03 for f9), and f4, f8, f11 are
  near point masses with tails. Irrelevant for trees, but it makes raw linear coefficients unreadable as
  an importance ranking - compare `coef * sd`, not `|coef|`. In notebook 03 the correction shrinks f11's
  apparent lead from 61x to 2.6x without changing who leads
- SHAP on the treated-arm outcome model and the standardized SparseLinearDML coefficients disagree
  (f8 first vs f11 first). Expected: a linear final stage sees only linear heterogeneity, and SHAP there
  explains E[Y|X,T=1] rather than tau(x)

### 3. Criteo-Specific IV Structure

The dataset contains both `treatment` (randomized offer) and `exposure` (actual ad view), which creates a natural instrumental variable setup:
```
Z = treatment_offer  →  T = exposure  →  Y = conversion
```
- ATE estimated via standard meta-learners on `treatment` ≈ Intent-to-Treat (ITT)
- LATE (effect on compliers) requires IV methods: `DMLIV`, `IntentToTreatDRIV` from econml
- Compliance is one-sided and this is exact, not approximate: `exposure=1 & treatment=0` has a count of
  0 across all 13,979,592 rows, so Pr(T=1 | Z=0) = 0 and monotonicity holds by construction. Offer
  take-up Pr(T=1 | Z=1) = 0.0360 on the full data

### 4. Baseline Conversion Rates (recomputed on the full v2.1 file)

| Group | Conversion Rate |
|-------|----------------|
| Control | 0.1938% |
| Treatment | 0.3089% |
| Overall | 0.2917% |
| ATE | +0.1152 pp full-data (SE 0.0000344, z = 33.5; relative lift +59.5%); +0.1252 pp on the 500K sample |

Computed directly over all 13,979,592 rows of the cached file, not derived. An earlier version of this
table listed control 0.22% and ATE +0.08 pp with a +35-40% lift; those are v1-dataset figures and do not
describe v2.1. The 500K sample's +0.1252 pp sits 0.5 SE from the full-data value, so the prototyping
sample is representative and needs no "sampling variation" explanation.

Very small absolute effects → metrics like AUUC and Qini are sensitive to noise; use large hold-out sets (≥1M rows) for stable evaluation. At the ~15% control share, control conversions are the scarcest cell and dominate the variance.

---

## Best Practices

### Data & Preprocessing

- No missing values in the dataset; skip imputation
- No categorical features; all features are continuous after projection
- Scale features for linear/neural models; tree models are scale-invariant
- Prototype on a 500K-row subsample (captures representative distributions); use 5M+ for final models
- Stratify splits by (treatment, conversion) to preserve rare conversion events in all folds

### Model Building

1. Always cross-fit nuisance models (propensity + outcome); fitting on the same data biases CATE estimates
2. In a randomized experiment, pass the known constant propensity instead of fitting one. Fitting a
   flexible propensity on a treatment that is Bernoulli(0.85) by design fits pure noise: the cross-fitted
   LGBM propensity here spans [0.497, 0.99] and inflates the AIPW standard error by 59% over a plain
   difference in means (notebook 04). Clipping to [0.01, 0.99] does not rescue this - it touches 0.0015%
   of rows. Clip anyway as a guard for confounded data, but do not treat it as the fix
3. LightGBM > XGBoost as base learner on this dataset (faster, similar accuracy)
4. For meta-learners, fix nuisance hyperparameters first, then tune the CATE model
5. CausalForestDML combines DML robustness with forest flexibility and holds up well across studies;
   on this repo's executed run it lands mid-table (rank 10 of 23), inside the leader's bootstrap tier
6. The S-Learner's built-in regularization is often argued to help when the effect is small, but it does
   not show up here: S-Learner ranks 13 (CatBoost) and 14 (LightGBM) of 23
7. Check the scale of tau_hat, not just its ranking. `R-Learner (CB)`, second on this leaderboard, predicts
   effects in [-1.62, +1.09] for a true effect of +0.00125, which is fine for Qini and useless for sizing
   a campaign

### HPO

- The known constant propensity matters for the DR pseudo-outcome: a fitted LGBM propensity supplied
  63% of its variance and biased its mean (notebook 05). It does not make DR-loss usable for HPO at this
  sample size: neither search (known or fitted e) beats a constant-mean model on paired out-of-fold error
  (winner z = +0.9 and -0.4), both pick the most shrunk config, and tuned-vs-baseline Qini goes opposite
  ways (0.271 vs 0.191 with the known e, 0.105 vs 0.159 with the fitted one)
- Signal check, run it every time: score a `DummyRegressor` on the same folds and report the winner's
  row-wise out-of-fold gain with a paired SE. A gain within about one SE of zero means the search is
  selecting noise. Do not bound `Var(tau)` from hold-out decile spreads; that leaks the hold-out
- Run HPO on a 500K subsample; notebook 05 uses 3-fold CV inside the training set
- Final model selection: Qini on the 20% hold-out (never touched during HPO)
- Optuna TPE sampler, 50 trials, no pruner (each trial is an atomic `cross_val_score`, nothing to prune)
- `econml.dml.CausalForestDML` has a built-in `.tune()`. `econml.grf.CausalForest` does **not** - it has
  no `tune` method in econml 0.16.0. Neither is used in this repo

### Evaluation

- AUUC and Qini can disagree in general, but not as sklift computes them on a constant-propensity
  design: the two curves differ pointwise only by `num_all/num_trmnt`, so the scores are proportional.
  Across all 23 rows of this repo's leaderboard the ratio AUUC/Qini is 0.0330-0.0338 and the rankings
  are identical. Reporting both here is convention, not corroboration
- Report Uplift@10% and @20% as business-interpretable metrics, but print the underlying counts with
  them. At this hold-out size the control slice of Uplift@10% holds 12-24 conversions, and the metric
  can rank a model first while it captures fewer treated conversions than the baseline
- For CausalForest, segment the population into persuadables/uncertain/sleeping dogs using CIs, and
  state the null rate: the persuadable rule is one-sided, so at alpha=0.10 chance alone yields ~5%
- Plot Qini curves for all models together; visual overlap tells you where models diverge

---

## Known Pitfalls on This Dataset

| Pitfall | Details |
|---------|---------|
| **ITT vs LATE confusion** | Using `treatment` as T gives ITT; using `exposure` gives biased ATE (non-random) |
| **Small ATE inflating noise** | ATE = +0.115 pp - ranking metrics are noisy on small hold-outs |
| **Memorizing nuisance** | Fitting propensity/outcome on the same data as CATE → overfit |
| **S-Learner underfitting** | Tree depth too small → treatment feature ignored; use min_child_samples carefully |
| **DA-Learner instability** | Domain adaptation can diverge if control/treatment distributions are too similar |

---

## Relevant Literature

| Paper | Key Contribution |
|-------|-----------------|
| Diemert et al. (2018) | Original Criteo dataset paper |
| Nie & Wager (2021) - "Quasi-oracle estimation" | R-Learner theory, τ-risk optimality |
| Kennedy (2023) - "Semiparametric doubly robust" | DR-Learner theory |
| Wager & Athey (2018) - "Estimation and Inference of HTE" | Causal Forest / GRF |
| Chernozhukov et al. (2018) - "Double/Debiased ML" | DML framework |
| Shi et al. (2019) - "Adapting Neural Networks for Uplift" | DragonNet |
| Künzel et al. (2019) - "Meta-learners for HTE" | X-Learner |
| Alaa, Ahmad & van der Laan (2023), arXiv:2308.14895 | Conformal meta-learners for ITE intervals |
| Lei & Candès (2021), arXiv:2006.06138 | Conformal counterfactual / ITE inference |
| arXiv:2604.06123 (2026) | Large-scale empirical comparison on Criteo v2.1 (referred to as "UpliftBench" in earlier drafts of these notes; cite the arXiv ID, the short name is not findable) |

---

## Useful Links

- [scikit-uplift documentation](https://www.uplift-modeling.com/)
- [Criteo dataset page](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
- [EconML documentation](https://www.pywhy.org/EconML/)
- [CausalML documentation](https://causalml.readthedocs.io/)
- [Criteo large-scale ITE benchmark (GitHub)](https://github.com/criteo-research/large-scale-ITE-UM-benchmark)
