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

- f8 is the dominant HTE driver (SHAP analysis across multiple studies)
- f0-f3 drive baseline conversion rate
- Features are randomly projected - no domain interpretation available

### 3. Criteo-Specific IV Structure

The dataset contains both `treatment` (randomized offer) and `exposure` (actual ad view), which creates a natural instrumental variable setup:
```
Z = treatment_offer  →  T = exposure  →  Y = conversion
```
- ATE estimated via standard meta-learners on `treatment` ≈ Intent-to-Treat (ITT)
- LATE (effect on compliers) requires IV methods: `DMLIV`, `IntentToTreatDRIV` from econml
- Non-compliance rate: fraction treated but not exposed (or vice versa) - check in EDA

### 4. Baseline Conversion Rates (approximate)

| Group | Conversion Rate |
|-------|----------------|
| Control | ~0.22% |
| Treatment | ~0.30% |
| Overall | ~0.29% |
| ATE | ~+0.08 pp full-data (relative lift ~+35-40%); ~+0.12 pp on the 500K sample |

(Approximate, derived from the documented overall CR ≈ 0.29% and treatment ratio ≈ 0.85 - confirm from notebook 00's executed output.)

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
2. Clip propensity scores to [0.01, 0.99]; even in RCTs, extreme scores appear in small subgroups
3. LightGBM > XGBoost as base learner on this dataset (faster, similar accuracy)
4. For meta-learners, fix nuisance hyperparameters first, then tune the CATE model
5. CausalForestDML combines DML robustness with forest flexibility and holds up well across studies
6. The S-Learner's built-in regularization helps here: the treatment effect is small, so a single model naturally shrinks it toward zero

### HPO

- Use DR-loss as the primary HPO objective (doubly robust, fast to compute)
- Run HPO on a 500K subsample with 5-fold CV; results transfer well to the full dataset
- Final model selection: Qini on the 20% hold-out (never touched during HPO)
- Optuna TPE sampler with MedianPruner works well (100 trials is sufficient)
- CausalForest and CausalForestDML have built-in `.tune()`; prefer it over manual Optuna

### Evaluation

- Always report both AUUC and Qini; they can disagree on model ranking
- Report Uplift@10% and @20% as business-interpretable metrics
- For CausalForest, segment the population into persuadables/uncertain/sleeping dogs using CIs
- Plot Qini curves for all models together; visual overlap tells you where models diverge

---

## Known Pitfalls on This Dataset

| Pitfall | Details |
|---------|---------|
| **ITT vs LATE confusion** | Using `treatment` as T gives ITT; using `exposure` gives biased ATE (non-random) |
| **Small ATE inflating noise** | ATE ≈ 0.08 pp - ranking metrics are noisy on small hold-outs |
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
| UpliftBench (2026) | Large-scale empirical comparison on Criteo v2.1 |

---

## Useful Links

- [scikit-uplift documentation](https://www.uplift-modeling.com/)
- [Criteo dataset page](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
- [EconML documentation](https://www.pywhy.org/EconML/)
- [CausalML documentation](https://causalml.readthedocs.io/)
- [Criteo large-scale ITE benchmark (GitHub)](https://github.com/criteo-research/large-scale-ITE-UM-benchmark)
