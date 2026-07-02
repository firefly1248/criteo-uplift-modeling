# Uplift Model Evaluation - Metrics Guide

## The Fundamental Problem

Unlike standard supervised learning, uplift models cannot be evaluated with simple MSE or accuracy because the true treatment effect τ(x) = Y(1) − Y(0) is never observed for any single unit. All evaluation metrics must work with observed (Y, T) pairs and aggregate over groups or use importance weighting.

---

## 1. Ranking Metrics (Primary)

### 1.1 Uplift Curve & AUUC

**Construction:**
1. Score all units by predicted uplift τ̂(x), sort descending
2. At each fraction φ of the population, compute:
   ```
   Uplift(φ) = [conversions in top-φ% treated] / [treated in top-φ%]
              − [conversions in top-φ% control] / [control in top-φ%]
   ```
3. AUUC = area under this curve minus area under the random (diagonal) line

**Interpretation:** AUUC > 0 means the model beats random targeting. The curve shows how much incremental conversion you capture at each targeting threshold.

**Key property:** Sensitive to the absolute scale of uplift - good for absolute business impact assessment.

```python
from sklift.metrics import uplift_auc_score
auuc = uplift_auc_score(y_true, uplift_scores, treatment)
```

> Scale note: sklift's `uplift_auc_score` returns a normalized coefficient (area rescaled against the ideal curve), not the raw "area minus diagonal" of step 3, so its values are comparable to `qini_auc_score`, not to the literal formula above. All AUUC/Qini numbers in this repo are on that normalized scale.

---

### 1.2 Qini Curve & Qini Coefficient

**Construction:**
1. Same sorting as uplift curve
2. At each fraction φ:
   ```
   Qini(φ) = cumulative_treated_conversions(φ)
            − cumulative_control_conversions(φ) × [treated(φ) / control(φ)]
   ```
3. Qini coefficient = area between model curve and random baseline

**Difference from AUUC:** Qini uses cumulative counts (not rates), so it weights early gains by population size - better aligned with a fixed-budget targeting scenario.

**On Criteo:** a best Qini ≈ 0.376 (S-Learner + LightGBM) is reported in the literature ("UpliftBench 2026"). Treat this as external and not directly comparable to these notebooks' numbers - sklift's `qini_auc_score` is normalized against the perfect model, so values here typically sit far below 0.376. Reconcile the normalization before quoting it.

```python
from sklift.metrics import qini_auc_score
qini = qini_auc_score(y_true, uplift_scores, treatment)
```

---

### 1.3 Uplift@k

Uplift at a fixed targeting fraction k (typically 10%, 20%). Two conventions:

```
# multiplier form (business framing, "3.9x vs random"):
Uplift@k_mult = [CR_treated(top-k%) − CR_control(top-k%)] / CR_overall

# absolute form - what sklift's uplift_at_k(strategy='by_group') actually returns:
Uplift@k_abs  =  CR_treated(top-k%) − CR_control(top-k%)
```

In this repo the notebooks report the absolute form (sklift `uplift_at_k`, values ~0.00x), not the CR-normalized multiplier. Don't read a leaderboard "Uplift@10% = 0.003" as a "3.9x" multiplier - divide by the hold-out overall CR yourself if you want the multiplier.

**When to use:** Most natural business metric when the marketing budget is fixed.

---

## 2. Surrogate Losses for HPO (No Ground Truth Needed)

### 2.1 R-loss (τ-risk)

Based on Robinson (1988) decomposition:
```
L_R(τ) = E[(Ỹ - τ(X)·T̃)²]

where  Ỹ = Y − ŷ(X)    (outcome residual)
       T̃ = T − ê(X)    (propensity residual)
```

**Theoretical guarantee:** Minimizing L_R is equivalent to minimizing MSE on τ(x) when nuisance models ŷ, ê are consistent (Nie & Wager, 2021).

**Weakness:** Sensitive to errors in both nuisance models simultaneously.

**Use for HPO:** Stage 1 fast search when nuisance models are reliable.

---

### 2.2 DR-loss (Doubly Robust pseudo-outcome MSE) ← Recommended

```
Ỹ_DR = ŷ₁(X) − ŷ₀(X)
       + T·(Y − ŷ₁(X)) / clip(ê(X), 0.01, 0.99)
       − (1−T)·(Y − ŷ₀(X)) / clip(1−ê(X), 0.01, 0.99)

L_DR(τ) = E[(Ỹ_DR − τ(X))²]
```

**Key property:** Doubly robust - consistent if either the outcome model or the propensity model is correctly specified.

**Use for HPO:** Preferred over R-loss in all cases; same computational cost.

**Implementation note:** Ỹ_DR pseudo-outcomes from a pretrained DR-Learner can be reused to tune any other model without recomputing nuisance models.

---

### 2.3 IPW / Transformed Outcome MSE

```
Y* = Y·T/ê(X) − Y·(1−T)/(1−ê(X))
L_IPW(τ) = E[(Y* − τ(X))²]
```

**Pros:** Simplest doubly robust proxy; no outcome model needed.
**Cons:** High variance when ê(X) is close to 0 or 1 (overlap violations).

**Use case:** Quick sanity check; not recommended as primary HPO objective.

---

### 2.4 ERUPT (Expected Response Under Proposed Treatments)

Evaluates the policy derived from τ̂(x) rather than τ̂(x) itself:
```
ERUPT(π) = E[Y · 1{π(X)=T} · (T/ê(X) + (1−T)/(1−ê(X)))]
```
where π(X) = 1{τ̂(X) > threshold}.

**When to use:** When the ultimate goal is a binary targeting decision (treat / don't treat), not CATE accuracy. Directly optimizes policy value.

**Critique:** Ignores ranking quality - two models with the same decision threshold but different score distributions look identical.

---

## 3. Calibration & Uncertainty Metrics

### 3.1 CausalForest Confidence Intervals

`econml.grf.CausalForest` produces pointwise CIs via the **bootstrap-of-little-bags** (a close cousin of the Wager-Athey infinitesimal jackknife). Note the API: the grf class uses `predict(..., interval=True)` and returns a 3-tuple, whereas the DML estimators (`CausalForestDML`) expose `effect_interval(...)` returning `(lb, ub)`:
```python
# econml.grf.CausalForest (needs inference=True):
tau_hat, tau_lb, tau_ub = cf.predict(X_test, interval=True, alpha=0.05)

# econml.dml.CausalForestDML:
lb, ub = cf_dml.effect_interval(X_test, alpha=0.05)
```

**Derived segments:**
- **Persuadables:** τ̂(x) > 0 and τ_lb > 0 (confident positive effect)
- **Sleeping dogs:** τ̂(x) < 0 and τ_ub < 0 (confident negative effect)
- **Uncertain:** CI crosses zero

**Criteo finding (this repo, notebook 02, CausalForest at alpha=0.10):** ~10.4% confident persuadables, ~0.006% confident sleeping dogs, ~89.6% uncertain (CI crosses 0). An external comparison (arXiv:2604.06123) reports far fewer confident persuadables (~1.9% at 95%); the gap is down to CI level, forest tuning, and sample, so quote the repo number for repo claims.

---

### 3.2 TMLE Bootstrap CIs

TMLE (Targeted Maximum Likelihood) produces valid asymptotic CIs for ATE and ATT. For CATE CIs use nonparametric bootstrap (200-500 iterations).

---

### 3.3 Conformal Prediction Intervals (individual treatment effects)

The CIs above quantify uncertainty of the CATE *function* `tau(x)` (an estimation target). Conformal prediction instead builds intervals for the *realization* `Y(1) - Y(0)` of a new unit, with a finite-sample, distribution-free, marginal guarantee. Two routes:

- **Counterfactual route** (Lei and Candes 2021): weighted split conformal on each arm; the weights are inverse-propensity and *collapse to plain split conformal under a constant randomized propensity*.
- **Pseudo-outcome route** (Alaa et al. 2023, "conformal meta-learners"): split conformal on DR pseudo-outcomes. Guarantees pseudo-outcome coverage always; ITE coverage follows for DR/IPW pseudo-outcomes by stochastic dominance. No weighting needed (the pseudo-outcome is computable for every unit).

Split-conformal quantile with the finite-sample correction:
```python
# scores V_i = |pseudo_i - tau_hat(x_i)| on a held-out calibration fold of size n
k = int(np.ceil((n + 1) * (1 - alpha)))
q_hat = np.sort(scores)[k - 1]          # interval = tau_hat(x) +/- q_hat
```

**Criteo caveat (notebook 07):** on a rare binary outcome the guarantee is real but nearly vacuous - `P(ITE = 0) >= 1 - p1 - p0 ~ 99.5%`, so `{0}` is already a valid 95% set (Zhang and Richardson 2025). Marginal ITE coverage can sit exactly on nominal while coverage on the affected units is zero. Report pseudo-outcome coverage (verifiable) or group-level statements instead.

---

## 4. Practical Metrics Hierarchy

```
Business decision stage          → Uplift@k, ERUPT
Model selection / HPO            → DR-loss (fast), then Qini on hold-out
Theoretical comparison           → AUUC, Qini coefficient
Uncertainty of tau(x) / ATE      → CausalForest CIs, TMLE CIs
Uncertainty of individual effect → conformal intervals (nb 07; caveat on rare binary outcomes)
Nuisance model quality check     → propensity calibration (Brier score), outcome AUC
```

---

## 5. Common Pitfalls

| Pitfall | Description | Fix |
|---------|-------------|-----|
| **Overlap violation** | ê(X) near 0 or 1 → IPW explodes | Clip propensity; check overlap via histograms |
| **SUTVA violation** | Treatment of one unit affects another | Flag network effects; consider spatial models |
| **Leaky nuisance** | Nuisance fitted on same fold as CATE | Always use cross-fitting (separate folds) |
| **Qini vs AUUC confusion** | Different normalization → not directly comparable | Report both; use Qini for budget-fixed scenarios |
| **Optimizing rank, evaluating calibration** | Qini-optimal model may have biased τ̂ scale | Add calibration check if absolute effects matter |

---

## 6. Evaluation Code Template

```python
from sklift.metrics import uplift_auc_score, qini_auc_score, uplift_at_k

results = {}
for name, scores in model_scores.items():
    results[name] = {
        'AUUC':       uplift_auc_score(y_test, scores, treat_test),
        'Qini':       qini_auc_score(y_test, scores, treat_test),
        'Uplift@10%': uplift_at_k(y_test, scores, treat_test, strategy='by_group', k=0.1),
        'Uplift@20%': uplift_at_k(y_test, scores, treat_test, strategy='by_group', k=0.2),
    }

import pandas as pd
pd.DataFrame(results).T.sort_values('Qini', ascending=False)
```
