---
marp: true
theme: default
paginate: true
style: |
  section {
    font-size: 21px;
    padding: 40px;
  }
  h1 { color: #1a237e; font-size: 36px; }
  h2 { color: #283593; font-size: 28px; }
  table { font-size: 17px; width: 100%; }
  th { background-color: #e8eaf6; }
  ul { margin-top: 8px; }
  li { margin-bottom: 4px; }
---

# Home Credit — Credit Risk Model Stability

**Milestone 2**

Imbrea Giulia · Goidan Matei

---

## Recap — Milestone 1

**Task:** binary loan-default classification, scored by a **Gini stability metric** (mean weekly Gini + slope penalty + residual-variance penalty)

**Dataset:** 1.5M applicants, 17 tables, 26 GB, 92 weeks → time-based CV mandatory

**M1 submissions (LightGBM):**

| Sub | Approach | Public | Private |
|---|---|---|---|
| 1 | Depth-0 only (224 features) | 0.4864 | **0.3995** |
| 2 | Depth-1 aggregates + 10% stability filter (673 features) | 0.4879 | 0.3892 |

**Key finding from M1:** Sub 2 improved every local metric but regressed on Kaggle private — our validation horizon (weeks 61–91) does not predict the hidden test horizon (≈100+). Set up the four directions for M2: CatBoost, ensembling, walk-forward CV, hyperparameter tuning.

---

## M2 — Submissions 3 & 4 (CatBoost + Ensemble)

**Submission 3 — CatBoost** on the same 673 features as Sub 2. Native categorical handling, better-calibrated probabilities.
- Val Gini-stab **0.6714** (vs LGB 0.6618) — slightly better locally
- Kaggle: 0.4781 / 0.3903 — comparable to Sub 2 on private

**Submission 4 — LightGBM + CatBoost 50/50 ensemble**

| Model | Val AUC | Val Gini-stab | Residual std |
|---|---|---|---|
| LightGBM alone | 0.8474 | 0.6618 | 0.0405 |
| CatBoost alone | 0.8539 | 0.6714 | 0.0406 |
| **Ensemble 50/50** | **0.8515** | **0.6703** | **0.0412** |

**Kaggle: public 0.4904 / private 0.4142** — best score, only one exceeding M1.

**Why it works:** different model families → similar mean Gini, *different errors* → averaging reduces residual variance → directly attacks the stability penalty.

---

## M2 — Submission 5 (Improved LightGBM)

Three procedural improvements on Sub 2's pipeline (features unchanged):

**1. Walk-forward CV** — 3 expanding-window folds: (0–40)/(41–60), (0–60)/(61–75), (0–75)/(76–91). Honest stability estimate; required to tune without overfitting one window.

**2. Custom `feval`** — early-stop directly on Gini stability instead of AUC. Trick: pass `WEEK_NUM` via a closure since LightGBM's eval signature only gets `(y_true, y_pred)`.

**3. Optuna search** (10 trials) over learning rate, num_leaves, min_child_samples, regularization, sampling ratios — objective is mean walk-forward Gini stability.

| | Walk-forward Gini-stab |
|---|---|
| Baseline params | 0.6401 |
| Optuna-tuned | **0.6541** (+0.014) |

**Best params:** lr=0.0228, num_leaves=55, min_child_samples=38, reg_alpha=0.79, reg_lambda=0.38

**Kaggle: public 0.4859 / private 0.3890** — essentially unchanged from Sub 2.

---

## M2 — Submission 6 (Improved Ensemble)

Same blend as Sub 4 but with the Optuna-tuned LightGBM from Sub 5. CatBoost untouched.

| Model | Val AUC | Val Gini-stab | Residual std |
|---|---|---|---|
| LightGBM (Optuna) | 0.8499 | 0.6703 | 0.0390 |
| CatBoost | 0.8489 | 0.6634 | 0.0414 |
| **Ensemble 50/50** | **0.8516** | **0.6710** | 0.0404 |

**Locally better** than Sub 4 (0.6710 vs 0.6703) — Optuna-tuned LightGBM individually outperforms its untuned version (0.6703 vs 0.6618).

**Kaggle: public 0.4851 / private 0.4029** — **worse** than Sub 4.

**Mechanism:** the tuned LightGBM uses heavy regularization that pushes it closer to CatBoost-like predictions → ensemble loses diversity → loses the variance-reduction advantage on the hidden test set.

---

## Results Summary & Discussion

| # | Approach | Val Gini-stab | Public | Private |
|---|---|---|---|---|
| 1 | LightGBM depth-0 | 0.5975 | 0.4864 | 0.3995 |
| 2 | LightGBM depth-1 + stability filter | 0.6618 | 0.4879 | 0.3892 |
| 3 | CatBoost (same features as Sub 2) | 0.6714 | 0.4781 | 0.3903 |
| **4** | **LightGBM + CatBoost ensemble** | **0.6703** | **0.4904** | **0.4142** |
| 5 | LightGBM improved (Optuna + walk-forward CV) | 0.6541 (WF mean) | 0.4859 | 0.3890 |
| 6 | Improved ensemble (Sub 5 LGB + CatBoost) | 0.6710 | 0.4851 | 0.4029 |

**What worked:** ensembling two different model families → mechanistic variance reduction → +0.025 public, +0.025 private over M1 best.

**What didn't transfer:** hyperparameter tuning. Sub 5 improved local CV by 0.014; both Sub 5 (alone) and Sub 6 (ensemble) lost on Kaggle. The Optuna search overfitted to drift patterns in weeks 0–91 that do not exist in the hidden test horizon.

---

## Qualitative Findings & Limitations

**The hidden test horizon dominates everything.** Across all 6 submissions, local validation Gini-stab correlates only weakly with Kaggle private. Walk-forward CV is better than a single split — but it still uses only weeks ≤91. The metric on hidden weeks ≈100+ is essentially a different distribution.

**Engineering challenges encountered:**
- Kaggle 30 GB notebook RAM → ran Optuna locally on M2 Max instead
- Kaggle re-runs notebooks with a *substituted hidden test set* at submission → cannot use pre-computed `.npy` predictions; must retrain in-notebook with tuned params

**Future directions:**
- Drop `credit_bureau_a_1` entirely (most volatile depth-1 table by drift)
- More aggressive stability filtering (20–30% instead of 10%)
- Diversify ensemble with a third model family (XGBoost, neural net) instead of tuning what we have

**LLM assistance:** Claude used for walk-forward CV scaffolding, Optuna setup, Polars schema debugging, and explanatory questions. All code read and tested before commit.

**Code:** https://github.com/MateiGoidan/home-credit
