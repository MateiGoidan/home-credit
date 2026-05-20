# Home Credit – Credit Risk Model Stability (Milestone 2)

**Matei-Constantin Goidan** (matei.goidan@stud.acs.upb.ro)  
**Giulia Ștefania Imbrea** (giulia.imbrea@stud.acs.upb.ro)

## 1. Introduction and Milestone 1 Recap

For this competition the task is binary classification of loan default, scored by a **Gini stability metric** that combines mean weekly Gini, a penalty on negative time-trend, and a penalty on residual variance around that trend. The data spans 92 weeks (`WEEK_NUM` 0–91), with 1,526,659 applicants spread across 17 tables organized hierarchically (depth 0, 1, 2), totalling ≈26 GB uncompressed. Class imbalance is 30.8:1 in favor of non-defaulters [1].

In Milestone 1 we submitted two LightGBM [2] solutions:

- **Submission 1** — depth-0 only (`train_base` + `train_static_0` + `train_static_cb_0`, 224 features). Kaggle: public **0.4864**, private **0.3995**.
- **Submission 2** — depth-0 + aggregates from 4 depth-1 tables + a 10% stability filter on the coefficient of variation of weekly feature means (673 features). Kaggle: public **0.4879**, private **0.3892**.

Submission 2 improved every validation metric (val AUC 0.8474 vs 0.8188, val Gini-stab 0.6618 vs 0.5975) but **regressed on the Kaggle private leaderboard**, confirming the gap between our validation horizon (weeks 61–91) and the hidden test horizon (~week 100+).

## 2. Milestone 2 Experiments

We addressed the four improvement directions proposed at the end of M1: a second model family, model ensembling, walk-forward cross-validation, and metric-aligned hyperparameter optimisation.

### 2.1 Submission 3 — CatBoost on the same feature set

We trained CatBoost [3] on the same 673-feature matrix used by Submission 2. CatBoost handles categorical columns natively (without our integer-encoding step) and tends to produce better-calibrated probabilities, which is relevant since the metric is AUC-based and benefits from well-ordered scores. Categorical columns were passed via `cat_features`; the same `WEEK_NUM <= 60` time-based split was used.

**Validation:** AUC 0.8539, Gini-stab 0.6714 (vs LightGBM Submission 2's 0.8474 / 0.6618 — CatBoost is slightly stronger locally). **Kaggle:** public **0.4781**, private **0.3903**. Comparable to Submission 2 on private (0.3892) but worse on public — useful primarily because the errors are different from LightGBM's, which sets up the ensemble.

### 2.2 Submission 4 — LightGBM + CatBoost 50/50 ensemble

Naive blend `0.5 * lgb_probs + 0.5 * cat_probs` on the test predictions, using both models with their Submission 2 / Submission 3 settings. Individually LightGBM scored val Gini-stab 0.6618 (AUC 0.8474) and CatBoost 0.6714 (AUC 0.8539); the ensemble achieved val Gini-stab **0.6703** (AUC 0.8515, residual std 0.0412). On Kaggle the ensemble scored public **0.4904**, private **0.4142** — the best score of all our submissions, and the only one to exceed M1's best private score. The mechanism is straightforward: the two model families have similar mean per-week Gini but make different errors, so averaging their probabilities keeps the mean roughly stable while reducing the residual variance around the weekly trend — exactly the term the stability metric penalises.

### 2.3 Submission 5 — Improved LightGBM (walk-forward CV + custom Gini-stability eval + Optuna)

Three changes layered on top of Submission 2's pipeline, applied only to the LightGBM training procedure (features unchanged):

1. **Walk-forward cross-validation** — replaced the single (0–60)/(61–91) split with three expanding-window folds: (0–40)/(41–60), (0–60)/(61–75), (0–75)/(76–91). This produces a more honest stability estimate than one fold and is necessary to tune hyperparameters without overfitting to one validation window.
2. **Custom `feval`** — instead of optimising AUC (the LightGBM default), early stopping is driven by a custom evaluation function computing the actual Gini stability score on the validation fold's weekly buckets. The function needs `WEEK_NUM` per row, which is not part of LightGBM's `eval_metric` signature; we pass it via a module-level variable set before each fold's `fit` call.
3. **Optuna** [4] hyperparameter search (10 trials) optimising mean Gini stability across the three walk-forward folds. The search space covered `learning_rate` (log 0.01–0.1), `num_leaves` (31–127), `min_child_samples` (20–200), `feature_fraction` (0.5–1.0), `bagging_fraction` (0.5–1.0), `reg_alpha` (0–1), `reg_lambda` (0–1).

**Local CV scores:**
- Baseline params, walk-forward Gini-stab: **0.6401**
- Optuna-tuned params: **0.6541** (+0.014)

**Best params found:** `learning_rate=0.0228`, `num_leaves=55`, `min_child_samples=38`, `feature_fraction=0.775`, `bagging_fraction=0.519`, `reg_alpha=0.793`, `reg_lambda=0.378`.

**Kaggle:** public **0.4859**, private **0.3890** — essentially unchanged from Submission 2 despite the local improvement.

### 2.4 Submission 6 — Improved ensemble (Optuna-tuned LightGBM + CatBoost)

Same ensemble structure as Submission 4 but with LightGBM replaced by the Submission 5 version (Optuna-tuned, custom-feval-driven). CatBoost untouched.

**Validation (weeks 61–91):**

| Model | AUC | Gini-stab | Residual std |
|---|---|---|---|
| LightGBM (Optuna-tuned) alone | 0.8499 | 0.6703 | 0.0390 |
| CatBoost alone | 0.8489 | 0.6634 | 0.0414 |
| Ensemble 50/50 | **0.8516** | **0.6710** | 0.0404 |

The Optuna-tuned LightGBM individually outperforms Submission 2's LightGBM on local validation (0.6703 vs 0.6618), confirming the tuning did improve in-sample performance. The ensemble barely beats LightGBM alone (+0.0007) because the two models are now closer in behaviour — the tuned LightGBM uses heavy regularisation that pushes it toward CatBoost-like predictions, reducing the diversity that made Submission 4's ensemble effective.

**Kaggle:** public **0.4851**, private **0.4029** — worse than Submission 4 (0.4904 / 0.4142) on both. The locally better LightGBM produced a worse ensemble on the hidden test horizon.

## 3. Results Summary

| Submission | Approach | Features | Val Gini-stab | Public | Private |
|---|---|---|---|---|---|
| 1 | LightGBM, depth-0 only | 224 | 0.5975 | 0.4864 | **0.3995** |
| 2 | LightGBM + depth-1 + stability filter | 673 | 0.6618 | 0.4879 | 0.3892 |
| 3 | CatBoost, same features as Submission 2 | 673 | 0.6714 | 0.4781 | 0.3903 |
| 4 | LightGBM + CatBoost ensemble (50/50) | 673 | 0.6703 | **0.4904** | **0.4142** |
| 5 | Submission 2 LightGBM + walk-forward CV + Optuna | 673 | 0.6541 (walk-forward mean) | 0.4859 | 0.3890 |
| 6 | Submission 5 LightGBM + CatBoost ensemble | 673 | 0.6710 | 0.4851 | 0.4029 |

## 4. Discussion

**Ensembling worked, hyperparameter tuning did not transfer.** Submission 4 is our only submission that exceeds M1's best private score (0.4142 vs 0.3995). The ensemble's advantage is mechanistic: averaging two model families with similar mean Gini but different error patterns reduces per-week residual variance directly, which is precisely the term the stability metric penalises. Submission 5's Optuna tuning showed real improvement on the walk-forward CV (0.6401 → 0.6541) but failed to translate to either Kaggle leaderboard. This repeats the M1 Submission 1 vs Submission 2 pattern at a finer scale: signal that exists in weeks 0–91 does not always exist in the hidden test horizon ≈100+. We hypothesise the Optuna search overfit to drift patterns specific to the training weeks, particularly via aggressive regularisation values (`reg_alpha = 0.79`) that suppressed signal which the default-params LightGBM was using productively.

**The hidden test horizon dominates everything.** Across six submissions, every method we tried produced a local validation score that correlates only weakly with Kaggle private. Walk-forward CV gave a better estimate than the single split (and let us tune Optuna at all), but it still uses only weeks ≤91 — the metric on the private weeks is essentially a different distribution. Methods that helped: ensembling (different model families → more robust mean predictions). Methods that hurt or were neutral: more features, hyperparameter tuning, custom evaluation metrics. The conclusion mirrors the competition's own framing — stability is hardest to estimate from in-sample data.

**Engineering challenges.** Kaggle's 30 GB notebook RAM cap was the binding constraint throughout M2. The Submission 2 pipeline alone uses ≈15 GB at peak (1.5M × 673 × float32 ≈ 4 GB per matrix, plus LightGBM histogram overhead). Running Optuna on Kaggle exhausted memory after one fold because the search ran on top of an already-loaded feature matrix; we worked around this by running Optuna locally on a 64 GB M2 Max and hardcoding the best params into the Kaggle notebook. We also encountered Kaggle's hidden test-set substitution mechanism the hard way: an early ensemble attempt that loaded pre-saved test predictions from a `.npy` file failed because Kaggle re-runs the notebook with a larger hidden test set at submission time, making pre-computed predictions the wrong length. The fix was to retrain LightGBM during the notebook run with the Optuna-tuned hyperparameters baked in.

## 5. Resources, Third-Party Code, and LLM Assistance

**Libraries:** Polars (data loading/joining/aggregation under memory pressure), LightGBM [2], CatBoost [3], Optuna [4], scikit-learn (`roc_auc_score`), scipy (`linregress`), matplotlib.

**Kaggle community references:** competition overview page for the exact stability metric formula and weekly bucketing rule [1]; competition forum discussions on memory management on the 30 GB notebook tier; competition forum discussions on the hidden test-set substitution at submission time, which informed our re-architecting of the ensemble notebook.

**LLM assistance.** We used Claude (Anthropic) as a coding assistant throughout M2. Concrete uses: (a) drafting the walk-forward CV loop and the closure pattern for passing `WEEK_NUM` into LightGBM's custom `eval_metric`; (b) drafting the Optuna `objective` function and parameter ranges; (c) debugging Polars schema mismatches when concatenating depth-1 parquet shards with mixed null-only columns (the `diagonal_relaxed` + per-shard cast pattern); (d) writing the patch to the ensemble notebook that loads pre-computed LightGBM test predictions (later reverted in favour of in-notebook retraining once we understood Kaggle's hidden-test-set behaviour). All generated code was read line-by-line, tested locally, and modified before being committed. The assistant was also used for explanatory questions ("what does Kaggle's submission re-run do?", "why might Optuna's best params not transfer?") which directly shaped Section 4. All experimental results, methodology choices, and final interpretations are our own.

**Code repository:** https://github.com/MateiGoidan/home-credit (branches: `main` for the LightGBM-side work in this milestone, `develop_catboost` for the CatBoost and ensemble work).

## References

[1] Home Credit. *Home Credit – Credit Risk Model Stability.* Kaggle Competition. Accessed: 2026-05-20. https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability

[2] G. Ke et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS*, vol. 30, 2017.

[3] L. Prokhorenkova et al. "CatBoost: Unbiased Boosting with Categorical Features." *NeurIPS*, vol. 31, 2018.

[4] T. Akiba et al. "Optuna: A Next-generation Hyperparameter Optimization Framework." *KDD*, 2019.

[5] N. Siddiqi. *Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards.* 2nd ed., Wiley, 2017.
