# Presentation Speech Script — Milestone 2

Target: 7 minutes total, ~1 min per slide.

---

## Slide 1 — Title (~20s)

"Hello, we are Giulia Imbrea and Matei Goidan. Today we present our Milestone 2 work on the Home Credit Credit Risk Model Stability competition. In M1 we had two LightGBM submissions; in M2 we added four more and we'll walk through what worked, what did not, and why."

---

## Slide 2 — Recap M1 (~50s)

"Quick recap. The competition scores models on a Gini stability metric that combines the mean weekly Gini, a penalty on negative time trends, and a penalty on residual variance — so consistency matters as much as accuracy. The dataset is 1.5 million applicants over 92 weeks, 26 gigabytes uncompressed.

In M1 we submitted two LightGBM solutions. Sub 1 used only the depth-0 join — 224 features — and scored 0.40 private. Sub 2 added aggregates from four depth-1 tables and a 10% stability filter — 673 features — improved every local metric but actually *regressed* on Kaggle private to 0.39.

That gap between local validation and the hidden test horizon is the central problem of this competition, and it sets up our four M2 directions: a second model family, ensembling, walk-forward cross-validation, and hyperparameter tuning."

---

## Slide 3 — Submissions 3 & 4 (CatBoost + Ensemble) (~75s)

"Submission 3 is CatBoost trained on the same 673 features as Sub 2. CatBoost handles categorical columns natively and produces better-calibrated probabilities. It scored val Gini-stab 0.67 — slightly better than LightGBM locally — and 0.39 on Kaggle private, comparable to Sub 2. Important on its own, but mostly useful because its errors are different from LightGBM's.

Submission 4 is the obvious next step: a 50/50 blend of the two test probabilities. The numbers tell the whole story. Individually LightGBM and CatBoost scored around 0.66–0.67 on validation. The ensemble scored 0.67 on validation but on *Kaggle* it scored **0.4904 public, 0.4142 private** — our best score, and the only submission in either milestone to exceed M1's best private score.

Why does this work? Two model families with similar mean Gini but different errors. Averaging keeps the mean roughly the same but reduces residual variance — exactly what the stability metric penalizes. It's a mechanistic improvement, not a lucky one."

---

## Slide 4 — Submission 5 (Improved LightGBM) (~75s)

"Submission 5 attacks the LightGBM training procedure itself, with three changes — features unchanged.

First, walk-forward cross-validation. Three expanding-window folds instead of one split: 0–40 train and 41–60 val, then 0–60 and 61–75, then 0–75 and 76–91. This gives an honest stability estimate across multiple time horizons.

Second, a custom evaluation function. The default LightGBM metric is AUC, but we're scored on Gini stability — these are not the same. The trick was that LightGBM's eval_metric signature only gets predictions and labels, not WEEK_NUM. We pass it through a module-level variable set before each fold's fit call.

Third, Optuna — 10 trials searching learning rate, num_leaves, regularization, sampling ratios, optimizing mean walk-forward Gini stability.

Result: baseline walk-forward Gini-stab 0.6401, Optuna-tuned 0.6541 — a clear 0.014 improvement locally. On Kaggle? 0.4859 public, 0.3890 private. Essentially flat versus Sub 2. The tuning worked locally but did not transfer."

---

## Slide 5 — Submission 6 (Improved Ensemble) (~70s)

"Submission 6 is the natural follow-up: take the Sub 5 Optuna-tuned LightGBM and put it into the Sub 4 ensemble structure.

On local validation it actually works — 0.6710 versus 0.6703 for Sub 4, the tuned LightGBM individually beats its untuned version 0.6703 versus 0.6618.

But on Kaggle: 0.4851 public, 0.4029 private. Worse than Sub 4 on both.

Why? The Optuna search pushed LightGBM into a heavily regularized regime — reg_alpha 0.79, low feature fraction — that produces predictions closer to CatBoost's. The ensemble loses diversity. Improvements in either component of an ensemble do not automatically improve the ensemble; what matters is how the components disagree."

---

## Slide 6 — Results & Discussion (~80s)

"Across six submissions: Sub 4 — the ensemble — is our best, and the only one that exceeds M1's best private score. We get +0.025 public and +0.015 private over Sub 1.

Two clear lessons. First, ensembling works and works for a principled reason: variance reduction targets exactly what the metric penalizes. Second, hyperparameter tuning did not transfer. Both Sub 5 alone and Sub 6 as ensemble component improved local CV but lost on Kaggle. The Optuna search appears to have overfitted to drift patterns in weeks 0–91 that do not exist in the hidden test horizon around week 100 and beyond.

This is the same pattern we saw between Sub 1 and Sub 2 in M1: more local signal does not automatically mean more out-of-sample signal when the validation distribution differs from the test distribution."

---

## Slide 7 — Qualitative & Limitations (~70s)

"Two practical lessons we hit hard. Kaggle has a 30 gigabyte notebook memory cap, so Optuna had to run locally on a 64 gigabyte MacBook, then we hardcoded the best parameters into the Kaggle notebook. And Kaggle re-runs notebooks with a substituted hidden test set at submission time, which means you cannot use pre-saved predictions — we discovered this when an attempt to load a `.npy` file produced predictions of the wrong length.

For future work: drop credit_bureau_a — the most volatile depth-1 table; try a more aggressive 20–30% stability filter; add a third model family rather than tuning the existing ones.

On LLM assistance: we used Claude for walk-forward CV scaffolding, Optuna setup, Polars debugging, and explanatory questions about Kaggle's submission mechanism. All code was read line-by-line and tested before being committed. The full code is on GitHub.

That's our work. Thank you — happy to take questions."
