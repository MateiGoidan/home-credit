# LightGBM Improvement Tasks — Home Credit Project

## Context

We've built 4 submissions so far:
- **Sub 1:** LightGBM on depth-0 only (baseline)
- **Sub 2:** LightGBM on depth-0 + 4 depth-1 aggregations + stability filter
- **Sub 3:** CatBoost on the same features as Sub 2
- **Sub 4:** 50/50 ensemble of LightGBM + CatBoost (in `notebooks/05_ensemble_kaggle.ipynb`)

The ensemble beats both individual models on local validation. But the LightGBM half of the ensemble is still using the *baseline* training setup (single time-based split, AUC as eval metric, default-ish hyperparameters). If we improve LightGBM, the ensemble improves too — likely +0.005 to +0.01 on the Kaggle score.

Your three tasks below, starting from `notebooks/03_lightGBM_depth1_stability_local.ipynb`.

---

## Task 1 — Walk-forward Cross-Validation

**Why:** the current code uses a single split (train weeks 0–60, validate 61–91). One fold = one noisy data point. Walk-forward folds give a more honest estimate of model stability and let us tune hyperparameters reliably.

**What to implement:**

```python
# Replace the single-split block in cell 11/15 with this loop
folds = [
    (slice(0, 40),   slice(41, 60)),   # fold 1
    (slice(0, 60),   slice(61, 75)),   # fold 2
    (slice(0, 75),   slice(76, 91)),   # fold 3
]

oof_preds = np.zeros(len(y))
scores = []

for i, (train_range, val_range) in enumerate(folds):
    train_mask = (week_num >= train_range.start) & (week_num <= train_range.stop)
    val_mask   = (week_num >= val_range.start)   & (week_num <= val_range.stop)

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_va, y_va = X[val_mask],   y[val_mask]
    week_va    = week_num[val_mask]

    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False)])

    preds = model.predict_proba(X_va)[:, 1]
    oof_preds[val_mask] = preds
    score, *_ = gini_stability(week_va, y_va, preds)
    print(f'Fold {i+1}: Gini-stab = {score:.4f}')
    scores.append(score)

print(f'Mean Gini-stab across folds: {np.mean(scores):.4f}')
```

**Output:** print mean Gini stability across folds; save `oof_preds` for potential stacking later.

---

## Task 2 — Custom `feval` for Gini Stability

**Why:** LightGBM's default eval metric is AUC. We're not scored on AUC — we're scored on Gini stability (mean weekly Gini + trend bonus − residual std penalty). Optimizing for AUC and hoping stability follows is suboptimal. We can pass a custom function and early-stop on that instead.

**What to implement:**

```python
def gini_stability_feval(preds, dataset):
    """LightGBM custom eval. Signature: (preds, Dataset) → (name, value, higher_is_better)"""
    y_true = dataset.get_label()
    # Need WEEK_NUM here — store it in a module-level variable or via closure
    week_nums = current_week_nums  # set this before each fold
    score, *_ = gini_stability(week_nums, y_true, preds)
    return 'gini_stability', score, True  # True = higher is better

# Then in fit():
model.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    eval_metric=gini_stability_feval,
    callbacks=[lgb.early_stopping(50, verbose=False)]
)
```

**Tricky bit:** the eval function only gets `y_true` and `preds`, not `week_num`. Easiest workaround is a closure or a module-level variable that you set to the current fold's validation `week_num` before calling `fit()`.

---

## Task 3 — Optuna Hyperparameter Tuning

**Why:** the current params (`num_leaves=63`, `learning_rate=0.05`, `min_child_samples=50`, etc.) are reasonable defaults but never tuned. Optuna can search the space and optimize on the Gini stability metric directly.

**What to implement:**

```python
import optuna

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',  # internal LightGBM metric; we override with feval
        'n_estimators': 1000,
        'learning_rate':      trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves':         trial.suggest_int('num_leaves', 31, 127),
        'min_child_samples':  trial.suggest_int('min_child_samples', 20, 200),
        'feature_fraction':   trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction':   trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq':       1,
        'reg_alpha':          trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda':         trial.suggest_float('reg_lambda', 0.0, 1.0),
        'scale_pos_weight':   (y == 0).sum() / (y == 1).sum(),
        'n_jobs': -1, 'verbose': -1, 'random_state': 42,
    }
    # Run walk-forward CV from Task 1 with these params
    # Return mean Gini stability across folds
    return mean_gini_stab_across_folds(params)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)  # 30 trials is enough to see signal
print('Best params:', study.best_params)
print('Best score:', study.best_value)
```

30 trials with 3-fold walk-forward CV on this data will take several hours. Run overnight or on Kaggle.

---

## Final Deliverable

A notebook `06_lightGBM_improved.ipynb` that:
1. Uses the **same features** as the existing LGB notebook (depth-0 + 4 depth-1 + stability filter — no changes needed there)
2. Implements walk-forward CV (Task 1)
3. Uses custom `feval` for Gini stability (Task 2)
4. Has the best-found Optuna params hard-coded (Task 3 — don't rerun Optuna in the final notebook, just paste the best params)
5. Outputs `lgbm_test_preds.npy` (test set probabilities, shape = number of test rows)

Then update `05_ensemble_kaggle.ipynb` — replace the LightGBM training block (cells 16–17) with a `np.load('lgbm_test_preds.npy')` call so the ensemble uses your improved predictions instead of training LightGBM from scratch. Submit the updated ensemble as Sub 5.

---

## Time Estimate

| Task | Effort |
|---|---|
| Task 1 (walk-forward CV) | ~2 hours |
| Task 2 (custom feval, closure trick) | ~1 hour |
| Task 3 (Optuna script + several hours of compute) | ~1 hour write + overnight run |
| Final notebook + cleanup | ~1 hour |

**Total:** roughly half a day of focused work + Optuna compute time.

---

Let me know if anything is unclear.