# Progress Notes

## Ce am făcut

- Implementat `06_lightGBM_improved.ipynb` cu:
  - Walk-forward CV (3 folduri: 0-40/41-60, 0-60/61-75, 0-75/76-91)
  - Custom Gini stability feval (early stopping pe metrica reală, nu AUC)
  - Optuna tuning (10 trials) — rulat local pe M2 Max

- Creat `07_lightGBM_final_kaggle.ipynb` — versiune Kaggle fără Optuna, cu best params hardcodați

## Rezultate obținute

| | Score |
|---|---|
| Baseline walk-forward CV (local) | 0.6401 |
| Tuned walk-forward CV (Optuna) | 0.6541 (+0.014) |
| Sub 4 ensemble (best Kaggle score) | 0.4904 public / 0.4142 private |

**Best params Optuna:**
```
learning_rate:     0.0228
num_leaves:        55
min_child_samples: 38
feature_fraction:  0.775
bagging_fraction:  0.519
reg_alpha:         0.793
reg_lambda:        0.378
```

## Ce așteptăm

- `07_lightGBM_final_kaggle.ipynb` rulează pe Kaggle (~30-40 min)
- Output: `submission_05_lgbm_improved.csv` + `lgbm_test_preds.npy`

## Ce facem după

1. Descarcă `submission_05_lgbm_improved.csv` din Kaggle Output → submit → notează scorul
2. Actualizează `05_ensemble_kaggle.ipynb` — înlocuiește blocul LightGBM cu `np.load('lgbm_test_preds.npy')` → rulează ensemblul → Sub 5 → submit → notează scorul
3. Compară cu Sub 4 (0.4904 public) — dacă e mai bun, gata
