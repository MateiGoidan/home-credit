# Progress — Milestone 2

## Rulat local (M2 Max, 64GB)

Notebook: `06_lightGBM_improved.ipynb`

- Walk-forward CV (3 folduri: 0–40/41–60, 0–60/61–75, 0–75/76–91)
- Custom Gini stability feval (early stopping pe metrica reală, nu AUC)
- Optuna tuning — 10 trials

| | Score |
|---|---|
| Baseline walk-forward CV | 0.6401 |
| Optuna best walk-forward CV | **0.6541** |

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

---

## Rulat pe Kaggle

Notebook executat: `notebooks/06_lightGBM_improved_executed.ipynb`  
Link Kaggle: https://www.kaggle.com/code/giuliastefaniaimbrea/06-improved-optuna-with-best-params

Same features ca Sub 2 (depth-0 + 4 depth-1 + stability filter, 673 features).  
Best params din Optuna hardcodați direct.

**Validare pe ultimul fold (weeks 76–91):**

| Metric | Valoare |
|--------|---------|
| Validation AUC | 0.8603 |
| Gini stability score | 0.7007 |
| Mean weekly Gini | 0.7157 |
| Trend slope | +0.002398 |
| Residual std | 0.0299 |

**Kaggle leaderboard — Sub 5:**
- Public score: **0.48590**
- Private score: **0.38895**

Screenshot: `submissions/scor-kaggle.png`  
Fisiere: `submissions/submission_05_lgbm_improved.csv`, `submissions/lgbm_test_preds.npy`

---

## Sub 6 — Improved Ensemble

Notebook: `notebooks/06_ensemble_improved_kaggle.ipynb` (copie din `05_ensemble_kaggle.ipynb` al colegului, fără modificări la original)  
Link Kaggle: https://www.kaggle.com/code/giuliastefaniaimbrea/06-ensemble-improved-kaggle

**Diferența față de Sub 4:** LightGBM-ul din ensemble e antrenat cu best params din Optuna (Sub 5), nu cu defaultul colegului. CatBoost rămâne identic.

**Rezultat:**
- Public score: **0.48513**
- Private score: **0.40289**

Sub 6 e mai slab decât Sub 4 (0.4904 / 0.4142). Params Optuna optimizați pe walk-forward CV local au îmbunătățit scorul de validare (0.6401 → 0.6541), dar nu au generalizat pe test set-ul ascuns — același pattern ca Sub 2 vs Sub 1.

**Concluzie:** Sub 4 rămâne cea mai bună submisie.

---

## Toate submisiile

| Sub | Model | Public | Private |
|-----|-------|--------|---------|
| Sub 1 | LightGBM depth-0 | 0.4864 | 0.3995 |
| Sub 2 | LightGBM depth-1 + stability filter | 0.4879 | 0.3892 |
| Sub 3 | CatBoost depth-1 (coleg) | 0.4781 | 0.3903 |
| Sub 4 | Ensemble LightGBM + CatBoost (coleg) | **0.4904** | **0.4142** |
| Sub 5 | LightGBM improved (Optuna + walk-forward CV) | 0.4859 | 0.3890 |
| Sub 6 | Improved LightGBM + CatBoost Ensemble | 0.48513 | 0.40289 |

**Best submission: Sub 4** (0.4904 public / 0.4142 private).
