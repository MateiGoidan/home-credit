# Report & Presentation Notes — Home Credit Credit Risk Model Stability

## Key Numbers to Use Everywhere

| Fact | Value |
|---|---|
| Training set size | 1,526,659 applicants |
| Number of tables | 14 (across depth 0, 1, 2) |
| Uncompressed dataset size | ~26 GB |
| Total weeks of data | 92 (WEEK_NUM 0–91) |
| Default rate | **3.14%** |
| Class imbalance ratio | **30.8:1** (non-default : default) |
| Train/val split | Weeks 0–60 train, 61–91 validate |

---

## EDA Highlights (from 01_eda notebook)

### Target Distribution
| Class | Count | Rate |
|---|---|---|
| No default (0) | 1,478,665 | 96.86% |
| Default (1) | 47,994 | **3.14%** |

- Imbalance ratio: **30.8:1** → handled with `scale_pos_weight` in LightGBM

### Temporal Structure
- Default rate has a **slight downward trend** over time (slope: −0.000039/week, p=0.23 — not statistically significant)
- This means the model must not treat time as a feature and must use **time-based CV**
- Random CV would leak future data and give falsely high validation scores

### Depth-1 Table Sizes (why scale matters)

| Table | Files | Columns | Rows (sample) | Avg rows/case |
|---|---|---|---|---|
| applprev_1 | 2 | 41 | 3,887,684 | 4.97 |
| credit_bureau_a_1 | 4 | 79 | 4,108,212 | 12.25 |
| credit_bureau_b_1 | 1 | 45 | 85,791 | 2.35 |
| person_1 | 1 | 37 | 2,973,991 | 1.95 |

### Missingness
- Base table: **0% missing** (clean identifier table)
- Depth-1 tables: **29–46% missing** per table → LightGBM handles NaN natively, no imputation needed

### Column naming convention
- `_A` — amount (numeric)
- `_D` — date
- `_L` — length/count (numeric)
- `_M` — categorical
- `_P` — percentage/rate (numeric)

---

## Submission 1 — LightGBM Depth-0 Only

**What it uses:** `train_base` + `train_static_0_*` + `train_static_cb_0`  
**Features:** 224 columns after join

### Results

| Metric | Value |
|---|---|
| Validation AUC | **0.8188** |
| **Gini stability score** | **0.5975** |
| Mean weekly Gini | 0.6216 |
| Trend slope | +0.0029 (positive = improving) |
| Residual std | 0.0483 |
| Best iteration | 280 trees |
| Kaggle public score | 0.4864 |
| Kaggle private score | 0.3995 |

### Key design choices
- Time-based CV: weeks 0–60 train, 61–91 validate
- `scale_pos_weight = 30.8` to handle class imbalance
- Early stopping (50 rounds) on validation AUC
- Date columns → days-since-epoch integer
- String columns → categorical integer codes

---

## Submission 2 — LightGBM Depth-0 + Depth-1 + Stability Filtering

**What it uses:** everything from Sub 1 + aggregations from 4 depth-1 tables  
**Features:** 747 before filtering → **673 after stability filter** (74 dropped)

### Aggregation strategy
For each depth-1 table, computed per `case_id`:
- Numeric columns: `mean`, `max`, `min`, `std`
- Categorical columns: `n_unique`

### Stability filter
- For each numeric feature, compute its mean per WEEK_NUM on training weeks
- Measure **coefficient of variation** (std / |mean|) of weekly means
- Drop top **10%** most temporally drifting features

### Top 10 most unstable features (dropped)
| Feature | Drift score |
|---|---|
| mindbdtollast24m_4525191P | 17.57 |
| mindbddpdlast24m_3658935P | 9.61 |
| forweek_1077L | 7.55 |
| cb_b_installmentamount_644A_std | 7.20 |
| cb_b_dpd_733P_max | 7.04 |

### Results

| Metric | Value |
|---|---|
| Validation AUC | **0.8474** |
| **Gini stability score** | **0.6618** |
| Mean weekly Gini | 0.6821 |
| Trend slope | +0.0026 |
| Residual std | **0.0405** (lower = better) |
| Best iteration | 244 trees |
| Kaggle public score | 0.4879 |
| Kaggle private score | 0.3892 |

### Top 5 most important features
| Feature | Importance |
|---|---|
| price_1097A | 262 |
| annuity_780A | 229 |
| pmtnum_254L | 213 |
| pmtssum_45A | 192 |
| validfrom_1069D | 185 |

---

## Comparison Table (put this in report AND slides)

| | Sub 1 (depth-0) | Sub 2 (depth-1 + stability) |
|---|---|---|
| Features | 224 | 673 |
| Tables used | 3 | 7 |
| Val AUC | 0.8188 | **0.8474** |
| Val Gini stability | 0.5975 | **0.6618** |
| Residual std | 0.0483 | **0.0405** |
| Kaggle public | 0.4864 | 0.4879 |
| Kaggle private | **0.3995** | 0.3892 |

---

## Discussion Points (critical for the report)

### Why does local validation not match Kaggle private score?
- Local val covers weeks 61–91 (still within training distribution)
- Kaggle private test covers **week 100+** (further in the future)
- Model degrades on data further from training period → temporal instability

### Why did Sub 2 private score go down despite better local validation?
- Adding more features from depth-1 tables introduced **temporally unstable patterns**
- Even with 10% drift filtering, remaining aggregates (especially credit_bureau_a) still drift on the private test weeks
- This is the core challenge of the competition — more features ≠ better stability

### How we handled the 26 GB dataset (scale discussion for report)
1. **Parquet over CSV** — 4–5x faster reads, column-pruning
2. **Polars instead of pandas** — lazy evaluation, 2x less memory than pandas for same operation
3. **Scope discipline** — used 4 out of 10+ depth-1 tables; tried to use all → OOM on Kaggle
4. **float32 instead of float64** — halves memory for feature matrices
5. **Explicit memory management** — `del` + `gc.collect()` after every join
6. **Hit real OOM on Kaggle** (30 GB notebook RAM) and solved it iteratively

### Proposals for improvement (for report)
1. More aggressive stability filtering (drop 20–30% instead of 10%)
2. Select only applprev + person depth-1 tables (drop credit_bureau_a which is most volatile)
3. Add trend features (slope of values over time) instead of just mean/max/min
4. Use walk-forward cross-validation (multiple folds) instead of single split
5. Try CatBoost (handles categoricals natively without encoding)

---

## For the Presentation Slides

### Slide: The Problem
- Predict loan default (binary classification)
- **Key twist**: scored on stability over time, not just AUC
- Metric: Gini stability = mean weekly Gini + trend bonus − variability penalty

### Slide: The Data
- 26 GB, 14 tables, 1.5M applicants, 92 weeks
- Show the depth-0/1/2 hierarchy diagram
- Highlight: 3.14% default rate, 30:1 imbalance

### Slide: Why Time-Based CV Matters
- Show a plot: random CV gives ~0.82 AUC but degrades on future weeks
- Time-based CV (weeks 0–60 / 61–91) gives honest estimate of temporal stability

### Slide: Results
- Use the comparison table above
- Key message: Sub 2 improved locally but not on Kaggle private → temporal instability is hard

### Slide: Scale Challenges
- Bullet list of the 5 scale handling strategies above
- Show the OOM error as evidence you actually hit the problem

---

## References to Include
- Competition page: https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability
- Polars docs: https://docs.pola.rs
- LightGBM docs: https://lightgbm.readthedocs.io
- Gini stability metric explanation: competition overview page
- Kaggle discussion on stability metric (search competition forum)
- Kaggle discussion on memory management with this dataset (search competition forum)
