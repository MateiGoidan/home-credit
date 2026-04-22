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

**Milestone 1**

Imbrea Giulia · Goidan Maeti

---

## The Problem & The Data

**Task:** Predict loan default for applicants with limited credit history

**Evaluation metric — Gini Stability Score:**
$$\text{Score} = \text{mean weekly Gini} + 88 \times \min(0,\text{slope}) - 0.5 \times \text{std(residuals)}$$
Rewards models that are **consistent over time**, not just accurate.

**Dataset scale:**

| | |
|---|---|
| Size | ~26 GB uncompressed |
| Applicants | 1,526,659 |
| Tables | 14 (depth 0, 1, 2) |
| Timeline | 92 weeks |
| Default rate | **3.14%** — 30.8:1 imbalance |

---

## Data Exploration

**Target imbalance:** 47,994 defaulters vs 1,478,665 non-defaulters

**Key finding — time-based CV is mandatory:**
- Default rate varies week to week
- Random CV leaks future data → falsely optimistic scores
- We always train on earlier weeks, validate on later weeks

| Split | Weeks | Rows | Default rate |
|---|---|---|---|
| Train | 0 – 60 | 1,235,013 | 3.26% |
| Validation | 61 – 91 | 291,646 | 2.66% |

**Scale challenges addressed:**
Parquet over CSV · Polars instead of pandas · float32 casting · lazy aggregation · explicit memory cleanup

---

## Submission 1 — Baseline (Depth-0 Only)

**Features:** 224 columns from 3 depth-0 tables only (no joins to historical records)

**Model:** LightGBM · `scale_pos_weight=30.8` · early stopping (50 rounds)

| Metric | Value |
|---|---|
| Validation AUC | 0.8188 |
| **Gini stability score** | **0.5975** |
| Mean weekly Gini | 0.6216 |
| Trend slope | +0.0029 ✅ |
| Kaggle public / private | 0.486 / 0.400 |

**Per-week Gini on validation (weeks 61–91):**
![width:750px](plots/02_depth0_weekly_gini.png)

---

## Submission 2 — Depth-1 + Stability Filtering

**Added:** aggregations (mean, max, min, std) from 4 depth-1 historical tables → 747 features

**Stability filter:** drop the 10% of features with highest temporal drift (CoV of weekly means) → **673 final features**

| Metric | Sub 1 | Sub 2 |
|---|---|---|
| Features | 224 | 673 |
| Val AUC | 0.8188 | **0.8474** |
| **Gini stability** | 0.5975 | **0.6618** |
| Residual std | 0.0483 | **0.0405** |
| Kaggle public | 0.4864 | 0.4879 |
| Kaggle private | **0.3995** | 0.3892 |

**Finding:** Sub 2 improved locally but not on private leaderboard — depth-1 features still carry temporal drift beyond the training period.

---

## Challenges & Improvements

**Challenges encountered:**
- **OOM on Kaggle** (30 GB RAM) when loading full depth-1 tables → fixed with lazy aggregation + float32
- **Private score gap** — model degrades on week 100+ data unseen during training
- **Polars version incompatibilities** — dtype mismatches across parquet shards

**Proposals for improvement:**
1. More aggressive stability filtering (20–30% drop instead of 10%)
2. Drop credit_bureau_a — largest table, most temporally volatile
3. Trend features — slope of historical values per applicant
4. Walk-forward cross-validation (multiple folds)
5. CatBoost — native categorical handling

**References:** Kaggle competition page · Polars docs · LightGBM docs · Competition forum discussions on stability metric and memory management
