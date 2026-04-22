# Home Credit - Credit Risk Model Stability (Kaggle Competition)

## Project Context

This is a university assignment based on the Kaggle competition **"Home Credit - Credit Risk Model Stability"** (https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability). The competition is closed but accepts late submissions to the post-competition leaderboard.

### Team
- 2-person team (me + one colleague)
- Using a shared GitHub repository

### The Problem
Predict whether a loan applicant will default on a loan. Home Credit provides consumer loans to people with limited credit history, often in emerging markets.

### The Key Twist: Stability Over Time
Unlike a typical credit scoring competition, this one evaluates models on a **gini stability metric** that rewards:
1. Raw predictive power (AUC / Gini)
2. A flat or improving trend across weekly buckets (penalizes time-based degradation)
3. Low variability in per-week Gini scores

Mechanically: the test set is split into weekly buckets via `WEEK_NUM`, Gini (`2*AUC - 1`) is computed per week, a linear regression is fit through the weekly Ginis to measure trend, and residual variance is penalized. Final score combines mean Gini, trend slope, and residual stability.

**Implication**: Time-based cross-validation using `WEEK_NUM` is essential. Random CV will give misleadingly high scores and hurt the leaderboard result.

### The Data
- **Large dataset**: ~26 GB uncompressed, multi-table, hierarchical
- Available as both CSV and Parquet — **use Parquet** (much faster)
- Tables are organized by "depth":
  - Depth 0: Applicant-level (base table)
  - Depth 1: One-to-many historical records (previous applications, credit bureau, etc.)
  - Depth 2: Deeper nested records (installment-level data, etc.)
- Column suffixes indicate type: `_P`, `_M`, `_D`, `_T`, `_A`, `_L` (number, category, date, etc.)
- Target is imbalanced (defaulters are minority class)
- `WEEK_NUM` column identifies the time bucket — central to both evaluation and CV strategy

---

## Assignment Requirements (Milestone 1)

The professor expects:
1. Understanding of the task and data (demonstrated via EDA)
2. **At least 2 different solutions submitted to Kaggle** (genuinely different approaches, not hyperparameter tweaks)
3. Since Home Credit is a **large dataset**: a substantive discussion of how to deal with resource constraints (sampling, memory management, feature selection) rather than just ranking above bottom 25%
4. A **3-page report** covering:
   - Problem and data overview
   - EDA highlights
   - Methodology (the two approaches)
   - Discussion of handling the scale
   - Results (leaderboard scores)
   - Challenges and proposed improvements
5. **References** to tools, papers, and Kaggle discussions used

### Planned Approach for the 2 Submissions

**Submission 1 — Baseline on depth-0 data only**
- Use only the `train_base.parquet` table
- LightGBM with minimal preprocessing
- Fast, cheap, demonstrates mechanics
- Time-based CV using `WEEK_NUM`

**Submission 2 — Full pipeline with aggregated features**
- Add aggregated features from 3-5 depth-1 tables (careful scope discipline — don't try to use every table)
- Use Polars for joins/aggregations due to memory
- Same LightGBM (or possibly CatBoost) model family
- Explicitly test feature temporal stability and drop features that hurt stability even if they help AUC
- This submission is where the "handling the scale" discussion lives

### Using Public Notebooks
Allowed and expected, per the professor's explicit mention of citing Kaggle discussions. Rules:
- Must understand every piece of code used
- Must cite every notebook/discussion borrowed from
- Can't submit two unmodified forks as "two different solutions"
- Write the report in our own words

---

## Environment Setup (Already Done)

### Hardware
- MacBook, Apple Silicon (M-series)
- 32 GB RAM
- macOS
- Hybrid workflow: local for most work + Kaggle notebooks for heavy experiments / GPU needs

### Stack
- **Python**: 3.11 (pinned in pyproject.toml)
- **Environment manager**: `uv` (not conda, not pip directly)
- **Package manager**: Homebrew (at `/opt/homebrew` for Apple Silicon)
- **Editor**: VS Code with Python + Jupyter extensions
- **Libraries**:
  - Data: `polars>=1.0.0`, `pandas>=2.2.0`, `pyarrow>=15.0.0`, `numpy<2.0.0`
  - ML: `scikit-learn`, `lightgbm`, `catboost`, `xgboost`
  - Viz: `matplotlib`, `seaborn`
  - Notebooks: `jupyterlab`, `ipykernel`, `ipywidgets`
  - Utilities: `tqdm`, `kaggle`, `optuna`, `shap`

**Important Apple Silicon note**: `libomp` is installed via Homebrew — required for LightGBM on macOS. GPU acceleration for LightGBM/CatBoost targets NVIDIA/CUDA only, so for GPU experiments we'd offload to Kaggle notebooks.

### Project Structure
```
home-credit/
├── .gitignore           # excludes data/, .venv/, kaggle.json, *.pkl
├── README.md
├── pyproject.toml       # uv-managed dependencies
├── CLAUDE.md            # this file
├── data/                # gitignored - each team member downloads separately
│   ├── raw/parquet_files/{train,test}/*.parquet
│   └── processed/       # our feature-engineered outputs
├── notebooks/
│   ├── 00_sanity_check.ipynb  # done, confirms data loads
│   ├── 01_eda.ipynb           # TODO
│   ├── 02_baseline.ipynb      # TODO
│   └── 03_feature_eng.ipynb   # TODO
├── src/
│   ├── __init__.py
│   ├── data.py          # loading, preprocessing
│   ├── features.py      # feature engineering
│   └── model.py         # training pipelines
├── submissions/         # kaggle submission CSVs
└── references/          # papers, notes
```

### Kaggle API
- Using **Legacy API Key** (classic `kaggle.json` at `~/.kaggle/kaggle.json`, chmod 600)
- The newer "Create New Token" on Kaggle generates a different format that doesn't work with the `kaggle` Python package — had to use the "Create Legacy API Key" button instead
- API is confirmed working (`kaggle competitions list` returns data)
- Competition rules accepted on the Kaggle website
- Data download command: `uv run kaggle competitions download -c home-credit-credit-risk-model-stability`

### Git / GitHub
- Repo is initialized and pushed to GitHub
- Data directory is gitignored — each team member downloads the data locally
- `kaggle.json` is also gitignored

---

## Current Status

Setup is complete and validated:
- ✅ Homebrew, `uv`, Python 3.11 installed
- ✅ All dependencies installed via `uv sync`
- ✅ Project structure created
- ✅ Git repo initialized, pushed to GitHub
- ✅ Kaggle API authenticated and working
- ⏳ Downloading competition data (or just completed)
- ⏳ Sanity check notebook validates data loads correctly

### Next Steps (Planned)
1. Proper EDA notebook — explore base table, target distribution over `WEEK_NUM`, missingness patterns
2. Understand table structure and key joins for depth-1 tables
3. Build Submission 1: baseline LightGBM on depth-0 data with time-based CV
4. Build Submission 2: add aggregated depth-1 features using Polars, test stability
5. Write the 3-page report

---

## Key Technical Gotchas to Remember

1. **Never use random CV** — use `WEEK_NUM`-based splits (earlier weeks train, later weeks validate)
2. **Don't leak features from the future** — aggregations must only use records with `WEEK_NUM` earlier than the target row's week
3. **Use Polars, not pandas**, for the multi-table work — pandas will OOM on 32GB with this dataset
4. **Pin NumPy to <2.0.0** — some libraries still have compatibility issues with NumPy 2.x
5. **Scope discipline matters more than cleverness** — teams that pick 3-5 depth-1 tables and do them well usually outperform teams that try to use every table
6. A feature can be predictive but unstable — winning solutions explicitly filtered features on temporal stability, not just predictive power

---

## Reference Material to Cite in the Report

(To be expanded as we go)

- Competition page: https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability
- Kaggle discussion forum on the competition (various threads on stability metric, memory, winning solutions)
- Tool docs: Polars, LightGBM, CatBoost, scikit-learn
- Public EDA and baseline notebooks (to be chosen and cited as we use them)
