# Presentation Speech Script

---

## Slide 1 — Title

"Hello everyone. We are Giulia Imbrea and Maeti Goidan, and today we'll present our work on the Home Credit Credit Risk Model Stability competition from Kaggle.
The goal was to build two genuinely different solutions and discuss how we dealt with the challenges of a very large dataset."

---

## Slide 2 — The Problem & The Data

"The task is to predict whether a loan applicant will default — a classic binary classification problem. Home Credit targets customers with limited credit history, often excluded from traditional scoring systems.

What makes this competition unusual is the evaluation metric. It's not just about AUC. The score combines mean weekly Gini, a trend component that rewards models that improve over time, and a penalty for week-to-week variability. A model that's accurate on average but inconsistent across weeks will score poorly.

The dataset itself is a major challenge: 26 gigabytes uncompressed, 1.5 million applicants, 14 tables organized in a hierarchy, and 92 weeks of historical data."

---

## Slide 3 — Data Exploration

"Before modeling, we explored the data to understand its structure.

The most important finding is the severe class imbalance: only 3.14% of applicants default — a 30 to 1 ratio. This directly influenced our modeling choices.

The second critical finding is the temporal structure. The default rate varies from week to week, and the data spans two years. This means we absolutely cannot use random cross-validation — it would leak future information into the training set. Instead, we always train on earlier weeks and validate on later weeks. Concretely, weeks 0 to 60 for training and 61 to 91 for validation.

We also spent significant effort on handling the scale: we used Parquet files for fast reads, Polars instead of pandas to avoid memory overflow, and cast all numeric features to float32 to halve memory usage."

---

## Slide 4 — Submission 1

"Our first submission is a baseline: LightGBM trained only on depth-0 tables — the applicant-level static features. No joins to historical records. This gives us 224 features and a fast, interpretable starting point.

We handled the class imbalance with scale_pos_weight set to 30.8, and used early stopping on validation AUC.

The results: validation AUC of 0.82 and a Gini stability score of 0.60. The per-week Gini plot shows a positive slope — the model actually improves over the validation period, which is exactly what the metric rewards.

On Kaggle, the public score was 0.486 and the private score was 0.40."

---

## Slide 5 — Submission 2

"Our second submission extends the baseline by aggregating features from four depth-1 tables: previous applications, credit bureau records A and B, and person demographics. These tables contain historical records — multiple rows per applicant — so we aggregate them using mean, max, min, and standard deviation per applicant.

This brings us to 747 features. But more features isn't always better for stability. So we apply what we call stability-first feature filtering: for each feature, we compute how much its weekly mean drifts over time using the coefficient of variation. We then drop the 10% most temporally drifting features, leaving 673 stable ones.

Locally, this improved significantly: AUC went from 0.82 to 0.85, and the Gini stability score from 0.60 to 0.66. The residual standard deviation also dropped, meaning the model is more consistent week to week.

However, on the Kaggle private leaderboard the score went slightly down. This tells us that even with our stability filter, the depth-1 aggregates still carry drift patterns that appear beyond the training period. This is the core difficulty of the competition."

---

## Slide 6 — Challenges & Improvements

"Let me close with the main challenges we encountered and what we'd do differently.

The biggest technical challenge was memory. Loading the full credit bureau table — 16 million rows, 79 columns — caused out-of-memory errors on Kaggle's 30 gigabyte notebooks. We solved this by reading files individually, casting to float32, and cleaning up memory explicitly after every join.

The more fundamental challenge is the gap between local validation and the private leaderboard. Our validation covers weeks 61 to 91, which are still relatively close to the training period. The private test goes beyond week 100 — much further in the future. The model degrades there in ways we can't directly measure locally.

For improvements, the most promising directions would be: more aggressive stability filtering, dropping the credit bureau A table entirely since it's the most volatile, and adding trend-based features — computing the slope of a value over time rather than just its average.

Thank you."
