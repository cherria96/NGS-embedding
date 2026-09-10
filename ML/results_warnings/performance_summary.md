# Multi-label anaerobic digestion warnings

CV-selected model: **NNET**. Selection uses training pooled OOF Macro AUPR, with Macro MCC as tie breaker.
The highest test Macro AUPR is **NNET**; this is a descriptive comparison, not a new tuning decision.

## Training 5-fold site-grouped CV (pooled out-of-fold)

| Model | Macro_AUPR | Macro_MCC | Micro_AUPR |
| --- | --- | --- | --- |
| NNET | 0.2757 | 0.1589 | 0.2411 |
| GLMNET | 0.2488 | 0.1316 | 0.2152 |
| RandomForest | 0.2386 | 0.0925 | 0.4275 |
| SVM | 0.2382 | 0.1319 | 0.3017 |
| KNN | 0.2170 | 0.1773 | 0.2114 |
| XGBoost | 0.1980 | 0.1520 | 0.3108 |

## Independent test

| Model | Macro_AUPR | Macro_MCC | Micro_AUPR |
| --- | --- | --- | --- |
| NNET | 0.5093 | 0.0037 | 0.3112 |
| RandomForest | 0.4874 | 0.1698 | 0.5858 |
| KNN | 0.4862 | 0.3382 | 0.3125 |
| XGBoost | 0.3879 | 0.3144 | 0.4858 |
| GLMNET | 0.3642 | 0.1634 | 0.2968 |
| SVM | 0.1531 | -0.0174 | 0.1443 |

All five classifiers are independent. Thresholds are fixed at 0.5. AUPR means average precision.
CV scores are tuning estimates for the selected hyperparameters, not nested-CV unbiased estimates.
No threshold tuning was performed. Full per-label MCC/AUPR appear in the CSV tables.

109 training samples from 28 groups; 29 test samples from 7 groups.
Missing archaeal samples were excluded explicitly in sample_audit.csv. Existing warning labels were preserved,
including zeros caused by missing target-source measurements. The smallest label has very few positives:
individual-fold AP may be undefined; pooled OOF scoring includes all training samples. MCC is 0 for degenerate cases.

GLMNET is elastic-net logistic regression (scikit-learn), not the R glmnet software.
SVM probabilities use sigmoid calibration on inner site-grouped OOF margins, with preprocessing refitted inside calibration folds.
RF/SVM/GLMNET use balanced class weights, XGBoost uses negative/positive class weighting,
and KNN/NNET use random minority oversampling confined to each training fold.
Imputation is training-median for numeric features (all-missing columns use zero), most-frequent for categories;
numerical predictors including abundances are z-scaled, and categories are one-hot encoded with unknown levels ignored.
Top-k selection is per sample and per block, union fitted on training only; zero ties are alphabetical.
Workbook annotations, including unidentified and off-domain entries, are retained as supplied.
See fit_warnings.csv for convergence diagnostics and pretraining_audit.txt for grouping and label support.

## Per-label test comparison

| Warning | Test positives | Highest AUPR (model) | Highest MCC (model) |
| --- | --- | --- | --- |
| acid_base_balance | 5/29 | 0.6178 (KNN) | 0.5167 (KNN) |
| buffer_capacity | 1/29 | 0.3333 (NNET) | 0.0000 (NNET) |
| acid_accumulation | 1/29 | 1.0000 (NNET) | 0.6944 (KNN) |
| ammonia_toxicity | 4/29 | 1.0000 (RandomForest) | 1.0000 (XGBoost) |
| biogas_quality | 2/29 | 0.4500 (RandomForest) | 0.4630 (XGBoost) |

Tied leaders are listed fully in `best_model_per_label.csv`. These test leaders are descriptive.

Labels with at most two positive test samples: buffer_capacity (1), acid_accumulation (1), biogas_quality (2). Their scores can change substantially with a single prediction. Use the full per-label table below to compare all algorithms.

| Model | acid_base_balance AP / MCC | buffer_capacity AP / MCC | acid_accumulation AP / MCC | ammonia_toxicity AP / MCC | biogas_quality AP / MCC |
| --- | --- | --- | --- | --- | --- |
| NNET | 0.543 / 0.236 | 0.333 / 0.000 | 1.000 / 0.000 | 0.525 / -0.109 | 0.146 / -0.109 |
| RandomForest | 0.487 / 0.000 | 0.167 / 0.000 | 0.333 / 0.000 | 1.000 / 0.849 | 0.450 / 0.000 |
| KNN | 0.618 / 0.517 | 0.034 / -0.076 | 1.000 / 0.694 | 0.710 / 0.709 | 0.069 / -0.154 |
| XGBoost | 0.379 / 0.145 | 0.043 / 0.000 | 0.100 / -0.036 | 1.000 / 1.000 | 0.417 / 0.463 |
| GLMNET | 0.518 / 0.145 | 0.167 / 0.000 | 0.500 / 0.000 | 0.469 / 0.475 | 0.167 / 0.197 |
| SVM | 0.129 / 0.000 | 0.038 / -0.036 | 0.034 / 0.000 | 0.430 / 0.000 | 0.133 / -0.051 |
