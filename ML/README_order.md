# Order-only microbiome prediction of five joint warning labels

Read [the analysis report](results_order/report.md) for the six-model comparison.
The final metrics are flattened Multilabel MCC, trapezoidal Micro-AUPR, and
Micro-ROC-AUC. Five binary-relevance outputs remain one multi-label task.

## Reproduce

From the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ngs-mpl \
/home/best/anaconda3/envs/tsf-ad/bin/python ML/train_order.py --seed 20260909 --n-iter 8

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ngs-mpl \
/home/best/anaconda3/envs/tsf-ad/bin/python -m unittest discover -s ML -p test_order.py
```

`--audit-only` saves integration, features, split and configuration without fitting.
`--output-dir` permits a separate run directory. Dependencies are recorded in
`results_order/run_config.json`; the script also reuses search-space, partition,
JSON, label-count and threshold helpers from `train_genomic.py`.

## Inputs and method

Only `data/final/Dat_BAC.xlsx` and `Dat_ARC.xlsx` sheets `O(%)` supply predictors.
Metadata supplies identities, site grouping, season auditing, and the five targets.
Each individual abundance strictly greater than 0.1 percent is retained; other
values become zero. The fixed feature schema is the union across the 138 matched
samples, as explicitly prescribed. This predefined all-sample union is the only
exception to test feature isolation; there is no fitted feature selection,
Top-N ranking, or renormalization. There are 221 BAC and 17 ARC features.

Missing entire archaeal profiles (`1-32'`, `4-24`) are explicitly excluded in
`sample_audit.csv`. Observed abundance cells contain no missing values, so no
imputation is performed. QC replicates BSIb and DGYb are grouped with BSI and DGY.
Actual coverage is 37 site labels in 35 groups, with incomplete seasons and QC
replicates; see `site_season_inventory.csv`. The split uses 28 training groups
(110 samples) and seven test groups (28 samples).

All six models use the same columns and five training-only site-grouped folds.
Scaling is fitted within folds for KNN, SVM, NNET and elastic-net regression;
trees receive unscaled percentages. RF/SVM/elastic-net use class weights,
XGBoost uses a training-class ratio, and KNN/NNET use training-only oversampling.
SVM probability calibration uses additional inner site-grouped folds.
GLMNET denotes sklearn elastic-net logistic regression, not the R glmnet package.

Eight seeded candidate configurations per model are compared using equal mean
ranks over all three pooled OOF metrics. Candidate MCC uses threshold 0.5.
A shared threshold is then optimized using only selected-candidate training OOF
predictions and flattened MCC. The same threshold is assigned to all five labels
to limit flexibility with rare positives. Final model selection uses equal mean
ranks over all three selected-model OOF metrics, with differing metric leaders
reported explicitly. All fits, thresholds and CV selection are saved before the
single test-prediction pass. OOF estimates reuse tuning data and are optimistic.

This cohort has been used in earlier analyses. The new seeded split is isolated
within this run but does not constitute external validation on unseen data.

## Outputs

- `cleaned_merged.csv`: aligned identities, targets and masked Order abundances.
- `final_feature_matrix.csv`: microbiome predictors with a SampleID index (not a feature).
- `selected_BAC_orders.csv`, `selected_ARC_orders.csv`, `selected_taxa.csv`,
  `taxa_audit.csv`, `feature_counts.csv`: complete taxa lists and selection counts.
- `sample_audit.csv`, `sample_id_mapping.csv`, `site_season_inventory.csv`:
  integration and coverage evidence, including all excluded samples.
- `split_and_cv_assignment.csv`, `label_distribution.csv`: group/fold membership
  and target counts; rare labels cannot be represented in every validation fold.
- `run_config.json`, `leakage_check.json`, `input_structure.json`: reproducibility,
  input hashes, preprocessing, search spaces and predictor whitelist.
- `hyperparameter_tuning.csv`, `*_best_params.json`, `cv_performance.csv`,
  `cv_selected_model.json`, `thresholds.csv`: all tuning and selection decisions.
- `*_CV_OOF_predictions.csv`, `*_test_predictions.csv`: five true labels,
  probabilities and binary predictions per sample.
- `test_performance.csv`: exactly the three requested performance metrics.
- `multilabel_mcc.png`, `micro_aupr.png`, `micro_roc_auc.png` and SVG equivalents:
  final metric bar charts. Pooled PR/ROC curves and coordinates are also saved.
- `*.joblib`, `*_preprocessing.joblib`: fitted model bundles and preprocessing.
- `fit_warnings.csv`: all estimator warnings, with model/candidate/fold context.

To load a model, add `ML` to `PYTHONPATH`, then use `joblib.load`. Supply percentage
features in the named columns recorded in `final_feature_matrix.csv`. The bundle
applies the fixed per-sample mask and its training-fitted preprocessing.
