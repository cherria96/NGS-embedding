# Class microbiome-only prediction with five-label macro metrics

Run code: [train_class_macro.py](train_class_macro.py). Results: [report.md](results_class_macro/report.md).
This separate analysis uses binary relevance for Random Forest, KNN, SVM, NNET,
XGBoost and elastic-net logistic regression (a sklearn GLMNET analogue).
Earlier analysis outputs are preserved.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ngs-mpl \
/home/best/anaconda3/envs/tsf-ad/bin/python ML/train_class_macro.py --seed 42 --n-iter 8

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ngs-mpl \
/home/best/anaconda3/envs/tsf-ad/bin/python -m unittest discover -s ML -p test_class_macro.py

/home/best/anaconda3/envs/tsf-ad/bin/python ML/verify_class_macro.py
```

Completed output directories cannot be overwritten by training. Use `--output-dir`
for reproduction in another directory. `--audit-only` performs data integration and
exports audits without model fitting. The verifier reads saved predictions and
source workbooks, never fits models or generates new test predictions.

## Inputs, integration, and predictor construction

Only `data/final/metadata.csv`, `Dat_BAC.xlsx`, and `Dat_ARC.xlsx` supply data.
Class integration is implemented in `class_macro_data.py`; stateless splitting and search utilities are local to `class_macro_utils.py`;
no prior model scores or selected taxa are read. C(%) sheets supply original Class
percentages; taxa are checked against OTUs Class annotations. Workbook IDs are
normalized to Round-No with replicate markers preserved. The extra ARC suffix in
2-29-SCHG-ARC is logged. Metadata Round/Season and Site/No mappings and workbook
suffix consistency are checked. These are internal consistency checks, not independent
confirmation of laboratory sample provenance.

There are 138 paired samples, 37 Site labels and 35 groups after documented QC
replicate grouping BSIb→BSI and DGYb→DGY. Samples 1-32' and 4-24 lack ARC data and
are excluded rather than interpreted as zero abundance. The expected 34 complete
four-season sites are not verified by these files. Exact seasonal coverage is in
samples_per_site.csv and sample-level assignments in split_and_cv_assignment.csv.

Each value strictly greater than 0.1% remains its original percentage. Values at
or below 0.1% become zero, without renormalization or root/log transformation.
All supplied Class annotation bins, including unidentified/off-domain annotations,
remain eligible; no additional taxonomic filter, prevalence rule or ML feature
selection is applied. BAC and ARC percentages have separate domain denominators.

The request contains conflicting rules: the all-sample union would use held-out
abundances, while another rule prohibits any test-informed feature construction.
This implementation prioritizes the latter: the modeling union is fitted on each
training portion, including CV and SVM calibration. Final training and descriptive union counts are recorded in report.md and taxa_audit.csv. The all-sample union is a separate descriptive export and
is never used to determine fitted model columns. This is an explicit protocol
reconciliation, not a claim of literal compliance with both conflicting rules.

All metadata columns are excluded from predictors, including IDs, Site, Season,
operating conditions, substrates, effluent chemistry, gas values and targets.
Missing/invalid genomic cells are rejected; missing domain profiles are excluded.
Existing binary target values are preserved, with missing defining chemistry
reported separately. Sampling chronology cannot be established: the analysis
classifies contemporaneous warning state, not prospective warning onset.

## Training and evaluation

Seed 42 gives 28 training groups/109 samples and seven test groups/29 samples.
Initial group stratification balances positive sample/site counts and sample sizes
over 5,000 candidate allocations, without model performance. Seed 43 constructs
five training-only group folds. The split is deterministic and has appeared in
previous project analyses; this is not a new external validation cohort.

Each model evaluates eight reproducibly sampled hyperparameter combinations, with
five binary estimators per fit. Full spaces are in run_config.json. Preprocessing
and balancing are fitted only on training rows. Median imputation is a safeguard;
no genomic abundance values are missing. KNN/SVM/NNET/GLMNET use z-score scaling;
RF/XGBoost are unscaled. RF/SVM/GLMNET use balanced weights, XGBoost uses training
negative/positive ratios, KNN/NNET use training-only random oversampling to majority
class size with seed+label index. Constant classifiers handle single-class training.
SVM uses inner three-fold site-grouped OOF logistic margin calibration with refitted
union and preprocessing. GLMNET regularizes all input coefficients; no subsequent
coefficient-based feature removal occurs.

Every primary metric is the arithmetic mean of exactly five binary-label metrics:
MCC, trapezoidal precision-recall AUPR, and ROC-AUC. AUPR is not average precision.
MCC uses zero for a zero denominator. Single-class truth makes AUPR and ROC-AUC
undefined; the macro remains undefined rather than silently averaging fewer labels.
Rare positives therefore prevent reporting all fold-level macro areas, but pooled
training OOF scores contain both classes for every label and are used for tuning.

Hyperparameters and the overall preferred model minimize equal-weight mean rank
across the three pooled OOF macro metrics. MCC uses predefined threshold 0.5 for
this selection; ties favor Macro AUPR, ROC-AUC, MCC, then candidate order. After
choosing each configuration, per-warning thresholds maximize MCC on its training
OOF scores over 0.01–0.99, step 0.01; ties favor the threshold nearest 0.5. Thresholds
and the preferred model are saved before all six final models predict test once.
Optimized OOF results are tuning estimates, not unbiased nested validation results.
AUPR/ROC-AUC use probabilities and are independent of classification thresholds.

## Outputs

- `cleaned_merged.csv`, `final_feature_matrix.csv`, `descriptive_all_sample_union.csv`:
  audited identities/labels/raw percentages, final training-union masked matrix,
  and descriptive all-sample union. SampleID is an index, never a predictor.
- `selected_taxa.csv`, `selected_Bacteria_classes.csv`, `selected_Archaea_classes.csv`,
  `final_feature_list.txt`, `taxa_audit.csv`, `cv_feature_columns.csv`: full feature
  lists, domains, Class level, per-taxon counts, and training/fold inclusion.
- `sample_audit.csv`, `sample_id_mapping.csv`, `sample_id_corrections.csv`,
  `file_structure.csv`, `class_source_counts.csv`, `samples_per_site.csv`,
  `missing_features.csv`, `target_source_missingness.csv`: input audits.
- `split_and_cv_assignment.csv`, `partition_summary.csv`, `label_distribution.csv`:
  exact sample/site/fold assignments, site lists, counts and prevalence.
- `run_config.json`, `leakage_check.json`, `pretraining_audit.txt`: methods, seed,
  package versions, input hashes, complete leakage audit and limitations.
- `hyperparameter_tuning.csv`, `*_best_params.json`, `cv_model_selection.csv`,
  `cv_selected_model.json`, `thresholds.csv`, `cv_performance.csv`: all tuning
  decisions and fixed final thresholds.
- `*_CV_OOF_predictions.csv`, `*_test_predictions.csv`, `per_warning_metrics.csv`,
  `test_performance.csv`: five-label probabilities, binary predictions, truth,
  individual metrics, and exact four-column final macro comparison.
- `macro_mcc`, `macro_aupr`, `macro_roc_auc` in PNG and SVG: model names and values,
  highlighting each metric leader.
- `*.joblib`, `*_preprocessing.joblib`: trained models and preprocessing. Load with
  ML on PYTHONPATH and supply raw Class percentages, not standardized values.
- `feature_importance_per_warning.csv`, `feature_importance_macro.csv`: complete
  RF impurity/XGBoost gain importances, computed from training-fitted models only.
  These are potentially unstable predictive associations, not causal effects.
- `fit_warnings.csv`, `verification.json`: fit diagnostics and independent artifact
  validation. Iteration-limit warnings are retained, never hidden.

This independent run reads no prior result directories, model bundles, chosen hyperparameters, thresholds, feature lists or matrices. Search spaces are prespecified; all candidate fits and selections are rerun on Class data. The fixed seed may reproduce site assignments seen in earlier analyses; this does not constitute a new external cohort.
