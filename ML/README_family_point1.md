# Family microbiome-only multilabel warnings: sample-specific >0.1%

This analysis treats five warning targets as one binary-relevance multilabel task.
Only BAC and ARC Family relative abundances enter the predictors. It is separate
from earlier analyses and does not reuse their selected taxa or model results.

## Reproduce

From the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/ngs-mpl \
/home/best/anaconda3/envs/tsf-ad/bin/python ML/train_family_point1.py --seed 42 --n-iter 8

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/ngs-mpl \
/home/best/anaconda3/envs/tsf-ad/bin/python -m unittest discover -s ML -p test_family_point1.py
```

Use `--audit-only` for integration and feature audits without training. Use
`--output-dir PATH` for a separate run. Dependencies are listed in requirements.txt;
run_config.json records exact versions, input SHA-256 hashes, seed, and search spaces.
Model bundles require `ML` on the Python import path when loading with joblib.

## Integration and feature construction

Inputs are only data/final/metadata.csv, Dat_BAC.xlsx, and Dat_ARC.xlsx.
The F(%) sheets supply Family percentages, in units 0–100. Taxon names are checked
against the OTUs Family annotations. There are 542 BAC and 27 ARC source families.
No other rank or reduced/ranked sheet supplies predictors. Original annotations,
including unidentified and off-domain annotations, are retained without an extra
filter; Family labels are annotation bins, not necessarily valid named families.
Workbook percentages are rounded: the strict threshold is applied to those
supplied values, not recomputed at greater precision from counts.

Every value <=0.1 is set to zero independently within its own sample; every value
>0.1 is preserved as its original percentage. The fixed schema is the union of
families exceeding 0.1 in at least one of the 138 matched samples: **346 BAC +
24 ARC = 370 features**. There is no prevalence filter, Top-N selection, root/log
transform, renormalization, or additional ML feature selection. All CV folds and
models use these same columns, even if a column is zero throughout a training fold.

The all-sample union is the sole feature-space operation that accesses held-out
abundances, as explicitly allowed by the supplied protocol. It does not use labels.
No test information fits preprocessing, tuning, thresholds, or model selection.
Per-sample selected taxa are the nonzero cells of final_feature_matrix.csv; the
union and each taxon's selected-sample count are in selected_taxa.csv and the
complete domain-specific retained_*_families.csv files. All source taxa, including
excluded taxa, are in taxa_audit.csv. Source and union counts are in feature_counts.csv.

Sample headers Round-No-suffix map to metadata Round-No, preserving apostrophes
and replicate markers. The redundant ARC suffix in 2-29-SCHG-ARC is logged.
BAC/ARC suffix agreement, within-site suffix consistency, unique and nonmissing
IDs, Round/Season identity, Site/No identity, binary targets, finite nonnegative
percentages, and sample totals near 100% are checked. This checks internal identity
consistency, not independent laboratory provenance.

Samples 1-32' and 4-24 lack ARC records and are explicitly excluded in
sample_audit.csv. Missing entire domain profiles are not interpreted as zeros.
The matched cohort has 138 samples, 37 Site labels, and 35 SiteGroup units after
mapping documented QC replicates BSIb→BSI and DGYb→DGY. All samples sharing a Site
remain together; this additional grouping also prevents their QC replicates from
crossing partitions. The files do not contain exactly 34 complete four-season
sites. samples_per_site.csv lists actual seasonal coverage.

Metadata supplies targets and identity/grouping information only. All metadata
columns are excluded from predictors and enumerated in leakage_check.json,
including eff_pH, eff_ALK, eff_TVFAs, eff_HPro, eff_HAc, eff_TAN, eff_CH4 and the
warning labels. Cleaned merged data include identities and targets for auditing;
these columns are never passed to a model. Existing target zeros are preserved;
missing source chemistry can mean an unflagged zero rather than confirmed normal.
See target_source_missingness.csv.

## Split and training

Seed 42 fixes an approximately 80:20 grouped holdout: 28 training groups/109
samples and 7 test groups/29 samples. A reproducible search over 5,000 group
allocations balances positive sample counts, positive site counts, and sample
numbers. Labels are used only for initial stratification. Seed 43 constructs five
training-only folds with the same strategy. Every sample and site assignment is
saved in split_and_cv_assignment.csv; label_distribution.csv contains counts for
training, test, and both sides of each CV fold. Rare positives cannot populate all
five validation folds. No split is chosen using model performance.

Eight ParameterSampler configurations per model are each assessed on the same five
folds. Search spaces, every candidate/fold score, selected parameters, and OOF
predictions are saved. Binary relevance fits five internal binary estimators but
returns one sample-by-five prediction matrix. Single-class training labels use a
constant classifier rather than failing or removing a label.

There are no missing abundance cells in the supplied Family sheets. Invalid or
missing percentages cause an error instead of silently treating missingness as
absence. Median imputation (zero fallback for all-empty columns) is nevertheless
inside each fitted pipeline. KNN, SVM, NNET, and GLMNET use training-fitted z-score
scaling; RF and XGBoost do not. No metadata encoding occurs.

RF, SVM, and GLMNET use balanced class weights. XGBoost uses each binary training
set's negative/positive ratio. KNN and NNET use per-label random oversampling only
on preprocessed training rows. No validation/test rows are resampled. NNET uses
MLPClassifier with lbfgs. GLMNET is sklearn elastic-net logistic regression with
saga, an analogue of GLMNET rather than the R package. SVM probabilities come from
logistic calibration on inner three-fold site-grouped OOF margins; preprocessing
is refitted within each calibration fold. Calibration sees no outer validation or
test labels. Fit warnings, including iteration-limit warnings, are saved.

## Exactly three performance metrics

- Multilabel MCC: binary Matthews correlation after flattening all sample-label
  truth/prediction pairs. A zero denominator yields zero.
- Micro-AUPR: trapezoidal area under one precision–recall curve from flattened
  truth and probabilities. This is explicitly not noninterpolated average precision.
- Micro-ROC-AUC: ROC area from flattened truth and probabilities.

Hyperparameters and the preferred model minimize equal-weight mean descending
rank across the three pooled OOF metrics, using MCC at threshold 0.5. This
predeclared combined rule avoids comparing the metrics' raw numerical scales.
Ties favor Micro-AUPR, then Micro-ROC-AUC, then MCC. All three scores and the ranking
are saved; divergent metric leaders are reported as trade-offs, not unanimous
superiority.

After each model's configuration is selected, a shared threshold across the five
labels maximizes flattened MCC on its training OOF predictions over 0.01–0.99 in
steps of 0.01; ties favor proximity to 0.5. A shared threshold limits extra fitting
with very few positive examples. Thresholds are saved for all five labels and
frozen before test prediction. The displayed CV MCC uses the optimized threshold;
cv_model_selection.csv contains the 0.5 MCC used for selection. CV scores are
optimistic tuning estimates, not nested unbiased validation estimates.

All six final bundles and the selected model decision are saved before final test
evaluation. Each bundle predicts the holdout once; plots use those saved scores.
Test metrics describe performance and trade-offs without changing the CV choice.
The deterministic seed-42 holdout has appeared in previous project analyses, so
it is isolated within this analysis but is not a never-before-used external cohort.
Pooled metrics describe sample-label pairs, not exact recovery of whole five-label
vectors; common labels can dominate. Few independent sites and rare positives
limit certainty. No causal interpretation is made.

## Outputs in results_family_point1/

- report.md: performance, complete taxa lists, methods, and limitations.
- final_feature_matrix.csv: 138 × 370 masked percentages, with SampleID as an index.
- cleaned_merged.csv: identities, five labels, and original Family percentages.
- retained_BAC_families.csv, retained_ARC_families.csv, selected_taxa.csv,
  taxa_audit.csv, selected_*_F.txt, feature_counts.csv, cv_feature_columns.csv:
  complete feature lists and sample counts.
- file_structure.csv, sample_id_mapping.csv, sample_id_corrections.csv,
  sample_audit.csv, samples_per_site.csv, missing_features.csv,
  target_source_missingness.csv: integration and data-quality records.
- split_and_cv_assignment.csv, label_distribution.csv: exact site/sample partitions.
- run_config.json, leakage_check.json, pretraining_audit.txt: reproducibility/methods.
- hyperparameter_tuning.csv, *_best_params.json, cv_model_selection.csv,
  cv_selected_model.json, cv_performance.csv, thresholds.csv: training decisions.
- *_CV_OOF_predictions.csv, *_test_predictions.csv: five true labels, probabilities,
  and binary predictions, alongside sample/site/season identity.
- test_performance.csv: the six-model comparison with only the three metrics.
- multilabel_mcc.png, micro_aupr.png, micro_roc_auc.png: requested bar charts.
- micro_precision_recall.png, micro_roc.png and coordinate CSVs: combined curves.
- *.joblib: final model/preprocessing bundles. *_encoded_feature_matrix.csv contains
  the final training rows after their model's fitted preprocessing.
- fit_warnings.csv: candidate/fold/final fit warnings.

For inference, supply the 370 named percentage columns (BAC_F:: and ARC_F::), in
original percentage units. The bundle applies the fixed mask and its fitted
preprocessing. Do not supply standardized encoded matrices as raw inputs.

Validation: `python ML/verify_family_point1.py` independently checks the saved
feature matrix against the source workbooks and recomputes all saved OOF/test
metrics without refitting or generating test predictions. Results are saved in
verification.json. Read probability CSVs with `float_precision="round_trip"` to
preserve floating-point ties when reproducing curve areas.
