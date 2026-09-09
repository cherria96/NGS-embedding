# Genomic/microbiome-only prediction of five warning conditions

This analysis predicts all five warning flags jointly using binary relevance with
Random Forest, KNN, SVM, NNET, XGBoost and elastic-net logistic regression (GLMNET
analogue implemented with sklearn saga, rather than the R glmnet package).
Results are in [results_family_genomic/report.md](results_family_genomic/report.md).

Reproduce from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/ngs-mpl XDG_CACHE_HOME=/tmp/ngs-cache \
/home/best/anaconda3/envs/tsf-ad/bin/python ML/train_family_genomic.py --seed 42 --n-iter 8

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/ngs-mpl XDG_CACHE_HOME=/tmp/ngs-cache \
/home/best/anaconda3/envs/tsf-ad/bin/python -m unittest discover -s ML -p test_family_genomic.py
```

Use `--audit-only` to inspect input/matching/feature selection without fitting
models, or `--output-dir PATH` to save a separate run. Dependencies are in
requirements.txt; exact package versions and three input SHA-256 hashes are
recorded in run_config.json. Earlier analysis files remain separate.

## Inputs and matching

Only data/final/metadata.csv, Dat_BAC.xlsx and Dat_ARC.xlsx supply analysis data.
The OTUs sheets contain 2,925 BAC and 111 ARC feature rows, taxonomy columns, and
sample counts. Sample headers are normalized from Round-No-siteSuffix to
Round-No while preserving apostrophes and replicate markers. The extra `-ARC`
in `2-29-SCHG-ARC` is removed for suffix consistency checks and logged. BAC/ARC
suffix agreement and within-site seasonal suffix consistency are checked against
the metadata Round/No/Site mapping. Source headers and mappings are preserved.
This verifies internal consistency, not independent laboratory provenance.

Duplicate/missing IDs, duplicate OTU IDs, target values, nonnegative finite
counts, nonempty sequencing samples, Round/Season consistency and Site/No
consistency are validated. Metadata IDs are unique, so Site/SampleID pairs are
also unique. Every source sample is listed in sample_audit.csv. Samples `1-32'`
and `4-24` lack ARC profiles and are explicitly excluded; 138 paired samples
remain. There are 37 Site labels. Documented BSIb/BSI and DGYb/DGY QC replicates
are grouped together, giving 35 SiteGroup units. Other site labels are distinct;
this is not a dataset of exactly 34 complete four-season sites.

## Family aggregation and selection

BAC and ARC are processed separately. All supplied OTU counts are summed by the
Family annotation, then divided by the domain-specific sample total and
multiplied by 100. This is algebraically equivalent to summing OTU relative
abundances. Unidentified and off-domain annotations remain eligible; there is no
unrequested taxonomic filtering. Missing Family annotations map to unidentified.
Do not use rounded F(%) sheets or the reduced F_rank(%) sheets as inputs.

Selection is fitted on each CV training portion and again on all 109 training
samples: retain a Family when its percentage is strictly >1 in at least 5% of
those samples. This means at least six of the 109 final training samples. Each
sample's values <=1 are replaced by zero; retained values are fourth-root
transformed in percentage units, without renormalization. The same fitted column
set is applied to validation/test data. Inner SVM calibration also refits this
selection. No test values/labels determine model feature inclusion.

There are 542 BAC Families and 27 ARC Families before selection. Final training
selection retains 64 BAC and 12 ARC features. All-138-sample descriptive counts
would retain 70 BAC and 13 ARC Families, but those counts are not used to select
features. All-sample counts refer to the matched cohort. Complete retained lists,
counts and percentages for both scopes are in family_summary.md,
retained_BAC_families.csv and retained_ARC_families.csv. Fold-specific columns are
in cv_feature_columns.csv. No importance-based or Top-N selection is performed.

## Genomic-only predictors and metadata use

Modeling condition: **Genomic/Microbiome features only**.
X contains only 64 BAC_F:: and 12 ARC_F:: Family columns. Y contains the five
warning labels. Metadata supplies sample/site identities, seasonal identity
checks, grouping, targets, target distributions and leakage audits only. All
physicochemical, operational, substrate and categorical metadata variables are
excluded from X. The selector rejects non-Family columns during fitting.

Numerical median imputation (all-empty columns use zero) and z-score scaling
are fitted on each training portion. There are no categorical predictor columns.
The raw genomic matrices have no missing abundance values. Missing ARC samples
are explicitly excluded rather than imputed as absent organisms.

All metadata columns, including all effluent variables, targets and identities,
are listed as excluded predictors in leakage_check.json. cleaned_merged.csv
contains identifiers and labels alongside raw Family percentages, but no
physicochemical or operational measurements. final_feature_matrix.csv contains
only the 76 selected fourth-root-transformed Family features and a SampleID
index, which is not a predictor. No target-derived features are calculated.
Missing underlying effluent measurements can yield warning zeros; existing labels
are retained and source missingness is reported separately.

## Split, tuning and imbalance

Seed 42 allocates 28 training and seven test site groups, balancing sample and
label-positive sample/site counts over 5,000 random candidate allocations.
Labels are used only for this initial stratification, not model performance.
When the previous seed-42 Family assignment is available, exact agreement of all
site/split/fold assignments is required; previous model performance is not read.
Seed 43 allocates training groups to five validation folds using the same
stratification objective. Every group remains in one split and one validation
fold. Rare labels cannot be represented in every validation fold. All sample,
site, split and fold assignments and label distributions are saved.

Each model evaluates eight ParameterSampler configurations, each in five grouped
folds. Search spaces and selected parameters are saved. RF/SVM/GLMNET use balanced
class weights; XGBoost uses the training negative/positive ratio. KNN/NNET use
per-label random oversampling only after training preprocessing. Single-class
training targets use constant classifiers. NNET is MLPClassifier with lbfgs.
SVM probabilities use logistic calibration of inner three-fold site-grouped OOF
margins, with selection/preprocessing refitted inside calibration folds.

## Three metrics and selection

- Multilabel MCC: standard binary MCC on flattened sample-by-five-label true and
  thresholded prediction matrices. Undefined denominators use sklearn's zero.
- Micro-AUPR: trapezoidal area under one precision–recall curve of flattened truth
  and probabilities; this is not noninterpolated average precision.
- Micro-ROC-AUC: ROC-AUC of flattened truth and probabilities.

No per-label or macro performance metrics, accuracy or F1 are reported or used.
Hyperparameters and the overall model minimize the equal-weight mean descending
rank across the three pooled OOF metrics. MCC uses threshold 0.5 for this ranking.
Ties favor Micro-AUPR, then Micro-ROC-AUC, then MCC. This criterion avoids imposing
comparable numerical scales on MCC and the two area measures.

After selecting each model's configuration, optimize a shared threshold over
0.01..0.99 in 0.01 steps on its training OOF predictions to maximize flattened MCC;
ties favor proximity to 0.5. A shared threshold limits fitting extra parameters
with very few positives. The five stored thresholds are consequently identical
within each model. Thresholds are frozen before test evaluation. Reported CV MCC
uses that optimized threshold; cv_model_selection.csv records the 0.5 scores used
for selection. Threshold optimization and hyperparameter reuse make OOF scores
optimistic; these are tuning estimates rather than nested validation estimates.

Each selected configuration is retrained on all training samples. All six models
and the CV selection are saved before each receives one final test evaluation.
The fixed holdout has been used in previous analyses: it is isolated during this
run but is not a never-before-used external validation cohort. Test comparisons
do not change the CV-selected model. The metrics pool sample-label pairs and do
not establish exact five-label-vector correctness; common warnings can dominate.
Few independent groups and rare positives limit generalization claims.

## Saved outputs

- report.md, test_performance.csv, cv_performance.csv: overall three-metric results.
- multilabel_mcc.png, micro_aupr.png, micro_roc_auc.png: six-model bar charts.
- micro_precision_recall.png, micro_roc.png and coordinate CSVs: pooled curves.
- cleaned_merged.csv, final_feature_matrix.csv, *_encoded_feature_matrix.csv:
  audited source merge, transformed selected features before fitted preprocessing,
  and final training-fitted encoded/scaled matrices.
- file_structure.csv, sample_id_mapping.csv, sample_id_corrections.csv,
  sample_audit.csv, samples_per_site.csv, numeric_coercions.csv (empty; no metadata parsing),
  missing_features.csv, target_source_missingness.csv: input audits.
- family_source_counts.csv, family_selection_summary.csv, taxa_audit.csv,
  selected_taxa.csv, retained_*_families.csv, selected_*_F.txt, feature_counts.csv,
  cv_feature_columns.csv, family_summary.md: complete selection evidence.
- split_and_cv_assignment.csv, label_distribution.csv, simultaneous_labels.csv:
  grouping and target distribution.
- hyperparameter_tuning.csv, *_best_params.json, cv_model_selection.csv,
  cv_selected_model.json, thresholds.csv: search, selection and fixed thresholds.
- *_CV_OOF_predictions.csv, *_test_predictions.csv: all five true labels,
  probabilities and binary predictions together with sample/site identities.
- *.joblib: trained binary-relevance bundles and preprocessing objects.
- run_config.json, leakage_check.json, pretraining_audit.txt, fit_warnings.csv:
  reproducibility, leakage checks, methods and convergence warnings.

To load joblib bundles, make ML importable (`PYTHONPATH=ML`). Supply only untransformed Family percentages named `BAC_F::Family` and
`ARC_F::Family`; the saved bundle selects, masks, fourth-roots, imputes, encodes,
scales and predicts. Do not feed the already transformed final_feature_matrix
back into the bundle, which would apply the transformation twice.

See this run’s fit_warnings.csv and report.md for convergence diagnostics.
