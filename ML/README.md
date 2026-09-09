# Site-grouped TVFAs / ALK regression

For the five-label warning classification comparison (RF, KNN, SVM, NNET,
XGBoost, and elastic-net logistic regression), see
[README_warnings.md](README_warnings.md). That workflow uses training-only,
sample-specific top-taxon selection and excludes all effluent predictors.

Run from the repository root:

```bash
python -m pip install -r ML/requirements.txt
python ML/train_vfa_alk.py --n-iter 20 --folds 5
```

Inputs default to `data/final/`; outputs default to `ML/results_selected/`, preserving
the earlier substrate-block experiment in `ML/results/`. Override using
`--data-dir`, `--output-dir`, `--seed` (42), or `--n-jobs` (1). A smoke test can use
`--n-iter 1 --folds 2`; it is not a full hyperparameter search.

## Features and assumptions

The script uses exactly 23 physicochemical numerical features, `substrate_type`,
and 52 microbial features: 10 phyla and 16 genera per workbook. Phyla come from
`P(%)`; genera must come from `G(%)` because phylum tables do not contain genus
abundances. Taxa are ranked by mean percentage across all samples in each source
sheet, as requested. Domain/rank prefixes prevent feature-name collisions.
Unidentified taxa remain eligible; percentages are not renormalized.

The requested `substrate_avg_*` columns are derived using component substrate
flows: `sum(Q_i * value_i) / sum(Q_i)`. Absent substrates (blank names) contribute
zero flow. Missing flow for a present substrate, negative flow, zero total flow,
or missing chemistry for a positive-flow substrate yields a missing average,
subsequently imputed inside training folds. Existing `Q_total_Tpy` is retained as
its own predictor. The pH average is an arithmetic descriptive feature, not a
prediction of the pH of a physical mixture. Separate substrate blocks and presence
indicators from the previous experiment are no longer model features.

Protein is `max(TKN - TAN, 0) * 6.25`; a missing input leaves protein missing.
Targets with missing/nonfinite values or nonpositive alkalinity are excluded,
never imputed. Sample IDs are normalized from `1-11b-DGYSb` to `1-11b`, preserving
replicate suffixes. Only samples shared by metadata and both workbooks are used.
Temperature ranges become midpoints and `35±2` becomes 35. Other nonnumeric
chemical entries (e.g. `Over range`) become missing and are logged.

## Training and interpretation

GroupShuffleSplit holds out approximately 20% of unique `Site` labels. All
available seasons of each site stay together; missing seasons are not fabricated.
Sample proportions may differ from 80:20. Site labels are used exactly as recorded,
including replicate codes; different labels are not automatically consolidated.
Neither Site nor Season is a predictor.

GroupKFold is applied only to training sites, and group labels are passed to
RandomizedSearchCV. No site crosses the holdout boundary or either side of a CV
fold. Numerical median imputation and standardization, categorical imputation and
one-hot encoding, and target standardization are fitted inside each training fold.
Entirely missing numerical columns are filled with zero. Target scaling supports
SVR/MLP/ElasticNet tuning; predictions and RMSE are returned in original ratio
units. Unknown categorical levels are accepted by the encoder.

Six models are tuned by mean CV RMSE, and the best parameters per model are refit
on all training samples. The overall ranking uses CV RMSE rather than test scores.
NNET is implemented as MLPRegressor, and GLMNET as scikit-learn ElasticNet (not the
R glmnet package). Train scores are fitted, in-sample scores. Convergence warnings
are left visible so problematic candidate fits can be reviewed.

Global taxon selection includes test feature distributions as explicitly requested;
this is not strictly training-only feature selection. Effluent predictors are
contemporaneous measurements, so the workflow does not establish forecasting
performance.

## Outputs

- `performance_summary.csv` and `performance_summary.md`: model comparison, with
  Train/Test RMSE and Train/Test R²; CSV additionally includes CV RMSE.
- Six PNGs, each showing Train and Test actual-versus-predicted scatter plots.
- Per-model predictions, full CV search results, best parameters and fitted
  `.joblib` preprocessing/model pipelines.
- `integrated_data.csv`, `microbial_ranking.csv`, `sample_audit.csv`,
  `coerced_values.csv`, `split_membership.csv`, `split_coverage.csv`,
  `cv_membership.csv`, and `run_config.json`.

Saved pipelines accept engineered features in `run_config.json`. New raw files
need the same engineering and the originally selected microbial feature panel;
do not independently rerank taxa for inference.

## Automated selection and importance

Within every CV training fold, zero-variance selection (`VarianceThreshold(0)`)
removes constant physicochemical columns after imputation and before scaling.
It also removes constant one-hot columns. The microbial panel is retained in a
separate preprocessing branch. This conservative selection removes uninformative
constants; it does not remove correlated or weakly predictive nonconstant columns.
No feature-selection statistic is computed from the test set. Final masks describe
the best pipeline refitted on the complete training set; masks can differ by fold.

The model with lowest mean grouped CV RMSE is selected for interpretation.
`best_model_feature_selection.csv` records retained and dropped encoded features.
`best_model_selected_features.png` displays the physicochemical selection mask.
`best_model_feature_importance.png` displays held-out permutation importance for
all retained input variables, with error bars over 30 permutations. Categorical
`substrate_type` is permuted jointly rather than independently permuting dummy
columns. The CSV exports signed RMSE increases, standard deviations, and relative
percentages normalized over positive importance values. Negative importances are
not evidence of a protective effect. Correlated phyla/genera can obscure importance.
These test-set explanations do not feed back into tuning or feature selection.
