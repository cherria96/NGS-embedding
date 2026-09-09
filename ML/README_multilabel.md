# Joint five-label warning analysis

Run `train_multilabel.py` for the sample-specific >0.1% analysis. This is separate
from the historical Top-N workflow in `train_warnings.py`.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ngs-mpl \
/home/best/anaconda3/envs/tsf-ad/bin/python ML/train_multilabel.py --seed 42 --n-iter 8

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ngs-mpl \
/home/best/anaconda3/envs/tsf-ad/bin/python -m unittest discover -s ML -p test_multilabel.py
```

Dependencies are listed in `requirements.txt`; exact versions and source input
SHA-256 hashes are captured in `results_multilabel/run_config.json`.
Use `--audit-only` for data inspection and `--output-dir` for a separate run.

## Data and feature construction

The original three files are read from `data/final/`. Workbook sample suffixes
are removed while preserving No markers such as `b` and `'`. Duplicates, missing
IDs, numeric abundance validity, target values, Round/Season consistency and
Site/No consistency are checked. Every source sample's presence is recorded in
`sample_audit.csv`; `1-32'` and `4-24` have no ARC record and are excluded from
joint modeling. The merged dataset has 138 samples.

The observed data have 37 Site labels and missing seasons, rather than 34 complete
four-season sites. Documented QC replicates BSIb and DGYb are grouped with BSI and
DGY, giving 35 grouping units. Other Site labels remain distinct. If additional
codes denote the same physical facility, that mapping must be established before
relying on independent-site generalization.

Only the requested 16 metadata predictors are used. All effluent and warning
columns and site identifiers are excluded from predictors. Supplied substrate
averages and binary targets are retained. Numeric ranges are replaced with their
midpoints and ± measurements with their centers; coercions are logged.
Missing source measurements can explain some zero warning labels; the original
labels are not reinterpreted as confirmed normal.

Microbial percentages come from BAC P(%)/O(%) and ARC O(%)/G(%), without Top-N
ranking. In each sample independently, values >0.1 are retained and values <=0.1
become zero. There is no renormalization or additional ML feature selection.
The union is fitted only on training samples, separately in each CV fold and
again on the full training set. Evaluation-only taxa cannot add feature columns.

`taxa_audit.csv` describes every supplied taxon, with total and training counts
above 0.1% and a final-inclusion flag. All-sample counts are descriptive only.
`selected_taxa.csv`, `selected_*.txt`, and `feature_counts.csv` describe the final
training union. `cv_feature_columns.csv` records fold-specific unions.
`final_feature_matrix.csv` contains the thresholded training-union matrix for all
matched samples before fitted imputation/encoding/scaling. SampleID is its index,
not a predictor. `*_encoded_feature_matrix.csv` records the final fitted transform.

## Validation and models

A seed-42 random search over 5,000 site allocations approximately balances sample
counts and label-positive sample/site counts, with fixed 28/7 train/test group
counts. This uses targets solely to stratify the initial split. Seed 43 allocates
training groups to five folds. Some warnings have too few positive sites to appear
in all five validation folds. All grouped assignments and label counts are saved.

Six binary-relevance model bundles each output five probabilities. Each algorithm
gets eight deterministic ParameterSampler configurations from the documented
spaces. Numeric median imputation and z-score scaling, categorical most-frequent
imputation and unknown-safe one-hot encoding are fitted on each training portion.
All-empty numerical columns are retained with zero imputation. Scaling is also
applied to tree models for a consistent implementation.

RF, SVM and elastic-net logistic regression use balanced class weights; XGBoost
uses the training negative/positive ratio. KNN and NNET use fold-local per-label
random oversampling. A label with only one training class gets a constant model.
SVM probabilities use logistic calibration of inner three-fold site-grouped OOF
margins; preprocessing and taxa unions are refitted in those inner folds.
NNET uses sklearn MLPClassifier with lbfgs. GLMNET is implemented as sklearn
elastic-net logistic regression with saga, not the R glmnet package.

## Only two performance metrics

- **Multilabel MCC**: binary Matthews correlation of flattened N × 5 true and
  predicted label matrices. Undefined denominators use sklearn's zero convention.
- **Micro-AUPR**: trapezoidal area under a single precision–recall curve built
  from flattened N × 5 true labels and probabilities. This is not the alternative
  non-interpolated average-precision definition.

Hyperparameters and the overall model are selected by maximum pooled OOF
Micro-AUPR, with pooled MCC at 0.5 breaking ties. No individual-label or macro
performance metric participates. After choosing a configuration, one shared
threshold is optimized on its training OOF probabilities for flattened MCC over
0.01..0.99 in 0.01 increments; ties favor proximity to 0.5. The shared threshold
reduces the number of tuned quantities given rare positives. Five identical
thresholds are stored in each final model bundle.

All six final models are fitted and the CV-selected model saved before the test
performance is computed. Each receives one final test evaluation. OOF scores are
tuning estimates: hyperparameter selection and threshold optimization make them
optimistic, especially OOF MCC. The holdout is the independent evaluation. Test
leaders are descriptive and do not replace the CV-selected model.

These pooled metrics score sample-label pairs. They do not measure exact recovery
of each entire five-label vector and can be driven by more prevalent warnings.
No causal claims follow from prediction performance.

## Outputs

`results_multilabel/report.md` contains the results and interpretation. The folder
also contains the cleaned merge; raw and encoded final matrices; complete taxa
and matching audits; train/test and CV assignments; preprocessing and model
joblib bundles; hyperparameter search results and selected settings; CV scores;
thresholds; OOF and test predictions/probabilities; test performance; pooled MCC
and Micro-AUPR bar charts; pooled PR curves and coordinates; fit warnings; and
reproducibility configuration. Only the two requested performance metrics appear
in the final tables and figures.

To load a model, make `ML` importable (e.g. `PYTHONPATH=ML`) before `joblib.load`.
The model accepts a pandas frame with the metadata predictors and original
`BAC_P::taxon`, `BAC_O::taxon`, `ARC_O::taxon`, `ARC_G::taxon` percentage columns,
then applies the saved per-sample mask, training union, and preprocessing.

## Physicochemical permutation importance

Run `python ML/importance_multilabel.py --seed 20260908 --repeats 30` in the same
environment for interpretation of the already CV-selected Random Forest. Results
are saved in `results_multilabel/physicochemical_importance/report.md`, with ranked
physicochemical and categorical CSVs and horizontal bar charts for the decrease
in pooled Micro-AUPR and flattened multilabel MCC.

This reuses the existing five training folds and selected settings, verifies
baseline OOF predictions, and permutes one raw metadata predictor at a time in
validation samples. Each repeat pools all five folds before scoring. Test
predictions and test scores are not used. Existing thresholds stay fixed. No
features or final fitted models are changed. Error bars represent permutation
randomness, not independent-site confidence intervals. These post-selection,
model-specific importance values measure predictive association, not causation
or the direction of a variable's effect.
