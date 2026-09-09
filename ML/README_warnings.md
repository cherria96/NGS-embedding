# Multi-label warning classification

This workflow predicts the five existing `warning_*` columns independently from
16 requested metadata predictors and sample-specific top microbiome taxa. It is
separate from the earlier TVFAs/ALK regression workflow.

Run from the repository root with dependencies in `ML/requirements.txt`:

```bash
python ML/train_warnings.py --seed 42 --n-iter 8
python -m unittest discover -s ML -p test_warning_models.py
```

The validated environment for this run is
`/home/best/anaconda3/envs/tsf-ad/bin/python`. Setting `MPLCONFIGDIR=/tmp/ngs-mpl`
and `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` avoids unwritable
plot caches and excessive numerical-library threads. Complete environment
versions and input SHA-256 hashes are recorded in `run_config.json`.
Use `--audit-only` for inspection without fitting, `--output-dir` for an isolated
run, and `--group-map mapping.json` to add explicit physical-site mappings.

## Inputs and matching

Inputs are `data/final/metadata.csv`, `Dat_BAC.xlsx`, and `Dat_ARC.xlsx`.
The CSV currently has 140 samples and 135 columns. It contains all 16 requested
predictors, including Season and the 11 requested substrate averages, and all five
binary labels. Supplied labels and averages are used without recomputing them.
The raw source writer `code/metadata_targetwriter.py` writes to a root-level
`metadata.csv`; this workflow does not invoke it or overwrite source data.

`P(%)` and `G(%)` have taxon row labels `Phylum` and `Genus`; remaining columns
are samples containing relative abundance percentages. `_rank(%)` is not used.
Workbook IDs such as `1-11b-DGYSb` map to `1-11b`, preserving number suffixes.
Metadata SampleID is checked against Round/No and Season against Round. Site/No
mapping is checked for consistency. The workbook suffix is exported alongside
the metadata Site code; their naming conventions differ, so literal equality is
not assumed. Duplicate or missing IDs and invalid abundance values fail loudly.

The shared intersection is 138 samples. `1-32'` and `4-24` lack ARC measurements
and are explicitly recorded in `sample_audit.csv` before exclusion. No missing
microbiome profiles are fabricated. Workbook taxa, including `unidentified`
and off-domain annotations, remain eligible as supplied; the workbook domain
prefix denotes the source assay, not an independently verified taxonomic domain.

## Grouping and selection

The files have 37 Site labels, not 34 complete sites. The default mapping groups
BSIb with BSI and DGYb with DGY because they are documented QC replicates. Other
labels stay distinct, producing 35 groups. This is an explicit working assumption;
if other labels share a physical site, provide a mapping and rerun the entire
analysis before interpreting independent-site performance. Missing seasons and
replicate rows are retained and reported rather than forcing four rows per group.

The holdout uses 28 training groups and 7 test groups. A deterministic search of
5,000 random group assignments balances label-positive sample counts, positive
site counts and sample sizes, with penalties for missing class coverage. Split
selection uses labels for stratification only, never model results. Training
sites are allocated to five validation folds by the same method with seed+1.
Every site is validated once; no group crosses a training/evaluation boundary.
Rare labels make positive coverage in every validation fold impossible.

Within each sample, top 10 phyla and top 16 genera are selected separately for
BAC and ARC. Ties, including zeros, break alphabetically by taxon name. The union
of selected taxa across **training samples only** defines the columns. Values
outside each sample's own top set are zero. Test samples are ranked against the
full source rank table before projecting onto training columns; unseen top taxa
do not create new columns or cause lower-ranked taxa to be promoted. Percentages
are not renormalized. This transformation is refitted in every CV training fold
and every inner SVM calibration fold; targets never enter taxon selection.

## Models, preprocessing, and scoring

Each model fits five independent binary classifiers:

- Random Forest: balanced class weights.
- KNN: fold-local random minority oversampling.
- SVM: balanced SVC, followed by sigmoid calibration fitted to inner three-fold
  site-grouped out-of-fold margins. Inner calibration refits feature selection
  and preprocessing; sample-level internal SVC probability fitting is disabled.
- NNET: MLPClassifier, fold-local random minority oversampling, no internal random
  validation split or early stopping.
- GLMNET: elastic-net logistic regression using scikit-learn's SAGA solver. This
  implements the penalized binomial objective family, not the R glmnet package
  or its exact lambda path. Uses balanced class weights.
- XGBoost: negative/positive training class-ratio weighting.

A one-class training fold uses a constant classifier and is not allowed to crash
or infer a class from evaluation data. Numerical missing values use the training
median; all-missing columns use zero. Categorical missing values use the training
mode, followed by one-hot encoding that ignores unseen categories. All numerical
features, including abundances, receive training-fitted z-score scaling. Temperature
ranges use their midpoint and `35±2` becomes 35; unparseable numeric values become
missing and are logged. Only the metadata whitelist and microbial columns enter
predictors; identities, targets and every effluent column are excluded.

Eight reproducible parameter candidates per algorithm are evaluated using the
same five folds. Full search spaces, sampled parameters, fold results, pooled OOF
results and best parameters are saved. Selection maximizes pooled OOF Macro AUPR,
with Macro MCC breaking ties. AUPR is non-interpolated **average precision**, using
probabilities. Individual folds with no positives report AP as missing; pooled
OOF AP uses all training observations. Macro AP averages defined labels and reports
the number evaluated; test label support must be checked when comparing models.
MCC uses the sklearn zero convention when its denominator is zero.

Thresholds are fixed at 0.5 for all five labels. No threshold optimization uses
CV or test observations. Pooled OOF metrics for selected parameters are tuning
estimates, not unbiased nested-CV estimates. The final model choice is recorded
from CV before calculating any test metrics. The highest-scoring test algorithms
are also identified descriptively without changing training or selecting a new
holdout. Each final bundle is fitted on training samples only.

## Output files (`ML/results_warnings/`)

| Output | Contents |
|---|---|
| `pretraining_audit.txt`, `file_structure.csv` | Detected structure, counts, grouping, leakage checks |
| `sample_audit.csv`, `sample_id_mapping.csv` | Matching coverage and workbook-to-metadata crosswalk |
| `samples_per_site.csv`, `split_and_cv_assignment.csv` | Sample counts, seasons, holdout and validation fold membership |
| `label_distribution.csv`, `simultaneous_labels.csv` | Binary-label support and 0–5 simultaneous-label counts |
| `missing_features.csv`, `numeric_coercions.csv`, `target_source_missingness.csv` | Missingness and parsing audit |
| `cleaned_merged.csv` | Matched metadata, labels and full source microbial tables |
| `final_feature_matrix.csv` | Training-union features, sample-specific masking, before imputation/scaling |
| `selected_taxa.csv`, `selected_*.txt` | Final and CV taxa, training/evaluation top-membership counts, final lists |
| `*_encoded_feature_matrix.csv` | Final training-fitted imputed/scaled/encoded matrix, indexed by SampleID |
| `hyperparameter_tuning.csv`, `*_best_params.json` | All candidates and fold scores, chosen parameters and thresholds |
| `cv_performance.csv`, `test_performance.csv` | Pooled OOF and independent test metrics, including every label |
| `performance_summary.csv`, `performance_summary.md` | Final comparison and interpretation |
| `best_model_per_label.csv`, `cv_selected_model.json` | Per-label leaders and pre-test model choice |
| `*_CV_OOF_predictions.csv`, `*_test_predictions.csv` | Actual labels, predicted labels and probabilities |
| `plots/{model}/{scope}/` | Five four-panel diagnostic figures, all-label heatmap, confusion counts, curve coordinates |
| `*.joblib`, `*_preprocessing.joblib` | Final model bundles and reusable selection/preprocessing objects |
| `fit_warnings.csv`, `run_config.json` | Fit diagnostics, environment, seed, inputs, search and thresholds |

Each label's diagnostic figure contains its confusion matrix, precision–recall
curve, ROC curve and positive/negative probability histograms. CV and test
figures live in separate directories. ROC and PR are explicitly marked undefined
where the evaluated labels do not support their calculation.

To load a saved bundle, make the source module importable:

```python
import sys
import joblib
import pandas as pd
sys.path.insert(0, 'ML')
model = joblib.load('ML/results_warnings/RandomForest.joblib')
raw = pd.read_csv('ML/results_warnings/cleaned_merged.csv', index_col='SampleID')
probabilities = model.predict_proba(raw)
labels = model.predict(raw)
```

For new samples, provide the same metadata fields and full BAC/ARC P/G abundance
columns, named `{domain}_{rank}::{taxon}`. The bundle applies its saved union,
preprocessor and classifiers. Omitted measurements must not be silently treated
as measured zeros. Training-set predictions are not independent validation.

The labels are rule-derived measurements, including existing zeros caused by
missing source effluent data. They are retained as requested; see the missingness
audit. Rare-label results, particularly those based on one positive test sample,
are highly uncertain. These data establish cross-sectional warning prediction,
not prospective forecasting without further sampling-time information.

## Visual overview

Run `python ML/visualize_warnings.py` after training. Open
`results_warnings/visualizations/index.html` in a browser to browse model and
warning diagnostics, switch between CV and test predictions, and download the
prediction CSVs. The gallery works locally without an internet connection.
Four overview figures (model comparison, per-label heatmaps, label support,
and overlaid precision–recall curves) are saved as PNG and PDF in the same folder.

## Per-label feature importance

After training, run:

```bash
python ML/feature_importance_warnings.py --repeats 20
```

This post-processing stage refits the saved best configurations in the original
five training CV folds and calculates model-agnostic permutation importance for
each warning label and all six models. No independent test rows are used.
The original final models, feature sets, and test results are unchanged.
Open `results_warnings/feature_importance/index.html` for the gallery and
`results_warnings/feature_importance/README.md` for findings and methodology.

Importance is the drop in each binary label's average precision after shuffling
one classifier input in held-out CV data. Fold-specific taxon masks and training
preprocessing are frozen; categorical one-hot blocks move together. This measures
reliance on the engineered inputs, not the effect of changing raw abundances and
reranking taxa. Numerical results include signed effects, repeat SD, fold SD,
feature eligibility, positive-fold frequency, positive top-10/top-20 frequency,
and pairwise Spearman/Jaccard consistency. Undefined AP folds are excluded
explicitly. Every model/label receives full/top10/top20 CSVs and PNG/PDF plots.

These are post-hoc interpretation results conditional on CV-tuned parameters,
not a validated feature-selection procedure or causal biological effects.
Subsequent selection would require training-only selection inside a fresh
validation procedure; do not use test interpretation to choose inputs.
