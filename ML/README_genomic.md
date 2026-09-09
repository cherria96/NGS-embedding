# Genomic-only joint warning prediction

Open `results_genomic/report.md` for the six-model results and
`results_genomic/genomic_importance/report.md` for taxa contribution results.

Reproduce from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ngs-mpl \
/home/best/anaconda3/envs/tsf-ad/bin/python ML/train_genomic.py --seed 42 --n-iter 8

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ngs-mpl \
/home/best/anaconda3/envs/tsf-ad/bin/python ML/importance_genomic.py --seed 20260908 --repeats 30

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ngs-mpl \
/home/best/anaconda3/envs/tsf-ad/bin/python -m unittest discover -s ML -p test_genomic.py
```

The training script reads only SampleID, Site and five warning targets from
metadata.csv. Predictors come exclusively from BAC P(%)/O(%) and ARC O(%)/G(%).
Relative abundance is stored in percent units: values strictly greater than 0.1
are retained separately in each sample; all others become zero. Columns are the
training-only union, refitted within every CV/calibration training fold. No
renormalization, Top-N ranking, or extra feature selection is performed.

The original assignment in `results_multilabel/split_and_cv_assignment.csv` is
required and reused directly, rather than regenerated. Matched sample identities
and site groups must agree exactly. There are 109 training samples from 28 groups
and 29 test samples from seven groups. The same five training validation folds
are used. Documented QC replicate groups remain joined. Two samples with missing
ARC profiles are recorded in `sample_audit.csv` and excluded from joint modeling.

RF, KNN, SVM, NNET, XGBoost and GLMNET use the same eight-candidate search spaces,
binary relevance strategy, fold-local preprocessing and imbalance handling as
the existing framework. GLMNET means sklearn elastic-net logistic regression,
not the R package. SVM calibration uses inner site-grouped OOF margins. Training
contains numeric genomic features only; the categorical transformer receives no
columns. Model objects retain training-only masking, imputation and scaling.

Only pooled Micro-AUPR and flattened multilabel MCC are used. Micro-AUPR is the
trapezoidal area under the flattened precision–recall curve. Model selection
maximizes training OOF Micro-AUPR, with MCC at 0.5 breaking ties. For each selected
configuration a shared threshold is optimized over 0.01..0.99 using training OOF
MCC, then fixed before test evaluation. OOF scores are post-tuning estimates,
not nested unbiased estimates. All models and the CV-selected model are saved
before the test scores are calculated. Because the same holdout has been used
in earlier requested analyses, these are fixed-holdout results, not validation
on a newly collected external dataset. No test results are used to tune this run.

`final_feature_matrix.csv` contains 290 genomic predictors and a SampleID index;
SampleID is not a predictor. `cleaned_merged.csv` also carries sample/site IDs and
targets, but contains no physicochemical or seasonal columns. All taxa counts,
fold columns, predictions, thresholds, tuning results, fitted model bundles,
metrics and charts are saved in `results_genomic/`.

Genomic importance uses the CV-selected genomic-only model and its fixed
settings on training validation folds, with 30 per-feature permutations. Scores
are pooled over all five folds and labels before calculating the performance
drop. Columnwise thresholding/scaling commutes with shuffling, enabling batched
predictions equivalent to raw-taxon permutations. A taxon absent from a fold's
training union has zero contribution in that fold. The full 290-taxon ranking
is saved; top-25 charts are a display choice, not feature selection.

Positive importance means permutation reduced performance; negative importance
means it improved performance. Taxonomic ranks overlap biologically, and taxa
are correlated and compositional. These dependencies can mask individual
importance; permutations can form unrealistic combinations and do not preserve
site trajectories. SD reflects permutation randomness, not confidence across
independent sites. No causal or higher/lower abundance direction is inferred.

To load a serialized bundle, make `ML` importable (for example `PYTHONPATH=ML`)
and use `joblib.load`. Supply original prefixed percentage columns in a pandas
frame. See `run_config.json` for input hashes, environment versions and settings.
