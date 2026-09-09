# Phylum-only multi-label warning analysis

Results and complete methods: [results_phylum/report.md](results_phylum/report.md).

From the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ngs-mpl \
/home/best/anaconda3/envs/tsf-ad/bin/python ML/train_phylum.py --seed 42 --n-iter 8

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ngs-mpl \
/home/best/anaconda3/envs/tsf-ad/bin/python -m unittest discover -s ML -p test_phylum.py
```

Inputs are data/final/metadata.csv, Dat_BAC.xlsx and Dat_ARC.xlsx. Only the
P(%) sheets provide predictors. A strict per-sample >0.1 percent mask is
applied without renormalization. The fixed union across matched samples is
retained in every fold, as explicitly requested. All fitted preprocessing,
resampling, SVM calibration, tuning and threshold estimation use training only.

Six binary-relevance models predict five warning labels. Equal-weight mean
ranks over flattened multilabel MCC, trapezoidal Micro-AUPR and Micro-ROC-AUC
select hyperparameters and the model. Eight configurations per model use five
site-grouped training folds. Shared thresholds use training OOF MCC. CV scores
are post-tuning estimates, not unbiased nested estimates.

The reproducible split matches an earlier repository holdout. No holdout-based
tuning occurs in this analysis, but these samples are not historically unseen.
QC replicate site aliases remain grouped with their parents. Metadata contains
35 such groups rather than the assumed 34 complete sites. Off-domain source
annotations are explicitly documented in the report.

The results directory contains all requested matrices, taxa/counts, identity
and season audits, assignments, searches, selected settings, OOF and test
predictions/probabilities, thresholds, models, and three metric bar charts.
Prediction CSVs contain actual labels, probabilities and binary predictions
in clearly named columns. No individual-warning performance metrics are reported.

Serialized models require phylum_models on the import path: use PYTHONPATH=ML
when loading with joblib. Input column names are domain-prefixed percentage
features from final_feature_matrix.csv; SampleID is an index, not a predictor.
To rebuild only the narrative from existing outputs, run ML/report_phylum.py.
