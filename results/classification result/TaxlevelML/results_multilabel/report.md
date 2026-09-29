# Joint five-label warning prediction
CV-selected model: **RandomForest**.

## Held-out test performance
| Model | Multilabel MCC | Micro-AUPR |
|---|---:|---:|
| RandomForest | 0.3798 | 0.6274 |
| KNN | 0.4314 | 0.4964 |
| SVM | 0.4196 | 0.5125 |
| NNET | 0.4930 | 0.5653 |
| XGBoost | 0.5196 | 0.6121 |
| GLMNET | 0.4483 | 0.4533 |

## Training OOF tuning estimates
| Model | Multilabel MCC | Micro-AUPR |
|---|---:|---:|
| RandomForest | 0.4782 | 0.4211 |
| KNN | 0.3564 | 0.3891 |
| SVM | 0.3636 | 0.3515 |
| NNET | 0.2776 | 0.2369 |
| XGBoost | 0.4164 | 0.3769 |
| GLMNET | 0.4830 | 0.3153 |

OOF MCC uses a threshold optimized on these same OOF predictions and is optimistic. Hyperparameters also use these folds; test is the independent evaluation.

Micro-AUPR uses trapezoidal PR area; MCC flattens all five labels. These pooled metrics do not measure exact five-label-vector correctness.

## Data and methods
Matched samples: 138; site labels: 37; groups: 35.

Train: 109 samples, 28 groups. Test: 29 samples, 7 groups.

Training sites: ADA, ADP, BSE, BSI, BSN, BSO, BSS, CGS, CWS, DGB, DGC, DGD, DGG, DGJ, GCG, GHGa, GHGb, GHH, GHJ, GSG, JJN, JJY, MGY, USJ, USO, USY, YCG, YSY

Test sites: CWM, DGY, MYS, SCH, UJG, YCD, YSD

Missing ARC: 1-32' and 4-24; excluded explicitly in sample_audit.csv.

Duplicate/missing metadata and normalized genomic IDs checked; SampleID and Site mapping validated.

QC replicates BSIb→BSI and DGYb→DGY; 35 groups, not the assumed 34 complete sites.

Final microbiome features: {'BAC_P': 45, 'BAC_O': 198, 'ARC_O': 15, 'ARC_G': 32}

Numeric ranges parsed to midpoint; ± values use center; other nonnumeric values become missing (logged).

Median numeric imputation, most-frequent categorical imputation, unknown-safe one-hot encoding, z-score scaling; all fitted on training rows.

RF/SVM/elastic-net logistic regression: balanced class weights; XGBoost: training negative/positive ratio.

KNN/NNET: per-label random oversampling on training rows only. Single-class training labels use constant prediction.

GLMNET denotes sklearn elastic-net logistic regression (saga), not the R glmnet package.

All supplied annotations remain eligible, including unidentified and off-domain taxa.

Existing warning labels retained; missing target-source measurements can produce unflagged zeros.

CV scores are tuning estimates, not nested unbiased performance estimates. Rare positives limit stability.

No additional ML feature selection. Test-only union taxa appear only in descriptive audit.

## Reproducibility
Run `python ML/train_multilabel.py --seed 42 --n-iter 8` with the versions in run_config.json.
Search spaces, input hashes, assignments, feature lists, fitted preprocessing, models, thresholds and predictions are saved alongside this report.
Test comparisons are descriptive; they do not replace the model selected from CV. No causal interpretation is supported.

## Interpretation
RandomForest was selected using training CV. RandomForest has the highest held-out Micro-AUPR and XGBoost has the highest held-out MCC. The two metrics favor different algorithms. The test set contains only 29 samples from seven groups, with very few positive examples for several warnings, so rankings have substantial uncertainty. These results support measurable pooled predictive signal in this holdout, not established deployment reliability.

Two convergence warnings occurred in folds 1 and 4 of unselected GLMNET candidate 2. The selected GLMNET candidate and final fit had no recorded convergence warnings. See fit_warnings.csv.

The initial test export encountered a threshold-vector broadcasting error after the first model's probabilities were computed. The export was corrected and completed from the saved fitted models and fixed thresholds. No hyperparameters, model selection, preprocessing, or thresholds changed in response to test results.

Validation: four targeted tests passed. Output checks verified site isolation, all 290 strict-filter microbial columns, six loadable model bundles, prediction IDs, threshold consistency, and both reported metrics recomputed from saved predictions.
