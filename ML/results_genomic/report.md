# Genomic-only joint five-label warning prediction
CV-selected model: **RandomForest**.

## Held-out test performance
| Model | Multilabel MCC | Micro-AUPR |
|---|---:|---:|
| RandomForest | 0.4696 | 0.6038 |
| KNN | 0.3659 | 0.4559 |
| SVM | 0.3798 | 0.5061 |
| NNET | 0.4109 | 0.4745 |
| XGBoost | 0.4483 | 0.5773 |
| GLMNET | 0.4675 | 0.4456 |

## Training OOF tuning estimates
| Model | Multilabel MCC | Micro-AUPR |
|---|---:|---:|
| RandomForest | 0.4766 | 0.3933 |
| KNN | 0.3292 | 0.3813 |
| SVM | 0.3679 | 0.3210 |
| NNET | 0.2524 | 0.2272 |
| XGBoost | 0.3863 | 0.3474 |
| GLMNET | 0.4706 | 0.3122 |

OOF MCC uses a threshold optimized on these same OOF predictions and is optimistic. Hyperparameters also use these folds; test is the independent evaluation.

Micro-AUPR uses trapezoidal PR area; MCC flattens all five labels. These pooled metrics do not measure exact five-label-vector correctness.

## Data and methods
Matched samples: 138; site labels: 37; groups: 35.

Train: 109 samples, 28 groups. Test: 29 samples, 7 groups.

Training sites: ADA, ADP, BSE, BSI, BSN, BSO, BSS, CGS, CWS, DGB, DGC, DGD, DGG, DGJ, GCG, GHGa, GHGb, GHH, GHJ, GSG, JJN, JJY, MGY, USJ, USO, USY, YCG, YSY

Test sites: CWM, DGY, MYS, SCH, UJG, YCD, YSD

Missing ARC: 1-32' and 4-24; excluded explicitly in sample_audit.csv.

Only SampleID, Site and five targets read from metadata; genomic IDs normalized and checked.

QC replicates BSIb→BSI and DGYb→DGY; 35 groups, not the assumed 34 complete sites.

Final microbiome features: {'BAC_P': 45, 'BAC_O': 198, 'ARC_O': 15, 'ARC_G': 32}

No metadata predictors: only BAC phylum/order and ARC order/genus percentages.

Numeric median imputation and z-score scaling fitted on training rows only. No categorical predictors.

RF/SVM/elastic-net logistic regression: balanced class weights; XGBoost: training negative/positive ratio.

KNN/NNET: per-label random oversampling on training rows only. Single-class training labels use constant prediction.

GLMNET denotes sklearn elastic-net logistic regression (saga), not the R glmnet package.

All supplied annotations remain eligible, including unidentified and off-domain taxa.

Existing binary warning targets retained without accessing source chemistry.

CV scores are tuning estimates, not nested unbiased performance estimates. Rare positives limit stability.

No additional ML feature selection. Test-only union taxa appear only in descriptive audit.

## Reproducibility
Run `python ML/train_genomic.py --seed 42 --n-iter 8` with the versions in run_config.json.
Search spaces, input hashes, assignments, feature lists, fitted preprocessing, models, thresholds and predictions are saved alongside this report.
Test comparisons are descriptive; they do not replace the model selected from CV. No causal interpretation is supported.

## Interpretation
RandomForest was selected using training CV. RandomForest has the highest held-out Micro-AUPR and RandomForest has the highest held-out MCC. The two metrics need not favor the same algorithm. The test set contains only 29 samples from seven groups, with very few positive examples for several warnings, so rankings have substantial uncertainty. These results support measurable pooled predictive signal in this holdout, not established deployment reliability.

Recorded fit warnings: 1; see fit_warnings.csv for candidate and fold details.


This reuses the previously evaluated holdout by request; no test-based tuning occurs, but it is not a new external validation cohort.

Genomic contribution analysis: [report](genomic_importance/report.md), [complete ranked taxa](genomic_importance/ranked_taxa.csv), and [Micro-AUPR importance plot](genomic_importance/micro_aupr_importance.png).

One convergence warning occurred in an unselected GLMNET candidate. Selected final fits had no recorded convergence warnings.
