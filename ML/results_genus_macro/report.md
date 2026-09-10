# Genus microbiome-only multi-label warning prediction

CV-selected model: **GLMNET**. Selection was frozen before test evaluation.

## Final test comparison

| Model | Macro MCC | Macro AUPR | Macro ROC-AUC |
|---|---:|---:|---:|
| Random Forest | 0.3772 | 0.4260 | 0.9272 |
| KNN | 0.1728 | 0.3801 | 0.6023 |
| SVM | -0.0522 | 0.1914 | 0.4927 |
| NNET | 0.1188 | 0.2576 | 0.7667 |
| XGBoost | 0.0558 | 0.3498 | 0.9340 |
| GLMNET | 0.2866 | 0.4629 | 0.8052 |

Test metric leaders: {"Macro MCC": "Random Forest", "Macro AUPR": "GLMNET", "Macro ROC-AUC": "XGBoost"}

## Verified cohort

Matched 138 samples; 37 Site labels; 35 groups after QC replicate grouping. Train 109 samples/28 groups; test 29 samples/7 groups. Final features 497.


Training site groups: ADA, ADP, BSE, BSI, BSN, BSO, BSS, CGS, CWS, DGB, DGC, DGD, DGG, DGJ, GCG, GHGa, GHGb, GHH, GHJ, GSG, JJN, JJY, MGY, USJ, USO, USY, YCG, YSY

Test site groups: CWM, DGY, MYS, SCH, UJG, YCD, YSD

See partition_summary.csv for all five folds, sites and counts; label_distribution.csv for every label in each partition. Missing ARC samples 1-32' and 4-24 were excluded. There are 37 Site labels and 35 groups, not 34 complete sites. BSIb/BSI and DGYb/DGY stay together.

## Methods and protocol reconciliation

**feature_rule**: Strict per-sample >0.1% from G(%) sheets; preserve original percentages, mask <=0.1 to zero. Union fitted exclusively on training portion, including inner calibration. No other selection or transformation.

**protocol_conflict**: All-sample union conflicts with prohibition on test-informed feature construction. All-sample union exported descriptively; modeling uses training-only union.

**selection**: Equal-weight mean descending rank of pooled training OOF Macro MCC at fixed 0.5, Macro AUPR, Macro ROC-AUC. Ties favor Macro AUPR, ROC-AUC, MCC, then candidate order.

**threshold**: Per-label maximize MCC on chosen-candidate training OOF scores, grid 0.01..0.99 step .01, ties closest to .5. Frozen before test. Selection CV uses predefined .5; optimized OOF MCC is a tuning estimate.

**metrics**: Each macro is arithmetic mean of exactly five binary-label metrics. AUPR is trapezoidal precision-recall area, not average precision. MCC zero-denominator=0. AUPR/ROC-AUC undefined for single-class truth; macro propagates NaN, never skips labels. Pooled OOF is used for tuning; single-class validation folds remain recorded.

**preprocessing**: Training-only median imputation (empty fallback zero); z-score KNN/SVM/NNET/GLMNET, trees unscaled. Abundances validated finite; missing entire domains excluded, never zero-imputed.

**imbalance**: RF/SVM/GLMNET balanced class weights; XGBoost train negative/positive ratio; KNN/NNET per-label random oversampling to majority count with seed+j, training only after preprocessing. Single-class training uses constant estimator.

**SVM**: Inner three-fold GroupKFold OOF margin logistic calibration with refitted feature union/scaling.

**GLMNET**: sklearn saga elastic-net logistic regression analogue, not the R glmnet implementation; coefficients regularized without subsequent feature removal.

## Training OOF tuning estimates

| Model | Macro MCC | Macro AUPR | Macro ROC-AUC |
|---|---:|---:|---:|
| Random Forest | 0.2539 | 0.2016 | 0.5610 |
| KNN | 0.2342 | 0.1823 | 0.6267 |
| SVM | 0.1828 | 0.1634 | 0.4218 |
| NNET | 0.2054 | 0.2069 | 0.5326 |
| XGBoost | 0.2570 | 0.2109 | 0.5624 |
| GLMNET | 0.3128 | 0.2421 | 0.5535 |

These optimized-threshold OOF scores reuse tuning data and are optimistic; they are not nested unbiased validation estimates. Selection uses fixed-0.5 MCC recorded separately. Some validation folds lack rare positives: their AUPR/ROC macro is undefined, and is never replaced by a four-label mean.

## Label consistency on held-out sites

| Model | Warning | Positives | MCC | AUPR | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Random Forest | warning_acid_base_balance | 5 | 0.668 | 0.751 | 0.958 |
| Random Forest | warning_buffer_capacity | 1 | 0.159 | 0.167 | 0.929 |
| Random Forest | warning_acid_accumulation | 1 | 0.000 | 0.250 | 0.964 |
| Random Forest | warning_ammonia_toxicity | 4 | 0.876 | 0.767 | 0.970 |
| Random Forest | warning_biogas_quality | 2 | 0.183 | 0.195 | 0.815 |
| KNN | warning_acid_base_balance | 5 | 0.517 | 0.658 | 0.900 |
| KNN | warning_buffer_capacity | 1 | -0.036 | 0.017 | 0.429 |
| KNN | warning_acid_accumulation | 1 | 0.000 | 0.517 | 0.500 |
| KNN | warning_ammonia_toxicity | 4 | 0.475 | 0.673 | 0.905 |
| KNN | warning_biogas_quality | 2 | -0.092 | 0.034 | 0.278 |
| SVM | warning_acid_base_balance | 5 | 0.000 | 0.269 | 0.542 |
| SVM | warning_buffer_capacity | 1 | -0.370 | 0.021 | 0.179 |
| SVM | warning_acid_accumulation | 1 | 0.000 | 0.018 | 0.036 |
| SVM | warning_ammonia_toxicity | 4 | 0.109 | 0.501 | 0.930 |
| SVM | warning_biogas_quality | 2 | 0.000 | 0.148 | 0.778 |
| NNET | warning_acid_base_balance | 5 | -0.086 | 0.266 | 0.542 |
| NNET | warning_buffer_capacity | 1 | -0.051 | 0.038 | 0.571 |
| NNET | warning_acid_accumulation | 1 | 0.000 | 0.250 | 0.964 |
| NNET | warning_ammonia_toxicity | 4 | 0.783 | 0.579 | 0.960 |
| NNET | warning_biogas_quality | 2 | -0.051 | 0.154 | 0.796 |
| XGBoost | warning_acid_base_balance | 5 | -0.086 | 0.391 | 0.858 |
| XGBoost | warning_buffer_capacity | 1 | 0.000 | 0.250 | 0.964 |
| XGBoost | warning_acid_accumulation | 1 | 0.000 | 0.250 | 0.964 |
| XGBoost | warning_ammonia_toxicity | 4 | -0.076 | 0.442 | 0.920 |
| XGBoost | warning_biogas_quality | 2 | 0.441 | 0.417 | 0.963 |
| GLMNET | warning_acid_base_balance | 5 | 0.236 | 0.571 | 0.875 |
| GLMNET | warning_buffer_capacity | 1 | 0.000 | 0.038 | 0.571 |
| GLMNET | warning_acid_accumulation | 1 | 0.414 | 1.000 | 1.000 |
| GLMNET | warning_ammonia_toxicity | 4 | 0.783 | 0.544 | 0.950 |
| GLMNET | warning_biogas_quality | 2 | 0.000 | 0.161 | 0.630 |

## Feature contributions

Training-fit RF impurity and XGBoost gain importances are predictive associations, not causal effects. Correlated compositional features can share importance; rankings may be unstable with this cohort. All features remain in the models. Below are the leading five per domain for presentation only; complete rankings are exported.

**RandomForest, Archaea**: Candidatus_Methanoplasma (0.0203), Methanosarcina (0.0160), Methanospirillum (0.0149), Candidatus_Methanofastidiosum (0.0141), Methanosaeta (0.0137)

**RandomForest, Bacteria**: Aminobacterium (0.0179), Fermentimonas (0.0178), Fastidiosipila (0.0163), Caldicoprobacter (0.0151), Bacteroidetes_vadinHA17 (0.0149)

**XGBoost, Archaea**: Methanosaeta (0.0315), Methanosarcina (0.0280), Methanospirillum (0.0213), Methanomassiliicoccus (0.0211), RumEn_M2 (0.0208)

**XGBoost, Bacteria**: Pseudomonas (0.0461), Olsenella (0.0425), Fermentimonas (0.0378), Proteiniphilum (0.0313), Bacteroidetes_vadinHA17 (0.0308)

## Limits

Sampling times do not establish prospective prediction. This is contemporaneous classification. Existing zeros with missing target-source chemistry mean not flagged, not confirmed normal. The deterministic seed-42 holdout appeared in earlier project analyses: isolated in this run, but not a previously unseen external cohort. Few sites and very few rare-warning positives make all comparisons uncertain. A macro mean weights all five warnings equally but does not imply consistent performance or exact five-label-vector recovery.

Recorded fit warnings: 6; consult fit_warnings.csv for convergence limitations.

## Reproducibility

Run `python ML/train_genus_macro.py --seed 42 --n-iter 8 --output-dir NEW_DIRECTORY`. See run_config.json for versions, source hashes and full search spaces. Load joblib with ML on PYTHONPATH and supply raw Genus percentages; preprocessing and thresholds are inside the bundle.

## Overall interpretation

Microbiome composition alone shows uneven predictive performance across the five warnings. The training-only joint three-metric ranking selected **GLMNET**; it remains the preferred model under the prespecified rule. Held-out metric leaders: Macro MCC: Random Forest (0.3772), Macro AUPR: GLMNET (0.4629), Macro ROC-AUC: XGBoost (0.9340). No model wins all three test metrics. Test results describe trade-offs and do not change the CV-selected model.

warning_acid_base_balance: 5 test positives; MCC ranges from -0.086 to 0.668. The highest MCC is from RandomForest.

warning_buffer_capacity: 1 test positives; MCC ranges from -0.370 to 0.159. The highest MCC is from RandomForest.

warning_acid_accumulation: 1 test positives; MCC ranges from 0.000 to 0.414. The highest MCC is from GLMNET.

warning_ammonia_toxicity: 4 test positives; MCC ranges from -0.076 to 0.876. The highest MCC is from RandomForest.

warning_biogas_quality: 2 test positives; MCC ranges from -0.092 to 0.441. The highest MCC is from XGBoost.

The largest mean test MCC across the six models is for warning_ammonia_toxicity (0.492), and the smallest is for warning_buffer_capacity (-0.050). This is descriptive only, not another model-selection criterion. Inspect both per-warning ranking areas and thresholded MCC: ranking one rare positive highly need not give a reliable threshold. Acid accumulation has very few examples, so even perfect test ranking would not establish robust generalization. The macro results do not establish dependable performance on all five warnings.

Rare-label ranking can differ substantially between training OOF and this small holdout. No model or threshold was changed in response to test results.

## Feature counts and label distribution

The final training-derived schema has 497 features: 32 Archaea, 465 Bacteria. Descriptive union counts by domain: {'Archaea': 34, 'Bacteria': 540}.

| Warning | Train positives / samples | Test positives / samples |
|---|---:|---:|
| warning_acid_base_balance | 15/109 (13.76%) | 5/29 (17.24%) |
| warning_buffer_capacity | 8/109 (7.34%) | 1/29 (3.45%) |
| warning_acid_accumulation | 3/109 (2.75%) | 1/29 (3.45%) |
| warning_ammonia_toxicity | 22/109 (20.18%) | 4/29 (13.79%) |
| warning_biogas_quality | 5/109 (4.59%) | 2/29 (6.90%) |

## Convergence diagnostics

{('GLMNET', 'ConvergenceWarning'): 6}

1 warnings occurred during final fits. Iteration-limit warnings mean those estimators may not have converged; no post-test retuning was performed. Full messages are preserved in fit_warnings.csv.

## Requirement audit

Implemented: five-label binary relevance; Genus-only genomic predictors; strict sample-wise >0.1% mask; no additional ML feature selection; grouped approximately 80:20 holdout; five grouped training folds; fold-local preprocessing/balancing and tuning; training-only thresholds; arithmetic five-label macro metrics with individual scores; six model objects; final predictions; three figures; complete feature contributions and reproducibility artifacts.

Explicit qualifications: the mutually incompatible all-sample-union and no-test-feature-use rules are reconciled with training-only modeling unions plus a descriptive full union; GLMNET is a sklearn elastic-net analogue; some validation-fold AUPR/ROC-AUC values are mathematically undefined; the existing holdout is reused at the project level; pre-warning sampling chronology is unavailable. The analysis therefore does not claim literal satisfaction of conflicting rules or a new external validation.

Run test_genus_macro.py for five protocol unit tests and verify_genus_macro.py for independent verification of all 12 saved prediction-table metrics, source masks, final/fold unions and site separation, without new test predictions. See verification.json for the completed verification status.
