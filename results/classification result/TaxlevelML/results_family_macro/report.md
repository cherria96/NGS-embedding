# Family microbiome-only multi-label warning prediction

CV-selected model: **NNET**. Selection was frozen before test evaluation.

## Final test comparison

| Model | Macro MCC | Macro AUPR | Macro ROC-AUC |
|---|---:|---:|---:|
| Random Forest | 0.2786 | 0.6352 | 0.9095 |
| KNN | 0.0846 | 0.2819 | 0.6967 |
| SVM | 0.2131 | 0.4631 | 0.5429 |
| NNET | 0.3079 | 0.5093 | 0.8216 |
| XGBoost | 0.1631 | 0.5788 | 0.9001 |
| GLMNET | 0.3663 | 0.5316 | 0.8507 |

Test metric leaders: {"Macro MCC": "GLMNET", "Macro AUPR": "Random Forest", "Macro ROC-AUC": "Random Forest"}

## Verified cohort

Matched 138 samples; 37 Site labels; 35 groups after QC replicate grouping. Train 109 samples/28 groups; test 29 samples/7 groups. Final features 329.


Training site groups: ADA, ADP, BSE, BSI, BSN, BSO, BSS, CGS, CWS, DGB, DGC, DGD, DGG, DGJ, GCG, GHGa, GHGb, GHH, GHJ, GSG, JJN, JJY, MGY, USJ, USO, USY, YCG, YSY

Test site groups: CWM, DGY, MYS, SCH, UJG, YCD, YSD

See partition_summary.csv for all five folds, sites and counts; label_distribution.csv for every label in each partition. Missing ARC samples 1-32' and 4-24 were excluded. There are 37 Site labels and 35 groups, not 34 complete sites. BSIb/BSI and DGYb/DGY stay together.

## Methods and protocol reconciliation

**feature_rule**: Strict per-sample >0.1% from F(%) sheets; preserve original percentages, mask <=0.1 to zero. Union fitted exclusively on training portion, including inner calibration. No other selection or transformation.

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
| Random Forest | 0.2538 | 0.2182 | 0.6173 |
| KNN | 0.2397 | 0.2124 | 0.6457 |
| SVM | 0.1423 | 0.1529 | 0.4186 |
| NNET | 0.3339 | 0.3022 | 0.5668 |
| XGBoost | 0.3207 | 0.2383 | 0.6059 |
| GLMNET | 0.3392 | 0.2829 | 0.5387 |

These optimized-threshold OOF scores reuse tuning data and are optimistic; they are not nested unbiased validation estimates. Selection uses fixed-0.5 MCC recorded separately. Some validation folds lack rare positives: their AUPR/ROC macro is undefined, and is never replaced by a four-label mean.

## Label consistency on held-out sites

| Model | Warning | Positives | MCC | AUPR | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Random Forest | warning_acid_base_balance | 5 | 0.347 | 0.523 | 0.925 |
| Random Forest | warning_buffer_capacity | 1 | 0.170 | 0.167 | 0.929 |
| Random Forest | warning_acid_accumulation | 1 | 0.000 | 1.000 | 1.000 |
| Random Forest | warning_ammonia_toxicity | 4 | 0.783 | 0.944 | 0.990 |
| Random Forest | warning_biogas_quality | 2 | 0.092 | 0.542 | 0.704 |
| KNN | warning_acid_base_balance | 5 | -0.086 | 0.398 | 0.742 |
| KNN | warning_buffer_capacity | 1 | -0.036 | 0.017 | 0.393 |
| KNN | warning_acid_accumulation | 1 | 0.000 | 0.250 | 0.964 |
| KNN | warning_ammonia_toxicity | 4 | 0.596 | 0.710 | 0.940 |
| KNN | warning_biogas_quality | 2 | -0.051 | 0.034 | 0.444 |
| SVM | warning_acid_base_balance | 5 | 0.414 | 0.798 | 0.933 |
| SVM | warning_buffer_capacity | 1 | -0.225 | 0.019 | 0.071 |
| SVM | warning_acid_accumulation | 1 | 0.000 | 0.019 | 0.071 |
| SVM | warning_ammonia_toxicity | 4 | 0.876 | 0.944 | 0.990 |
| SVM | warning_biogas_quality | 2 | 0.000 | 0.536 | 0.648 |
| NNET | warning_acid_base_balance | 5 | 0.236 | 0.309 | 0.683 |
| NNET | warning_buffer_capacity | 1 | 0.000 | 1.000 | 1.000 |
| NNET | warning_acid_accumulation | 1 | -0.036 | 0.033 | 0.500 |
| NNET | warning_ammonia_toxicity | 4 | 0.876 | 0.871 | 0.980 |
| NNET | warning_biogas_quality | 2 | 0.463 | 0.333 | 0.944 |
| XGBoost | warning_acid_base_balance | 5 | -0.086 | 0.279 | 0.767 |
| XGBoost | warning_buffer_capacity | 1 | 0.107 | 0.125 | 0.893 |
| XGBoost | warning_acid_accumulation | 1 | 0.086 | 1.000 | 1.000 |
| XGBoost | warning_ammonia_toxicity | 4 | 0.709 | 0.908 | 0.980 |
| XGBoost | warning_biogas_quality | 2 | 0.000 | 0.581 | 0.861 |
| GLMNET | warning_acid_base_balance | 5 | 0.145 | 0.517 | 0.858 |
| GLMNET | warning_buffer_capacity | 1 | 0.260 | 0.071 | 0.786 |
| GLMNET | warning_acid_accumulation | 1 | 0.694 | 1.000 | 1.000 |
| GLMNET | warning_ammonia_toxicity | 4 | 0.783 | 0.908 | 0.980 |
| GLMNET | warning_biogas_quality | 2 | -0.051 | 0.161 | 0.630 |

## Feature contributions

Training-fit RF impurity and XGBoost gain importances are predictive associations, not causal effects. Correlated compositional features can share importance; rankings may be unstable with this cohort. All features remain in the models. Below are the leading five per domain for presentation only; complete rankings are exported.

**RandomForest, Archaea**: Methanosarcinaceae (0.0156), Methanofastidiosaceae (0.0147), Methanomethylophilaceae (0.0140), Methanoregulaceae (0.0130), Methanospirillaceae (0.0119)

**RandomForest, Bacteria**: Cloacimonadaceae (0.0275), Peptostreptococcales-Tissierellales (0.0259), Hungateiclostridiaceae (0.0247), Oscillospiraceae (0.0244), Synergistaceae (0.0220)

**XGBoost, Archaea**: Methanosarcinaceae (0.0229), Methanospirillaceae (0.0181), Methanosaetaceae (0.0168), Methanomethylophilaceae (0.0151), Methanocorpusculaceae (0.0125)

**XGBoost, Bacteria**: Cloacimonadaceae (0.0485), Synergistaceae (0.0395), Pseudomonadaceae (0.0381), Spirochaetaceae (0.0364), Peptostreptococcales-Tissierellales (0.0362)

## Limits

Sampling times do not establish prospective prediction. This is contemporaneous classification. Existing zeros with missing target-source chemistry mean not flagged, not confirmed normal. The deterministic seed-42 holdout appeared in earlier project analyses: isolated in this run, but not a previously unseen external cohort. Few sites and very few rare-warning positives make all comparisons uncertain. A macro mean weights all five warnings equally but does not imply consistent performance or exact five-label-vector recovery.

Recorded fit warnings: 2; consult fit_warnings.csv for convergence limitations.

## Reproducibility

Run `python ML/train_family_macro.py --seed 42 --n-iter 8 --output-dir NEW_DIRECTORY`. See run_config.json for versions, source hashes and full search spaces. Load joblib with ML on PYTHONPATH and supply raw Family percentages; preprocessing and thresholds are inside the bundle.

## Overall interpretation

Microbiome composition alone shows uneven predictive performance across the five warnings. The training-only joint three-metric ranking selected **NNET**; it remains the preferred model under the prespecified rule. Held-out metric leaders: Macro MCC: GLMNET (0.3663), Macro AUPR: Random Forest (0.6352), Macro ROC-AUC: Random Forest (0.9095). No model wins all three test metrics. Test results describe trade-offs and do not change the CV-selected model.

warning_acid_base_balance: 5 test positives; MCC ranges from -0.086 to 0.414. The highest MCC is from SVM.

warning_buffer_capacity: 1 test positives; MCC ranges from -0.225 to 0.260. The highest MCC is from GLMNET.

warning_acid_accumulation: 1 test positives; MCC ranges from -0.036 to 0.694. The highest MCC is from GLMNET.

warning_ammonia_toxicity: 4 test positives; MCC ranges from 0.596 to 0.876. The highest MCC is from SVM.

warning_biogas_quality: 2 test positives; MCC ranges from -0.051 to 0.463. The highest MCC is from NNET.

Ammonia toxicity is comparatively consistent in this run; the other warnings vary much more across models. Inspect both the per-warning ranking areas and thresholded MCC: ranking one rare positive highly need not give a reliable threshold. In particular, acid accumulation has very few examples, so even perfect test ranking is not evidence of robust generalization. The macro results do not establish dependable performance on all five warnings.

Rare-label ranking can differ substantially between training OOF and this small holdout. No model or threshold was changed in response to test results.

## Feature counts and label distribution

The final training-derived schema has 329 features: 22 Archaea, 307 Bacteria. Descriptive union counts by domain: {'Archaea': 24, 'Bacteria': 346}.

| Warning | Train positives / samples | Test positives / samples |
|---|---:|---:|
| warning_acid_base_balance | 15/109 (13.76%) | 5/29 (17.24%) |
| warning_buffer_capacity | 8/109 (7.34%) | 1/29 (3.45%) |
| warning_acid_accumulation | 3/109 (2.75%) | 1/29 (3.45%) |
| warning_ammonia_toxicity | 22/109 (20.18%) | 4/29 (13.79%) |
| warning_biogas_quality | 5/109 (4.59%) | 2/29 (6.90%) |

## Convergence diagnostics

{('GLMNET', 'ConvergenceWarning'): 2}

0 warnings occurred during final fits. Iteration-limit warnings mean those estimators may not have converged; no post-test retuning was performed. Full messages are preserved in fit_warnings.csv.

## Requirement audit

Implemented: five-label binary relevance; Family-only genomic predictors; strict sample-wise >0.1% mask; no additional ML feature selection; grouped approximately 80:20 holdout; five grouped training folds; fold-local preprocessing/balancing and tuning; training-only thresholds; arithmetic five-label macro metrics with individual scores; six model objects; final predictions; three figures; complete feature contributions and reproducibility artifacts.

Explicit qualifications: the mutually incompatible all-sample-union and no-test-feature-use rules are reconciled with training-only modeling unions plus a descriptive full union; GLMNET is a sklearn elastic-net analogue; some validation-fold AUPR/ROC-AUC values are mathematically undefined; the existing holdout is reused at the project level; pre-warning sampling chronology is unavailable. The analysis therefore does not claim literal satisfaction of conflicting rules or a new external validation.

Run test_family_macro.py for five protocol unit tests and verify_family_macro.py for independent verification of all 12 saved prediction-table metrics, source masks, final/fold unions and site separation, without new test predictions. See verification.json for the completed verification status.
