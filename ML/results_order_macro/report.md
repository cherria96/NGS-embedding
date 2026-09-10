# Order microbiome-only multi-label warning prediction

CV-selected model: **GLMNET**. Selection was frozen before test evaluation.

## Final test comparison

| Model | Macro MCC | Macro AUPR | Macro ROC-AUC |
|---|---:|---:|---:|
| Random Forest | 0.5654 | 0.5750 | 0.8655 |
| KNN | 0.4412 | 0.6241 | 0.7769 |
| SVM | 0.2075 | 0.5134 | 0.8890 |
| NNET | 0.2933 | 0.4967 | 0.8898 |
| XGBoost | 0.5715 | 0.3944 | 0.8587 |
| GLMNET | 0.2983 | 0.5082 | 0.8780 |

Test metric leaders: {"Macro MCC": "XGBoost", "Macro AUPR": "KNN", "Macro ROC-AUC": "NNET"}

## Verified cohort

Matched 138 samples; 37 Site labels; 35 groups after QC replicate grouping. Train 109 samples/28 groups; test 29 samples/7 groups. Final features 213.


Training site groups: ADA, ADP, BSE, BSI, BSN, BSO, BSS, CGS, CWS, DGB, DGC, DGD, DGG, DGJ, GCG, GHGa, GHGb, GHH, GHJ, GSG, JJN, JJY, MGY, USJ, USO, USY, YCG, YSY

Test site groups: CWM, DGY, MYS, SCH, UJG, YCD, YSD

See partition_summary.csv for all five folds, sites and counts; label_distribution.csv for every label in each partition. Missing ARC samples 1-32' and 4-24 were excluded. There are 37 Site labels and 35 groups, not 34 complete sites. BSIb/BSI and DGYb/DGY stay together.

## Methods and protocol reconciliation

**feature_rule**: Strict per-sample >0.1% from O(%) sheets; preserve original percentages, mask <=0.1 to zero. Union fitted exclusively on training portion, including inner calibration. No other selection or transformation.

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
| Random Forest | 0.4433 | 0.3098 | 0.7746 |
| KNN | 0.2939 | 0.3912 | 0.6331 |
| SVM | 0.1422 | 0.1446 | 0.4918 |
| NNET | 0.3212 | 0.2667 | 0.6761 |
| XGBoost | 0.4206 | 0.2803 | 0.7342 |
| GLMNET | 0.5929 | 0.4413 | 0.7818 |

These optimized-threshold OOF scores reuse tuning data and are optimistic; they are not nested unbiased validation estimates. Selection uses fixed-0.5 MCC recorded separately. Some validation folds lack rare positives: their AUPR/ROC macro is undefined, and is never replaced by a four-label mean.

## Label consistency on held-out sites

| Model | Warning | Positives | MCC | AUPR | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Random Forest | warning_acid_base_balance | 5 | 0.517 | 0.560 | 0.800 |
| Random Forest | warning_buffer_capacity | 1 | 0.370 | 0.250 | 0.964 |
| Random Forest | warning_acid_accumulation | 1 | 0.694 | 1.000 | 1.000 |
| Random Forest | warning_ammonia_toxicity | 4 | 0.783 | 0.908 | 0.980 |
| Random Forest | warning_biogas_quality | 2 | 0.463 | 0.157 | 0.583 |
| KNN | warning_acid_base_balance | 5 | 0.000 | 0.586 | 0.500 |
| KNN | warning_buffer_capacity | 1 | 0.694 | 0.250 | 0.964 |
| KNN | warning_acid_accumulation | 1 | 1.000 | 1.000 | 1.000 |
| KNN | warning_ammonia_toxicity | 4 | 0.512 | 0.750 | 0.920 |
| KNN | warning_biogas_quality | 2 | 0.000 | 0.534 | 0.500 |
| SVM | warning_acid_base_balance | 5 | 0.000 | 0.266 | 0.742 |
| SVM | warning_buffer_capacity | 1 | 0.414 | 0.250 | 0.964 |
| SVM | warning_acid_accumulation | 1 | 0.051 | 1.000 | 1.000 |
| SVM | warning_ammonia_toxicity | 4 | 0.109 | 0.871 | 0.980 |
| SVM | warning_biogas_quality | 2 | 0.463 | 0.180 | 0.759 |
| NNET | warning_acid_base_balance | 5 | -0.086 | 0.271 | 0.700 |
| NNET | warning_buffer_capacity | 1 | 0.306 | 0.250 | 0.964 |
| NNET | warning_acid_accumulation | 1 | 0.370 | 1.000 | 1.000 |
| NNET | warning_ammonia_toxicity | 4 | 0.876 | 0.767 | 0.970 |
| NNET | warning_biogas_quality | 2 | 0.000 | 0.195 | 0.815 |
| XGBoost | warning_acid_base_balance | 5 | 0.331 | 0.357 | 0.800 |
| XGBoost | warning_buffer_capacity | 1 | 0.694 | 0.250 | 0.964 |
| XGBoost | warning_acid_accumulation | 1 | 0.694 | 0.250 | 0.964 |
| XGBoost | warning_ammonia_toxicity | 4 | 0.783 | 1.000 | 1.000 |
| XGBoost | warning_biogas_quality | 2 | 0.354 | 0.115 | 0.565 |
| GLMNET | warning_acid_base_balance | 5 | 0.236 | 0.282 | 0.733 |
| GLMNET | warning_buffer_capacity | 1 | 0.472 | 0.250 | 0.964 |
| GLMNET | warning_acid_accumulation | 1 | 0.000 | 1.000 | 1.000 |
| GLMNET | warning_ammonia_toxicity | 4 | 0.783 | 0.835 | 0.970 |
| GLMNET | warning_biogas_quality | 2 | 0.000 | 0.173 | 0.722 |

## Feature contributions

Training-fit RF impurity and XGBoost gain importances are predictive associations, not causal effects. Correlated compositional features can share importance; rankings may be unstable with this cohort. All features remain in the models. Below are the leading five per domain for presentation only; complete rankings are exported.

**RandomForest, Archaea**: Methanomicrobiales (0.0213), Methanofastidiosales (0.0207), Methanosarciniales (0.0181), Methanomassiliicoccales (0.0138), unidentified (0.0060)

**RandomForest, Bacteria**: Oscillospirales (0.1085), Bacteroidales (0.0841), Spirochaetales (0.0591), Lactobacillales (0.0577), Clostridia (0.0550)

**XGBoost, Archaea**: Methanomicrobiales (0.0249), Methanofastidiosales (0.0192), Bathyarchaeia (0.0192), Methanosarciniales (0.0096), Methanomassiliicoccales (0.0095)

**XGBoost, Bacteria**: Bacteroidales (0.0937), Oscillospirales (0.0697), Eubacteriales (0.0519), Spirochaetales (0.0513), Coriobacteriales (0.0488)

## Limits

Sampling times do not establish prospective prediction. This is contemporaneous classification. Existing zeros with missing target-source chemistry mean not flagged, not confirmed normal. The deterministic seed-42 holdout appeared in earlier project analyses: isolated in this run, but not a previously unseen external cohort. Few sites and very few rare-warning positives make all comparisons uncertain. A macro mean weights all five warnings equally but does not imply consistent performance or exact five-label-vector recovery.

Recorded fit warnings: 2; consult fit_warnings.csv for convergence limitations.

## Reproducibility

Run `python ML/train_order_macro.py --seed 42 --n-iter 8 --output-dir NEW_DIRECTORY`. See run_config.json for versions, source hashes and full search spaces. Load joblib with ML on PYTHONPATH and supply raw Order percentages; preprocessing and thresholds are inside the bundle.

## Overall interpretation

Microbiome composition alone shows uneven predictive performance across the five warnings. The training-only joint three-metric ranking selected **GLMNET**; it remains the preferred model under the prespecified rule. Held-out metric leaders: Macro MCC: XGBoost (0.5715), Macro AUPR: KNN (0.6241), Macro ROC-AUC: NNET (0.8898). No model wins all three test metrics. Test results describe trade-offs and do not change the CV-selected model.

warning_acid_base_balance: 5 test positives; MCC ranges from -0.086 to 0.517. The highest MCC is from RandomForest.

warning_buffer_capacity: 1 test positives; MCC ranges from 0.306 to 0.694. The highest MCC is from KNN.

warning_acid_accumulation: 1 test positives; MCC ranges from 0.000 to 1.000. The highest MCC is from KNN.

warning_ammonia_toxicity: 4 test positives; MCC ranges from 0.109 to 0.876. The highest MCC is from NNET.

warning_biogas_quality: 2 test positives; MCC ranges from 0.000 to 0.463. The highest MCC is from RandomForest.

The largest mean test MCC across the six models is for warning_ammonia_toxicity (0.641), and the smallest is for warning_acid_base_balance (0.166). This is descriptive only, not another model-selection criterion. Inspect both per-warning ranking areas and thresholded MCC: ranking one rare positive highly need not give a reliable threshold. Acid accumulation has very few examples, so even perfect test ranking would not establish robust generalization. The macro results do not establish dependable performance on all five warnings.

Rare-label ranking can differ substantially between training OOF and this small holdout. No model or threshold was changed in response to test results.

## Feature counts and label distribution

The final training-derived schema has 213 features: 15 Archaea, 198 Bacteria. Descriptive union counts by domain: {'Archaea': 17, 'Bacteria': 221}.

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

Implemented: five-label binary relevance; Order-only genomic predictors; strict sample-wise >0.1% mask; no additional ML feature selection; grouped approximately 80:20 holdout; five grouped training folds; fold-local preprocessing/balancing and tuning; training-only thresholds; arithmetic five-label macro metrics with individual scores; six model objects; final predictions; three figures; complete feature contributions and reproducibility artifacts.

Explicit qualifications: the mutually incompatible all-sample-union and no-test-feature-use rules are reconciled with training-only modeling unions plus a descriptive full union; GLMNET is a sklearn elastic-net analogue; some validation-fold AUPR/ROC-AUC values are mathematically undefined; the existing holdout is reused at the project level; pre-warning sampling chronology is unavailable. The analysis therefore does not claim literal satisfaction of conflicting rules or a new external validation.

Run test_order_macro.py for five protocol unit tests and verify_order_macro.py for independent verification of all 12 saved prediction-table metrics, source masks, final/fold unions and site separation, without new test predictions. See verification.json for the completed verification status.
