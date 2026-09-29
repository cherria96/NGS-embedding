# Phylum microbiome-only multi-label warning prediction

CV-selected model: **NNET**. Selection was frozen before test evaluation.

## Final test comparison

| Model | Macro MCC | Macro AUPR | Macro ROC-AUC |
|---|---:|---:|---:|
| Random Forest | 0.2760 | 0.4862 | 0.8072 |
| KNN | 0.3657 | 0.4197 | 0.7651 |
| SVM | 0.2555 | 0.2632 | 0.6882 |
| NNET | 0.3865 | 0.3733 | 0.7257 |
| XGBoost | 0.1955 | 0.4798 | 0.7325 |
| GLMNET | 0.4153 | 0.4551 | 0.7658 |

Test metric leaders: {"Macro MCC": "GLMNET", "Macro AUPR": "Random Forest", "Macro ROC-AUC": "Random Forest"}

## Verified cohort

Matched 138 samples; 37 Site labels; 35 groups after QC replicate grouping. Train 109 samples/28 groups; test 29 samples/7 groups. Final features 57.


Training site groups: ADA, ADP, BSE, BSI, BSN, BSO, BSS, CGS, CWS, DGB, DGC, DGD, DGG, DGJ, GCG, GHGa, GHGb, GHH, GHJ, GSG, JJN, JJY, MGY, USJ, USO, USY, YCG, YSY

Test site groups: CWM, DGY, MYS, SCH, UJG, YCD, YSD

See partition_summary.csv for all five folds, sites and counts; label_distribution.csv for every label in each partition. Missing ARC samples 1-32' and 4-24 were excluded. There are 37 Site labels and 35 groups, not 34 complete sites. BSIb/BSI and DGYb/DGY stay together.

## Methods and protocol reconciliation

**feature_rule**: Strict per-sample >0.1% from P(%) sheets; preserve original percentages, mask <=0.1 to zero. Union fitted exclusively on training portion, including inner calibration. No other selection or transformation.

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
| Random Forest | 0.3288 | 0.2518 | 0.6970 |
| KNN | 0.1980 | 0.1898 | 0.6096 |
| SVM | 0.1562 | 0.1897 | 0.4528 |
| NNET | 0.2834 | 0.2854 | 0.7122 |
| XGBoost | 0.3339 | 0.2617 | 0.7060 |
| GLMNET | 0.3895 | 0.2882 | 0.6856 |

These optimized-threshold OOF scores reuse tuning data and are optimistic; they are not nested unbiased validation estimates. Selection uses fixed-0.5 MCC recorded separately. Some validation folds lack rare positives: their AUPR/ROC macro is undefined, and is never replaced by a four-label mean.

## Label consistency on held-out sites

| Model | Warning | Positives | MCC | AUPR | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Random Forest | warning_acid_base_balance | 5 | 0.444 | 0.691 | 0.942 |
| Random Forest | warning_buffer_capacity | 1 | 0.107 | 0.042 | 0.607 |
| Random Forest | warning_acid_accumulation | 1 | 0.282 | 1.000 | 1.000 |
| Random Forest | warning_ammonia_toxicity | 4 | 0.783 | 0.544 | 0.950 |
| Random Forest | warning_biogas_quality | 2 | -0.236 | 0.154 | 0.537 |
| KNN | warning_acid_base_balance | 5 | 0.740 | 0.963 | 0.992 |
| KNN | warning_buffer_capacity | 1 | -0.064 | 0.017 | 0.446 |
| KNN | warning_acid_accumulation | 1 | 0.556 | 0.250 | 0.964 |
| KNN | warning_ammonia_toxicity | 4 | 0.648 | 0.833 | 0.960 |
| KNN | warning_biogas_quality | 2 | -0.051 | 0.034 | 0.463 |
| SVM | warning_acid_base_balance | 5 | 0.124 | 0.487 | 0.892 |
| SVM | warning_buffer_capacity | 1 | 0.370 | 0.083 | 0.821 |
| SVM | warning_acid_accumulation | 1 | 0.000 | 0.017 | 0.000 |
| SVM | warning_ammonia_toxicity | 4 | 0.783 | 0.544 | 0.950 |
| SVM | warning_biogas_quality | 2 | 0.000 | 0.184 | 0.778 |
| NNET | warning_acid_base_balance | 5 | 0.275 | 0.329 | 0.708 |
| NNET | warning_buffer_capacity | 1 | 0.000 | 0.033 | 0.500 |
| NNET | warning_acid_accumulation | 1 | 1.000 | 1.000 | 1.000 |
| NNET | warning_ammonia_toxicity | 4 | 0.709 | 0.438 | 0.920 |
| NNET | warning_biogas_quality | 2 | -0.051 | 0.066 | 0.500 |
| XGBoost | warning_acid_base_balance | 5 | 0.347 | 0.614 | 0.900 |
| XGBoost | warning_buffer_capacity | 1 | -0.117 | 0.031 | 0.464 |
| XGBoost | warning_acid_accumulation | 1 | 0.127 | 1.000 | 1.000 |
| XGBoost | warning_ammonia_toxicity | 4 | 0.709 | 0.704 | 0.965 |
| XGBoost | warning_biogas_quality | 2 | -0.089 | 0.050 | 0.333 |
| GLMNET | warning_acid_base_balance | 5 | 0.443 | 0.543 | 0.825 |
| GLMNET | warning_buffer_capacity | 1 | -0.076 | 0.033 | 0.500 |
| GLMNET | warning_acid_accumulation | 1 | 1.000 | 1.000 | 1.000 |
| GLMNET | warning_ammonia_toxicity | 4 | 0.709 | 0.626 | 0.930 |
| GLMNET | warning_biogas_quality | 2 | 0.000 | 0.074 | 0.574 |

## Feature contributions

Training-fit RF impurity and XGBoost gain importances are predictive associations, not causal effects. Correlated compositional features can share importance; rankings may be unstable with this cohort. All features remain in the models. Below are the leading five per domain for presentation only; complete rankings are exported.

**RandomForest, Archaea**: Euryarchaeota (0.0471), Halobacterota (0.0330), Thermoplasmatota (0.0272), unidentified (0.0176), Crenarchaeota (0.0128)

**RandomForest, Bacteria**: Spirochaetota (0.0733), Planctomycetota (0.0650), Cloacimonadota (0.0581), Caldatribacteriota (0.0562), Firmicutes (0.0554)

**XGBoost, Archaea**: Euryarchaeota (0.0437), Thermoplasmatota (0.0418), unidentified (0.0198), Halobacterota (0.0193), Nanoarchaeota (0.0102)

**XGBoost, Bacteria**: Bacteroidota (0.0944), Synergistota (0.0794), Spirochaetota (0.0687), Caldatribacteriota (0.0680), Thermotogota (0.0675)

## Limits

Sampling times do not establish prospective prediction. This is contemporaneous classification. Existing zeros with missing target-source chemistry mean not flagged, not confirmed normal. The deterministic seed-42 holdout appeared in earlier project analyses: isolated in this run, but not a previously unseen external cohort. Few sites and very few rare-warning positives make all comparisons uncertain. A macro mean weights all five warnings equally but does not imply consistent performance or exact five-label-vector recovery.

Recorded fit warnings: 8; consult fit_warnings.csv for convergence limitations.

## Reproducibility

Run `python ML/train_phylum_macro.py --seed 42 --n-iter 8 --output-dir NEW_DIRECTORY`. See run_config.json for versions, source hashes and full search spaces. Load joblib with ML on PYTHONPATH and supply raw Phylum percentages; preprocessing and thresholds are inside the bundle.

## Overall interpretation

Microbiome composition alone shows uneven predictive performance across the five warnings. The training-only joint three-metric ranking selected **NNET**; it remains the preferred model under the prespecified rule. Held-out metric leaders: Macro MCC: GLMNET (0.4153), Macro AUPR: Random Forest (0.4862), Macro ROC-AUC: Random Forest (0.8072). No model wins all three test metrics. Test results describe trade-offs and do not change the CV-selected model.

warning_acid_base_balance: 5 test positives; MCC ranges from 0.124 to 0.740. The highest MCC is from KNN.

warning_buffer_capacity: 1 test positives; MCC ranges from -0.117 to 0.370. The highest MCC is from SVM.

warning_acid_accumulation: 1 test positives; MCC ranges from 0.000 to 1.000. The highest MCC is from NNET.

warning_ammonia_toxicity: 4 test positives; MCC ranges from 0.648 to 0.783. The highest MCC is from RandomForest.

warning_biogas_quality: 2 test positives; MCC ranges from -0.236 to 0.000. The highest MCC is from SVM.

The largest mean test MCC across the six models is for warning_ammonia_toxicity (0.724), and the smallest is for warning_biogas_quality (-0.071). This is descriptive only, not another model-selection criterion. Inspect both per-warning ranking areas and thresholded MCC: ranking one rare positive highly need not give a reliable threshold. Acid accumulation has very few examples, so even perfect test ranking would not establish robust generalization. The macro results do not establish dependable performance on all five warnings.

Rare-label ranking can differ substantially between training OOF and this small holdout. No model or threshold was changed in response to test results.

## Feature counts and label distribution

The final training-derived schema has 57 features: 12 Archaea, 45 Bacteria. Descriptive union counts by domain: {'Archaea': 13, 'Bacteria': 50}.

| Warning | Train positives / samples | Test positives / samples |
|---|---:|---:|
| warning_acid_base_balance | 15/109 (13.76%) | 5/29 (17.24%) |
| warning_buffer_capacity | 8/109 (7.34%) | 1/29 (3.45%) |
| warning_acid_accumulation | 3/109 (2.75%) | 1/29 (3.45%) |
| warning_ammonia_toxicity | 22/109 (20.18%) | 4/29 (13.79%) |
| warning_biogas_quality | 5/109 (4.59%) | 2/29 (6.90%) |

## Convergence diagnostics

{('GLMNET', 'ConvergenceWarning'): 6, ('NNET', 'ConvergenceWarning'): 2}

0 warnings occurred during final fits. Iteration-limit warnings mean those estimators may not have converged; no post-test retuning was performed. Full messages are preserved in fit_warnings.csv.

## Requirement audit

Implemented: five-label binary relevance; Phylum-only genomic predictors; strict sample-wise >0.1% mask; no additional ML feature selection; grouped approximately 80:20 holdout; five grouped training folds; fold-local preprocessing/balancing and tuning; training-only thresholds; arithmetic five-label macro metrics with individual scores; six model objects; final predictions; three figures; complete feature contributions and reproducibility artifacts.

Explicit qualifications: the mutually incompatible all-sample-union and no-test-feature-use rules are reconciled with training-only modeling unions plus a descriptive full union; GLMNET is a sklearn elastic-net analogue; some validation-fold AUPR/ROC-AUC values are mathematically undefined; the existing holdout is reused at the project level; pre-warning sampling chronology is unavailable. The analysis therefore does not claim literal satisfaction of conflicting rules or a new external validation.

Run test_phylum_macro.py for five protocol unit tests and verify_phylum_macro.py for independent verification of all 12 saved prediction-table metrics, source masks, final/fold unions and site separation, without new test predictions. See verification.json for the completed verification status.

## Supplementary interpretation and resolution comparison
All five resolutions have identical source hashes, sample/site partitions and CV folds. This comparison uses frozen results only; no models or thresholds are changed. The models below were selected independently by training CV. The repeatedly inspected holdout is not external validation, and resolution selection based on it would introduce selection bias.
| Resolution | CV-selected model | Test Macro MCC | Test Macro AUPR | Test Macro ROC-AUC |
|---|---|---:|---:|---:|
| Family | NNET | 0.3079 | 0.5093 | 0.8216 |
| Genus | GLMNET | 0.2866 | 0.4629 | 0.8052 |
| Order | GLMNET | 0.2983 | 0.5082 | 0.8780 |
| Class | GLMNET | 0.1393 | 0.4120 | 0.8976 |
| Phylum | NNET | 0.3865 | 0.3733 | 0.7257 |

Among these CV-selected models, the held-out leaders are Macro MCC: Phylum, Macro AUPR: Family, Macro ROC-AUC: Class. These descriptive rankings do not establish superiority of a taxonomic resolution.

### Training permutation importance for all six models

Each feature was shuffled independently three times across the 109 training rows (seeds 42, 43, 44 in the default run). Importance is baseline minus shuffled macro score at frozen thresholds. No test rows were predicted or permuted. This training-resubstitution diagnostic may overstate generalizable importance; permutations can create unrealistic compositional profiles and disrupt site dependence. Repetition SD is not a confidence interval. These are predictive contributions, not statistical-significance tests or causal biological effects. No feature is removed. Complete mean/SD values for all three metrics are exported.

**GLMNET, Archaea (Phylum)**: Vertebrata (0.0972), Halobacterota (0.0618), Euryarchaeota (0.0569)

**GLMNET, Bacteria (Phylum)**: Actinobacteriota (0.3438), Spirochaetota (0.2073), Bacteroidota (0.1652)

**KNN, Archaea (Phylum)**: Thermoplasmatota (0.0010), unidentified (0.0003), Aenigmarchaeota (0.0000)

**KNN, Bacteria (Phylum)**: Halanaerobiaeota (0.0370), Acidobacteriota (0.0000), Actinobacteriota (0.0000)

**NNET, Archaea (Phylum)**: Thermoplasmatota (0.0213), Euryarchaeota (0.0206), Halobacterota (0.0129)

**NNET, Bacteria (Phylum)**: Spirochaetota (0.2849), Actinobacteriota (0.1401), Bacteroidota (0.0818)

**RandomForest, Archaea (Phylum)**: Euryarchaeota (0.0149), Halobacterota (0.0148), unidentified (0.0036)

**RandomForest, Bacteria (Phylum)**: Spirochaetota (0.0470), Caldatribacteriota (0.0243), Proteobacteria (0.0237)

**SVM, Archaea (Phylum)**: Halobacterota (0.0203), Euryarchaeota (0.0148), Micrarchaeota (0.0133)

**SVM, Bacteria (Phylum)**: Patescibacteria (0.0303), Synergistota (0.0216), Dependentiae (0.0207)

**XGBoost, Archaea (Phylum)**: Halobacterota (0.0373), Nanoarchaeota (0.0116), unidentified (0.0035)

**XGBoost, Bacteria (Phylum)**: Spirochaetota (0.1586), Caldatribacteriota (0.1521), Synergistota (0.0984)

