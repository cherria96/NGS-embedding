# Order-only joint five-label warning prediction

CV-selected compromise model: **RandomForest**, using equal average ranks across all three metrics. CV leaders: {'Multilabel MCC': ['GLMNET'], 'Micro-AUPR': ['RandomForest'], 'Micro-ROC-AUC': ['RandomForest']}. Differing leaders indicate a trade-off, not a unanimous best model.

## Final held-out performance

| Model | Multilabel MCC | Micro-AUPR | Micro-ROC-AUC |
|---|---:|---:|---:|
| RandomForest | 0.4085 | 0.5660 | 0.8722 |
| KNN | 0.2914 | 0.3806 | 0.6729 |
| SVM | 0.2102 | 0.2480 | 0.7729 |
| NNET | 0.3666 | 0.3217 | 0.7553 |
| XGBoost | 0.4064 | 0.3454 | 0.8422 |
| GLMNET | 0.3332 | 0.3881 | 0.8334 |

Descriptive test metric leaders: {'Multilabel MCC': ['RandomForest'], 'Micro-AUPR': ['RandomForest'], 'Micro-ROC-AUC': ['RandomForest']}. These test comparisons do not change the CV selection.

## Training OOF performance

| Model | Multilabel MCC | Micro-AUPR | Micro-ROC-AUC |
|---|---:|---:|---:|
| RandomForest | 0.5260 | 0.4662 | 0.8865 |
| KNN | 0.4018 | 0.4065 | 0.7875 |
| SVM | 0.3345 | 0.2721 | 0.7214 |
| NNET | 0.4129 | 0.2817 | 0.7895 |
| XGBoost | 0.4692 | 0.4434 | 0.8606 |
| GLMNET | 0.5551 | 0.4059 | 0.8706 |

Hyperparameter selection and threshold optimization reuse these OOF predictions; these are optimistic tuning estimates, not nested unbiased validation estimates.

## Data and integration

138 matched samples; 37 original site labels grouped into 35 sites. QC replicates BSIb→BSI and DGYb→DGY remain with their parent sites. The supplied data do not contain exactly 34 complete four-season sites.

Excluded records: SampleID
1-32'    missing ARC
4-24     missing ARC

No missing microbial cells were found. Missing entire ARC profiles are excluded explicitly, not filled with biological zeros. Target values are retained as supplied; zeros can reflect missing source chemistry, as documented in the root README.

Fixed Order features: 238 total; BAC 221, ARC 17. Full lists and sample-specific counts are in selected_BAC_orders.csv and selected_ARC_orders.csv.

Union is taken over matched samples according to the prespecified >0.1% rule, the only permitted use of all-sample feature presence. Test-only taxa, if any, remain fixed zero training columns. This is a schema decision explicitly requested in the protocol; no model-driven selection follows. Rank-sheet annotations, including unidentified and off-domain annotations, are retained as supplied.

Training: 110 samples / 28 sites. Test: 28 samples / 7 sites.

Training sites: ADA, ADP, BSI, BSN, BSO, BSS, CGS, CWM, CWS, DGB, DGC, DGG, DGJ, DGY, GCG, GHGa, GHH, GHJ, GSG, JJN, JJY, SCH, USJ, USO, USY, YCG, YSD, YSY

Test sites: BSE, DGD, GHGb, MGY, MYS, UJG, YCD

Sample/site/season coverage and each CV assignment are recorded in site_season_inventory.csv and split_and_cv_assignment.csv. Label distributions, including folds without rare positives, are in label_distribution.csv.

## Methods

O(%) only; each sample value <=0.1 percent becomes zero; global union across matched samples under the explicitly predefined rule. No further selection or renormalization. Same fixed columns in every model/fold.

Fold-local StandardScaler for KNN/SVM/NNET/GLMNET; identity transform for RF/XGBoost. No missing abundance cells: no imputation. Incomplete ARC profiles excluded with audit.

{'RandomForest': 'balanced class weights', 'SVM': 'balanced weights; 3-fold inner site-grouped sigmoid calibration', 'GLMNET': 'sklearn elastic-net logistic regression with saga and balanced weights (not R glmnet)', 'XGBoost': 'training negative/positive scale_pos_weight', 'KNN': 'training-only per-label random oversampling', 'NNET': 'training-only per-label random oversampling'}

Constant classifier when a training label contains only one class.

Equal average descending rank of all three pooled OOF metrics across candidates; stable candidate order breaks exact rank ties. MCC uses 0.5 during hyperparameter selection. Across final models use all three thresholded OOF metrics; report leaders/trade-offs.

One shared threshold across labels, maximizing flattened training OOF MCC over 0.01..0.99, ties closest to 0.5; fixed before test prediction.

{'Multilabel MCC': 'Binary MCC after flattening all sample-label pairs.', 'Micro-AUPR': 'Trapezoidal area under PR curve of flattened true labels and probabilities, not average precision.', 'Micro-ROC-AUC': 'ROC AUC of flattened true labels and probabilities.'}

No metadata enters predictors. See leakage_check.json for every excluded metadata column. Grouped SVM calibration, oversampling and scaling use training portions only. All six final fits, thresholds and CV selection are saved before test prediction.

## Interpretation

The CV-selected RandomForest achieved held-out MCC 0.4085, Micro-AUPR 0.5660, and Micro-ROC-AUC 0.8722. It led all three held-out metrics. This shows pooled predictive signal in this split, with imperfect binary warning prediction.

These results quantify pooled sample-label discrimination and binary association for five simultaneous outputs. They do not measure exact five-label-vector correctness, and common labels contribute more to pooled metrics. Sparse positives and few independent held-out sites limit precision; five-fold balancing cannot place a rare label in every fold when too few positive sites exist.

A fresh seeded split is generated without reading previous results or assignments. However, the same source cohort has been analyzed previously: this is an internally isolated run, not a previously unseen external validation cohort. No causal claims or deployment reliability follow from this comparison.

Recorded fitting warnings: 0; see fit_warnings.csv. GLMNET is an elastic-net logistic-regression implementation using sklearn, not the R glmnet package.

## Reproduce

`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ngs-mpl /home/best/anaconda3/envs/tsf-ad/bin/python ML/train_order.py --seed 20260909 --n-iter 8`

Inputs, hashes, package versions and complete search spaces are in run_config.json. Saved models require ML on PYTHONPATH. Predict using percentage-unit Order columns named exactly as final_feature_matrix.csv; its SampleID index is not a predictor.
