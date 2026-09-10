# Phylum-only multi-label warning prediction

Five warnings are predicted jointly using binary relevance and microbiome predictors only. CV selected **RandomForest** using the equal-weight mean rank across the three metrics.

## Final test performance

| Model        |   Multilabel MCC |   Micro-AUPR |   Micro-ROC-AUC |
|:-------------|-----------------:|-------------:|----------------:|
| RandomForest |           0.4571 |       0.5022 |          0.8730 |
| KNN          |           0.5568 |       0.5811 |          0.8467 |
| SVM          |           0.3798 |       0.4097 |          0.7867 |
| NNET         |           0.3263 |       0.2798 |          0.7920 |
| XGBoost      |           0.3878 |       0.4418 |          0.8459 |
| GLMNET       |           0.4056 |       0.4605 |          0.8590 |

Test metric leaders: {'Multilabel MCC': 'KNN', 'Micro-AUPR': 'KNN', 'Micro-ROC-AUC': 'RandomForest'}. These are descriptive results; they do not change the CV-selected model.

A unanimous winner exists only when all three metrics favor the same model. Otherwise the rankings reflect a trade-off, not a single best model on every criterion.

## Training cross-validation

| Model        |   Multilabel MCC |   Micro-AUPR |   Micro-ROC-AUC |
|:-------------|-----------------:|-------------:|----------------:|
| RandomForest |           0.5247 |       0.4498 |          0.8148 |
| KNN          |           0.3261 |       0.3068 |          0.7310 |
| SVM          |           0.4185 |       0.3443 |          0.7551 |
| NNET         |           0.3287 |       0.3087 |          0.7433 |
| XGBoost      |           0.4451 |       0.4142 |          0.8175 |
| GLMNET       |           0.3635 |       0.2637 |          0.7669 |

CV metric leaders: {'Multilabel MCC': 'RandomForest', 'Micro-AUPR': 'RandomForest', 'Micro-ROC-AUC': 'XGBoost'}.

These are pooled out-of-fold tuning estimates, not nested unbiased estimates. Hyperparameter selection and shared-threshold optimization use these same predictions, so CV scores, especially MCC, are optimistic.

## Data integration and identity audit

metadata.csv has 140 rows and 135 columns. BAC P(%) has 60 annotation rows and 140 samples; ARC P(%) has 15 annotation rows and 138 samples. Workbook sample suffixes are removed by retaining the first two hyphen-separated components; b and apostrophe markers remain intact. All normalized workbook IDs match metadata. No duplicate or missing SampleID/Site IDs were found; unique SampleID also guarantees unique Site/SampleID pairs.

Samples 1-32' (JJY, Summer) and 4-24 (GHGa, Spring) lack ARC profiles and are explicitly excluded, not filled with zero. See sample_audit.csv and sample_id_mapping.csv. The joint cohort contains 138 samples, 37 Site codes, and 35 conservative groups after joining BSIb with BSI and DGYb with DGY. The supplied files do not support an assumption of exactly 34 complete four-season sites.

ADP and YCG lack Spring metadata; JJY lacks Summer ARC and GHGa lacks Spring ARC. BSIb and DGYb are Summer-only QC records grouped with their parent sites. season_audit.csv lists every retained site/season/sample. GHGa/GHGb retain distinct metadata Site codes; facility-level identity would require source confirmation before treating this as facility-external validation.

## Split and label prevalence

Seed 42 creates an approximately 80:20 site-grouped split; seed 43 assigns five training CV folds. A 5,000-candidate random allocation minimizes label/sample imbalance while preserving groups. Labels are used only for initial stratification, not to select a split based on model performance. All six models use identical assignments.

| split   |   samples |   site_groups |
|:--------|----------:|--------------:|
| test    |        29 |             7 |
| train   |       109 |            28 |

Training groups: ADA, ADP, BSE, BSI, BSN, BSO, BSS, CGS, CWS, DGB, DGC, DGD, DGG, DGJ, GCG, GHGa, GHGb, GHH, GHJ, GSG, JJN, JJY, MGY, USJ, USO, USY, YCG, YSY

Test groups: CWM, DGY, MYS, SCH, UJG, YCD, YSD

| scope   | target                    |   samples |   positive |
|:--------|:--------------------------|----------:|-----------:|
| train   | warning_acid_base_balance |       109 |         15 |
| train   | warning_buffer_capacity   |       109 |          8 |
| train   | warning_acid_accumulation |       109 |          3 |
| train   | warning_ammonia_toxicity  |       109 |         22 |
| train   | warning_biogas_quality    |       109 |          5 |
| test    | warning_acid_base_balance |        29 |          5 |
| test    | warning_buffer_capacity   |        29 |          1 |
| test    | warning_acid_accumulation |        29 |          1 |
| test    | warning_ammonia_toxicity  |        29 |          4 |
| test    | warning_biogas_quality    |        29 |          2 |

Rare positives cannot populate every validation fold. This limits stability despite grouped stratification. split_and_cv_assignment.csv lists sample IDs, Site, SiteGroup, Season, split, and validation fold.

## Feature construction

Use only BAC P(%) and ARC P(%), in percentage units. Retain a value only when strictly >0.1 in that sample; otherwise set it to zero. Do not renormalize. The union across the 138 matched samples defines 50 BAC-source and 13 ARC-source columns, 63 total, retained in every fold. This is the explicitly allowed predefined feature-space rule; no global means, Top-N, or learned feature selection are used. Six columns are zero throughout the training set and are retained under that rule.

Domain prefixes denote workbook origin. Supplied off-domain/unidentified annotations remain included under the requested abundance-only criterion. ARC includes Cryptomycota, Holozoa, Mollusca and Vertebrata annotations; these must not be described as biologically confirmed archaeal phyla. No unrequested taxonomy cleanup was applied.

Complete selected annotations and sample counts:

| block   | taxon                         |   samples_above_0_1_percent |   training_samples_above_0_1_percent |
|:--------|:------------------------------|----------------------------:|-------------------------------------:|
| BAC_P   | Acetothermia                  |                          12 |                                    4 |
| BAC_P   | Acidobacteriota               |                          77 |                                   56 |
| BAC_P   | Actinobacteriota              |                         123 |                                   95 |
| BAC_P   | Armatimonadota                |                          68 |                                   50 |
| BAC_P   | Bacteroidota                  |                         138 |                                  109 |
| BAC_P   | Bdellovibrionota              |                          22 |                                   16 |
| BAC_P   | CK-2C2-2                      |                           9 |                                    7 |
| BAC_P   | Caldatribacteriota            |                         108 |                                   84 |
| BAC_P   | Caldisericota                 |                          84 |                                   64 |
| BAC_P   | Calditrichota                 |                           1 |                                    0 |
| BAC_P   | Campilobacterota              |                          26 |                                   16 |
| BAC_P   | Chloroflexi                   |                         100 |                                   77 |
| BAC_P   | Cloacimonadota                |                         133 |                                  104 |
| BAC_P   | Coprothermobacterota          |                           8 |                                    8 |
| BAC_P   | Cyanobacteria                 |                           2 |                                    2 |
| BAC_P   | Dadabacteria                  |                           3 |                                    0 |
| BAC_P   | Deinococcota                  |                           6 |                                    3 |
| BAC_P   | Dependentiae                  |                          34 |                                   25 |
| BAC_P   | Desulfobacterota              |                         111 |                                   82 |
| BAC_P   | Elusimicrobiota               |                           8 |                                    7 |
| BAC_P   | Fermentibacterota             |                          16 |                                   11 |
| BAC_P   | Fibrobacterota                |                          16 |                                   13 |
| BAC_P   | Firmicutes                    |                         138 |                                  109 |
| BAC_P   | Fusobacteriota                |                          21 |                                   14 |
| BAC_P   | Gemmatimonadota               |                          19 |                                   13 |
| BAC_P   | Halanaerobiaeota              |                           2 |                                    2 |
| BAC_P   | Hydrogenedentes               |                          65 |                                   49 |
| BAC_P   | Hydrothermae                  |                           1 |                                    1 |
| BAC_P   | LCP-89                        |                           1 |                                    0 |
| BAC_P   | Latescibacterota              |                           3 |                                    1 |
| BAC_P   | MBNT15                        |                           1 |                                    0 |
| BAC_P   | Marinimicrobia_(SAR406_clade) |                          30 |                                   24 |
| BAC_P   | Myxococcota                   |                          19 |                                   12 |
| BAC_P   | NB1-j                         |                           2 |                                    0 |
| BAC_P   | NKB15                         |                           6 |                                    6 |
| BAC_P   | Nitrospirota                  |                          15 |                                    7 |
| BAC_P   | Patescibacteria               |                          76 |                                   58 |
| BAC_P   | Planctomycetota               |                          92 |                                   71 |
| BAC_P   | Proteobacteria                |                         121 |                                   92 |
| BAC_P   | Spirochaetota                 |                         129 |                                  101 |
| BAC_P   | Sumerlaeota                   |                          22 |                                   15 |
| BAC_P   | Sva0485                       |                           8 |                                    2 |
| BAC_P   | Synergistota                  |                         136 |                                  107 |
| BAC_P   | TA06                          |                           8 |                                    4 |
| BAC_P   | Thermotogota                  |                         105 |                                   85 |
| BAC_P   | Verrucomicrobiota             |                         130 |                                  101 |
| BAC_P   | WPS-2                         |                           4 |                                    3 |
| BAC_P   | WS1                           |                          47 |                                   37 |
| BAC_P   | Zixibacteria                  |                           5 |                                    3 |
| BAC_P   | unidentified                  |                          84 |                                   65 |
| ARC_P   | Aenigmarchaeota               |                           2 |                                    2 |
| ARC_P   | Asgardarchaeota               |                           4 |                                    2 |
| ARC_P   | Crenarchaeota                 |                          76 |                                   58 |
| ARC_P   | Cryptomycota                  |                           1 |                                    0 |
| ARC_P   | Euryarchaeota                 |                         127 |                                   98 |
| ARC_P   | Halobacterota                 |                         138 |                                  109 |
| ARC_P   | Holozoa                       |                           1 |                                    1 |
| ARC_P   | Micrarchaeota                 |                          18 |                                   13 |
| ARC_P   | Mollusca                      |                           2 |                                    2 |
| ARC_P   | Nanoarchaeota                 |                          52 |                                   40 |
| ARC_P   | Thermoplasmatota              |                         132 |                                  104 |
| ARC_P   | Vertebrata                    |                           1 |                                    1 |
| ARC_P   | unidentified                  |                         112 |                                   91 |

## Pipeline, tuning and thresholds

Binary relevance fits five classifiers internally; all evaluation pools their outputs. Eight seeded random hyperparameter candidates per model are tested in five grouped folds: 48 configurations and 240 outer-fold multi-label fits. Search spaces and versions are in run_config.json; all candidate/fold scores are in hyperparameter_tuning.csv. Best settings are in each model’s best_params.json.

KNN, SVM, NNET and GLMNET use fold-local standardization. RF and XGBoost are unscaled. All abundance values are finite and nonnegative; no missing microbiome cells required imputation. Fold-local median imputation remains in the saved pipeline as a safeguard. Missing whole ARC profiles are excluded as above.

RF/SVM/GLMNET use balanced class weights; XGBoost uses the training-label negative/positive ratio. KNN and NNET use per-label random oversampling only on fold training rows. Single-class training labels use constant predictions. SVM probabilities use three-fold site-grouped inner out-of-fold sigmoid calibration with preprocessing refitted in each inner fold.

GLMNET is implemented as elastic-net logistic regression using sklearn saga, not the R glmnet package. NNET uses sklearn MLP. Model objects and preprocessing objects are saved as joblib files.

Candidate and model selection use equal-weight mean rank over all three pooled CV metrics; deterministic ties retain search/model order. Candidate MCC uses a shared threshold optimized on its OOF predictions. The final shared threshold maximizes flattened MCC over 0.01–0.99, with ties closest to 0.5. A shared threshold limits extra fitting with very few positives; each of the five labels receives the same fixed threshold for a given model.

| Model        |   warning_acid_base_balance |   warning_buffer_capacity |   warning_acid_accumulation |   warning_ammonia_toxicity |   warning_biogas_quality |
|:-------------|----------------------------:|--------------------------:|----------------------------:|---------------------------:|-------------------------:|
| RandomForest |                      0.3200 |                    0.3200 |                      0.3200 |                     0.3200 |                   0.3200 |
| KNN          |                      0.5700 |                    0.5700 |                      0.5700 |                     0.5700 |                   0.5700 |
| SVM          |                      0.3800 |                    0.3800 |                      0.3800 |                     0.3800 |                   0.3800 |
| NNET         |                      0.0500 |                    0.0500 |                      0.0500 |                     0.0500 |                   0.0500 |
| XGBoost      |                      0.6000 |                    0.6000 |                      0.6000 |                     0.6000 |                   0.6000 |
| GLMNET       |                      0.3300 |                    0.3300 |                      0.3300 |                     0.3300 |                   0.3300 |

6 fitting warnings were recorded; see fit_warnings.csv. Warnings include convergence limits; convergence is not guaranteed for those fits.

## Metric definitions and isolation

Multilabel MCC = binary MCC calculated after flattening all sample-label pairs across the five warning labels. Micro-AUPR = trapezoidal area under the precision–recall curve of flattened labels and probabilities, not average precision. Micro-ROC-AUC = ROC-AUC of flattened labels and probabilities. Only MCC uses thresholded predictions. These pooled metrics do not measure exact five-label-vector correctness.

Every metadata column is excluded from predictors, including all effluent warning-definition measurements, substrate chemistry, operation, Season, Site and identifiers. The full excluded list and genomic whitelist are in leakage_check.json. Identifiers and Season appear only in integration/audit/prediction outputs. Feature matrix SampleID is an index, not a predictor.

The test set was not used for fitting, preprocessing fitting, tuning, thresholds, or model selection in this run. Models and thresholds were saved before test evaluation. The fixed sample-specific union is the sole permitted feature-space exception. This deterministic split coincides with a previously evaluated repository holdout: it is not historically untouched or external validation.

## Interpretation

The results describe pooled prediction across five warning conditions from microbiome composition alone. The small test cohort (29 samples from seven groups), rare positives, prior use of this holdout, and imperfect supplied taxonomy limit generalization claims. Warning zeros may include missing source chemistry, as documented in the root README. No causal claims or deployment reliability are established.

## Reproduce

Run `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ngs-mpl /home/best/anaconda3/envs/tsf-ad/bin/python ML/train_phylum.py --seed 42 --n-iter 8` from the repository root. Input SHA256 hashes and package versions are saved in run_config.json.

All required artifacts are alongside this report: cleaned_merged.csv, final_feature_matrix.csv, selected_taxa.csv, selected_BAC_P.txt, selected_ARC_P.txt, assignments, tuning scores, CV scores, thresholds, model/preprocessing objects, and combined actual/probability/binary prediction CSVs. Figures: multilabel_mcc.png, micro_aupr.png, micro_roc_auc.png, micro_precision_recall.png.
