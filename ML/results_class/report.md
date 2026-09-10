# Class-only multi-label warning prediction

Five warnings are predicted jointly using binary relevance and microbiome predictors only. CV selected **RandomForest** using the equal-weight mean rank across the three metrics.

## Final test performance

| Model        |   Multilabel MCC |   Micro-AUPR |   Micro-ROC-AUC |
|:-------------|-----------------:|-------------:|----------------:|
| RandomForest |           0.4528 |       0.6193 |          0.9167 |
| KNN          |           0.3924 |       0.3736 |          0.8534 |
| SVM          |           0.3471 |       0.4290 |          0.7821 |
| NNET         |           0.5191 |       0.5049 |          0.9382 |
| XGBoost      |           0.3645 |       0.5658 |          0.8508 |
| GLMNET       |           0.4505 |       0.3685 |          0.9033 |

Test metric leaders: {'Multilabel MCC': 'NNET', 'Micro-AUPR': 'RandomForest', 'Micro-ROC-AUC': 'NNET'}. These are descriptive results; they do not change the CV-selected model.

A unanimous winner exists only when all three metrics favor the same model. Otherwise the rankings reflect a trade-off, not a single best model on every criterion.

## Training cross-validation

| Model        |   Multilabel MCC |   Micro-AUPR |   Micro-ROC-AUC |
|:-------------|-----------------:|-------------:|----------------:|
| RandomForest |           0.4715 |       0.3946 |          0.7962 |
| KNN          |           0.3289 |       0.2508 |          0.7511 |
| SVM          |           0.4028 |       0.3266 |          0.7503 |
| NNET         |           0.3492 |       0.2840 |          0.7584 |
| XGBoost      |           0.3473 |       0.3341 |          0.7604 |
| GLMNET       |           0.3929 |       0.2939 |          0.7806 |

CV metric leaders: {'Multilabel MCC': 'RandomForest', 'Micro-AUPR': 'RandomForest', 'Micro-ROC-AUC': 'RandomForest'}.

These are pooled out-of-fold tuning estimates, not nested unbiased estimates. Hyperparameter selection and shared-threshold optimization use these same predictions, so CV scores, especially MCC, are optimistic.

## Data integration and identity audit

metadata.csv has 140 rows and 135 columns. BAC C(%) has 150 annotation rows and 140 samples; ARC C(%) has 19 annotation rows and 138 samples. Workbook sample suffixes are removed by retaining the first two hyphen-separated components; b and apostrophe markers remain intact. All normalized workbook IDs match metadata. No duplicate or missing SampleID/Site IDs were found; unique SampleID also guarantees unique Site/SampleID pairs.

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

Use only BAC C(%) and ARC C(%), in percentage units. Retain a value only when strictly >0.1 in that sample; otherwise set it to zero. Do not renormalize. The union across the 138 matched samples defines 112 BAC-source and 17 ARC-source columns, 129 total, retained in every fold. This is the explicitly allowed predefined feature-space rule; no global means, Top-N, or learned feature selection are used. 16 columns are zero throughout the training set and are retained under that rule.

Domain prefixes denote workbook origin. Supplied off-domain/unidentified annotations remain included under the requested abundance-only criterion. Workbook origin is not proof that every retained annotation belongs to that biological domain; unresolved annotations are preserved as supplied. No unrequested taxonomy cleanup was applied.

Complete selected annotations and sample counts:

| block   | taxon                         |   samples_above_0_1_percent |   training_samples_above_0_1_percent |
|:--------|:------------------------------|----------------------------:|-------------------------------------:|
| BAC_C   | ABY1                          |                           2 |                                    1 |
| BAC_C   | Acetothermiia                 |                          12 |                                    4 |
| BAC_C   | Acidimicrobiia                |                          59 |                                   44 |
| BAC_C   | Acidobacteriae                |                          11 |                                   11 |
| BAC_C   | Actinobacteria                |                         112 |                                   85 |
| BAC_C   | Alphaproteobacteria           |                          95 |                                   73 |
| BAC_C   | Aminicenantia                 |                          61 |                                   41 |
| BAC_C   | Anaerolineae                  |                          98 |                                   75 |
| BAC_C   | BRH-c20a                      |                           8 |                                    6 |
| BAC_C   | Babeliae                      |                          34 |                                   25 |
| BAC_C   | Bacilli                       |                         126 |                                  102 |
| BAC_C   | Bacteroidia                   |                         138 |                                  109 |
| BAC_C   | Bdellovibrionia               |                           3 |                                    2 |
| BAC_C   | Berkelbacteria                |                           1 |                                    0 |
| BAC_C   | Blastocatellia                |                          33 |                                   25 |
| BAC_C   | Brachyspirae                  |                          36 |                                   24 |
| BAC_C   | CK-2C2-2                      |                           9 |                                    7 |
| BAC_C   | Caldatribacteriia             |                          83 |                                   68 |
| BAC_C   | Caldisericia                  |                          84 |                                   64 |
| BAC_C   | Calditrichia                  |                           1 |                                    0 |
| BAC_C   | Campylobacteria               |                          26 |                                   16 |
| BAC_C   | Chlamydiae                    |                          26 |                                   17 |
| BAC_C   | Chthonomonadetes              |                          16 |                                   14 |
| BAC_C   | Cloacimonadia                 |                         133 |                                  104 |
| BAC_C   | Clostridia                    |                         138 |                                  109 |
| BAC_C   | Coprothermobacteria           |                           8 |                                    8 |
| BAC_C   | Coriobacteriia                |                          29 |                                   25 |
| BAC_C   | D8A-2                         |                         103 |                                   89 |
| BAC_C   | Dadabacteriia                 |                           3 |                                    0 |
| BAC_C   | Deinococci                    |                           6 |                                    3 |
| BAC_C   | Desulfitobacteriia            |                           3 |                                    3 |
| BAC_C   | Desulfobacteria               |                           2 |                                    0 |
| BAC_C   | Desulfobulbia                 |                          37 |                                   27 |
| BAC_C   | Desulfotomaculia              |                          83 |                                   70 |
| BAC_C   | Desulfovibrionia              |                          51 |                                   40 |
| BAC_C   | Desulfuromonadia              |                          24 |                                   21 |
| BAC_C   | Dethiobacteria                |                          47 |                                   41 |
| BAC_C   | Elusimicrobia                 |                           1 |                                    0 |
| BAC_C   | Endomicrobia                  |                           7 |                                    7 |
| BAC_C   | Fermentibacteria              |                          16 |                                   11 |
| BAC_C   | Fibrobacteria                 |                          16 |                                   13 |
| BAC_C   | Fimbriimonadia                |                          23 |                                   15 |
| BAC_C   | Fusobacteriia                 |                          21 |                                   14 |
| BAC_C   | Gammaproteobacteria           |                         120 |                                   91 |
| BAC_C   | Gemmatimonadetes              |                          16 |                                   11 |
| BAC_C   | Gracilibacteria               |                          19 |                                   16 |
| BAC_C   | Halanaerobiia                 |                           2 |                                    2 |
| BAC_C   | Holophagae                    |                          36 |                                   25 |
| BAC_C   | Hydrogenedentia               |                          65 |                                   49 |
| BAC_C   | Hydrothermae                  |                           1 |                                    1 |
| BAC_C   | Ignavibacteria                |                          29 |                                   20 |
| BAC_C   | Incertae_Sedis                |                         124 |                                  103 |
| BAC_C   | JS1                           |                          64 |                                   46 |
| BAC_C   | KD4-96                        |                           4 |                                    0 |
| BAC_C   | Kapabacteria                  |                          44 |                                   33 |
| BAC_C   | Kiritimatiellae               |                         107 |                                   84 |
| BAC_C   | LCP-89                        |                           1 |                                    0 |
| BAC_C   | Latescibacteria               |                           1 |                                    0 |
| BAC_C   | Latescibacterota              |                           2 |                                    0 |
| BAC_C   | Lentisphaeria                 |                          16 |                                   11 |
| BAC_C   | Leptospirae                   |                          66 |                                   50 |
| BAC_C   | Limnochordia                  |                          66 |                                   57 |
| BAC_C   | MBNT15                        |                           1 |                                    0 |
| BAC_C   | MD2902-B12                    |                           2 |                                    0 |
| BAC_C   | MVP-15                        |                          55 |                                   41 |
| BAC_C   | Marinimicrobia_(SAR406_clade) |                          30 |                                   24 |
| BAC_C   | Microgenomatia                |                           2 |                                    1 |
| BAC_C   | Moorellia                     |                           7 |                                    7 |
| BAC_C   | Myxococcia                    |                           1 |                                    1 |
| BAC_C   | NB1-j                         |                           2 |                                    0 |
| BAC_C   | NKB15                         |                           6 |                                    6 |
| BAC_C   | Negativicutes                 |                          45 |                                   37 |
| BAC_C   | Nitrospiria                   |                          15 |                                    7 |
| BAC_C   | OLB14                         |                          14 |                                   10 |
| BAC_C   | OM190                         |                           1 |                                    0 |
| BAC_C   | Oligoflexia                   |                          16 |                                   12 |
| BAC_C   | Omnitrophia                   |                          11 |                                    5 |
| BAC_C   | Parcubacteria                 |                           8 |                                    6 |
| BAC_C   | Phycisphaerae                 |                          24 |                                   16 |
| BAC_C   | Pla3_lineage                  |                           6 |                                    5 |
| BAC_C   | Planctomycetes                |                          81 |                                   62 |
| BAC_C   | Polyangia                     |                          15 |                                    8 |
| BAC_C   | RBG-16-55-12                  |                           8 |                                    5 |
| BAC_C   | Rubrobacteria                 |                           1 |                                    0 |
| BAC_C   | SJA-28                        |                          70 |                                   53 |
| BAC_C   | Saccharimonadia               |                          43 |                                   32 |
| BAC_C   | Spirochaetia                  |                         128 |                                  101 |
| BAC_C   | Sumerlaeia                    |                          22 |                                   15 |
| BAC_C   | Sva0485                       |                           8 |                                    2 |
| BAC_C   | Symbiobacteriia               |                           4 |                                    4 |
| BAC_C   | Synergistia                   |                         136 |                                  107 |
| BAC_C   | Syntrophia                    |                          91 |                                   67 |
| BAC_C   | Syntrophobacteria             |                          45 |                                   29 |
| BAC_C   | Syntrophomonadia              |                         129 |                                  103 |
| BAC_C   | Syntrophorhabdia              |                          80 |                                   60 |
| BAC_C   | TA06                          |                           8 |                                    4 |
| BAC_C   | Thermacetogenia               |                          27 |                                   27 |
| BAC_C   | Thermoanaerobacteria          |                          14 |                                   13 |
| BAC_C   | Thermoanaerobaculia           |                          35 |                                   26 |
| BAC_C   | Thermoleophilia               |                          58 |                                   44 |
| BAC_C   | Thermotogae                   |                         105 |                                   85 |
| BAC_C   | Thermovenabulia               |                          23 |                                   23 |
| BAC_C   | Vampirivibrionia              |                           2 |                                    2 |
| BAC_C   | Verrucomicrobiae              |                          96 |                                   70 |
| BAC_C   | Vicinamibacteria              |                           2 |                                    2 |
| BAC_C   | WCHB1-81                      |                          39 |                                   27 |
| BAC_C   | WPS-2                         |                           4 |                                    3 |
| BAC_C   | WS1                           |                          47 |                                   37 |
| BAC_C   | WWE3                          |                           3 |                                    2 |
| BAC_C   | Zixibacteria                  |                           5 |                                    3 |
| BAC_C   | c5LKS83                       |                          11 |                                    3 |
| BAC_C   | unidentified                  |                         117 |                                   92 |
| ARC_C   | Actinopterygii                |                           1 |                                    1 |
| ARC_C   | Aenigmarchaeia                |                           2 |                                    2 |
| ARC_C   | Bathyarchaeia                 |                          67 |                                   51 |
| ARC_C   | Bivalvia                      |                           2 |                                    2 |
| ARC_C   | Ichthyosporea                 |                           1 |                                    1 |
| ARC_C   | LKM11                         |                           1 |                                    0 |
| ARC_C   | Lokiarchaeia                  |                           2 |                                    1 |
| ARC_C   | Methanomethylicia             |                          20 |                                   12 |
| ARC_C   | Methanomicrobia               |                         138 |                                  109 |
| ARC_C   | Methanosarcinia               |                         135 |                                  106 |
| ARC_C   | Micrarchaeia                  |                          18 |                                   13 |
| ARC_C   | Nanoarchaeia                  |                          52 |                                   40 |
| ARC_C   | Nitrososphaeria               |                          13 |                                   10 |
| ARC_C   | Odinarchaeia                  |                           1 |                                    0 |
| ARC_C   | Thermococci                   |                         127 |                                   98 |
| ARC_C   | Thermoplasmata                |                         132 |                                  104 |
| ARC_C   | unidentified                  |                         115 |                                   93 |

## Pipeline, tuning and thresholds

Binary relevance fits five classifiers internally; all evaluation pools their outputs. Eight seeded random hyperparameter candidates per model are tested in five grouped folds: 48 configurations and 240 outer-fold multi-label fits. Search spaces and versions are in run_config.json; all candidate/fold scores are in hyperparameter_tuning.csv. Best settings are in each model’s best_params.json.

KNN, SVM, NNET and GLMNET use fold-local standardization. RF and XGBoost are unscaled. All abundance values are finite and nonnegative; no missing microbiome cells required imputation. Fold-local median imputation remains in the saved pipeline as a safeguard. Missing whole ARC profiles are excluded as above.

RF/SVM/GLMNET use balanced class weights; XGBoost uses the training-label negative/positive ratio. KNN and NNET use per-label random oversampling only on fold training rows. Single-class training labels use constant predictions. SVM probabilities use three-fold site-grouped inner out-of-fold sigmoid calibration with preprocessing refitted in each inner fold.

GLMNET is implemented as elastic-net logistic regression using sklearn saga, not the R glmnet package. NNET uses sklearn MLP. Model objects and preprocessing objects are saved as joblib files.

Candidate and model selection use equal-weight mean rank over all three pooled CV metrics; deterministic ties retain search/model order. Candidate MCC uses a shared threshold optimized on its OOF predictions. The final shared threshold maximizes flattened MCC over 0.01–0.99, with ties closest to 0.5. A shared threshold limits extra fitting with very few positives; each of the five labels receives the same fixed threshold for a given model.

| Model        |   warning_acid_base_balance |   warning_buffer_capacity |   warning_acid_accumulation |   warning_ammonia_toxicity |   warning_biogas_quality |
|:-------------|----------------------------:|--------------------------:|----------------------------:|---------------------------:|-------------------------:|
| RandomForest |                      0.3200 |                    0.3200 |                      0.3200 |                     0.3200 |                   0.3200 |
| KNN          |                      0.6500 |                    0.6500 |                      0.6500 |                     0.6500 |                   0.6500 |
| SVM          |                      0.2400 |                    0.2400 |                      0.2400 |                     0.2400 |                   0.2400 |
| NNET         |                      0.3300 |                    0.3300 |                      0.3300 |                     0.3300 |                   0.3300 |
| XGBoost      |                      0.3100 |                    0.3100 |                      0.3100 |                     0.3100 |                   0.3100 |
| GLMNET       |                      0.4100 |                    0.4100 |                      0.4100 |                     0.4100 |                   0.4100 |

3 fitting warnings were recorded; see fit_warnings.csv. Warnings include convergence limits; convergence is not guaranteed for those fits.

## Metric definitions and isolation

Multilabel MCC = binary MCC calculated after flattening all sample-label pairs across the five warning labels. Micro-AUPR = trapezoidal area under the precision–recall curve of flattened labels and probabilities, not average precision. Micro-ROC-AUC = ROC-AUC of flattened labels and probabilities. Only MCC uses thresholded predictions. These pooled metrics do not measure exact five-label-vector correctness.

Every metadata column is excluded from predictors, including all effluent warning-definition measurements, substrate chemistry, operation, Season, Site and identifiers. The full excluded list and genomic whitelist are in leakage_check.json. Identifiers and Season appear only in integration/audit/prediction outputs. Feature matrix SampleID is an index, not a predictor.

The test set was not used for fitting, preprocessing fitting, tuning, thresholds, or model selection in this run. Models and thresholds were saved before test evaluation. The fixed sample-specific union is the sole permitted feature-space exception. This deterministic split coincides with a previously evaluated repository holdout: it is not historically untouched or external validation.

## Interpretation

The results describe pooled prediction across five warning conditions from microbiome composition alone. The small test cohort (29 samples from seven groups), rare positives, prior use of this holdout, and imperfect supplied taxonomy limit generalization claims. Warning zeros may include missing source chemistry, as documented in the root README. No causal claims or deployment reliability are established.

## Reproduce

Run `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ngs-mpl /home/best/anaconda3/envs/tsf-ad/bin/python ML/train_class.py --seed 42 --n-iter 8` from the repository root. Input SHA256 hashes and package versions are saved in run_config.json.

All required artifacts are alongside this report: cleaned_merged.csv, final_feature_matrix.csv, selected_taxa.csv, selected_BAC_C.txt, selected_ARC_C.txt, assignments, tuning scores, CV scores, thresholds, model/preprocessing objects, and combined actual/probability/binary prediction CSVs. Figures: multilabel_mcc.png, micro_aupr.png, micro_roc_auc.png, micro_precision_recall.png.
