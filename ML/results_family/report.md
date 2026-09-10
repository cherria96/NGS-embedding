# Family-level joint five-label warning prediction
CV-selected model: **RandomForest**.
Minimize equal-weight mean descending rank across pooled OOF flattened MCC at 0.5, Micro-AUPR and Micro-ROC-AUC; ties favor AUPR then ROC-AUC then MCC. Select before test evaluation.

## Held-out test performance
| Model | Multilabel MCC | Micro-AUPR | Micro-ROC-AUC |
|---|---:|---:|---:|
| RandomForest | 0.4196 | 0.5857 | 0.9120 |
| KNN | 0.3799 | 0.4609 | 0.7788 |
| SVM | 0.4196 | 0.5764 | 0.7867 |
| NNET | 0.4862 | 0.4357 | 0.8619 |
| XGBoost | 0.3240 | 0.3462 | 0.8161 |
| GLMNET | 0.4109 | 0.3129 | 0.8823 |

## Training OOF tuning estimates
| Model | Multilabel MCC | Micro-AUPR | Micro-ROC-AUC |
|---|---:|---:|---:|
| RandomForest | 0.4566 | 0.3875 | 0.7884 |
| KNN | 0.2981 | 0.2924 | 0.7137 |
| SVM | 0.4624 | 0.4271 | 0.7938 |
| NNET | 0.3580 | 0.2909 | 0.7017 |
| XGBoost | 0.4301 | 0.3258 | 0.7496 |
| GLMNET | 0.3937 | 0.2796 | 0.7396 |
OOF estimates are optimistic because hyperparameters and the shared threshold were tuned on these folds. Selection ranking uses MCC at 0.5; this table uses the optimized threshold.
Multilabel MCC is binary MCC after flattening all sample-label pairs. Micro-AUPR is trapezoidal PR area from flattened probabilities; Micro-ROC-AUC is ROC area from those same pairs. Only these three metrics are used.
Pooled metrics assess sample-label pairs and do not measure exact recovery of entire five-label vectors. More prevalent labels can dominate.

## Family selection
Counts use 109 training samples for selection; 138-sample counts are descriptive only.

domain: ARC, original_OTUs: 111, unique_Families: 27, all_samples_exceed_1_any: 17, all_samples_meet_5_percent_descriptive: 13, training_exceed_1_any: 17, training_meet_5_percent_retained: 12

| Family | Training >1% count | Training % | All-sample >1% count | All-sample % |
|---|---:|---:|---:|---:|
| Bathyarchaeia | 7 | 6.42 | 8 | 5.80 |
| Methanofastidiosaceae | 78 | 71.56 | 104 | 75.36 |
| Methanomassiliicoccaceae | 73 | 66.97 | 90 | 65.22 |
| Methanomethylophilaceae | 30 | 27.52 | 37 | 26.81 |
| Methanomicrobiaceae | 85 | 77.98 | 93 | 67.39 |
| Methanomicrobiales | 13 | 11.93 | 16 | 11.59 |
| Methanoregulaceae | 70 | 64.22 | 92 | 66.67 |
| Methanosaetaceae | 97 | 88.99 | 126 | 91.30 |
| Methanosarcinaceae | 40 | 36.70 | 47 | 34.06 |
| Methanospirillaceae | 88 | 80.73 | 117 | 84.78 |
| SCGC_AAA011-D5 | 11 | 10.09 | 15 | 10.87 |
| unidentified | 67 | 61.47 | 81 | 58.70 |

domain: BAC, original_OTUs: 2925, unique_Families: 542, all_samples_exceed_1_any: 136, all_samples_meet_5_percent_descriptive: 70, training_exceed_1_any: 124, training_meet_5_percent_retained: 64

| Family | Training >1% count | Training % | All-sample >1% count | All-sample % |
|---|---:|---:|---:|---:|
| 009E01-B-SD-P15 | 10 | 9.17 | 15 | 10.87 |
| Acholeplasmataceae | 29 | 26.61 | 36 | 26.09 |
| Aminicenantales | 8 | 7.34 | 11 | 7.97 |
| Anaerolineaceae | 10 | 9.17 | 13 | 9.42 |
| Anaerovoracaceae | 21 | 19.27 | 23 | 16.67 |
| Atopobiaceae | 7 | 6.42 | 7 | 5.07 |
| Bacteroidales_UCG-001 | 11 | 10.09 | 13 | 9.42 |
| Bacteroidetes_vadinHA17 | 61 | 55.96 | 83 | 60.14 |
| Brachyspirales_Incertae_Sedis | 8 | 7.34 | 12 | 8.70 |
| Caldatribacteriaceae | 9 | 8.26 | 9 | 6.52 |
| Caldicoprobacteraceae | 30 | 27.52 | 31 | 22.46 |
| Chitinophagaceae | 7 | 6.42 | 12 | 8.70 |
| Christensenellaceae | 80 | 73.39 | 96 | 69.57 |
| Cloacimonadaceae | 72 | 66.06 | 93 | 67.39 |
| Comamonadaceae | 64 | 58.72 | 86 | 62.32 |
| Competibacteraceae | 7 | 6.42 | 13 | 9.42 |
| Coprothermobacteraceae | 7 | 6.42 | 7 | 5.07 |
| D8A-2 | 10 | 9.17 | 10 | 7.25 |
| DTU014 | 73 | 66.97 | 84 | 60.87 |
| Desulfotomaculales | 8 | 7.34 | 9 | 6.52 |
| Dethiobacteraceae | 19 | 17.43 | 20 | 14.49 |
| Dysgonomonadaceae | 71 | 65.14 | 82 | 59.42 |
| Fermentibacteraceae | 7 | 6.42 | 8 | 5.80 |
| Gracilibacteraceae | 11 | 10.09 | 11 | 7.97 |
| Hungateiclostridiaceae | 89 | 81.65 | 107 | 77.54 |
| Incertae_Sedis | 9 | 8.26 | 9 | 6.52 |
| Intrasporangiaceae | 12 | 11.01 | 15 | 10.87 |
| Kosmotogaceae | 20 | 18.35 | 30 | 21.74 |
| LD1-PA32 | 6 | 5.50 | 7 | 5.07 |
| Lachnospiraceae | 23 | 21.10 | 28 | 20.29 |
| Lactobacillaceae | 6 | 5.50 | 7 | 5.07 |
| Lentimicrobiaceae | 33 | 30.28 | 42 | 30.43 |
| MBA03 | 20 | 18.35 | 20 | 14.49 |
| Marinilabiliaceae | 11 | 10.09 | 11 | 7.97 |
| Methylophilaceae | 8 | 7.34 | 16 | 11.59 |
| Paludibacteraceae | 43 | 39.45 | 54 | 39.13 |
| Pedosphaeraceae | 11 | 10.09 | 15 | 10.87 |
| Peptostreptococcales-Tissierellales | 44 | 40.37 | 52 | 37.68 |
| Petrotogaceae | 34 | 31.19 | 37 | 26.81 |
| Prevotellaceae | 10 | 9.17 | 12 | 8.70 |
| Prolixibacteraceae | 52 | 47.71 | 59 | 42.75 |
| Rhodanobacteraceae | 7 | 6.42 | 11 | 7.97 |
| Rhodobacteraceae | 7 | 6.42 | 9 | 6.52 |
| Rhodocyclaceae | 46 | 42.20 | 64 | 46.38 |
| Rikenellaceae | 57 | 52.29 | 70 | 50.72 |
| Ruminococcaceae | 14 | 12.84 | 16 | 11.59 |
| SBR1031 | 22 | 20.18 | 30 | 21.74 |
| SC-I-84 | 6 | 5.50 | 11 | 7.97 |
| SJA-15 | 8 | 7.34 | 10 | 7.25 |
| SJA-28 | 6 | 5.50 | 13 | 9.42 |
| ST-12K33 | 50 | 45.87 | 58 | 42.03 |
| Saprospiraceae | 23 | 21.10 | 35 | 25.36 |
| Sedimentibacteraceae | 72 | 66.06 | 88 | 63.77 |
| Smithellaceae | 48 | 44.04 | 68 | 49.28 |
| Spirochaetaceae | 65 | 59.63 | 81 | 58.70 |
| Synergistaceae | 82 | 75.23 | 100 | 72.46 |
| Syntrophomonadaceae | 76 | 69.72 | 84 | 60.87 |
| Syntrophorhabdaceae | 27 | 24.77 | 39 | 28.26 |
| Thermacetogeniaceae | 13 | 11.93 | 13 | 9.42 |
| UCG-010 | 8 | 7.34 | 9 | 6.52 |
| W27 | 30 | 27.52 | 35 | 25.36 |
| Williamwhitmaniaceae | 34 | 31.19 | 45 | 32.61 |
| Xanthomonadaceae | 12 | 11.01 | 14 | 10.14 |
| unidentified | 100 | 91.74 | 127 | 92.03 |

Total retained microbiome features: 76.


## Data and methods
Matched samples: 138; site labels: 37; groups: 35.

Train: 109 samples, 28 groups. Test: 29 samples, 7 groups.

Training sites: ADA, ADP, BSE, BSI, BSN, BSO, BSS, CGS, CWS, DGB, DGC, DGD, DGG, DGJ, GCG, GHGa, GHGb, GHH, GHJ, GSG, JJN, JJY, MGY, USJ, USO, USY, YCG, YSY

Test sites: CWM, DGY, MYS, SCH, UJG, YCD, YSD

Missing ARC: 1-32' and 4-24; excluded explicitly in sample_audit.csv.

ARC header 2-29-SCHG-ARC has redundant domain suffix removed for site checks; logged in sample_id_corrections.csv.

Duplicate/missing metadata and normalized genomic IDs checked; SampleID and Site mapping validated.

QC replicates BSIb→BSI and DGYb→DGY; 35 groups, not the assumed 34 complete sites.

Final microbiome features: {'BAC_F': 64, 'ARC_F': 12}

Numeric ranges parsed to midpoint; ± values use center; other nonnumeric values become missing (logged).

Median numeric imputation, most-frequent categorical imputation, unknown-safe one-hot encoding, z-score scaling; all fitted on training rows.

RF/SVM/elastic-net logistic regression: balanced class weights; XGBoost: training negative/positive ratio.

KNN/NNET: per-label random oversampling on training rows only. Single-class training labels use constant prediction.

GLMNET denotes sklearn elastic-net logistic regression (saga), not the R glmnet package.

All supplied annotations remain eligible, including unidentified and off-domain taxa.

Existing warning labels retained; missing target-source measurements can produce unflagged zeros.

CV scores are tuning estimates, not nested unbiased performance estimates. Rare positives limit stability.

No additional ML feature selection. All-sample Family counts are descriptive; selection uses training samples only.

## Interpretation
RandomForest is the preferred model under the predeclared joint CV ranking. Test results describe generalization to the held-out groups and do not change that selection.
The test set contains 29 samples from 7 groups. Rare labels and few independent sites limit precision; no causal inference is supported.
The fixed holdout has appeared in earlier analyses. This run excludes it from feature fitting, preprocessing, tuning, threshold optimization and model selection, but it is not a never-before-used external validation set.

## Reproducibility
Run `python ML/train_family.py --seed 42 --n-iter 8`. See run_config.json for input hashes, versions and search spaces.
See split_and_cv_assignment.csv for every sample/site/fold, label_distribution.csv for label counts, selected_taxa.csv for the complete Family list, and README_family.md for pipeline details.
Recorded fit warnings: 15; see fit_warnings.csv for candidate and fold details.


Convergence limits: NNET and GLMNET have iteration-limit warnings in training CV; the final GLMNET fit also has nonconverged coefficients. Their comparisons require caution. The selected Random Forest has no recorded fit warnings. No refitting or tuning was done in response to test scores.
