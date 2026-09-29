# Genomic/microbiome-only prediction of the five anaerobic digestion warning conditions
CV-selected model: **RandomForest**.
Minimize equal-weight mean descending rank across pooled OOF flattened MCC at 0.5, Micro-AUPR and Micro-ROC-AUC; ties favor AUPR then ROC-AUC then MCC. Select before test evaluation.

## Held-out test performance
| Model | Multilabel MCC | Micro-AUPR | Micro-ROC-AUC |
|---|---:|---:|---:|
| RandomForest | 0.3798 | 0.5453 | 0.9108 |
| KNN | 0.2243 | 0.2997 | 0.7232 |
| SVM | 0.4924 | 0.5522 | 0.7733 |
| NNET | 0.4483 | 0.3474 | 0.8695 |
| XGBoost | 0.2562 | 0.3453 | 0.8849 |
| GLMNET | 0.3645 | 0.3665 | 0.8712 |

## Training OOF tuning estimates
| Model | Multilabel MCC | Micro-AUPR | Micro-ROC-AUC |
|---|---:|---:|---:|
| RandomForest | 0.4857 | 0.3960 | 0.7910 |
| KNN | 0.3823 | 0.3181 | 0.7611 |
| SVM | 0.3682 | 0.3151 | 0.7048 |
| NNET | 0.4043 | 0.3375 | 0.7086 |
| XGBoost | 0.3731 | 0.3092 | 0.7487 |
| GLMNET | 0.4528 | 0.3649 | 0.7690 |
OOF estimates are optimistic because hyperparameters and the shared threshold were tuned on these folds. Selection ranking uses MCC at 0.5; this table uses the optimized threshold.
Multilabel MCC is binary MCC after flattening all sample-label pairs. Micro-AUPR is trapezoidal PR area from flattened probabilities; Micro-ROC-AUC is ROC area from those same pairs. Only these three metrics are used.
Pooled metrics assess sample-label pairs and do not measure exact recovery of entire five-label vectors. More prevalent labels can dominate.

## Family selection
Union of families exceeding 0.1% in at least one of 138 matched samples. Values <=0.1% are zero; other values remain original percentages. No renormalization, root transform, or prevalence filter.

## ARC_F: 24 features
| Family | Samples >0.1% |
|---|---:|
| Aenigmarchaeales | 2 |
| Bathyarchaeia | 67 |
| LKM11 | 1 |
| Lokiarchaeia | 2 |
| Methanocorpusculaceae | 29 |
| Methanofastidiosaceae | 127 |
| Methanomassiliicoccaceae | 117 |
| Methanomethyliaceae | 20 |
| Methanomethylophilaceae | 47 |
| Methanomicrobiaceae | 118 |
| Methanomicrobiales | 46 |
| Methanoregulaceae | 111 |
| Methanosaetaceae | 132 |
| Methanosarcinaceae | 107 |
| Methanospirillaceae | 127 |
| Micrarchaeales | 18 |
| Nitrososphaeraceae | 13 |
| Odinarchaeia | 1 |
| Rhinosporideacae | 1 |
| SCGC_AAA011-D5 | 46 |
| Teleostei | 1 |
| Veneroida | 2 |
| Woesearchaeales | 11 |
| unidentified | 123 |

## BAC_F: 346 features
| Family | Samples >0.1% |
|---|---:|
| 009E01-B-SD-P15 | 63 |
| 0319-6G20 | 14 |
| 053A03-B-DI-P58 | 1 |
| 1-20 | 10 |
| 113B434 | 4 |
| 37-13 | 3 |
| 4572-13 | 1 |
| 67-14 | 5 |
| A0839 | 15 |
| A21b | 1 |
| AKYH767 | 47 |
| Absconditabacteriales_(SR1) | 13 |
| Acetobacteraceae | 16 |
| Acetothermiia | 12 |
| Acholeplasmataceae | 72 |
| Acidaminococcaceae | 27 |
| Acidobacteriae | 1 |
| Actinomycetaceae | 18 |
| Aerococcaceae | 4 |
| Aeromonadaceae | 21 |
| Alcaligenaceae | 10 |
| Aminicenantales | 61 |
| Anaerofustaceae | 10 |
| Anaerolineaceae | 85 |
| Anaeromyxobacteraceae | 1 |
| Anaerovoracaceae | 119 |
| Aquaspirillaceae | 7 |
| Arcobacteraceae | 21 |
| Ardenticatenaceae | 1 |
| Arenicellaceae | 1 |
| Atopobiaceae | 18 |
| B1-7BS | 14 |
| B55-F-B-G02 | 5 |
| BRH-c20a | 8 |
| Babeliales | 26 |
| Bacillaceae | 15 |
| Bacteriovoracaceae | 1 |
| Bacteroidaceae | 50 |
| Bacteroidales_UCG-001 | 49 |
| Bacteroidetes_BD2-2 | 5 |
| Bacteroidetes_VC2.1_Bac22 | 12 |
| Bacteroidetes_vadinHA17 | 109 |
| Bdellovibrionaceae | 2 |
| Beijerinckiaceae | 34 |
| Berkelbacteria | 1 |
| Bifidobacteriaceae | 9 |
| Blastocatellaceae | 15 |
| Bogoriellaceae | 4 |
| Brachyspirales_Incertae_Sedis | 36 |
| Bradymonadales | 1 |
| Bryobacteraceae | 2 |
| Burkholderiaceae | 62 |
| Butyricicoccaceae | 1 |
| C10-SB1A | 2 |
| CCM19a | 4 |
| CK-2C2-2 | 9 |
| Caldatribacteriaceae | 83 |
| Caldicoprobacteraceae | 71 |
| Caldilineaceae | 25 |
| Caldisericaceae | 30 |
| Calditrichaceae | 1 |
| Caloramatoraceae | 6 |
| Candidatus_Falkowbacteria | 2 |
| Candidatus_Magasanikbacteria | 1 |
| Candidatus_Peregrinibacteria | 5 |
| Candidatus_Shapirobacteria | 1 |
| Carnobacteriaceae | 24 |
| Caulobacteraceae | 5 |
| Chitinimonadaceae | 2 |
| Chitinophagaceae | 64 |
| Christensenellaceae | 133 |
| Chromobacteriaceae | 1 |
| Chthoniobacteraceae | 1 |
| Chthonomonadales | 16 |
| Cloacimonadaceae | 111 |
| Cloacimonadales | 3 |
| Clostridia_UCG-014 | 19 |
| Clostridiaceae | 32 |
| Comamonadaceae | 106 |
| Competibacteraceae | 54 |
| Coprothermobacteraceae | 8 |
| Coriobacteriales_Incertae_Sedis | 1 |
| Corynebacteriaceae | 10 |
| Crocinitomicaceae | 1 |
| D8A-2 | 103 |
| DEV007 | 7 |
| DS-100 | 25 |
| DTU014 | 124 |
| Dadabacteriales | 3 |
| Defluviicoccaceae | 9 |
| Defluviitaleaceae | 10 |
| Deinococcaceae | 2 |
| Dermatophilaceae | 6 |
| Desulfitibacteraceae | 6 |
| Desulfobacteraceae | 1 |
| Desulfobulbaceae | 36 |
| Desulfocapsaceae | 2 |
| Desulfomicrobiaceae | 37 |
| Desulforegulaceae | 1 |
| Desulfosarcinaceae | 1 |
| Desulfotomaculales | 83 |
| Desulfovibrionaceae | 17 |
| Desulfuromonadaceae | 1 |
| Dethiobacteraceae | 47 |
| Devosiaceae | 14 |
| Dietziaceae | 1 |
| Diplorickettsiaceae | 1 |
| Dongiaceae | 1 |
| Dysgonomonadaceae | 128 |
| Eggerthellaceae | 1 |
| Elsteraceae | 1 |
| Endomicrobiaceae | 7 |
| Enterobacteriaceae | 16 |
| Enterococcaceae | 4 |
| Erwiniaceae | 3 |
| Erysipelatoclostridiaceae | 21 |
| Erysipelotrichaceae | 7 |
| Eubacteriaceae | 8 |
| F082 | 7 |
| FTLpost3 | 9 |
| Fermentibacteraceae | 16 |
| Fervidobacteriaceae | 11 |
| Fibrobacteraceae | 9 |
| Fibrobacterales | 4 |
| Fimbriimonadaceae | 21 |
| Flavobacteriaceae | 11 |
| Fusibacteraceae | 6 |
| Fusobacteriaceae | 1 |
| GZKB124 | 3 |
| GZKB75 | 13 |
| Gallionellaceae | 1 |
| Garciellaceae | 9 |
| Gemmataceae | 32 |
| Gemmatimonadaceae | 16 |
| Geobacteraceae | 12 |
| Gracilibacteraceae | 82 |
| HOC36 | 1 |
| Hahellaceae | 1 |
| Halanaerobiaceae | 2 |
| Haliangiaceae | 6 |
| Halomonadaceae | 3 |
| Holophagaceae | 11 |
| Hungateiclostridiaceae | 135 |
| Hydrogenedensaceae | 65 |
| Hydrogenophilaceae | 19 |
| Hydrothermae | 1 |
| Hyphomicrobiaceae | 36 |
| Hyphomonadaceae | 33 |
| IMCC26256 | 5 |
| Iamiaceae | 4 |
| Ignavibacteriaceae | 6 |
| Ilumatobacteraceae | 1 |
| Incertae_Sedis | 35 |
| Intrasporangiaceae | 57 |
| Isosphaeraceae | 7 |
| Izemoplasmataceae | 2 |
| Izemoplasmatales | 58 |
| JG36-GS-52 | 2 |
| JS1 | 64 |
| KCLunmb-38-53 | 13 |
| KD1-131 | 2 |
| KD4-96 | 4 |
| Kapabacteriales | 44 |
| Kosmotogaceae | 47 |
| LCP-89 | 1 |
| LD1-PA32 | 16 |
| LD1-PB3 | 32 |
| LF045 | 28 |
| LWQ8 | 4 |
| Lachnospiraceae | 108 |
| Lactobacillaceae | 45 |
| Latescibacteraceae | 1 |
| Latescibacterota | 2 |
| Leeiaceae | 4 |
| Legionellaceae | 1 |
| Lenti-02 | 3 |
| Lentimicrobiaceae | 113 |
| Leptospiraceae | 66 |
| Leptotrichiaceae | 20 |
| Leuconostocaceae | 9 |
| LiUU-11-161 | 1 |
| Limnochordaceae | 2 |
| Limnochordia | 8 |
| Lineage_IV | 1 |
| M55-D21 | 12 |
| MAT-CR-H4-C10 | 2 |
| MBA03 | 60 |
| MBNT15 | 1 |
| MD2902-B12 | 2 |
| MVP-15 | 55 |
| Marinifilaceae | 1 |
| Marinilabiliaceae | 79 |
| Marinimicrobia_(SAR406_clade) | 30 |
| Methylococcaceae | 9 |
| Methylomonadaceae | 21 |
| Methylophilaceae | 25 |
| MgMjR-022 | 24 |
| Microbacteriaceae | 56 |
| Micrococcaceae | 2 |
| Micropepsaceae | 1 |
| Microscillaceae | 18 |
| Microtrichaceae | 47 |
| MidBa8 | 2 |
| Monoglobaceae | 12 |
| Moorellaceae | 1 |
| Moraxellaceae | 23 |
| Morganellaceae | 2 |
| Mycobacteriaceae | 27 |
| NB1-j | 2 |
| NK-L14 | 2 |
| NKB15 | 6 |
| Nannocystaceae | 1 |
| Neisseriaceae | 6 |
| Nitrosococcaceae | 5 |
| Nitrosomonadaceae | 30 |
| Nitrospiraceae | 15 |
| OLB14 | 14 |
| OM190 | 1 |
| OPB41 | 9 |
| Obscuribacteraceae | 2 |
| Oligosphaeraceae | 10 |
| Omnitrophaceae | 11 |
| Oscillospiraceae | 93 |
| Oxalobacteraceae | 3 |
| PB19 | 2 |
| PBS-18 | 4 |
| PHOS-HE36 | 21 |
| PLTA13 | 12 |
| Paludibacteraceae | 129 |
| Parachlamydiaceae | 1 |
| PeH15 | 8 |
| PeM15 | 22 |
| Pedosphaeraceae | 85 |
| Peptococcaceae | 26 |
| Peptostreptococcaceae | 18 |
| Peptostreptococcales-Tissierellales | 83 |
| Petrotogaceae | 95 |
| Phycisphaeraceae | 3 |
| Pirellulaceae | 51 |
| Pla3_lineage | 6 |
| Planococcaceae | 1 |
| Porticoccaceae | 5 |
| Prevotellaceae | 46 |
| Prolixibacteraceae | 93 |
| Propionibacteriaceae | 9 |
| Proteiniboraceae | 7 |
| Pseudomonadaceae | 25 |
| RBG-13-54-9 | 12 |
| RBG-16-55-12 | 8 |
| RF39 | 1 |
| Reyranellaceae | 4 |
| Rhizobiaceae | 25 |
| Rhizobiales_Incertae_Sedis | 24 |
| Rhodanobacteraceae | 65 |
| Rhodobacteraceae | 81 |
| Rhodocyclaceae | 89 |
| Rickettsiaceae | 1 |
| Rikenellaceae | 106 |
| Rubinisphaeraceae | 26 |
| Rubrobacteriaceae | 1 |
| Ruminococcaceae | 80 |
| Run-SP154 | 20 |
| S15A-MN91 | 22 |
| SB-5 | 18 |
| SBR1031 | 69 |
| SC-I-84 | 60 |
| SHA-4 | 2 |
| SHA-41 | 1 |
| SJA-15 | 50 |
| SJA-28 | 70 |
| SM1A07 | 1 |
| SRB2 | 14 |
| ST-12K33 | 81 |
| ST-NAGAB-D1 | 2 |
| Saccharimonadaceae | 5 |
| Saccharimonadales | 24 |
| Saprospiraceae | 75 |
| Schlesneriaceae | 2 |
| Sedimentibacteraceae | 129 |
| Selenomonadaceae | 2 |
| Simkaniaceae | 1 |
| Smithellaceae | 84 |
| Solibacteraceae | 2 |
| Sphingobacteriaceae | 2 |
| Sphingomonadaceae | 59 |
| Spirochaetaceae | 128 |
| Spongiibacteraceae | 34 |
| Sporichthyaceae | 9 |
| Sporolactobacillaceae | 1 |
| Sporomusaceae | 12 |
| Steroidobacteraceae | 7 |
| Streptococcaceae | 46 |
| Succinivibrionaceae | 1 |
| Sulfurimonadaceae | 2 |
| Sulfurospirillaceae | 2 |
| Sulfurovaceae | 1 |
| Sumerlaeaceae | 14 |
| Sutterellaceae | 43 |
| Sva0485 | 8 |
| Symbiobacteraceae | 4 |
| Synergistaceae | 136 |
| Syntrophaceae | 40 |
| Syntrophobacteraceae | 45 |
| Syntrophobotulaceae | 1 |
| Syntrophomonadaceae | 129 |
| Syntrophorhabdaceae | 80 |
| Syntrophotaleaceae | 8 |
| T34 | 1 |
| TA06 | 8 |
| TTA-B1 | 10 |
| Tannerellaceae | 65 |
| Thermaceae | 3 |
| Thermacetogeniaceae | 27 |
| Thermoanaerobaculaceae | 35 |
| Thermotaleaceae | 6 |
| Thermovenabulales | 23 |
| Thioalkalispiraceae | 1 |
| Trueperaceae | 2 |
| UASB-TL25 | 1 |
| UCG-010 | 72 |
| VHS-B3-70 | 1 |
| Veillonellaceae | 9 |
| Vermiphilaceae | 4 |
| Verrucomicrobiaceae | 1 |
| Vibrionaceae | 1 |
| W27 | 90 |
| WCHB1-02 | 79 |
| WCHB1-41 | 106 |
| WCHB1-81 | 39 |
| WD260 | 3 |
| WPS-2 | 4 |
| WS1 | 47 |
| WWE3 | 3 |
| Weeksellaceae | 29 |
| Williamwhitmaniaceae | 67 |
| Wohlfahrtiimonadaceae | 1 |
| Xanthobacteraceae | 39 |
| Xanthomonadaceae | 68 |
| Yersiniaceae | 4 |
| Z4MB62 | 2 |
| Zixibacteria | 5 |
| c5LKS83 | 11 |
| cvE6 | 1 |
| env.OPS_17 | 2 |
| p-251-o5 | 15 |
| unidentified | 137 |

Total microbiome features: 370.


## Data and methods
Matched samples: 138; site labels: 37; groups: 35.

Train: 109 samples, 28 groups. Test: 29 samples, 7 groups.

Training sites: ADA, ADP, BSE, BSI, BSN, BSO, BSS, CGS, CWS, DGB, DGC, DGD, DGG, DGJ, GCG, GHGa, GHGb, GHH, GHJ, GSG, JJN, JJY, MGY, USJ, USO, USY, YCG, YSY

Test sites: CWM, DGY, MYS, SCH, UJG, YCD, YSD

Missing ARC: 1-32' and 4-24; excluded explicitly in sample_audit.csv.

ARC header 2-29-SCHG-ARC has redundant domain suffix removed for site checks; logged in sample_id_corrections.csv.

Duplicate/missing metadata and normalized genomic IDs checked; SampleID and Site mapping validated.

QC replicates BSIb→BSI and DGYb→DGY; 35 groups, not the assumed 34 complete sites.

Final microbiome features: {'BAC_F': 346, 'ARC_F': 24}

Genomic/microbiome-only prediction: all metadata variables are excluded from X.

No missing Family values observed. Median imputation fitted within training folds as a safeguard. Scale KNN/SVM/NNET/GLMNET only; RF/XGBoost are unscaled.

RF/SVM/elastic-net logistic regression: balanced class weights; XGBoost: training negative/positive ratio.

KNN/NNET: per-label random oversampling on training rows only. Single-class training labels use constant prediction.

GLMNET denotes sklearn elastic-net logistic regression (saga), not the R glmnet package.

All supplied annotations remain eligible, including unidentified and off-domain taxa.

Existing warning labels retained; missing target-source measurements can produce unflagged zeros.

CV scores are tuning estimates, not nested unbiased performance estimates. Rare positives limit stability.

No additional ML feature selection. The fixed all-sample union is the explicitly permitted predefined >0.1% rule; test values do not fit preprocessing or influence tuning.

## Interpretation
RandomForest is the preferred model under the predeclared joint CV ranking. Test results describe generalization to the held-out groups and do not change that selection.
The test set contains 29 samples from 7 groups. Rare labels and few independent sites limit precision; no causal inference is supported.
The fixed holdout has appeared in earlier analyses. This run excludes it from preprocessing fitting, tuning, threshold optimization and model selection; the predeclared sample-wise union is permitted, but it is not a never-before-used external validation set.

## Reproducibility
Run `python ML/train_family_point1.py --seed 42 --n-iter 8`. See run_config.json for input hashes, versions and search spaces.
See split_and_cv_assignment.csv for every sample/site/fold, label_distribution.csv for label counts, selected_taxa.csv for the complete Family list, and README_family_point1.md for pipeline details.
Test metric leaders: {"Multilabel MCC": "SVM", "Micro-AUPR": "SVM", "Micro-ROC-AUC": "RandomForest"}.
The test metrics favor different models; there is no unanimous best test model. This comparison does not change CV selection.

Recorded fit warnings: 2; see fit_warnings.csv for candidate and fold details.

Fit warning counts: {('GLMNET', 'ConvergenceWarning'): 2}. See fit_warnings.csv; iteration-limit fits may not have converged.


## Train/test label distribution

| Warning | Training positives (109) | Test positives (29) |
|---|---:|---:|
| warning_acid_base_balance | 15 | 5 |
| warning_buffer_capacity | 8 | 1 |
| warning_acid_accumulation | 3 | 1 |
| warning_ammonia_toxicity | 22 | 4 |
| warning_biogas_quality | 5 | 2 |

## Seasonal coverage exceptions

| Site | Site group | Included samples | Missing seasons |
|---|---|---:|---|
| ADP | ADP | 3 | Spring |
| BSIb | BSI | 1 | Fall, Winter, Spring |
| DGYb | DGY | 1 | Fall, Winter, Spring |
| GHGa | GHGa | 3 | Spring |
| JJY | JJY | 3 | Summer |
| YCG | YCG | 3 | Spring |

CV selection metric leaders (MCC at 0.5): {"Multilabel MCC": "RandomForest", "Micro-AUPR": "RandomForest", "Micro-ROC-AUC": "RandomForest"}.

Artifact verification: run `python ML/verify_family_point1.py`; checks use saved probabilities with round-trip float parsing to preserve numerical ties.
