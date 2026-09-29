# Per-label feature importance

Primary model: **NNET**, chosen by the original training CV Macro AUPR. All six models are compared using their saved best hyperparameters.

No warning-model permutation importance or SHAP outputs existed before this analysis. Earlier importance outputs in `ML/results_selected/` belong to the regression experiment.

Method: 20 repeated permutations per input feature in each of five held-out training CV folds. Importance is baseline average precision minus permuted average precision, independently for each binary label. Rankings average the evaluable fold means with equal fold weights.

No independent test rows are fitted, scored, or permuted. This is post-hoc CV interpretation, not a feature-selection run. Best hyperparameters were previously selected on these folds, so the interpretation is conditional on that tuning and not nested-CV validation of a newly selected subset.

Taxon masks/unions and preprocessing are fitted on each training fold and frozen before permutation. We permute the resulting classifier input feature, not the raw abundance followed by reranking. One-hot columns for a categorical variable move together. This answers which supplied inputs the classifier relies on. All training-union input features receive a ranking; a taxon absent from a fold model has zero effect in that fold.

Permutations shuffle rows within the held-out fold; training remains grouped by site. They do not preserve within-site trajectories or abundance compositional constraints. Correlated metadata, phyla/genera, and taxa can dilute or redistribute importance; importance is not causal and does not indicate whether a feature raises or lowers biological risk.

Folds without positive labels have undefined AP and are excluded for that label. Feature eligibility and evaluable-fold counts are reported separately. Negative values mean shuffling improved AP. Error bars are fold SD, not confidence intervals; repeat-level variation is also saved. Zero effects and tied ranks are retained, not interpreted as evidence of biological irrelevance. Positive-only top-10 stability excludes zero ties.

Taxonomic caution: BAC/ARC categories identify source assays, not verified domains. The ARC workbook OTUs sheet labels Holozoa and Rhinosporideacae as Eukaryota; they must not be interpreted as archaeal biomarkers. Their small acid-accumulation importance is based on only two evaluable folds.

Observed NNET consistency is low: mean pairwise rank correlations range from approximately -0.03 to 0.07, and positive top-10 Jaccard overlaps from 0 to 0.10. None of the five rankings is strongly consistent across folds.

## Most influential features for the CV-selected model

### warning_acid_base_balance

Evaluable folds: 5; mean pairwise rank correlation: 0.071; positive top-10 Jaccard overlap: 0.034.

| Feature | Category | Mean AUPR decrease | Positive folds | Top-10 folds |
| --- | --- | --- | --- | --- |
| BAC_G::Lactobacillus | Bacteria | 0.1206 | 1/5 | 1/5 |
| ARC_G::Methanimicrococcus | Archaea | 0.0642 | 1/5 | 1/5 |
| BAC_G::Alkaliphilus | Bacteria | 0.0259 | 2/5 | 2/5 |
| BAC_G::Guggenheimella | Bacteria | 0.0176 | 2/5 | 2/5 |
| BAC_G::Gelria | Bacteria | 0.0127 | 2/5 | 2/5 |
| BAC_G::WCHB1-41 | Bacteria | 0.0126 | 2/5 | 2/5 |
| BAC_G::Bacteroides | Bacteria | 0.0089 | 1/5 | 1/5 |
| BAC_G::Sphaerochaeta | Bacteria | 0.0087 | 2/5 | 1/5 |
| substrate_avg_pH | Metadata: substrate | 0.0064 | 1/5 | 1/5 |
| HRT_d | Metadata: operation/context | 0.0056 | 2/5 | 2/5 |

### warning_buffer_capacity

Evaluable folds: 5; mean pairwise rank correlation: 0.041; positive top-10 Jaccard overlap: 0.040.

| Feature | Category | Mean AUPR decrease | Positive folds | Top-10 folds |
| --- | --- | --- | --- | --- |
| BAC_P::Spirochaetota | Bacteria | 0.0024 | 2/5 | 1/5 |
| ARC_P::Thermoplasmatota | Archaea | 0.0016 | 2/5 | 1/5 |
| BAC_G::Pseudomonas | Bacteria | 0.0015 | 1/5 | 1/5 |
| ARC_G::Methanomassiliicoccus | Archaea | 0.0014 | 3/5 | 2/5 |
| BAC_G::Syner-01 | Bacteria | 0.0014 | 2/5 | 2/5 |
| BAC_G::Sedimentibacter | Bacteria | 0.0014 | 1/5 | 1/5 |
| substrate_avg_Protein | Metadata: substrate | 0.0011 | 3/5 | 3/5 |
| BAC_P::Proteobacteria | Bacteria | 0.0010 | 2/5 | 1/5 |
| BAC_G::SJA-15 | Bacteria | 0.0010 | 1/5 | 1/5 |
| BAC_G::SC103 | Bacteria | 0.0010 | 2/5 | 1/5 |

### warning_acid_accumulation

Evaluable folds: 2; mean pairwise rank correlation: -0.032; positive top-10 Jaccard overlap: 0.000.

| Feature | Category | Mean AUPR decrease | Positive folds | Top-10 folds |
| --- | --- | --- | --- | --- |
| ARC_G::Rhinosporideacae | Archaea | 0.0022 | 1/2 | 1/2 |
| ARC_P::Holozoa | Archaea | 0.0022 | 1/2 | 1/2 |
| BAC_G::Gelria | Bacteria | 0.0011 | 1/2 | 1/2 |
| BAC_G::Syntrophomonas | Bacteria | 0.0010 | 1/2 | 1/2 |
| ARC_G::Candidatus_Nitrocosmicus | Archaea | 0.0007 | 1/2 | 1/2 |
| BAC_G::Guggenheimella | Bacteria | 0.0007 | 1/2 | 1/2 |
| BAC_G::MVP-15 | Bacteria | 0.0006 | 1/2 | 1/2 |
| BAC_G::ST-12K33 | Bacteria | 0.0006 | 1/2 | 1/2 |
| substrate_avg_ALK | Metadata: substrate | 0.0004 | 1/2 | 1/2 |
| BAC_G::D8A-2 | Bacteria | 0.0001 | 1/2 | 1/2 |

### warning_ammonia_toxicity

Evaluable folds: 5; mean pairwise rank correlation: 0.043; positive top-10 Jaccard overlap: 0.096.

| Feature | Category | Mean AUPR decrease | Positive folds | Top-10 folds |
| --- | --- | --- | --- | --- |
| ARC_G::Methanimicrococcus | Archaea | 0.0461 | 1/5 | 1/5 |
| BAC_G::Keratinibaculum | Bacteria | 0.0323 | 1/5 | 1/5 |
| BAC_G::W5053 | Bacteria | 0.0286 | 3/5 | 3/5 |
| BAC_G::Proteiniphilum | Bacteria | 0.0190 | 2/5 | 1/5 |
| substrate_avg_sCOD | Metadata: substrate | 0.0151 | 4/5 | 3/5 |
| substrate_avg_VS | Metadata: substrate | 0.0149 | 4/5 | 1/5 |
| BAC_G::HN-HF0106 | Bacteria | 0.0145 | 2/5 | 2/5 |
| BAC_G::Acholeplasma | Bacteria | 0.0134 | 3/5 | 2/5 |
| BAC_G::Anaerovorax | Bacteria | 0.0133 | 3/5 | 3/5 |
| BAC_G::UCG-010 | Bacteria | 0.0132 | 3/5 | 2/5 |

### warning_biogas_quality

Evaluable folds: 3; mean pairwise rank correlation: 0.024; positive top-10 Jaccard overlap: 0.018.

| Feature | Category | Mean AUPR decrease | Positive folds | Top-10 folds |
| --- | --- | --- | --- | --- |
| BAC_G::Atopobium | Bacteria | 0.1737 | 1/3 | 1/3 |
| BAC_G::Bacteroides | Bacteria | 0.0392 | 2/3 | 1/3 |
| ARC_G::Methanomethylovorans | Archaea | 0.0375 | 2/3 | 2/3 |
| BAC_G::Petrimonas | Bacteria | 0.0370 | 1/3 | 1/3 |
| BAC_G::Sphaerochaeta | Bacteria | 0.0370 | 1/3 | 1/3 |
| BAC_P::unidentified | Bacteria | 0.0370 | 1/3 | 1/3 |
| ARC_G::Methanospirillum | Archaea | 0.0363 | 1/3 | 1/3 |
| BAC_P::Thermotogota | Bacteria | 0.0352 | 1/3 | 1/3 |
| ARC_G::SCGC_AAA011-D5 | Archaea | 0.0333 | 1/3 | 1/3 |
| BAC_G::Fermentimonas | Bacteria | 0.0333 | 1/3 | 1/3 |

## Reading fold consistency

Rank correlation near zero and low top-10 overlap indicate unstable rankings. Compare positive-fold frequencies and taxon eligibility before claiming consistency. Small validation samples and very rare labels make fine-grained rankings exploratory. Full pairwise values and summaries are in the stability CSV files.

## Files

Each model directory contains per-label full/top10/top20 CSVs and top-20 PNG/PDF plots, fold-level importance, and compressed repeat-level CSV data. Root files contain all-model rankings, cross-model figures, baseline scores and fold stability. `index.html` provides a local gallery.
