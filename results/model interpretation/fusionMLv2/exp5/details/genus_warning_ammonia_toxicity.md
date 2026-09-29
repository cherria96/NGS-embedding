# Ammonia toxicity: GENUS biological interpretation

Groups are specific to a fitted training partition. The examples below are the three largest training-score standard deviations per model/domain, restricted to active groups. This describes fitted score variation, not independent or causal importance. The table may include more than one fold-specific version of a taxon group. See the complete member tables for all groups.

A positive coefficient contributes toward flag=1 through a within-domain log-composition contrast. It does not mean that increasing absolute abundance causes failure. Group IDs are local; identical IDs across fits do not identify the same group.

## ARC fusion (`fusion_arc_only`)

Available fits: 3/3. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G0 | 6 | -0.115 | 0.259 | d__Eukaryota (unresolved genus); Unassigned; Unassigned; d__Eukaryota (unresolved genus); Unassigned; Unassigned | unassigned: 6 | 2 |
| 0:0:G1 | 1 | -0.098 | 0.231 | f__Bathyarchaeia (unresolved genus) (3) | unassigned: 1 | 0 |
| 0:0:G4 | 1 | 0.077 | 0.200 | Methanomassiliicoccus (3) | hydrogen-dependent methyl reduction: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:0:G70 | 1 | 0.284 | 0.725 | f__UCG-010 (unresolved genus) (3) | unassigned: 1 | 0 |
| 0:0:G6 | 1 | -0.172 | 0.563 | f__Prolixibacteraceae (unresolved genus) | unassigned: 1 | 0 |
| 0:0:G20 | 1 | -0.147 | 0.437 | Blvii28_wastewater-sludge_group | unassigned: 1 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAC | BAC::F00322 | f__UCG-010 (unresolved genus) (3) | 2 | 3 | 2 | 0 | unassigned |
| BAC | BAC::F00018 | f__Prolixibacteraceae (unresolved genus) | 3 | 3 | 0 | 3 | unassigned |
| BAC | BAC::F00121 | Caldicoprobacter (5) | 3 | 3 | 3 | 0 | unassigned |
| BAC | BAC::F00116 | Blvii28_wastewater-sludge_group | 3 | 3 | 0 | 3 | unassigned |
| BAC | BAC::F00297 | Syntrophomonas (4) | 3 | 3 | 3 | 0 | syntrophic fatty-acid oxidation |
| BAC | BAC::F00346 | f__Paludibacteraceae (unresolved genus) | 2 | 3 | 2 | 0 | unassigned |
| BAC | BAC::F00145 | DEV114 (2) | 1 | 3 | 1 | 0 | unassigned |
| BAC | BAC::F00128 | Candidatus_Cloacimonas (2) | 3 | 3 | 0 | 3 | unassigned |

## BAC fusion (`fusion_bac_only`)

Available fits: 3/3. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G6 | 1 | 0.302 | 0.859 | Methanoculleus (14) | hydrogenotrophic methanogenesis: 1 | 0 |
| 0:1:G7 | 1 | -0.194 | 0.651 | Methanolinea (6) | hydrogenotrophic methanogenesis: 1 | 0 |
| 0:0:G6 | 1 | 0.231 | 0.593 | Methanoculleus (14) | hydrogenotrophic methanogenesis: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:0:G3 | 6 | -0.636 | 1.342 | Lautropia; f__SC-I-84 (unresolved genus) (6); Zoogloea (3); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus) | unassigned: 6 | 0 |
| 0:1:G3 | 4 | -0.358 | 0.821 | Candidatus_Cloacimonas (2); Candidatus_Cloacimonas (2); f__W27 (unresolved genus); W5 (2) | unassigned: 4 | 0 |
| 0:0:G2 | 2 | -0.259 | 0.715 | Candidatus_Cloacimonas (2); Candidatus_Cloacimonas (2) | unassigned: 2 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARC | ARC::F00014 | Methanoculleus (14) | 3 | 3 | 3 | 0 | hydrogenotrophic methanogenesis |
| ARC | ARC::F00015 | Methanolinea (6) | 3 | 3 | 0 | 3 | hydrogenotrophic methanogenesis |
| BAC | BAC::F00203 | Lautropia | 1 | 3 | 0 | 1 | unassigned |
| BAC | BAC::F00268 | f__SC-I-84 (unresolved genus) (6) | 1 | 3 | 0 | 1 | unassigned |
| ARC | ARC::F00027 | Methanosarcina (5) | 3 | 3 | 3 | 0 | multiple methanogenic pathways |
| ARC | ARC::F00026 | Methanosaeta | 3 | 3 | 0 | 3 | acetoclastic methanogenesis |
| ARC | ARC::F00021 | Methanomethylovorans (2) | 3 | 3 | 2 | 1 | unassigned |
| BAC | BAC::F00324 | f__W27 (unresolved genus) | 1 | 3 | 0 | 1 | unassigned |

## Shared hybrid (`shared_hybrid`)

Available fits: 3/3. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G0 | 7 | -0.300 | 0.658 | d__Eukaryota (unresolved genus); Unassigned; Unassigned; Unassigned; d__Eukaryota (unresolved genus); Unassigned; Unassigned | unassigned: 7 | 2 |
| 0:1:G1 | 12 | 0.278 | 0.343 | f__Bathyarchaeia (unresolved genus) (3); Candidatus_Methanofastidiosum (5); Methanoculleus (14); Methanolinea (6); Methanolinea; Methanolinea; Methanomethylovorans (2); Methanosaeta (13); Methanosaeta; Methanosarcina (5); Methanosphaerula (3); Methanospiril… | acetoclastic methanogenesis: 2; hydrogen-dependent methylated-thiol reduction: 1; hydrogenotrophic methanogenesis: 4; multiple methanogenic pathways: 1; unassigned: 4 | 0 |
| 0:0:G1 | 2 | -0.143 | 0.282 | f__Bathyarchaeia (unresolved genus) (3); Candidatus_Methanofastidiosum (5) | hydrogen-dependent methylated-thiol reduction: 1; unassigned: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G4 | 4 | -0.901 | 2.070 | Candidatus_Cloacimonas (2); Candidatus_Cloacimonas (2); f__W27 (unresolved genus); W5 (2) | unassigned: 4 | 0 |
| 0:0:G5 | 6 | -0.514 | 1.085 | Lautropia; f__SC-I-84 (unresolved genus) (6); Zoogloea (3); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus) | unassigned: 6 | 0 |
| 0:0:G4 | 2 | -0.357 | 0.985 | Candidatus_Cloacimonas (2); Candidatus_Cloacimonas (2) | unassigned: 2 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAC | BAC::F00324 | f__W27 (unresolved genus) | 1 | 3 | 0 | 1 | unassigned |
| BAC | BAC::F00129 | Candidatus_Cloacimonas (2) | 3 | 3 | 0 | 3 | unassigned |
| BAC | BAC::F00128 | Candidatus_Cloacimonas (2) | 3 | 3 | 0 | 3 | unassigned |
| BAC | BAC::F00327 | W5 (2) | 2 | 3 | 0 | 2 | unassigned |
| BAC | BAC::F00108 | f__Bacteroidales_UCG-001 (unresolved genus) | 3 | 3 | 3 | 0 | unassigned |
| BAC | BAC::F00268 | f__SC-I-84 (unresolved genus) (6) | 1 | 3 | 0 | 1 | unassigned |
| BAC | BAC::F00203 | Lautropia | 1 | 3 | 0 | 1 | unassigned |
| ARC | ARC::F00009 | f__Bathyarchaeia (unresolved genus) (3) | 3 | 3 | 1 | 2 | unassigned |

## Domain separate (`domain_adaptive_hybrid`)

Available fits: 3/3. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G0 | 7 | -0.370 | 0.810 | d__Eukaryota (unresolved genus); Unassigned; Unassigned; Unassigned; d__Eukaryota (unresolved genus); Unassigned; Unassigned | unassigned: 7 | 2 |
| 0:2:G6 | 1 | 0.196 | 0.579 | Methanosarcina (5) | multiple methanogenic pathways: 1 | 0 |
| 0:2:G5 | 1 | -0.120 | 0.329 | Methanosaeta | acetoclastic methanogenesis: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G11 | 7 | -0.333 | 0.666 | Acidovorax; Dechloromonas; Thauera (3); Zoogloea (3); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus) | unassigned: 7 | 0 |
| 0:0:G34 | 1 | 0.258 | 0.659 | f__UCG-010 (unresolved genus) (3) | unassigned: 1 | 0 |
| 0:0:G21 | 6 | -0.260 | 0.548 | Lautropia; f__SC-I-84 (unresolved genus) (6); Zoogloea (3); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus) | unassigned: 6 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAC | BAC::F00254 | Rikenellaceae_RC9_gut_group (2) | 1 | 3 | 1 | 0 | unassigned |
| BAC | BAC::F00322 | f__UCG-010 (unresolved genus) (3) | 2 | 3 | 2 | 0 | unassigned |
| ARC | ARC::F00027 | Methanosarcina (5) | 3 | 3 | 3 | 0 | multiple methanogenic pathways |
| BAC | BAC::F00297 | Syntrophomonas (4) | 3 | 3 | 3 | 0 | syntrophic fatty-acid oxidation |
| BAC | BAC::F00116 | Blvii28_wastewater-sludge_group | 3 | 3 | 0 | 3 | unassigned |
| BAC | BAC::F00018 | f__Prolixibacteraceae (unresolved genus) | 3 | 3 | 0 | 3 | unassigned |
| BAC | BAC::F00267 | SC103 (3) | 3 | 3 | 0 | 3 | unassigned |
| BAC | BAC::F00129 | Candidatus_Cloacimonas (2) | 3 | 3 | 1 | 2 | unassigned |

## Domain interaction (`domain_adaptive_interaction`)

Available fits: 3/3. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G0 | 7 | -0.371 | 0.813 | d__Eukaryota (unresolved genus); Unassigned; Unassigned; Unassigned; d__Eukaryota (unresolved genus); Unassigned; Unassigned | unassigned: 7 | 2 |
| 0:2:G9 | 1 | 0.250 | 0.740 | Methanosarcina (5) | multiple methanogenic pathways: 1 | 0 |
| 0:2:G8 | 1 | -0.195 | 0.534 | Methanosaeta | acetoclastic methanogenesis: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:2:G5 | 7 | 0.485 | 1.186 | Acholeplasma (7); Fermentimonas; Rikenellaceae_RC9_gut_group (2); Sphaerochaeta (4); f__UCG-010 (unresolved genus) (3); f__Paludibacteraceae (unresolved genus); f__uncultured (unresolved genus) | unassigned: 7 | 0 |
| 0:1:G11 | 7 | -0.332 | 0.664 | Acidovorax; Dechloromonas; Thauera (3); Zoogloea (3); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus) | unassigned: 7 | 0 |
| 0:0:G34 | 1 | 0.257 | 0.658 | f__UCG-010 (unresolved genus) (3) | unassigned: 1 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAC | BAC::F00322 | f__UCG-010 (unresolved genus) (3) | 2 | 3 | 2 | 0 | unassigned |
| ARC | ARC::F00027 | Methanosarcina (5) | 3 | 3 | 3 | 0 | multiple methanogenic pathways |
| BAC | BAC::F00297 | Syntrophomonas (4) | 3 | 3 | 3 | 0 | syntrophic fatty-acid oxidation |
| BAC | BAC::F00018 | f__Prolixibacteraceae (unresolved genus) | 3 | 3 | 0 | 3 | unassigned |
| BAC | BAC::F00031 | f__Dysgonomonadaceae (unresolved genus) | 3 | 3 | 3 | 0 | unassigned |
| BAC | BAC::F00116 | Blvii28_wastewater-sludge_group | 3 | 3 | 0 | 3 | unassigned |
| ARC | ARC::F00026 | Methanosaeta | 3 | 3 | 1 | 2 | acetoclastic methanogenesis |
| BAC | BAC::F00129 | Candidatus_Cloacimonas (2) | 3 | 3 | 1 | 2 | unassigned |

## Interaction interpretation

| repeat | outer_fold | status | fitted_products | active_products |
| --- | --- | --- | --- | --- |
| 0 | 0 | ok | 4 | 0 |
| 0 | 1 | ok | 4 | 0 |
| 0 | 2 | ok | 4 | 0 |

No active product means no fitted interaction to interpret. Even an active centered-balance product would be a statistical interaction, not proof of syntrophy or direct metabolic exchange.
