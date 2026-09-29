# Biogas quality: GENUS biological interpretation

Groups are specific to a fitted training partition. The examples below are the three largest training-score standard deviations per model/domain, restricted to active groups. This describes fitted score variation, not independent or causal importance. The table may include more than one fold-specific version of a taxon group. See the complete member tables for all groups.

A positive coefficient contributes toward flag=1 through a within-domain log-composition contrast. It does not mean that increasing absolute abundance causes failure. Group IDs are local; identical IDs across fits do not identify the same group.

## ARC fusion (`fusion_arc_only`)

Available fits: 3/3. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:0:G2 | 1 | -0.082 | 0.212 | Methanomassiliicoccus (3) | hydrogen-dependent methyl reduction: 1 | 0 |
| 0:0:G0 | 7 | 0.082 | 0.178 | d__Eukaryota (unresolved genus); Unassigned; Unassigned; Unassigned; d__Eukaryota (unresolved genus); Unassigned; Unassigned | unassigned: 7 | 2 |
| 0:2:G2 | 10 | 0.145 | 0.156 | Methanoculleus (14); Methanolinea (6); Methanolinea; Methanolinea; Methanomethylovorans (2); Methanosaeta (13); Methanosaeta; Methanosarcina (5); Methanosphaerula (3); Methanospirillum (15) | acetoclastic methanogenesis: 2; hydrogenotrophic methanogenesis: 4; multiple methanogenic pathways: 1; unassigned: 3 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G69 | 1 | -0.451 | 1.421 | Thermovirga (3) | amino-acid degradation: 1 | 0 |
| 0:1:G71 | 1 | 0.413 | 1.103 | f__W27 (unresolved genus) | unassigned: 1 | 0 |
| 0:0:G34 | 1 | 0.350 | 1.027 | Fermentimonas | unassigned: 1 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAC | BAC::F00324 | f__W27 (unresolved genus) | 1 | 3 | 1 | 0 | unassigned |
| BAC | BAC::F00145 | DEV114 (2) | 1 | 3 | 1 | 0 | unassigned |
| BAC | BAC::F00147 | f__DTU014 (unresolved genus) (6) | 3 | 3 | 0 | 3 | unassigned |
| BAC | BAC::F00168 | Fermentimonas | 2 | 3 | 2 | 0 | unassigned |
| BAC | BAC::F00314 | Thermovirga (3) | 3 | 3 | 0 | 3 | amino-acid degradation |
| BAC | BAC::F00132 | Christensenellaceae_R-7_group (29) | 3 | 3 | 1 | 2 | unassigned |
| BAC | BAC::F00052 | f__Paludibacteraceae (unresolved genus) | 2 | 3 | 2 | 0 | unassigned |
| BAC | BAC::F00332 | f__WS1 (unresolved genus) | 2 | 3 | 2 | 0 | unassigned |

## BAC fusion (`fusion_bac_only`)

Available fits: 3/3. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G13 | 1 | 0.272 | 0.851 | Methanosaeta | acetoclastic methanogenesis: 1 | 0 |
| 0:0:G16 | 1 | 0.340 | 0.766 | Methanospirillum (15) | unassigned: 1 | 0 |
| 0:1:G8 | 1 | -0.197 | 0.715 | Methanolinea | hydrogenotrophic methanogenesis: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:0:G4 | 6 | 0.318 | 0.672 | Lautropia; f__SC-I-84 (unresolved genus) (6); Zoogloea (3); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus) | unassigned: 6 | 0 |
| 0:1:G1 | 8 | -0.233 | 0.443 | Acidovorax; Dechloromonas; Thauera (3); Thermomonas (4); Zoogloea (3); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus) | unassigned: 8 | 0 |
| 0:1:G2 | 4 | 0.177 | 0.407 | Candidatus_Cloacimonas (2); Candidatus_Cloacimonas (2); f__W27 (unresolved genus); W5 (2) | unassigned: 4 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARC | ARC::F00029 | Methanospirillum (15) | 3 | 3 | 3 | 0 | unassigned |
| ARC | ARC::F00026 | Methanosaeta | 3 | 3 | 3 | 0 | acetoclastic methanogenesis |
| ARC | ARC::F00009 | f__Bathyarchaeia (unresolved genus) (3) | 3 | 3 | 0 | 3 | unassigned |
| ARC | ARC::F00027 | Methanosarcina (5) | 3 | 3 | 3 | 0 | multiple methanogenic pathways |
| ARC | ARC::F00016 | Methanolinea | 3 | 3 | 1 | 2 | hydrogenotrophic methanogenesis |
| ARC | ARC::F00015 | Methanolinea (6) | 3 | 3 | 0 | 3 | hydrogenotrophic methanogenesis |
| ARC | ARC::F00010 | Candidatus_Methanofastidiosum (5) | 2 | 3 | 2 | 0 | hydrogen-dependent methylated-thiol reduction |
| ARC | ARC::F00020 | Methanomassiliicoccus (3) | 3 | 3 | 0 | 3 | hydrogen-dependent methyl reduction |

## Shared hybrid (`shared_hybrid`)

Available fits: 3/3. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:2:G3 | 4 | 0.421 | 0.557 | Methanomethylovorans (2); Methanosaeta (13); Methanosaeta; Methanosarcina (5) | acetoclastic methanogenesis: 2; multiple methanogenic pathways: 1; unassigned: 1 | 0 |
| 0:0:G4 | 1 | -0.198 | 0.513 | Methanomassiliicoccus (3) | hydrogen-dependent methyl reduction: 1 | 0 |
| 0:0:G1 | 1 | -0.193 | 0.453 | f__Bathyarchaeia (unresolved genus) (3) | unassigned: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G6 | 4 | 0.362 | 0.832 | Candidatus_Cloacimonas (2); Candidatus_Cloacimonas (2); f__W27 (unresolved genus); W5 (2) | unassigned: 4 | 0 |
| 0:1:G0 | 62 | -0.563 | 0.484 | f__009E01-B-SD-P15 (unresolved genus); f__Spirochaetaceae (unresolved genus); f__uncultured (unresolved genus); f__Marinilabiliaceae (unresolved genus); f__Prolixibacteraceae (unresolved genus); f__Dysgonomonadaceae (unresolved genus); f__Hungateiclostridia… | proteolytic fermentation: 1; syntrophic fatty-acid oxidation: 2; unassigned: 59 | 0 |
| 0:1:G9 | 2 | 0.180 | 0.471 | f__SBR1031 (unresolved genus) (6); f__SJA-15 (unresolved genus) (3) | unassigned: 2 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARC | ARC::F00020 | Methanomassiliicoccus (3) | 3 | 3 | 0 | 3 | hydrogen-dependent methyl reduction |
| BAC | BAC::F00324 | f__W27 (unresolved genus) | 1 | 3 | 1 | 0 | unassigned |
| BAC | BAC::F00270 | f__SJA-15 (unresolved genus) (3) | 1 | 3 | 1 | 0 | unassigned |
| ARC | ARC::F00009 | f__Bathyarchaeia (unresolved genus) (3) | 3 | 3 | 0 | 3 | unassigned |
| BAC | BAC::F00327 | W5 (2) | 2 | 3 | 2 | 0 | unassigned |
| BAC | BAC::F00265 | f__SBR1031 (unresolved genus) (6) | 3 | 3 | 2 | 1 | unassigned |
| ARC | ARC::F00026 | Methanosaeta | 3 | 3 | 3 | 0 | acetoclastic methanogenesis |
| ARC | ARC::F00027 | Methanosarcina (5) | 3 | 3 | 3 | 0 | multiple methanogenic pathways |

## Domain separate (`domain_adaptive_hybrid`)

Available fits: 3/3. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G10 | 1 | 0.218 | 0.680 | Methanosaeta | acetoclastic methanogenesis: 1 | 0 |
| 0:2:G4 | 3 | 0.396 | 0.635 | Methanomethylovorans (2); Methanosaeta (13); Methanosaeta | acetoclastic methanogenesis: 2; unassigned: 1 | 0 |
| 0:1:G3 | 3 | 0.328 | 0.561 | Candidatus_Methanofastidiosum (5); Methanosaeta (13); Methanospirillum (15) | acetoclastic methanogenesis: 1; hydrogen-dependent methylated-thiol reduction: 1; unassigned: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:2:G22 | 2 | 0.253 | 0.625 | Smithella (3); Smithella (5) | unassigned: 2 | 0 |
| 0:2:G11 | 1 | 0.206 | 0.564 | Fermentimonas | unassigned: 1 | 0 |
| 0:1:G30 | 2 | 0.208 | 0.543 | f__SBR1031 (unresolved genus) (6); f__SJA-15 (unresolved genus) (3) | unassigned: 2 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAC | BAC::F00168 | Fermentimonas | 2 | 3 | 2 | 0 | unassigned |
| BAC | BAC::F00147 | f__DTU014 (unresolved genus) (6) | 3 | 3 | 0 | 3 | unassigned |
| ARC | ARC::F00026 | Methanosaeta | 3 | 3 | 3 | 0 | acetoclastic methanogenesis |
| BAC | BAC::F00145 | DEV114 (2) | 1 | 3 | 1 | 0 | unassigned |
| BAC | BAC::F00324 | f__W27 (unresolved genus) | 1 | 3 | 1 | 0 | unassigned |
| ARC | ARC::F00020 | Methanomassiliicoccus (3) | 3 | 3 | 0 | 3 | hydrogen-dependent methyl reduction |
| ARC | ARC::F00009 | f__Bathyarchaeia (unresolved genus) (3) | 3 | 3 | 0 | 3 | unassigned |
| ARC | ARC::F00027 | Methanosarcina (5) | 3 | 3 | 3 | 0 | multiple methanogenic pathways |

## Domain interaction (`domain_adaptive_interaction`)

Available fits: 3/3. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:2:G7 | 1 | 0.218 | 0.646 | Methanosarcina (5) | multiple methanogenic pathways: 1 | 0 |
| 0:1:G5 | 4 | 0.427 | 0.609 | Methanomethylovorans (2); Methanosaeta (13); Methanosaeta; Methanosarcina (5) | acetoclastic methanogenesis: 2; multiple methanogenic pathways: 1; unassigned: 1 | 0 |
| 0:1:G1 | 1 | -0.215 | 0.498 | f__Bathyarchaeia (unresolved genus) (3) | unassigned: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:2:G22 | 2 | 0.238 | 0.589 | Smithella (3); Smithella (5) | unassigned: 2 | 0 |
| 0:2:G11 | 1 | 0.198 | 0.543 | Fermentimonas | unassigned: 1 | 0 |
| 0:1:G33 | 2 | 0.206 | 0.537 | f__SBR1031 (unresolved genus) (6); f__SJA-15 (unresolved genus) (3) | unassigned: 2 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAC | BAC::F00168 | Fermentimonas | 2 | 3 | 2 | 0 | unassigned |
| ARC | ARC::F00027 | Methanosarcina (5) | 3 | 3 | 3 | 0 | multiple methanogenic pathways |
| BAC | BAC::F00147 | f__DTU014 (unresolved genus) (6) | 3 | 3 | 0 | 3 | unassigned |
| BAC | BAC::F00145 | DEV114 (2) | 1 | 3 | 1 | 0 | unassigned |
| BAC | BAC::F00324 | f__W27 (unresolved genus) | 1 | 3 | 1 | 0 | unassigned |
| ARC | ARC::F00026 | Methanosaeta | 3 | 3 | 3 | 0 | acetoclastic methanogenesis |
| ARC | ARC::F00020 | Methanomassiliicoccus (3) | 3 | 3 | 0 | 3 | hydrogen-dependent methyl reduction |
| BAC | BAC::F00052 | f__Paludibacteraceae (unresolved genus) | 2 | 3 | 2 | 0 | unassigned |

## Interaction interpretation

| repeat | outer_fold | status | fitted_products | active_products |
| --- | --- | --- | --- | --- |
| 0 | 0 | ok | 4 | 0 |
| 0 | 1 | ok | 4 | 0 |
| 0 | 2 | ok | 4 | 0 |

No active product means no fitted interaction to interpret. Even an active centered-balance product would be a statistical interaction, not proof of syntrophy or direct metabolic exchange.
