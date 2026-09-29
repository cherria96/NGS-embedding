# Acid base balance: GENUS biological interpretation

Groups are specific to a fitted training partition. The examples below are the three largest training-score standard deviations per model/domain, restricted to active groups. This describes fitted score variation, not independent or causal importance. The table may include more than one fold-specific version of a taxon group. See the complete member tables for all groups.

A positive coefficient contributes toward flag=1 through a within-domain log-composition contrast. It does not mean that increasing absolute abundance causes failure. Group IDs are local; identical IDs across fits do not identify the same group.

## ARC fusion (`fusion_arc_only`)

Available fits: 3/3. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:0:G0 | 7 | 0.047 | 0.102 | d__Eukaryota (unresolved genus); Unassigned; Unassigned; Unassigned; d__Eukaryota (unresolved genus); Unassigned; Unassigned | unassigned: 7 | 2 |
| 0:1:G0 | 6 | -0.045 | 0.101 | d__Eukaryota (unresolved genus); Unassigned; Unassigned; d__Eukaryota (unresolved genus); Unassigned; Unassigned | unassigned: 6 | 2 |
| 0:1:G2 | 5 | 0.062 | 0.091 | f__Bathyarchaeia (unresolved genus) (3); Candidatus_Methanofastidiosum (5); Methanomethylovorans (2); Methanosaeta (13); Methanospirillum (15) | acetoclastic methanogenesis: 1; hydrogen-dependent methylated-thiol reduction: 1; unassigned: 3 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:0:G49 | 1 | 0.072 | 0.216 | Proteiniphilum (5) | proteolytic fermentation: 1 | 0 |
| 0:2:G58 | 1 | -0.066 | 0.186 | f__WCHB1-02 (unresolved genus) (3) | unassigned: 1 | 0 |
| 0:0:G19 | 1 | -0.074 | 0.166 | f__Bacteroidetes_vadinHA17 (unresolved genus) (12) | unassigned: 1 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAC | BAC::F00122 | Caldicoprobacter (4) | 2 | 3 | 2 | 0 | unassigned |
| BAC | BAC::F00251 | Proteiniphilum (5) | 3 | 3 | 3 | 0 | proteolytic fermentation |
| BAC | BAC::F00121 | Caldicoprobacter (5) | 3 | 3 | 3 | 0 | unassigned |
| BAC | BAC::F00115 | f__Bacteroidetes_vadinHA17 (unresolved genus) (12) | 3 | 3 | 0 | 3 | unassigned |
| BAC | BAC::F00329 | f__WCHB1-02 (unresolved genus) (3) | 3 | 3 | 0 | 3 | unassigned |
| BAC | BAC::F00132 | Christensenellaceae_R-7_group (29) | 3 | 3 | 0 | 3 | unassigned |
| BAC | BAC::F00114 | f__Bacteroidetes_vadinHA17 (unresolved genus) (4) | 3 | 3 | 0 | 3 | unassigned |
| BAC | BAC::F00263 | f__SAR324_clade(Marine_group_B) (unresolved genus) (3) | 3 | 3 | 0 | 3 | unassigned |

## BAC fusion (`fusion_bac_only`)

Available fits: 3/3. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G12 | 1 | 0.204 | 0.397 | Methanosaeta (13) | acetoclastic methanogenesis: 1 | 0 |
| 0:1:G16 | 1 | -0.187 | 0.390 | Methanospirillum (15) | unassigned: 1 | 0 |
| 0:1:G11 | 1 | 0.161 | 0.376 | Methanomethylovorans (2) | unassigned: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G17 | 2 | -0.254 | 0.807 | Mesotoga (2); f__SAR324_clade(Marine_group_B) (unresolved genus) (3) | unassigned: 2 | 0 |
| 0:1:G28 | 1 | 0.191 | 0.601 | Thermovirga (3) | amino-acid degradation: 1 | 0 |
| 0:1:G30 | 1 | 0.183 | 0.458 | f__[Eubacterium]_coprostanoligenes_group (unresolved genus) (5) | unassigned: 1 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARC | ARC::F00024 | Methanosaeta (13) | 3 | 3 | 2 | 1 | acetoclastic methanogenesis |
| ARC | ARC::F00021 | Methanomethylovorans (2) | 3 | 3 | 3 | 0 | unassigned |
| ARC | ARC::F00029 | Methanospirillum (15) | 3 | 3 | 0 | 3 | unassigned |
| BAC | BAC::F00314 | Thermovirga (3) | 3 | 3 | 3 | 0 | amino-acid degradation |
| BAC | BAC::F00334 | f__[Eubacterium]_coprostanoligenes_group (unresolved genus) (5) | 3 | 3 | 3 | 0 | unassigned |
| ARC | ARC::F00015 | Methanolinea (6) | 3 | 3 | 0 | 3 | hydrogenotrophic methanogenesis |
| BAC | BAC::F00079 | ADurb.Bin063-1 | 2 | 3 | 2 | 0 | unassigned |
| BAC | BAC::F00263 | f__SAR324_clade(Marine_group_B) (unresolved genus) (3) | 3 | 3 | 0 | 3 | unassigned |

## Shared hybrid (`shared_hybrid`)

Available fits: 3/3. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:2:G3 | 1 | -0.088 | 0.228 | Methanomassiliicoccus (3) | hydrogen-dependent methyl reduction: 1 | 0 |
| 0:0:G0 | 7 | 0.087 | 0.189 | d__Eukaryota (unresolved genus); Unassigned; Unassigned; Unassigned; d__Eukaryota (unresolved genus); Unassigned; Unassigned | unassigned: 7 | 2 |
| 0:2:G4 | 4 | 0.123 | 0.163 | Methanomethylovorans (2); Methanosaeta (13); Methanosaeta; Methanosarcina (5) | acetoclastic methanogenesis: 2; multiple methanogenic pathways: 1; unassigned: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:0:G3 | 2 | -0.162 | 0.446 | Candidatus_Cloacimonas (2); Candidatus_Cloacimonas (2) | unassigned: 2 | 0 |
| 0:0:G7 | 1 | 0.149 | 0.445 | Proteiniphilum (5) | proteolytic fermentation: 1 | 0 |
| 0:2:G3 | 1 | 0.097 | 0.299 | Proteiniphilum (5) | proteolytic fermentation: 1 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAC | BAC::F00251 | Proteiniphilum (5) | 3 | 3 | 3 | 0 | proteolytic fermentation |
| BAC | BAC::F00129 | Candidatus_Cloacimonas (2) | 3 | 3 | 0 | 3 | unassigned |
| BAC | BAC::F00128 | Candidatus_Cloacimonas (2) | 2 | 3 | 0 | 2 | unassigned |
| ARC | ARC::F00020 | Methanomassiliicoccus (3) | 2 | 3 | 1 | 1 | hydrogen-dependent methyl reduction |
| ARC | ARC::F00009 | f__Bathyarchaeia (unresolved genus) (3) | 3 | 3 | 0 | 3 | unassigned |
| ARC | ARC::F00010 | Candidatus_Methanofastidiosum (5) | 3 | 3 | 1 | 2 | hydrogen-dependent methylated-thiol reduction |
| BAC | BAC::F00263 | f__SAR324_clade(Marine_group_B) (unresolved genus) (3) | 2 | 3 | 0 | 2 | unassigned |
| BAC | BAC::F00327 | W5 (2) | 2 | 3 | 0 | 2 | unassigned |

## Domain separate (`domain_adaptive_hybrid`)

Available fits: 3/3. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:2:G5 | 4 | 0.431 | 0.572 | Methanomethylovorans (2); Methanosaeta (13); Methanosaeta; Methanosarcina (5) | acetoclastic methanogenesis: 2; multiple methanogenic pathways: 1; unassigned: 1 | 0 |
| 0:1:G5 | 4 | 0.328 | 0.468 | Methanomethylovorans (2); Methanosaeta (13); Methanosaeta; Methanosarcina (5) | acetoclastic methanogenesis: 2; multiple methanogenic pathways: 1; unassigned: 1 | 0 |
| 0:2:G4 | 1 | -0.174 | 0.454 | Methanomassiliicoccus (3) | hydrogen-dependent methyl reduction: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G26 | 3 | -0.345 | 0.873 | Mesotoga (2); SC103 (3); f__WCHB1-02 (unresolved genus) (3) | unassigned: 3 | 0 |
| 0:1:G7 | 8 | -0.394 | 0.750 | Acidovorax; Dechloromonas; Thauera (3); Thermomonas (4); Zoogloea (3); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus) | unassigned: 8 | 0 |
| 0:2:G28 | 1 | 0.207 | 0.642 | Proteiniphilum (5) | proteolytic fermentation: 1 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAC | BAC::F00251 | Proteiniphilum (5) | 3 | 3 | 3 | 0 | proteolytic fermentation |
| BAC | BAC::F00263 | f__SAR324_clade(Marine_group_B) (unresolved genus) (3) | 3 | 3 | 0 | 3 | unassigned |
| BAC | BAC::F00267 | SC103 (3) | 3 | 3 | 0 | 3 | unassigned |
| BAC | BAC::F00329 | f__WCHB1-02 (unresolved genus) (3) | 3 | 3 | 0 | 3 | unassigned |
| BAC | BAC::F00079 | ADurb.Bin063-1 | 2 | 3 | 2 | 0 | unassigned |
| BAC | BAC::F00256 | Rikenellaceae_RC9_gut_group (7) | 3 | 3 | 1 | 2 | unassigned |
| BAC | BAC::F00314 | Thermovirga (3) | 3 | 3 | 3 | 0 | amino-acid degradation |
| BAC | BAC::F00147 | f__DTU014 (unresolved genus) (6) | 3 | 3 | 1 | 2 | unassigned |

## Domain interaction (`domain_adaptive_interaction`)

Available fits: 3/3. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:2:G5 | 1 | -0.180 | 0.469 | Methanomassiliicoccus (3) | hydrogen-dependent methyl reduction: 1 | 0 |
| 0:1:G0 | 7 | -0.173 | 0.380 | d__Eukaryota (unresolved genus); Unassigned; Unassigned; Unassigned; d__Eukaryota (unresolved genus); Unassigned; Unassigned | unassigned: 7 | 2 |
| 0:2:G6 | 1 | 0.142 | 0.341 | Methanomethylovorans (2) | unassigned: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G26 | 3 | -0.343 | 0.869 | Mesotoga (2); SC103 (3); f__WCHB1-02 (unresolved genus) (3) | unassigned: 3 | 0 |
| 0:1:G8 | 8 | -0.386 | 0.733 | Acidovorax; Dechloromonas; Thauera (3); Thermomonas (4); Zoogloea (3); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus) | unassigned: 8 | 0 |
| 0:0:G33 | 1 | 0.240 | 0.717 | Proteiniphilum (5) | proteolytic fermentation: 1 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAC | BAC::F00251 | Proteiniphilum (5) | 3 | 3 | 3 | 0 | proteolytic fermentation |
| BAC | BAC::F00299 | Syntrophomonas | 1 | 3 | 1 | 0 | syntrophic fatty-acid oxidation |
| BAC | BAC::F00263 | f__SAR324_clade(Marine_group_B) (unresolved genus) (3) | 3 | 3 | 0 | 3 | unassigned |
| BAC | BAC::F00115 | f__Bacteroidetes_vadinHA17 (unresolved genus) (12) | 3 | 3 | 0 | 3 | unassigned |
| BAC | BAC::F00329 | f__WCHB1-02 (unresolved genus) (3) | 3 | 3 | 0 | 3 | unassigned |
| ARC | ARC::F00021 | Methanomethylovorans (2) | 3 | 3 | 3 | 0 | unassigned |
| BAC | BAC::F00099 | Anaerovorax (3) | 3 | 3 | 2 | 1 | unassigned |
| BAC | BAC::F00244 | Pelotomaculum | 3 | 3 | 2 | 1 | unassigned |

## Interaction interpretation

| repeat | outer_fold | status | fitted_products | active_products |
| --- | --- | --- | --- | --- |
| 0 | 0 | ok | 4 | 0 |
| 0 | 1 | ok | 4 | 0 |
| 0 | 2 | ok | 4 | 0 |

No active product means no fitted interaction to interpret. Even an active centered-balance product would be a statistical interaction, not proof of syntrophy or direct metabolic exchange.
