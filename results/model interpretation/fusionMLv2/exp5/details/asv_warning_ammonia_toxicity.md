# Ammonia toxicity: ASV biological interpretation

Groups are specific to a fitted training partition. The examples below are the three largest training-score standard deviations per model/domain, restricted to active groups. This describes fitted score variation, not independent or causal importance. The table may include more than one fold-specific version of a taxon group. See the complete member tables for all groups.

A positive coefficient contributes toward flag=1 through a within-domain log-composition contrast. It does not mean that increasing absolute abundance causes failure. Group IDs are local; identical IDs across fits do not identify the same group.

## ARC fusion (`fusion_arc_only`)

Available fits: 6/6. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G9 | 1 | 0.315 | 1.075 | Methanosaeta | acetoclastic methanogenesis: 1 | 0 |
| 1:2:G4 | 2 | 0.289 | 0.844 | Methanoculleus; Methanosaeta | acetoclastic methanogenesis: 1; hydrogenotrophic methanogenesis: 1 | 0 |
| 0:1:G0 | 1 | -0.204 | 0.580 | Methanosaeta | acetoclastic methanogenesis: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G9 | 1 | 0.323 | 1.088 | Fastidiosipila | unassigned: 1 | 0 |
| 0:1:G24 | 1 | 0.351 | 0.822 | f__Hungateiclostridiaceae (unresolved genus) | unassigned: 1 | 0 |
| 0:1:G35 | 1 | -0.276 | 0.792 | f__Bacteroidetes_vadinHA17 (unresolved genus) | unassigned: 1 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAC | BAC::4b90c775ff1aa9cc71122804f9d75158 | f__Hungateiclostridiaceae (unresolved genus) | 2 | 6 | 2 | 0 | unassigned |
| BAC | BAC::98ac7d13ebdedd798069b2e7f8f1588d | Acholeplasma | 3 | 6 | 3 | 0 | unassigned |
| BAC | BAC::1374aaab66407b2a57923b2edb9c43a1 | Fastidiosipila | 6 | 6 | 6 | 0 | unassigned |
| BAC | BAC::2e6f7e61545c37522edb1a3a46092f9d | f__Prolixibacteraceae (unresolved genus) | 6 | 6 | 0 | 6 | unassigned |
| BAC | BAC::7f7370b40b1736512fd60f8724a88ab9 | Syntrophomonas | 3 | 6 | 3 | 0 | syntrophic fatty-acid oxidation |
| BAC | BAC::82ab92ff25107467c6fdfac5af36ee1a | f__Bacteroidetes_vadinHA17 (unresolved genus) | 6 | 6 | 0 | 6 | unassigned |
| BAC | BAC::96affa23ea568d2373afae040e5aa9cc | Acidovorax | 5 | 6 | 0 | 5 | unassigned |
| BAC | BAC::62730fd9676d050cb6a25c5f3c1d5bc2 | f__WS1 (unresolved genus) | 4 | 6 | 2 | 2 | unassigned |

## BAC fusion (`fusion_bac_only`)

Available fits: 6/6. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:0:G5 | 1 | 0.424 | 1.214 | Methanoculleus | hydrogenotrophic methanogenesis: 1 | 0 |
| 1:0:G8 | 1 | 0.266 | 1.045 | Methanoculleus | hydrogenotrophic methanogenesis: 1 | 0 |
| 1:1:G11 | 1 | 0.243 | 0.879 | Methanoculleus | hydrogenotrophic methanogenesis: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G6 | 1 | 0.143 | 0.480 | Fastidiosipila | unassigned: 1 | 0 |
| 0:1:G16 | 1 | 0.166 | 0.388 | f__Hungateiclostridiaceae (unresolved genus) | unassigned: 1 | 0 |
| 0:2:G2 | 4 | 0.148 | 0.364 | Fastidiosipila; Fermentimonas; f__Dysgonomonadaceae (unresolved genus); f__Paludibacteraceae (unresolved genus) | unassigned: 4 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARC | ARC::5a72697a45f01e085759fa49e855d45e | Methanoculleus | 4 | 6 | 4 | 0 | hydrogenotrophic methanogenesis |
| ARC | ARC::8995879a30350364c8d6147a828733ee | Methanoculleus | 4 | 6 | 4 | 0 | hydrogenotrophic methanogenesis |
| ARC | ARC::a6cdde7256831dcfab80f676aa3f4af1 | Methanosaeta | 6 | 6 | 6 | 0 | acetoclastic methanogenesis |
| ARC | ARC::02bc9f1c8263ce565cfa8174b56af2de | Methanosaeta | 6 | 6 | 0 | 6 | acetoclastic methanogenesis |
| ARC | ARC::a2ff073fb23039e822fba8a653a34527 | Methanosaeta | 2 | 6 | 0 | 2 | acetoclastic methanogenesis |
| ARC | ARC::ff2ddcb0273425f900af1f8d1b4d90a8 | Methanoculleus | 4 | 6 | 4 | 0 | hydrogenotrophic methanogenesis |
| ARC | ARC::7fb850c5cd377da752f7b081daed5032 | f__Bathyarchaeia (unresolved genus) | 5 | 6 | 0 | 5 | unassigned |
| BAC | BAC::4b90c775ff1aa9cc71122804f9d75158 | f__Hungateiclostridiaceae (unresolved genus) | 1 | 6 | 1 | 0 | unassigned |

## Shared hybrid (`shared_hybrid`)

Available fits: 6/6. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 1:0:G1 | 7 | 0.890 | 0.913 | Methanospirillum; Methanolinea; Methanospirillum; Methanoculleus; Methanolinea; Methanospirillum; Methanoculleus | hydrogenotrophic methanogenesis: 4; unassigned: 3 | 0 |
| 1:0:G0 | 5 | -0.650 | 0.861 | Methanosaeta; Methanosaeta; Methanosaeta; Methanosaeta; Methanosaeta | acetoclastic methanogenesis: 5 | 0 |
| 1:0:G2 | 1 | -0.247 | 0.746 | Methanomassiliicoccus | hydrogen-dependent methyl reduction: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 1:0:G1 | 31 | -0.982 | 0.996 | f__Synergistaceae (unresolved genus); Syntrophomonas; f__WCHB1-41 (unresolved genus); Fastidiosipila; Proteiniphilum; Christensenellaceae_R-7_group; Candidatus_Caldatribacterium; Microbacterium; ADurb.Bin063-1; Christensenellaceae_R-7_group; Christensenella… | amino-acid degradation: 2; proteolytic fermentation: 1; syntrophic fatty-acid oxidation: 2; unassigned: 26 | 0 |
| 1:0:G0 | 15 | 0.970 | 0.966 | Paludibacter; f__Marinilabiliaceae (unresolved genus); f__ST-12K33 (unresolved genus); f__Bacteroidetes_vadinHA17 (unresolved genus); f__009E01-B-SD-P15 (unresolved genus); Cloacibacterium; f__Prolixibacteraceae (unresolved genus); Fermentimonas; f__Dysgono… | unassigned: 15 | 0 |
| 0:2:G1 | 8 | -0.517 | 0.962 | Lautropia; f__uncultured (unresolved genus); f__Comamonadaceae (unresolved genus); SWB02; f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus); Syntrophorhabdus | unassigned: 8 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARC | ARC::7fb850c5cd377da752f7b081daed5032 | f__Bathyarchaeia (unresolved genus) | 4 | 6 | 0 | 4 | unassigned |
| ARC | ARC::3b768ad6b26d688837b3511e5d2316d2 | Methanomassiliicoccus | 6 | 6 | 1 | 5 | hydrogen-dependent methyl reduction |
| BAC | BAC::4b90c775ff1aa9cc71122804f9d75158 | f__Hungateiclostridiaceae (unresolved genus) | 2 | 6 | 2 | 0 | unassigned |
| BAC | BAC::8a952a2427fd133802ed3054049ab72b | f__Bacteroidales_UCG-001 (unresolved genus) | 6 | 6 | 6 | 0 | unassigned |
| ARC | ARC::a2ff073fb23039e822fba8a653a34527 | Methanosaeta | 2 | 6 | 1 | 1 | acetoclastic methanogenesis |
| BAC | BAC::b46f07b6978dbb1a7183270f72a12f52 | SWB02 | 1 | 6 | 0 | 1 | unassigned |
| ARC | ARC::02bc9f1c8263ce565cfa8174b56af2de | Methanosaeta | 6 | 6 | 4 | 2 | acetoclastic methanogenesis |
| BAC | BAC::fb839e4738eb146439c7ca93239968b2 | Candidatus_Cloacimonas | 5 | 6 | 0 | 5 | unassigned |

## Domain separate (`domain_adaptive_hybrid`)

Available fits: 6/6. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 1:1:G5 | 1 | 0.168 | 0.553 | Methanosaeta | acetoclastic methanogenesis: 1 | 0 |
| 1:2:G4 | 1 | -0.246 | 0.541 | f__Bathyarchaeia (unresolved genus) | unassigned: 1 | 0 |
| 1:0:G0 | 5 | -0.396 | 0.525 | Methanosaeta; Methanosaeta; Methanosaeta; Methanosaeta; Methanosaeta | acetoclastic methanogenesis: 5 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 1:0:G2 | 2 | -0.267 | 0.749 | Blvii28_wastewater-sludge_group; f__Prolixibacteraceae (unresolved genus) | unassigned: 2 | 0 |
| 0:2:G1 | 7 | -0.385 | 0.743 | Lautropia; f__uncultured (unresolved genus); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus); Syntrophorhabdus | unassigned: 7 | 0 |
| 1:2:G4 | 2 | 0.286 | 0.728 | Fastidiosipila; Acholeplasma | unassigned: 2 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARC | ARC::7fb850c5cd377da752f7b081daed5032 | f__Bathyarchaeia (unresolved genus) | 5 | 6 | 0 | 5 | unassigned |
| BAC | BAC::7f7370b40b1736512fd60f8724a88ab9 | Syntrophomonas | 3 | 6 | 3 | 0 | syntrophic fatty-acid oxidation |
| BAC | BAC::98ac7d13ebdedd798069b2e7f8f1588d | Acholeplasma | 3 | 6 | 3 | 0 | unassigned |
| BAC | BAC::1374aaab66407b2a57923b2edb9c43a1 | Fastidiosipila | 6 | 6 | 6 | 0 | unassigned |
| BAC | BAC::4b90c775ff1aa9cc71122804f9d75158 | f__Hungateiclostridiaceae (unresolved genus) | 2 | 6 | 2 | 0 | unassigned |
| ARC | ARC::abb98aa85e9528ef54e3078ff40ac726 | Candidatus_Methanofastidiosum | 6 | 6 | 5 | 1 | hydrogen-dependent methylated-thiol reduction |
| ARC | ARC::a6cdde7256831dcfab80f676aa3f4af1 | Methanosaeta | 6 | 6 | 5 | 1 | acetoclastic methanogenesis |
| ARC | ARC::02bc9f1c8263ce565cfa8174b56af2de | Methanosaeta | 6 | 6 | 0 | 6 | acetoclastic methanogenesis |

## Domain interaction (`domain_adaptive_interaction`)

Available fits: 6/6. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 1:1:G5 | 1 | 0.168 | 0.553 | Methanosaeta | acetoclastic methanogenesis: 1 | 0 |
| 1:2:G4 | 1 | -0.246 | 0.541 | f__Bathyarchaeia (unresolved genus) | unassigned: 1 | 0 |
| 1:0:G0 | 5 | -0.396 | 0.525 | Methanosaeta; Methanosaeta; Methanosaeta; Methanosaeta; Methanosaeta | acetoclastic methanogenesis: 5 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 1:0:G2 | 2 | -0.267 | 0.749 | Blvii28_wastewater-sludge_group; f__Prolixibacteraceae (unresolved genus) | unassigned: 2 | 0 |
| 0:2:G1 | 7 | -0.385 | 0.743 | Lautropia; f__uncultured (unresolved genus); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus); Syntrophorhabdus | unassigned: 7 | 0 |
| 1:2:G4 | 2 | 0.286 | 0.728 | Fastidiosipila; Acholeplasma | unassigned: 2 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARC | ARC::7fb850c5cd377da752f7b081daed5032 | f__Bathyarchaeia (unresolved genus) | 5 | 6 | 0 | 5 | unassigned |
| BAC | BAC::7f7370b40b1736512fd60f8724a88ab9 | Syntrophomonas | 3 | 6 | 3 | 0 | syntrophic fatty-acid oxidation |
| BAC | BAC::98ac7d13ebdedd798069b2e7f8f1588d | Acholeplasma | 3 | 6 | 3 | 0 | unassigned |
| BAC | BAC::1374aaab66407b2a57923b2edb9c43a1 | Fastidiosipila | 6 | 6 | 6 | 0 | unassigned |
| BAC | BAC::4b90c775ff1aa9cc71122804f9d75158 | f__Hungateiclostridiaceae (unresolved genus) | 2 | 6 | 2 | 0 | unassigned |
| ARC | ARC::abb98aa85e9528ef54e3078ff40ac726 | Candidatus_Methanofastidiosum | 6 | 6 | 5 | 1 | hydrogen-dependent methylated-thiol reduction |
| ARC | ARC::a6cdde7256831dcfab80f676aa3f4af1 | Methanosaeta | 6 | 6 | 5 | 1 | acetoclastic methanogenesis |
| ARC | ARC::02bc9f1c8263ce565cfa8174b56af2de | Methanosaeta | 6 | 6 | 0 | 6 | acetoclastic methanogenesis |

## Interaction interpretation

| repeat | outer_fold | status | fitted_products | active_products |
| --- | --- | --- | --- | --- |
| 0 | 0 | ok | 4 | 0 |
| 0 | 1 | ok | 4 | 0 |
| 0 | 2 | ok | 4 | 0 |
| 1 | 0 | ok | 4 | 0 |
| 1 | 1 | ok | 4 | 0 |
| 1 | 2 | ok | 4 | 0 |

No active product means no fitted interaction to interpret. Even an active centered-balance product would be a statistical interaction, not proof of syntrophy or direct metabolic exchange.
