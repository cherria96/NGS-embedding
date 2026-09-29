# Biogas quality: ASV biological interpretation

Groups are specific to a fitted training partition. The examples below are the three largest training-score standard deviations per model/domain, restricted to active groups. This describes fitted score variation, not independent or causal importance. The table may include more than one fold-specific version of a taxon group. See the complete member tables for all groups.

A positive coefficient contributes toward flag=1 through a within-domain log-composition contrast. It does not mean that increasing absolute abundance causes failure. Group IDs are local; identical IDs across fits do not identify the same group.

## ARC fusion (`fusion_arc_only`)

Available fits: 4/6. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:0:G0 | 1 | 0.104 | 0.304 | Methanosaeta | acetoclastic methanogenesis: 1 | 0 |
| 0:1:G7 | 1 | 0.083 | 0.221 | Methanospirillum | unassigned: 1 | 0 |
| 0:0:G4 | 1 | -0.095 | 0.210 | f__Bathyarchaeia (unresolved genus) | unassigned: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 1:0:G11 | 1 | 0.544 | 1.593 | f__009E01-B-SD-P15 (unresolved genus) | unassigned: 1 | 0 |
| 1:0:G14 | 1 | 0.516 | 1.553 | Fermentimonas | unassigned: 1 | 0 |
| 1:0:G30 | 1 | -0.396 | 1.243 | f__Spirochaetaceae (unresolved genus) | unassigned: 1 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAC | BAC::33abb774bb089c7a1c33cdc1341c8751 | Fermentimonas | 4 | 4 | 4 | 0 | unassigned |
| BAC | BAC::835807472970ae51e3ba46cfe94841e5 | f__Paludibacteraceae (unresolved genus) | 3 | 4 | 3 | 0 | unassigned |
| BAC | BAC::1b9bdb8182f6ee15da8d4ff1bdff03c8 | f__009E01-B-SD-P15 (unresolved genus) | 4 | 4 | 4 | 0 | unassigned |
| BAC | BAC::62730fd9676d050cb6a25c5f3c1d5bc2 | f__WS1 (unresolved genus) | 4 | 4 | 4 | 0 | unassigned |
| BAC | BAC::8ab62dfb84acaaf13c629a5d72dadcd8 | f__Spirochaetaceae (unresolved genus) | 4 | 4 | 0 | 4 | unassigned |
| BAC | BAC::47a8022ade8fe9825dd1ff3e66baa5f8 | f__Dysgonomonadaceae (unresolved genus) | 4 | 4 | 0 | 4 | unassigned |
| BAC | BAC::44dd28cba5f7f60f757152ea18910a34 | Microbacterium | 4 | 4 | 0 | 4 | unassigned |
| BAC | BAC::dc78bc8f02fbf156350d61d153238cb1 | f__uncultured (unresolved genus) | 2 | 4 | 0 | 2 | unassigned |

## BAC fusion (`fusion_bac_only`)

Available fits: 4/6. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G10 | 1 | 0.399 | 1.063 | Methanospirillum | unassigned: 1 | 0 |
| 1:0:G7 | 1 | 0.354 | 0.954 | Methanospirillum | unassigned: 1 | 0 |
| 1:0:G0 | 1 | 0.234 | 0.661 | Methanosaeta | acetoclastic methanogenesis: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G2 | 1 | 0.078 | 0.241 | f__SBR1031 (unresolved genus) | unassigned: 1 | 0 |
| 0:0:G11 | 1 | 0.063 | 0.180 | Fermentimonas | unassigned: 1 | 0 |
| 1:0:G1 | 1 | 0.059 | 0.172 | f__SBR1031 (unresolved genus) | unassigned: 1 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARC | ARC::87078040a976e7ceb82734c0713fd4e2 | Methanospirillum | 4 | 4 | 4 | 0 | unassigned |
| ARC | ARC::02bc9f1c8263ce565cfa8174b56af2de | Methanosaeta | 4 | 4 | 4 | 0 | acetoclastic methanogenesis |
| ARC | ARC::ff2ddcb0273425f900af1f8d1b4d90a8 | Methanoculleus | 2 | 4 | 0 | 2 | hydrogenotrophic methanogenesis |
| ARC | ARC::7fb850c5cd377da752f7b081daed5032 | f__Bathyarchaeia (unresolved genus) | 2 | 4 | 0 | 2 | unassigned |
| ARC | ARC::a2ff073fb23039e822fba8a653a34527 | Methanosaeta | 2 | 4 | 0 | 2 | acetoclastic methanogenesis |
| ARC | ARC::3b768ad6b26d688837b3511e5d2316d2 | Methanomassiliicoccus | 4 | 4 | 0 | 4 | hydrogen-dependent methyl reduction |
| ARC | ARC::7029dd7b1fe970d81fbff18315097ae2 | Methanolinea | 3 | 4 | 0 | 3 | hydrogenotrophic methanogenesis |
| ARC | ARC::327914bbcbfe425fb2d5caaf9d8688a1 | Methanospirillum | 4 | 4 | 3 | 1 | unassigned |

## Shared hybrid (`shared_hybrid`)

Available fits: 4/6. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:0:G5 | 1 | 0.154 | 0.396 | Candidatus_Methanofastidiosum | hydrogen-dependent methylated-thiol reduction: 1 | 0 |
| 1:0:G2 | 1 | -0.119 | 0.359 | Methanomassiliicoccus | hydrogen-dependent methyl reduction: 1 | 0 |
| 0:1:G3 | 1 | -0.153 | 0.349 | f__Bathyarchaeia (unresolved genus) | unassigned: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 1:0:G7 | 1 | 0.230 | 0.692 | Fermentimonas | unassigned: 1 | 0 |
| 1:2:G1 | 8 | -0.322 | 0.581 | Lautropia; f__uncultured (unresolved genus); Zoogloea; Acidovorax; f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus); Syntrophorhabdus | unassigned: 8 | 0 |
| 1:2:G3 | 4 | 0.286 | 0.580 | Candidatus_Cloacimonas; W5; f__W27 (unresolved genus); Candidatus_Cloacimonas | unassigned: 4 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARC | ARC::abb98aa85e9528ef54e3078ff40ac726 | Candidatus_Methanofastidiosum | 4 | 4 | 4 | 0 | hydrogen-dependent methylated-thiol reduction |
| ARC | ARC::7fb850c5cd377da752f7b081daed5032 | f__Bathyarchaeia (unresolved genus) | 2 | 4 | 0 | 2 | unassigned |
| BAC | BAC::835807472970ae51e3ba46cfe94841e5 | f__Paludibacteraceae (unresolved genus) | 3 | 4 | 3 | 0 | unassigned |
| BAC | BAC::c7f1a74be381c2080962c5af1b6cd08c | f__SBR1031 (unresolved genus) | 3 | 4 | 2 | 1 | unassigned |
| BAC | BAC::33abb774bb089c7a1c33cdc1341c8751 | Fermentimonas | 4 | 4 | 3 | 1 | unassigned |
| BAC | BAC::44dd28cba5f7f60f757152ea18910a34 | Microbacterium | 3 | 4 | 0 | 3 | unassigned |
| BAC | BAC::7f7370b40b1736512fd60f8724a88ab9 | Syntrophomonas | 2 | 4 | 0 | 2 | syntrophic fatty-acid oxidation |
| ARC | ARC::3b768ad6b26d688837b3511e5d2316d2 | Methanomassiliicoccus | 4 | 4 | 1 | 3 | hydrogen-dependent methyl reduction |

## Domain separate (`domain_adaptive_hybrid`)

Available fits: 4/6. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G5 | 1 | -0.232 | 0.527 | f__Bathyarchaeia (unresolved genus) | unassigned: 1 | 0 |
| 0:0:G0 | 1 | 0.167 | 0.488 | Methanosaeta | acetoclastic methanogenesis: 1 | 0 |
| 1:0:G2 | 1 | -0.119 | 0.359 | Methanomassiliicoccus | hydrogen-dependent methyl reduction: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 1:2:G13 | 1 | 0.331 | 0.938 | Fermentimonas | unassigned: 1 | 0 |
| 0:0:G8 | 1 | 0.313 | 0.889 | Fermentimonas | unassigned: 1 | 0 |
| 1:0:G7 | 1 | 0.230 | 0.692 | Fermentimonas | unassigned: 1 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAC | BAC::33abb774bb089c7a1c33cdc1341c8751 | Fermentimonas | 4 | 4 | 4 | 0 | unassigned |
| BAC | BAC::835807472970ae51e3ba46cfe94841e5 | f__Paludibacteraceae (unresolved genus) | 3 | 4 | 3 | 0 | unassigned |
| BAC | BAC::dc78bc8f02fbf156350d61d153238cb1 | f__uncultured (unresolved genus) | 2 | 4 | 0 | 2 | unassigned |
| ARC | ARC::7fb850c5cd377da752f7b081daed5032 | f__Bathyarchaeia (unresolved genus) | 3 | 4 | 0 | 3 | unassigned |
| BAC | BAC::1b9bdb8182f6ee15da8d4ff1bdff03c8 | f__009E01-B-SD-P15 (unresolved genus) | 4 | 4 | 3 | 1 | unassigned |
| BAC | BAC::da7902cef5ac5dc0ee628713d7357235 | f__W27 (unresolved genus) | 2 | 4 | 2 | 0 | unassigned |
| ARC | ARC::02bc9f1c8263ce565cfa8174b56af2de | Methanosaeta | 4 | 4 | 4 | 0 | acetoclastic methanogenesis |
| BAC | BAC::16ef1ff98dad2cf16777aaa2c0d33635 | f__Synergistaceae (unresolved genus) | 2 | 4 | 0 | 2 | unassigned |

## Domain interaction (`domain_adaptive_interaction`)

Available fits: 4/6. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G5 | 1 | -0.232 | 0.527 | f__Bathyarchaeia (unresolved genus) | unassigned: 1 | 0 |
| 0:0:G0 | 1 | 0.167 | 0.488 | Methanosaeta | acetoclastic methanogenesis: 1 | 0 |
| 1:0:G2 | 1 | -0.119 | 0.359 | Methanomassiliicoccus | hydrogen-dependent methyl reduction: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 1:2:G13 | 1 | 0.331 | 0.938 | Fermentimonas | unassigned: 1 | 0 |
| 0:0:G8 | 1 | 0.313 | 0.889 | Fermentimonas | unassigned: 1 | 0 |
| 1:0:G7 | 1 | 0.230 | 0.692 | Fermentimonas | unassigned: 1 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAC | BAC::33abb774bb089c7a1c33cdc1341c8751 | Fermentimonas | 4 | 4 | 4 | 0 | unassigned |
| BAC | BAC::835807472970ae51e3ba46cfe94841e5 | f__Paludibacteraceae (unresolved genus) | 3 | 4 | 3 | 0 | unassigned |
| BAC | BAC::dc78bc8f02fbf156350d61d153238cb1 | f__uncultured (unresolved genus) | 2 | 4 | 0 | 2 | unassigned |
| ARC | ARC::7fb850c5cd377da752f7b081daed5032 | f__Bathyarchaeia (unresolved genus) | 3 | 4 | 0 | 3 | unassigned |
| BAC | BAC::1b9bdb8182f6ee15da8d4ff1bdff03c8 | f__009E01-B-SD-P15 (unresolved genus) | 4 | 4 | 3 | 1 | unassigned |
| BAC | BAC::da7902cef5ac5dc0ee628713d7357235 | f__W27 (unresolved genus) | 2 | 4 | 2 | 0 | unassigned |
| ARC | ARC::02bc9f1c8263ce565cfa8174b56af2de | Methanosaeta | 4 | 4 | 4 | 0 | acetoclastic methanogenesis |
| BAC | BAC::16ef1ff98dad2cf16777aaa2c0d33635 | f__Synergistaceae (unresolved genus) | 2 | 4 | 0 | 2 | unassigned |

## Interaction interpretation

| repeat | outer_fold | status | fitted_products | active_products |
| --- | --- | --- | --- | --- |
| 0 | 0 | ok | 4 | 0 |
| 0 | 1 | ok | 4 | 0 |
| 0 | 2 | unavailable | 0 | 0 |
| 1 | 0 | ok | 4 | 0 |
| 1 | 1 | unavailable | 0 | 0 |
| 1 | 2 | ok | 4 | 0 |

No active product means no fitted interaction to interpret. Even an active centered-balance product would be a statistical interaction, not proof of syntrophy or direct metabolic exchange.
