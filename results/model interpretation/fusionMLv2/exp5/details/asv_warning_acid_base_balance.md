# Acid base balance: ASV biological interpretation

Groups are specific to a fitted training partition. The examples below are the three largest training-score standard deviations per model/domain, restricted to active groups. This describes fitted score variation, not independent or causal importance. The table may include more than one fold-specific version of a taxon group. See the complete member tables for all groups.

A positive coefficient contributes toward flag=1 through a within-domain log-composition contrast. It does not mean that increasing absolute abundance causes failure. Group IDs are local; identical IDs across fits do not identify the same group.

## ARC fusion (`fusion_arc_only`)

Available fits: 6/6. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 1:0:G3 | 1 | -0.156 | 0.519 | Methanosaeta | acetoclastic methanogenesis: 1 | 0 |
| 1:2:G2 | 6 | -0.195 | 0.494 | d__Eukaryota (unresolved genus); Unassigned; Unassigned; d__Eukaryota (unresolved genus); Unassigned; Unassigned | unassigned: 6 | 2 |
| 1:2:G8 | 1 | 0.137 | 0.467 | Methanolinea | hydrogenotrophic methanogenesis: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 1:0:G8 | 1 | 0.454 | 1.480 | Proteiniphilum | proteolytic fermentation: 1 | 0 |
| 1:0:G17 | 1 | -0.564 | 1.278 | f__Dysgonomonadaceae (unresolved genus) | unassigned: 1 | 0 |
| 1:0:G41 | 1 | -0.366 | 1.113 | W5 | unassigned: 1 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAC | BAC::14a311b4802d5606cbf72b2dc465ae56 | Proteiniphilum | 6 | 6 | 6 | 0 | proteolytic fermentation |
| BAC | BAC::47a8022ade8fe9825dd1ff3e66baa5f8 | f__Dysgonomonadaceae (unresolved genus) | 6 | 6 | 2 | 4 | unassigned |
| BAC | BAC::835807472970ae51e3ba46cfe94841e5 | f__Paludibacteraceae (unresolved genus) | 3 | 6 | 1 | 2 | unassigned |
| ARC | ARC::a2ff073fb23039e822fba8a653a34527 | Methanosaeta | 2 | 6 | 0 | 2 | acetoclastic methanogenesis |
| BAC | BAC::845e0b2e08620241ecbfc479bc644a98 | Thermovirga | 5 | 6 | 5 | 0 | amino-acid degradation |
| BAC | BAC::ab550775cc16043888330ee065fc30b9 | f__M2PB4-65_termite_group (unresolved genus) | 6 | 6 | 6 | 0 | unassigned |
| BAC | BAC::1b9bdb8182f6ee15da8d4ff1bdff03c8 | f__009E01-B-SD-P15 (unresolved genus) | 6 | 6 | 2 | 4 | unassigned |
| BAC | BAC::ba41674647f59350f02ba53a1d4d2eca | W5 | 6 | 6 | 1 | 5 | unassigned |

## BAC fusion (`fusion_bac_only`)

Available fits: 6/6. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 1:0:G8 | 1 | 0.247 | 0.970 | Methanoculleus | hydrogenotrophic methanogenesis: 1 | 0 |
| 1:0:G11 | 1 | 0.303 | 0.844 | Methanosaeta | acetoclastic methanogenesis: 1 | 0 |
| 0:0:G11 | 1 | 0.206 | 0.736 | Methanoculleus | hydrogenotrophic methanogenesis: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:0:G5 | 1 | 0.165 | 0.478 | Proteiniphilum | proteolytic fermentation: 1 | 0 |
| 0:0:G1 | 1 | -0.176 | 0.421 | f__Synergistaceae (unresolved genus) | unassigned: 1 | 0 |
| 0:0:G14 | 1 | -0.134 | 0.374 | f__uncultured (unresolved genus) | unassigned: 1 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARC | ARC::8995879a30350364c8d6147a828733ee | Methanoculleus | 4 | 6 | 4 | 0 | hydrogenotrophic methanogenesis |
| ARC | ARC::a875c453fc275b8f21ef13228c53d599 | Methanosaeta | 6 | 6 | 6 | 0 | acetoclastic methanogenesis |
| BAC | BAC::14a311b4802d5606cbf72b2dc465ae56 | Proteiniphilum | 6 | 6 | 6 | 0 | proteolytic fermentation |
| ARC | ARC::a2ff073fb23039e822fba8a653a34527 | Methanosaeta | 2 | 6 | 0 | 2 | acetoclastic methanogenesis |
| BAC | BAC::dc78bc8f02fbf156350d61d153238cb1 | f__uncultured (unresolved genus) | 1 | 6 | 0 | 1 | unassigned |
| ARC | ARC::12041ed58f0b14a5efb2687b72fc5883 | Methanospirillum | 3 | 6 | 0 | 3 | unassigned |
| ARC | ARC::5a72697a45f01e085759fa49e855d45e | Methanoculleus | 4 | 6 | 4 | 0 | hydrogenotrophic methanogenesis |
| ARC | ARC::7029dd7b1fe970d81fbff18315097ae2 | Methanolinea | 6 | 6 | 0 | 6 | hydrogenotrophic methanogenesis |

## Shared hybrid (`shared_hybrid`)

Available fits: 6/6. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:2:G2 | 1 | -0.108 | 0.289 | Methanomassiliicoccus | hydrogen-dependent methyl reduction: 1 | 0 |
| 0:2:G3 | 7 | 0.122 | 0.284 | d__Eukaryota (unresolved genus); Unassigned; Unassigned; Unassigned; d__Eukaryota (unresolved genus); Unassigned; Unassigned | unassigned: 7 | 2 |
| 0:2:G4 | 1 | -0.138 | 0.267 | f__Bathyarchaeia (unresolved genus) | unassigned: 1 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 1:0:G3 | 1 | 0.314 | 1.022 | Proteiniphilum | proteolytic fermentation: 1 | 0 |
| 1:0:G10 | 2 | 0.304 | 0.771 | Thermovirga; Thermovirga | amino-acid degradation: 2 | 0 |
| 0:2:G1 | 8 | -0.330 | 0.614 | Lautropia; f__uncultured (unresolved genus); f__Comamonadaceae (unresolved genus); SWB02; f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus); f__Comamonadaceae (unresolved genus); Syntrophorhabdus | unassigned: 8 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAC | BAC::14a311b4802d5606cbf72b2dc465ae56 | Proteiniphilum | 6 | 6 | 6 | 0 | proteolytic fermentation |
| BAC | BAC::ba41674647f59350f02ba53a1d4d2eca | W5 | 6 | 6 | 0 | 6 | unassigned |
| ARC | ARC::7fb850c5cd377da752f7b081daed5032 | f__Bathyarchaeia (unresolved genus) | 5 | 6 | 1 | 4 | unassigned |
| BAC | BAC::845e0b2e08620241ecbfc479bc644a98 | Thermovirga | 4 | 6 | 4 | 0 | amino-acid degradation |
| BAC | BAC::b1e26f0101b1c0178f1d56f169f75955 | Thermovirga | 4 | 6 | 4 | 0 | amino-acid degradation |
| BAC | BAC::fb839e4738eb146439c7ca93239968b2 | Candidatus_Cloacimonas | 6 | 6 | 0 | 6 | unassigned |
| BAC | BAC::b4bfed3fff749b89ea9d77f7870e0fa4 | Candidatus_Cloacimonas | 6 | 6 | 0 | 6 | unassigned |
| BAC | BAC::835807472970ae51e3ba46cfe94841e5 | f__Paludibacteraceae (unresolved genus) | 2 | 6 | 2 | 0 | unassigned |

## Domain separate (`domain_adaptive_hybrid`)

Available fits: 6/6. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G6 | 1 | -0.166 | 0.541 | Methanosaeta | acetoclastic methanogenesis: 1 | 0 |
| 0:2:G4 | 3 | 0.360 | 0.469 | Methanosaeta; Methanosaeta; Methanosaeta | acetoclastic methanogenesis: 3 | 0 |
| 0:2:G1 | 6 | -0.329 | 0.446 | Methanospirillum; Methanospirillum; Methanolinea; Methanospirillum; Methanospirillum; Methanoculleus | hydrogenotrophic methanogenesis: 2; unassigned: 4 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 1:1:G15 | 2 | 0.415 | 1.047 | Thermovirga; Thermovirga | amino-acid degradation: 2 | 0 |
| 0:1:G10 | 1 | 0.277 | 0.825 | Proteiniphilum | proteolytic fermentation: 1 | 0 |
| 1:0:G4 | 1 | 0.246 | 0.801 | Proteiniphilum | proteolytic fermentation: 1 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAC | BAC::14a311b4802d5606cbf72b2dc465ae56 | Proteiniphilum | 6 | 6 | 6 | 0 | proteolytic fermentation |
| BAC | BAC::845e0b2e08620241ecbfc479bc644a98 | Thermovirga | 5 | 6 | 5 | 0 | amino-acid degradation |
| BAC | BAC::b1e26f0101b1c0178f1d56f169f75955 | Thermovirga | 5 | 6 | 5 | 0 | amino-acid degradation |
| BAC | BAC::ab550775cc16043888330ee065fc30b9 | f__M2PB4-65_termite_group (unresolved genus) | 6 | 6 | 6 | 0 | unassigned |
| BAC | BAC::c82d6f1cbea3d3c882367938f9654c0b | f__WCHB1-02 (unresolved genus) | 6 | 6 | 0 | 6 | unassigned |
| BAC | BAC::8a952a2427fd133802ed3054049ab72b | f__Bacteroidales_UCG-001 (unresolved genus) | 6 | 6 | 1 | 5 | unassigned |
| BAC | BAC::d15dcc7926d096955df451c670464a35 | Rikenellaceae_RC9_gut_group | 2 | 6 | 0 | 2 | unassigned |
| ARC | ARC::a2ff073fb23039e822fba8a653a34527 | Methanosaeta | 2 | 6 | 0 | 2 | acetoclastic methanogenesis |

## Domain interaction (`domain_adaptive_interaction`)

Available fits: 6/6. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors.

### ARC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 0:1:G6 | 1 | -0.166 | 0.541 | Methanosaeta | acetoclastic methanogenesis: 1 | 0 |
| 0:2:G4 | 3 | 0.360 | 0.469 | Methanosaeta; Methanosaeta; Methanosaeta | acetoclastic methanogenesis: 3 | 0 |
| 0:2:G1 | 6 | -0.329 | 0.446 | Methanospirillum; Methanospirillum; Methanolinea; Methanospirillum; Methanospirillum; Methanoculleus | hydrogenotrophic methanogenesis: 2; unassigned: 4 | 0 |

### BAC

| fit | n_features | theta | train_score_sd | taxa | roles | domain_mismatch_asvs |
| --- | --- | --- | --- | --- | --- | --- |
| 1:1:G15 | 2 | 0.415 | 1.047 | Thermovirga; Thermovirga | amino-acid degradation: 2 | 0 |
| 0:1:G10 | 1 | 0.277 | 0.825 | Proteiniphilum | proteolytic fermentation: 1 | 0 |
| 1:0:G4 | 1 | 0.246 | 0.801 | Proteiniphilum | proteolytic fermentation: 1 | 0 |

Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):

| domain | feature_id | taxon_label | active_fits | successful_fits | positive_fits | negative_fits | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAC | BAC::14a311b4802d5606cbf72b2dc465ae56 | Proteiniphilum | 6 | 6 | 6 | 0 | proteolytic fermentation |
| BAC | BAC::845e0b2e08620241ecbfc479bc644a98 | Thermovirga | 5 | 6 | 5 | 0 | amino-acid degradation |
| BAC | BAC::b1e26f0101b1c0178f1d56f169f75955 | Thermovirga | 5 | 6 | 5 | 0 | amino-acid degradation |
| BAC | BAC::ab550775cc16043888330ee065fc30b9 | f__M2PB4-65_termite_group (unresolved genus) | 6 | 6 | 6 | 0 | unassigned |
| BAC | BAC::c82d6f1cbea3d3c882367938f9654c0b | f__WCHB1-02 (unresolved genus) | 6 | 6 | 0 | 6 | unassigned |
| BAC | BAC::8a952a2427fd133802ed3054049ab72b | f__Bacteroidales_UCG-001 (unresolved genus) | 6 | 6 | 1 | 5 | unassigned |
| BAC | BAC::d15dcc7926d096955df451c670464a35 | Rikenellaceae_RC9_gut_group | 2 | 6 | 0 | 2 | unassigned |
| ARC | ARC::a2ff073fb23039e822fba8a653a34527 | Methanosaeta | 2 | 6 | 0 | 2 | acetoclastic methanogenesis |

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
