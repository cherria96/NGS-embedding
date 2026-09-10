# Genomic-only taxa importance

CV-selected model: **RandomForest**. All 290 training-union features tested with 30 repetitions on the existing training folds. Test predictions are not used.

Mean pooled Micro-AUPR decrease determines rank. The complete CSV also reports pooled MCC decrease and permutation SD.

| Rank | Taxon feature | Micro-AUPR drop | MCC drop |
|---:|---|---:|---:|
| 1 | BAC_O::Anaerolineales | 0.00717 | 0.00948 |
| 2 | BAC_P::Planctomycetota | 0.00711 | 0.00879 |
| 3 | BAC_O::Clostridia | 0.00711 | 0.01561 |
| 4 | BAC_P::Chloroflexi | 0.00705 | 0.00632 |
| 5 | BAC_O::Oscillospirales | 0.00678 | -0.00109 |
| 6 | ARC_G::Candidatus_Methanofastidiosum | 0.00626 | 0.00185 |
| 7 | BAC_O::MBA03 | 0.00609 | 0.00747 |
| 8 | BAC_O::Rhizobiales | 0.00494 | 0.01338 |
| 9 | BAC_O::Caldicoprobacterales | 0.00473 | 0.00355 |
| 10 | BAC_P::Firmicutes | 0.00398 | 0.01427 |
| 11 | BAC_O::Spirochaetales | 0.00396 | 0.03538 |
| 12 | ARC_O::Methanofastidiosales | 0.00367 | 0.00586 |
| 13 | BAC_P::Spirochaetota | 0.00352 | 0.03652 |
| 14 | BAC_P::Synergistota | 0.00343 | 0.01079 |
| 15 | BAC_O::Synergistales | 0.00322 | 0.00101 |
| 16 | BAC_O::Syntrophorhabdales | 0.00301 | 0.00000 |
| 17 | BAC_O::unidentified | 0.00301 | 0.02593 |
| 18 | BAC_O::Eubacteriales | 0.00245 | 0.00235 |
| 19 | BAC_O::Xanthomonadales | 0.00228 | 0.01180 |
| 20 | BAC_P::Verrucomicrobiota | 0.00225 | -0.00036 |
| 21 | BAC_P::Desulfobacterota | 0.00210 | 0.00666 |
| 22 | BAC_O::Micrococcales | 0.00204 | 0.00466 |
| 23 | BAC_O::Thermacetogeniales | 0.00190 | 0.00067 |
| 24 | BAC_O::Syntrophales | 0.00160 | 0.00000 |
| 25 | BAC_O::Lactobacillales | 0.00159 | -0.00510 |

Largest Micro-AUPR contributions: BAC_O::Anaerolineales, BAC_P::Planctomycetota, BAC_O::Clostridia, BAC_P::Chloroflexi, BAC_O::Oscillospirales.
Largest MCC contributions: BAC_P::Spirochaetota, BAC_O::Spirochaetales, BAC_O::unidentified, BAC_P::Actinobacteriota, BAC_O::Caldisericales.
Negative importance means permutation improved the score, not that a taxon protects against warnings. Small drops indicate little detected individual reliance, not biological irrelevance.
Importance is predictive association in this model, not causation or direction of effect. Related ranks and correlated/compositional taxa can share information. Shuffling does not preserve site trajectories or compositional/hierarchical consistency. Error bars are permutation SD, not confidence intervals across sites.
Settings and thresholds were selected using the same training CV previously, so this is post-selection interpretation. The original splits and final models are unchanged. No taxa are removed. Displaying the top 25 is only a visualization choice; all taxa are in ranked_taxa.csv.
Reproduce: `python ML/importance_genomic.py --seed 20260908 --repeats 30`.
