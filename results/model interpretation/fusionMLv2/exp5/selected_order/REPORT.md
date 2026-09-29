# Experiment 5 (selected order-level models) — biological interpretation

Two configurations, chosen in Experiment 1 as the highest-MCC member of their
fusion arm, are read as biology here:

| Flag | Label definition | Arm | Model | Resolution | MCC | F2 | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|
| Ammonia toxicity | effluent NH4-N (TAN) > 2.5 | Both domains fused | `shared_hybrid` | order | 0.659 | 0.852 | 0.895 | 0.488 |
| Acid high | effluent pH > 8.0 | ARC-only fusion | `fusion_arc_only` | order | 0.363 | 0.595 | 0.808 | 0.273 |

No model is fitted here. Every input is a hash-verified snapshot of
Experiment 1's domain-restricted fusion runs, and every number below is
recomputed from the saved coefficients and out-of-fold probabilities.

**The acid-high flag is `effluent pH > 8.0`, not 8.5.** That is the threshold
in `fusionML/chemistry_targets.py`, the definition the selected model was
trained on, and no sample in this cohort reaches 8.5 at all — the maximum
effluent pH is 8.32. A pH > 8.5 flag would have
zero positives and could not be fitted. Everything below therefore describes
the > 8.0 model.

## The selection premise checks out

Both configurations are rank 1 of 20 candidates
(4 resolutions x 5 models) for their flag, by **both** MCC and F2, under the
minimum-inner-loss rule Experiment 1 reports:

| flag | resolution | model | mcc | f2 | auc | ap | rank_by_mcc | rank_by_f2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| warning_acid_high | order | fusion_arc_only | 0.363 | 0.595 | 0.808 | 0.273 | 1 | 1 |
| warning_acid_high | asv | fusion_arc_only | 0.352 | 0.583 | 0.756 | 0.237 | 2 | 2 |
| warning_acid_high | asv | domain_adaptive_hybrid | 0.343 | 0.576 | 0.817 | 0.383 | 3 | 4 |
| warning_acid_high | genus | fusion_bac_only | 0.341 | 0.578 | 0.785 | 0.263 | 4 | 3 |
| warning_ammonia_toxicity | order | shared_hybrid | 0.659 | 0.852 | 0.895 | 0.488 | 1 | 1 |
| warning_ammonia_toxicity | order | fusion_arc_only | 0.649 | 0.839 | 0.899 | 0.503 | 2 | 3 |
| warning_ammonia_toxicity | genus | shared_hybrid | 0.644 | 0.822 | 0.901 | 0.502 | 3 | 7 |
| warning_ammonia_toxicity | order | fusion_bac_only | 0.638 | 0.840 | 0.897 | 0.496 | 4 | 2 |

That is a best-of-20 choice read off held-out scores with no multiplicity
correction. Treat the *margin* over the runner-up as noise: the top four
ammonia configurations sit within
0.022
MCC of each other, the top four acid-high configurations within
0.022.

**The two selections are not equally robust to the selection rule.** Under the
one-standard-error rule the acid-high winner holds
(MCC 0.349, rank 1 of
20), but order-level `shared_hybrid` for
ammonia falls to MCC 0.414, rank
20 of 20 —
last. The one-SE rule takes the sparsest candidate within one standard error of
the inner-loss minimum, and for this model that is a far heavier fusion penalty:
5-10
active groups per fit against
18-74 under
minimum inner loss. The ammonia reading below therefore belongs to the
minimum-inner-loss fit specifically, not to `shared_hybrid` at order level in
general, and the drop is itself evidence that this flag's signal does not
survive aggressive fusion. Both rules are exported in `selection_ranking.csv`.

`fusion_arc_only` does **not** mean an archaea-only model. Both domains are
predictors in every arm; only the named domain receives fusion edges. The
restriction is verified per fit — the acid-high fits carry
144 ARC edges and 0 BAC
edges in total, while the ammonia fits carry 134 and
606.

## Main findings

1. **The ammonia model reproduces the textbook methanogen split, and it does so
   by refusing to fuse.** `Methanosarciniales__b5` (5 *Methanosarcina* + 3
   *Methanomethylovorans* ASVs) is the largest archaeal coefficient and is
   positive in 6/6
   fits (+0.032); the single-ASV
   *Methanosaeta* block `Methanosarciniales__b4` is negative in
   5/6
   fits (-0.027). In
   5 of 6 fits the fusion penalty collapses most of the
   archaeal domain into a near-zero background group, but it repeatedly keeps
   these two blocks out of it.

2. **The bacterial numerator is a proteolytic / amino-acid-fermenting guild plus
   one syntroph.** The blocks that rise with the flag are *Proteiniphilum*,
   Dysgonomonadaceae, *Fastidiosipila*, *Caldicoprobacter*, *Acholeplasma*,
   Christensenellaceae R-7, the Anaerovorax/Tissierella/Tepidimicrobium block,
   and *Syntrophomonas*. The denominator is led by the propionate and aromatic
   syntrophs (*Smithella*/*Syntrophus*, *Syntrophorhabdus*) and by aerobic,
   feed-associated Proteobacteria and Actinobacteria.

3. **The "ARC-only fusion" winner for acid high wins partly by deleting the
   archaea.** In 2 of 6 outer fits (r0f0, r1f1) the entire ARC
   domain collapses into 1 or 2 group(s)
   whose coefficients are numerically zero, so archaea contribute nothing to
   those predictions. Even in the fits where they do contribute, the largest
   archaeal coefficient is
   0.046 against
   0.074 on the bacterial side. This arm is
   better described as *compress the archaea, leave the bacteria free* than as
   an archaeal model.

4. **The two flags are largely the same event, and the two models find largely
   the same contrast.** 9 of 15
   acid-high positives are also ammonia positives, effluent pH and TAN correlate
   at Spearman 0.74, and the mean coefficients of the
   two models correlate at Pearson r = 0.56 over 85 shared blocks
   (65% share sign). The acid-high model should not be
   read as an independent line of evidence about alkaline pH.

5. **Order-level blocks, not order names, are the interpretable unit.**
   7 orders (ammonia) and
   6 (acid high) are split by the monophyly-
   consistent rank tree into blocks that end up on *opposite* sides of the
   contrast. *Bacteroidales* is the clearest: the *Proteiniphilum* block
   `Bacteroidales__b9` and the Dysgonomonadaceae block `Bacteroidales__b1` are
   numerator, the two *Bacteroidetes vadinHA17* blocks `__b7`/`__b8` are
   denominator. A naive taxonomy-label collapse to the order name would cancel
   these against each other.

6. **Two fused groups recur across folds and both are interpretable.** The
   largest is `Acholeplasmatales__b1` + `Caldicoprobacterales__b1` + `Clostridia__b2` + `Clostridia__b7` + `MBA03`
   (3 of 6 ammonia fits, always positive, mean theta
   +0.197) — five Firmicutes blocks (*Acholeplasma*,
   *Caldicoprobacter*, *Ruminiclostridium*, *Fastidiosipila* and the uncultured
   MBA03) carrying one shared coefficient. Against it,
   `Anaerolineales` + `Syntrophales__b1` + `Syntrophales__b2` + `Syntrophorhabdales`
   recurs in 2 fits, always negative (mean theta
   -0.096): the propionate/aromatic syntrophs and
   Anaerolineaceae fused into a single denominator term. A shared coefficient is
   an equality constraint the penalty imposed, not evidence that these taxa do
   the same job — but these two groups are at least coherent guilds.

7. **A fifth to two fifths of the leading signal has no literature role at all.**
   Of the 20 largest-|beta| blocks per model, 80%
   (ammonia) and 60% (acid high) match a curated role
   hypothesis; the remainder are uncultured lineage labels (MBA03, D8A_2,
   MVP_15, Bacteroidetes vadinHA17, SAR324, JS1) that are deliberately left
   unassigned. Coefficient size and marginal association are also not the same
   thing: for ammonia every one of
   the top 20 blocks also separates sites on its own (site-level AUROC at least
   0.1 away from 0.5), but for acid high
   3 of them do not
   (`Synergistales__b1`, `Bacteroidales__b3`, `Sphingobacteriales__b4`).
   Those blocks act only in combination with the rest of the contrast, which is
   what a log-ratio model is for and what makes single-taxon biomarker readings
   of one unsafe.

## The cohort, and what the two labels actually overlap

134 samples from 35 QC-adjusted site groups,
3 outer folds x 2 repeats, sites never split across a fold. Positives:
24 for ammonia over 10 sites,
15 for acid high over 9 sites,
of which 7 sites carry both.

| acid_high \\ ammonia | 0 | 1 |
| --- | --- | --- |
| 0 | 104 | 15 |
| 1 | 6 | 9 |

Sample-level phi between the two labels is 0.39; effluent pH and
TAN correlate at Pearson 0.60 / Spearman
0.74 over 130 samples
with both measurements. This is chemistry, not coincidence: protein and urea
hydrolysis releases ammonia, ammonium/bicarbonate buffering raises alkalinity,
and alkaline pH in turn shifts the NH4+/NH3 equilibrium toward free ammonia.

The practical consequence for this report: **a taxon that appears on the same
side of both contrasts has been found once, not twice.** The independent part
of the acid-high signal is the 6 samples that are
alkaline without being ammonia-positive, which is too few to interpret
separately.

## How to read one of these coefficients

Each model is a log-contrast logistic fit with a **separate zero-sum constraint
per domain**:

    logit(p) = b0 + sum_ARC beta_a log q_a + sum_BAC beta_b log q_b,
    sum_ARC beta_a = 0,  sum_BAC beta_b = 0

so within each domain the score is a weighted log-ratio: blocks with positive
beta form the numerator, blocks with negative beta the denominator, and only
the *ratio* is identified. A positive coefficient therefore means "higher
relative to the denominator blocks in the same domain", never "higher in
absolute abundance" and never "more active". Because ARC and BAC were amplified
with different primer pairs and normalised separately, the two contrasts are
independent of each other and read counts are not a biomass ratio between
domains.

Fusion sets coefficients exactly equal within a group. A group G with common
coefficient b contributes `|G| * b * log GM(q_G)` — a geometric mean, not a
summed abundance — and `theta = |G| * b` is what `group_catalog.csv` exports.
Equality is an estimated constraint; it is not evidence of shared function, and
taxonomic tidiness inside a group is partly imposed by the phylogenetic graph
that generated the candidate edges.

Coefficient magnitudes are comparable *within* a domain and a fit. They are not
comparable across domains without care: ARC carries
13 blocks and BAC 64 blocks after prevalence filtering, so the same
total contrast mass is divided differently.

## Ammonia toxicity — both domains fused, order level

Six outer fits, all available. Per fit the model keeps
13.0 ARC and 64.2
BAC blocks after training-partition prevalence filtering and reduces them to
5.5 + 38.2 =
43.7 active coefficient groups.

### Archaea: two blocks are pulled out of the background

| block | n_asv | genus_composition | mean_beta | min_beta | max_beta | n_positive_fits | n_negative_fits | mean_group_size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Methanosarciniales__b5 | 9 | Methanosarcina (5); Methanomethylovorans (3); f__Methanosarcinaceae | 0.032 | 0.002 | 0.076 | 6 | 0 | 1.667 |
| c17275cccc27dc9a4ab351a310c063c8 | 1 | unclassified | 0.003 | -0.002 | 0.021 | 3 | 3 | 5.167 |
| Methanosarciniales__b2 | 13 | Methanosaeta (13) | 0.002 | -0.002 | 0.009 | 3 | 3 | 4.167 |
| Methanomassiliicoccales | 8 | Methanomassiliicoccus (3); Candidatus_Methanoplasma (2); uncultured (2); RumEn_M2 | 0.001 | -0.022 | 0.021 | 4 | 2 | 2.333 |
| 4194fdb699f4d5c9620e1e7cfb8b009b | 1 | unclassified | 0.001 | -0.002 | 0.008 | 3 | 3 | 5.167 |
| 58de505db56b2250b12c7cd7ec5bfa50 | 1 | unclassified | 0.000 | 0.000 | 0.000 | 1 | 0 | 4.000 |
| c4a2579e95fee657cb355e865aa027af | 1 | unclassified | -0.000 | -0.002 | 0.002 | 3 | 3 | 5.167 |
| 783749c187ad3cb6e60ccecdd73fd8ef | 1 | unclassified | -0.001 | -0.002 | 0.001 | 3 | 3 | 5.167 |
| eb0898e38a817c5648a19a5e3db2c1be | 1 | unclassified | -0.001 | -0.002 | 0.001 | 3 | 3 | 5.167 |
| 88fa6e2c26361b8b7ca8a72ae1ed781f | 1 | unclassified | -0.001 | -0.002 | 0.001 | 2 | 2 | 6.250 |
| Methanomicrobiales | 50 | Methanospirillum (15); Methanoculleus (14); Methanolinea (10); uncultured (4); Methanosphaerula (3); Methanoregula (2); f__Methanoregulaceae; Methanocorpusculum | -0.001 | -0.006 | 0.003 | 2 | 4 | 4.167 |
| Methanofastidiosales | 5 | Candidatus_Methanofastidiosum (5) | -0.004 | -0.023 | 0.003 | 2 | 4 | 4.167 |
| Bathyarchaeia | 3 | Bathyarchaeia (3) | -0.005 | -0.017 | 0.003 | 1 | 5 | 3.500 |
| Methanosarciniales__b4 | 1 | Methanosaeta | -0.027 | -0.050 | 0.002 | 1 | 5 | 2.333 |

The pattern is the canonical ammonia response and it is visible without any
prompting from the model: the versatile/methylotrophic *Methanosarcina* block
goes up with the flag, the obligately acetoclastic *Methanosaeta*
(*Methanothrix*) block goes down, and everything else — including the 50-ASV
hydrogenotrophic `Methanomicrobiales` block — is fused into a near-zero
background whose mean group size is
4.2 blocks.
Acetoclastic methanogens are the most ammonia-sensitive functional group in the
literature and *Methanosarcina* is the tolerant one, so the *direction* of both
coefficients matches the established ordering.

Two cautions that matter more than the match does. First, `Methanosarciniales__b2`
is also *Methanosaeta* — 13 ASVs of it — and its mean coefficient is
+0.002 with
3/6
fits positive, i.e. the genus does **not** move as one unit; the strong negative
signal is one ASV that the tree happens to separate. Second, hydrogenotrophic
methanogens do not rise here, which is what a clean syntrophic-acetate-oxidation
story would predict. The model found the tolerant-methanogen half of the
expected shift, not the whole of it.

### Bacteria: proteolysis and amino-acid fermentation on the numerator

Numerator (relative abundance higher when TAN > 2.5):

| block | n_asv | genus_composition | mean_beta | n_positive_fits | fits_retained | mean_relative_abundance | site_auc_ammonia |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bacteroidales__b1 | 8 | f__Dysgonomonadaceae (6); Fermentimonas; uncultured | 0.049 | 5 | 6 | 0.020 | 0.904 |
| Acholeplasmatales__b1 | 7 | Acholeplasma (7) | 0.036 | 6 | 6 | 0.006 | 0.956 |
| Caldicoprobacterales__b1 | 5 | Caldicoprobacter (5) | 0.035 | 4 | 4 | 0.004 | 0.924 |
| Clostridia__b7 | 23 | Fastidiosipila (23) | 0.034 | 6 | 6 | 0.091 | 0.888 |
| MBA03 | 7 | MBA03 (7) | 0.034 | 6 | 6 | 0.006 | 0.940 |
| Clostridia__b2 | 4 | Ruminiclostridium (2); HN-HF0106 (2) | 0.034 | 5 | 5 | 0.003 | 0.880 |
| Methanosarciniales__b5 | 9 | Methanosarcina (5); Methanomethylovorans (3); f__Methanosarcinaceae | 0.032 | 6 | 6 | 0.034 | 0.632 |
| Bacteroidales__b9 | 5 | Proteiniphilum (5) | 0.031 | 6 | 6 | 0.016 | 0.936 |
| Acholeplasmatales__b2 | 2 | Acholeplasma (2) | 0.027 | 6 | 6 | 0.002 | 0.840 |
| Syntrophomonadales | 13 | Syntrophomonas (10); uncultured (2); Syntrophothermus | 0.024 | 6 | 6 | 0.019 | 0.876 |

Denominator:

| block | n_asv | genus_composition | mean_beta | n_negative_fits | fits_retained | mean_relative_abundance | site_auc_ammonia |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bacteroidales__b7 | 4 | Bacteroidetes_vadinHA17 (4) | -0.038 | 6 | 6 | 0.015 | 0.152 |
| Petrotogales | 5 | SC103 (3); AUTHM297; Defluviitoga | -0.037 | 6 | 6 | 0.012 | 0.188 |
| MVP_15 | 3 | MVP-15 (3) | -0.031 | 6 | 6 | 0.001 | 0.128 |
| Burkholderiales | 54 | f__Comamonadaceae (13); SC-I-84 (6); Dechloromonas (4); f__Rhodocyclaceae (4); Lautropia (3); Thauera (3); Zoogloea (3); uncultured (2); Candidatus_Accumulibacter (2); unclassified (2); B1-7BS; Azospira; Denitratisoma; Brachymonas; Methylotenera; Propionivibrio; Acidovorax; AAP99; Sulfuritalea; Simplicispira; Comamonas; Limnobacter | -0.031 | 6 | 6 | 0.069 | 0.076 |
| Micrococcales | 9 | Tetrasphaera (5); f__Intrasporangiaceae (2); Microbacterium; Galbitalea | -0.030 | 6 | 6 | 0.006 | 0.108 |
| Methanosarciniales__b4 | 1 | Methanosaeta | -0.027 | 5 | 6 | 0.028 | 0.376 |
| Syntrophales__b1 | 18 | Smithella (12); Syntrophus (3); uncultured (3) | -0.023 | 6 | 6 | 0.025 | 0.128 |
| Syntrophorhabdales | 3 | Syntrophorhabdus (3) | -0.023 | 6 | 6 | 0.007 | 0.064 |
| Syntrophales__b2 | 3 | uncultured (3) | -0.022 | 6 | 6 | 0.005 | 0.104 |
| SAR324_clade_Marine_group_B_ | 3 | SAR324_clade(Marine_group_B) (3) | -0.022 | 6 | 6 | 0.005 | 0.176 |

Read as a guild, the numerator is protein-degrading and amino-acid-fermenting:
*Proteiniphilum* (proteolytic, and reported as a dominant acetate utiliser under
strong ammonia stress), the Dysgonomonadaceae/*Fermentimonas* block,
*Fastidiosipila*, the Anaerovorax/*Tissierella*/*Tepidimicrobium* block,
*Caldicoprobacter*, *Acholeplasma* and Christensenellaceae R-7. That is the
guild that *produces* ammonia from a protein-rich feed, so its association with
a high-TAN effluent is at least as consistent with shared substrate as with
tolerance to the inhibitor.

*Syntrophomonadales* (10 *Syntrophomonas* + *Syntrophothermus*) is the one
clear syntroph on the numerator, positive in
6/6
fits. Syntrophic butyrate oxidisers of this genus recur in ammonia-loaded
syntrophic-acetate-oxidising consortia. The denominator's syntrophs are the
*other* kind: `Syntrophales__b1` (12 *Smithella* + 3 *Syntrophus*) and
`Syntrophorhabdales` (*Syntrophorhabdus*), i.e. propionate and aromatic
oxidisers — and ammonia dosing of digester sludge has been reported to inhibit
syntrophic propionate oxidation, with *Smithella* and *Methanosaetaceae*
inhibited together. Those two blocks moving down while *Methanosaeta* moves
down is the most internally coherent part of this model.

The rest of the denominator is not a digester story at all: `Burkholderiales`
(54 ASVs: Comamonadaceae, *Dechloromonas*, *Thauera*, *Zoogloea*),
`Micrococcales` (*Tetrasphaera*), `Rhodobacterales`, `Xanthomonadales`,
`Microtrichales`. These are aerobic and facultative activated-sludge organisms.
In full-scale digesters a large share of such populations are inactive
immigrants arriving with the feed, and inactive populations are specifically
associated with aerobes and facultative anaerobes. The honest reading is that
these blocks index **feed composition and retention**, not inhibition — and
because a log-contrast is a ratio, a denominator that tracks dilution by feed
will look like a predictor of the numerator's flag.

### What fusion actually did here

Fused sets that recur across outer fits, largest contribution first:

| domain | n_members | n_fits | fits | mean_group_beta | mean_theta | n_positive | n_negative | members |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BAC | 5 | 3 | r0f1; r1f0; r1f1 | 0.039 | 0.197 | 3 | 0 | Acholeplasmatales__b1; Caldicoprobacterales__b1; Clostridia__b2; Clostridia__b7; MBA03 |
| BAC | 4 | 2 | r0f1; r1f1 | -0.024 | -0.096 | 0 | 2 | Anaerolineales; Syntrophales__b1; Syntrophales__b2; Syntrophorhabdales |
| BAC | 2 | 2 | r0f1; r1f2 | -0.022 | -0.043 | 0 | 2 | Kosmotogales; SAR324_clade_Marine_group_B_ |
| BAC | 3 | 2 | r0f1; r1f1 | -0.012 | -0.036 | 0 | 2 | Flavobacteriales__b1; Fusobacteriales; Lachnospirales |
| BAC | 2 | 2 | r1f0; r1f2 | 0.011 | 0.021 | 1 | 0 | Bacteroidales__b2; Synergistales__b2 |
| BAC | 2 | 4 | r0f1; r1f0; r1f1; r1f2 | 0.010 | 0.019 | 3 | 1 | JS1; Sphingobacteriales__b4 |
| BAC | 2 | 2 | r0f1; r1f1 | -0.004 | -0.008 | 1 | 1 | Caldatribacteriales; DTU014__b1 |
| ARC | 6 | 3 | r0f1; r1f1; r1f2 | -0.001 | -0.005 | 1 | 2 | 4194fdb699f4d5c9620e1e7cfb8b009b; 783749c187ad3cb6e60ccecdd73fd8ef; 88fa6e2c26361b8b7ca8a72ae1ed781f; c17275cccc27dc9a4ab351a310c063c8; c4a2579e95fee657cb355e865aa027af; eb0898e38a817c5648a19a5e3db2c1be |
| ARC | 5 | 2 | r0f1; r1f2 | 0.001 | 0.004 | 1 | 1 | Bathyarchaeia; Methanofastidiosales; Methanomassiliicoccales; Methanomicrobiales; Methanosarciniales__b2 |

Only the sets that keep one sign every time they appear are worth naming:
5 of 9 do, led
by the Firmicutes numerator group and the syntroph denominator group quoted in
the findings. The six-member archaeal group is the six
unclassified single-ASV blocks pinned together at a coefficient near zero — that
is the penalty removing them from the model, not a biological grouping.

Fusion is also doing most of the compression on the archaeal side:
13.0 retained ARC blocks become
5.5 active groups
(57% fewer),
against 64.2 to
38.2 for bacteria
(40% fewer).

## Acid high (pH > 8.0) — ARC-only fusion, order level

Six outer fits, all available; 1 of them
has no positive test sample, so its fold-level MCC is 0 by construction and the
pooled out-of-fold metrics in the header are the ones to read.

Per fit: 13.0 ARC and
64.2 BAC blocks retained,
3.7 + 60.3
active groups. The BAC side is unfused by construction, so its "groups" are
simply its retained blocks.

### The archaeal side is mostly switched off

| repeat | outer_fold | arc_groups | max_abs_arc_beta | arc_contributes |
| --- | --- | --- | --- | --- |
| 0 | 0 | 1 | 0.000 | False |
| 0 | 1 | 4 | 0.021 | True |
| 0 | 2 | 6 | 0.046 | True |
| 1 | 0 | 4 | 0.040 | True |
| 1 | 1 | 2 | 0.000 | False |
| 1 | 2 | 8 | 0.020 | True |

In 2 of 6 fits every archaeal block is fused into
1 or 2 group(s) at a coefficient of
order 1e-10 — numerically zero — and is reported inactive. In the remaining fits
ARC splits into 4 to 8
groups with coefficients up to 0.046, against up to
0.074 on the bacterial side. The only
archaeal block with a consistent direction is again
`Methanosarciniales__b4`, the single *Methanosaeta* ASV
(-0.019,
4/6
fits negative).

So the winning "ARC-only fusion" arm for this flag is winning by *regularising
the archaea away* and letting the 62-67
bacterial blocks fit freely. That is a
legitimate answer to Experiment 1's question — where should the fusion penalty
act — but it is not an archaeal biomarker result, and it should not be reported
as one.

### The bacterial contrast, and why it looks familiar

| block | n_asv | genus_composition | mean_beta | n_positive_fits | fits_retained | site_auc_acid_high |
| --- | --- | --- | --- | --- | --- | --- |
| Caldicoprobacterales__b1 | 5 | Caldicoprobacter (5) | 0.057 | 4 | 4 | 0.940 |
| Bacteroidales__b9 | 5 | Proteiniphilum (5) | 0.053 | 6 | 6 | 0.927 |
| Acholeplasmatales__b1 | 7 | Acholeplasma (7) | 0.044 | 6 | 6 | 0.910 |
| Clostridia__b7 | 23 | Fastidiosipila (23) | 0.038 | 6 | 6 | 0.846 |
| Bacteroidales__b1 | 8 | f__Dysgonomonadaceae (6); Fermentimonas; uncultured | 0.037 | 6 | 6 | 0.799 |
| Oscillospirales__b1 | 7 | [Eubacterium]_coprostanoligenes_group (5); uncultured; Ruminococcus | 0.034 | 6 | 6 | 0.611 |
| MBA03 | 7 | MBA03 (7) | 0.031 | 6 | 6 | 0.893 |
| Synergistales__b1 | 8 | uncultured (3); Thermovirga (3); JGI-0000079-D21; Acetomicrobium | 0.027 | 6 | 6 | 0.457 |
| Peptostreptococcales_Tissierellales__b1 | 25 | uncultured (5); Anaerovorax (4); Tissierella (3); Tepidimicrobium (3); Guggenheimella (2); Alkaliphilus (2); W5053 (2); f__Anaerovoracaceae (2); Fusibacter; Amnipila | 0.021 | 6 | 6 | 0.786 |
| Desulfotomaculales | 1 | Pelotomaculum | 0.020 | 5 | 6 | 0.385 |

| block | n_asv | genus_composition | mean_beta | n_negative_fits | fits_retained | site_auc_acid_high |
| --- | --- | --- | --- | --- | --- | --- |
| Bacteroidales__b7 | 4 | Bacteroidetes_vadinHA17 (4) | -0.038 | 6 | 6 | 0.132 |
| Bacteroidales__b8 | 12 | Bacteroidetes_vadinHA17 (12) | -0.034 | 6 | 6 | 0.132 |
| Synergistales__b2 | 8 | uncultured (3); Syner-01 (2); f__Synergistaceae; Lactivibrio; Aminobacterium | -0.034 | 5 | 6 | 0.338 |
| SAR324_clade_Marine_group_B_ | 3 | SAR324_clade(Marine_group_B) (3) | -0.029 | 6 | 6 | 0.068 |
| Izemoplasmatales | 3 | Izemoplasmatales (3) | -0.028 | 6 | 6 | 0.150 |
| Caldisericales | 5 | WCHB1-02 (3); Caldisericum; LF045 | -0.025 | 6 | 6 | 0.141 |
| Bacteroidales__b3 | 2 | SB-5; MgMjR-022 | -0.024 | 5 | 6 | 0.462 |
| Christensenellales__b2 | 29 | Christensenellaceae_R-7_group (29) | -0.024 | 6 | 6 | 0.368 |
| Spirochaetales | 10 | uncultured (5); Sphaerochaeta (4); Treponema | -0.020 | 6 | 6 | 0.214 |
| Methanosarciniales__b4 | 1 | Methanosaeta | -0.019 | 4 | 6 | 0.457 |

The numerator is the ammonia numerator again — *Caldicoprobacter*,
*Proteiniphilum*, *Acholeplasma*, *Fastidiosipila*, Dysgonomonadaceae, MBA03,
the Anaerovorax/Tissierella block — with `Synergistales__b1`
(*Thermovirga*/*Acetomicrobium*, amino-acid degraders) added. Given that
protein degradation is what supplies both the ammonium and the bicarbonate
alkalinity that holds pH above 8, a proteolytic numerator for an alkaline-pH
flag is the expected result rather than a new finding.

Of the 19 blocks that are substantial in both models
(|mean beta| > 0.015), 2 switch sides:
`Clostridia__b3` (+0.019 for ammonia, -0.016 here); `Christensenellales__b2` (+0.017 for ammonia, -0.024 here).
Neither flip is visible in the raw abundances. `Christensenellales__b2`, a
29-ASV Christensenellaceae R-7 block, has a site-level AUROC of
0.37
for acid high and
0.45
for ammonia — both *below* 0.5, i.e. marginally lower in positives for both
flags, even though the ammonia model gives it a positive coefficient. The
positive coefficient is therefore entirely conditional on the rest of that
model's contrast, and the sign disagreement between the two models is a property
of the fits rather than a physiological distinction between the two conditions.
Do not read either sign for this block on its own.

## One contrast, two labels

| domain | block | mean_beta_ammonia | mean_beta_acid_high | same_sign |
| --- | --- | --- | --- | --- |
| BAC | Caldicoprobacterales__b2 | 0.033 | 0.068 | True |
| BAC | Caldicoprobacterales__b1 | 0.035 | 0.057 | True |
| BAC | Bacteroidales__b1 | 0.049 | 0.037 | True |
| BAC | Oscillospirales__b2 | 0.059 | 0.027 | True |
| BAC | Bacteroidales__b9 | 0.031 | 0.053 | True |
| BAC | Acholeplasmatales__b1 | 0.036 | 0.044 | True |
| BAC | Bacteroidales__b7 | -0.038 | -0.038 | True |
| BAC | Clostridia__b7 | 0.034 | 0.038 | True |
| BAC | MBA03 | 0.034 | 0.031 | True |
| BAC | SAR324_clade_Marine_group_B_ | -0.022 | -0.029 | True |
| BAC | Clostridia__b2 | 0.034 | 0.017 | True |
| BAC | MVP_15 | -0.031 | -0.016 | True |
| BAC | Petrotogales | -0.037 | -0.008 | True |
| BAC | Micrococcales | -0.030 | -0.016 | True |
| ARC | Methanosarciniales__b4 | -0.027 | -0.019 | True |
| BAC | Peptostreptococcales_Tissierellales__b1 | 0.022 | 0.021 | True |

Across all 85 blocks retained by both models the mean coefficients
correlate at Pearson r = 0.56 (65% sign agreement);
restricted to bacteria, r = 0.56. Of the 19 blocks that are
substantial in both models (|mean beta| > 0.015), 17
agree in sign.

This is the single most important caveat in this report. The two models were
selected independently, on different arms, and they are describing the same
gradient: protein-rich, ammonia-loaded, alkaline digesters versus the rest. The
acid-high model's apparent independent confirmation of the ammonia signature is
not independent.

## Functional annotation and its limits

`functional_annotation.csv` carries 37 block-level role hypotheses;
`unassigned_lineages.csv` records what was deliberately left blank and why;
`background_literature.csv` records the general claims used in the discussion
above. Each annotation states its `basis`: `expectation` means no primary
source was retrieved for that block in this session and the role is a taxonomic
expectation only.

| domain | block | role | basis |
| --- | --- | --- | --- |
| ARC | Methanomicrobiales | hydrogenotrophic methanogenesis | isolate genome + community evidence |
| ARC | Methanosarciniales__b2 | acetoclastic methanogenesis | culture/community evidence |
| ARC | Methanosarciniales__b3 | acetoclastic methanogenesis | culture/community evidence |
| ARC | Methanosarciniales__b4 | acetoclastic methanogenesis | culture/community evidence |
| ARC | Methanosarciniales__b5 | versatile methanogenesis (acetoclastic, hydrogenotrophic, methylotrophic) | culture/community evidence; species dependent |
| ARC | Methanomassiliicoccales | hydrogen-dependent methyl reduction | culture evidence outside digesters |
| ARC | Methanofastidiosales | hydrogen-dependent methylated-thiol reduction | genome-inferred potential |
| ARC | Methanomethyliales | methyl-reducing methanogenesis | genome-inferred potential |
| BAC | Bacteroidales__b9 | proteolytic fermentation; acetate utilisation under ammonia stress | culture evidence + reactor community evidence |
| BAC | Bacteroidales__b1 | saccharolytic/proteolytic fermentation | expectation |
| BAC | Bacteroidales__b4 | mixed fermentation | expectation |
| BAC | Clostridia__b7 | amino-acid fermentation | expectation |
| BAC | Peptostreptococcales_Tissierellales__b1 | proteolytic and amine/amino-acid fermentation | expectation |
| BAC | Synergistales__b1 | amino-acid degradation | culture/genome evidence outside digesters |
| BAC | Synergistales__b2 | amino-acid degradation | expectation |
| BAC | Syntrophomonadales | syntrophic fatty-acid (butyrate) oxidation | digester isotope/genome evidence |
| BAC | Syntrophales__b1 | syntrophic propionate oxidation | culture + ammonia-dosing evidence |
| BAC | Syntrophales__b2 | syntrophic oxidation (family-level) | expectation |
| BAC | Syntrophorhabdales | syntrophic aromatic-compound oxidation | expectation |
| BAC | Desulfotomaculales | syntrophic propionate oxidation / sulfate reduction | expectation |
| BAC | Kosmotogales | saccharolytic Thermotogota; syntroph-associated | expectation |
| BAC | Petrotogales | thermophilic saccharolytic Thermotogota | expectation |
| BAC | Caldicoprobacterales__b1 | xylanolytic/proteolytic thermophilic fermentation | expectation |
| BAC | Caldicoprobacterales__b2 | xylanolytic/proteolytic thermophilic fermentation | expectation |
| BAC | Acholeplasmatales__b1 | fermentative Mollicutes | expectation |
| BAC | Acholeplasmatales__b2 | fermentative Mollicutes | expectation |
| BAC | Christensenellales__b2 | fermentation | expectation |
| BAC | Clostridia__b2 | cellulolytic/saccharolytic fermentation | expectation |
| BAC | Oscillospirales__b1 | fermentation | expectation |
| BAC | Spirochaetales | fermentation | expectation |
| BAC | Anaerolineales | fermentative Chloroflexi | expectation |
| BAC | Sphingobacteriales__b2 | fermentation | expectation |
| BAC | Burkholderiales | aerobic/denitrifying activated-sludge taxa; likely influent-associated | digester immigration evidence |
| BAC | Micrococcales | activated-sludge Actinobacteriota; likely influent-associated | digester immigration evidence |
| BAC | Rhodobacterales | aerobic/facultative Alphaproteobacteria; likely influent-associated | digester immigration evidence |
| BAC | Xanthomonadales | aerobic Gammaproteobacteria; likely influent-associated | digester immigration evidence |
| BAC | Microtrichales | aerobic Actinobacteriota; likely influent-associated | digester immigration evidence |

What this cannot establish, stated plainly:

- These are 16S rRNA amplicons. No gene, transcript, pathway, flux or syntrophic
  interaction was measured. A role hypothesis describes *cultured relatives of
  the named genera in that block*, not the organisms here.
- Extracted DNA does not distinguish active cells from inactive immigrants. This
  is not a marginal concern for the aerobic denominator blocks; it is the leading
  alternative explanation for them.
- Every coefficient is a conditional, relative association inside a log-ratio. It
  identifies no causal effect and no absolute abundance change.
- A fused group is an equality constraint that improved held-out loss. It is not
  a functional guild, and its taxonomic coherence is partly imposed by the graph
  that proposed the edges.
- Directions are cross-validation fold counts, not confidence intervals or
  independent replication. With 24 and
  15 positives over
  10 and 9 sites, a handful of
  sites can drive any one of these blocks.
- Both configurations were chosen as the best of 20 on held-out scores. No
  multiplicity correction was applied, and no held-out cohort remains.

## Files

| File | Contents |
|---|---|
| `tables/selection_ranking.csv` | All 4 resolutions x 5 models x 2 rules per flag, recomputed; rank by MCC and F2 |
| `tables/fit_inventory.csv` | The 6 outer fits per model: candidate, threshold, retained/active/edge counts per domain, residuals, fold metrics |
| `tables/taxon_coefficients.csv` | Per block: genus composition, per-fit coefficient summary, sign agreement, group size, abundance and site-level AUROC |
| `tables/group_catalog.csv` | Every fused group in every fit: exact members, group coefficient, theta, activity, phyla |
| `tables/recurrent_groups.csv` | Multi-member fused sets recurring across outer fits |
| `tables/domain_contrasts.csv` | Per fit and domain: numerator/denominator mass, top members, zero-sum residual |
| `tables/cross_flag_agreement.csv` | Mean coefficient of each block in both models, with sign agreement |
| `tables/abundance_context.csv` | Model-free size and direction of each block: relative abundance, detection prevalence, site-level AUROC per flag |
| `tables/block_catalog.csv` | feature_id -> order block -> ASV count, genus composition, phylum, minimum taxonomy confidence |
| `tables/functional_annotation.csv` | Curated block-level role hypotheses with basis and source |
| `tables/unassigned_lineages.csv` | Lineages deliberately left unannotated, with reasons |
| `tables/background_literature.csv` | Literature claims used in the discussion |
| `tables/metric_reproduction.csv` | Saved fold metrics reproduced from saved probabilities |
| `figures/01_taxon_coefficients` | Largest mean coefficients per flag, with across-fit range and sign agreement |
| `figures/02_archaeal_blocks` | Every archaeal block, every outer fit, both flags |
| `figures/03_cross_flag_agreement` | The two models' coefficients against each other |

`validation.json` records the snapshot verification, zero-sum and
group-reconstruction residuals, the fusion-restriction checks, site isolation of
every split, metric reproduction, and script hashes.
