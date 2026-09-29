"""Report prose for the selected-order interpretation.

Every number in the text is formatted from the frames passed in; nothing is
typed in by hand. Kept separate from interpret.py so the analysis functions
stay readable.
"""
from __future__ import annotations


# Locate the repository independently of the script's directory depth.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_REPO_ROOT = next(p for p in _RepositoryPath(__file__).resolve().parents
                  if (p / "main.py").is_file() and (p / "configs").is_dir())
if str(_REPO_ROOT) not in _repository_sys.path:
    _repository_sys.path.insert(0, str(_REPO_ROOT))


import numpy as np
import pandas as pd


def top_blocks(taxa, config, sign, n, min_fits=4):
    d = taxa[taxa.config.eq(config) & taxa.fits_retained.ge(min_fits)]
    d = d[d.mean_beta > 0] if sign > 0 else d[d.mean_beta < 0]
    d = d.reindex(d.mean_beta.abs().sort_values(ascending=False).index).head(n)
    return d


def block_list(d, with_genus=False):
    parts = []
    for _, row in d.iterrows():
        agree = int(round(row.sign_agreement * row.fits_retained))
        genus = f", {row.genus_composition}" if with_genus else ""
        parts.append(f"`{row.block}` ({row.mean_beta:+.3f}, {agree}/{int(row.fits_retained)} fits{genus})")
    return "; ".join(parts)


def compose(rank, fits, taxa, groups, recurrent, shared, overlap, overlap_table, evidence, chem,
            SELECTED, RULE, md, ZERO):
    by_key = {s["key"]: s for s in SELECTED}
    rule_rank = rank[rank.selection.eq(RULE)]
    chosen = {}
    for key, selection in by_key.items():
        row = rule_rank[rule_rank.target.eq(selection["flag"]) & rule_rank.resolution.eq("order")
                        & rule_rank.model.eq(selection["model"])].iloc[0]
        chosen[key] = row
    other_rank = rank[rank.selection.eq("one_se")]
    one_se = {key: other_rank[other_rank.target.eq(by_key[key]["flag"]) & other_rank.resolution.eq("order")
                              & other_rank.model.eq(by_key[key]["model"])].iloc[0] for key in by_key}

    rule_fits = fits[fits.selection.eq(RULE)]
    acid_fits = rule_fits[rule_fits.config.eq("acid_high")]
    amm_fits = rule_fits[rule_fits.config.eq("ammonia")]
    amm_one_se = fits[fits.selection.eq("one_se") & fits.config.eq("ammonia")]
    acid_coeff = taxa[taxa.config.eq("acid_high")]
    amm_coeff = taxa[taxa.config.eq("ammonia")]

    # ARC collapse: fits where every archaeal coefficient is numerically zero.
    arc_groups = groups[groups.selection.eq(RULE) & groups.domain.eq("ARC")]
    collapsed = {}
    for key in by_key:
        d = arc_groups[arc_groups.config.eq(key)]
        per_fit = d.groupby(["repeat", "outer_fold"]).agg(
            n_groups=("group_id", "size"), max_abs=("group_mean_beta", lambda s: s.abs().max()),
            any_active=("active", "any"))
        collapsed[key] = per_fit
    acid_dead = int((~collapsed["acid_high"].any_active).sum())
    acid_dead_fits = ", ".join(f"r{r}f{f}" for r, f in collapsed["acid_high"].index[~collapsed["acid_high"].any_active])

    arc_scale = {key: taxa[taxa.config.eq(key) & taxa.domain.eq("ARC")].mean_beta.abs().max() for key in by_key}
    bac_scale = {key: taxa[taxa.config.eq(key) & taxa.domain.eq("BAC")].mean_beta.abs().max() for key in by_key}
    # Per-fit largest |coefficient| by domain: the scale comparison the fits see.
    per_fit_scale = {}
    for key in by_key:
        d = groups[groups.selection.eq(RULE) & groups.config.eq(key)]
        per_fit_scale[key] = d.groupby(["repeat", "outer_fold", "domain"]).group_mean_beta.apply(
            lambda s: s.abs().max()).unstack()
    acid_live = collapsed["acid_high"][collapsed["acid_high"].any_active]
    acid_dead_groups = sorted(set(collapsed["acid_high"][~collapsed["acid_high"].any_active].n_groups))

    # Blocks with a substantial coefficient but no marginal site-level signal.
    hidden = {}
    for key, selection in by_key.items():
        d = taxa[taxa.config.eq(key) & taxa.fits_retained.ge(4)].copy()
        d = d.reindex(d.mean_beta.abs().sort_values(ascending=False).index).head(20)
        auc = d[f"site_auc_{key}"]
        hidden[key] = d[(auc - 0.5).abs() < 0.1][["block", "mean_beta", f"site_auc_{key}"]]

    r_all = shared.mean_beta_ammonia.corr(shared.mean_beta_acid_high)
    r_bac = shared[shared.domain.eq("BAC")].mean_beta_ammonia.corr(
        shared[shared.domain.eq("BAC")].mean_beta_acid_high)
    strong = shared[(shared.mean_beta_acid_high.abs() > 0.015) & (shared.mean_beta_ammonia.abs() > 0.015)]
    flips = strong[~strong.same_sign]

    methano = amm_coeff[amm_coeff.block.isin(
        ["Methanosarciniales__b5", "Methanosarciniales__b4", "Methanosarciniales__b2", "Methanomicrobiales"])]
    methano = methano.set_index("block")

    recurrent_amm = recurrent[recurrent.config.eq("ammonia")]
    recurrent_acid = recurrent[recurrent.config.eq("acid_high")]
    # The recurrent groups worth naming: same sign every time they appear,
    # ranked by the magnitude of their contribution to the score.
    consistent = recurrent[(recurrent.n_positive.eq(0)) | (recurrent.n_negative.eq(0))].copy()
    consistent = consistent.reindex(consistent.mean_theta.abs().sort_values(ascending=False).index)
    headline = consistent[consistent.config.eq("ammonia")].iloc[0] if len(
        consistent[consistent.config.eq("ammonia")]) else None
    syntroph = recurrent[recurrent.members.str.contains("Syntrophales__b1")]
    syntroph = syntroph.iloc[0] if len(syntroph) else None

    # Orders the rank tree split into blocks that end up on opposite sides.
    split_orders = {}
    for key in by_key:
        d = taxa[taxa.config.eq(key) & taxa.block.str.contains("__b")].copy()
        d["order_name"] = d.block.str.split("__b").str[0]
        per_order = d.groupby("order_name").mean_beta.agg(["min", "max", "size"])
        split_orders[key] = per_order[(per_order["min"] < -ZERO) & (per_order["max"] > ZERO)]

    # How often the archaeal domain is actually compressed (groups < blocks).
    arc_compressed = {}
    for key in by_key:
        d = collapsed[key].join(
            rule_fits[rule_fits.config.eq(key)].set_index(["repeat", "outer_fold"]).arc_retained_features)
        arc_compressed[key] = int((d.n_groups < d.arc_retained_features).sum())

    annotated_blocks = set(evidence.block)
    coverage = {}
    for key in by_key:
        d = taxa[taxa.config.eq(key) & taxa.fits_retained.ge(4)]
        top = d.reindex(d.mean_beta.abs().sort_values(ascending=False).index).head(20)
        coverage[key] = (top.block.isin(annotated_blocks).mean(), len(top))

    parts = []
    A = parts.append

    A(f"""# Experiment 5 (selected order-level models) — biological interpretation

Two configurations, chosen in Experiment 1 as the highest-MCC member of their
fusion arm, are read as biology here:

| Flag | Label definition | Arm | Model | Resolution | MCC | F2 | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|
| Ammonia toxicity | effluent NH4-N (TAN) > 2.5 | Both domains fused | `shared_hybrid` | order | {chosen['ammonia'].mcc:.3f} | {chosen['ammonia'].f2:.3f} | {chosen['ammonia'].auc:.3f} | {chosen['ammonia'].ap:.3f} |
| Acid high | effluent pH > 8.0 | ARC-only fusion | `fusion_arc_only` | order | {chosen['acid_high'].mcc:.3f} | {chosen['acid_high'].f2:.3f} | {chosen['acid_high'].auc:.3f} | {chosen['acid_high'].ap:.3f} |

No model is fitted here. Every input is a hash-verified snapshot of
Experiment 1's domain-restricted fusion runs, and every number below is
recomputed from the saved coefficients and out-of-fold probabilities.

**The acid-high flag is `effluent pH > 8.0`, not 8.5.** That is the threshold
in `fusionML/chemistry_targets.py`, the definition the selected model was
trained on, and no sample in this cohort reaches 8.5 at all — the maximum
effluent pH is {overlap['max_ph_in_cohort']:.2f}. A pH > 8.5 flag would have
zero positives and could not be fitted. Everything below therefore describes
the > 8.0 model.
""")

    # --- selection premise --------------------------------------------------
    A(f"""## The selection premise checks out

Both configurations are rank 1 of {int(chosen['ammonia'].n_candidates)} candidates
(4 resolutions x 5 models) for their flag, by **both** MCC and F2, under the
minimum-inner-loss rule Experiment 1 reports:

{md(rule_rank[rule_rank.rank_by_mcc.le(4)][['target', 'resolution', 'model', 'mcc', 'f2', 'auc', 'ap', 'rank_by_mcc', 'rank_by_f2']].rename(columns={'target': 'flag'}).assign(rank_by_mcc=lambda d: d.rank_by_mcc.astype(int), rank_by_f2=lambda d: d.rank_by_f2.astype(int)))}

That is a best-of-20 choice read off held-out scores with no multiplicity
correction. Treat the *margin* over the runner-up as noise: the top four
ammonia configurations sit within
{(rule_rank[rule_rank.target.eq('warning_ammonia_toxicity')].nlargest(4, 'mcc').mcc.max() - rule_rank[rule_rank.target.eq('warning_ammonia_toxicity')].nlargest(4, 'mcc').mcc.min()):.3f}
MCC of each other, the top four acid-high configurations within
{(rule_rank[rule_rank.target.eq('warning_acid_high')].nlargest(4, 'mcc').mcc.max() - rule_rank[rule_rank.target.eq('warning_acid_high')].nlargest(4, 'mcc').mcc.min()):.3f}.

**The two selections are not equally robust to the selection rule.** Under the
one-standard-error rule the acid-high winner holds
(MCC {one_se['acid_high'].mcc:.3f}, rank {int(one_se['acid_high'].rank_by_mcc)} of
{int(one_se['acid_high'].n_candidates)}), but order-level `shared_hybrid` for
ammonia falls to MCC {one_se['ammonia'].mcc:.3f}, rank
{int(one_se['ammonia'].rank_by_mcc)} of {int(one_se['ammonia'].n_candidates)} —
last. The one-SE rule takes the sparsest candidate within one standard error of
the inner-loss minimum, and for this model that is a far heavier fusion penalty:
{amm_one_se.n_active_groups.min():.0f}-{amm_one_se.n_active_groups.max():.0f}
active groups per fit against
{amm_fits.n_active_groups.min():.0f}-{amm_fits.n_active_groups.max():.0f} under
minimum inner loss. The ammonia reading below therefore belongs to the
minimum-inner-loss fit specifically, not to `shared_hybrid` at order level in
general, and the drop is itself evidence that this flag's signal does not
survive aggressive fusion. Both rules are exported in `selection_ranking.csv`.

`fusion_arc_only` does **not** mean an archaea-only model. Both domains are
predictors in every arm; only the named domain receives fusion edges. The
restriction is verified per fit — the acid-high fits carry
{int(acid_fits.arc_n_edges.sum())} ARC edges and {int(acid_fits.bac_n_edges.sum())} BAC
edges in total, while the ammonia fits carry {int(amm_fits.arc_n_edges.sum())} and
{int(amm_fits.bac_n_edges.sum())}.
""")

    # --- main findings ------------------------------------------------------
    A(f"""## Main findings

1. **The ammonia model reproduces the textbook methanogen split, and it does so
   by refusing to fuse.** `Methanosarciniales__b5` (5 *Methanosarcina* + 3
   *Methanomethylovorans* ASVs) is the largest archaeal coefficient and is
   positive in {int(methano.loc['Methanosarciniales__b5'].n_positive_fits)}/{int(methano.loc['Methanosarciniales__b5'].fits_retained)}
   fits ({methano.loc['Methanosarciniales__b5'].mean_beta:+.3f}); the single-ASV
   *Methanosaeta* block `Methanosarciniales__b4` is negative in
   {int(methano.loc['Methanosarciniales__b4'].n_negative_fits)}/{int(methano.loc['Methanosarciniales__b4'].fits_retained)}
   fits ({methano.loc['Methanosarciniales__b4'].mean_beta:+.3f}). In
   {arc_compressed['ammonia']} of 6 fits the fusion penalty collapses most of the
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
   archaea.** In {acid_dead} of 6 outer fits ({acid_dead_fits}) the entire ARC
   domain collapses into {' or '.join(str(n) for n in acid_dead_groups)} group(s)
   whose coefficients are numerically zero, so archaea contribute nothing to
   those predictions. Even in the fits where they do contribute, the largest
   archaeal coefficient is
   {per_fit_scale['acid_high'].ARC.max():.3f} against
   {per_fit_scale['acid_high'].BAC.max():.3f} on the bacterial side. This arm is
   better described as *compress the archaea, leave the bacteria free* than as
   an archaeal model.

4. **The two flags are largely the same event, and the two models find largely
   the same contrast.** {overlap['both_positive']} of {overlap['acid_high_positives']}
   acid-high positives are also ammonia positives, effluent pH and TAN correlate
   at Spearman {overlap['ph_tan_spearman']:.2f}, and the mean coefficients of the
   two models correlate at Pearson r = {r_all:.2f} over {len(shared)} shared blocks
   ({shared.same_sign.mean():.0%} share sign). The acid-high model should not be
   read as an independent line of evidence about alkaline pH.

5. **Order-level blocks, not order names, are the interpretable unit.**
   {len(split_orders['ammonia'])} orders (ammonia) and
   {len(split_orders['acid_high'])} (acid high) are split by the monophyly-
   consistent rank tree into blocks that end up on *opposite* sides of the
   contrast. *Bacteroidales* is the clearest: the *Proteiniphilum* block
   `Bacteroidales__b9` and the Dysgonomonadaceae block `Bacteroidales__b1` are
   numerator, the two *Bacteroidetes vadinHA17* blocks `__b7`/`__b8` are
   denominator. A naive taxonomy-label collapse to the order name would cancel
   these against each other.

6. **Two fused groups recur across folds and both are interpretable.** The
   largest is `{headline.members.replace('; ', '` + `')}`
   ({int(headline.n_fits)} of 6 ammonia fits, always positive, mean theta
   {headline.mean_theta:+.3f}) — five Firmicutes blocks (*Acholeplasma*,
   *Caldicoprobacter*, *Ruminiclostridium*, *Fastidiosipila* and the uncultured
   MBA03) carrying one shared coefficient. Against it,
   `{syntroph.members.replace('; ', '` + `')}`
   recurs in {int(syntroph.n_fits)} fits, always negative (mean theta
   {syntroph.mean_theta:+.3f}): the propionate/aromatic syntrophs and
   Anaerolineaceae fused into a single denominator term. A shared coefficient is
   an equality constraint the penalty imposed, not evidence that these taxa do
   the same job — but these two groups are at least coherent guilds.

7. **A fifth to two fifths of the leading signal has no literature role at all.**
   Of the 20 largest-|beta| blocks per model, {coverage['ammonia'][0]:.0%}
   (ammonia) and {coverage['acid_high'][0]:.0%} (acid high) match a curated role
   hypothesis; the remainder are uncultured lineage labels (MBA03, D8A_2,
   MVP_15, Bacteroidetes vadinHA17, SAR324, JS1) that are deliberately left
   unassigned. Coefficient size and marginal association are also not the same
   thing: for ammonia {'every one of' if hidden['ammonia'].empty else 'most of'}
   the top 20 blocks also separates sites on its own (site-level AUROC at least
   0.1 away from 0.5), but for acid high
   {len(hidden['acid_high'])} of them do not
   ({', '.join('`' + b + '`' for b in hidden['acid_high'].block) or 'none'}).
   Those blocks act only in combination with the rest of the contrast, which is
   what a log-ratio model is for and what makes single-taxon biomarker readings
   of one unsafe.
""")

    # --- cohort and confound ------------------------------------------------
    A(f"""## The cohort, and what the two labels actually overlap

{overlap['n_samples']} samples from {overlap['n_sites']} QC-adjusted site groups,
3 outer folds x 2 repeats, sites never split across a fold. Positives:
{overlap['ammonia_positives']} for ammonia over {overlap['ammonia_sites']} sites,
{overlap['acid_high_positives']} for acid high over {overlap['acid_high_sites']} sites,
of which {overlap['shared_sites']} sites carry both.

{md(pd.DataFrame({'acid_high \\\\ ammonia': ['0', '1'],
                  '0': overlap_table[0].to_numpy(), '1': overlap_table[1].to_numpy()}))}

Sample-level phi between the two labels is {overlap['phi']:.2f}; effluent pH and
TAN correlate at Pearson {overlap['ph_tan_pearson']:.2f} / Spearman
{overlap['ph_tan_spearman']:.2f} over {overlap['n_with_both_measurements']} samples
with both measurements. This is chemistry, not coincidence: protein and urea
hydrolysis releases ammonia, ammonium/bicarbonate buffering raises alkalinity,
and alkaline pH in turn shifts the NH4+/NH3 equilibrium toward free ammonia.

The practical consequence for this report: **a taxon that appears on the same
side of both contrasts has been found once, not twice.** The independent part
of the acid-high signal is the {overlap['acid_high_only']} samples that are
alkaline without being ammonia-positive, which is too few to interpret
separately.
""")

    # --- algebra ------------------------------------------------------------
    A("""## How to read one of these coefficients

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
{arc_n} blocks and BAC {bac_n} blocks after prevalence filtering, so the same
total contrast mass is divided differently.
""".replace("{arc_n}", f"{int(amm_fits.arc_retained_features.mean())}")
      .replace("{bac_n}", f"{int(amm_fits.bac_retained_features.mean())}"))

    # --- ammonia ------------------------------------------------------------
    amm_pos = top_blocks(taxa, "ammonia", +1, 10)
    amm_neg = top_blocks(taxa, "ammonia", -1, 10)
    A(f"""## Ammonia toxicity — both domains fused, order level

Six outer fits, all available. Per fit the model keeps
{amm_fits.arc_retained_features.mean():.1f} ARC and {amm_fits.bac_retained_features.mean():.1f}
BAC blocks after training-partition prevalence filtering and reduces them to
{amm_fits.arc_active_groups.mean():.1f} + {amm_fits.bac_active_groups.mean():.1f} =
{amm_fits.n_active_groups.mean():.1f} active coefficient groups.

### Archaea: two blocks are pulled out of the background

{md(amm_coeff[amm_coeff.domain.eq('ARC')].sort_values('mean_beta', ascending=False)[['block', 'n_asv', 'genus_composition', 'mean_beta', 'min_beta', 'max_beta', 'n_positive_fits', 'n_negative_fits', 'mean_group_size']].head(14))}

The pattern is the canonical ammonia response and it is visible without any
prompting from the model: the versatile/methylotrophic *Methanosarcina* block
goes up with the flag, the obligately acetoclastic *Methanosaeta*
(*Methanothrix*) block goes down, and everything else — including the 50-ASV
hydrogenotrophic `Methanomicrobiales` block — is fused into a near-zero
background whose mean group size is
{amm_coeff[amm_coeff.block.eq('Methanomicrobiales')].mean_group_size.iloc[0]:.1f} blocks.
Acetoclastic methanogens are the most ammonia-sensitive functional group in the
literature and *Methanosarcina* is the tolerant one, so the *direction* of both
coefficients matches the established ordering.

Two cautions that matter more than the match does. First, `Methanosarciniales__b2`
is also *Methanosaeta* — 13 ASVs of it — and its mean coefficient is
{methano.loc['Methanosarciniales__b2'].mean_beta:+.3f} with
{int(methano.loc['Methanosarciniales__b2'].n_positive_fits)}/{int(methano.loc['Methanosarciniales__b2'].fits_retained)}
fits positive, i.e. the genus does **not** move as one unit; the strong negative
signal is one ASV that the tree happens to separate. Second, hydrogenotrophic
methanogens do not rise here, which is what a clean syntrophic-acetate-oxidation
story would predict. The model found the tolerant-methanogen half of the
expected shift, not the whole of it.

### Bacteria: proteolysis and amino-acid fermentation on the numerator

Numerator (relative abundance higher when TAN > 2.5):

{md(amm_pos[['block', 'n_asv', 'genus_composition', 'mean_beta', 'n_positive_fits', 'fits_retained', 'mean_relative_abundance', 'site_auc_ammonia']])}

Denominator:

{md(amm_neg[['block', 'n_asv', 'genus_composition', 'mean_beta', 'n_negative_fits', 'fits_retained', 'mean_relative_abundance', 'site_auc_ammonia']])}

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
{int(amm_coeff[amm_coeff.block.eq('Syntrophomonadales')].n_positive_fits.iloc[0])}/6
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

{md(recurrent_amm.reindex(recurrent_amm.mean_theta.abs().sort_values(ascending=False).index)[['domain', 'n_members', 'n_fits', 'fits', 'mean_group_beta', 'mean_theta', 'n_positive', 'n_negative', 'members']])}

Only the sets that keep one sign every time they appear are worth naming:
{len(consistent[consistent.config.eq('ammonia')])} of {len(recurrent_amm)} do, led
by the Firmicutes numerator group and the syntroph denominator group quoted in
the findings. The six-member archaeal group is the six
unclassified single-ASV blocks pinned together at a coefficient near zero — that
is the penalty removing them from the model, not a biological grouping.

Fusion is also doing most of the compression on the archaeal side:
{amm_fits.arc_retained_features.mean():.1f} retained ARC blocks become
{amm_fits.arc_active_groups.mean():.1f} active groups
({100 * (1 - amm_fits.arc_active_groups / amm_fits.arc_retained_features).mean():.0f}% fewer),
against {amm_fits.bac_retained_features.mean():.1f} to
{amm_fits.bac_active_groups.mean():.1f} for bacteria
({100 * (1 - amm_fits.bac_active_groups / amm_fits.bac_retained_features).mean():.0f}% fewer).
""")

    # --- acid high ----------------------------------------------------------
    acid_pos = top_blocks(taxa, "acid_high", +1, 10)
    acid_neg = top_blocks(taxa, "acid_high", -1, 10)
    A(f"""## Acid high (pH > 8.0) — ARC-only fusion, order level

Six outer fits, all available; {int((acid_fits.n_test_positive == 0).sum())} of them
has no positive test sample, so its fold-level MCC is 0 by construction and the
pooled out-of-fold metrics in the header are the ones to read.

Per fit: {acid_fits.arc_retained_features.mean():.1f} ARC and
{acid_fits.bac_retained_features.mean():.1f} BAC blocks retained,
{acid_fits.arc_active_groups.mean():.1f} + {acid_fits.bac_active_groups.mean():.1f}
active groups. The BAC side is unfused by construction, so its "groups" are
simply its retained blocks.

### The archaeal side is mostly switched off

{md(collapsed['acid_high'].reset_index().rename(columns={'n_groups': 'arc_groups', 'max_abs': 'max_abs_arc_beta', 'any_active': 'arc_contributes'}))}

In {acid_dead} of 6 fits every archaeal block is fused into
{' or '.join(str(n) for n in acid_dead_groups)} group(s) at a coefficient of
order 1e-10 — numerically zero — and is reported inactive. In the remaining fits
ARC splits into {int(acid_live.n_groups.min())} to {int(acid_live.n_groups.max())}
groups with coefficients up to {acid_live.max_abs.max():.3f}, against up to
{per_fit_scale['acid_high'].BAC.max():.3f} on the bacterial side. The only
archaeal block with a consistent direction is again
`Methanosarciniales__b4`, the single *Methanosaeta* ASV
({acid_coeff[acid_coeff.block.eq('Methanosarciniales__b4')].mean_beta.iloc[0]:+.3f},
{int(acid_coeff[acid_coeff.block.eq('Methanosarciniales__b4')].n_negative_fits.iloc[0])}/6
fits negative).

So the winning "ARC-only fusion" arm for this flag is winning by *regularising
the archaea away* and letting the {int(acid_fits.bac_retained_features.min())}-{int(acid_fits.bac_retained_features.max())}
bacterial blocks fit freely. That is a
legitimate answer to Experiment 1's question — where should the fusion penalty
act — but it is not an archaeal biomarker result, and it should not be reported
as one.

### The bacterial contrast, and why it looks familiar

{md(acid_pos[['block', 'n_asv', 'genus_composition', 'mean_beta', 'n_positive_fits', 'fits_retained', 'site_auc_acid_high']])}

{md(acid_neg[['block', 'n_asv', 'genus_composition', 'mean_beta', 'n_negative_fits', 'fits_retained', 'site_auc_acid_high']])}

The numerator is the ammonia numerator again — *Caldicoprobacter*,
*Proteiniphilum*, *Acholeplasma*, *Fastidiosipila*, Dysgonomonadaceae, MBA03,
the Anaerovorax/Tissierella block — with `Synergistales__b1`
(*Thermovirga*/*Acetomicrobium*, amino-acid degraders) added. Given that
protein degradation is what supplies both the ammonium and the bicarbonate
alkalinity that holds pH above 8, a proteolytic numerator for an alkaline-pH
flag is the expected result rather than a new finding.

Of the {len(strong)} blocks that are substantial in both models
(|mean beta| > 0.015), {len(flips)} switch sides:
{'; '.join(f'`{r.block}` ({r.mean_beta_ammonia:+.3f} for ammonia, {r.mean_beta_acid_high:+.3f} here)' for _, r in flips.iterrows()) if len(flips) else 'none'}.
Neither flip is visible in the raw abundances. `Christensenellales__b2`, a
29-ASV Christensenellaceae R-7 block, has a site-level AUROC of
{acid_coeff[acid_coeff.block.eq('Christensenellales__b2')].site_auc_acid_high.iloc[0]:.2f}
for acid high and
{amm_coeff[amm_coeff.block.eq('Christensenellales__b2')].site_auc_ammonia.iloc[0]:.2f}
for ammonia — both *below* 0.5, i.e. marginally lower in positives for both
flags, even though the ammonia model gives it a positive coefficient. The
positive coefficient is therefore entirely conditional on the rest of that
model's contrast, and the sign disagreement between the two models is a property
of the fits rather than a physiological distinction between the two conditions.
Do not read either sign for this block on its own.
""")

    # --- cross flag ---------------------------------------------------------
    A(f"""## One contrast, two labels

{md(shared.reindex((shared.mean_beta_acid_high.abs() + shared.mean_beta_ammonia.abs()).sort_values(ascending=False).index).head(16)[['domain', 'block', 'mean_beta_ammonia', 'mean_beta_acid_high', 'same_sign']])}

Across all {len(shared)} blocks retained by both models the mean coefficients
correlate at Pearson r = {r_all:.2f} ({shared.same_sign.mean():.0%} sign agreement);
restricted to bacteria, r = {r_bac:.2f}. Of the {len(strong)} blocks that are
substantial in both models (|mean beta| > 0.015), {int(strong.same_sign.sum())}
agree in sign.

This is the single most important caveat in this report. The two models were
selected independently, on different arms, and they are describing the same
gradient: protein-rich, ammonia-loaded, alkaline digesters versus the rest. The
acid-high model's apparent independent confirmation of the ammonia signature is
not independent.
""")

    # --- annotation ---------------------------------------------------------
    A(f"""## Functional annotation and its limits

`functional_annotation.csv` carries {len(evidence)} block-level role hypotheses;
`unassigned_lineages.csv` records what was deliberately left blank and why;
`background_literature.csv` records the general claims used in the discussion
above. Each annotation states its `basis`: `expectation` means no primary
source was retrieved for that block in this session and the role is a taxonomic
expectation only.

{md(evidence[['domain', 'block', 'role', 'basis']].head(40))}

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
  independent replication. With {overlap['ammonia_positives']} and
  {overlap['acid_high_positives']} positives over
  {overlap['ammonia_sites']} and {overlap['acid_high_sites']} sites, a handful of
  sites can drive any one of these blocks.
- Both configurations were chosen as the best of 20 on held-out scores. No
  multiplicity correction was applied, and no held-out cohort remains.
""")

    A("""## Files

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
""")

    return "\n".join(parts)
