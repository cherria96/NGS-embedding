# Task: build genus-level feature tables and matching phylogenetic trees for Phylo-Spec / DeepPhylo

## 1. Context

I have 16S amplicon data from 140 full-scale anaerobic digesters, processed in QIIME 2.
I want to train Phylo-Spec and DeepPhylo (both in the `Phylospec/` repo) on it. Both models
take **two** inputs that must agree with each other: an abundance table and a rooted
phylogenetic tree. Your job is to produce that pair at genus level, correctly.

Read this whole file before writing code. The failure modes below are not hypothetical —
they are things I verified in the Phylo-Spec source and in its own shipped example data.

## 2. Why genus level, and why the obvious way to get there is wrong

**Why reduce at all.** I have 140 samples and roughly 10^4 ASVs. That is p >> n. Beyond the
statistical problem there are two mechanical limits: DeepPhylo builds a patristic distance
matrix that is O(p^2) before its PCA (20,000 ASVs is ~3.2 GB in float64), and Phylo-Spec
instantiates one `nn.Conv1d` per tree node, so cost scales with node count. Target
roughly 300–1500 features.

**Why genus specifically.** In my data, species-level taxonomy is ~80% unclassified, but
genus-level unclassified is only 3–5% of read abundance. Genus is therefore the deepest rank
where the labels are actually informative, and it is the rank at which the anaerobic digestion
literature and the MiDAS functional guilds are organised. Genus features are interpretable;
ASV hashes are not.

**The trap.** `level-7.csv` (and a `level-6.csv` if you generate one) is a table collapsed by
taxonomy *string*. Do not feed it to either model. Neither model reads taxonomic labels at all —
Phylo-Spec walks the tree topology and branch lengths via `get_conv_order`, DeepPhylo consumes
a PCA of the patristic distance matrix. Both consume **tree geometry**. A taxonomy-collapsed
table has no tree: the moment you merge ASVs by label, you have to answer "where does this
merged feature sit on the tree?" and for a non-monophyletic genus there is no correct answer.
Genera are frequently non-monophyletic on a 16S tree, and `uncultured` / `unassigned` is not a
taxon at all — its members are scattered across the whole tree. Merging those into one column
produces a feature with no defensible branch. **Both models will accept such a file without
error and train on a prior that is simply wrong.**

**The correct operation.** Collapse *on the tree*, not on the label. For each genus, find the
ASVs carrying that label, check whether they form a clade on the rooted tree, and if they do,
replace that clade with a single tip named for the genus. If they do not, split the genus into
its maximal monophyletic blocks (`Genus__b1`, `Genus__b2`, …) and collapse each block
separately. Features whose genus is unassigned are carried through individually as their own
ASV tips — never pooled. Every output column is then a real tip with a real branch length, and
the table/tree correspondence is exact by construction.

## 3. Inputs

Two parallel datasets, `data/qiimeresult/ARC` (archaea) and `data/qiimeresult/BAC` (bacteria),
each containing:

| file | shape |
|---|---|
| `table_filtered.tsv` | index = ASV ID, header = sample name, values = reads. **This is the abundance source.** |
| `rooted-tree.nwk` | Newick, tips = ASV IDs. **This is the tree source.** |
| `metadata.csv` | `Feature ID` = ASV ID, `Taxon` = taxonomy string. **This is the label source.** Despite the filename it is a taxonomy file, not sample metadata. |
| `level-7.csv` | index = sample, header = taxonomy string, values = reads. **Reference only — do not use as model input** (see §2). Useful as a cross-check that your collapsed abundances are sane. |

Note `table_filtered.tsv` is samples-in-columns and `level-7.csv` is samples-in-rows. Transpose
so that the model input is samples-in-rows.

### 3.1 Targets

The per-sample labels are five binary columns already appended to `data/final/metadata.csv`
(written by `code/metadata_targetwriter.py`), each a threshold on an effluent stability
indicator. **They are multi-label, not mutually exclusive** — 17 of 140 samples trigger more
than one.

| column | trigger | positives / 140 |
|---|---|---|
| `warning_acid_base_balance` | `eff_pH < 6.5` or `eff_pH > 8.0` | 21 |
| `warning_buffer_capacity` | `eff_TVFAs / eff_ALK >= 0.30` | 9 |
| `warning_acid_accumulation` | (`eff_HPro/eff_HAc > 1.4` and `eff_HPro > 0.1`) or `eff_HPro >= 0.8` | 4 |
| `warning_ammonia_toxicity` | `eff_TAN > 2.5` | 26 |
| `warning_biogas_quality` | `eff_CH4 < 50.0` | 7 |

Phylo-Spec's per-flag mode wants a single `Group` column of `Normal` / `Warning`, so emit one
subfolder per flag plus a `multi_output/` folder carrying all five as 0/1 columns. DeepPhylo
takes the (N, 5) matrix directly. Both must use the **same** fold assignment so the metrics are
comparable.

**The 140 samples are 36 sites sampled over up to four seasons, so they are not 140 independent
observations.** All seasons of a site must stay on the same side of every split. The full
evaluation protocol — site-grouped repeated CV, pooled out-of-fold scoring, per-flag MCC-tuned
thresholds, macro-averaged MCC/F2, and the 5-element `pos_weight` — is specified in §5A and
implemented in `evaluate_multilabel.py`. Read §5A before writing any training or scoring code.

## 4. Hard invariants — these are the acceptance criteria

Write an assertion for each. A run that violates any of them is a failed run, not a warning.

1. **Tip set == column set.** `set(tree.tip_labels) == set(table.columns) - {"Group"}`, exactly,
   in every output. Not a subset either way.
2. **No feature name is a substring of any other feature name.** See §5.2 — this is not
   cosmetic, it silently corrupts training.
3. **No column name contains the string `unclassified`** (case-insensitive). See §5.1.
4. **Abundance is conserved.** Total reads per sample after collapsing == total reads per
   sample after filtering, to floating point tolerance. Collapsing is a sum, never a mean.
5. **Archaea survive filtering** in the merged dataset. Methanogens are a few percent of
   digester reads; an abundance threshold tuned on bacterial counts deletes exactly the
   organisms this project is about. Apply a separate, laxer threshold to archaeal features.
6. **No branch length >= 1.** See §5.3.

## 5. Gotchas verified in the Phylo-Spec source — read these before designing

These are in `Phylospec/src/model/`. Line references are to `PhyloSpec.py`,
`data_processing.py`, and `PhyloSpec_train_test.py`.

### 5.1 The unmatched-column bypass ("the unclassified node")

In `PhyloSpec.AuxiliaryModel.forward`, Step 1 processes columns that match a tree tip. **Step 2
then loops over every column that does *not* match a tip**, gives it its own 1x1 conv, and
concatenates it directly into the pooled feature vector:

```python
matched_columns = set(leaf_to_species.values())
for column in data.columns[1:-1]:
    if column not in matched_columns:
        ...
        all_features.append(unmatched_feature)
```

Such a feature never enters `conv_order`. It gets no node weight, no parent, and contributes to
no internal node — it bypasses the phylogeny entirely and becomes a bare MLP channel. **This is
triggered by a name failing to match a tip, not by taxonomy.** If your column names are exactly
the tip labels, Step 2 is empty and 100% of features go through the tree. That is the target
state; assert it.

Separately, `data_processing.process_unclassified_features` fires on any column whose *name*
contains `Unclassified`. It looks the feature up in a taxonomy table by
`taxonomy_table['Species'] == feature`, then splits that column's abundance **uniformly** across
reference species that are absent from the table. It never modifies the tree. I ran it on the
repo's own `example/Unclassified/` data: the table went from 60 columns to 10,151, of which 50
were tree tips and **10,101 landed in the Step 2 bypass**. It is designed for a closed-reference
setting with a reference tree containing unobserved species; it is actively harmful with a
de novo QIIME 2 tree. **Do not pass `-taxo`, and make sure no column name contains
`unclassified`** — name unassigned features by their ASV ID instead.

### 5.2 `match_leaf_nodes` binds columns to tips by SUBSTRING

```python
for leaf in tree.get_terminals():
    for species in data.columns[1:-1]:
        if species in leaf.name:      # substring, not equality
            leaf_to_species[leaf.name] = species
            break
```

The repo's own `example/Unclassified/` data is already corrupted by this: 5 of its 50 leaves bind
to the wrong column (leaf `sp44` reads feature `sp4`; `sp42` also reads `sp4`; `sp37`, `sp30` and
`sp35` all read `sp3`). Only 45 distinct features feed the 50 leaves — five are read twice, five
are never read.

QIIME 2 md5 ASV hashes are safe. **Taxonomic names are not**: `Clostridium` is a substring of
`Clostridium_sensu_stricto_1`, and `Methanobacterium` of `Methanobacterium_b2`. So a
genus-labelled table walks straight into this bug. **Emit opaque fixed-width feature ids
(`F00001`, `F00002`, …) in both the table and the tree, and keep the readable genus name in a
separate mapping file.** Invariant 2 above is the check.

Also note `re.sub(r'\W+', '_', leaf)` is used to key the conv `ModuleDict`, so two labels
differing only in punctuation would share one layer. Opaque ids avoid this too.

### 5.3 Node weights are `1 - branch_length`

`calculate_node_weights` returns `1 - branch_length`. Any branch longer than 1 substitution/site
produces a **negative** multiplier that flips the sign of that node's activations and propagates
upward. QIIME 2's `align-to-tree-mafft-fasttree` on V3–V4 normally keeps lengths well under 1,
but chimeric or poorly aligned ASVs are exactly the ones that do not. Report the branch length
distribution; drop the offending tips (they are usually artefacts anyway) or, if you prefer to
keep them, patch the weight to `exp(-branch_length)` and say so in the report.

### 5.4 Rooting matters for Phylo-Spec but not DeepPhylo

`rooted-tree.nwk` from QIIME 2 is midpoint-rooted, which places the root on the longest path.
Phylo-Spec's `conv_order` is a post-order traversal **from the root**, so the root position
determines the entire nesting. DeepPhylo uses patristic distances, which are invariant to
rooting. For the merged dataset, confirm the root actually separates Archaea from Bacteria; if a
single long-branch artefact ASV has captured it instead, re-root with the archaea as outgroup.

### 5.5 The ARC / BAC merge — build one tree, do not graft two

Representative sequences **do** exist: `data/qiimeresult/{ARC,BAC}/dada2_rep_seqs.qza`. The
existing merge in `prepare_warning_targets_data.py` grafts the two separately midpoint-rooted
trees under a new root with `DOMAIN_CONNECT_BRANCH_LEN = 1.0`. Replace it.

**Why the graft is not acceptable as-is.** Grafting makes every cross-domain distance
`d(arc_tip → arc_root) + 2L + d(bac_root → bac_tip)`, so `L` alone fixes how far every archaeon
sits from every bacterium — the exact quantity the syntrophy work is meant to estimate. Measured
on the reference tree that ships with this repo (`Phylospec/database/16S/gg_13.nwk`, 2,446
archaeal and 96,876 bacterial tips), the median Bacteria↔Archaea patristic distance is **1.350**,
against within-Bacteria 0.735 and within-Archaea 0.600 — a ratio of only 1.84×. `L = 1.0`
contributes 2.0 *before* either tip depth is counted, i.e. roughly triple the real separation.
It also lands on the one value that breaks Phylo-Spec (§5.3): `1 - 1.0 = 0` zeroes both domain
roots, and since `feature_map[parent]` is what the level above consumes, the whole-domain and
whole-community aggregates go with them.

**ARC and BAC were amplified with different primer pairs.** This is settled, not something to
check. It rules out any joint de novo alignment: MAFFT over the union of the two rep-seq sets
would be aligning two different hypervariable regions with little real overlap, and FastTree
would then read branch lengths off the gap structure. Do not attempt
`align-to-tree-mafft-fasttree` on the combined sequences under any circumstances.

**What to build instead.**

1. **One tree by fragment insertion (do this).** Insert *both* rep-seq sets into a single
   reference backbone with `qiime fragment-insertion sepp`, using a reference that contains
   Archaea (SILVA; the Greengenes backbone is bacteria-dominated). SEPP is the correct tool here
   precisely *because* the primers differ: each fragment is placed independently against a
   full-length reference alignment, so the two amplicon regions never have to align to each
   other — they only each have to align to the reference. Cross-domain distances are then
   inherited from the reference rather than invented, and there is no `L`. Report the number of
   fragments SEPP rejected, per domain.
2. **Calibrated graft** (fallback only, if SEPP cannot be run). Use `calibrate_connect_len()` from
   `domain_merge_utils.py`, which solves `L = (1.350 − mean_arc_tip_depth − mean_bac_tip_depth)/2`
   floored at zero. On the reference tree the two tip depths already sum to ~1.37 against a
   cross-domain median of 1.35, so the true inter-domain stem is essentially zero; expect a small
   positive `L` here only because separate midpoint rooting makes the tip depths shallower.

Whichever route you take, **prune the one tree three ways** (ARC tips only, BAC tips only, both)
rather than building three trees. Then the only thing that differs between the three arms is
which features are present, not how distance is defined, and the arms are comparable.

### 5.6 Normalise each domain separately, and record why

`build_merged()` computes `arc_rel` and `bac_rel` independently, each summing to 1. **Keep it
that way.** Because the two domains were amplified with different primers, ARC and BAC read
counts measure different things and their ratio reflects primer efficiency and sequencing effort,
not community composition. Closing the concatenated table jointly would treat that artefact as
biology. Do not add a read-depth-ratio covariate for the same reason.

The cost has to be stated rather than hidden: the archaea:bacteria ratio is a first-order
digester-stability signal — methanogen washout under organic overload sits behind three of the
five flags — and with domain-specific primers **it is not recoverable from this data**. Every
model here can therefore only use *within-archaea* and *within-bacteria* composition plus
co-occurrence between them, never their relative mass. Put this in the limitations section; it
bounds what the syntrophy result can claim.

### 5.7 The distance matrix as written will not finish

`patristic_distance_matrix()` calls `Bio.Phylo`'s `tree.distance()` once per pair, and each call
re-walks the tree from the root: at 3,000 features that is 4.5M traversals. Use `fast_patristic()`
from `domain_merge_utils.py` — one post-order pass, every pair written once, verified equal to
`Bio.Phylo` to 2e-15. Likewise `prune_tree()` calls `Phylo.prune()` once per dropped tip; if the
raw trees have tens of thousands of ASVs, build the pruned tree by copying the retained subtree
instead.

## 5A. Evaluation protocol — this is prescriptive, not advisory

The task is five imbalanced binary outputs at *different* imbalance ratios, on 140 samples that
are really only 36 independent units. `evaluate_multilabel.py` implements everything below; use
it rather than reimplementing, and do not substitute accuracy, micro-averaged F1, or a single
80/20 split anywhere in the reporting.

### 5A.1 Splitting: group by site, always

The 140 samples are **36 sites sampled over up to four seasons**. Two seasons of the same
digester share a community lineage, a feedstock and an operator, so a sample-level split lets the
model recognise the site instead of the instability, and the reported score is then a measure of
site memorisation. Every split — outer CV folds, the inner validation split for early stopping,
and the split used to pick decision thresholds — is built on **sites**, then expanded to samples.
`site_grouped_cv()` does this, stratifying sites by their label vector aggregated with `max`.

Run `audit_site_label_support()` first and put its output in the report. The number that matters
is not positives-per-flag but **positive *sites* per flag**: if a flag's positives are
concentrated in one or two digesters, no split can give it independent evidence, and that is a
property of the study design that no method choice fixes.

### 5A.2 Score pooled out-of-fold predictions, not individual folds

Site-grouped 5-fold puts about 7 of the 36 sites in each test fold, so a flag whose positives sit
in one or two sites is simply **absent** from most test folds. Scoring fold-by-fold and averaging
then discards most of the evidence — on a synthetic cohort with exactly this design,
`acid_accumulation` was scorable in 5 of 25 folds and `buffer_capacity` in 15.

Because k-fold predicts every sample exactly once per repeat, pool the out-of-fold predictions
within a repeat and score once on all 140. Every flag then carries its full positive count in
every repeat. Repeat 5× with different site partitions and report the mean with a 95% percentile
interval across repeats. `evaluate_cv()` does this.

### 5A.3 Three reporting strategies, all three required

1. AUPRC-guided epoch monitoring and checkpoint selection. During model training and early stopping, model checkpoints are selected based on Macro-AUPRC rather than threshold-dependent metrics. Because output probability distributions shift continuously across epochs—especially when using pos_weight rebalancing—evaluating training progress at a fixed default threshold (e.g., 0.5) introduces severe artifacts and false performance drops. Macro-AUPRC acts as a threshold-free metric that tracks the model's fundamental ranking capacity across all five target flags, ensuring early stopping captures the checkpoint with the strongest feature representations before thresholding.
2. Macro-average aggregation, never micro. Performance is evaluated using the unweighted mean across the five flags, ensuring rare events like acid_accumulation (4 positives) contribute equally to common ones like ammonia_toxicity (26). Micro-averaging or raw accuracy is heavily dominated by the majority negative class, which masks severe class imbalance and misleadingly rewards trivial all-negative predictors. Per-flag performance tables must be reported alongside the macro-average, as an unweighted macro metric across disparate prevalences is non-interpretable without its underlying components.
3. Per-flag decision threshold tuning via MCC. After selecting the best checkpoint via Macro-AUPRC, optimal decision thresholds are tuned independently for each flag using inner-validation scores to maximize MCC. Given flag prevalences ranging from 3% to 19%, no global threshold (and strictly not 0.5) is optimal across all outputs. tune_thresholds_mcc() grid-searches each flag's operating point independently. MCC is the required tuning objective under high imbalance because it incorporates all four quadrants of the confusion matrix, penalizing majority-class cheating. Thresholds are strictly fitted within inner-validation folds via evaluate_cv() to prevent data leakage.
4. Asymmetric operating metrics ($F_2$, Precision, Recall). Model performance at the tuned MCC thresholds is reported using $F_2$ alongside precision and recall. $F_2$ weights recall four times higher than precision, aligning with the domain asymmetry: a missed instability risks digester failure, whereas a false alarm merely incurs an inspection. Reporting $F_2$, precision, and recall together exposes operational trade-offs that single-scalar summaries obscure.
5. AUPRC reporting and prevalence baseline lift. Final test set results must report AUPRC against each flag's baseline prevalence to quantify the "lift over baseline." Raw AUPRC values are context-dependent—an AUPRC of 0.15 represents strong predictive power at 3% prevalence but random performance at 19%. AUROC may be included for comparison but remains secondary due to its optimistic bias under extreme imbalance.

### 5A.4 Loss: `pos_weight` as a 5-element tensor

Use `torch.nn.BCEWithLogitsLoss(pos_weight=w)` with `w` a **5-dimensional** tensor, one entry per
flag — a scalar cannot express five different imbalance ratios. The conventional value is
`n_negative / n_positive` per flag, computed **on the training fold only**; deriving it from the
full dataset leaks the test fold's class balance. `pos_weight_from_labels()` returns it.

Full-dataset values, for reference only: acid_base_balance 5.7, buffer_capacity 14.6,
acid_accumulation 34.0, ammonia_toxicity 4.4, biogas_quality 19.0. **A weight of 34 on four
positives will destabilise training** — that one flag's gradient swamps the other four and the
usual failure is a model that predicts it everywhere. Pass `cap=10` or `cap="sqrt"`, report which
you used, and treat it as a declared modelling choice rather than a hyperparameter to quietly
tune. Note that `pos_weight` and the tuned thresholds interact: both shift the operating point,
so the threshold search must be re-run for every `pos_weight` setting, never carried over.

### 5A.5 What to report per arm

One table per experimental arm, rows = the five flags, columns = n_pos, positive sites, AUPRC,
AUPRC lift, MCC, F2, precision, recall, tuned threshold — each as mean [2.5th, 97.5th percentile]
over the 5 repeats — plus a macro row. Arms are compared on **macro MCC and macro F2**, and a
difference between arms is only claimed if the repeat-level intervals separate.
`acid_accumulation` is reported in every table but must not carry a conclusion at n = 4.

## 6. What to build

There is an existing script, `reduce_features_for_tree_models.py`, that already implements the
three reduction modes (`asv`, `phylo`, `genus`), the monophyly audit, the archaea-protecting
threshold (`--protect`), the opaque-id option (`--safe-ids`), and a `phylospec_audit` block
covering §5.1–§5.3. **Read it first and extend it rather than starting over.** It expects an
abundance CSV, a Newick tree, and a taxonomy file; the QIIME outputs above need adapting to
those shapes.

Deliverables:

1. `prepare_qiime_inputs.py` — reads a `data/qiimeresult/{ARC,BAC}` directory, transposes
   `table_filtered.tsv`, parses the `Taxon` strings from `metadata.csv` into ranked columns, and
   writes the abundance CSV + taxonomy CSV that the reduction script consumes. Handle both
   `;`-delimited Greengenes-style strings and any missing-rank padding.
2. `build_merged.py` — the §5.5 SEPP insertion, producing one backbone tree pruned three ways
   (ARC tips, BAC tips, both) and the merged abundance table under separate per-domain closure
   (§5.6). Import `fast_patristic`, `audit_tree_for_phylospec` and `rescale_for_node_weights`
   from `domain_merge_utils.py`; do not reimplement them.
3. A driver that runs the pipeline for `ARC`, `BAC`, and `merged`, at genus level with
   `--safe-ids`, and writes for each: `<name>_table.csv`, `<name>_tree.nwk`,
   `<name>_mapping.csv`, `<name>_report.json`.
4. `tests/test_invariants.py` — one test per invariant in §4, run against all three outputs.
5. An evaluation driver built on `evaluate_multilabel.py` (§5A). It supplies `fit_predict` for
   each model family and emits the per-arm table of §5A.5 as `results_<arm>.csv`. Do not write
   your own splitter, threshold search, or metric aggregation.
6. `REPORT.md` summarising, per dataset: input ASV count, features surviving filtering, output
   feature count, genera that were non-monophyletic and how they were split, percent of reads
   retained, percent of reads with no genus assignment, branch-length distribution, the
   `phylospec_audit` block, the SEPP rejection counts, and the `audit_site_label_support()`
   table.

Also produce, as a sanity check, an `asv`-mode and a `phylo`-mode output at a comparable feature
count. If genus beats distance-based agglomeration at the same p, the taxonomic grouping carries
information the tree geometry does not, and that is a result worth reporting; if they tie, the
tree prior is doing the work.

## 7. Do not

- Do not use `level-7.csv` (or any `level-N.csv`) as a model input table.
- Do not pool unassigned features into a single `Unclassified` column.
- Do not merge features by taxonomy label without checking monophyly on the tree.
- Do not average when collapsing — sum.
- Do not truncate or rename ASV hashes to short ids like `ASV1`, `ASV2`.
- Do not pass `-taxo` to `PhyloSpec_train_test.py`.
- Do not silently drop ASVs that are in the table but absent from the tree, or vice versa —
  report the count and ask me before deciding.
- Do not run MAFFT/FastTree over the combined ARC + BAC representative sequences. The primers
  differ; the alignment is meaningless (§5.5).
- Do not close the merged abundance table jointly across domains, and do not add a read-depth
  ratio feature (§5.6).
- Do not put two seasons of the same `Site` on opposite sides of any split (§5A.1).
- Do not report accuracy, micro-averaged F1, or a single 80/20 held-out score. Do not use 0.5 as
  a decision threshold (§5A.3).
- Do not select decision thresholds, `pos_weight`, early-stopping epochs, or any hyperparameter
  using test-fold data.
- Do not pass a scalar `pos_weight` to `BCEWithLogitsLoss` — it must be a 5-element tensor
  computed on the training fold (§5A.4).
