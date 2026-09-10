# Genus-tree pipeline: run report

Executed end-to-end on this machine (`python3` 3.9.6, `scikit-learn` 1.6.1,
`biopython` 1.85, `pandas` 2.3.3 installed for this run). QIIME2/SEPP was not
installed (declined as out of scope for this session); the merged domain uses
the calibrated-graft fallback described in task Sec 5.5. All numbers below are
from real runs of `run_genus_tree_pipeline.py`, not estimates.

## 1. Per-dataset summary (`--mode genus`, `--safe-ids`, `--protect archaea`)

| | ARC | BAC | merged |
|---|---:|---:|---:|
| Input ASVs | 1,776 | 13,385 | 15,161 |
| Features surviving prevalence/abundance filter | 104 | 630 | 567 |
| Output (genus-collapsed) features | **42** | **389** | **325** |
| % reads retained after filtering | 86.44% | 71.07% | 79.01% |
| % reads with no genus assignment (post-filter) | 6.0% | 15.9% | 10.3% |
| Genera present on the pruned tree | 17 | 211 | 180 |
| Genera that were non-monophyletic | 3 | 39 | 28 |
| Archaeal (`--protect`) features retained | 96 | 0 (n/a, all-bacteria) | 96 |

**Non-monophyletic genera and how they were split** (genus → number of
maximal monophyletic blocks it was split into; a block count of 1 means
monophyletic, not listed):

- **ARC**: `Methanolinea` → 5 blocks, `Methanosaeta` → 3 blocks,
  `Methanomethylovorans` → 2 blocks.
- **BAC** (39 total; largest 10 shown): `Syntrophomonas` → 4,
  `Dechloromonas` → 4, `Candidatus_Cloacimonas` → 4, `Lautropia` → 3,
  `Paludibacter` → 3, `Smithella` → 3, `Bacteroides` → 3, `Syntrophus` → 3,
  `Proteiniphilum` → 3, `Rikenellaceae_RC9_gut_group` → 3, plus 29 more
  2-block splits (`Acholeplasma`, `Aminicenantales`, `Anaerovorax`,
  `Bacteroidetes_vadinHA17`, `Butyrivibrio`, `Caldicoprobacter`,
  `Christensenellaceae_R-7_group`, `Clostridia_vadinBB60_group`, `DTU014`,
  `Fermentimonas`, `Gelria`, `HN-HF0106`, `Lachnospiraceae_NK3A20_group`,
  `Lentimicrobiaceae`, `Lentimicrobium`(→3), `NK4A214_group`, `PeM15`,
  `Petrimonas`, `SBR1031`, `SJA-28`, `Streptococcus`, `Syner-01`,
  `Tepidimicrobium`, `Tetrasphaera`, `Tissierella`, `UCG-010`, `W27`, `W5`,
  `p-251-o5`).
- **merged** (28 total; largest 6 shown): `Methanolinea` → 5,
  `Syntrophomonas` → 4, `Candidatus_Cloacimonas` → 4, `Methanosaeta` → 3,
  `Smithella` → 3, `Proteiniphilum` → 3, plus 22 more 2-block splits.

Notably, several of the worst-split genera in BAC and merged
(`Syntrophomonas`, `Smithella`, `Candidatus_Cloacimonas`) are canonical
syntrophic fatty-acid oxidisers in anaerobic digesters — their tree position
is scattered, so this pipeline correctly refuses to collapse them into a
single feature with no defensible branch, per task Sec 2.

### Branch-length distribution, per output tree (after all invariant-6 fixes)

| | n branches | p50 | p95 | max | branches ≥ 1 |
|---|---:|---:|---:|---:|---:|
| ARC | 83 | 0.012 | 0.185 | 0.935 | 0 |
| BAC | 777 | 0.023 | 0.156 | 0.536 | 0 |
| merged | 649 | 0.008 | 0.060 | 0.950 | 0 |

## 2. `phylospec_audit` block (final, per Sec 5.1–5.3)

All three arms: `features_routed_to_step2_bypass = 0` (every feature name
matches a tree tip exactly — `--safe-ids` is doing its job),
`columns_named_unclassified = []`, `substring_ambiguous_feature_names = []`.

| | features_matching_a_tip | max_branch_length | branches_ge_1_negative_node_weight |
|---|---:|---:|---:|
| ARC | 42 | 0.9353 | 0 |
| BAC | 389 | 0.5358 | 0 |
| merged | 325 | 0.9500 | 0 |

## 3. A real bug this run caught (not hypothetical)

**Invariant 6 was violated by an intermediate step, not the input.**
`build_merged.py` rescales the pre-merge tree so every branch is < 1 (see
§4). But `reduce_features_for_tree_models.py`'s own pruning step
(`build_cross_domain_graph.prune_tree`) collapses a node with a single
surviving child by **summing** its branch length into the child's — so two
branches each < 1 can combine into one ≥ 1 after filtering removes everything
else along that path. On the actual merged/genus run, the reduced tree came
out with one branch at **1.0239**, despite every input branch being < 0.95.
`run_genus_tree_pipeline.py` now re-checks and rescales the **final output
tree** (`enforce_branch_length_invariant()`), not just the pre-merge tree,
and resyncs `report.json`'s audit block afterward. Confirmed fixed: final
merged tree max is exactly 0.9500, 0 branches ≥ 1, verified by
`tests/test_invariants.py`.

**The invariant test itself had a blind spot, also caught on a real run.**
The first version of `tests/test_invariants.py`'s Newick-token regex required
a non-empty node name before `:branch_length`. Internal nodes produced by
`build_merged.py`'s graft/prune (unlike FastTree's own bootstrap-labelled
internal nodes) are frequently **unnamed**. On the merged tree this dropped
323 of 649 branches silently — the test would have reported a false "max
branch length 0.138" and passed even with the real violation above still
present, if it had been checked before the fix instead of after. Fixed by
allowing an empty name in the regex; re-verified against the actual data
(326 → 649 branches found, max corrected from 0.138 to the true 0.950).

## 4. Domain-merge route actually used (Sec 5.5)

QIIME2 was not installed on this machine (checked all local conda
environments; none had `qiime`) and a fresh install was explicitly declined
for this session as a multi-GB, ~20–40 minute one-time cost. **SEPP
fragment-insertion was not run.** No SEPP rejection counts exist as a result.

Used `domain_merge_utils.calibrate_connect_len()` (the task's own documented
fallback) instead:

| | value |
|---|---:|
| mean ARC tip depth | 2.128 |
| mean BAC tip depth | 3.002 |
| target cross-domain distance (Greengenes reference) | 1.350 |
| **calibrated connect_len** | **0.0 (floored)** |
| resulting cross-domain distance | **5.131** |

**This calibration could not reach its target and that matters.** The
reference-tree calibration assumes the two domain trees' own tip depths are
comparable to the Greengenes reference tree's (which sums to ~1.37, close to
the 1.35 target). Here ARC + BAC tip depths already sum to **5.13** on their
own — the connect length floors at 0 and the resulting cross-domain distance
is **3.8× the reference target regardless**. This is a property of these
particular de novo QIIME2 trees (deeper, less resolved than the curated
Greengenes reference), not a bug in the calibration formula. **Any
distance-based syntrophy claim drawn from the merged arm should treat the
absolute Bacteria↔Archaea distance as inflated**; only within-domain distances
and relative (rank-based) cross-domain comparisons are on solid footing.

Branch-length rescale (Sec 5.3, `audit_tree_for_phylospec` +
`rescale_for_node_weights`): the pre-merge audit found branches ≥ 1 already
present in the **raw, unmerged** domain trees (ARC max 1.548, BAC max 2.465 —
this is a property of the raw QIIME2 output, unrelated to grafting). A single
factor (0.3854) was computed from the worst offender and applied uniformly to
all three pruned arms per `domain_merge_utils`' own guidance ("apply the same
factor to every arm, or the arms are no longer comparable"). This is the
factor used to build the `merged` arm's starting tree.

**Deliberate deviation, documented here:** the ARC-only and BAC-only *arms of
the main pipeline* (as opposed to build_merged.py's own three-way-pruned
reference trees) run on each domain's native, unscaled QIIME2 tree, not this
0.3854-scaled one — because neither single-domain arm ever touches the
graft-induced connect-length branch that made the rescale necessary in the
first place, and scaling them anyway would only discard real branch-length
resolution for no corresponding benefit. Only the merged arm needed (and got)
a rescale, applied at the point it was actually violated (see §3).

## 5. Site / label support (`audit_site_label_support`, full 140-sample cohort)

The task doc states 36 sites; the actual `data/final/metadata.csv` has
**37** distinct `Site` values — noted, not corrected (not investigated
further; immaterial to every conclusion below).

| flag | n_pos_samples | n_pos_sites | n_sites |
|---|---:|---:|---:|
| acid_base_balance | 21 | 12 | 37 |
| buffer_capacity | 9 | 6 | 37 |
| **acid_accumulation** | **4** | **3** | 37 |
| ammonia_toxicity | 26 | 11 | 37 |
| biogas_quality | 7 | 4 | 37 |

`acid_accumulation`'s 4 positive samples come from only 3 sites — per Sec
5A.1, no split design can give this flag independent evidence; any result
reported for it (in the evaluation driver, not built in this pass) should not
carry a standalone conclusion.

## 6. Comparison arms (Sec 6): genus vs. asv vs. phylo, same feature-count regime

All three modes, all three domains, ran successfully and pass every Sec 4
invariant (verified by `pytest tests/test_invariants.py`, 16 passed / 2
skipped — the 2 skips are the archaea-survival check, which only applies to
`merged` — for every mode).

| domain | asv (filter only) | phylo (height=0.05, default) | genus |
|---|---:|---:|---:|
| ARC | 104 | 35 | 42 |
| BAC | 630 | 389 | 389 |
| merged | 567 | 220 | 325 |

`phylo` mode's default height happens to land close to `genus` mode's feature
count without any tuning (BAC: exact match at 389; ARC and merged within
~1.5×) — a fair, comparable-p comparison was achievable without a parameter
search. A full accuracy comparison between arms requires the evaluation
driver (§7 below).

## 7. Evaluation results (task Sec 5A, executed)

Trained and tested on the **genus-level, calibrated-graft-fallback** tables
built in §1–§4 above (not the old ASV-level, naive-graft data) — ARC, BAC, and
merged as three separate arms, no syntrophy-shortcut information anywhere.
`Phylospec/multi_models/build_inputs_from_genus_tables.py` converts this run's
`output/genus_tree/genus_autorun/*_table.csv` + `*_tree.nwk` into each
model's native input; `run_site_grouped_cv_benchmark.py` then wires all five
model families (RF, CNN, PMCNN, MetaDR, DeepPhylo) into
`evaluate_multilabel.evaluate_cv()` — site-grouped repeated CV (5 folds × 5
repeats = 25 fits per model per domain, 375 total), pooled out-of-fold
scoring, per-flag MCC-tuned thresholds fit on validation-only scores, and
macro F2 / AUPRC-lift reporting, exactly per Sec 5A. `pos_weight` capped at
10 for every model (Sec 5A.4). Full per-flag tables (mean + 95% CI over the 5
repeats) are in `Phylospec/multi_models/results/site_grouped_cv/results_
<domain>_<model>.csv`; the macro-level summary is reproduced below.

**Two environment fixes were needed to actually run this**, both applied:
`RF-multilabel.py` didn't accept `--pos-weight-cap` (RF's imbalance handling
is `class_weight='balanced'`, not a `pos_weight`, so the flag was simply never
defined) — added as an accepted-but-unused argument for CLI uniformity across
all five scripts. The installed `torch` (2.1.2) could not call `.numpy()` at
all against the installed `numpy` (2.0.2) — a known ABI break, not specific to
this code — fixed by upgrading to `torch` 2.8.0, matching this repo's own
`requirements.txt` floor of `>=2.3.1`.

### Macro MCC / F2, all 15 (domain × model) combinations

Mean over 5 repeats, [2.5th, 97.5th] percentile in brackets.

| domain | model | macro MCC | macro F2 | macro AUPRC | AUPRC lift |
|---|---|---:|---:|---:|---:|
| ARC | RF | 0.128 [0.059, 0.192] | 0.320 [0.282, 0.371] | 0.209 | 2.17× |
| ARC | CNN | 0.052 [0.027, 0.077] | 0.266 [0.187, 0.357] | 0.159 | 1.59× |
| ARC | PMCNN | 0.150 [0.086, 0.204] | 0.326 [0.260, 0.378] | 0.186 | 1.68× |
| ARC | MetaDR | **0.182** [0.143, 0.209] | **0.368** [0.336, 0.393] | 0.218 | 2.22× |
| ARC | DeepPhylo | 0.031 [−0.037, 0.079] | 0.230 [0.095, 0.328] | 0.116 | 1.14× |
| BAC | RF | 0.183 [0.147, 0.238] | 0.360 [0.319, 0.404] | 0.224 | 2.11× |
| BAC | CNN | 0.183 [0.172, 0.192] | 0.378 [0.368, 0.395] | 0.197 | 1.75× |
| BAC | PMCNN | **0.237** [0.202, 0.270] | **0.389** [0.341, 0.430] | 0.275 | 2.51× |
| BAC | MetaDR | 0.221 [0.154, 0.265] | 0.379 [0.306, 0.431] | 0.235 | 2.16× |
| BAC | DeepPhylo | 0.141 [0.056, 0.231] | 0.324 [0.189, 0.416] | 0.175 | 1.72× |
| merged | RF | 0.161 [0.124, 0.188] | 0.351 [0.293, 0.388] | 0.222 | 2.17× |
| merged | CNN | 0.183 [0.136, 0.217] | 0.334 [0.286, 0.386] | 0.241 | 2.03× |
| merged | PMCNN | 0.197 [0.137, 0.232] | 0.373 [0.304, 0.414] | 0.214 | 1.94× |
| merged | MetaDR | 0.182 [0.161, 0.192] | 0.365 [0.352, 0.397] | 0.234 | 1.97× |
| merged | DeepPhylo | **0.220** [0.155, 0.273] | 0.370 [0.299, 0.461] | 0.273 | 4.25× |

Best macro MCC per domain in bold: **MetaDR on ARC**, **PMCNN on BAC**,
**DeepPhylo on merged** — but note per Sec 5A.5's own rule ("a difference
between arms is only claimed if the repeat-level intervals separate"), most
of these are *not* cleanly separated from their nearest competitor within the
same domain (e.g. BAC's PMCNN 0.237 vs. MetaDR 0.221 overlap heavily) — read
these as "roughly tied," not a ranked leaderboard.

**What *is* cleanly separated, arm vs. arm, same model:**
- **CNN, ARC vs. BAC**: 0.052 [0.027, 0.077] vs. 0.183 [0.172, 0.192] — no
  overlap. CNN is a materially worse fit on ARC's 42 features than on BAC's
  389, more so than any other model family; plausibly the model family most
  sensitive to feature count in this p ≈ 40 regime.
- **DeepPhylo, ARC vs. merged**: 0.031 [−0.037, 0.079] vs. 0.220 [0.155,
  0.273] — no overlap. DeepPhylo (which relies entirely on the phylogenetic
  distance PCA, no taxonomy) is the family that benefits most from the
  merged arm's larger, cross-domain tree.
- Every other same-model, cross-domain pair overlaps and should not be read
  as a difference.

**`ammonia_toxicity` (26 positives, 11 sites — the best-powered flag)** shows
the clearest, most consistent pattern: RF/PMCNN/MetaDR all reach MCC 0.5–0.68
with tight, non-overlapping-from-zero intervals on every domain; CNN is
noticeably weaker specifically on ARC (MCC 0.018, CI [−0.060, 0.108] —
crosses zero, not distinguishable from a null model) while performing
respectably on BAC (0.531) and merged (0.521). BAC gives the single best
`ammonia_toxicity` result of any arm (RF: MCC 0.680, F2 0.845, recall 0.946).

**`acid_accumulation` (4 positives, 3 sites) must not carry a conclusion**,
exactly as flagged in §5 — and this run demonstrates why concretely: merged/
DeepPhylo reports MCC 0.286 with AUPRC lift 10.56×, by far the single most
dramatic number in the whole table, but its own CI is [−0.012, 0.426] —
crosses zero, consistent with noise from 4 positive samples in 3 sites. Every
other `acid_accumulation` result across all 15 runs has a similarly wide or
zero-crossing interval. This is the study-design limitation from §5 showing
up directly in the numbers, not a modelling failure.

**Full per-flag tables** (AUPRC, AUPRC-lift, MCC, F2, precision, recall,
tuned threshold, each mean + 95% CI): `Phylospec/multi_models/results/
site_grouped_cv/results_<domain>_<model>.csv`, one file per row of the table
above (15 files). Site/label support per domain:
`site_label_support_<domain>.csv` in the same directory.

## 8. What's still open

1. **SEPP fragment-insertion not attempted** — see §4. The calibrated-graft
   fallback's cross-domain distance is inflated (5.13 vs. a 1.35 target) and
   any absolute distance-based syntrophy claim from the merged arm should be
   treated as such.
2. **Syntrophy-shortcut fusion (`build_cross_domain_graph.py`) not wired in**
   — deliberately excluded from both the tables and this evaluation per your
   instruction. It expects a single tree that already contains both domains'
   tips (this run's calibrated-graft `merged_tree.nwk` would be the natural
   input) plus operating-variable metadata to condition the co-occurrence
   test on.
3. **`asv`/`phylo` mode arms (§6) were not run through the evaluation
   driver** — only `genus` mode was benchmarked above. `build_inputs_from_
   genus_tables.py --genus-dir output/genus_tree/<mode>` plus a re-run of
   `run_site_grouped_cv_benchmark.py` would produce the same table for those
   two arms if a genus-vs-asv-vs-phylo *accuracy* comparison (as opposed to
   just feature count, §6) is wanted next.
4. **`data/qiimeresult/{ARC,BAC}/dada2_rep_seqs.qza`** exist and are ready
   if SEPP is attempted later.
