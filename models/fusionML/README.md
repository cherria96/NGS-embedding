# fusionML

Domain-constrained log-contrast fusion regression for anaerobic-digestion
(AD) microbiome data. Implements
`docs/superpowers/specs/ad-microbial-model-design.md` (also copied to
`models/fusionML/ad-microbial-model-design.md` with a concrete project config) —
items 9-18 are the implementation contract, items 1-8 the scientific
framing. Read that document before changing anything here; this README
only covers how to run and extend the code.

The core model: for each domain (ARC = archaea, BAC = bacteria, amplified
with different primers and closed separately), a log-contrast linear
regression `y = b0 + lA'betaA + lB'betaB` with a per-domain zero-sum
constraint (`sum(betaA) = sum(betaB) = 0`), fit by a convex QP with
sparsity (L1), ridge (L2) and graph-fusion penalties. The fusion graph
mixes a phylogenetic-distance component and a training-only abundance
(log-ratio-variance) component per domain, with an independently
learned mixture weight `alpha_ARC`/`alpha_BAC`. Everything is selected by
nested, site-grouped cross-validation — no held-out sample ever touches
feature selection, graph construction, or hyperparameter choice.

## Install

```bash
conda activate torch   # this repo's verified env (pandas, sklearn, biopython, torch, iterstrat)
pip install -r models/fusionML/requirements.txt   # installs cvxpy (+ CLARABEL/OSQP), openpyxl on top of that env
```

Installed/verified versions in this repo's `torch` env at the time of
writing: cvxpy 1.9.2 (solvers CLARABEL, SCS, SCIPY, HIGHS, OSQP), numpy,
pandas, scipy, scikit-learn (all already present), biopython 1.88,
matplotlib 3.9.2, openpyxl (added). Run `python -m pytest models/fusionML/tests/`
after installing to confirm your environment matches what was tested.

Base/system `python3` in this repo is broken (numpy2/numexpr ABI conflict)
— always use `conda activate torch`.

## Classification (`task: classification`)

Item 7's explicit extension: "The same feature construction can use
logistic loss for an operational-state classification task." Set
`task: classification` and point `columns.target` at a `{0,1}` column.
Mechanically: `estimator.fit_log_contrast_logistic` replaces squared error
with weighted negative log-likelihood (`cp.logistic(eta) - y*eta`, same
per-domain zero-sum constraints, same L1/ridge/fusion penalties); no target
standardization (item 10's y-scaling is a regression-only concern); the
decision threshold is selected by maximizing MCC on **that fit's own
training data only** (`evaluate.select_threshold_on_training`, applied
per outer/inner fold — never on validation/test rows); inner-CV candidate
selection uses site-balanced log-loss in place of site-balanced MSE (same
one-standard-error rule); reporting adds pooled AUC-ROC/AUC-PR/Brier,
site-balanced log-loss, and precision/recall/F1/F2/MCC/accuracy at the
pooled, per-sample-fold-specific threshold. See
`configs/real_smoke_classification.yaml`.

This repo has 5 real, documented binary targets for this: the "warning
flag" columns computed by `code/metadata_targetwriter.py` from `eff_*`
Chem_Se.Avg chemistry triggers (see the root `CLAUDE.md`'s table). **These
flags, and every other real-data config in this repo, are now computed
directly from `datasets/raw/final/Dat_chem.xlsx`'s `Chem_Se.Avg` sheet via
`models/fusionML/chemistry_targets.py` — `datasets/raw/final/metadata.csv` is not read
anywhere in fusionML.** That file went stale: a live comparison this
session found its `eff_CH4` differs from the current workbook by up to
19.4 percentage points on some records (`eff_pH`/`eff_TAN` differ only at
rounding level), and `datasets/raw/final/Dat_chem.xlsx` shows as locally modified
in `git status`. The 5 flag formulas themselves are untouched — copied
verbatim from `code/metadata_targetwriter.py` — only their inputs are now
read live. `chemistry_targets.py` reconstructs `sample_id` as
`"{season_code}-{No.}"` straight from `Chem_Se.Avg`'s own columns (no
crosswalk file needed) and excludes the one effluent record with a
genuinely blank "No." cell (`MGYe2`) rather than guessing its identity.

Real-data classification smoke run (genus resolution, reduced
folds/candidates, same 0.40 smoke-only prevalence override as
`real_smoke.yaml`), pooled OOF AUC-ROC / MCC / F2, best family per flag:

| Flag | positives (of 134 scorable) | best pooled AUC-ROC | MCC | F2 | Verdict |
| --- | --- | --- | --- | --- | --- |
| `warning_ammonia_toxicity` | 24 | 0.921 (abundance_only) / 0.91-0.92 (others) | up to 0.58 | up to 0.66 | Real, well-powered signal — clearly above the intercept baseline's 0.46 AUC. Worth a full (non-smoke) research run. |
| `warning_acid_base_balance` | 20 | 0.72 | up to 0.25 | up to 0.35 | Real but modest signal above chance. |
| `warning_acid_accumulation` | 4 | up to 0.96 | ~0.49 | ~0.29 | **Not a reliable result** — 4 positives is too few for a trustworthy AUC estimate at any fold count; treat this number as noise, not a finding. |
| `warning_biogas_quality` | 6 | 0.27-0.74 (unstable across families) | undefined (constant predictions) | 0 | No usable signal detected at this sample size. |
| `warning_buffer_capacity` | 5 | 0.155 (identical across every family, including intercept) | ~0 or slightly negative | 0 | No signal; sparse positives collapsed every family to intercept-like behavior. |

All 5 are genuinely real, site-grouped-CV, non-placeholder classification
results — but folds are **not stratified by outcome** (item 14: "balance
sample counts without inspecting target values"), so the rarer flags
inevitably produce folds with too few or zero positives to estimate
discrimination reliably; this is a real power limitation of the dataset,
not a pipeline defect. Full per-flag artifacts are under
`results/classification result/fusionML/real_results_classification*/smoke/`. Coefficients for
`warning_ammonia_toxicity` (the best-powered flag) implicate several ARC
methanogen genera with nonzero, non-negligible weights
(`coefficients_and_groups.csv`) — biologically plausible given methanogen
sensitivity to free ammonia, but this is an association from one
smoke-scale run, not a validated biological claim.

### All 7 item-14 families, for the two best-powered flags

`configs/real_smoke_classification_phylo.yaml` adds `inputs.rank_trees.genus`
(the same monophyly-consistent genus tree + ASV→block crosswalk validated
in `real_smoke_family_rank.yaml`) and widens `search.alpha_arc`/`alpha_bac`
to `[0, 0.5, 1]`, so `phylogeny_only`/`shared_hybrid`/`domain_adaptive_hybrid`
(the primary model) become available — completing all 7 required-comparison
rows (ASV ridge/elastic net still not run at real scale; they need
`resolution: asv`, which is implemented but not yet exercised here):

| Family | `warning_ammonia_toxicity` (24 pos.) AUC-ROC / MCC / F2 | `warning_acid_base_balance` (20 pos.) AUC-ROC / MCC / F2 |
| --- | --- | --- |
| Intercept only | 0.463 / n/a / 0 | 0.444 / n/a / 0 |
| Genus ridge | 0.909 / 0.543 / 0.675 | 0.698 / 0.131 / 0.213 |
| Genus elastic net | 0.903 / 0.493 / 0.610 | 0.680 / 0.183 / 0.263 |
| Phylogeny only | 0.856 / 0.528 / 0.669 | 0.639 / 0.187 / 0.337 |
| Abundance only | 0.901 / 0.450 / 0.595 | 0.664 / 0.142 / 0.255 |
| Shared hybrid | 0.870 / 0.496 / 0.635 | 0.629 / 0.132 / 0.289 |
| **Domain-adaptive hybrid (primary)** | 0.887 / 0.469 / 0.625 | **0.663 / 0.254 / 0.388** |

This table predates the interaction extension and still uses the
now-superseded in-sample-MCC threshold rule (item 14's inner-OOF-F2 rule
above applies to every family from this session onward, so a from-scratch
rerun would shift every row's precision/recall-derived numbers somewhat,
not just add a row). A fresh `warning_ammonia_toxicity` run with the new
threshold rule *and* the new 8th family
(`configs/real_smoke_classification_phylo.yaml`, reduced interaction grid,
`real_results_classification_phylo_interaction/`) gave: intercept only
0.463/n/a/0, genus ridge 0.909/0.616/0.816, genus elastic net
0.903/0.595/0.758, phylogeny only 0.856/0.596/0.791, abundance only
0.901/0.595/0.758, shared hybrid 0.870/0.547/0.777, domain-adaptive hybrid
0.887/0.554/0.769, **domain-adaptive interaction 0.897/0.523/0.725**
(AUC-ROC/MCC/F2). The final deployable model selected the interaction
family but its one-standard-error selection picked an **interaction-OFF**
candidate (4 candidate products existed and were tried, but the simpler
off-candidate won) — i.e. on this one flag, at this seed/fold count, the
interaction terms did not earn their added complexity; the tree-free
`genus_ridge` baseline remains the strongest single family by MCC/F2. This
is one seed, one flag, a reduced interaction-penalty grid, and no repeats —
not evidence that the interaction mechanism doesn't help in general, only
that it didn't win a model-selection contest run this small.

Two honest observations, not conclusions (still smoke-scale, one seed, no
stability check across repeats): for `warning_ammonia_toxicity`, the
tree-free `genus_ridge` baseline is at least as good as every fusion
variant, including the primary domain-adaptive hybrid — phylogeny doesn't
add discrimination beyond a plain regularized log-contrast here. For
`warning_acid_base_balance`, the primary domain-adaptive hybrid has the
best MCC/F2 of the 7 (though not the best AUC), i.e. the picture is
flag-dependent, exactly the kind of thing a real (non-smoke) run with
proper repeats would need to characterize before drawing any conclusion
about whether adaptive fusion earns its complexity.

## Cross-domain predictive interaction (`domain_adaptive_interaction`)

`ad-microbial-model-design-arc-bac-interactions.md` (items 8, 10-18) adds an
eighth family that extends `domain_adaptive_hybrid` (now "the required
additive reference") with ARC×BAC predictive interaction terms, fit
*jointly* with the additive model rather than as a post-hoc add-on:

1. **Tree-split balances** (`interactions.enumerate_tree_splits`): every
   tree-edge bipartition of a domain's retained+mapped taxa is a candidate
   zero-sum "balance" (generalizing the exploratory 13/8 block from item
   9's earlier smoke work to every split the tree actually contains).
   Deterministically deduplicated and oriented (lexicographically-smaller
   side is "positive").
2. **Selection** (`interactions.select_balances`): candidates must clear a
   *group*-detection gate (≥1 taxon per side jointly detected in
   ≥`min_joint_group_samples` training samples spanning
   ≥`min_joint_group_sites` sites — distinct from item 11's pairwise
   taxon-graph gate), then are greedily selected by decreasing equal-site
   weighted variance, capped at `max_balances_arc`/`max_balances_bac`,
   rejecting a candidate that doesn't increase design-matrix rank or that
   correlates above `max_abs_balance_correlation` with an already-selected
   balance.
3. **Products**: selected balances are standardized, crossed pairwise
   (ARC × BAC) into up to `max_products` standardized product features,
   again rejecting rank-deficient additions.
4. **Joint fit** (`estimator.fit_log_contrast_with_interaction`):
   `eta = b0 + L_A@beta_A + L_B@beta_B + X_I@gamma`, with an *unconstrained*
   `gamma` (no zero-sum constraint — products aren't taxa) penalized by
   `lambda_i1*||gamma||_1 + lambda_i2*||gamma||_2^2`. No cross-domain
   fusion-graph edges are created; interactions are a wholly separate
   predictor mechanism from `lambda_f`'s graph fusion.
5. **On/off candidates**: for every base beta/alpha point in the *identical*
   deterministic grid `domain_adaptive_hybrid` uses (same seed derivation —
   `search.INTERACTION_BASE_FAMILY_ALIAS`), the family evaluates one
   interaction-OFF candidate (literally refit via the plain additive
   estimator, not just a large `lambda_i2`) plus every
   `lambda_i_l1 × lambda_i_l2` on-setting. An on-candidate with zero
   surviving products in a given training partition raises
   `InteractionUnavailableError` and is dropped from that partition's
   search (the off-candidate remains eligible) rather than fabricating an
   empty-but-successful fit.
6. **Model selection**: one-standard-error selection now takes a
   family-specific complexity tie-break —
   `(n_active_products, n_required_raw_taxa, n_active_groups)` for this
   family vs. `(n_active_groups, n_active_taxa)` for every additive family
   (`search.one_standard_error_select`).
7. **Classification threshold rule (supersedes the old in-sample MCC rule
   for every family, not just this one)**: per `classification.*`, the
   threshold is chosen from `threshold_grid` using the *selected*
   candidate's pooled **inner**-CV OOF probabilities only (never outer/test
   data, never in-sample training predictions) with equal-site-weighted
   confusion counts, F2 criterion, ties toward the larger threshold, frozen
   before the one-time outer-fold prediction
   (`evaluate.select_threshold_via_inner_oof_f2`). The final deployable
   model freezes its threshold from its own full-data inner-CV OOF the same
   way.
8. **Single-class training gate**: only `intercept_only` may report a
   clipped constant-probability baseline when a training partition has a
   single class; every other family (including this one) raises
   `preprocessing.SingleClassTrainingError`, caught upstream and reported
   as an unavailable partition, never a fabricated fit.

**Config**: set `optional.cross_domain_interactions: true` *and*
`interaction.enabled: true` together (config validation rejects them
disagreeing) and add `domain_adaptive_interaction` to `cv.families`. See
`configs/real_smoke_classification_phylo.yaml` for a real-data example
(reduced to one `lambda_i_l1`/`lambda_i_l2` pair, explicitly labeled SMOKE
per item 16's permission to shrink the interaction grid for smoke runs) and
`models/fusionML/tests/test_interactions.py` for the full defaults.

**Artifacts**: `balance_definitions.csv` (domain, canonical ID, memberships,
loading weights, mean/sd/variance) and `interaction_coefficients.csv`
(parent balance IDs, gamma in both fitted and original-target units,
activity flag) are written twice — once concatenated across outer folds as
`*_outer_folds.csv`, once for the final deployable model at the plain
names, matching `final_model.pkl`. The saved `FusionModel` carries the
frozen `InteractionTransform` + `gamma_orig` and applies the interaction
term in `predict()`.

**Validation**: `models/fusionML/tests/test_interactions.py` covers item 18's
interaction-specific acceptance tests — invariance of balances/products/eta
to any per-sample rescaling of a domain's positive composition (loading
vectors are zero-sum by construction), on/off equivalence for both squared
and logistic loss, the single-class gate, full prediction algebra against a
manual eta reconstruction, and a synthetic nested end-to-end run including
the zero-product data-generating case (an impossible detection gate leaves
every on-candidate unavailable while off-candidates still fit). Not yet
implemented from the design doc: the parameter-matched null arm and
per-fold edge discovery are a *different* extension
(`CLAUDE_CODE_TASK_cross_domain_injection.md`'s cross-domain-injection
benchmark, not this one) and don't apply here.

## CLI

One entry point, four modes:

```bash
python models/fusionML/run_model.py --config <config.yaml> --mode validate
python models/fusionML/run_model.py --config <config.yaml> --mode diagnose
python models/fusionML/run_model.py --config <config.yaml> --mode train
python models/fusionML/run_model.py --config <config.yaml> --mode predict --model-path <final_model.pkl> --output <predictions.csv>
```

- **validate**: inspects every configured input (existence, schema,
  duplicate IDs, orientation, taxonomy columns, chemistry workbook
  headers) and reports `ready_to_run` plus a list of blockers — it never
  fits anything.
- **diagnose**: exploratory, descriptive-only redundancy diagnostics
  (pairwise log-ratio variance, site decomposition, pseudocount
  sensitivity) on the full joint cohort. Never read by `train`.
- **train**: the full nested site-grouped CV pipeline — inner
  hyperparameter search, outer refit + prediction, all required model
  families, artifacts, figures, and a final deployable model refit on all
  eligible data.
- **predict**: loads a saved model and applies its frozen preprocessing —
  never refits anything, regardless of what the config's other fields say.

## Two config schemas

There are two YAML shapes in `configs/fusionML/` and this repo root,
because the spec's config evolved mid-project:

1. **item-16 schema** (`configs/synthetic_demo.yaml`,
   `configs/real_smoke.yaml`, `configs/real_smoke_family_rank.yaml`,
   `configs/real_research_template.yaml`) — matches the YAML template in
   design-spec item 16 almost exactly (a few explicitly-documented
   extension keys, see `config.py`'s "EXTENSION" comments). Relative input
   paths resolve against **the config file's own directory**.
2. **project schema** (`ad-microbial-config.yaml`, `models/fusionML/ad-microbial-model-design.md`'s
   embedded example) — a later, more concrete config carrying `project_root`
   (paths resolve against it, or the invocation cwd if it's relative),
   `chemistry:` (a full Dat_chem.xlsx wide-layout reader spec), a separate
   `target:`/`sample_metadata:` block instead of `columns.target`, and
   `inputs.rank_trees.{genus,family,order,class}` for tree-consistent rank
   aggregation. `Config.is_project_schema` (true iff `project_root` is a
   top-level key) is what the code branches on; `pipeline.run_validate`
   dispatches to `run_validate_project()` for this schema.

`--mode train`/`diagnose` refuse to run against a project-schema config
until its target and sample crosswalk are resolved (see below) — this is
intentional, not a missing feature: **that resolution is a scientific
decision for a human, not something this pipeline may guess** (item 9).

## What's runnable today vs. blocked

| Config | Runs? | Notes |
| --- | --- | --- |
| `configs/synthetic_demo.yaml` | Yes, fully | Synthetic data (`generate_synthetic_demo_data.py`), all 9 model families, ~53s. |
| `configs/real_smoke.yaml` | Yes, fully | Real ARC/BAC/taxonomy/tree; sample_id/site_id/target built live from `Dat_chem.xlsx` (`chemistry_targets.py`, no `metadata.csv`), genus resolution, `effluent.CH4` as an explicitly-labeled **placeholder** target. ~110s. |
| `configs/real_smoke_family_rank.yaml` | Yes, fully | Same real data/chemistry source, `resolution: family` via the tree-consistent rank-mapping mechanism (phylogeny-enabled families included). ~90s. |
| `configs/real_smoke_classification.yaml` | Yes, fully | `task: classification`, real `warning_ammonia_toxicity` target computed live from `Dat_chem.xlsx` (a genuine, documented target, not a placeholder) — see "Classification" section above. ~2 min per flag. |
| `configs/real_smoke_classification_phylo.yaml` | Yes, fully | Same, plus `inputs.rank_trees.genus` and a widened alpha search, so all 7 item-14 families run (phylogeny-enabled ones included). ~2 min per flag. |
| `configs/real_research_template.yaml` | `validate` only | item-16 default settings (6/4 folds, 500 bootstrap, 0.10 prevalence); `columns.target` is deliberately `null` — no approved continuous target exists in this dataset yet (see below). |
| `ad-microbial-config.yaml` | `validate` only | A separate, more elaborate project-schema config (see below); `sample_metadata.crosswalk_path` and `target.source_column` are `null` by the document's own admission; `train`/`diagnose` raise a clear `NotImplementedError` rather than guessing. |

**`datasets/raw/final/metadata.csv` is not read anywhere in fusionML** (explicit
instruction). It went stale relative to the current `datasets/raw/final/Dat_chem.xlsx`
(see the "Classification" section above for the live diff), so every
real-data config builds sample_id/site_id/target directly from
`Chem_Se.Avg` via `chemistry_targets.py` instead.

### Why there's no literal "methane yield" target

Neither `datasets/raw/final/metadata.csv` nor `Dat_chem.xlsx`'s `Chem_Se.Avg` has a
methane-yield column. The closest continuous, non-sCOD/non-ALK candidates,
read live via the `chemistry:` block, are `effluent.CH4` (% CH4 in biogas —
composition, not a rate/yield), `effluent.CO2`, `effluent.H2S`,
`effluent.O2`. Per item 9, picking one (or deriving a real yield from flow
× composition, if that data exists) is a documented, user-approved
decision this pipeline must not make on its own. `configs/real_smoke*.yaml`
uses `effluent.CH4` **only** to exercise the pipeline end to end on real
inputs — its results are not a scientific baseline.
`configs/real_research_template.yaml` leaves `columns.target: null` for
exactly this reason (it does, however, document the 5 warning flags as
real, approved classification targets that ARE ready to use).

### What `ad-microbial-config.yaml` is, and why it's still blocked

This is a separate, more elaborate config (from a later spec update) that
also reads `Dat_chem.xlsx` directly via the same `chemistry.py` machinery,
plus `inputs.rank_trees` for all 4 ranks, `inputs.orientation: auto_validate`,
and a `project_root`-relative path convention (`Config.is_project_schema`).
Run `--mode validate` against it and every configured chemistry cell
coordinate, season header, and join key is checked live against the
current workbook — as of this session, **all 200 header checks pass** and
the join is clean (148 rows, one-to-one, `MGYe1` correctly retained
unmerged) despite the workbook's sha256 no longer matching
`chemistry.inspected_sha256` (`datasets/raw/final/Dat_chem.xlsx` is `git status`-modified
in this working tree; the two sheets fusionML reads are structurally
unaffected, but re-confirm before trusting this over a longer time
horizon). What's still genuinely unresolved, exactly as the config itself
declares: which chemistry column is the target (the `configs/real_smoke*.yaml`
family answers this question for the item-16 schema, using
`chemistry_targets.py`; that mechanism has not been back-ported into this
config's own `target:`/`sample_metadata:` blocks, which are a different,
more general assembly path not yet wired to `run_train`). `--mode validate`
reports this as an explicit blocker; nothing downstream guesses around it.

## Package layout

```
config.py            YAML schema (both flavors), validation, path resolution
io_counts.py          count-table loading: TSV/CSV, "#OTU ID" header handling, orientation auto-detect
io_metadata.py         sample_id/site_id/target/covariate loading (item-16 schema)
io_taxonomy.py          SILVA lineage parsing, lineage-aware genus grouping
io_tree.py               Newick loading, O(depth)-per-pair patristic distance
season.py                 S/F/W/Sp <-> 1-4 exact-token season mapping
chemistry.py              Dat_chem.xlsx wide-layout reader (season/header validation, join)
chemistry_targets.py       warning-flag formulas + sample_id/site_id/target assembly straight
                            from Chem_Se.Avg -- the metadata.csv-free path every real config uses
genus.py                   naive lineage-string aggregation (genus_ridge/EN baseline families)
rank_mapping.py             tree-consistent rank aggregation via a supplied ASV->block crosswalk
cohort.py                    joint ARC+BAC+metadata cohort assembly and exclusion reporting
preprocessing.py               per-partition filtering, zero handling, log composition, target/covariate scaling
graphs.py                       phylogenetic + abundance graph construction, bootstrap stability, mixture
estimator.py                     CVXPY constrained log-contrast solver, retry policy
groups.py                         group extraction algebra, optional b12 balance comparator
folds.py                           site-grouped outer/inner fold construction
search.py                           model-family definitions, candidate hyperparameter generation, 1-SE selection
evaluate.py                          pooled/site-balanced metrics, bootstrap paired comparisons
diagnostics.py                       exploratory redundancy diagnostics (--mode diagnose)
model_io.py                           save/load a trained model bundle; predict()
artifacts.py, figures.py               run-manifest/CSV/JSON writers, matplotlib figures
synthetic.py                           end-to-end synthetic data generator (item 18.11)
pipeline.py                            orchestration for all four CLI modes
cli.py, run_model.py                    argparse entry point
tests/                                  item 18's 11 verification checks (see below)
configs/                                example YAML configs (see table above)
```

## Real-data mechanics worth knowing

- **Sample-ID mismatch**: `datasets/raw/qiimeresult/{ARC,BAC}/table_filtered.tsv`
  columns are `{round}-{No}[-{site_code}]` (e.g. `1-1-BSSG`); this repo's
  canonical sample ID is `{round}-{No}`, which is exactly
  `"{season_code}-{No.}"` from `Dat_chem.xlsx`'s own columns
  (S/F/W/Sp = round 1/2/3/4) -- `chemistry_targets.py` reconstructs it
  directly, with no separate crosswalk file. `models/fusionML/prepare_real_data_smoke_inputs.py`
  separately writes a round-No-normalized **copy** of the ASV tables to
  `datasets/fusionML/real_data_cache/` (never edits the checked-in QIIME tables),
  using the same transform `prepare_warning_targets_data.py` uses
  elsewhere in this repo. Run it once before `configs/real_smoke*.yaml`.
- **ARC/BAC coverage differs**: several chemistry records have no ARC
  assay at all (4 are the documented poor-archaeal-sequencing exclusions
  in the root `CLAUDE.md`; others never had an ARC run, or fall in a
  season a site wasn't sampled). `FusionModel.predict`
  matches rows by `sample_id`, not position, and reports a sample present
  in only one domain's table as a missing assay, not a zero-count row —
  this was a real bug caught by testing on the actual data (a synthetic
  fixture with always-matched domains never exercised it); see
  `tests/test_prediction_replay.py::test_sample_missing_from_one_domain_is_reported_not_misaligned`.
- **`#OTU ID` header**: QIIME's biom-tsv export has a true one-line comment
  (`# Constructed from biom file`, no delimiter) followed by the real
  header (`#OTU ID\t<sample1>\t...`, which also starts with `#`). An
  earlier version of `io_counts.py` skipped both, silently misreading the
  first ASV row as column headers. Fixed by only skipping "#"-prefixed
  lines that contain **no delimiter** (see `io_counts.py`'s comment for
  the exact rule); regression test in `test_source_assembly.py`.
- **Rank resolution, two different mechanisms**: `resolution: genus` with
  no `inputs.rank_trees.genus` configured uses `genus.py`'s naive
  lineage-string aggregation (fine for the tree-free `genus_ridge`/
  `genus_elastic_net` baseline families, item 14's own definition). Any
  resolution with a configured `inputs.rank_trees.{rank}` uses
  `rank_mapping.py` instead, consuming this repo's own
  `datasets/processed/genus_tree/taxlevel_autorun/{rank}/{arc,bac}_mapping.csv` —
  an already-built, monophyly-aware ASV->block crosswalk whose blocks
  (`F00000`, `F00001`, ...) are exactly that rank tree's tip labels. This
  is what lets `phylogeny_only`/`shared_hybrid`/`domain_adaptive_hybrid`
  run at genus/family/order/class resolution without assigning a genus an
  arbitrary representative ASV (item 9/12 explicitly forbid that
  shortcut). Verified one-to-one tree-tip coverage on the real data for
  all four ranks (`configs/real_smoke_family_rank.yaml`'s run log).
- **Orientation auto-detection** (`inputs.orientation: auto_validate`,
  project schema only): detects features-by-samples vs. samples-by-features
  by ID overlap against taxonomy+tree reference IDs, and raises rather
  than guessing if both or neither axis overlaps. Verified against real
  ARC/BAC tables (100% correct on the actual files).

## Runtime drivers

- **Abundance-graph bootstrap** (`graph.bootstrap_sites`, default 500) is
  the dominant cost: each replicate redoes the full mutual-nearest-neighbor
  search over all *(p choose 2)* pairs among retained taxa. At the item-9
  default 0.10 prevalence threshold, real BAC retains ~367 genus features
  (~67k pairs/replicate); `configs/real_smoke.yaml` raises the threshold to
  0.40 (~116 features, ~6.7k pairs) specifically to keep the smoke run to
  ~2 minutes — **the research config must not inherit that override**.
  `graph.approximate_bootstrap` (candidate pool of top-`k` per taxon) is
  implemented for large ASV sets but not exercised at real ASV scale in
  this session (out of time budget, not a correctness gap).
- **CVXPY solves** themselves are fast (well under a second each, even at
  a few hundred features per domain) — the search-loop fit count
  (`outer_folds * (inner_folds * n_candidates + 1)`, printed before every
  `train` run) is what multiplies runtime, not the solver.
- A full item-16-default research run (6 outer x 4 inner folds, 48
  candidates/family, 500 bootstrap reps, 9 families) was not executed in
  this session — bounded by the missing approved target as much as by
  runtime. `configs/real_smoke_family_rank.yaml` demonstrates every model
  family and graph mechanism at reduced scale/settings instead.

## Optional / explicitly unimplemented extensions

- **b12 balance comparator** (`optional.b12_comparator`, item 15):
  implemented (`groups.compute_b12_feature`/`expand_b12_coefficient`) but
  gated off by default and requires `inputs.exploratory_block_list` (a
  CSV with explicit `group,feature_id` rows) — never guessed. Not wired
  into the default `train` family list (the primary model must not carry
  b12 alongside its source log-abundance columns, per the task
  instruction and item 15).
- **Cross-domain interaction terms** (`optional.cross_domain_interactions`
  + `interaction.enabled`, items 8/10-18): implemented as the
  `domain_adaptive_interaction` family — see
  "Cross-domain predictive interaction" below. `interaction.balance_source:
  external_balances` (item 10.6) is the one still-unimplemented sub-piece;
  config validation rejects it explicitly rather than silently falling
  back to `tree_splits`.
- **PhILR / trac baselines** (item 7): not implemented; reported
  unavailable rather than fabricated.
- **Domain-only baseline** (item 10, for a domain with <2 retained taxa):
  `estimator.fit_domain_only_baseline` exists and is directly callable but
  not wired into the default family list (must be requested explicitly).
- **Standardized microbial columns** (`preprocessing.standardize_microbial_columns`):
  raises rather than silently changing what "equal coefficients" means in
  the fusion penalty (item 5 requires carrying inverse scaling into the
  constraints and penalty if this is ever done — not implemented).

## Tests

```bash
conda activate torch
python -m pytest models/fusionML/tests/ -v
```

57 tests, all passing as of this session: the original 47 covering all 11
items of the additive design spec's item 18 verification list (item
18.10's later addition, "source assembly," is in `test_source_assembly.py`;
the rest are one file per item as named), plus 10 more in
`test_interactions.py` covering the ARC-BAC interaction extension's items
12 (invariance), 14 (on/off equivalence), 15 (prediction algebra), 17
(classification gates) and 18 (synthetic interaction demonstration,
including the zero-product case). `test_end_to_end.py` runs the actual CLI
`train` entry point against a small synthetic nested grouped configuration
in a tmp directory (~2s) — this tests implementation correctness, not the
scientific hypothesis (item 18.11 / originally 18.10).

## Regenerating the synthetic demo

```bash
conda activate torch
python models/fusionML/generate_synthetic_demo_data.py     # writes models/fusionML/demo_data/
python models/fusionML/run_model.py --config configs/fusionML/synthetic_demo.yaml --mode train
```

Every artifact this produces is written under `models/fusionML/demo_results/smoke/`
and labeled synthetic in its figures/report.

## Deviations from the spec, summarized

All of these are called out inline (`# EXTENSION:` comments in `config.py`,
docstrings elsewhere) — collected here for convenience:

1. `preprocessing.relative_input_mode` and `preprocessing.missing_retained_feature_means_zero`
   are added config keys (both default `false`) implementing the
   "explicit alternative preprocessing mode" item 9 asks for when it
   forbids silently guessing.
2. `cv.families` is an added optional whitelist (default: every family
   applicable to the configured resolution/inputs) used only to bound
   smoke-config runtime.
3. `solver.max_retries`/`solver.time_limit_seconds` are added
   resource-limit keys per item 16's explicit permission to add these.
4. Genus-resolution phylogenetic fusion falls back to naive lineage
   aggregation only for the `genus_ridge`/`genus_elastic_net` baselines
   (which item 14 defines as tree-free); every other family requires a
   real tree (ASV `rooted-tree.nwk` or a configured `rank_trees` entry) —
   never an arbitrary representative-ASV shortcut.
5. The constrained solver has never been substituted with an unconstrained
   fit anywhere in this codebase, including on failure (`estimator.SolverFailure`
   propagates; candidates that fail are excluded from the inner-CV search
   and logged, not silently retried with a different objective).
6. `solver.classification_fallback` (default `SCS`) is an added key: OSQP
   cannot solve `cp.logistic()`'s exponential-cone formulation at all, so
   classification needs its own fallback distinct from `solver.fallback`
   (regression-only). This was a latent, never-triggered bug (CLARABEL, the
   preferred solver, always succeeded first in every real run so far) found
   while implementing the interaction extension's classification path.
7. `interaction.group_detection_count` must equal `preprocessing.detection_count`
   (validated, not silently ignored) — item 10.2's group-detection gate
   uses "the raw threshold," i.e. the same detection table already built
   for graph construction; there is no second, independently-thresholded
   detection table.
