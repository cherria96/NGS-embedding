# NGS-embedding — project map for Claude Code

Anaerobic digestion (AD) microbiome study: 16S rRNA amplicon data from ~34-36
full-scale digesters × up to 4 seasons, split into **two parallel domains**
(archaea `ARC`, bacteria `BAC`) because they were amplified with different
primer pairs. Goal: predict 5 binary digester-stability "warning flags" from
microbiome composition, comparing classical-ML baselines against
phylogeny-aware deep models. See `README.md` for the full data dictionary
(metadata columns, chemistry sheets, sample-ID convention `{round}-{No}`).

**Read `Phylospec/CLAUDE_CODE_TASK_genus_tree.md` before touching the tree/
merge pipeline.** It documents verified bugs in the Phylo-Spec source
(substring tip-matching, unclassified-column bypass, branch-length sign
flips) and the required evaluation protocol. Its architecture is current;
only its §5.5 SEPP recommendation has since been *carried out* (see below —
don't re-read that section as "still TODO").

## The 5 classification targets

Appended to `data/final/metadata.csv` by `code/metadata_targetwriter.py`.
Multi-label (not mutually exclusive), computed from `eff_*` effluent
chemistry columns:

| flag | trigger | positives (of 140 original) |
|---|---|---|
| `warning_acid_base_balance` | `eff_pH < 6.5` or `> 8.0` | 21 |
| `warning_buffer_capacity` | `eff_TVFAs / eff_ALK >= 0.30` | 9 |
| `warning_acid_accumulation` | HPro/HAc ratio or absolute HPro threshold | 4 |
| `warning_ammonia_toxicity` | `eff_TAN > 2.5` | 26 |
| `warning_biogas_quality` | `eff_CH4 < 50.0` | 7 |

All evaluation must be **site-grouped** (a site's seasons never split across
train/test), scored on **pooled out-of-fold predictions**, **macro**-averaged
(never micro/accuracy), with **per-flag MCC-tuned thresholds** — this is all
implemented in `Phylospec/evaluate_multilabel.py` (`site_grouped_cv()`,
`evaluate_cv()`); no benchmark runner should reimplement splitting/scoring.

## Data flow, in order

1. **Raw QIIME2 artifacts**: `data/qiimeresult/{ARC,BAC}/` — `table_filtered.tsv`
   (ASV × sample counts, **this is the sample-inclusion source of truth**),
   `rooted-tree.nwk`, `metadata.tsv` (despite the name: ASV taxonomy, *not*
   sample metadata), `dada2_rep_seqs.qza`. To drop/add samples, edit
   `table_filtered.tsv`/`.qza` here (both are git-tracked; use
   `qiime feature-table filter-samples --p-exclude-ids` in the `qiime2-2023.2`
   conda env, then re-export to `.tsv` — don't hand-edit the tsv with a text
   tool since QIIME auto-drops now-all-zero features).
2. **SEPP fragment-insertion tree** (already built, reused — see below):
   `data/qiimeresult/sepp_output/{sepp_tree.qza, sepp_tree_export/tree.nwk}`.
   This inserts both domains' rep-seqs into one SILVA reference backbone, so
   cross-domain distances come from the reference rather than an invented
   graft branch length (`build_merged.py`'s `--use-sepp-tree` route, task doc
   §5.5 route 1). **SEPP placement is per-ASV, not per-sample** — dropping
   samples only removes now-zero ASVs from downstream *tables*; it does not
   require re-running SEPP (~2.8h) unless you need to physically prune the
   backbone tree itself. `build_merged.py` defaults to this tree
   automatically via `run_genus_tree_pipeline.py`.
3. **`prepare_warning_targets_data.py`** (repo root) — reads
   `data/qiimeresult/{ARC,BAC}/table_filtered.tsv` directly, joins the 5-flag
   labels, and writes the **original ungrafted ASV-level** per-domain +
   merged (naive `DOMAIN_CONNECT_BRANCH_LEN=1.0` graft) train/test splits to
   `Phylospec/example/warnings_<domain>/<flag>/example_{train,test}.csv` and
   DeepPhylo npys to `Phylospec/multi_models/DeepPhylo/data/warnings_<domain>/`.
4. **`Phylospec/multi_models/data_preprocessing/build_multilabel_datasets.py`**
   — merges the 5 per-flag folders from step 3 into one multi-label
   train/test CSV per domain at
   `Phylospec/multi_models/input_for_all_models/Warnings_<domain>/`.
5. **`run_genus_tree_pipeline.py --mode genus --out-dir output/genus_tree/genus_autorun`**
   (repo root) — the *tree-aware* reduction: collapses ASVs to genus **on the
   tree** (not by label string — non-monophyletic genera get split into
   blocks, see task doc §2), for ARC, BAC, and merged (via `build_merged.py`
   + the SEPP tree from step 2). Writes `<domain>_table.csv`,
   `<domain>_tree.nwk`, `<domain>_mapping.csv`, `<domain>_report.json` per
   domain, plus per-flag label views.
   **`output/genus_tree/genus_autorun/` is the directory every downstream
   script actually reads — the sibling `output/genus_tree/genus/` (no
   `_autorun` suffix) is a stale one-off, not read by anything.**
   `--mode asv` / `--mode phylo` write `output/genus_tree/{asv,phylo}/` as
   standalone sanity-check arms (task doc §6) — **not** wired into any
   benchmark runner; don't confuse these with step 4's ASV-level inputs.
6. **Per-model-family input builders**, all in `Phylospec/multi_models/`:
   - `build_inputs_from_genus_tables.py` → `input_for_all_models_genus/Warnings_<domain>/` (from step 5)
   - `build_inputs_from_asv_tables.py` → `input_for_all_models_asv/Warnings_<domain>/` (from step 4)
   - `build_taxlevel_inputs.py` → `input_for_all_models_taxlevel/<Level>/Warnings_<domain>/` (naive taxonomy-label collapse at Phylum/Class/Order/Family/Genus, from step 5's raw abundance CSVs — flat classical-ML models don't need tree correspondence, task doc §2's monophyly rule doesn't apply to them)
   Each writes `table.csv` + `phylogeny.nwk` + PMCNN orderings + DeepPhylo `X/Y/c.npy`.
7. **Benchmark runners** (all in `Phylospec/multi_models/`, all build on
   `evaluate_multilabel.evaluate_cv()`, all output `results_<...>.csv` — one
   row per flag + a macro row, per task doc §5A.5):
   - `run_site_grouped_cv_benchmark.py` — **current/authoritative** GPU-aware
     harness for RF/CNN/PMCNN/MetaDR/DeepPhylo/PhyloSpec. Explicitly
     *replaces* `run_cv_benchmark.py` (sample-level splits, fold-averaging,
     accuracy/0.5-threshold — all of which the eval protocol forbids).
     `--csv-data-dir` defaults to the genus arm (`input_for_all_models_genus`);
     point it at `input_for_all_models_asv` for the ASV arm.
     Output → `results/site_grouped_cv/` or `results/site_grouped_cv_asv/`.
   - `run_taxlevel_baseline_benchmark.py` — 6 classical models (RF, KNN, SVM,
     NNET, XGBoost, GLMNET) × 5 ranks × 3 domains, CPU only, reuses the same
     harness/scoring. Output → `results/site_grouped_cv_taxlevel/`.
   - `run_cross_domain_injection_benchmark.py` + `build_cross_domain_prior_merged.py`
     — a **separate, later research question** (syntrophic cross-domain prior
     injection ablation), consumes the already-built genus merged table. Not
     part of the core 4-benchmark-family comparison; only touch when asked
     about cross-domain injection specifically. See "Cross-domain injection"
     below for the current (per-model) injection mechanisms and what's still
     unimplemented from the spec.
   - `run_cv_benchmark.py`, `run_multilabel_benchmark.py`, `Multi-comparison-models.py`,
     `build_cv_folds.py`, `build_site_grouped_cv_folds.py` — **superseded**,
     kept for reference only. Don't build on these.

## Environments

Base/system `python3` is broken for this repo (numpy2/numexpr ABI conflict,
missing `iterstrat`). Use conda envs:

- **`conda activate torch`** — for everything Python in this repo (pandas,
  sklearn, biopython, torch 2.4.1+cuda, iterstrat all present and verified).
  All 3 GPU model families (DeepPhylo, MetaDR, PhyloSpec) auto-select
  `cuda if torch.cuda.is_available()` — no `--device` flag exists or is
  needed, just run inside this env. GPU: RTX 4090, driver 560.35, CUDA 12.6.
- **`conda activate qiime2-2023.2`** — only for `qiime` CLI calls (feature
  table filtering, fragment-insertion SEPP). SEPP reference:
  `data/qiimeresult/sepp-refs-silva-128.qza` (gitignored, ~181MB, don't
  re-download unless missing).

## Cross-domain injection

Read `CLAUDE_CODE_TASK_cross_domain_injection.md` (repo root) before touching
this. It **supersedes** `Phylospec/CROSS_DOMAIN_INJECTION_BY_MODEL.md` (the
older doc — its PhyloSpec node-weight-reweighting and MetaDR 1-D-seriation
mechanisms are explicitly replaced, not complementary approaches). The idea:
`build_cross_domain_prior_merged.py` estimates which bacterium↔archaeon pairs
covary after conditioning on operating variables (`{prefix}_edges.csv`:
bacterium, archaeon, rho, q), and each model gets that edge set injected
through a *different* mechanism because each model reads the tree
differently:

- **DeepPhylo** — unaffected by this task; its channel is `--embedding_path`
  in `deepphylo_classification_multi_label_bce.py` (swap the whole
  PCA-of-patristic-matrix step for a precomputed fused/tree-only embedding).
- **Phylo-Spec** (`Phylospec/src/model/data_processing.py:inject_cross_domain_edges`,
  wired via `PhyloSpec_train_test.py`'s `-cross_domain_edges` CLI flag) —
  appends one synthetic `(children=[bac_leaf, arc_leaf], parent=f"syn_{i}")`
  merge to `conv_order` **per retained edge**, weight `node_weights[f"syn_{i}"]
  = abs(rho)`. Always appended last (conv_order has no topological sort — by
  the time these run, every leaf and every real internal node is already in
  `feature_map`, since `forward()` never pops it — a DAG walk, not a tree
  walk). **No modified Newick file, ever.** `-alpha` no longer exists —
  edge weight is fixed at `abs(rho)`, not a tunable scalar (that was the old
  doc's mechanism).
- **MetaDR** (`build_cross_domain_prior_merged.py`, consumed via
  `MetaDR-multilabel.py --seriation-order`, **only** on the post-order/zigzag
  branch — the level-order branch is left untouched as an internal control)
  — places taxa on a 2-D grid (spectral embedding of a fused similarity graph
  `W` + Hungarian assignment to grid cells) instead of a 1-D
  optimal-leaf-ordering seriation, so a taxon can have up to 8 grid neighbors
  (not 2), letting `--topk`-many cross-domain partners actually land in one
  5×5 conv footprint. **Critical detail**: `transform_image()` reshapes the
  order-file's flat list row-major into `img_size × img_size` *then* reverses
  every odd row (zigzag) — so the order file must be written with each
  taxon's column **pre-flipped** on odd target rows (`col = n-1-c if
  target_row % 2 else c`), or the zigzag scrambles the carefully-computed 2-D
  layout back into something arbitrary. Reports a placement-quality number
  (fraction of retained edges whose two endpoints land within one 5×5
  footprint of each other in the final grid) in `{prefix}_diagnostics.json`
  as `metadr_placement_quality` — it came out **13.6%** on the current
  134-sample merged data (330 features, 19×19 grid, topk=5), i.e. most edges
  do *not* end up co-located; report this honestly rather than treating it as
  a bug, per the task doc — a low number means that *arm* is uninformative,
  not that the code is wrong.

**What the task doc asks for that is NOT implemented yet** (scope was "fix
the injection mechanism + named PhyloSpec bugs", not the full spec):
- No parameter-matched **null arm** (`plus_null`) for either model — task doc
  §2's null-edge-list export (permuted-weight rewiring in
  `build_cross_domain_graph.py`) was never wired into
  `build_cross_domain_prior_merged.py`. Right now there is only
  with-vs-without, which the task doc explicitly says is not evidence by
  itself (§7: "a with-vs-baseline difference alone is not evidence").
- No per-fold edge discovery (§3) — the edge set is still estimated once on
  all samples, not inside each CV fold.
- No `run_phylospec_arms.py`/`run_metadr_arms.py`, no figures, no
  `REPORT_cross_domain_injection.md` (§6 deliverables).
- Fixed 3 of Phylo-Spec's 5 named bugs this session (§A.5): the optimizer
  bug (below), `match_leaf_nodes` substring matching, and the resulting
  `tree_p` disagreement. `out_feature=5` and multi-label indexing were
  already fixed by a prior session.

## Phylo-Spec optimizer bug (found and fixed 2026-09-11)

`PhyloSpec.conv1x1_layers` (one 1×1 conv per leaf / per unmatched feature)
is created **lazily inside `forward()`**, not in `__init__`. Every earlier
version of `PhyloSpec_train_test.py` built `torch.optim.Adam(final_model
.parameters())` *before* the model's first forward pass, so those layers
didn't exist yet when Adam snapshotted its parameter list — meaning **every
PhyloSpec model ever trained in this repo had random, never-updated leaf
convolutions**; only `res_blocks`/`fc1`/`fc2`/`fc3` actually trained. Fixed
by running one dummy forward pass on a single training sample right after
constructing `final_model`, discarding its `accumulated_node_features` side
effect, *then* building the optimizer, with an assertion
(`len(final_model.parameters()) == len(optimizer.param_groups[0]['params'])`)
guarding against this regressing silently again. This affects **every**
PhyloSpec result in `results/site_grouped_cv*`, not just the cross-domain
injection arms — the full PhyloSpec arm of the main benchmark was rerun
after this fix (see session log).

## Session log

**2026-09-11**: Dropped 4 samples with poor ARC (archaeal) sequencing —
`3-1-BSSG`, `3-9-DGGB`, `2-33-CWMS`, `3-22-UJGN` — from the ARC domain only
(BAC keeps all 140; ARC/merged now 134). Filtered
`data/qiimeresult/ARC/table_filtered.{tsv,qza}` via
`qiime feature-table filter-samples --p-exclude-ids` (8 ASVs that were
nonzero only in those 4 samples were auto-dropped by QIIME). Re-ran the full
chain: `prepare_warning_targets_data.py` → `build_multilabel_datasets.py` →
`run_genus_tree_pipeline.py --mode genus` (reused the existing SEPP tree —
no rerun needed, since SEPP places ASVs not samples and no ASV set changed
requiring backbone pruning) → all 3 `build_inputs_from_*` builders → full
benchmark suite (`run_taxlevel_baseline_benchmark.py`, then
`run_site_grouped_cv_benchmark.py` genus arm all 6 models, then its ASV arm
for DeepPhylo/MetaDR/PhyloSpec), on GPU (`conda activate torch`). Reran all
6 genus-level models (not just the 4 named by the user) since RF/CNN/PMCNN
results for ARC/merged would otherwise sit stale next to freshly-rerun ones
in the same results directory. Did **not** touch the cross-domain-injection
benchmark (out of scope for that request — it's a separate ablation study,
not one of the named benchmark families).

**2026-09-11 (later same day)**: Implemented `CLAUDE_CODE_TASK_cross_domain_injection.md`'s
injection mechanisms for Phylo-Spec and MetaDR (see "Cross-domain injection"
above), replacing the older `CROSS_DOMAIN_INJECTION_BY_MODEL.md` mechanisms.
While auditing Phylo-Spec's injection point, found and fixed a pre-existing,
unrelated bug affecting *every* PhyloSpec run in the repo — see "Phylo-Spec
optimizer bug" above — and fixed the `match_leaf_nodes` substring-matching
bug + its resulting `tree_p` disagreement (task doc §A.5). Reran the
PhyloSpec arm only (not the other 5 models, which weren't affected) of both
`results/site_grouped_cv/` and `results/site_grouped_cv_asv/` for all 3
domains. New shared module: `Phylospec/cross_domain_edges.py`
(`load_edges`/`check_edges`, task doc §4). Did **not** rebuild the
cross-domain-injection benchmark's own results (`results/cross_domain_injection*/`)
— those still reflect the old mechanisms and are now stale; smoke-tested the
new mechanisms there (2 folds, 5 epochs, scratch dir) but a full rerun needs
an explicit ask since it's ~1-2h of additional GPU time. Also not implemented:
the parameter-matched null arm (needs task doc §2's null-edge-list export,
never wired in) and per-fold edge discovery (§3) — both flagged in CLAUDE.md,
not silently skipped.
