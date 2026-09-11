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
     about cross-domain injection specifically.
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
