# Experiment 1: prediction–compression tradeoff

Reanalysis of existing genus-level `models/fusionML/` outputs, extended with two new ASV-level fusionML experiments: the eight-family resolution comparison, and an ARC-only/BAC-only/both-domain fusion comparison across three warning flags. The report generator does not fit models; the dedicated runners do.

## Layout

- `data/`: byte-for-byte copies of the selected source outputs, protocol documents, and source-code snapshots. `source_manifest.json` records each original path, size and SHA-256 hash.
- `scripts/copy_inputs.py`: copies and verifies the selected inputs from `models/fusionML/`.
- `scripts/generate_results.py`: standalone reanalysis using only the copied data.
- `scripts/run_asv_fusion.py`: fits ASV-level fusion on the original ARC/BAC rooted trees with the existing fusionML estimator.
- `scripts/asv_results.py`: summarizes the copied ASV run and produces resolution comparisons.
- `scripts/run_domain_restricted_fusion.py`: fits the ARC-only, BAC-only and both-domain fusion arms for three warning flags at a chosen taxonomic resolution, on the same protocol.
- `scripts/domain_restricted_results.py`: summarizes those runs under the minimum-inner-loss rule, ranked by MCC across resolutions.
- `results/REPORT.md`: results, figures, methods, and limits of the existing evidence.
- `results/tables/`: machine-readable summaries and intermediate results.
- `results/figures/`: PNG and PDF figures.
- `results/validation.json`: coverage, split-isolation, metric-reproduction and provenance checks.
- `results/artifact_manifest.json`: hashes of generated artifacts.

## Reproduce

Run from the repository root with the existing environment:

```bash
/home/best/anaconda3/envs/torch/bin/python models/fusionML/experiments/exp1/scripts/copy_inputs.py
/home/best/anaconda3/envs/torch/bin/python models/fusionML/experiments/exp1/scripts/generate_results.py
```

To fit the ASV extension and include it in the report:

```bash
/home/best/anaconda3/envs/torch/bin/python models/fusionML/experiments/exp1/scripts/run_asv_fusion.py --workers 3
/home/best/anaconda3/envs/torch/bin/python models/fusionML/experiments/exp1/scripts/copy_inputs.py --include-asv
/home/best/anaconda3/envs/torch/bin/python models/fusionML/experiments/exp1/scripts/generate_results.py
```

To fit the domain-restricted fusion arms (three flags x five models per resolution, about five minutes each on three workers) and include them:

```bash
for res in order family genus asv; do
  /home/best/anaconda3/envs/torch/bin/python models/fusionML/experiments/exp1/scripts/run_domain_restricted_fusion.py --workers 3 --resolution $res
done
/home/best/anaconda3/envs/torch/bin/python models/fusionML/experiments/exp1/scripts/copy_inputs.py --include-asv --include-domain-fusion
/home/best/anaconda3/envs/torch/bin/python models/fusionML/experiments/exp1/scripts/generate_results.py
```

Each resolution writes its own `results/classification result/fusionML/real_results_domain_fusion_<resolution>_exp1_20260922/`; the reporting module includes whichever ones were copied. `fusion_arc_only` and `fusion_bac_only` keep both domains as predictors and give a fusion graph to one domain only; they are not single-assay models. The ASV both-domain families reproduce the ASV run's overlapping fits exactly, which `generate_results.py` checks rather than assumes.

Rank resolutions aggregate ASVs on the monophyly-consistent trees under `datasets/processed/genus_tree/taxlevel_autorun/<rank>/`, using each tree's co-located `*_mapping.csv` crosswalk, not taxonomy labels. All resolutions use the 35-group QC protocol with two repeats, so they are comparable to each other and to the rest of this report. They are **not** the older 37-site single-repeat smoke runs under `results/classification result/fusionML/real_results_*/smoke/`, and the two sets must not be pooled or ranked together.

The ASV runner writes a separate checkpointed run under `results/classification result/fusionML/real_results_asv_exp1_20260922/`. It verifies cohort labels and outer split membership against the copied genus outputs and records its full configuration and source/input hashes. It uses `datasets/raw/qiimeresult/ARC/rooted-tree.nwk` and `datasets/raw/qiimeresult/BAC/rooted-tree.nwk` directly. It preserves the original runs. The runner requires the existing fusionML dependencies and original input files; reanalysis after copying requires neither raw inputs nor the estimator.

Once copied, the second command does not require access to `models/fusionML/` or raw data. Dependencies are NumPy, pandas, Matplotlib, and scikit-learn; actual versions are recorded in `results/validation.json`. The copy command intentionally refreshes the snapshot from the current source files. Both scripts accept alternate input/output directory arguments; use `--help` for details.

The source run manifest records code/config hashes that differ from the currently available fitting script and configuration. Their copies are explicitly labeled context snapshots rather than original training versions. Saved-prediction reanalysis is verified independently; exact refitting of the original models is not established. See the report's provenance note.

## Analysis scope

The regularization paths use the original 37-site convention and fixed ridge/sparsity penalties. The primary model comparisons use 35 QC-adjusted groups and only fraction=1.0 of the saved nested CV experiment. These protocols are analyzed separately. No learning-curve results are generated, and no test-based deployment settings are selected.

The primary comparison uses saved one-standard-error selections for fusion and saved minimum-loss selections for ordinary ML. Both source records use the `one_se` tag; `selection_rule` disambiguates them. Alternative saved minimum-loss fusion selections are supplementary. These are results for existing fusionML models, not newly validated fusionMLv2 models.

The new ASV experiment uses eight families, the same candidate-generation rules, two targets, two repeats, three outer folds and two inner folds. Both selection rules are saved. The 324 new ASV path evaluations use the QC-adjusted 35-group protocol, whereas the reused genus paths use the old 37-site protocol. Only the selected-model resolution comparison shares the same outer site assignments. Changing resolution also changes aggregation and prevalence-filter survival; it is not an isolated test of the tree alone.

ASV dimensions are reported in three stages: 15,153 input ASVs, ASVs retained by training-only prevalence filtering, and active fused groups. Do not attribute the large filtering reduction to fusion.

The domain-restricted comparison adds `warning_biogas_quality`, the two restricted families, and the order/family/genus/ASV resolutions on the same QC protocol. It reports the minimum-inner-loss rule and ranks by MCC with F2 alongside, because F2 at 8-18% prevalence is maximized by near-all-positive predictors; inner-fold log-loss still selects every candidate. Each arm's reported resolution and model are the observed highest-MCC combination on held-out folds, so that choice reads test scores: it is a descriptive best in this cohort, not a validated deployment setting, and picking the best of four resolutions x five models selects some noise upward with no multiplicity correction applied. Biogas quality has 11 positives and loses two of its six outer splits to single-class inner training partitions at every resolution, so its metrics are partial out-of-fold results and its models sit at or below chance discrimination.

## Table guide

| File | Contents |
| --- | --- |
| `qc_primary_comparison.csv` | Full-size selected models; mean group counts, mean repeat-pooled metrics, compression, paired loss differences |
| `qc_all_selection_rules.csv` | The same summaries for both saved fusion selection rules |
| `qc_selection_sensitivity.csv` | Paired changes in group count and performance between the two saved selection rules |
| `qc_repeat_metrics.csv` | Metrics recomputed independently from the saved OOF probabilities |
| `qc_full_size_fits.csv` | Selected outer-fit records, including candidate settings and counts |
| `qc_fixed_oof_site_losses.csv` | Site losses averaged across repeats for paired comparisons |
| `qc_paired_loss_vs_ridge.csv` | Paired site-bootstrap loss differences conditional on fixed predictions |
| `regularization_fold_results.csv` | Original path rows plus within-fold compression and unfused-reference loss differences |
| `regularization_summary.csv` | Three-fold summaries of each graph/strength combination |
| `historical_saved_runs.csv` | Run-specific appendix joining old scores and compression; not pooled across protocols |
| `asv_primary_comparison.csv` | New ASV-tree models selected by the one-SE rule |
| `asv_all_selection_rules.csv` | New ASV results for both selection rules |
| `asv_vs_genus_qc.csv` | Paired resolution summaries with the same cohort and outer site splits |
| `asv_repeat_metrics.csv` | Recomputed repeat-pooled metrics for the ASV run |
| `asv_full_size_fits.csv` | ASV selected fits, constraints, group reconstruction checks and candidate settings |
| `asv_regularization_summary.csv` | New ASV fixed-penalty compression paths |
| `asv_filtering_and_compression.csv` | Original counts, retained ASVs, abundance mass retained, and shared-hybrid group counts |
| `domain_fusion_min_loss_comparison.csv` | Every flag x resolution x model under the minimum-inner-loss rule, with per-domain group counts and both compression denominators |
| `domain_fusion_best_by_resolution.csv` | The resolution and model representing each arm, how it was ranked, and whether F2 would have chosen differently |
| `domain_fusion_resolution_grid.csv` | Active groups, reduction and MCC laid out as flag x arm x model by resolution |
| `domain_fusion_prior_best_configurations.csv` | The resolution/model pairs previously read off the 37-site smoke runs, recomputed on this protocol, with whether each is also the best here |
| `domain_fusion_all_selection_rules.csv` | The same summaries for both saved selection rules |
| `domain_fusion_repeat_metrics.csv` | Repeat-pooled metrics recomputed from the saved probabilities, with an out-of-fold completeness flag |
| `domain_fusion_fits.csv` | Available outer fits with ARC/BAC retained, edge and group counts |
| `domain_fusion_fit_inventory.csv` | Every planned fit including the unavailable biogas-quality splits |

## Definitions

- `retained_features_mean`: mean number of input taxa after that pipeline's preprocessing.
- `active_taxa_mean`: nonzero taxon coefficients for linear models, or nonzero-importance taxa for the saved XGBoost models.
- `active_groups_mean`: active coefficient groups for log-contrast models only. Missing for ordinary ML; its active taxa are not fused groups.
- `structural_compression_pct`: mean over outer fits of `100 × (1 − active_groups / retained_taxa)`. It excludes filtering from the claimed compression and is not a certified reduction in required assays or exported input columns.
- `logloss`: site-balanced loss, with probabilities clipped to `[1e-6, 1−1e-6]` as in the original implementation. QC predictions are pooled within each repeat before scoring; repeat scores are then averaged.
- `auc` / `ap`: sample-level AUROC / average precision. `brier`, `mcc`, and `f2` also preserve the source sample-level definitions. Thresholds are the saved training-selected thresholds.
- `ci_low` / `ci_high`: 95% percentile intervals for loss differences from 5,000 paired resamples of site-level fixed OOF losses. Repeated predictions are not treated as independent samples. These intervals omit model-refitting uncertainty.
- `observed_nondominated_logcontrast`: descriptive comparison of mean active groups and mean loss among saved log-contrast families only. It is not an outer-test model-selection rule.
- Path `test_auc_fold_mean` is a fold average, not pooled AUROC. Path uncertainty is not combined with QC comparison uncertainty.

The original 134 samples provide 35 QC-adjusted site groups, not 134 independent sites. Small coefficient residuals, active interaction terms, and selected/used original taxa are distinct from group counts. Missing compression baselines and matched-size experiments are identified in the report.
