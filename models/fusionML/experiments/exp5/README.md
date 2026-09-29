# Experiment 5: biological interpretation

Two sub-experiments, both interpretation-only (no model is fitted in either):

| Directory | Scope |
|---|---|
| [`selected_order/`](selected_order/) | **Current.** The two configurations Experiment 1 selected as the highest-MCC member of their fusion arm: `warning_ammonia_toxicity` x `shared_hybrid` x order (both domains fused) and `warning_acid_high` (effluent pH > 8.0) x `fusion_arc_only` x order. |
| `scripts/`, `data/`, `results/` (this directory) | **Earlier scope.** Five model families for ammonia toxicity, the two-sided acid-base flag and biogas quality, at ASV and genus resolution. Predates the order-level domain-restricted runs and does not cover `warning_acid_high`. |

The two do not overlap in flag, resolution or model set, so neither supersedes
the other; read `selected_order/` for the current selected-model interpretation.

## Earlier scope (this directory)

This analyzes `fusion_arc_only`, `fusion_bac_only`, `shared_hybrid`, `domain_adaptive_hybrid`, and `domain_adaptive_interaction` for ammonia toxicity, acid–base balance, and biogas quality, at ASV and genus resolution.

`fusion_arc_only` and `fusion_bac_only` retain both domains as predictors; only the named domain receives fusion edges. These are not single-assay prediction models.

### Run and reproduce

From the repository root, using the `torch` environment:

```bash
/home/best/anaconda3/envs/torch/bin/python models/fusionML/experiments/exp5/scripts/copy_inputs.py
XDG_CACHE_HOME=/tmp/fusionmlv2-exp5-cache /home/best/anaconda3/envs/torch/bin/python models/fusionML/experiments/exp5/scripts/run_missing_asv.py --workers 3
/home/best/anaconda3/envs/torch/bin/python models/fusionML/experiments/exp5/scripts/generate_results.py
```

To regenerate interpretation without training or recopying, run only the last command. `--results-dir /tmp/fusionml-exp5-check` generates a separate copy of the report artifacts while reading the existing model fits. The runner resumes completed partitions only when its source/input-manifest fingerprint matches. Do not refresh input snapshots mid-run.

### Inputs and protocols

- `data/`: hash-verified copies of the count tables, chemistry workbook, ASV trees, taxonomy, genus trees and ASV-to-genus-block mappings, existing genus/ASV output tables, configuration, and fusionML source used for the new fits.
- Reused genus fits: three outer folds, 37 original site labels, all five models and three flags. These historical smoke runs are not the later QC-corrected genus prediction runs from Experiment 1.
- Reused ASV fits: Experiment 1's three hybrid models for ammonia and acid–base; two repeats, three outer folds, 35 corrected sites.
- New ASV fits: the two domain-restricted models for all three flags, and three hybrid models for biogas; same splitting, candidate generation and one-SE selection rules as Experiment 1. Training uses only the copied implementation and inputs.
- Biogas has 13 positive samples across six sites. Two ASV outer splits have a single-class inner training partition; all five models are unavailable for those splits. Their missing fits are retained in the inventory, and incomplete out-of-fold performance is not reported as a full benchmark.

Historical taxonomy hashes are checked against saved manifests. The old manifests did not hash genus crosswalks, so the copied mappings are current context rather than authenticated historical files. Resolution comparisons are descriptive because the genus and ASV protocols differ.

### Outputs

Start with [results/REPORT.md](results/REPORT.md). Six linked detail reports cover every flag/resolution and each model. Figures are exported as PNG and PDF.

| File under `results/tables/` | Purpose |
|---|---|
| `model_fit_inventory.csv` | All planned fits, provenance and unavailable-fit status |
| `group_catalog.csv` | Every group, exact members, coefficients, geometric-mean weight, size, taxonomy and role hypotheses |
| `feature_asv_taxonomy.csv` | Exact model-feature → ASV → original taxonomy mapping |
| `coefficients_annotated.csv` | Original feature coefficients, residuals and annotations |
| `feature_associations.csv` | Feature retention, activity and sign counts across available fits |
| `model_domain_summary.csv` | ARC/BAC group counts, taxonomic coherence and annotation coverage |
| `domain_contrasts.csv` | Positive/negative weighted log-ratio members per model/domain/fit |
| `interaction_summary.csv` | Active interaction counts, including unavailable fits |
| `active_interactions_annotated.csv` | Taxonomic explanations of selected interaction products, if any |
| `prediction_context.csv` | Complete-repeat OOF metrics only; partial coverage remains explicit |
| `functional_evidence.csv` | Conservative manually curated genus-role hypotheses and primary sources |
| `taxonomy_domain_mismatches.csv` | Assay labels inconsistent with assigned taxonomic domain |
| `cohort_summary.csv`, `fit_validation.csv` | Cohort/event counts, site isolation and score reconstruction checks |

`results/model_fits/new_asv/` contains the new fits, predictions, group and interaction exports, search records, failures, splits and resumable checkpoints. Those files have their own run manifest. `results/artifact_manifest.json` hashes the interpretation artifacts; `results/validation.json` records checks and script hashes.

Run `python models/fusionML/experiments/exp5/scripts/verify_results.py` in the same environment for independent integrity checks and uncached refits of the ARC-restricted and BAC-restricted models. This verifies the graph restrictions and compares their predictions against the saved cached fits.

### Interpretation limits

A fused group represents equal fitted log-contrast coefficients, not established functional redundancy. Group scores use geometric means rather than summed abundance. Coefficient directions are relative, conditional associations; they do not identify causal effects or absolute microbial activity. Taxonomic compactness is partly imposed by the graph. Cross-fold sign counts are descriptive and do not constitute an independent stability experiment. Literature annotations apply to related named organisms; these 16S data do not directly measure genes, pathways, fluxes or syntrophy.
