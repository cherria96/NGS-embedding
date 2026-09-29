# Experiment 5 — biological interpretation of the two selected order-level models

Reads, as biology, the two configurations Experiment 1 identified as the
highest-MCC member of their fusion arm:

| Flag | Label | Arm | Model | Resolution |
|---|---|---|---|---|
| `warning_ammonia_toxicity` | effluent NH4-N (TAN) > 2.5 | both domains fused | `shared_hybrid` | order |
| `warning_acid_high` | effluent pH > 8.0 | ARC-only fusion | `fusion_arc_only` | order |

Both are rank 1 of 20 (4 resolutions x 5 models) for their flag by MCC **and**
by F2 under the minimum-inner-loss rule, which the interpretation recomputes
from the saved probabilities rather than taking on trust.

**The acid-high flag is pH > 8.0, not 8.5.** That is the threshold in
`models/fusionML/chemistry_targets.py` and the definition the selected model was
trained on. The cohort's maximum effluent pH is 8.32, so a > 8.5 flag would
have no positives at all.

This is a separate sub-experiment from the one in the parent directory, which
interpreted five model families for ammonia, the two-sided acid-base flag and
biogas quality at ASV and genus resolution. That analysis is still valid for
its own scope; it predates the order-level domain-restricted runs and does not
cover `warning_acid_high`. Nothing there is modified by this directory.

## Reproduce

From the repository root, in the `torch` environment:

```bash
/home/best/anaconda3/envs/torch/bin/python models/fusionML/experiments/exp5/selected_order/scripts/copy_inputs.py
/home/best/anaconda3/envs/torch/bin/python models/fusionML/experiments/exp5/selected_order/scripts/interpret.py
```

`copy_inputs.py` snapshots the inputs with SHA-256 hashes; `interpret.py` reads
only that snapshot and writes `results/`. **No model is fitted here.** Both
scripts accept `--data-dir` / `--results-dir` if you want a scratch copy.

## Inputs

`data/source_manifest.json` records the original path, size and hash of every
copied file.

- `data/runs/{ammonia,acid_high}__order/` — the full export of the two selected
  runs: `rows`, `predictions`, `coefficients_and_groups`, `domain_counts`,
  `filter_audit`, `splits`, `search`, `failures`, `interactions`, run manifest.
- `data/runs/{ammonia,acid_high}__{family,genus,asv}/` — `rows` and
  `predictions` only, enough to recompute where the selected configurations
  rank among all 20 candidates per flag.
- `data/reference/order/` — the monophyly-consistent order trees, the
  ASV-to-block crosswalks (`*_mapping.csv`) and the block count tables.
- `data/reference/taxonomy/` — the ARC and BAC ASV taxonomy assignments.
- `data/reference/chemistry/Dat_chem.xlsx` — the chemistry workbook the labels
  are computed from.
- `data/fusionML/*.py` — source snapshot, so the label and lineage definitions
  are pinned. Only the config, chemistry and taxonomy helpers are imported; the
  estimator is never called.

acid_high was fit in its own run directory
(`results/classification result/fusionML/real_results_domain_fusion_acidhigh_<resolution>_exp1_20260922`);
the other flags share `real_results_domain_fusion_<resolution>_exp1_20260922`.

## Outputs

Start with [results/REPORT.md](results/REPORT.md). Tables and figures are listed
at the end of it. `results/validation.json` records the snapshot verification,
zero-sum and group-reconstruction residuals, the per-fit fusion-restriction
checks, site isolation of every split, reproduction of the saved fold metrics
from the saved probabilities, and script hashes.
`results/artifact_manifest.json` hashes everything generated.

## Interpretation limits

Stated in full in the report. The short version: these are 16S amplicons, so no
gene, pathway or activity was measured; every coefficient is a conditional
association inside a per-domain log-ratio, so a positive sign means "high
relative to this model's denominator", not "abundant" or "active"; a fused
group is an equality constraint that improved held-out loss, not a functional
guild; the two flags overlap heavily (9 of 15 acid-high positives are also
ammonia positives, pH-TAN Spearman 0.74), so agreement between the two models
is not independent confirmation; and both configurations were chosen as the
best of 20 on held-out scores with no multiplicity correction and no remaining
held-out cohort.
