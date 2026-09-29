# NGS embedding and fusionML

This repository studies whether bacterial and archaeal community profiles can predict anaerobic-digester warning indicators, and whether phylogenetic and abundance relationships can compress those profiles into interpretable predictive groups. Repeated seasonal samples from a digester are related observations, so the principal comparisons use site-grouped validation.

The objective is to compare fixed-taxonomic-rank machine learning, phylogeny-aware deep learning, and domain-constrained log-contrast fusion models. Phylogenetic proximity and similar abundance patterns provide grouping evidence; neither establishes shared biological function. Existing analyses distinguish prediction, compression, and biological interpretation.

## Repository layout

```text
configs/                       YAML configurations and configuration schemas
function/                      pipeline orchestration, evaluation, plotting, reporting
models/
  baselines/
    TaxlevelML/                fixed-rank ML models and benchmark runner
    PhylogenicDL/              Phylo-Spec, MetaDR, DeepPhylo, PMCNN, CNN
  fusionML/                    estimator, search, interactions, coefficient groups
    experiments/               existing exp1 and exp5 scripts and documentation
datasets/
  graph/                       phylogeny and log-ratio-variance graph construction
  datapreprocess/              input readers, warning labels, tree resolution, zero handling
  raw/                         original data/final and data/qiimeresult contents
  processed/                   reduced trees, mappings, prepared tables
  fusionML/                    normalized input cache and synthetic demo inputs
  baselines/                   prepared baseline inputs and reference data
  experiments/                 preserved experiment input/source snapshots
results/
  data investigation/report.md
  classification result/report.md
  model interpretation/report.md
  ablation analysis/report.md
main.py
README.md
```

The spelling `PhylogenicDL` and the spaces in result directory names follow the requested layout. Quote result paths in shell commands. The published root contains only this revised layout plus `.gitignore`; the previous uploaded layout is preserved on `archive/main-before-restructure-2026-09-29`. Internal compatibility links preserve existing experiment layouts, while saved fusionML models load through the canonical modules. Tests and supporting design documents live in `function/tests/` and `function/docs/`. Frozen experiment inputs and historical reports retain their original content and provenance paths.

## Data

- **Relative abundance and taxonomy:** `datasets/raw/final/Dat_ARC.xlsx` and `Dat_BAC.xlsx` contain domain-specific taxonomic summaries, including abundance at phylum, class, order, family, genus, and species ranks. QIIME exports and taxonomy assignments are under `datasets/raw/qiimeresult/{ARC,BAC}/`.
- **Count tables and phylogenetic trees:** QIIME feature tables and rooted Newick trees are under the same domain directories. Normalized sample-ID tables used by fusionML are in `datasets/fusionML/real_data_cache/`. Reduced trees and their ASV-to-block mappings are under `datasets/processed/genus_tree/`.
- **Chemistry:** `datasets/raw/final/Dat_chem.xlsx` contains substrate and effluent measurements. `metadata.csv` is the existing derived metadata/label table. The newer fusionML configurations assemble targets directly from the chemistry workbook and their explicit site/season crosswalks.
- **Observation unit:** a site-season sample, with repeated visits grouped by site. Sample IDs commonly use `{round}-{No}`. Available sample counts and site groupings differ across historical protocols; use each report's cohort definition rather than combining their scores.

Warning labels include acid-base balance, buffer capacity, acid accumulation, ammonia toxicity, biogas quality, and one-sided acid-low/acid-high flags. Definitions are implemented in `datasets/datapreprocess/chemistry_targets.py` and the historical `datasets/datapreprocess/legacy/metadata_targetwriter.py`. Thresholds and missing-value treatment differ between historical workflows; inspect the selected configuration and source report before comparing them. The metadata writer modifies `metadata.csv` when explicitly run.

fusionML expects raw counts by default. Set `preprocessing.relative_input_mode: true` explicitly to accept relative-abundance inputs, and choose detection/pseudocount settings appropriate to their scale. An absent assay and a measured zero are distinct. Feature filtering, zero handling, graph fitting, and hyperparameter selection happen within training partitions. Each microbial domain is normalized separately.

## Models and graphs

| Component | Implementation and role |
|---|---|
| TaxlevelML | RF, KNN, SVM, NNET, XGBoost, GLMNET and existing rank-specific training scripts; fixed-rank abundance predictors |
| PhylogenicDL | Existing Phylo-Spec, MetaDR, DeepPhylo, PMCNN and CNN implementations and benchmark scripts |
| fusionML | Convex log-contrast regression/classification with separate zero-sum constraints for ARC and BAC; sparsity, ridge, and graph-fusion penalties |
| Phylogenetic graph | Nearest-neighbor relationships from patristic tree distances |
| Abundance graph | Site-weighted variance of log abundance ratios, joint-detection requirements, mutual neighbors, and bootstrap support |
| Fusion controls | No fusion, phylogeny-only, abundance-only, shared/domain-adaptive mixtures, domain-restricted fusion, and optional cross-domain interactions |

`datasets/graph/fusion_graphs.py` computes log-ratio variance as `Var(log(q_j) - log(q_k))`. The training pipeline rebuilds these graphs inside each training partition. Full-cohort exports made by the descriptive stages are not inputs to nested cross-validation.

The fusion implementation supports both classification and regression. `task` in the YAML determines which is run; the synthetic demonstration configuration uses regression.

## Environment

Use the existing Python 3.12 `torch` environment for the full repository, including the experiment report scripts:

```bash
conda activate torch
python main.py --help
```

For a new environment, the existing dependency lists are [fusionML requirements](models/fusionML/requirements.txt), [taxonomic ML requirements](models/baselines/TaxlevelML/requirements.txt), and [phylogenetic benchmark requirements](models/baselines/PhylogenicDL/multi_models/requirements.txt). These originated in separate workflows and are not a newly unified environment lockfile.

## Main workflow

```text
import abundance/counts + taxonomy + phylogenetic trees + chemistry
  -> preprocessing and resolution mapping
  -> descriptive graph construction and data investigation
  -> baseline and/or fusionML training/testing
  -> fitted-model coefficient/group interpretation
  -> comparison of the configured ablation families
```

Start with validation or a dry run:

```bash
python main.py --stage validate --config configs/fusionML/real_smoke_classification_phylo.yaml
python main.py --stage all --model all --config configs/fusionML/real_smoke_classification_phylo.yaml --dry-run
```

Run the fusionML workflow, or individual stages:

```bash
python main.py --stage all --config configs/fusionML/real_smoke_classification_phylo.yaml
python main.py --stage preprocess --config configs/fusionML/synthetic_demo.yaml
python main.py --stage graph --config configs/fusionML/synthetic_demo.yaml
python main.py --stage investigate --config configs/fusionML/synthetic_demo.yaml
python main.py --stage train --config configs/fusionML/real_smoke_classification_phylo.yaml
python main.py --stage interpret --config configs/fusionML/real_smoke_classification_phylo.yaml
python main.py --stage ablate --config configs/fusionML/real_smoke_classification_phylo.yaml
```

`all` performs validation first and stops on failure. `--model all` also launches both existing baseline benchmarks at the training stage. Those benchmarks use their own prepared inputs and evaluation settings; launching them together does not make their cohorts or splits identical to the fusionML config. A full run can be expensive. `--dry-run` prints the plan without loading scientific dependencies or fitting models.

Prediction uses a saved model and does not refit:

```bash
python main.py --stage predict --config configs/fusionML/real_smoke_classification_phylo.yaml \
  --model-path 'results/classification result/fusionML/real_results_classification_phylo/smoke/final_model.pkl' \
  --output /tmp/predictions.csv
```

Baseline-only examples:

```bash
python main.py --stage train --model TaxlevelML \
  --baseline-args='--levels Genus --domains ARC --models RF --n-splits 3 --n-repeats 1'
python main.py --stage train --model PhylogenicDL --baseline-args='--help'
```

Preparation commands remain separate when their outputs are needed:

```bash
python datasets/datapreprocess/prepare_real_data_smoke_inputs.py
python datasets/datapreprocess/run_genus_tree_pipeline.py --mode genus
python datasets/datapreprocess/generate_synthetic_demo_data.py
```

These commands write prepared inputs. The first command normalizes existing QIIME sample IDs; the second rebuilds reduced trees/tables; the third regenerates the synthetic demonstration data. The staged `preprocess` command uses the input readers, chemistry targets, and resolution mappings configured in YAML and exports full-cohort log compositions. It does not implicitly run every historical data preparation script.

## Arguments and configuration

| Argument | Purpose |
|---|---|
| `--stage` | `validate` (default), `preprocess`, `graph`, `investigate`, `train`, `predict`, `interpret`, `ablate`, `all` |
| `--config` | fusionML YAML; required when running fusionML stages |
| `--model` | `fusionML` (default), `TaxlevelML`, `PhylogenicDL`, or `all`; baseline-only runs use `train` |
| `--model-path` | saved model for prediction; optional override for interpretation |
| `--output` | prediction CSV or output directory for descriptive/interpretation/ablation stages; training uses YAML `output_dir` |
| `--baseline-args` | quoted arguments forwarded to baseline benchmark parsers; use `--baseline-args='--help'` for their full options |
| `--dry-run` | print the requested stages and baseline commands without executing |

Configurations live in `configs/fusionML/`; schema/defaults are in `configs/fusionml.py`. Relative paths resolve against the actual configuration file's directory, including when it is opened through a compatibility link. The older project-schema template `configs/ad-microbial-config.yaml` has its own validation contract and is not a drop-in training configuration.

| YAML section | Main settings |
|---|---|
| `task`, `resolution`, `seed` | classification/regression, ASV or configured taxonomic rank, random seed |
| `inputs`, `columns`, `chemistry` | counts, metadata/chemistry, taxonomy, trees, orientation, target, site/sample identifiers |
| `preprocessing` | detection/prevalence filters, pseudocount, sensitivity pseudocounts, explicit relative-abundance mode |
| `graph` | phylogenetic/abundance neighbor counts, joint support, bootstrap repetitions and support threshold |
| `cv` | outer/inner folds, selection rule, eligible `families` |
| `search` | candidate budget, sparsity/ridge/fusion penalties, domain mixture weights |
| `solver` | solver selection and tolerances |
| `classification`, `interaction` | classification thresholds/metrics and optional ARC–BAC interaction controls |
| `compression` | coefficient grouping tolerance and residual retention |
| `output_dir`, `smoke`, `run_id` | training output location and run metadata |

Preprocess/graph exports default to `datasets/processed/fusionML/<config-name>/<stage>/`. Investigation, interpretation, and ablation reports default to their respective result category under `<config-name>/`. Interpretation exports fitted coefficients and explicit group memberships. The ablation stage summarizes already-trained families from that config's `results.csv`; select controls through `cv.families` and train them first. It reports when only one family is available.

## Existing results and reproduction

- [Data investigation](results/data%20investigation/report.md)
- [Classification results](results/classification%20result/report.md), including the preserved [Experiment 1 report](results/classification%20result/fusionMLv2/exp1/REPORT.md)
- [Model interpretation](results/model%20interpretation/report.md)
- [Ablation analysis](results/ablation%20analysis/report.md)

Archived Experiment 1 and Experiment 5 scripts remain under `models/fusionML/experiments/`. Their `data/` and `results/` links lead to the centralized snapshots and results. These analyses describe saved experiments, while `main.py --stage interpret/ablate` operates on the selected config's fitted model/results. Do not refresh archived snapshots merely to regenerate their reports. See each experiment README for its original protocol and reproduction commands.

Run the implementation checks with:

```bash
python -m pytest models/fusionML/tests function/tests models/baselines/TaxlevelML -q
```
