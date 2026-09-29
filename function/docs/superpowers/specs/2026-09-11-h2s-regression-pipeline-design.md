# H2S regression pipeline — design spec

Date: 2026-09-11
Status: approved for planning

## 1. Goal

Add a new regression target ("Group" column = H2S) to the existing AD
warning-flag pipeline, and run it through:

1. **TaxLevel ML models** — the 6 flat classical models (RF, KNN, SVM,
   NNET, XGBoost, GLMNET) at Phylum/Class/Order/Family/Genus rank x
   ARC/BAC/merged domain.
2. **ASV-tree**, with and without cross-domain fusion.
3. **Genus-tree**, with and without cross-domain fusion.

The existing pipeline (`Phylospec/evaluate_multilabel.py`,
`run_site_grouped_cv_benchmark.py`, `run_taxlevel_baseline_benchmark.py`,
`build_taxlevel_inputs.py`, `build_site_grouped_cv_folds.py`) is entirely
built around 5 binary warning flags scored with MCC/F-beta/AUPRC/AUROC.
None of that classification machinery (pos_weight, threshold tuning,
multilabel stratification) applies to a single continuous target, so this
spec adds a parallel regression track that reuses the site-grouped CV
*protocol* (repeated, site-grouped, pooled out-of-fold) but not the
classification-specific pieces.

## 2. Target construction ("Group" = H2S)

`data/final/metadata.csv` has `eff_H2S` for 137/140 samples (3 missing,
dropped). Distribution is strongly right-skewed (skew 2.36, range
0-9.65). The regression target is `log1p(eff_H2S)`; all reported metrics
are back-transformed to real H2S units via `expm1` except Spearman ρ
(scale-invariant).

"Group column" follows the existing Phylo-Spec CSV convention already
used elsewhere in this repo (`feature_1...feature_n, Group` as the last
CSV column, e.g. `Real_Dateset_16S_CRC/16S_CRC.csv`) — for the H2S task,
`Group` = `log1p(eff_H2S)`.

New module: `Phylospec/multi_models/data_preprocessing/h2s_target.py`
- `load_h2s(metadata_path) -> pd.Series` indexed by SampleID, log1p scale,
  NaNs dropped.
- `back_transform(y) -> expm1(y)`.

## 3. Regression CV + evaluation protocol

New module `Phylospec/evaluate_regression.py`, sibling of
`evaluate_multilabel.py`, same repeated/pooled/site-grouped shape:

- `site_grouped_cv_regression(y, sites, n_splits, n_repeats, seed)`: bins
  sites into quartiles by site-mean log1p(H2S), then
  `sklearn.model_selection.StratifiedGroupKFold` (available,
  scikit-learn>=1.4.2 per requirements.txt) grouped by site, stratified
  by quartile bin. Mirrors `site_grouped_cv`'s signature/semantics.
- `split_train_val_regression`: inner split out of the outer-training
  fold only, same discipline as `split_train_val` (needed here for
  early-stopping in the torch models, not for threshold tuning — there
  is none).
- `score_regression(y_true, y_pred)`: RMSE, MAE, R², Spearman ρ, computed
  after `back_transform`.
- `evaluate_cv_regression(y, sites, fit_predict, ...)`: pooled-OOF per
  repeat, mean + 95% percentile interval over repeats. No threshold
  tuning step (`fit_predict` signature drops `pos_weight`).

## 4. TaxLevel ML models

`build_taxlevel_h2s_inputs.py` (mirrors `build_taxlevel_inputs.py`):
same per-domain abundance loading and rank-collapse, but joins
`h2s_target.load_h2s()` as the single `Group` column instead of the 5
`FLAG_COLS`. Writes
`input_for_all_models_taxlevel_h2s/<Level>/H2S_<Domain>/table.csv`.

Each of RF/KNN/SVM/NNET/XGBoost/GLMNET gets a `*-regression.py` sibling
next to its existing `*-multilabel.py`, same `--train/--val/--test
/--scores-out-dir` CLI contract, single continuous `Group` column instead
of the 5-column `FLAGS` matrix:

| Model | Classifier (existing) | Regressor (new) |
|---|---|---|
| RF | RandomForestClassifier | RandomForestRegressor |
| KNN | KNeighborsClassifier | KNeighborsRegressor |
| SVM | SVC (MultiOutputClassifier) | SVR |
| NNET | MLPClassifier | MLPRegressor |
| XGBoost | XGBClassifier x5 | XGBRegressor |
| GLMNET | LogisticRegression(elasticnet) x5 | ElasticNet |

Same hyperparameter choices as the classifier versions where they carry
over (n_estimators=500/max_depth=5 for RF, etc.); no `class_weight` /
`pos_weight` concept for regression.

New `run_taxlevel_h2s_benchmark.py` (mirrors
`run_taxlevel_baseline_benchmark.py`): 5 ranks x 3 domains x 6 models,
each through `evaluate_cv_regression`. Output:
`results_<level>_<domain>_<model>.csv` in
`Phylospec/multi_models/results/site_grouped_cv_h2s_taxlevel/`.

## 5. ASV-tree / Genus-tree, with and without cross-domain

Applies only to the **merged** (BAC+ARC) domain — cross-domain
association is undefined for a single-domain table, and
`build_cross_domain_graph.py` already requires both domains present.

`build_cross_domain_graph.py --regression` already writes **both**
`<out>_DeepPhylo_embeding.npy` (cross-domain fused) and
`<out>_treeonly_DeepPhylo_embeding.npy` (tree-only, no cross-domain) from
one invocation — reused as-is, not modified:

- ASV-tree: `-c` merged ASV table (features + `Group` last column) built
  from `output/genus_tree/asv/merged_build/merged_relative_abundance.csv`
  + `merged_taxonomy.csv`, `-t output/genus_tree/asv/merged_tree.nwk`,
  `-m data/final/metadata.csv --covariates HRT_d,T_C,Q_total_Tpy` (upstream
  operating variables; the script already refuses `eff_*`/`warning_*`
  covariates by default, so H2S cannot leak into its own edge-selection).
- Genus-tree: same inputs from `output/genus_tree/genus_autorun/merged_*`.

### 5a. DeepPhylo (embedding-based) — with AND without cross-domain

New `Phylospec/multi_models/DeepPhylo/deepphylo_regression_cv.py`:
subprocess-contract sibling of
`deepphylo_classification_multi_label_bce.py` (reads `--data_dir
--val_dir --scores-out-dir`, writes `val_scores.npy`/`test_scores.npy`),
built on the existing `DeepPhylo_regression` model
(`deepphylo/model.py`) and its `SmoothL1Loss(beta=0.8)` from
`deepphylo_regression.py`. New `--embedding <path>` argument accepts a
precomputed `phy_embedding.npy` (the cross-domain fused or tree-only
array from step 5) instead of deriving one from `c.npy` via
`reducer(...)` — this is the hook that makes "with/without cross-domain"
a one-flag difference for this model.

### 5b. MetaDR and PhyloSpec (topology-based) — tree-only, no cross-domain split

Both traverse the Newick tree's hierarchy directly (MetaDR:
level-order/post-order leaf sequence for its pseudo-image; PhyloSpec:
parent-child convolution order via `get_conv_order`). Neither consumes a
distance matrix or embedding, and the cross-domain fusion is deliberately
expressed in distance space specifically because a Bacteria-Archaea
shortcut is not a valid tree edge. Per decision: **run these two
tree-only, one config per tree level, no cross-domain ablation** — adding
a topology-surgery mechanism (grafting shortcut edges into the Newick) is
out of scope here.

Both need a new regression branch added:
- `MetaDR-multilabel.py` -> new `MetaDR-regression.py`: `SimpleCNN`
  `output_dim=1`, no sigmoid at inference, `SmoothL1Loss(beta=0.8)`
  replaces `BCEWithLogitsLoss`, same level-order/post-order dual-branch
  averaging.
- `PhyloSpec_train_test.py` / `training_evaluating.py`: new `--regression`
  flag that sets `num_classes=1`, `criterion=SmoothL1Loss(beta=0.8)`,
  skips the `sigmoid`/`roc_auc` branches in `training_evaluating.py`
  (currently only classification-shaped), returns raw predictions.

### Fold construction

New `build_site_grouped_cv_folds_h2s.py`: one shared set of 25
(rep, fold) train/val/test splits over the merged-domain H2S sample set
(same 137 samples regardless of ASV vs genus level — tree-matching only
filters *features*, not samples), quantile-stratified site-grouped via
`evaluate_regression.site_grouped_cv_regression`. Writes both the CSV
form (for MetaDR/PhyloSpec) and the X/Y npy form (for DeepPhylo) per
fold, mirroring `build_site_grouped_cv_folds.py`'s two output shapes.

New `run_treelevel_h2s_benchmark.py`: for each of {ASV, Genus}:
- DeepPhylo x {cross-domain embedding, tree-only embedding} — 2 runs
- MetaDR — 1 run (tree-only)
- PhyloSpec — 1 run (tree-only)

= 8 runs total (not 4), each x 25 folds, through `evaluate_cv_regression`.
Output: `results_<treelevel>_<model>[_<crossdomain|treeonly>].csv` in
`Phylospec/multi_models/results/site_grouped_cv_h2s_tree/`.

## 6. Output layout

```
Phylospec/multi_models/results/
  site_grouped_cv_h2s_taxlevel/
    results_<Level>_<Domain>_<Model>.csv   # RMSE/MAE/R2/Spearman + 95% CI
  site_grouped_cv_h2s_tree/
    results_<ASV|Genus>_DeepPhylo_<crossdomain|treeonly>.csv
    results_<ASV|Genus>_MetaDR.csv
    results_<ASV|Genus>_PhyloSpec.csv
```

## 7. Smoke test before the full grid

Before launching the full grid, verify end-to-end correctness cheaply:
- One TaxLevel config (Genus x merged x RF-regression), 1 repeat / 2
  folds: check finite RMSE/MAE/R²/Spearman, correct sample counts.
- One tree config (Genus-tree x DeepPhylo x cross-domain), 1 repeat / 2
  folds: check the embedding loads, training converges without NaN loss,
  scores write out with the right shape.

Then launch the full grid: 5 levels x 3 domains x 6 models = 90 TaxLevel
configs, and 2 tree levels x 4 tree-level models (DeepPhylo x2 +
MetaDR + PhyloSpec) = 8 tree configs, each x 25 site-grouped folds.

## 8. Out of scope

- Grafting cross-domain shortcuts into MetaDR/PhyloSpec's tree topology.
- CNN and PMCNN (neither is tree-aware; not named in the request).
- Re-deriving covariates beyond HRT_d/T_C/Q_total_Tpy without user
  confirmation.
