"""Read the completed ASV run snapshot and extend Experiment 1 reporting.

Imported by generate_results.py; no estimator imports or model fitting.
"""
from __future__ import annotations


# Locate the repository independently of the script's directory depth.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_REPO_ROOT = next(p for p in _RepositoryPath(__file__).resolve().parents
                  if (p / "main.py").is_file() and (p / "configs").is_dir())
if str(_REPO_ROOT) not in _repository_sys.path:
    _repository_sys.path.insert(0, str(_REPO_ROOT))


import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ASV_DIR = "real_results_asv_exp1_20260922"


def build(data, out, genus, api):
    base = data / ASV_DIR
    if not base.exists():
        return None
    manifest = json.loads((base / "run_manifest.json").read_text())
    assert manifest["resolution"] == "asv" and manifest["completed_tasks"] == 12
    for filename, expected in manifest["output_hashes"].items():
        assert hashlib.sha256((base / filename).read_bytes()).hexdigest() == expected
    rows = pd.read_csv(base / "rows.csv")
    pred = pd.read_csv(base / "predictions.csv")
    splits = pd.read_csv(base / "splits.csv")
    filters = pd.read_csv(base / "filter_audit.csv")
    paths = pd.read_csv(base / "regularization_paths.csv")
    groups = pd.read_csv(base / "domain_counts.csv")
    failures = pd.read_csv(base / "failures.csv")
    assert len(rows) == 192 and rows.status.eq("ok").all() and rows.coverage.eq(1).all()
    assert np.isfinite(rows[["test_logloss", "maximum_residual", "group_reconstruction_error", "equality_residual"]]).all().all()
    assert rows.group_reconstruction_error.max() < 1e-8
    assert rows.equality_residual.max() <= 1e-6
    assert filters.tree_uncovered_retained.eq(0).all()
    assert pred.p.between(0, 1).all() and pred.y.isin([0, 1]).all()
    assert not pred.duplicated(["target", "repeat", "model", "selection", "sample_id"]).any()
    assert len(paths) == 324 and paths.coverage.eq(1).all()
    original_splits = pd.read_csv(data / "investigation_20260921/learning_qc/splits.csv")
    original_splits = original_splits[original_splits.fraction.eq(1)]
    old_labels = pd.read_csv(data / "investigation_20260921/learning_qc/predictions.csv")
    old_labels = old_labels.drop_duplicates(["target", "sample_id"])[["target", "sample_id", "site_id", "y"]]
    checked_labels = pred.merge(old_labels, on=["target", "sample_id"], suffixes=("", "_old"), validate="many_to_one")
    assert checked_labels.y.eq(checked_labels.y_old).all() and checked_labels.site_id.eq(checked_labels.site_id_old).all()
    for key, d in splits.groupby(["target", "repeat", "outer_fold"]):
        old = original_splits[(original_splits.target == key[0]) & (original_splits.repeat == key[1]) & (original_splits.outer_fold == key[2])]
        for role in ["train", "test"]:
            assert set(d[d.role.eq(role)].sample_id) == set(old[old.role.eq(role)].sample_id)
        test_sites = set(d[d.role.eq("test")].site_id)
        assert test_sites.isdisjoint(d[d.role.eq("train")].site_id)
        for k in range(2):
            tr, va = d[d.role.eq(f"inner{k}_train")], d[d.role.eq(f"inner{k}_validation")]
            assert set(tr.site_id).isdisjoint(va.site_id)
            assert test_sites.isdisjoint(tr.site_id) and test_sites.isdisjoint(va.site_id)
        pp = pred[(pred.target == key[0]) & (pred.repeat == key[1]) & (pred.outer_fold == key[2])]
        for _, group in pp.groupby(["model", "selection"]):
            assert set(group.sample_id) == set(d[d.role.eq("test")].sample_id)
    scored = []
    for key, d in pred.groupby(api.PRED_KEYS):
        assert len(d) == 134 and d.site_id.nunique() == 35 and d.outer_fold.nunique() == 3
        scored.append({**dict(zip(api.PRED_KEYS, key)), **api.metrics(d)})
    scored = pd.DataFrame(scored)
    summary = []
    for key, d in rows.groupby(["target", "model", "selection"]):
        target, model, selection = key
        m = scored[(scored.target == target) & (scored.model == model) & (scored.selection == selection)]
        assert len(d) == 6 and len(m) == 2
        summary.append(dict(target=target, model=model, selection_tag=selection, resolution="asv", n_fits=len(d), n_repeats=len(m),
            retained_features_mean=d.retained_taxa.mean(), retained_arc_mean=d.retained_arc.mean(), retained_bac_mean=d.retained_bac.mean(),
            active_taxa_mean=d.n_active_taxa.mean(), active_groups_mean=d.n_active_groups.mean(),
            active_products_mean=d.n_active_products.mean(), structural_compression_pct=100 * (1 - d.n_active_groups / d.retained_taxa).mean(),
            maximum_residual=d.maximum_residual.max(), group_reconstruction_error=d.group_reconstruction_error.max(),
            **{metric: m[metric].mean() for metric in api.METRICS}))
    summary = pd.DataFrame(summary)
    primary = summary[summary.selection_tag.eq("one_se")].copy()
    canonical = lambda name: {"asv_ridge": "genus_ridge", "asv_elastic_net": "genus_elastic_net"}.get(name, name)
    primary["comparison_family"] = primary.model.map(canonical)
    g = genus[genus.model.isin(api.GROUP_MODELS)].copy()
    g["comparison_family"] = g.model
    columns = ["target", "comparison_family", "model", "retained_features_mean", "active_groups_mean", "active_products_mean", "structural_compression_pct", "logloss", "auc", "ap"]
    comparison = primary[columns].merge(g[columns], on=["target", "comparison_family"], suffixes=("_asv", "_genus"), validate="one_to_one")
    comparison["delta_logloss_asv_minus_genus"] = comparison.logloss_asv - comparison.logloss_genus
    comparison["delta_auc_asv_minus_genus"] = comparison.auc_asv - comparison.auc_genus

    path_summary = paths.groupby(["target", "graph", "lambda_f"], as_index=False).agg(
        n_fits=("outer_fold", "size"), retained_asvs_mean=("retained_taxa", "mean"), active_groups_mean=("n_active_groups", "mean"),
        test_logloss_mean=("test_logloss", "mean"), test_logloss_sd=("test_logloss", "std"), test_auc_fold_mean=("test_auc", "mean"))
    audit = filters.groupby(["target", "domain"], as_index=False).agg(input_asvs=("input_asvs", "first"),
        retained_asvs_mean=("retained_asvs", "mean"), retained_asvs_min=("retained_asvs", "min"), retained_asvs_max=("retained_asvs", "max"),
        mean_training_retained_mass=("mean_retained_mass", "mean"), min_training_retained_mass=("min_retained_mass", "min"))
    shared = groups[(groups.model == "shared_hybrid") & groups.selection.eq("one_se")].groupby(["target", "domain"], as_index=False).active_groups.mean()
    audit = audit.merge(shared.rename(columns={"active_groups": "shared_hybrid_active_groups_mean"}), on=["target", "domain"], validate="one_to_one")
    for df, name in [(rows, "asv_full_size_fits.csv"), (scored, "asv_repeat_metrics.csv"), (summary, "asv_all_selection_rules.csv"),
        (primary, "asv_primary_comparison.csv"), (comparison, "asv_vs_genus_qc.csv"), (path_summary, "asv_regularization_summary.csv"), (audit, "asv_filtering_and_compression.csv")]:
        api.save_table(df, out, name)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for i, target in enumerate(api.TARGETS):
        for graph, color in api.COLORS.items():
            d = path_summary[(path_summary.target == target) & path_summary.graph.eq(graph)].sort_values("lambda_f")
            for j, metric in enumerate(["test_logloss_mean", "test_auc_fold_mean"]):
                axes[i, j].plot(d.active_groups_mean, d[metric], "o-", color=color, label=graph.title(), ms=5)
        for ax in axes[i]:
            ax.set_title(api.TITLES[target]); ax.set_xlabel("Mean active ASV coefficient groups (6 fits)"); ax.grid(alpha=.18)
        axes[i, 0].set_ylabel("Mean held-out site log-loss ↓")
        axes[i, 1].set_ylabel("Mean fold AUROC ↑ (not pooled)")
    axes[0, 0].legend()
    fig.suptitle("ASV-tree compression paths: QC-corrected 35-site-group protocol\nActual ARC/BAC rooted trees; fixed ridge=0.1 and sparsity=0", y=1.01)
    fig.tight_layout(); api.save_figure(fig, out, "05_asv_regularization_tradeoff")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    yy = np.arange(len(api.GROUP_MODELS))
    for i, target in enumerate(api.TARGETS):
        d = comparison[comparison.target.eq(target)].set_index("comparison_family").loc[api.GROUP_MODELS]
        for j, metric in enumerate(["active_groups_mean", "logloss"]):
            axes[i, j].barh(yy - .18, d[metric + "_genus"], height=.34, color="#7b9eae", label="Genus input / tree")
            axes[i, j].barh(yy + .18, d[metric + "_asv"], height=.34, color="#c97732", label="ASV input / tree")
            axes[i, j].set_yticks(yy, [api.LABELS[m] for m in api.GROUP_MODELS]); axes[i, j].invert_yaxis()
            axes[i, j].set_title(api.TITLES[target]); axes[i, j].grid(axis="x", alpha=.18)
        axes[i, 0].set_xlabel("Mean active coefficient groups; products reported separately")
        axes[i, 1].set_xlabel("Mean repeat-pooled held-out site log-loss ↓")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Genus versus ASV: saved one-SE selections on the same outer site splits\nInput aggregation and retained feature sets differ; this is not an isolated tree-resolution effect", y=1.01)
    fig.tight_layout(); api.save_figure(fig, out, "06_asv_vs_genus")

    checks = {"completed_tasks": 12, "selected_fit_rows": len(rows), "prediction_rows": len(pred), "regularization_rows": len(paths),
        "same_cohort_labels_and_outer_splits_as_genus": True, "retained_asvs_missing_from_trees": 0,
        "maximum_equality_residual": float(rows.equality_residual.max()), "maximum_group_reconstruction_error": float(rows.group_reconstruction_error.max()),
        "maximum_coefficient_residual": float(rows.maximum_residual.max()), "failed_candidate_records": len(failures),
        "active_product_counts": sorted(rows.n_active_products.unique().tolist()),
        "input_tree_hashes": {p: h for p, h in manifest["hashes"].items() if p.endswith("rooted-tree.nwk")}}
    (out / "asv_validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    lines = []
    for target in api.TARGETS:
        d = primary[primary.target.eq(target)].set_index("model")
        ridge, shared_row, adaptive = d.loc["asv_ridge"], d.loc["shared_hybrid"], d.loc["domain_adaptive_hybrid"]
        lines.append(f"- **{api.TITLES[target]}:** {ridge.retained_features_mean:.1f} ASVs remain after filtering on average. Shared hybrid uses {shared_row.active_groups_mean:.1f} active groups ({shared_row.structural_compression_pct:.1f}% additional reduction) with log-loss {shared_row.logloss:.3f}, versus ASV ridge {ridge.logloss:.3f}. Adaptive hybrid uses {adaptive.active_groups_mean:.1f} groups and has loss {adaptive.logloss:.3f}.")
    tables = []
    for target in api.TARGETS:
        d = primary[primary.target.eq(target)][["model", "retained_features_mean", "active_groups_mean", "active_products_mean", "structural_compression_pct", "logloss", "auc", "ap"]]
        tables.append(f"### {api.TITLES[target]} — ASV-level fusion\n\n" + api.md(d.rename(columns={"model": "Model", "retained_features_mean": "Retained ASVs", "active_groups_mean": "Active groups", "active_products_mean": "Active products", "structural_compression_pct": "Reduction (%)", "logloss": "Log-loss", "auc": "AUROC", "ap": "AP"})))
    section = f"""## New ASV-level tree experiment

This section contains **newly fitted fusionML models**, added after the original saved-output reanalysis. The run uses `data/qiimeresult/ARC/rooted-tree.nwk` and `data/qiimeresult/BAC/rooted-tree.nwk` directly with ASV counts; no genus tree or genus aggregation enters these fits. Original outputs and configuration are in `../data/{ASV_DIR}/`.

{chr(10).join(lines)}

The 134 samples, 35 QC-adjusted site groups, outcome labels, three outer folds, two repeats and two inner folds match the genus comparison. The candidate and split generation rules are retained, including the limited main grids and separate expanded adaptive grid. Both one-SE and minimum-inner-loss selections are exported. The trees and current code/configuration/input hashes are recorded in the new run manifest. The historical genus fitting-version limitation still applies to the reused comparator.

{(chr(10) * 2).join(tables)}

![Genus versus ASV](figures/06_asv_vs_genus.png)

This comparison changes the input aggregation and which features pass prevalence filtering as well as the tree resolution. It cannot isolate a causal effect of using an ASV tree. Ridge/elastic-net models use ASV abundances but no phylogenetic penalty; the phylogenetic and hybrid candidates use the actual supplied trees. Candidate failures are recorded rather than replaced by substitute predictions. All selected outer fits have complete coverage.

### Filtering versus fusion

{api.md(audit.rename(columns={"target": "Target", "domain": "Domain", "input_asvs": "Input ASVs", "retained_asvs_mean": "Retained mean", "retained_asvs_min": "Retained min", "retained_asvs_max": "Retained max", "mean_training_retained_mass": "Mean retained mass", "min_training_retained_mass": "Min retained mass", "shared_hybrid_active_groups_mean": "Shared active groups"}))}

There are 1,768 archaeal and 13,385 bacterial input ASVs (15,153 total). The inherited 40% prevalence filter removes most before model fitting. Report **input ASVs → retained ASVs → active groups**; the entire raw-to-group reduction is not attributable to fusion. Retained mass is a fraction of training-sample abundance within each separately measured domain. The current run tests this filtered ASV representation, not unfiltered modeling of all 15,153 ASVs.

### ASV compression paths

![ASV compression paths](figures/05_asv_regularization_tradeoff.png)

These new ASV paths use the QC-corrected protocol: 324 evaluations across two targets, two repeats, three folds, three graph types, and nine strengths. Ridge=0.1 and sparsity=0 are fixed. The graph types are phylogeny, abundance, and a shared 50:50 mixture. Curves show averages of fold scores; the selected-model table instead uses repeat-pooled metrics. These curves are descriptive and do not select a deployment strength from test scores. They should not be pooled with the old genus paths, which used the original 37-site convention.

### ASV validation and reproducibility

All 192 selected fit records and 324 path evaluations completed. The selected fits have maximum zero-sum constraint residual {checks['maximum_equality_residual']:.3g} and group-plus-residual reconstruction error {checks['maximum_group_reconstruction_error']:.3g}. All retained ASVs are covered by the specified trees. There are {checks['failed_candidate_records']} recorded candidate failures; see the copied `failures.csv`. Product counts are reported separately from group counts, and group-only lossless prediction is not asserted when residual coefficients remain. Full coefficients and graphs are saved for future interpretation.

To refit, run `scripts/run_asv_fusion.py --workers 3`; it preserves old runs and writes a separate checkpointed run under `fusionML/real_results_asv_exp1_20260922`. To copy completed results, run `scripts/copy_inputs.py --include-asv`, then `scripts/generate_results.py`. Report regeneration uses only the copied outputs and does not refit models. New ASV tables use the `asv_` prefix; the primary comparison includes the same saved one-SE rule, and both policies are available in `asv_all_selection_rules.csv`.

"""
    return {"section": section, "findings": "\n".join(lines), "validation": checks}
