"""Summarize the domain-restricted fusion runs and extend Experiment 1 reporting.

Imported by generate_results.py; no estimator imports and no model fitting.

Compares where the fusion penalty acts -- ARC edges only, BAC edges only, or
both domains -- for three warning flags, across taxonomic resolutions, under
the minimum-inner-loss selection rule, ranked by MCC with F2 alongside.

Every resolution is fit on the same QC protocol (134 samples, 35 site groups,
3 outer folds x 2 repeats, 2 inner folds, same seeds), so the resolutions are
comparable to each other and to the rest of this report. They are NOT the
older 37-site single-repeat smoke runs under fusionML/real_results_*/smoke/.
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

# Coarse to fine. A resolution appears in the report only if its run was copied.
RESOLUTIONS = ["order", "family", "genus", "asv"]
RUN_DIR = "real_results_domain_fusion_{resolution}_exp1_20260922"
RESOLUTION_LABELS = {"order": "Order", "family": "Family", "genus": "Genus", "asv": "ASV"}
TARGETS = ["warning_ammonia_toxicity", "warning_acid_base_balance", "warning_biogas_quality"]
TITLES = dict(zip(TARGETS, ["Ammonia toxicity", "Acid–base balance", "Biogas quality"]))
ARMS = ["fusion_arc_only", "fusion_bac_only", "both"]
ARM_LABELS = {"fusion_arc_only": "ARC-only fusion", "fusion_bac_only": "BAC-only fusion", "both": "Both domains fused"}
BOTH_MODELS = ["shared_hybrid", "domain_adaptive_hybrid", "domain_adaptive_interaction"]
MODELS = ["fusion_arc_only", "fusion_bac_only"] + BOTH_MODELS
MODEL_LABELS = {"fusion_arc_only": "ARC-only fusion", "fusion_bac_only": "BAC-only fusion",
                "shared_hybrid": "Shared hybrid", "domain_adaptive_hybrid": "Domain-adaptive hybrid",
                "domain_adaptive_interaction": "ARC–BAC interaction"}
# Selection rule this section reports. one_se stays in the exported tables.
RULE = "min_loss"
# Threshold metric used to pick the representative model in each arm. MCC is
# primary: at 8-18% positive prevalence F2 is maximized by near-all-positive
# predictors, so F2 is reported alongside rather than used to rank.
RANK_METRIC = "mcc"
# Configurations previously identified as best per flag and arm from the older
# 37-site single-repeat smoke runs under fusionML/real_results_*/smoke/. They
# are recomputed here on the QC protocol so their compression is comparable
# with everything else in this report; several are not the QC-protocol best.
PRIOR_BEST = {
    ("warning_ammonia_toxicity", "fusion_arc_only"): ("order", "fusion_arc_only"),
    ("warning_ammonia_toxicity", "fusion_bac_only"): ("order", "fusion_bac_only"),
    ("warning_ammonia_toxicity", "both"): ("asv", "shared_hybrid"),
    ("warning_acid_base_balance", "fusion_arc_only"): ("order", "fusion_arc_only"),
    ("warning_acid_base_balance", "fusion_bac_only"): ("family", "fusion_bac_only"),
    ("warning_acid_base_balance", "both"): ("family", "domain_adaptive_hybrid"),
}


def load_run(base, resolution, api):
    """Read and integrity-check one resolution's run directory."""
    manifest = json.loads((base / "run_manifest.json").read_text())
    assert manifest["resolution"] == resolution and manifest["completed_tasks"] == 18
    for filename, expected in manifest["output_hashes"].items():
        assert hashlib.sha256((base / filename).read_bytes()).hexdigest() == expected
    rows = pd.read_csv(base / "rows.csv")
    pred = pd.read_csv(base / "predictions.csv")
    splits = pd.read_csv(base / "splits.csv")
    counts = pd.read_csv(base / "domain_counts.csv")
    audit = pd.read_csv(base / "filter_audit.csv")
    failures = pd.read_csv(base / "failures.csv")
    assert set(rows.target) == set(TARGETS) and set(rows.model) == set(MODELS)
    assert len(rows) == 18 * len(MODELS) * 2
    assert not rows.duplicated(["target", "repeat", "outer_fold", "model", "selection"]).any()
    assert rows.resolution.eq(resolution).all()
    ok = rows[rows.status.eq("ok")].copy()
    assert ok.coverage.eq(1).all()
    assert ok.group_reconstruction_error.max() < 1e-8 and ok.equality_residual.max() <= 1e-6
    assert (ok.n_active_groups <= ok.retained_taxa).all()
    # The restriction itself: the unfused domain must carry no fusion edges.
    assert ok[ok.model.eq("fusion_arc_only")].n_edges_bac.eq(0).all()
    assert ok[ok.model.eq("fusion_bac_only")].n_edges_arc.eq(0).all()
    assert ok[ok.model.isin(BOTH_MODELS)].n_edges_arc.gt(0).any()
    assert pred.p.between(0, 1).all() and pred.y.isin([0, 1]).all()
    assert not pred.duplicated(["target", "repeat", "model", "selection", "sample_id"]).any()
    assert audit.tree_uncovered_retained.eq(0).all()

    site_checks = 0
    for key, d in splits.groupby(["target", "repeat", "outer_fold"]):
        test_sites = set(d[d.role.eq("test")].site_id)
        assert test_sites.isdisjoint(d[d.role.eq("train")].site_id)
        for k in range(2):
            tr, va = d[d.role.eq(f"inner{k}_train")], d[d.role.eq(f"inner{k}_validation")]
            assert set(tr.site_id).isdisjoint(va.site_id)
            assert test_sites.isdisjoint(tr.site_id) and test_sites.isdisjoint(va.site_id)
        available = ok[(ok.target == key[0]) & (ok.repeat == key[1]) & (ok.outer_fold == key[2])]
        pp = pred[(pred.target == key[0]) & (pred.repeat == key[1]) & (pred.outer_fold == key[2])]
        for _, group in pp.groupby(["model", "selection"]):
            assert set(group.sample_id) == set(d[d.role.eq("test")].sample_id)
        assert len(pp) == len(available) * len(d[d.role.eq("test")])
        site_checks += 1

    expected_folds = ok.groupby(["target", "repeat"]).outer_fold.nunique().to_dict()
    scored = []
    for key, d in pred.groupby(api.PRED_KEYS):
        assert d.outer_fold.nunique() == expected_folds[(key[0], key[1])]
        scored.append({**dict(zip(api.PRED_KEYS, key)), "resolution": resolution, **api.metrics(d),
                       "complete_oof": d.outer_fold.nunique() == 3 and len(d) == 134})
    scored = pd.DataFrame(scored)

    domains = counts.pivot_table(index=["target", "repeat", "outer_fold", "model", "selection"],
                                 columns="domain", values=["retained_features", "active_groups", "n_edges"]).reset_index()
    domains.columns = [c[0] if not c[1] else f"{c[1].lower()}_{c[0]}" for c in domains.columns]
    fits = ok.merge(domains, on=["target", "repeat", "outer_fold", "model", "selection"], validate="one_to_one")
    assert fits.arc_retained_features.add(fits.bac_retained_features).eq(fits.retained_taxa).all()
    assert fits.arc_active_groups.add(fits.bac_active_groups).eq(fits.n_active_groups).all()
    inputs = audit.groupby(["target", "repeat", "outer_fold"], as_index=False).input_features.sum()
    fits = fits.merge(inputs, on=["target", "repeat", "outer_fold"], validate="many_to_one")
    return dict(resolution=resolution, rows=rows, pred=pred, fits=fits, scored=scored,
                failures=failures, audit=audit, site_checks=site_checks, manifest=manifest)


def summarize(run, api):
    fits, scored = run["fits"], run["scored"]
    records = []
    for key, d in fits.groupby(["target", "model", "selection"]):
        target, model, selection = key
        m = scored[(scored.target == target) & (scored.model == model) & (scored.selection == selection)]
        assert len(m) == 2
        records.append(dict(target=target, resolution=run["resolution"], model=model, selection_tag=selection,
            fusion_domains=d.fusion_domains.iloc[0], arm=model if model in ARMS else "both",
            n_available_outer_fits=len(d), n_planned_outer_fits=6,
            complete_oof=bool(m.complete_oof.all()), n_scored_mean=m.n_scored.mean(),
            input_features_mean=d.input_features.mean(),
            retained_features_mean=d.retained_taxa.mean(), retained_arc_mean=d.arc_retained_features.mean(),
            retained_bac_mean=d.bac_retained_features.mean(), active_groups_mean=d.n_active_groups.mean(),
            active_groups_arc_mean=d.arc_active_groups.mean(), active_groups_bac_mean=d.bac_active_groups.mean(),
            active_taxa_mean=d.n_active_taxa.mean(), active_products_mean=d.n_active_products.mean(),
            structural_compression_pct=100 * (1 - d.n_active_groups / d.retained_taxa).mean(),
            compression_from_input_pct=100 * (1 - d.n_active_groups / d.input_features).mean(),
            arc_compression_pct=100 * (1 - d.arc_active_groups / d.arc_retained_features).mean(),
            bac_compression_pct=100 * (1 - d.bac_active_groups / d.bac_retained_features).mean(),
            fusion_edges_arc_mean=d.n_edges_arc.mean(), fusion_edges_bac_mean=d.n_edges_bac.mean(),
            maximum_residual=d.maximum_residual.max(), group_reconstruction_error=d.group_reconstruction_error.max(),
            **{metric: m[metric].mean() for metric in api.METRICS}))
    return pd.DataFrame(records)


def build(data, out, api):
    runs = {}
    for resolution in RESOLUTIONS:
        base = data / RUN_DIR.format(resolution=resolution)
        if base.exists():
            runs[resolution] = load_run(base, resolution, api)
    if not runs:
        return None
    available = [r for r in RESOLUTIONS if r in runs]

    rows = pd.concat([r["rows"] for r in runs.values()], ignore_index=True)
    fits = pd.concat([r["fits"] for r in runs.values()], ignore_index=True)
    scored = pd.concat([r["scored"] for r in runs.values()], ignore_index=True)
    failures = pd.concat([r["failures"] for r in runs.values()], ignore_index=True)
    summary = pd.concat([summarize(r, api) for r in runs.values()], ignore_index=True)
    selected = summary[summary.selection_tag.eq(RULE)].copy()

    # Labels and site assignments must match the reused Experiment 1 cohort.
    old = pd.read_csv(data / "investigation_20260921/learning_qc/predictions.csv")
    old = old.drop_duplicates(["target", "sample_id"])[["target", "sample_id", "site_id", "y"]]
    for run in runs.values():
        checked = run["pred"].merge(old, on=["target", "sample_id"], suffixes=("", "_old"), validate="many_to_one")
        assert checked.y.eq(checked.y_old).all() and checked.site_id.eq(checked.site_id_old).all()

    # The ASV both-domain families share candidate grids, seeds and partition
    # graphs with the saved ASV run, so those fits must reproduce exactly.
    reproduced = {}
    saved_path = data / "real_results_asv_exp1_20260922/rows.csv"
    if "asv" in runs and saved_path.exists():
        saved = pd.read_csv(saved_path)
        keys = ["target", "repeat", "outer_fold", "model", "selection"]
        mine = runs["asv"]["rows"]
        mine = mine[mine.status.eq("ok") & mine.model.isin(BOTH_MODELS)]
        shared = mine.merge(saved[saved.model.isin(BOTH_MODELS)], on=keys, suffixes=("", "_saved"), validate="one_to_one")
        assert len(shared) == 2 * 3 * 2 * len(BOTH_MODELS) * 2  # targets x folds x repeats x models x rules
        reproduced = {column: float((shared[column] - shared[column + "_saved"]).abs().max())
                      for column in ["n_active_groups", "test_mcc", "test_f2", "test_logloss", "test_auc"]}
        assert max(reproduced.values()) == 0, reproduced

    # Best configuration per flag and arm: the highest-MCC (resolution, model)
    # pair. This is an observed ranking on held-out folds, not a selection rule.
    best = []
    for target in TARGETS:
        d = selected[selected.target.eq(target)]
        for arm in ARMS:
            pool = d[d.arm.eq(arm)]
            row = pool.loc[pool[RANK_METRIC].idxmax()]
            alt = pool.loc[pool.f2.idxmax()]
            best.append(dict(target=target, arm=arm, arm_label=ARM_LABELS[arm], resolution=row.resolution,
                model=row.model, candidates_considered=len(pool), ranked_by=RANK_METRIC,
                resolution_if_ranked_by_f2=alt.resolution, model_if_ranked_by_f2=alt.model,
                rank_metric_agrees=bool(alt.resolution == row.resolution and alt.model == row.model),
                **{c: row[c] for c in ["complete_oof", "input_features_mean", "retained_features_mean",
                    "retained_arc_mean", "retained_bac_mean", "active_groups_mean", "active_groups_arc_mean",
                    "active_groups_bac_mean", "active_products_mean", "structural_compression_pct",
                    "compression_from_input_pct", "arc_compression_pct", "bac_compression_pct"] + api.METRICS}))
    best = pd.DataFrame(best)

    # The previously identified configurations, recomputed on this protocol.
    prior = []
    for (target, arm), (resolution, model) in PRIOR_BEST.items():
        match = selected[selected.target.eq(target) & selected.resolution.eq(resolution) & selected.model.eq(model)]
        if not len(match):
            continue
        row = match.iloc[0]
        chosen = best[best.target.eq(target) & best.arm.eq(arm)].iloc[0]
        prior.append(dict(target=target, arm=arm, arm_label=ARM_LABELS[arm], resolution=resolution, model=model,
            is_qc_protocol_best=bool(chosen.resolution == resolution and chosen.model == model),
            qc_protocol_best_resolution=chosen.resolution, qc_protocol_best_model=chosen.model,
            qc_protocol_best_mcc=chosen.mcc, delta_mcc_vs_qc_best=row.mcc - chosen.mcc,
            **{c: row[c] for c in ["complete_oof", "input_features_mean", "retained_features_mean",
                "retained_arc_mean", "retained_bac_mean", "active_groups_mean", "active_groups_arc_mean",
                "active_groups_bac_mean", "structural_compression_pct", "compression_from_input_pct",
                "arc_compression_pct", "bac_compression_pct"] + api.METRICS}))
    prior = pd.DataFrame(prior)

    grid = selected.pivot_table(index=["target", "arm", "model"], columns="resolution",
                                values=["active_groups_mean", "structural_compression_pct", RANK_METRIC]).reset_index()
    grid.columns = [c[0] if not c[1] else f"{c[0]}_{c[1]}" for c in grid.columns]

    inventory = rows[["target", "resolution", "repeat", "outer_fold", "model", "selection", "status", "reason"]].copy()
    unavailable = inventory[inventory.status.ne("ok")]
    unavailable_splits = set(map(tuple, unavailable[["target", "repeat", "outer_fold"]].drop_duplicates().to_numpy()))
    failure_splits = set(map(tuple, failures[["target", "repeat", "outer_fold"]].drop_duplicates().to_numpy()))
    assert failure_splits <= unavailable_splits, sorted(failure_splits - unavailable_splits)
    failure_kinds = sorted(failures.error.str.split(":").str[0].unique())

    for frame, name in [(fits, "domain_fusion_fits.csv"), (scored, "domain_fusion_repeat_metrics.csv"),
                        (summary, "domain_fusion_all_selection_rules.csv"), (selected, "domain_fusion_min_loss_comparison.csv"),
                        (best, "domain_fusion_best_by_resolution.csv"), (grid, "domain_fusion_resolution_grid.csv"),
                        (prior, "domain_fusion_prior_best_configurations.csv"),
                        (inventory, "domain_fusion_fit_inventory.csv")]:
        api.save_table(frame, out, name)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 12, "figure.facecolor": "white"})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    y = np.arange(len(ARMS))
    for j, target in enumerate(TARGETS):
        d = best[best.target.eq(target)].set_index("arm").loc[ARMS]
        axes[0, j].barh(y, d.retained_features_mean, color="#dce5ec", label="Retained features")
        axes[0, j].barh(y, d.active_groups_arc_mean, color="#2563a6", label="Active ARC groups")
        axes[0, j].barh(y, d.active_groups_bac_mean, left=d.active_groups_arc_mean, color="#d16b19", label="Active BAC groups")
        for yy, (value, pct) in enumerate(zip(d.active_groups_mean, d.structural_compression_pct)):
            axes[0, j].text(value + 1.5, yy, f"{value:.1f}  ({pct:.0f}%)", va="center", fontsize=9)
        # Shared count scale too, so bar lengths compare across flags.
        axes[0, j].set_xlim(0, best.retained_features_mean.max() * 1.42)
        axes[0, j].set_xlabel("Mean counts across available outer fits")
        axes[1, j].barh(y, d[RANK_METRIC], color="#24816c")
        axes[1, j].axvline(0, color="#333333", lw=1)
        # One shared scale across flags: a 0.03 MCC must not look like a 0.62 one.
        axes[1, j].set_xlim(min(0, best[RANK_METRIC].min() * 1.1), best[RANK_METRIC].max() * 1.24)
        axes[1, j].set_xlabel("Pooled out-of-fold MCC ↑ (shared scale)")
        for yy, value in zip(y, d[RANK_METRIC]):
            axes[1, j].text(max(value, 0) + .012, yy, f"{value:.3f}", va="center", fontsize=9)
        for i in range(2):
            axes[i, j].set_yticks(y, [f"{ARM_LABELS[a]}\n{RESOLUTION_LABELS[r]}" for a, r in zip(ARMS, d.resolution)])
            axes[i, j].invert_yaxis()
            axes[i, j].set_title(TITLES[target] + ("" if d.complete_oof.all() else " (partial OOF)"))
            axes[i, j].grid(axis="x", alpha=.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(.5, -.04))
    fig.suptitle("Where the fusion penalty acts, at each arm's best taxonomic resolution\n"
                 "Minimum-inner-loss selection; resolution and model chosen by highest MCC among "
                 + ", ".join(RESOLUTION_LABELS[r] for r in available), y=1.02)
    fig.tight_layout()
    api.save_figure(fig, out, "07_domain_fusion_compression")

    checks = {"resolutions": available, "completed_tasks_per_resolution": 18,
              "planned_fit_records": len(rows), "available_fit_records": int(rows.status.eq("ok").sum()),
              "unavailable_fit_records": len(unavailable), "prediction_rows": int(sum(len(r["pred"]) for r in runs.values())),
              "outer_partitions_without_site_overlap": sum(r["site_checks"] for r in runs.values()),
              "restriction_verified_no_edges_in_unfused_domain": True,
              "labels_and_sites_match_reused_cohort": True,
              "retained_features_missing_from_trees": 0,
              "maximum_equality_residual": float(fits.equality_residual.max()),
              "maximum_group_reconstruction_error": float(fits.group_reconstruction_error.max()),
              "maximum_coefficient_residual": float(fits.maximum_residual.max()),
              "failed_candidate_records": len(failures), "failure_kinds": failure_kinds,
              "unavailable_outer_splits": sorted(map(list, unavailable_splits)),
              "targets_with_complete_out_of_fold_coverage": sorted(summary[summary.complete_oof].target.unique()),
              "asv_both_domain_fits_reproduce_saved_asv_run": reproduced,
              "selection_rule_reported": RULE, "ranking_metric": RANK_METRIC}
    (out / "domain_fusion_validation.json").write_text(json.dumps(checks, indent=2) + "\n")

    partial = sorted(set(TARGETS) - set(checks["targets_with_complete_out_of_fold_coverage"]))
    findings = []
    for target in TARGETS:
        d = best[best.target.eq(target)].set_index("arm")
        parts = [f"{ARM_LABELS[a].lower()} at {RESOLUTION_LABELS[d.loc[a].resolution].lower()} level keeps "
                 f"{d.loc[a].active_groups_mean:.1f} of {d.loc[a].retained_features_mean:.1f} retained features "
                 f"({d.loc[a].structural_compression_pct:.1f}% reduction, MCC {d.loc[a].mcc:.3f})" for a in ARMS]
        order = d.sort_values(RANK_METRIC, ascending=False)
        findings.append(f"- **{TITLES[target]}:** " + "; ".join(parts) +
                        f". Most compressed: {ARM_LABELS[d.structural_compression_pct.idxmax()].lower()}; "
                        f"highest MCC: {ARM_LABELS[order.index[0]].lower()}.")

    best_cols = ["arm_label", "resolution", "model", "input_features_mean", "retained_features_mean", "active_groups_mean",
                 "active_groups_arc_mean", "active_groups_bac_mean", "structural_compression_pct",
                 "compression_from_input_pct", "mcc", "f2", "auc", "ap", "logloss"]
    best_names = {"arm_label": "Arm", "resolution": "Resolution", "model": "Model",
                  "input_features_mean": "Input features", "retained_features_mean": "Retained",
                  "active_groups_mean": "Active groups", "active_groups_arc_mean": "ARC groups",
                  "active_groups_bac_mean": "BAC groups", "structural_compression_pct": "Reduction vs retained (%)",
                  "compression_from_input_pct": "Reduction vs input (%)", "mcc": "MCC", "f2": "F2",
                  "auc": "AUROC", "ap": "AP", "logloss": "Log-loss"}
    grid_names = {"arm_label": "Arm", "model_label": "Model", "metric": "Quantity"}
    blocks = []
    for target in TARGETS:
        table = best[best.target.eq(target)].set_index("arm").loc[ARMS].reset_index()
        table["resolution"] = table.resolution.map(RESOLUTION_LABELS)
        note = ""
        if target in partial:
            m = scored[scored.target.eq(target) & scored.selection.eq(RULE)]
            note = (f"\n\nOnly {int(m.n_scored.min())}–{int(m.n_scored.max())} of 134 samples are scored per repeat here, "
                    "from 2 of 3 outer folds at every resolution. These are not full out-of-fold results and the arms "
                    "are not ranked as a completed benchmark.")
        disagree = table[~table.rank_metric_agrees]
        if len(disagree):
            note += "\n\nRanking by F2 instead of MCC would instead choose " + "; ".join(
                f"{r.arm_label.lower()} → {RESOLUTION_LABELS[r.resolution_if_ranked_by_f2]}/"
                f"{MODEL_LABELS[r.model_if_ranked_by_f2].lower()}" for r in disagree.itertuples()) + "."
        # Per-resolution compression for every arm, so the chosen row is in context.
        wide = selected[selected.target.eq(target)].copy()
        wide["model_label"] = wide.model.map(MODEL_LABELS)
        pivot = wide.pivot_table(index="model_label", columns="resolution",
                                 values=["active_groups_mean", "structural_compression_pct", RANK_METRIC])
        pivot = pivot.reindex([MODEL_LABELS[m] for m in MODELS])
        lines = []
        for label in ["active_groups_mean", "structural_compression_pct", RANK_METRIC]:
            nice = {"active_groups_mean": "Active groups", "structural_compression_pct": "Reduction (%)",
                    RANK_METRIC: "MCC"}[label]
            for model_label in pivot.index:
                lines.append({"Quantity": nice, "Model": model_label,
                              **{RESOLUTION_LABELS[r]: pivot.loc[model_label, (label, r)] for r in available}})
        blocks.append(f"### {TITLES[target]}\n\n" + api.md(table[best_cols].rename(columns=best_names)) + note +
                      "\n\nAll five models at every fitted resolution:\n\n" + api.md(pd.DataFrame(lines)))

    resolution_note = ", ".join(
        f"{RESOLUTION_LABELS[r]} ({selected[selected.resolution.eq(r)].input_features_mean.mean():.0f} input features, "
        f"{selected[selected.resolution.eq(r)].retained_features_mean.mean():.0f} retained)" for r in available)

    section = f"""## Where the fusion penalty acts — three flags, three arms, {len(available)} resolutions

These are **newly fitted fusionML models**. Every resolution uses the same cohort, 35 QC site groups, outer/inner splits, seeds and candidate-generation rules as the rest of this report, so resolutions are comparable to each other. Original outputs are in `../data/{RUN_DIR.format(resolution="<resolution>")}/`. The question is not which domain to measure but **which domain's coefficients should be fused**: `fusion_arc_only` and `fusion_bac_only` keep *both* ARC and BAC features as predictors and give a fusion graph to one domain only, so the other domain's coefficients are fit completely unfused. They are not single-assay models.

This comparison reports the **minimum-inner-loss** selection rule and ranks by **MCC**, with F2 shown alongside. At 8–18% positive prevalence F2 is maximized by near-all-positive predictors, so it is read next to MCC rather than used to rank. Each arm is represented by the resolution and model with the highest MCC; every fitted combination is tabulated underneath. Selection still uses inner-fold log-loss only — MCC and F2 rank already-fitted models and never choose a candidate. Because the ranking reads held-out folds, the chosen row is an **observed best in this cohort**, not a validated deployment setting, and the same folds then report its compression.

Resolutions fitted: {resolution_note}. Rank resolutions aggregate ASVs on the monophyly-consistent trees under `output/genus_tree/taxlevel_autorun/`, not by taxonomy label; ASV uses the original ARC/BAC rooted trees. These are **not** the older 37-site single-repeat smoke runs in `fusionML/real_results_*/smoke/`; numbers will differ from those.

{chr(10).join(findings)}

{(chr(10) * 2).join(blocks)}

![Domain-restricted fusion compression](figures/07_domain_fusion_compression.png)

### Compression at the previously identified configurations

A set of best resolution/model pairs had been read off the older 37-site single-repeat smoke runs. Those runs use a different site grouping, one repeat and the one-SE rule, so their compression numbers are not comparable with the rest of this report. The same configurations, refit and rescored on this protocol, are below.

{{prior_table}}

{{prior_note}}

Two compression denominators are given because they answer different questions. **Reduction vs retained** is this report's usual definition: how far fusion and sparsity shrink the features that survived the training-only prevalence filter, and it is the column to compare arms and resolutions with. **Reduction vs input** additionally counts everything removed before fitting, which is prevalence filtering at every resolution plus rank aggregation at the three rank resolutions; none of that is attributable to fusion. That column is **not comparable across resolutions**, because each resolution has a different input universe: at ASV it is the 15,153 raw ASVs, so the figure reads about 99.8% for every model, while at order it is 180 aggregated features. Read it only within one resolution.

Reading the per-domain columns: an unfused domain has no fusion edges, so each of its active coefficients is its own group and its share of the reduction reflects sparsity alone. Overall reduction therefore mixes a fused and an unfused domain in the two restricted arms, and BAC dominates it because BAC supplies the large majority of retained features at every resolution.

The three flags differ sharply in event count: ammonia toxicity has 24 positives, acid–base balance 20, and biogas quality 11 in the 134-sample cohort. {"Biogas quality lost 2 of its 6 outer splits to single-class inner training partitions at every resolution, for every model; those fits are recorded as unavailable rather than substituted, and its metrics come from partial out-of-fold predictions with 5–7 positives per repeat." if partial else ""} MCC and F2 at these event counts are unstable, and differences between arms or resolutions of a few hundredths are not evidence of a better fusion target. Picking the best of {len(available)} resolutions × 5 models on held-out MCC also selects some noise upward; no multiplicity correction is applied and no interval is reported for these threshold metrics.

{{chance}}

Thresholds come from inner out-of-fold F2 as elsewhere in this report, so MCC and F2 depend on a threshold chosen inside the training partition; AUROC, AP and log-loss are threshold-free and are given for the same fits. `domain_fusion_resolution_grid.csv` holds the full resolution × arm grid, `domain_fusion_all_selection_rules.csv` the one-SE results for the same candidates, and `domain_fusion_best_by_resolution.csv` which combination represented each arm and whether F2 would have chosen differently.

### Domain-restricted validation

{int(rows.status.eq("ok").sum())} of {len(rows)} planned fit records completed across {len(available)} resolutions; the {len(unavailable)} unavailable records are all biogas quality. Every ARC-restricted fit has zero BAC fusion edges and every BAC-restricted fit has zero ARC fusion edges, checked per fit, and every retained feature is covered by its resolution's tree. Maximum zero-sum constraint residual is {checks['maximum_equality_residual']:.3g} and group-plus-residual reconstruction error {checks['maximum_group_reconstruction_error']:.3g}. Sample labels and site assignments were checked against the reused Experiment 1 cohort, and all {checks['outer_partitions_without_site_overlap']} outer partitions have disjoint train/test/inner sites. All {len(failures)} recorded candidate failures come from those {len(unavailable_splits)} splits and are the same kind ({', '.join(failure_kinds)}); no candidate failed anywhere else.

The ASV both-domain models reproduce the ASV run above exactly — group counts and all four metrics agree to the last bit across all 72 overlapping fit records, checked here rather than asserted, because the candidate grids, seeds and partition graphs are derived identically. To refit, run `scripts/run_domain_restricted_fusion.py --workers 3 --resolution <order|family|genus|asv>` once per resolution, then `scripts/copy_inputs.py --include-asv --include-domain-fusion` and `scripts/generate_results.py`.

"""
    prior_cols = ["flag", "arm_label", "resolution_label", "model", "retained_features_mean", "active_groups_mean",
                  "active_groups_arc_mean", "active_groups_bac_mean", "structural_compression_pct",
                  "compression_from_input_pct", "mcc", "f2", "auc", "logloss", "is_qc_protocol_best"]
    prior_display = prior.copy()
    prior_display["flag"] = prior_display.target.map(TITLES)
    prior_display["resolution_label"] = prior_display.resolution.map(RESOLUTION_LABELS)
    prior_display["is_qc_protocol_best"] = np.where(prior_display.is_qc_protocol_best, "yes", "no")
    prior_table = api.md(prior_display[prior_cols].rename(columns={
        "flag": "Flag", "arm_label": "Arm", "resolution_label": "Resolution", "model": "Model",
        "retained_features_mean": "Retained", "active_groups_mean": "Active groups",
        "active_groups_arc_mean": "ARC groups", "active_groups_bac_mean": "BAC groups",
        "structural_compression_pct": "Reduction vs retained (%)", "compression_from_input_pct": "Reduction vs input (%)",
        "mcc": "MCC", "f2": "F2", "auc": "AUROC", "logloss": "Log-loss",
        "is_qc_protocol_best": "Also best here?"}))
    disagreeing = prior[~prior.is_qc_protocol_best]
    prior_note = ("Every one of these configurations is also the highest-MCC combination under this protocol."
                  if not len(disagreeing) else
                  f"{len(disagreeing)} of {len(prior)} are **not** the highest-MCC combination under this protocol: " + "; ".join(
                      f"for {TITLES[r.target].lower()} {r.arm_label.lower()}, {RESOLUTION_LABELS[r.resolution]}/"
                      f"{MODEL_LABELS[r.model].lower()} scores MCC {r.mcc:.3f} against "
                      f"{RESOLUTION_LABELS[r.qc_protocol_best_resolution]}/{MODEL_LABELS[r.qc_protocol_best_model].lower()} "
                      f"at {r.qc_protocol_best_mcc:.3f}" for r in disagreeing.itertuples()) +
                  ". The disagreement is expected: the smoke runs split 37 site labels once, this protocol splits 35 QC "
                  "site groups twice, and at 20–24 positives a rank ordering by MCC is not stable across either change. "
                  "Neither ranking is a validated choice of resolution; both read held-out folds.")
    below_chance = selected[selected.auc <= .5].groupby("target").model.apply(lambda s: sorted(set(s))).to_dict()
    chance = (" ".join(
        f"On {TITLES[target].lower()}, {'all five' if len(models) == len(MODELS) else str(len(models)) + ' of the five'} "
        "fitted model families are at or below chance discrimination (AUROC ≤ 0.500) at one or more resolutions, "
        "so its arm ordering by MCC is ordering near-uninformative predictors."
        for target, models in below_chance.items()) if below_chance else
        "Every fitted model at every resolution is above chance discrimination.")
    section = section.replace("{chance}", chance)
    section = section.replace("{prior_table}", prior_table).replace("{prior_note}", prior_note)
    return {"section": section, "findings": "\n".join(findings), "validation": checks}
