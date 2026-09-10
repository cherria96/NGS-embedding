#!/usr/bin/env python3
"""
run_holdout_benchmark.py -- single train:test split version of the site-grouped
CV benchmarks (run_site_grouped_cv_benchmark.py / run_taxlevel_baseline_benchmark.py),
for the user-requested "train:test split instead of CV" comparison across three
feature/tree regimes, for the same 5 warning flags:

  taxlevel    naive taxonomy-label collapse at Phylum/Class/Order/Family/Genus
              rank (build_taxlevel_inputs.py; no tree). 6 flat models: RF, KNN,
              SVM, NNET, XGBoost, GLMNET.
  asv-tree    ASV-level, prevalence/abundance-filtered-but-NOT-collapsed
              features + their pruned tree (output/genus_tree/asv/, built by
              run_genus_tree_pipeline.py --mode asv -- the calibrated-graft-
              fallback merged tree, NOT prepare_warning_targets_data.py's older
              naive DOMAIN_CONNECT_BRANCH_LEN=1.0 graft).
  genus-tree  genus-level, monophyly-checked tree collapse (output/genus_tree/
              genus_autorun/, the main pipeline this whole repo was built
              around).

  asv-tree and genus-tree both run the same 6 tree-capable models: RF, CNN,
  PMCNN, MetaDR, DeepPhylo, PhyloSpec (the actual target model).

THIS SCRIPT IS THE "WITHOUT cross-domain-syntrophy-injection" ARM ONLY (task
doc Phylospec/CROSS_DOMAIN_INJECTION_BY_MODEL.md). The "with injection" arms
for asv-tree/genus-tree need build_cross_domain_graph.py's fused-distance
artefacts, which do not exist yet for this dataset -- see
run_holdout_benchmark_injection.py once that lands. taxlevel has no tree at
all, so "with/without injection" does not apply to it either way.

Split: reuses ML/results_warnings/split_and_cv_assignment.csv's FIXED
109-train / 29-test site partition (28 train site-groups, 7 test site-groups)
verbatim -- not a fresh split -- so results here are directly comparable,
same held-out sites, to that separate metadata+top-k-taxa pipeline's own
train:test numbers. That file only covers the 138 ARC-aligned SampleIDs, so
every arm/domain here (including BAC, which otherwise has 140) is filtered
to exactly those 138 samples, for one consistent partition across the whole
benchmark -- not a per-domain-specific split.

Threshold tuning: an inner site-grouped 80/20 split carved OUT OF THE 109
TRAIN samples only (evaluate_multilabel.split_train_val, same helper
evaluate_cv() itself uses per fold) -- never touches the 29 test samples.
This is a single split, not evaluate_cv()'s repeated-and-pooled protocol, so
there is no confidence interval here, only a point estimate; read
evaluate_multilabel.evaluate_holdout()'s docstring before treating any one
number here as more than that, ESPECIALLY for acid_accumulation (see
audit_site_label_support() run against the actual 29-sample test set below,
not the full-cohort numbers reported elsewhere in REPORT.md).

Output: results_<arm>[_<level>]_<domain>_<model>.csv, one row per flag plus a
macro row (point estimates, no CI), in
Phylospec/multi_models/results/holdout/.
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PHYLOSPEC_DIR = os.path.join(HERE, "..")
ROOT = os.path.join(PHYLOSPEC_DIR, "..")

sys.path.insert(0, HERE)
sys.path.insert(0, PHYLOSPEC_DIR)
from run_site_grouped_cv_benchmark import (  # noqa: E402
    FLAG_COLS, PY, POS_WEIGHT_CAP, PHYLOSPEC_EPOCHS,
    make_csv_fit_predict, make_deepphylo_fit_predict, run_subprocess,
)
from evaluate_multilabel import audit_site_label_support, evaluate_holdout  # noqa: E402

SPLIT_PATH = os.path.join(ROOT, "ML", "results_warnings", "split_and_cv_assignment.csv")
TAXLEVEL_DIR = os.path.join(HERE, "input_for_all_models_taxlevel")
ASVTREE_DIR = os.path.join(HERE, "input_for_all_models_asvtree")
GENUSTREE_DIR = os.path.join(HERE, "input_for_all_models_genus")
RESULTS_DIR = os.path.join(HERE, "results", "holdout")

LEVELS = ["Phylum", "Class", "Order", "Family", "Genus"]
DOMAINS = ["ARC", "BAC", "merged"]
TAXLEVEL_MODELS = ["RF", "KNN", "SVM", "NNET", "XGBoost", "GLMNET"]
TREE_MODELS = ["RF", "CNN", "PMCNN", "MetaDR", "DeepPhylo", "PhyloSpec"]
DOMAIN_TO_ARM = {"ARC": "arc", "BAC": "bac", "merged": "merged"}


def load_split():
    """ML/results_warnings/split_and_cv_assignment.csv: SampleID -> 'train'/'test'.
    Fixed, external, shared with that pipeline -- not regenerated here."""
    df = pd.read_csv(SPLIT_PATH)
    return dict(zip(df["SampleID"].astype(str), df["split"]))


def restrict_to_split(combined_df, split_map):
    """Filters to exactly the SampleIDs present in split_map (138 ARC-aligned
    samples, same set for every domain/arm) and returns (df, train_idx,
    test_idx) as 0-based positions into the FILTERED df -- what
    evaluate_holdout()'s fit_predict index arrays are relative to."""
    df = combined_df[combined_df["SampleID"].astype(str).isin(split_map)].reset_index(drop=True)
    dropped = len(combined_df) - len(df)
    if dropped:
        print(f"    ({dropped} samples not in the shared 138-sample split, dropped)")
    split = df["SampleID"].astype(str).map(split_map).values
    train_idx = np.flatnonzero(split == "train")
    test_idx = np.flatnonzero(split == "test")
    return df, train_idx, test_idx


def load_site_map():
    meta = pd.read_csv(os.path.join(ROOT, "data", "final", "metadata.csv"), encoding="utf-8-sig")
    return dict(zip(meta["SampleID"].astype(str), meta["Site"]))


def make_phylospec_fit_predict(domain, combined_df, tree_path, scratch_root):
    """Same contract as run_site_grouped_cv_benchmark.make_phylospec_fit_predict,
    generalised to an explicit tree_path instead of hardcoding the genus-tree
    directory, so the same function serves asv-tree and genus-tree alike."""
    script = os.path.join(PHYLOSPEC_DIR, "src", "model", "PhyloSpec_train_test.py")
    labels_arg = ",".join(FLAG_COLS)
    env = dict(os.environ, PYTHONPATH=os.path.abspath(ROOT))

    def fit_predict(tr, va, te, pos_weight):
        with tempfile.TemporaryDirectory(dir=scratch_root) as tmp:
            train_csv, val_csv, test_csv = (os.path.join(tmp, f"{s}.csv")
                                            for s in ("train", "val", "test"))
            combined_df.iloc[tr].to_csv(train_csv, index=False)
            combined_df.iloc[va].to_csv(val_csv, index=False)
            combined_df.iloc[te].to_csv(test_csv, index=False)
            model_dir = os.path.join(tmp, "model") + os.sep
            os.makedirs(model_dir, exist_ok=True)

            run_subprocess([PY, script, "--PhyloSpec", "train",
                            "-c", train_csv, "-t", tree_path, "-o", model_dir,
                            "-labels", labels_arg, "-ep", str(PHYLOSPEC_EPOCHS),
                            "-pos_weight_cap", str(POS_WEIGHT_CAP)], env=env)

            val_scores_path = os.path.join(tmp, "val_scores.npy")
            test_scores_path = os.path.join(tmp, "test_scores.npy")
            run_subprocess([PY, script, "--PhyloSpec", "test",
                            "-c", val_csv, "-t", tree_path, "-o", model_dir,
                            "-labels", labels_arg, "-scores_out", val_scores_path], env=env)
            run_subprocess([PY, script, "--PhyloSpec", "test",
                            "-c", test_csv, "-t", tree_path, "-o", model_dir,
                            "-labels", labels_arg, "-scores_out", test_scores_path], env=env)
            val_scores = np.load(val_scores_path)
            test_scores = np.load(test_scores_path)
        return val_scores, test_scores
    return fit_predict


def load_taxlevel_table(level, domain):
    csv_dir = os.path.join(TAXLEVEL_DIR, level, f"Warnings_{domain}")
    return pd.read_csv(os.path.join(csv_dir, "table.csv"))


def load_tree_arm(base_dir, domain):
    """asv-tree / genus-tree: table.csv + X.npy/Y.npy/c.npy/label_names.txt +
    PMCNN_list.csv + phylogeny.nwk, all written by build_inputs_from_genus_tables.py."""
    csv_dir = os.path.join(base_dir, f"Warnings_{domain}")
    combined_df = pd.read_csv(os.path.join(csv_dir, "table.csv"))
    X_all = np.load(os.path.join(csv_dir, "X.npy"))
    Y_all_raw = np.load(os.path.join(csv_dir, "Y.npy"))
    return combined_df, X_all, Y_all_raw, csv_dir


def build_taxlevel_fit_predict(model, domain, combined_df, scratch_root):
    script = os.path.join(HERE, model, f"{model}-multilabel.py")
    return make_csv_fit_predict(script, domain, combined_df, [], scratch_root)


def build_tree_fit_predict(model, domain, combined_df, X_all, Y_all_raw, csv_dir, scratch_root):
    if model == "RF":
        return make_csv_fit_predict(os.path.join(HERE, "RF", "RF-multilabel.py"),
                                    domain, combined_df, [], scratch_root)
    if model == "CNN":
        return make_csv_fit_predict(os.path.join(HERE, "CNN", "CNN-multilabel.py"),
                                    domain, combined_df, [], scratch_root)
    if model == "PMCNN":
        pmcnn_list = os.path.join(csv_dir, "PMCNN_list.csv")
        return make_csv_fit_predict(os.path.join(HERE, "PMCNN", "PMCNN-multilabel.py"),
                                    domain, combined_df, ["--list", pmcnn_list], scratch_root)
    if model == "MetaDR":
        tree = os.path.join(csv_dir, "phylogeny.nwk")
        return make_csv_fit_predict(os.path.join(HERE, "MetaDR", "MetaDR-multilabel.py"),
                                    domain, combined_df, ["-t", tree], scratch_root)
    if model == "DeepPhylo":
        return make_deepphylo_fit_predict(domain, X_all, Y_all_raw, csv_dir, scratch_root)
    if model == "PhyloSpec":
        tree_path = os.path.join(csv_dir, "phylogeny.nwk")
        return make_phylospec_fit_predict(domain, combined_df, tree_path, scratch_root)
    raise ValueError(model)


def result_table(res, arm, domain, model, level=None):
    """Holdout analogue of run_site_grouped_cv_benchmark.score_multilabel_table
    -- same columns, but point estimates only (no _lo/_hi, single split)."""
    rows = []
    for r in res["per_label"]:
        row = {"arm": arm, "domain": domain, "model": model, "flag": r["label"],
              "n_pos": r["n_pos"], "prevalence": r["prevalence"]}
        for k in ["auprc", "auprc_lift", "auroc", "mcc", "f2", "precision", "recall", "threshold"]:
            row[k] = r.get(k, float("nan"))
        rows.append(row)
    macro = {"arm": arm, "domain": domain, "model": model, "flag": "MACRO", "n_pos": None,
            "prevalence": None}
    for k in ["auprc", "auprc_lift", "auroc", "mcc", "f2", "precision", "recall"]:
        macro[k] = res["macro"].get(f"macro_{k}", float("nan"))
    macro["threshold"] = float("nan")
    rows.append(macro)
    table = pd.DataFrame(rows)
    if level:
        table.insert(1, "level", level)
    return table


def run_taxlevel(levels, domains, models, split_map, out_dir, scratch_root):
    for level in levels:
        for domain in domains:
            raw = load_taxlevel_table(level, domain)
            df, train_idx, test_idx = restrict_to_split(raw, split_map)
            site_map = load_site_map()
            sites = df["SampleID"].astype(str).map(site_map).values
            y = df[FLAG_COLS].values.astype(int)
            support = audit_site_label_support(y[test_idx], sites[test_idx], label_names=FLAG_COLS)
            pd.DataFrame(support).to_csv(
                os.path.join(out_dir, f"test_label_support_taxlevel_{level}_{domain}.csv"), index=False)
            print(f"=== taxlevel / {level} / {domain} === "
                 f"({df.shape[1] - 1 - len(FLAG_COLS)} features, "
                 f"{len(train_idx)} train / {len(test_idx)} test)")
            for model in models:
                print(f"--- taxlevel / {level} / {domain} / {model} ---")
                fit_predict = build_taxlevel_fit_predict(model, domain, df, scratch_root)
                res = evaluate_holdout(y, sites, fit_predict, train_idx, test_idx,
                                       label_names=FLAG_COLS, pos_weight_cap=POS_WEIGHT_CAP)
                table = result_table(res, "taxlevel", domain, model, level=level)
                out_path = os.path.join(out_dir, f"results_taxlevel_{level}_{domain}_{model}.csv")
                table.to_csv(out_path, index=False)
                print(table[["flag", "n_pos", "mcc", "f2", "auprc", "auprc_lift"]].to_string(index=False))
                print(f"-> {out_path}")


def run_tree_arm(arm, base_dir, domains, models, split_map, out_dir, scratch_root):
    for domain in domains:
        raw, X_all_full, Y_all_raw_full, csv_dir = load_tree_arm(base_dir, domain)
        keep_mask = raw["SampleID"].astype(str).isin(split_map).values
        df, train_idx, test_idx = restrict_to_split(raw, split_map)
        X_all = X_all_full[keep_mask]
        Y_all_raw = Y_all_raw_full[keep_mask]
        assert len(df) == len(X_all), f"{arm}/{domain}: filtered table/npy row mismatch"

        site_map = load_site_map()
        sites = df["SampleID"].astype(str).map(site_map).values
        y = df[FLAG_COLS].values.astype(int)
        support = audit_site_label_support(y[test_idx], sites[test_idx], label_names=FLAG_COLS)
        pd.DataFrame(support).to_csv(
            os.path.join(out_dir, f"test_label_support_{arm}_{domain}.csv"), index=False)
        print(f"=== {arm} / {domain} === ({df.shape[1] - 1 - len(FLAG_COLS)} features, "
             f"{len(train_idx)} train / {len(test_idx)} test)")

        for model in models:
            print(f"--- {arm} / {domain} / {model} ---")
            fit_predict = build_tree_fit_predict(model, domain, df, X_all, Y_all_raw,
                                                 csv_dir, scratch_root)
            res = evaluate_holdout(y, sites, fit_predict, train_idx, test_idx,
                                   label_names=FLAG_COLS, pos_weight_cap=POS_WEIGHT_CAP)
            table = result_table(res, arm, domain, model)
            out_path = os.path.join(out_dir, f"results_{arm}_{domain}_{model}.csv")
            table.to_csv(out_path, index=False)
            print(table[["flag", "n_pos", "mcc", "f2", "auprc", "auprc_lift"]].to_string(index=False))
            print(f"-> {out_path}")


def run_all(arms, levels, domains, taxlevel_models, tree_models):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    scratch_root = os.path.join(RESULTS_DIR, "_scratch")
    os.makedirs(scratch_root, exist_ok=True)
    split_map = load_split()
    print(f"Loaded split: {sum(v=='train' for v in split_map.values())} train / "
         f"{sum(v=='test' for v in split_map.values())} test "
         f"({len(split_map)} samples total) from {SPLIT_PATH}")

    if "taxlevel" in arms:
        run_taxlevel(levels, domains, taxlevel_models, split_map, RESULTS_DIR, scratch_root)
    if "asv-tree" in arms:
        run_tree_arm("asvtree", ASVTREE_DIR, domains, tree_models, split_map,
                    RESULTS_DIR, scratch_root)
    if "genus-tree" in arms:
        run_tree_arm("genustree", GENUSTREE_DIR, domains, tree_models, split_map,
                    RESULTS_DIR, scratch_root)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=["taxlevel", "asv-tree", "genus-tree"],
                    choices=["taxlevel", "asv-tree", "genus-tree"])
    ap.add_argument("--levels", nargs="+", default=LEVELS, choices=LEVELS)
    ap.add_argument("--domains", nargs="+", default=DOMAINS, choices=DOMAINS)
    ap.add_argument("--taxlevel-models", nargs="+", default=TAXLEVEL_MODELS, choices=TAXLEVEL_MODELS)
    ap.add_argument("--tree-models", nargs="+", default=TREE_MODELS, choices=TREE_MODELS)
    a = ap.parse_args()
    run_all(a.arms, a.levels, a.domains, a.taxlevel_models, a.tree_models)
