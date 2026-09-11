"""
run_cross_domain_injection_benchmark.py -- CROSS_DOMAIN_INJECTION_BY_MODEL.md end
to end: build the Bacteria x Archaea prior on the merged-domain genus data
(build_cross_domain_prior_merged.py), inject it into DeepPhylo, MetaDR and
Phylo-Spec via each model's own channel, and benchmark "without" vs "with" at
the SAME epoch budget, through evaluate_multilabel.evaluate_cv() (site-grouped
CV, pooled out-of-fold scoring, per-flag MCC-tuned thresholds -- Sec 5A).

Only the "merged" domain is run: the cross-domain network needs both Bacteria
and Archaea features, which ARC-only and BAC-only tables don't have.

Per-model injection, each the SAME code path across arms (injection parameter
at its neutral value for "without"):
  DeepPhylo   -- c.npy: patristic Dp (without) vs syntrophy-shortcut Dfused
                 (with). Both come out of ONE build_cross_domain_prior_merged.py
                 call to shortcut_distance(), so "without" is not a separate
                 script (CROSS_DOMAIN_INJECTION_BY_MODEL.md Sec 1).
  MetaDR      -- post-order/zigzag branch column order: tree post-order
                 (without) vs optimal-leaf-ordering seriation of Dfused (with),
                 via --seriation-order (Sec 3).
  Phylo-Spec  -- Arm 2a node-weight reweighting: alpha=0 (without, bit-for-bit
                 baseline weights) vs alpha=1.0 with the retained cross-domain
                 edges (with), via -cross_domain_edges/-alpha (Sec 2).

Usage:
    python run_cross_domain_injection_benchmark.py
    python run_cross_domain_injection_benchmark.py --models DeepPhylo --n-repeats 1
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PHYLOSPEC_DIR = os.path.join(HERE, "..")
ROOT = os.path.join(PHYLOSPEC_DIR, "..")
GENUS_CSV_DATA_DIR = os.path.join(HERE, "input_for_all_models_genus")
CSV_DATA_DIR = GENUS_CSV_DATA_DIR
RESULTS_DIR = os.path.join(HERE, "results", "cross_domain_injection")
PRIOR_PREFIX = os.path.join(RESULTS_DIR, "merged")
EPOCHS = 60
PATIENCE = 10
POS_WEIGHT_CAP = 10.0
PHYLOSPEC_ALPHA = 1.0
MODELS = ["DeepPhylo", "MetaDR", "PhyloSpec"]
ARMS = ["without", "with"]
PY = sys.executable

sys.path.insert(0, PHYLOSPEC_DIR)
sys.path.insert(0, HERE)
from evaluate_multilabel import audit_site_label_support, evaluate_cv  # noqa: E402
import run_site_grouped_cv_benchmark as sgb  # noqa: E402
from run_site_grouped_cv_benchmark import (  # noqa: E402
    load_site_map, _deepphylo_col_order, run_subprocess,
    score_multilabel_table, FLAG_COLS,
)


def load_domain(domain):
    """load_domain() reads run_site_grouped_cv_benchmark's own CSV_DATA_DIR
    global at call time -- keep it in sync with this script's CSV_DATA_DIR
    (which --csv-data-dir may have overridden) before calling."""
    sgb.CSV_DATA_DIR = CSV_DATA_DIR
    return sgb.load_domain(domain)


def build_prior_if_needed(force=False):
    if not force and all(os.path.exists(f"{PRIOR_PREFIX}_{s}") for s in
                         ["Dp.npy", "Dfused.npy", "edges.csv", "seriation.txt"]):
        print(f"reusing existing cross-domain prior at {PRIOR_PREFIX}_*")
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cmd = [PY, os.path.join(HERE, "build_cross_domain_prior_merged.py"), PRIOR_PREFIX,
           "--data-dir", CSV_DATA_DIR]
    if CSV_DATA_DIR != GENUS_CSV_DATA_DIR:
        # Not the genus arm -- e.g. input_for_all_models_asv, whose raw ASV-hash
        # feature ids have no reduced-feature mapping.csv to join against, so
        # domain-of-feature must come from ARC/BAC table membership instead
        # (see build_cross_domain_prior_merged.py's module docstring).
        cmd += ["--arc-dir", os.path.join(CSV_DATA_DIR, "Warnings_ARC"),
               "--bac-dir", os.path.join(CSV_DATA_DIR, "Warnings_BAC")]
    run_subprocess(cmd)


# --------------------------------------------------------------- DeepPhylo
def make_deepphylo_fit_predict(X_all, Y_all_raw, dp_dir, c_array, scratch_root):
    col_perm = _deepphylo_col_order(dp_dir)
    script = os.path.join(HERE, "DeepPhylo", "deepphylo_classification_multi_label_bce.py")
    label_path = os.path.join(dp_dir, "label_names.txt")

    def fit_predict(tr, va, te, pos_weight):
        with tempfile.TemporaryDirectory(dir=scratch_root) as tmp:
            np.save(os.path.join(tmp, "X_train.npy"), X_all[tr])
            np.save(os.path.join(tmp, "Y_train.npy"), Y_all_raw[tr])
            np.save(os.path.join(tmp, "X_val.npy"), X_all[va])
            np.save(os.path.join(tmp, "Y_val.npy"), Y_all_raw[va])
            np.save(os.path.join(tmp, "X_eval.npy"), X_all[te])
            np.save(os.path.join(tmp, "Y_eval.npy"), Y_all_raw[te])
            np.save(os.path.join(tmp, "c.npy"), c_array)
            shutil.copyfile(label_path, os.path.join(tmp, "label_names.txt"))
            scores_dir = os.path.join(tmp, "scores")
            cmd = [PY, script, "--data_dir", tmp, "--val_dir", tmp, "--domain", "merged",
                   "--epochs", str(EPOCHS), "--pos-weight-cap", str(POS_WEIGHT_CAP),
                   "--scores-out-dir", scores_dir]
            if PATIENCE:
                cmd += ["--patience", str(PATIENCE)]
            run_subprocess(cmd)
            val_scores = np.load(os.path.join(scores_dir, "val_scores.npy"))[:, col_perm]
            test_scores = np.load(os.path.join(scores_dir, "test_scores.npy"))[:, col_perm]
        return val_scores, test_scores
    return fit_predict


# ------------------------------------------------------------------ MetaDR
def make_metadr_fit_predict(combined_df, tree_path, seriation_path, scratch_root):
    script = os.path.join(HERE, "MetaDR", "MetaDR-multilabel.py")
    extra_args = ["-t", tree_path, "--epochs", str(EPOCHS)]
    if PATIENCE:
        extra_args += ["--patience", str(PATIENCE)]
    if seriation_path:
        extra_args += ["--seriation-order", seriation_path]

    def fit_predict(tr, va, te, pos_weight):
        with tempfile.TemporaryDirectory(dir=scratch_root) as tmp:
            train_csv, val_csv, test_csv = (os.path.join(tmp, f"{s}.csv")
                                            for s in ("train", "val", "test"))
            combined_df.iloc[tr].to_csv(train_csv, index=False)
            combined_df.iloc[va].to_csv(val_csv, index=False)
            combined_df.iloc[te].to_csv(test_csv, index=False)
            scores_dir = os.path.join(tmp, "scores")
            cmd = [PY, script, "--train", train_csv, "--val", val_csv, "--test", test_csv,
                   "--domain", "merged", "--pos-weight-cap", str(POS_WEIGHT_CAP),
                   "--scores-out-dir", scores_dir] + extra_args
            run_subprocess(cmd)
            val_scores = np.load(os.path.join(scores_dir, "val_scores.npy"))
            test_scores = np.load(os.path.join(scores_dir, "test_scores.npy"))
        return val_scores, test_scores
    return fit_predict


# --------------------------------------------------------------- Phylo-Spec
def make_phylospec_fit_predict(combined_df, tree_path, edges_path, alpha, scratch_root):
    script = os.path.join(PHYLOSPEC_DIR, "src", "model", "PhyloSpec_train_test.py")
    labels_arg = ",".join(FLAG_COLS)
    env = dict(os.environ, PYTHONPATH=os.path.abspath(ROOT))
    injection_args = ["-cross_domain_edges", edges_path, "-alpha", str(alpha)] if edges_path else []

    def fit_predict(tr, va, te, pos_weight):
        with tempfile.TemporaryDirectory(dir=scratch_root) as tmp:
            train_csv, val_csv, test_csv = (os.path.join(tmp, f"{s}.csv")
                                            for s in ("train", "val", "test"))
            combined_df.iloc[tr].to_csv(train_csv, index=False)
            combined_df.iloc[va].to_csv(val_csv, index=False)
            combined_df.iloc[te].to_csv(test_csv, index=False)
            model_dir = os.path.join(tmp, "model") + os.sep
            os.makedirs(model_dir, exist_ok=True)

            train_cmd = [PY, script, "--PhyloSpec", "train",
                        "-c", train_csv, "-t", tree_path, "-o", model_dir,
                        "-labels", labels_arg, "-ep", str(EPOCHS),
                        "-pos_weight_cap", str(POS_WEIGHT_CAP)] + injection_args
            if PATIENCE:
                train_cmd += ["-val_c", val_csv, "-patience", str(PATIENCE)]
            run_subprocess(train_cmd, env=env)

            val_scores_path = os.path.join(tmp, "val_scores.npy")
            test_scores_path = os.path.join(tmp, "test_scores.npy")
            run_subprocess([PY, script, "--PhyloSpec", "test",
                            "-c", val_csv, "-t", tree_path, "-o", model_dir,
                            "-labels", labels_arg, "-scores_out", val_scores_path] + injection_args, env=env)
            run_subprocess([PY, script, "--PhyloSpec", "test",
                            "-c", test_csv, "-t", tree_path, "-o", model_dir,
                            "-labels", labels_arg, "-scores_out", test_scores_path] + injection_args, env=env)

            val_scores = np.load(val_scores_path)
            test_scores = np.load(test_scores_path)
        return val_scores, test_scores
    return fit_predict


def build_fit_predict(model, arm, combined_df, X_all, Y_all_raw, dp_dir, tree_path, scratch_root):
    with_injection = arm == "with"
    if model == "DeepPhylo":
        c_array = np.load(f"{PRIOR_PREFIX}_Dfused.npy" if with_injection else f"{PRIOR_PREFIX}_Dp.npy")
        return make_deepphylo_fit_predict(X_all, Y_all_raw, dp_dir, c_array, scratch_root)
    if model == "MetaDR":
        seriation_path = f"{PRIOR_PREFIX}_seriation.txt" if with_injection else None
        return make_metadr_fit_predict(combined_df, tree_path, seriation_path, scratch_root)
    if model == "PhyloSpec":
        edges_path = f"{PRIOR_PREFIX}_edges.csv" if with_injection else None
        alpha = PHYLOSPEC_ALPHA if with_injection else 0.0
        return make_phylospec_fit_predict(combined_df, tree_path, edges_path, alpha, scratch_root)
    raise ValueError(model)


def run_all(models=MODELS, arms=ARMS, n_splits=5, n_repeats=5, force_prior=False):
    build_prior_if_needed(force=force_prior)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    scratch_root = os.path.join(RESULTS_DIR, "_scratch")
    os.makedirs(scratch_root, exist_ok=True)

    combined_df, X_all, Y_all_raw, dp_dir, sites, y = load_domain("merged")
    tree_path = os.path.join(CSV_DATA_DIR, "Warnings_merged", "phylogeny.nwk")

    support = audit_site_label_support(y, sites, label_names=FLAG_COLS)
    pd.DataFrame(support).to_csv(os.path.join(RESULTS_DIR, "site_label_support_merged.csv"), index=False)

    for model in models:
        for arm in arms:
            print(f"--- merged / {model} / {arm} ({EPOCHS} epochs) ---")
            fit_predict = build_fit_predict(model, arm, combined_df, X_all, Y_all_raw,
                                            dp_dir, tree_path, scratch_root)
            res = evaluate_cv(y, sites, fit_predict, label_names=FLAG_COLS,
                             n_splits=n_splits, n_repeats=n_repeats,
                             pos_weight_cap=POS_WEIGHT_CAP)
            table = score_multilabel_table(res, "merged", model)
            table["arm"] = arm
            out_path = os.path.join(RESULTS_DIR, f"results_merged_{model}_{arm}.csv")
            table.to_csv(out_path, index=False)
            print(table[["flag", "n_pos", "mcc", "f2", "auprc", "auprc_lift"]].to_string(index=False))
            print(f"-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--rebuild-prior", action="store_true",
                    help="Recompute the cross-domain prior even if cached outputs exist.")
    ap.add_argument("--csv-data-dir", default=CSV_DATA_DIR,
                    help="Per-domain Warnings_<domain>/{table.csv,phylogeny.nwk,...} root "
                         "(needs Warnings_ARC/BAC/merged, all three domains). Default: the "
                         "genus-level arm. Point this at input_for_all_models_asv (built by "
                         "build_inputs_from_asv_tables.py) to run cross-domain injection "
                         "against the original, ungrafted ASV-level features/tree instead -- "
                         "switches the prior's domain-of-feature lookup to ARC/BAC table "
                         "membership automatically (see build_cross_domain_prior_merged.py).")
    ap.add_argument("--results-dir", default=RESULTS_DIR,
                    help="Where to write results_merged_<model>_<arm>.csv etc. and the cached "
                         "prior. Default: results/cross_domain_injection.")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE,
                    help="Early-stopping patience (epochs) for all 3 models, using the inner "
                         "validation split. 0 disables early stopping.")
    a = ap.parse_args()
    # run_subprocess() runs child scripts with cwd=HERE, so a relative
    # --csv-data-dir/--results-dir (resolved against the invoking shell's cwd)
    # must be made absolute here, or paths built from it break.
    CSV_DATA_DIR = os.path.abspath(a.csv_data_dir)
    RESULTS_DIR = os.path.abspath(a.results_dir)
    PRIOR_PREFIX = os.path.join(RESULTS_DIR, "merged")
    EPOCHS = a.epochs
    PATIENCE = a.patience
    run_all(a.models, a.arms, a.n_splits, a.n_repeats, force_prior=a.rebuild_prior)
