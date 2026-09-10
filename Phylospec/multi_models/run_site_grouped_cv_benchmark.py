"""
run_site_grouped_cv_benchmark.py -- REPLACES run_cv_benchmark.py.

Wires RF/CNN/PMCNN/MetaDR/DeepPhylo into evaluate_multilabel.evaluate_cv()
(task Sec 5A -- already implemented, not reimplemented here) instead of
run_cv_benchmark.py's plain `sklearn.KFold(shuffle=True)` + fold-by-fold
accuracy/MCC/ROC-AUC/AUPR averaging, which:

  1. splits at the SAMPLE level, so two seasons of the same digester can land
     on opposite sides of a fold -- Sec 5A.1 calls this out explicitly as
     something no result can be trusted through;
  2. scores each fold independently and averages, which -- given
     acid_accumulation has 4 positives total across 36 sites -- means most
     folds contribute a NaN for that flag and the "mean +/- std" silently
     drops them instead of reporting that the flag was barely evaluated
     (Sec 5A.2);
  3. reports accuracy (explicitly excluded, Sec 7) and thresholds at 0.5
     (explicitly excluded, Sec 5A.3) instead of per-flag MCC-tuned thresholds
     fit on validation scores.

evaluate_cv() owns the site-grouped repeated splitting, the inner train/val
carve-out, pooling test predictions within a repeat, MCC threshold tuning on
pooled validation scores, and per-label + macro AUPRC / AUPRC-lift / AUROC /
MCC / F2 / precision / recall with 95% percentile intervals across repeats.
This script's only job is to give it a `fit_predict(tr, va, te, pos_weight) ->
(val_scores, test_scores)` per model that writes those three index sets out
in that model's native input format, subprocess-calls the (already patched
to support --val/--scores-out-dir) model script, and reads back raw
probability arrays.

*** COLUMN-ORDER WARNING, read before touching this file ***
RF/CNN/PMCNN/MetaDR select training columns by NAME (`train_df[FLAGS]`,
FLAGS sorted alphabetically), so their y_prob columns always come out in that
alphabetical order regardless of on-disk column order. DeepPhylo's Y_*.npy
files carry NO column names -- their column order is whatever
prepare_warning_targets_data.write_deepphylo_npys used, which is
evaluate_multilabel.LABEL_COLS order (acid_base_balance, buffer_capacity,
acid_accumulation, ammonia_toxicity, biogas_quality), a DIFFERENT order than
the alphabetical FLAG_COLS used everywhere else in this file. Silently
pooling DeepPhylo's raw output against `y` (built in FLAG_COLS order) would
score, e.g., DeepPhylo's acid_base_balance predictions against the
acid_accumulation ground-truth column -- wrong, silent, and not something a
shape check would catch (both are length-5 vectors). `_deepphylo_col_order()`
below reads each fold's label_names.txt and computes the permutation back to
FLAG_COLS explicitly; do not remove it.

Output: results_<domain>_<model>.csv, one row per flag plus a macro row,
per Sec 5A.5.
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
# Genus-level, calibrated-graft-fallback data (built by
# build_inputs_from_genus_tables.py from output/genus_tree/genus_autorun/),
# NOT the older input_for_all_models/ + DeepPhylo/data/ trees -- those were
# built by prepare_warning_targets_data.py from raw ASV features under a
# naive DOMAIN_CONNECT_BRANCH_LEN=1.0 graft (REPORT.md Sec 4).
CSV_DATA_DIR = os.path.join(HERE, "input_for_all_models_genus")
METADATA_PATH = os.path.join(ROOT, "data", "final", "metadata.csv")
RESULTS_DIR = os.path.join(HERE, "results", "site_grouped_cv")
DOMAINS = ["ARC", "BAC", "merged"]
MODELS = ["RF", "CNN", "PMCNN", "MetaDR", "DeepPhylo"]
POS_WEIGHT_CAP = 10.0
PY = sys.executable

sys.path.insert(0, PHYLOSPEC_DIR)
from evaluate_multilabel import LABEL_COLS, audit_site_label_support, evaluate_cv  # noqa: E402

FLAG_COLS = sorted(c.replace("warning_", "") for c in LABEL_COLS)  # alphabetical,
# matches the FLAGS constant hardcoded in RF/CNN/PMCNN/MetaDR-multilabel.py


def load_site_map():
    df = pd.read_csv(METADATA_PATH, encoding="utf-8-sig")
    return dict(zip(df["SampleID"].astype(str), df["Site"]))


def load_domain(domain):
    """build_inputs_from_genus_tables.py writes ONE table.csv + ONE set of
    DeepPhylo npys per domain (no train/test split at rest -- the outer
    test / inner val split is entirely evaluate_cv()'s job), so there is no
    concat step here unlike the old ASV-level loader."""
    csv_dir = os.path.join(CSV_DATA_DIR, f"Warnings_{domain}")
    combined_df = pd.read_csv(os.path.join(csv_dir, "table.csv"))
    X_all = np.load(os.path.join(csv_dir, "X.npy"))
    Y_all_raw = np.load(os.path.join(csv_dir, "Y.npy"))
    assert len(combined_df) == len(X_all), (
        f"{domain}: table.csv rows ({len(combined_df)}) != X.npy rows ({len(X_all)})")

    site_map = load_site_map()
    sites = combined_df["SampleID"].astype(str).map(site_map).values
    assert not pd.isna(sites).any(), f"{domain}: samples with no Site in {METADATA_PATH}"

    y = combined_df[FLAG_COLS].values.astype(int)  # alphabetical order, authoritative
    return combined_df, X_all, Y_all_raw, csv_dir, sites, y


def _deepphylo_col_order(dp_dir):
    """Permutation p such that Y_all_raw[:, p] == columns in FLAG_COLS order."""
    label_path = os.path.join(dp_dir, "label_names.txt")
    raw_order = [n.replace("warning_", "") for n in open(label_path).read().split()]
    assert sorted(raw_order) == FLAG_COLS, (
        f"DeepPhylo label_names.txt flags {sorted(raw_order)} != FLAG_COLS {FLAG_COLS}")
    return [raw_order.index(flag) for flag in FLAG_COLS]


def run_subprocess(cmd):
    subprocess.run(cmd, check=True, cwd=HERE)


def make_csv_fit_predict(script, domain, combined_df, extra_args, scratch_root):
    """RF / CNN / PMCNN / MetaDR: all read --train/--val/--test CSVs with a
    SampleID-first, FLAG_COLS-last column layout and select feature/label
    columns BY NAME, so alphabetical FLAG_COLS order is guaranteed on output."""
    def fit_predict(tr, va, te, pos_weight):
        with tempfile.TemporaryDirectory(dir=scratch_root) as tmp:
            train_csv, val_csv, test_csv = (os.path.join(tmp, f"{s}.csv")
                                            for s in ("train", "val", "test"))
            combined_df.iloc[tr].to_csv(train_csv, index=False)
            combined_df.iloc[va].to_csv(val_csv, index=False)
            combined_df.iloc[te].to_csv(test_csv, index=False)
            scores_dir = os.path.join(tmp, "scores")
            cmd = [PY, script, "--train", train_csv, "--val", val_csv, "--test", test_csv,
                   "--domain", domain, "--pos-weight-cap", str(POS_WEIGHT_CAP),
                   "--scores-out-dir", scores_dir] + extra_args
            run_subprocess(cmd)
            val_scores = np.load(os.path.join(scores_dir, "val_scores.npy"))
            test_scores = np.load(os.path.join(scores_dir, "test_scores.npy"))
        return val_scores, test_scores
    return fit_predict


def make_deepphylo_fit_predict(domain, X_all, Y_all_raw, dp_dir, scratch_root):
    col_perm = _deepphylo_col_order(dp_dir)
    script = os.path.join(HERE, "DeepPhylo", "deepphylo_classification_multi_label_bce.py")
    c_path = os.path.join(dp_dir, "c.npy")
    label_path = os.path.join(dp_dir, "label_names.txt")

    def fit_predict(tr, va, te, pos_weight):
        with tempfile.TemporaryDirectory(dir=scratch_root) as tmp:
            np.save(os.path.join(tmp, "X_train.npy"), X_all[tr])
            np.save(os.path.join(tmp, "Y_train.npy"), Y_all_raw[tr])
            np.save(os.path.join(tmp, "X_val.npy"), X_all[va])
            np.save(os.path.join(tmp, "Y_val.npy"), Y_all_raw[va])
            np.save(os.path.join(tmp, "X_eval.npy"), X_all[te])   # outer test
            np.save(os.path.join(tmp, "Y_eval.npy"), Y_all_raw[te])
            shutil.copyfile(c_path, os.path.join(tmp, "c.npy"))
            shutil.copyfile(label_path, os.path.join(tmp, "label_names.txt"))
            scores_dir = os.path.join(tmp, "scores")
            cmd = [PY, script, "--data_dir", tmp, "--val_dir", tmp, "--domain", domain,
                   "--pos-weight-cap", str(POS_WEIGHT_CAP), "--scores-out-dir", scores_dir]
            run_subprocess(cmd)
            val_scores = np.load(os.path.join(scores_dir, "val_scores.npy"))[:, col_perm]
            test_scores = np.load(os.path.join(scores_dir, "test_scores.npy"))[:, col_perm]
        return val_scores, test_scores
    return fit_predict


def build_fit_predict(model, domain, combined_df, X_all, Y_all_raw, dp_dir, scratch_root):
    d = os.path.join(CSV_DATA_DIR, f"Warnings_{domain}")
    if model == "RF":
        return make_csv_fit_predict(os.path.join(HERE, "RF", "RF-multilabel.py"),
                                    domain, combined_df, [], scratch_root)
    if model == "CNN":
        return make_csv_fit_predict(os.path.join(HERE, "CNN", "CNN-multilabel.py"),
                                    domain, combined_df, [], scratch_root)
    if model == "PMCNN":
        pmcnn_list = os.path.join(d, "PMCNN_list.csv")
        return make_csv_fit_predict(os.path.join(HERE, "PMCNN", "PMCNN-multilabel.py"),
                                    domain, combined_df, ["--list", pmcnn_list], scratch_root)
    if model == "MetaDR":
        tree = os.path.join(d, "phylogeny.nwk")
        return make_csv_fit_predict(os.path.join(HERE, "MetaDR", "MetaDR-multilabel.py"),
                                    domain, combined_df, ["-t", tree], scratch_root)
    if model == "DeepPhylo":
        return make_deepphylo_fit_predict(domain, X_all, Y_all_raw, dp_dir, scratch_root)
    raise ValueError(model)


def score_multilabel_table(res, domain, model):
    """Sec 5A.5 table shape: one row per flag + a macro row, mean [2.5, 97.5]
    percentile over repeats, for AUPRC / AUPRC-lift / AUROC / MCC / F2 /
    precision / recall / threshold, plus n_pos / n_repeats_scored."""
    rows = []
    for r in res["per_label"]:
        row = {"domain": domain, "model": model, "flag": r["label"], "n_pos": r["n_pos"],
              "prevalence": r["prevalence"], "n_repeats_scored": r["n_repeats_scored"]}
        for k in ["auprc", "auprc_lift", "auroc", "mcc", "f2", "precision", "recall", "threshold"]:
            row[k] = r.get(k, float("nan"))
            row[f"{k}_lo"] = r.get(f"{k}_lo", float("nan"))
            row[f"{k}_hi"] = r.get(f"{k}_hi", float("nan"))
        rows.append(row)
    macro = {"domain": domain, "model": model, "flag": "MACRO", "n_pos": None,
            "prevalence": None, "n_repeats_scored": res["macro"].get("n_labels_scored")}
    for k in ["auprc", "auprc_lift", "auroc", "mcc", "f2", "precision", "recall"]:
        macro[k] = res["macro"].get(f"macro_{k}", float("nan"))
        macro[f"{k}_lo"] = res["macro"].get(f"macro_{k}_lo", float("nan"))
        macro[f"{k}_hi"] = res["macro"].get(f"macro_{k}_hi", float("nan"))
    macro["threshold"] = macro["threshold_lo"] = macro["threshold_hi"] = float("nan")
    rows.append(macro)
    return pd.DataFrame(rows)


def run_all(domains=DOMAINS, models=MODELS, n_splits=5, n_repeats=5):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    scratch_root = os.path.join(RESULTS_DIR, "_scratch")
    os.makedirs(scratch_root, exist_ok=True)

    for domain in domains:
        combined_df, X_all, Y_all_raw, dp_dir, sites, y = load_domain(domain)
        support = audit_site_label_support(y, sites, label_names=FLAG_COLS)
        pd.DataFrame(support).to_csv(
            os.path.join(RESULTS_DIR, f"site_label_support_{domain}.csv"), index=False)
        print(f"=== {domain} === site/label support:")
        for s in support:
            print(f"  {s['label']}: {s['n_pos_samples']} positive samples, "
                 f"{s['n_pos_sites']}/{s['n_sites']} positive sites")

        for model in models:
            print(f"--- {domain} / {model} ---")
            fit_predict = build_fit_predict(model, domain, combined_df, X_all,
                                            Y_all_raw, dp_dir, scratch_root)
            res = evaluate_cv(y, sites, fit_predict, label_names=FLAG_COLS,
                             n_splits=n_splits, n_repeats=n_repeats,
                             pos_weight_cap=POS_WEIGHT_CAP)
            table = score_multilabel_table(res, domain, model)
            out_path = os.path.join(RESULTS_DIR, f"results_{domain}_{model}.csv")
            table.to_csv(out_path, index=False)
            print(table[["flag", "n_pos", "mcc", "f2", "auprc", "auprc_lift"]]
                 .to_string(index=False))
            print(f"-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="+", default=DOMAINS, choices=DOMAINS)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-repeats", type=int, default=5)
    a = ap.parse_args()
    run_all(a.domains, a.models, a.n_splits, a.n_repeats)
