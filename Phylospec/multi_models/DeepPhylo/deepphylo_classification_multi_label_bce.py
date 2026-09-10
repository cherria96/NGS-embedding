"""
DeepPhylo baseline for multi-label warning prediction (5 flags at once),
matching the evaluation protocol used for the other baselines (RF, CNN,
PMCNN, MetaDR): a single train/test split, per-flag Accuracy / MCC /
ROC-AUC / AUPR, macro-averaged across the 5 flags.

This differs from deepphylo_classification_multi_label.py in one important
way: that script trains DeepPhylo_multi_label (which ends in nn.Sigmoid())
with nn.MSELoss(), which has no mechanism to up-weight the rare "Warning"
class -- several flags have under 5% positives in this data. Here we use
DeepPhylo_multi_label_logits (same architecture, no final Sigmoid) with
nn.BCEWithLogitsLoss(pos_weight=...), where pos_weight is set per-flag from
the training-set class balance, consistent with the other neural baselines.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_preprocessing"))
from deepphylo.pre_dataset import set_seed, reducer, DeepPhyDataset
from deepphylo.model import DeepPhylo_multi_label_logits
from multilabel_metrics import compute_multilabel_metrics, save_results

DEFAULT_FLAGS = [
    "acid_accumulation",
    "acid_base_balance",
    "ammonia_toxicity",
    "biogas_quality",
    "buffer_capacity",
]


def clean_flag_names(label_names):
    return [name.replace("warning_", "") for name in label_names]


def fit_deepphylo(X_train, Y_train, phy_embedding, pos_weight,
                   hidden_size=32, kernel_size_conv=5, kernel_size_pool=1,
                   lr=5e-2, batch_size=8, epochs=150, activation=nn.ReLU()):
    """Fit and return (model, device, train_dataset) -- separated from
    prediction so the same fitted model can be scored on both the inner-val
    and outer-test splits without retraining (site-grouped CV path).
    train_dataset is returned only to reuse its custom_collate_fn."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_labels = Y_train.shape[1]

    train_dataset = DeepPhyDataset(phy_embedding, X_train, Y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=train_dataset.custom_collate_fn,
        drop_last=len(train_dataset) % batch_size == 1,
    )

    model = DeepPhylo_multi_label_logits(
        hidden_size, train_dataset.embeddings, kernel_size_conv,
        kernel_size_pool, activation=activation, n_labels=n_labels,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["X"], batch["nonzero_indices"])
            loss = criterion(logits, batch["y"].reshape(-1, n_labels))
            loss.backward()
            optimizer.step()

    return model, device, train_dataset


def predict_deepphylo(model, device, template_dataset, X, Y, phy_embedding, batch_size=8):
    dataset = DeepPhyDataset(phy_embedding, X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=template_dataset.custom_collate_fn)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            logits = model(batch["X"], batch["nonzero_indices"])
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def train_bce(X_train, Y_train, X_eval, Y_eval, phy_embedding, pos_weight,
               hidden_size=32, kernel_size_conv=5, kernel_size_pool=1,
               lr=5e-2, batch_size=8, epochs=150, activation=nn.ReLU()):
    """Kept for backward compatibility with the old fit+predict-in-one-call
    interface; delegates to fit_deepphylo + predict_deepphylo."""
    model, device, train_dataset = fit_deepphylo(
        X_train, Y_train, phy_embedding, pos_weight, hidden_size,
        kernel_size_conv, kernel_size_pool, lr, batch_size, epochs, activation)
    return predict_deepphylo(model, device, train_dataset, X_eval, Y_eval,
                             phy_embedding, batch_size)


def main(data_dir, out_path, domain, seed=1234, epochs=150, hidden_size=32,
         use_pos_weight=True, pos_weight_cap=10.0, val_dir=None, scores_out_dir=None,
         embedding_path=None):
    set_seed(seed)

    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_eval = np.load(os.path.join(data_dir, "X_eval.npy"))
    Y_train = np.load(os.path.join(data_dir, "Y_train.npy")).astype(np.float32)
    Y_eval = np.load(os.path.join(data_dir, "Y_eval.npy")).astype(np.float32)

    label_path = os.path.join(data_dir, "label_names.txt")
    if os.path.exists(label_path):
        flag_names = clean_flag_names(open(label_path).read().split())
    else:
        flag_names = DEFAULT_FLAGS

    if embedding_path:
        # Cross-domain syntrophy injection (Phylospec/CROSS_DOMAIN_INJECTION_
        # BY_MODEL.md Sec 1): a precomputed embedding -- either the fused
        # (with-injection) or tree-only (without, nested baseline) PCA from
        # build_cross_domain_graph.py -- replaces the usual "PCA the raw
        # patristic matrix here" step entirely. Do not also pass c.npy-based
        # computation; the whole point is that the *same* reduction method
        # (plain PCA, not classical MDS -- a pre-existing quirk, see the doc)
        # was already applied identically to both arms upstream, so nothing
        # here should re-derive or re-fit anything.
        phy_embedding = np.load(embedding_path)
        if phy_embedding.shape[1] != hidden_size:
            print(f"NOTE: --embedding_path has dim {phy_embedding.shape[1]}, "
                 f"not --hidden_size {hidden_size}; using the embedding's own "
                 f"dimensionality (hidden_size is only a PCA target when no "
                 f"embedding_path is given).")
    else:
        C = np.load(os.path.join(data_dir, "c.npy"))
        phy_embedding = reducer(C, "pca", hidden_size, whiten=True)

    # pos_weight cap defaults to 10.0 to match evaluate_multilabel.pos_weight_
    # from_labels()'s recommendation (task Sec 5A.4); override via --pos-weight-cap.
    if use_pos_weight:
        pos_counts = Y_train.sum(axis=0)
        neg_counts = len(Y_train) - pos_counts
        pos_weight = torch.tensor(
            np.clip(neg_counts / np.maximum(pos_counts, 1), 1.0, pos_weight_cap),
            dtype=torch.float32)
    else:
        pos_weight = None

    model, device, train_dataset = fit_deepphylo(
        X_train, Y_train, phy_embedding, pos_weight, hidden_size=hidden_size, epochs=epochs)
    y_prob = predict_deepphylo(model, device, train_dataset, X_eval, Y_eval, phy_embedding)

    if scores_out_dir:
        # Site-grouped CV path: dump raw scores, score centrally (see
        # run_site_grouped_cv_benchmark.py). phy_embedding (PCA of the
        # patristic distance matrix) is fit once on this fold's tree and used
        # for train/val/test alike, same as the non-CV path.
        os.makedirs(scores_out_dir, exist_ok=True)
        np.save(os.path.join(scores_out_dir, "test_scores.npy"), y_prob)
        np.save(os.path.join(scores_out_dir, "test_y.npy"), Y_eval)
        if val_dir:
            X_val = np.load(os.path.join(val_dir, "X_val.npy"))
            Y_val = np.load(os.path.join(val_dir, "Y_val.npy")).astype(np.float32)
            val_prob = predict_deepphylo(model, device, train_dataset, X_val, Y_val, phy_embedding)
            np.save(os.path.join(scores_out_dir, "val_scores.npy"), val_prob)
            np.save(os.path.join(scores_out_dir, "val_y.npy"), Y_val)
        return

    result = compute_multilabel_metrics(Y_eval, y_prob, flag_names)
    save_results(result, "DeepPhylo", domain, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-label DeepPhylo with BCEWithLogitsLoss + pos_weight.")
    parser.add_argument("--data_dir", required=True, help="Dir with X_train.npy, X_eval.npy, Y_train.npy, Y_eval.npy, c.npy")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--out", default=None, help="required unless --scores-out-dir is given")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--no_pos_weight", action="store_true", help="Ablation: use plain BCEWithLogitsLoss (no class-imbalance weighting)")
    parser.add_argument("--pos-weight-cap", type=float, default=10.0)
    parser.add_argument("--val_dir", default=None, help="Dir with X_val.npy, Y_val.npy (inner-validation split)")
    parser.add_argument("--scores-out-dir", default=None)
    parser.add_argument("--embedding_path", default=None,
                        help="Cross-domain syntrophy injection (CROSS_DOMAIN_INJECTION_BY_"
                             "MODEL.md Sec 1): a precomputed (n_taxa, d) embedding .npy from "
                             "build_cross_domain_graph.py, replacing the usual PCA-of-c.npy "
                             "step. Pass the *_treeonly_DeepPhylo_embeding.npy for the "
                             "without-injection arm, *_DeepPhylo_embeding.npy for with -- "
                             "same code path, neutral vs. fused input, per the doc's own rule.")
    args = parser.parse_args()
    if not args.scores_out_dir and not args.out:
        parser.error("--out is required unless --scores-out-dir is given")
    main(args.data_dir, args.out, args.domain, epochs=args.epochs,
         use_pos_weight=not args.no_pos_weight, pos_weight_cap=args.pos_weight_cap,
         val_dir=args.val_dir, scores_out_dir=args.scores_out_dir,
         embedding_path=args.embedding_path)
