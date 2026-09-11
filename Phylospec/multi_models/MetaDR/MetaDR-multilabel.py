"""
MetaDR baseline for multi-label warning prediction (5 flags at once).

Same idea as MetaDR.py -- rearrange the abundance vector into a pseudo-image
using level-order and post-order phylogenetic tree traversals, then classify
with a small 2D CNN, averaging the level-order and post-order branch
predictions -- but the final FC layer now emits 5 logits with a sigmoid
multi-label head (BCEWithLogitsLoss + per-flag pos_weight) instead of a
softmax over mutually-exclusive classes, and training uses the explicit
train/test split instead of internal 5-fold CV.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from ete3 import Tree
from torch import nn, optim
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_preprocessing"))
from multilabel_metrics import compute_multilabel_metrics, save_results

FLAGS = [
    "acid_accumulation",
    "acid_base_balance",
    "ammonia_toxicity",
    "biogas_quality",
    "buffer_capacity",
]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def transform_image(X, zigzag=False):
    X = np.array(X)
    raw_dim = X.shape[1]
    img_size = int(np.ceil(raw_dim ** 0.5))
    new_dim = img_size ** 2
    n_pad = new_dim - raw_dim
    pad = np.zeros((X.shape[0], n_pad))
    new_X = np.hstack((X, pad)).reshape(X.shape[0], img_size, img_size)
    # is_real tracks which cells came from an actual feature vs. zero-padding,
    # carried through the same zigzag permutation as new_X so it still lines up
    # with new_X's cells afterwards.
    is_real = np.hstack((np.ones((1, raw_dim)), np.zeros((1, n_pad)))).reshape(img_size, img_size)
    is_real = np.broadcast_to(is_real, new_X.shape).copy()

    if zigzag:
        for img, real in zip(new_X, is_real):
            for row in range(img.shape[0]):
                if row % 2 != 0:
                    img[row] = img[row][::-1]
                    real[row] = real[row][::-1]

    new_X = np.log(new_X + 1) / np.log(4)
    # Quantile bin edges from the UNPADDED values only -- CROSS_DOMAIN_INJECTION_
    # BY_MODEL.md Sec 3: computing them on the flattened array including the
    # zero-padding shifts every bin edge, substantially so when raw_dim is far
    # from a perfect square. The padded cells still get binned afterwards (they
    # carry no feature and fall in whichever bin their value -- 0 -- lands in);
    # only the edges are fixed.
    flat_real = new_X[is_real.astype(bool)] if n_pad else new_X.flatten()
    quantiles = np.quantile(flat_real, np.linspace(0, 1, 11))
    bins = [[quantiles[i], quantiles[i + 1]] for i in range(10)]
    color_vals = [0.1 * (i + 1) for i in range(10)]

    for i, (low, high) in enumerate(bins):
        mask = (new_X >= low) & (new_X < high)
        new_X[mask] = color_vals[i]

    return new_X[:, np.newaxis, :, :]


class SimpleCNN(nn.Module):
    def __init__(self, img_size, output_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(20, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        flat_dim = self._flat_dim(img_size)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 500),
            nn.ReLU(),
            nn.Linear(500, output_dim),
        )

    def _flat_dim(self, img_size):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_size, img_size)
            return self.features(dummy).view(1, -1).shape[1]

    def forward(self, x):
        return self.fc(self.features(x))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_metadr(X_train, y_train, output_dim, pos_weight, epochs, lr, seed,
              X_val=None, y_val=None, patience=None):
    """Fit and return the trained model -- separated from prediction so the
    same fitted model can be scored on both the inner-val and outer-test
    splits without retraining.

    When X_val/y_val and patience are given, validation BCE loss is checked
    after every epoch and training stops once it hasn't improved for
    `patience` epochs, restoring the best-val-loss weights instead of always
    running the full `epochs`."""
    set_seed(seed)
    img_size = X_train.shape[-1]
    model = SimpleCNN(img_size, output_dim).to(DEVICE)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(DEVICE) if pos_weight is not None else None)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_data = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

    X_val_t = y_val_t = None
    if X_val is not None and patience:
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    best_state = None
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        if X_val_t is not None:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t).item()

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}/{epochs} "
                         f"(no val-loss improvement for {patience} epochs)")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model


def predict_metadr(model, X):
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        return torch.sigmoid(model(X_t)).cpu().numpy()


def train_and_predict(X_train, y_train, X_test, output_dim, pos_weight, epochs, lr, seed,
                      X_val=None, y_val=None, patience=None):
    """Kept for backward compatibility with the old fit+predict-in-one-call
    interface; delegates to fit_metadr + predict_metadr."""
    model = fit_metadr(X_train, y_train, output_dim, pos_weight, epochs, lr, seed,
                       X_val=X_val, y_val=y_val, patience=patience)
    return predict_metadr(model, X_test)


def main(train_csv, test_csv, tree_path, out_path, domain, seed=42, epochs=60, lr=1e-3,
         use_pos_weight=True, pos_weight_cap=10.0, val_csv=None, scores_out_dir=None,
         seriation_order_path=None, patience=None):
    set_seed(seed)

    train_df = pd.read_csv(train_csv, index_col=0)
    test_df = pd.read_csv(test_csv, index_col=0)
    val_df = pd.read_csv(val_csv, index_col=0) if val_csv else None

    feature_cols = [c for c in train_df.columns if c not in FLAGS]
    X_train_raw = train_df[feature_cols]
    X_test_raw = test_df[feature_cols]
    y_train = train_df[FLAGS].values.astype(np.float32)
    y_test = test_df[FLAGS].values.astype(np.float32)

    tree = Tree(tree_path, format=1)
    level_order = [leaf.name for leaf in tree.traverse("levelorder") if leaf.is_leaf()]
    post_order = [leaf.name for leaf in tree.traverse("postorder") if leaf.is_leaf()]

    taxa_level = [t for t in level_order if t in feature_cols]
    if seriation_order_path:
        # CROSS_DOMAIN_INJECTION_BY_MODEL.md Sec 3, with-syntrophy arm: swap the
        # post-order branch's column order for an optimal-leaf-ordering
        # seriation of the fused distance matrix, so syntrophic partners land
        # in the same convolution window. zigzag stays True (unchanged) so the
        # two arms differ only in this ordering, per the doc's "use the same
        # setting in both arms" note. Without-arm callers simply omit this path.
        seriation = [ln.strip() for ln in open(seriation_order_path) if ln.strip()]
        taxa_post = [t for t in seriation if t in feature_cols]
        missing = set(feature_cols) - set(taxa_post)
        if missing:
            sys.exit(f"ERROR: --seriation-order is missing {len(missing)} of this fold's "
                     f"feature columns, e.g. {sorted(missing)[:5]}")
    else:
        taxa_post = [t for t in post_order if t in feature_cols]

    Xl_train = transform_image(X_train_raw[taxa_level], zigzag=False)
    Xp_train = transform_image(X_train_raw[taxa_post], zigzag=True)
    Xl_test = transform_image(X_test_raw[taxa_level], zigzag=False)
    Xp_test = transform_image(X_test_raw[taxa_post], zigzag=True)

    y_val = None
    Xl_val = Xp_val = None
    if val_df is not None:
        X_val_raw = val_df[feature_cols]
        y_val = val_df[FLAGS].values.astype(np.float32)
        Xl_val = transform_image(X_val_raw[taxa_level], zigzag=False)
        Xp_val = transform_image(X_val_raw[taxa_post], zigzag=True)

    # pos_weight cap defaults to 10.0 to match evaluate_multilabel.pos_weight_
    # from_labels()'s recommendation (task Sec 5A.4); override via --pos-weight-cap.
    if use_pos_weight:
        pos_counts = y_train.sum(axis=0)
        neg_counts = len(y_train) - pos_counts
        pos_weight = torch.tensor(
            np.clip(neg_counts / np.maximum(pos_counts, 1), 1.0, pos_weight_cap),
            dtype=torch.float32)
    else:
        pos_weight = None

    # Fit once per branch (with early stopping against the val split, if
    # given) and reuse the same fitted model to predict both val and test --
    # replaces the old refit-per-target approach, which (with patience unset)
    # is bit-identical since set_seed(seed) at fit_metadr's top made both
    # refits deterministic and equal anyway.
    model_l = fit_metadr(Xl_train, y_train, len(FLAGS), pos_weight, epochs, lr, seed,
                         X_val=Xl_val, y_val=y_val, patience=patience)
    model_p = fit_metadr(Xp_train, y_train, len(FLAGS), pos_weight, epochs, lr, seed,
                         X_val=Xp_val, y_val=y_val, patience=patience)

    probs_l = predict_metadr(model_l, Xl_test)
    probs_p = predict_metadr(model_p, Xp_test)
    y_prob = (probs_l + probs_p) / 2

    if scores_out_dir:
        # Site-grouped CV path: dump raw scores, score centrally (see
        # run_site_grouped_cv_benchmark.py).
        os.makedirs(scores_out_dir, exist_ok=True)
        np.save(os.path.join(scores_out_dir, "test_scores.npy"), y_prob)
        np.save(os.path.join(scores_out_dir, "test_y.npy"), y_test)
        if val_df is not None:
            vprobs_l = predict_metadr(model_l, Xl_val)
            vprobs_p = predict_metadr(model_p, Xp_val)
            np.save(os.path.join(scores_out_dir, "val_scores.npy"), (vprobs_l + vprobs_p) / 2)
            np.save(os.path.join(scores_out_dir, "val_y.npy"), y_val)
        return

    result = compute_multilabel_metrics(y_test, y_prob, FLAGS)
    save_results(result, "MetaDR", domain, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-label MetaDR.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("-t", "--tree", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--out", default=None, help="required unless --scores-out-dir is given")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--no_pos_weight", action="store_true", help="Ablation: use plain BCEWithLogitsLoss (no class-imbalance weighting)")
    parser.add_argument("--pos-weight-cap", type=float, default=10.0)
    parser.add_argument("--val", default=None)
    parser.add_argument("--scores-out-dir", default=None)
    parser.add_argument("--patience", type=int, default=None,
                        help="Early-stopping patience in epochs, using --val's validation "
                             "loss. None (default) disables early stopping.")
    parser.add_argument("--seriation-order", default=None,
                        help="CROSS_DOMAIN_INJECTION_BY_MODEL.md with-syntrophy arm: path to a "
                             "newline-delimited feature-name file (optimal-leaf-ordering "
                             "seriation of the fused distance matrix) replacing the tree's own "
                             "post-order traversal as the post_order/zigzag branch's column "
                             "order. Omit for the without-syntrophy arm.")
    args = parser.parse_args()
    if not args.scores_out_dir and not args.out:
        parser.error("--out is required unless --scores-out-dir is given")
    main(args.train, args.test, args.tree, args.out, args.domain, epochs=args.epochs,
         use_pos_weight=not args.no_pos_weight, pos_weight_cap=args.pos_weight_cap,
         val_csv=args.val, scores_out_dir=args.scores_out_dir,
         seriation_order_path=args.seriation_order, patience=args.patience)
