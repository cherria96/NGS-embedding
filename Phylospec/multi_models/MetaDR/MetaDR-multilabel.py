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
    pad = np.zeros((X.shape[0], new_dim - raw_dim))
    new_X = np.hstack((X, pad)).reshape(X.shape[0], img_size, img_size)

    if zigzag:
        for img in new_X:
            for row in range(img.shape[0]):
                if row % 2 != 0:
                    img[row] = img[row][::-1]

    new_X = np.log(new_X + 1) / np.log(4)
    flat = new_X.flatten()
    quantiles = np.quantile(flat, np.linspace(0, 1, 11))
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


def train_and_predict(X_train, y_train, X_test, output_dim, pos_weight, epochs, lr, seed):
    set_seed(seed)
    img_size = X_train.shape[-1]
    model = SimpleCNN(img_size, output_dim)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_data = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

    model.train()
    for _ in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test_t)).numpy()
    return probs


def main(train_csv, test_csv, tree_path, out_path, domain, seed=42, epochs=60, lr=1e-3,
         use_pos_weight=True, pos_weight_cap=10.0, val_csv=None, scores_out_dir=None):
    set_seed(seed)

    train_df = pd.read_csv(train_csv, index_col=0)
    test_df = pd.read_csv(test_csv, index_col=0)

    feature_cols = [c for c in train_df.columns if c not in FLAGS]
    X_train_raw = train_df[feature_cols]
    X_test_raw = test_df[feature_cols]
    y_train = train_df[FLAGS].values.astype(np.float32)
    y_test = test_df[FLAGS].values.astype(np.float32)

    tree = Tree(tree_path, format=1)
    level_order = [leaf.name for leaf in tree.traverse("levelorder") if leaf.is_leaf()]
    post_order = [leaf.name for leaf in tree.traverse("postorder") if leaf.is_leaf()]

    taxa_level = [t for t in level_order if t in feature_cols]
    taxa_post = [t for t in post_order if t in feature_cols]

    Xl_train = transform_image(X_train_raw[taxa_level], zigzag=False)
    Xp_train = transform_image(X_train_raw[taxa_post], zigzag=True)
    Xl_test = transform_image(X_test_raw[taxa_level], zigzag=False)
    Xp_test = transform_image(X_test_raw[taxa_post], zigzag=True)

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

    probs_l = train_and_predict(Xl_train, y_train, Xl_test, len(FLAGS), pos_weight, epochs, lr, seed)
    probs_p = train_and_predict(Xp_train, y_train, Xp_test, len(FLAGS), pos_weight, epochs, lr, seed)
    y_prob = (probs_l + probs_p) / 2

    if scores_out_dir:
        # Site-grouped CV path: dump raw scores, score centrally (see
        # run_site_grouped_cv_benchmark.py). NOTE: train_and_predict() refits
        # from scratch each call (same set_seed(seed) at its top), so calling
        # it again for the val split reproduces the exact same fitted model
        # rather than training twice with different weights.
        os.makedirs(scores_out_dir, exist_ok=True)
        np.save(os.path.join(scores_out_dir, "test_scores.npy"), y_prob)
        np.save(os.path.join(scores_out_dir, "test_y.npy"), y_test)
        if val_csv:
            val_df = pd.read_csv(val_csv, index_col=0)
            X_val_raw = val_df[feature_cols]
            y_val = val_df[FLAGS].values.astype(np.float32)
            Xl_val = transform_image(X_val_raw[taxa_level], zigzag=False)
            Xp_val = transform_image(X_val_raw[taxa_post], zigzag=True)
            vprobs_l = train_and_predict(Xl_train, y_train, Xl_val, len(FLAGS), pos_weight, epochs, lr, seed)
            vprobs_p = train_and_predict(Xp_train, y_train, Xp_val, len(FLAGS), pos_weight, epochs, lr, seed)
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
    args = parser.parse_args()
    if not args.scores_out_dir and not args.out:
        parser.error("--out is required unless --scores-out-dir is given")
    main(args.train, args.test, args.tree, args.out, args.domain, epochs=args.epochs,
         use_pos_weight=not args.no_pos_weight, pos_weight_cap=args.pos_weight_cap,
         val_csv=args.val, scores_out_dir=args.scores_out_dir)
