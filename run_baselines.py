"""
Run RF, PM-CNN, and MetaDR (from Phylospec/multi_models/) on the same
per-domain per-flag train/test CSVs already built for Phylo-Spec
(Phylospec/example/warnings_{domain}/{flag}/example_{train,test}.csv),
so all four single-label baselines (these three + Phylo-Spec) are compared
on identical, site-grouped splits.

These are faithful-but-generalized reimplementations of the vendored
models' actual architectures (RandomForestClassifier config from RF.py,
the 4-branch Conv1d net from PMCNN.py, the tree-order-image CNN from
MetaDR.py) with two changes needed to run on our data at all:
  1. they use OUR pre-built train/test split instead of re-splitting with
     their own internal (non-site-aware) StratifiedKFold/KFold, and
  2. their fully-connected layer input sizes are computed dynamically
     instead of hardcoded to the original demo dataset's feature count
     (Phylospec/multi_models/Multi_Models_README.md itself flags this as
     a required manual step for PM-CNN and MetaDR).

Usage: python run_baselines.py > baseline_results.json
"""
import json
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from ete3 import Tree
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_BASE = os.path.join(ROOT, "Phylospec", "example")
DOMAINS = ["ARC", "BAC", "merged"]
FLAGS = [
    "acid_base_balance",
    "buffer_capacity",
    "acid_accumulation",
    "ammonia_toxicity",
    "biogas_quality",
]
SEED = 42


def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(domain, flag):
    base = os.path.join(EXAMPLE_BASE, f"warnings_{domain}", flag)
    train_df = pd.read_csv(os.path.join(base, "example_train.csv"))
    test_df = pd.read_csv(os.path.join(base, "example_test.csv"))
    feature_cols = list(train_df.columns[1:-1])
    le = LabelEncoder()
    le.fit(train_df["Group"])
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = le.transform(train_df["Group"])
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = le.transform(test_df["Group"])
    return X_train, y_train, X_test, y_test, feature_cols


def safe_smote(X_train, y_train):
    min_count = np.min(np.bincount(y_train))
    if min_count < 2:
        return X_train, y_train
    k = min(5, min_count - 1)
    return SMOTE(random_state=SEED, k_neighbors=k).fit_resample(X_train, y_train)


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


# ---------------------------------------------------------------- RF -----
def run_rf(X_train, y_train, X_test, y_test):
    set_seed()
    X_train_s, y_train_s = safe_smote(X_train, y_train)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_s)
    X_test_s = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=500, max_depth=5, max_features="log2", random_state=SEED)
    clf.fit(X_train_s, y_train_s)
    probs = clf.predict_proba(X_test_s)[:, 1]
    return safe_auc(y_test, probs)


# ------------------------------------------------------------ PM-CNN -----
class PMCNNNet(nn.Module):
    """4-branch Conv1d net, matching PMCNN.py's Net but with the input
    contiguously quartered (no phylogenetic -list clustering file) and the
    fc1 input size computed from the actual data instead of hardcoded."""

    def __init__(self, input_size, n_branches=4):
        super().__init__()
        self.n_branches = n_branches
        bounds = np.linspace(0, input_size, n_branches + 1).astype(int)
        self.bounds = list(zip(bounds[:-1], bounds[1:]))

        def make_branch(branch_len):
            # PMCNN.py hardcodes kernel_size=8, stride=7, which needs a
            # fairly long input to survive two stacked conv layers; scale
            # both down for domains with few features per branch (e.g. ARC)
            k1 = min(8, max(branch_len, 1))
            s1 = max(1, min(7, branch_len - 1)) if branch_len > 1 else 1
            len1 = (branch_len + 2 * 1 - k1) // s1 + 1
            k2 = min(8, max(len1, 1))
            s2 = max(1, min(7, len1 - 1)) if len1 > 1 else 1
            return nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=k1, stride=s1, padding=1),
                nn.ReLU(),
                nn.Conv1d(16, 16, kernel_size=k2, stride=s2, padding=1),
                nn.ReLU(),
            )

        self.branches = nn.ModuleList([make_branch(hi - lo) for lo, hi in self.bounds])
        with torch.no_grad():
            flat_dim = 0
            for i, (lo, hi) in enumerate(self.bounds):
                dummy = torch.zeros(1, 1, max(hi - lo, 1))
                flat_dim += self.branches[i](dummy).reshape(1, -1).shape[1]
        self.fc1 = nn.Linear(flat_dim, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        parts = []
        for i, (lo, hi) in enumerate(self.bounds):
            xi = x[:, lo:hi].unsqueeze(1)
            parts.append(self.branches[i](xi).reshape(x.size(0), -1))
        h = torch.cat(parts, dim=1)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


def run_pmcnn(X_train, y_train, X_test, y_test, epochs=10):
    set_seed()
    X_train_s, y_train_s = safe_smote(X_train, y_train)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_s)
    X_test_s = scaler.transform(X_test)

    model = PMCNNNet(X_train_s.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_s, dtype=torch.long)
    n = len(y_train_t)
    batch_size = min(64, n)
    drop_last = (n % batch_size == 1)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            if drop_last and len(idx) == 1:
                continue
            optimizer.zero_grad()
            out = model(X_train_t[idx])
            loss = criterion(out, y_train_t[idx])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(torch.tensor(X_test_s, dtype=torch.float32)), dim=1).numpy()
    return safe_auc(y_test, probs[:, 1])


# ------------------------------------------------------------ MetaDR -----
def transform_image(X, zigzag=False):
    X = np.asarray(X, dtype=np.float64)
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
    return new_X[:, np.newaxis, :, :].astype(np.float32), img_size


class MetaDRCNN(nn.Module):
    def __init__(self, img_size, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(20, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_size, img_size)
            flat_dim = self.features(dummy).reshape(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 500),
            nn.ReLU(),
            nn.Linear(500, num_classes),
        )

    def forward(self, x):
        return self.fc(self.features(x))


def _metadr_train_predict(X_train_img, y_train, X_test_img, img_size, num_classes, epochs=5):
    model = MetaDRCNN(img_size, num_classes)
    X_train_t = torch.tensor(X_train_img, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test_img, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(X_test_t), dim=1).numpy()
    return probs


def run_metadr(X_train, y_train, X_test, y_test, feature_cols, tree_path, epochs=5):
    set_seed()
    tree = Tree(tree_path, format=1)
    level_order = [leaf.name for leaf in tree.traverse("levelorder") if leaf.is_leaf()]
    post_order = [leaf.name for leaf in tree.traverse("postorder") if leaf.is_leaf()]
    taxa_level = [c for c in level_order if c in feature_cols]
    taxa_post = [c for c in post_order if c in feature_cols]

    col_idx = {c: i for i, c in enumerate(feature_cols)}
    idx_level = [col_idx[c] for c in taxa_level]
    idx_post = [col_idx[c] for c in taxa_post]

    X_train_s, y_train_s = safe_smote(X_train, y_train)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_s)
    X_test_s = scaler.transform(X_test)
    # transform_image applies its own log/quantile normalization; feed
    # non-negative magnitudes (relative abundances are already >=0, but
    # SMOTE/scaling can introduce negatives, so shift to non-negative)
    X_train_s = X_train_s - X_train_s.min()
    X_test_s = X_test_s - X_test_s.min()

    Xl_train, img_size = transform_image(X_train_s[:, idx_level], zigzag=False)
    Xl_test, _ = transform_image(X_test_s[:, idx_level], zigzag=False)
    Xp_train, img_size_p = transform_image(X_train_s[:, idx_post], zigzag=True)
    Xp_test, _ = transform_image(X_test_s[:, idx_post], zigzag=True)

    num_classes = len(np.unique(y_train_s))
    probs_l = _metadr_train_predict(Xl_train, y_train_s, Xl_test, img_size, num_classes, epochs)
    probs_p = _metadr_train_predict(Xp_train, y_train_s, Xp_test, img_size_p, num_classes, epochs)
    probs_avg = (probs_l + probs_p) / 2
    return safe_auc(y_test, probs_avg[:, 1])


# ----------------------------------------------------------------- main --
def main():
    results = []
    for domain in DOMAINS:
        tree_path = os.path.join(EXAMPLE_BASE, f"warnings_{domain}", "phylogeny.nwk")
        for flag in FLAGS:
            X_train, y_train, X_test, y_test, feature_cols = load_split(domain, flag)
            row = {"domain": domain, "flag": flag}
            try:
                row["RF"] = run_rf(X_train, y_train, X_test, y_test)
            except Exception as e:
                row["RF"] = f"ERROR: {e}"
            try:
                row["PMCNN"] = run_pmcnn(X_train, y_train, X_test, y_test)
            except Exception as e:
                row["PMCNN"] = f"ERROR: {e}"
            try:
                row["MetaDR"] = run_metadr(X_train, y_train, X_test, y_test, feature_cols, tree_path)
            except Exception as e:
                row["MetaDR"] = f"ERROR: {e}"
            print(json.dumps(row), flush=True)
            results.append(row)

    with open(os.path.join(ROOT, "baseline_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
