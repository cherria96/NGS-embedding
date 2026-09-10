"""
PM-CNN baseline for multi-label warning prediction (5 flags at once).

Same four-branch phylogeny-ordered conv architecture as PMCNN.py, but the
final FC layer emits 5 logits with a sigmoid multi-label head
(BCEWithLogitsLoss + per-flag pos_weight) instead of a 2-way softmax, and
training uses the explicit train/test split (with the four phylogenetic
feature orderings from build_pmcnn_groups.py) instead of internal 5-fold CV.
"""
import argparse
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_group_list(list_path):
    df = pd.read_csv(list_path)
    return [list(map(str, row.dropna())) for _, row in df.iterrows()]


class MyDataset(Dataset):
    def __init__(self, x1, x2, x3, x4, y):
        self.x1, self.x2, self.x3, self.x4, self.y = x1, x2, x3, x4, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (self.x1[index], self.x2[index], self.x3[index], self.x4[index]), self.y[index]


class Net(nn.Module):
    def __init__(self, n_features, output_dim):
        super().__init__()
        self.conv1_1 = nn.Conv1d(1, 16, kernel_size=8, stride=7, padding=1)
        self.conv1_2 = nn.Conv1d(16, 16, kernel_size=8, stride=7, padding=1)
        self.conv2_1 = nn.Conv1d(1, 16, kernel_size=8, stride=7, padding=1)
        self.conv2_2 = nn.Conv1d(16, 16, kernel_size=8, stride=7, padding=1)
        self.conv3_1 = nn.Conv1d(1, 16, kernel_size=8, stride=7, padding=1)
        self.conv3_2 = nn.Conv1d(16, 16, kernel_size=8, stride=7, padding=1)
        self.conv4_1 = nn.Conv1d(1, 16, kernel_size=8, stride=7, padding=1)
        self.conv4_2 = nn.Conv1d(16, 16, kernel_size=8, stride=7, padding=1)

        flat_dim = self._branch_out_dim(n_features) * 4
        self.fc1 = nn.Linear(flat_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def _branch_out_dim(self, n_features):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_features)
            out = F.relu(self.conv1_1(dummy))
            out = F.relu(self.conv1_2(out))
            return out.view(1, -1).shape[1]

    def conv_block(self, x, conv1, conv2):
        x = F.relu(conv1(x))
        x = F.relu(conv2(x))
        return x

    def forward(self, x1, x2, x3, x4):
        x1 = self.conv_block(x1.unsqueeze(1), self.conv1_1, self.conv1_2).view(x1.size(0), -1)
        x2 = self.conv_block(x2.unsqueeze(1), self.conv2_1, self.conv2_2).view(x2.size(0), -1)
        x3 = self.conv_block(x3.unsqueeze(1), self.conv3_1, self.conv3_2).view(x3.size(0), -1)
        x4 = self.conv_block(x4.unsqueeze(1), self.conv4_1, self.conv4_2).view(x4.size(0), -1)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def to_branches(X_df, group_list):
    return [torch.tensor(X_df[cols].values, dtype=torch.float32) for cols in group_list]


def main(train_csv, test_csv, list_csv, out_path, domain, seed=42, epochs=60, lr=1e-4,
         use_pos_weight=True, pos_weight_cap=10.0, val_csv=None, scores_out_dir=None):
    set_seed(seed)
    group_list = load_group_list(list_csv)

    train_df = pd.read_csv(train_csv, index_col=0)
    test_df = pd.read_csv(test_csv, index_col=0)

    X_train_raw = train_df.drop(columns=FLAGS)
    X_test_raw = test_df.drop(columns=FLAGS)
    y_train = train_df[FLAGS].values.astype(np.float32)
    y_test = test_df[FLAGS].values.astype(np.float32)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)

    train_branches = to_branches(X_train_scaled, group_list)
    test_branches = to_branches(X_test_scaled, group_list)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    n_features = train_branches[0].shape[1]

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

    train_dataset = MyDataset(*train_branches, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = Net(n_features=n_features, output_dim=len(FLAGS))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            x1, x2, x3, x4 = inputs
            y_pred = model(x1, x2, x3, x4)
            loss = criterion(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()

    def predict(branches):
        with torch.no_grad():
            return torch.sigmoid(model(*branches)).numpy()

    y_prob = predict(test_branches)

    if scores_out_dir:
        # Site-grouped CV path: dump raw scores, score centrally (see
        # run_site_grouped_cv_benchmark.py).
        os.makedirs(scores_out_dir, exist_ok=True)
        np.save(os.path.join(scores_out_dir, "test_scores.npy"), y_prob)
        np.save(os.path.join(scores_out_dir, "test_y.npy"), y_test)
        if val_csv:
            val_df = pd.read_csv(val_csv, index_col=0)
            X_val_raw = val_df.drop(columns=FLAGS)
            y_val = val_df[FLAGS].values.astype(np.float32)
            X_val_scaled = pd.DataFrame(scaler.transform(X_val_raw), columns=X_val_raw.columns, index=X_val_raw.index)
            val_branches = to_branches(X_val_scaled, group_list)
            np.save(os.path.join(scores_out_dir, "val_scores.npy"), predict(val_branches))
            np.save(os.path.join(scores_out_dir, "val_y.npy"), y_val)
        return

    result = compute_multilabel_metrics(y_test, y_prob, FLAGS)
    save_results(result, "PMCNN", domain, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-label PM-CNN.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--list", required=True, help="PMCNN feature-ordering CSV")
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
    main(args.train, args.test, args.list, args.out, args.domain, epochs=args.epochs,
         use_pos_weight=not args.no_pos_weight, pos_weight_cap=args.pos_weight_cap,
         val_csv=args.val, scores_out_dir=args.scores_out_dir)
