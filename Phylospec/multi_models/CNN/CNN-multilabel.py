"""
1D-CNN baseline for multi-label warning prediction (5 flags at once).

Same convolutional backbone as CNN.py / CNN-multi.py, but the final FC layer
now emits 5 logits (one per flag) trained jointly with BCEWithLogitsLoss
(sigmoid multi-label head) instead of a single softmax/sigmoid output over
mutually-exclusive classes. Per-flag pos_weight compensates for the strong
class imbalance (some flags have <5% positives) instead of SMOTE, which does
not generalize cleanly to multi-label targets.
"""
import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

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


def convolution_block(in_channels, out_channels, kernel_size=3, padding="same"):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.Dropout(p=0.5),
        nn.ReLU(inplace=False),
    )


class CNNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = convolution_block(1, 16, kernel_size=3)
        self.conv2 = convolution_block(16, 8, kernel_size=5)
        self.conv3 = convolution_block(8, 4, kernel_size=7)
        self.conv4 = convolution_block(4, 2, kernel_size=9)
        self.fc1 = nn.Linear(2 * input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def main(train_csv, test_csv, out_path, domain, seed=42, epochs=60, lr=1e-4,
         use_pos_weight=True, pos_weight_cap=10.0, val_csv=None, scores_out_dir=None):
    set_seed(seed)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    X_train = train_df.drop(columns=["SampleID"] + FLAGS).values
    y_train = train_df[FLAGS].values.astype(np.float32)
    X_test = test_df.drop(columns=["SampleID"] + FLAGS).values
    y_test = test_df[FLAGS].values.astype(np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    input_dim = X_train.shape[1]

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    # pos_weight = n_neg / n_pos per flag, capped to avoid exploding on rare
    # flags -- cap defaults to 10.0 to match evaluate_multilabel.pos_weight_
    # from_labels()'s recommendation (task Sec 5A.4), overridable via
    # --pos-weight-cap so every model in a benchmark run uses the same cap.
    if use_pos_weight:
        pos_counts = y_train.sum(axis=0)
        neg_counts = len(y_train) - pos_counts
        pos_weight = torch.tensor(
            np.clip(neg_counts / np.maximum(pos_counts, 1), 1.0, pos_weight_cap),
            dtype=torch.float32)
    else:
        pos_weight = None

    model = CNNModel(input_dim=input_dim, output_dim=len(FLAGS))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_data = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()

    def predict(X_t):
        with torch.no_grad():
            return torch.sigmoid(model(X_t)).numpy()

    y_prob = predict(X_test_t)

    if scores_out_dir:
        # Site-grouped CV path: dump raw scores, score centrally (see
        # run_site_grouped_cv_benchmark.py) so threshold tuning and macro
        # metrics are identical across all 5 model families.
        os.makedirs(scores_out_dir, exist_ok=True)
        np.save(os.path.join(scores_out_dir, "test_scores.npy"), y_prob)
        np.save(os.path.join(scores_out_dir, "test_y.npy"), y_test)
        if val_csv:
            val_df = pd.read_csv(val_csv)
            X_val = scaler.transform(val_df.drop(columns=["SampleID"] + FLAGS).values)
            y_val = val_df[FLAGS].values.astype(np.float32)
            X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
            np.save(os.path.join(scores_out_dir, "val_scores.npy"), predict(X_val_t))
            np.save(os.path.join(scores_out_dir, "val_y.npy"), y_val)
        return

    result = compute_multilabel_metrics(y_test, y_prob, FLAGS)
    save_results(result, "CNN", domain, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-label 1D-CNN.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
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
    main(args.train, args.test, args.out, args.domain, epochs=args.epochs,
         use_pos_weight=not args.no_pos_weight, pos_weight_cap=args.pos_weight_cap,
         val_csv=args.val, scores_out_dir=args.scores_out_dir)
