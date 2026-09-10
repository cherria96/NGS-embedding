import os
import sys
import json
import torch
import numpy as np
import joblib
from Bio import Phylo
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from Phylospec.src.model.data_processing import load_and_preprocess_data, match_leaf_nodes, assign_unique_names, get_conv_order, \
    calculate_node_weights, save_node_features_with_pickle, process_unclassified_features, tree_p
from Phylospec.src.model.PhyloSpec import PhyloSpec, AuxiliaryModel, calculate_fc1_input_dim
sys.path.append('./')
from Phylospec.src.global_config import get_config_train_test
from Phylospec.src.model.training_evaluating import train_model, evaluate_model_on_test

def set_seed(seed):
    """Ensure reproducibility by setting all relevant seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

label_encoder = LabelEncoder()

def parse_label_cols(config):
    """Comma-separated -labels arg -> list of column names, or None for single-label mode."""
    raw = getattr(config, 'labels', None)
    return [s.strip() for s in raw.split(',')] if raw else None

def train_model_function(config, seed):
    """Train PhyloSpec model on the training set and save results."""
    newick_path = config.t
    train_csv_path = config.c
    taxonomy_path = config.taxo
    label_cols = parse_label_cols(config)

    # Load phylogenetic tree and assign unique names to nodes
    tree = tree_p(train_csv_path, newick_path, label_cols=label_cols)
    tree = Phylo.read(tree, 'newick')
    tree = assign_unique_names(tree)

    # Load and preprocess training data
    X_train, y_train, encoder, data_train = load_and_preprocess_data(train_csv_path, tree, label_cols=label_cols)

    label_cols_resolved = label_cols if label_cols else [data_train.columns[-1]]
    n_labels = len(label_cols_resolved)
    multi_label = n_labels > 1

    # Handle unclassified features if present
    if any("Unclassified" in col or "unclassified" in col for col in data_train.columns):
        data_train, tree = process_unclassified_features(tree, data_train, taxonomy_path, label_cols=label_cols_resolved)
        X_train = data_train.iloc[:, 1:-n_labels].values
        if multi_label:
            y_train = data_train[label_cols_resolved].values.astype(np.float32)
        else:
            y_train = label_encoder.fit_transform(data_train.iloc[:, -1].values)

    num_classes = n_labels if multi_label else len(np.unique(y_train))
    data_features = data_train.drop(columns=label_cols_resolved)

    # Match leaves and build convolution order
    leaf_to_species = match_leaf_nodes(tree, data_features)
    nodes, parents, conv_order, node_relations = get_conv_order(tree)
    node_weights = calculate_node_weights(tree)

    if multi_label:
        # imblearn's SMOTE only supports single-column labels; skip resampling
        # and instead weight each flag's positive class in the loss (below).
        print("Skipping SMOTE for multi-label classification")
        pos_counts = y_train.sum(axis=0)
        neg_counts = y_train.shape[0] - pos_counts
        pos_weight = neg_counts / np.clip(pos_counts, 1, None)
        pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)
    else:
        # SMOTE's default k_neighbors=5 needs >=6 samples in the smallest class;
        # fall back to a smaller k (or skip SMOTE entirely below 2 samples) for
        # rare classes instead of crashing.
        min_class_count = np.min(np.bincount(y_train))
        if min_class_count < 2:
            print(f"Skipping SMOTE: smallest class has only {min_class_count} sample(s)")
        else:
            k_neighbors = min(5, min_class_count - 1)
            smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
            X_train, y_train = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32 if multi_label else torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Prepare auxiliary model to determine input dimension for FC layer
    aux_model = AuxiliaryModel(channel=config.ch, kernel_size=config.ks)
    fc1_input_dim = calculate_fc1_input_dim(aux_model, X_train, conv_order, data_features, leaf_to_species,
                                            node_weights)

    # Initialize model and select loss function based on classification type
    if multi_label:
        final_model = PhyloSpec(fc1_input_dim=fc1_input_dim, num_res_blocks=1, channel=config.ch,
                                kernel_size=config.ks, out_feature=n_labels).to('cpu')
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    elif num_classes == 2:
        final_model = PhyloSpec(fc1_input_dim=fc1_input_dim, num_res_blocks=1, channel=config.ch,
                                kernel_size=config.ks, out_feature=1).to('cpu')
        criterion = nn.BCEWithLogitsLoss()
    else:
        final_model = PhyloSpec(fc1_input_dim=fc1_input_dim, num_res_blocks=1, channel=config.ch,
                                kernel_size=config.ks, out_feature=num_classes).to('cpu')
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(final_model.parameters(), lr=config.lr, weight_decay=0.0001)

    print("Training the model on the entire training set...")

    # Start training loop
    final_model = train_model(
        final_model, train_loader, criterion, optimizer, conv_order, data_features, leaf_to_species,
        node_weights=node_weights, num_epochs=config.ep, num_classes=num_classes
    )

    # Save with node features
    all_train_labels = []
    for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
        all_train_labels.append(batch_labels)
    all_train_labels = torch.cat(all_train_labels, dim=0).numpy()

    # Save training data
    node_features = final_model.accumulated_node_features
    save_node_features_with_pickle(
        node_features,
        node_relations,
        node_weights,
        all_train_labels,
        os.path.join(config.o, 'Node_Features.pkl')
    )

    scaler_path = os.path.join(config.o, 'StandardScaler.pkl')
    joblib.dump(scaler, scaler_path)
    torch.save(final_model, config.o + 'train_model.pth')

    print("End of training")

def test_model_function(config, seed):
    """Load model and evaluate it on the test set."""
    set_seed(seed)

    newick_path = config.t
    test_csv_path = config.c
    taxonomy_path = config.taxo
    label_cols = parse_label_cols(config)

    # Load phylogenetic tree and test data
    tree = tree_p(test_csv_path, newick_path, label_cols=label_cols)
    tree = Phylo.read(tree, 'newick')
    tree = assign_unique_names(tree)

    X_test, y_test, _, data_test = load_and_preprocess_data(test_csv_path, tree, label_cols=label_cols)

    label_cols_resolved = label_cols if label_cols else [data_test.columns[-1]]
    n_labels = len(label_cols_resolved)
    multi_label = n_labels > 1

    # Handle unclassified features if present
    if any("Unclassified" in col or "unclassified" in col for col in data_test.columns):
        data_test, tree = process_unclassified_features(tree, data_test, taxonomy_path, label_cols=label_cols_resolved)
        X_test = data_test.iloc[:, 1:-n_labels].values
        if multi_label:
            y_test = data_test[label_cols_resolved].values.astype(np.float32)
        else:
            y_test = label_encoder.fit_transform(data_test.iloc[:, -1].values)

    num_classes = n_labels if multi_label else len(np.unique(y_test))
    data_features = data_test.drop(columns=label_cols_resolved)

    # Preprocess tree structure
    leaf_to_species = match_leaf_nodes(tree, data_features)
    nodes, parents, conv_order, node_relations = get_conv_order(tree)
    node_weights = calculate_node_weights(tree)

    # Normalize test data using training scaler
    scaler = joblib.load(config.o + 'StandardScaler.pkl')
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32 if multi_label else torch.long)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load trained model
    final_model = torch.load(config.o + 'train_model.pth')

    print("Testing the model on the test set...")

    # Predict and evaluate
    y_true, y_scores = evaluate_model_on_test(final_model, test_loader, conv_order, data_features, leaf_to_species,
                                              node_weights, num_classes=num_classes, multi_label=multi_label)

    # Compute AUC / AUPR / MCC scores
    if multi_label:
        y_pred = (y_scores >= 0.5).astype(int)
        per_flag_auc = {}
        per_flag_aupr = {}
        per_flag_mcc = {}
        for i, flag in enumerate(label_cols_resolved):
            try:
                per_flag_auc[flag] = float(roc_auc_score(y_true[:, i], y_scores[:, i]))
            except ValueError:
                per_flag_auc[flag] = float('nan')  # only one class present in this flag's test split
            try:
                per_flag_aupr[flag] = float(average_precision_score(y_true[:, i], y_scores[:, i]))
            except ValueError:
                per_flag_aupr[flag] = float('nan')
            try:
                per_flag_mcc[flag] = float(matthews_corrcoef(y_true[:, i], y_pred[:, i]))
            except ValueError:
                per_flag_mcc[flag] = float('nan')  # only one class present in this flag's test split

        macro_auc = float(np.nanmean(list(per_flag_auc.values())))
        macro_aupr = float(np.nanmean(list(per_flag_aupr.values())))
        macro_mcc = float(np.nanmean(list(per_flag_mcc.values())))

        for flag in label_cols_resolved:
            print(f"{flag}: AUC={per_flag_auc[flag]:.4f}  AUPR={per_flag_aupr[flag]:.4f}  MCC={per_flag_mcc[flag]:.4f}")
        print(f"Macro-average across {n_labels} flags: AUC={macro_auc:.4f}  AUPR={macro_aupr:.4f}  MCC={macro_mcc:.4f}")

        with open(os.path.join(config.o, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({
                "per_flag_auc": per_flag_auc,
                "per_flag_aupr": per_flag_aupr,
                "per_flag_mcc": per_flag_mcc,
                "macro_auc": macro_auc,
                "macro_aupr": macro_aupr,
                "macro_mcc": macro_mcc,
                "hamming_loss": hamming,
            }, f, indent=2)
    elif num_classes == 2:
        auc_val = roc_auc_score(y_true, y_scores)
        print(f"ROC AUC (Binary): {auc_val:.4f}")
    else:
        for i in range(num_classes):
            auc_val = roc_auc_score(y_true == i, y_scores[:, i])
            print(f"Class {i} AUC (Multiclass): {auc_val:.4f}")

def main():
    config = get_config_train_test()
    seed = 42
    set_seed(seed)

    if config.PhyloSpec == 'train':
        train_model_function(config, seed)
    elif config.PhyloSpec == 'test':
        test_model_function(config, seed)
    else:
        print("Invalid mode. Use 'train' or 'test'.")
        return

if __name__ == '__main__':
    main()
