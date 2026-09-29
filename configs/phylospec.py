

# Locate the repository independently of the script's directory depth.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_REPO_ROOT = next(p for p in _RepositoryPath(__file__).resolve().parents
                  if (p / "main.py").is_file() and (p / "configs").is_dir())
if str(_REPO_ROOT) not in _repository_sys.path:
    _repository_sys.path.insert(0, str(_REPO_ROOT))
_repository_sys.path.insert(0, str(_REPO_ROOT / "models/baselines/PhylogenicDL/src"))

import argparse


# train_test
def get_config_train_test():
    parser = argparse.ArgumentParser(description="Configuration for training and testing the model.")

    parser.add_argument('--PhyloSpec', type=str, required=True, choices=['train', 'test' , 'cv'],
                        help="Mode of operation: 'train' , 'test' or 'cv'.")
    parser.add_argument('-c', type=str, required=True, help="Path to the CSV file (train , test or cv).")
    parser.add_argument('-bs', type=int, default=64, help="Batch size.")
    parser.add_argument('-ep', type=int, default=10, help="Number of epochs.")
    parser.add_argument('-lr', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('-ch', type=int, default=16, help="Number of channels in the model.")
    parser.add_argument('-ks', type=int, default=3, help="Kernel size for convolution layers.")
    parser.add_argument('-t', type=str, required=True, help="Path to the Newick format tree file.")
    parser.add_argument('-o', type=str, default='./output/',help="Directory to save model and features. Default is './output/'.")
    parser.add_argument('-taxo', type=str, help="Path to the taxonomy file.")
    parser.add_argument('-labels', type=str, default=None,
                        help="Comma-separated label column names for multi-output/multi-label "
                             "classification (e.g. 'acid_base_balance,buffer_capacity,acid_accumulation,"
                             "ammonia_toxicity,biogas_quality'). If omitted, single-label mode is used "
                             "(the CSV's last column, as before).")
    # os.path.join(os.getcwd(), 'output') or './output/'

    parser.add_argument('-pkl', type=str, default=None,
                        help="Optional path to a .pkl file with predefined 5/10-fold splits.")
    parser.add_argument('-pos_weight_cap', type=float, default=None,
                        help="Cap the per-flag BCEWithLogitsLoss pos_weight (train mode, "
                             "multi-label only) at this value. Uncapped by default; the "
                             "site-grouped CV benchmark passes 10.0 to match the other "
                             "5 model families (task Sec 5A.4).")
    parser.add_argument('-scores_out', type=str, default=None,
                        help="'test' mode only: dump raw (unthresholded) y_scores as a .npy "
                             "to this path instead of computing/printing metrics inline. For "
                             "site-grouped CV: run 'test' once per split (val, test) against "
                             "the same trained model in -o, pointing -scores_out at a "
                             "different file each time.")
    parser.add_argument('-cross_domain_edges', type=str, default=None,
                        help="CLAUDE_CODE_TASK_cross_domain_injection.md Sec A.2: path to a CSV "
                             "with 'bacterium','archaeon','rho' (or 'rho_conditioned') columns "
                             "(feature/leaf names on -t's tree) listing retained cross-domain "
                             "edges (real or, for the parameter-matched null arm, rewired). One "
                             "conv_order merge is appended per edge, weight abs(rho). Omit for "
                             "the tree_only arm. Must be passed identically to 'train' and "
                             "'test'.")
    parser.add_argument('-cross_domain_topk', type=int, default=None,
                        help="Max partners per archaeon allowed in -cross_domain_edges (checked "
                             "via cross_domain_edges.check_edges). None (default) = no cap.")
    parser.add_argument('-val_c', type=str, default=None,
                        help="'train' mode only: path to a validation CSV (same column "
                             "layout as -c). When given together with -patience, training "
                             "checks validation loss after every epoch and stops early "
                             "(restoring the best-val-loss weights) instead of always "
                             "running the full -ep epochs.")
    parser.add_argument('-patience', type=int, default=None,
                        help="Early-stopping patience in epochs (train mode, requires "
                             "-val_c). None (default) disables early stopping.")
    return parser.parse_args()


# feature_importance
def get_config_feature_importance():
    parser = argparse.ArgumentParser(description="Configuration for calculating feature importance.")
    parser.add_argument('-pkl', type=str,  default='./output/Node_Features.pkl', help="Path to the pickle file with data.")
    parser.add_argument('-o', type=str, default='./output/', help="Directory to save model and features. Default is './output/'.")

    return parser.parse_args()


# 16S_Phlyogeny
def get_config_16S_Phlyogeny():
    parser = argparse.ArgumentParser(description="Process OTU sequences and generate phylogenetic tree.")
    parser.add_argument('-c', type=str, required=True, help="Path to the CSV file(species table) with feature names")
    parser.add_argument('-f',  type=str, required=True, help="Path to the FASTA file(database) with sequences")
    parser.add_argument('-o', type=str, default='./output/', help="Directory to save model and features. Default is './output/'.")
    return parser.parse_args()


# WGS_Phlyogeny
def get_config_WGS_Phlyogeny():
    parser = argparse.ArgumentParser(description="Process feature table and generate pruned phylogenetic tree.")
    parser.add_argument('-c',type=str, required=True, help="Path to the CSV file with feature names")
    parser.add_argument('-t',type=str, required=True, help="Path to the phylogenetic tree (nwk format)")
    parser.add_argument('-o', type=str, default='./output/', help="Directory to save model and features. Default is './output/'.")
    return parser.parse_args()