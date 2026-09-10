"""Stateless splitting, search spaces and serialization for independent Order analysis."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from order_macro_models import TARGETS
ROOT=Path(__file__).resolve().parents[1]
SPACES = {
    'RandomForest': {'n_estimators': [150, 300], 'max_depth': [3, 6, None], 'min_samples_leaf': [1, 3, 5], 'max_features': ['sqrt', .5]},
    'KNN': {'n_neighbors': [3, 5, 9, 15], 'weights': ['uniform', 'distance'], 'p': [1, 2]},
    'SVM': {'C': [.1, 1, 10, 100], 'gamma': ['scale', .001, .01], 'kernel': ['rbf', 'linear']},
    'NNET': {'hidden_layer_sizes': [(16,), (32,), (32, 16)], 'alpha': [.01, .1, 1, 10], 'solver': ['lbfgs']},
    'XGBoost': {'n_estimators': [50, 100, 200], 'max_depth': [1, 2, 3], 'learning_rate': [.03, .1], 'reg_lambda': [1, 10], 'subsample': [.8, 1.]},
    'GLMNET': {'C': [.01, .1, 1, 10], 'l1_ratio': [.1, .5, .9]}
}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')


def label_report(y, scope):
    return [{'scope': scope, 'target': t, 'samples': len(y), 'positive': int(y[:, j].sum()),
             'positive_percent': float(100*y[:, j].mean())} for j, t in enumerate(TARGETS)]


def partition(groups, y, proportions, seed, trials=5000):
    """Random-search group allocation balancing sample and site-positive counts.

    The objective uses labels only to stratify the split, never model performance.
    Site counts per partition are fixed; all samples of a group stay together.
    """
    sites = np.array(sorted(set(groups)))
    counts = np.array([y[groups == s].sum(axis=0) for s in sites])
    positives = np.array([y[groups == s].max(axis=0) for s in sites])
    sizes = np.array([(groups == s).sum() for s in sites])
    cuts = np.rint(np.cumsum(proportions)[:-1]*len(sites)).astype(int)
    rng = np.random.default_rng(seed)
    best_score, best = np.inf, None
    for _ in range(trials):
        pieces = np.split(rng.permutation(len(sites)), cuts)
        score = 0.
        for part, fraction in zip(pieces, proportions):
            observed = counts[part].sum(axis=0)
            score += np.square((observed-fraction*counts.sum(axis=0))/np.maximum(counts.sum(axis=0), 1)).sum()
            score += np.square((positives[part].sum(axis=0)-fraction*positives.sum(axis=0))/np.maximum(positives.sum(axis=0), 1)).sum()
            score += ((sizes[part].sum()/sizes.sum())-fraction)**2
            # Prefer coverage when feasible, especially positive training coverage.
            score += 5 * ((observed == 0) & (counts.sum(axis=0) > 0)).sum()
            score += 20 * ((counts.sum(axis=0)-observed == 0) & (counts.sum(axis=0) > 0)).sum()
        if score < best_score:
            best_score, best = score, pieces
    return [np.flatnonzero(np.isin(groups, sites[part])) for part in best]


def predictions(out, name, scope, frame, y, p, threshold):
    df = frame[['Site', 'SiteGroup', 'Season']].copy()
    for j, t in enumerate(TARGETS):
        df[t+'_actual'] = y[:, j]
        df[t+'_probability'] = p[:, j]
        cutoff = float(threshold) if np.ndim(threshold) == 0 else threshold[j]
        df[t+'_predicted'] = (p[:, j] >= cutoff).astype(int)
    df.to_csv(out/f'{name}_{scope}_predictions.csv')
