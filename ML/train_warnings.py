#!/usr/bin/env python3
"""Site-grouped multi-label warning comparison; see README_warnings.md."""
from pathlib import Path
import argparse
import hashlib
import json
import warnings
import platform
import sys
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import (average_precision_score, matthews_corrcoef, confusion_matrix,
                             precision_recall_curve, roc_curve, roc_auc_score)
from warning_models import TARGETS, CATS, NUMS, BLOCKS, SampleTopTaxa, WarningModel, numeric

ROOT = Path(__file__).resolve().parents[1]
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


def metrics(y, p, threshold=.5):
    result = {}
    for j, label in enumerate(TARGETS):
        # No positive examples: AP cannot measure detection; explicitly report NA.
        result['MCC_' + label] = float(matthews_corrcoef(y[:, j], p[:, j] >= threshold))
        result['AUPR_' + label] = float(average_precision_score(y[:, j], p[:, j])) if y[:, j].sum() else np.nan
    result['Macro_MCC'] = float(np.mean([result['MCC_' + t] for t in TARGETS]))
    ap = [result['AUPR_' + t] for t in TARGETS]
    result['Macro_AUPR'] = float(np.nanmean(ap)) if np.isfinite(ap).any() else np.nan
    result['AUPR_evaluable_labels'] = int(np.isfinite(ap).sum())
    result['Micro_AUPR'] = float(average_precision_score(y.ravel(), p.ravel())) if y.sum() else np.nan
    return result


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


def inspect(data_dir, out, group_map):
    m = pd.read_csv(data_dir/'metadata.csv', dtype={'SampleID': str, 'No': str})
    required = ['SampleID', 'Site', 'Season', 'Round', 'No'] + NUMS + ['substrate_type'] + TARGETS
    if set(required)-set(m.columns):
        raise ValueError(f'Missing required metadata columns: {set(required)-set(m.columns)}')
    if m.SampleID.isna().any() or m.SampleID.duplicated().any() or m[['Site', 'Season']].isna().any().any():
        raise ValueError('Missing/duplicate metadata IDs or missing Site/Season')
    if not m[TARGETS].isin([0, 1]).all().all():
        raise ValueError('Targets must contain only 0 or 1')
    if not (m.SampleID == m.Round.astype(str) + '-' + m.No).all():
        raise ValueError('SampleID disagrees with Round/No')
    seasons = {1: 'Summer', 2: 'Fall', 3: 'Winter', 4: 'Spring'}
    if not (m.Round.map(seasons) == m.Season).all():
        raise ValueError('Round/Season mismatch')
    if m.groupby('No').Site.nunique().max() != 1 or m.groupby('Site').No.nunique().max() != 1:
        raise ValueError('Site/No mapping is inconsistent')
    m['SiteGroup'] = m.Site.replace(group_map)
    m = m.set_index('SampleID')
    sheets, structures, mappings = {}, [], []
    for domain in ['BAC', 'ARC']:
        path = data_dir/f'Dat_{domain}.xlsx'
        book = pd.ExcelFile(path)
        for sheet in book.sheet_names:
            frame = pd.read_excel(book, sheet_name=sheet)
            structures.append({'file': path.name, 'sheet': sheet, 'data_rows': len(frame), 'columns': len(frame.columns), 'leading_columns': '|'.join(map(str, frame.columns[:8]))})
        for rank in ['P', 'G']:
            frame = pd.read_excel(book, sheet_name=f'{rank}(%)', index_col=0)
            expected = {'P': 'Phylum', 'G': 'Genus'}[rank]
            if frame.index.name != expected or frame.index.has_duplicates or frame.index.isna().any():
                raise ValueError(f'Invalid {expected} taxonomic labels')
            original = frame.columns.tolist()
            ids = []
            for full in original:
                parts = str(full).split('-', 2)
                if len(parts) != 3:
                    raise ValueError(f'Unexpected genomic ID: {full}')
                sid = '-'.join(parts[:2]); ids.append(sid)
                mappings.append({'domain': domain, 'rank': rank, 'workbook_sample': full, 'SampleID': sid, 'workbook_suffix': parts[2], 'Site': m.loc[sid, 'Site'] if sid in m.index else None})
            if len(set(ids)) != len(ids):
                raise ValueError('Duplicate normalized workbook sample IDs')
            frame.columns = ids
            frame = frame.T.apply(pd.to_numeric, errors='raise')
            if not np.isfinite(frame.values).all() or (frame.values < 0).any():
                raise ValueError('Invalid relative abundance')
            frame.columns = [f'{domain}_{rank}::{taxon}' for taxon in frame.columns]
            sheets[f'{domain}_{rank}'] = frame
        if not sheets[f'{domain}_P'].index.equals(sheets[f'{domain}_G'].index):
            raise ValueError('Phylum/Genus sample columns do not agree')
        book.close()
    pd.DataFrame(structures).to_csv(out/'file_structure.csv', index=False)
    pd.DataFrame(mappings).to_csv(out/'sample_id_mapping.csv', index=False)
    all_ids = sorted(set(m.index).union(*(set(f.index) for f in sheets.values())))
    audit = pd.DataFrame(index=pd.Index(all_ids, name='SampleID'))
    audit['metadata'] = audit.index.isin(m.index)
    for block, frame in sheets.items():
        audit[block] = audit.index.isin(frame.index)
    audit['included'] = audit.all(axis=1)
    audit.to_csv(out/'sample_audit.csv')
    print('Unmatched samples (excluded and recorded):', audit.index[~audit.included].tolist(), flush=True)
    original_targets = m[TARGETS].to_numpy(int)
    label_rows = label_report(original_targets, 'all_metadata')
    simultaneous = []
    for scope, yy in [('all_metadata', original_targets)]:
        simultaneous.extend({'scope': scope, 'simultaneous_labels': n, 'samples': int((yy.sum(axis=1) == n).sum()), 'percent': float(100*(yy.sum(axis=1) == n).mean())} for n in range(6))
    included = m.index[m.index.isin(audit.index[audit.included])]
    m = m.loc[included].copy()
    coercions = []
    for c in NUMS:
        parsed = m[c].map(numeric).replace([np.inf, -np.inf], np.nan)
        for sid in m.index[m[c].notna() & pd.to_numeric(m[c], errors='coerce').isna()]:
            coercions.append({'SampleID': sid, 'column': c, 'original': m.loc[sid, c], 'parsed': parsed.loc[sid]})
        m[c] = parsed
    pd.DataFrame(coercions, columns=['SampleID', 'column', 'original', 'parsed']).to_csv(out/'numeric_coercions.csv', index=False)
    X = pd.concat([m[CATS+NUMS]]+[f.loc[included] for f in sheets.values()], axis=1)
    y = m[TARGETS].to_numpy(int)
    label_rows += label_report(y, 'matched')
    simultaneous.extend({'scope': 'matched', 'simultaneous_labels': n, 'samples': int((y.sum(axis=1) == n).sum()), 'percent': float(100*(y.sum(axis=1) == n).mean())} for n in range(6))
    pd.DataFrame(simultaneous).to_csv(out/'simultaneous_labels.csv', index=False)
    pd.DataFrame({'missing': X.isna().sum(), 'missing_percent': 100*X.isna().mean()}).to_csv(out/'missing_features.csv')
    pd.concat([m, X.drop(columns=CATS+NUMS)], axis=1).to_csv(out/'cleaned_merged.csv')
    m.groupby(['SiteGroup', 'Site']).agg(samples=('Season', 'size'), seasons=('Season', lambda s: '|'.join(s))).to_csv(out/'samples_per_site.csv')
    source = ['eff_pH', 'eff_ALK', 'eff_TVFAs', 'eff_HPro', 'eff_HAc', 'eff_TAN', 'eff_CH4']
    m[source].isna().to_csv(out/'target_source_missingness.csv')
    assert not set(X.columns) & set(TARGETS+source)
    return m, X, y, label_rows, structures


def taxa_report(selector, a, b, scope):
    rows = []
    for block, cols in selector.selected_.items():
        train_counts, test_counts = selector.mask(a, block).sum(), selector.mask(b, block).sum()
        for c in cols:
            rows.append({'scope': scope, 'block': block, 'feature': c, 'taxon': c.split('::', 1)[1], 'training_top_count': int(train_counts[c]), 'evaluation_top_count': int(test_counts[c])})
    return rows


def predictions(out, name, scope, frame, y, p):
    df = frame[['Site', 'SiteGroup', 'Season']].copy()
    for j, t in enumerate(TARGETS):
        df[t+'_actual'] = y[:, j]; df[t+'_probability'] = p[:, j]; df[t+'_predicted'] = (p[:, j] >= .5).astype(int)
    df.to_csv(out/f'{name}_{scope}_predictions.csv')


def plots(out, name, scope, ids, y, p):
    directory = out/'plots'/name/scope
    directory.mkdir(parents=True, exist_ok=True)
    matrices, curves = [], []
    for j, target in enumerate(TARGETS):
        yy, pp = y[:, j], p[:, j]
        fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
        cm = confusion_matrix(yy, pp >= .5, labels=[0, 1])
        ax[0].imshow(cm, cmap='Blues')
        for r in range(2):
            for c in range(2):
                ax[0].text(c, r, str(cm[r, c]), ha='center', va='center')
        ax[0].set(xticks=[0, 1], yticks=[0, 1], xlabel='Predicted', ylabel='Actual', title='Confusion matrix')
        matrices.append({'target': target, 'TN': cm[0, 0], 'FP': cm[0, 1], 'FN': cm[1, 0], 'TP': cm[1, 1]})
        if yy.sum():
            precision, recall, thresholds = precision_recall_curve(yy, pp)
            ax[1].step(recall, precision, where='post', label=f'AP={average_precision_score(yy, pp):.3f}')
            ax[1].axhline(yy.mean(), ls='--', color='grey', label=f'Prevalence={yy.mean():.3f}')
            ax[1].legend(fontsize=7)
            curves.extend({'target': target, 'curve': 'PR', 'x': r, 'y': pr} for r, pr in zip(recall, precision))
        else:
            ax[1].text(.1, .5, 'Undefined: no positives')
        ax[1].set(xlabel='Recall', ylabel='Precision', title='Precision–recall', xlim=(0, 1), ylim=(0, 1.05))
        if len(np.unique(yy)) == 2:
            fpr, tpr, thresholds = roc_curve(yy, pp)
            ax[2].plot(fpr, tpr, label=f'AUC={roc_auc_score(yy, pp):.3f}'); ax[2].legend(fontsize=7)
            curves.extend({'target': target, 'curve': 'ROC', 'x': f, 'y': t} for f, t in zip(fpr, tpr))
        else:
            ax[2].text(.1, .5, 'Undefined: one class')
        ax[2].plot([0, 1], [0, 1], '--', color='grey')
        ax[2].set(xlabel='False positive rate', ylabel='True positive rate', title='ROC', xlim=(0, 1), ylim=(0, 1.05))
        for val, color in [(0, 'steelblue'), (1, 'darkorange')]:
            ax[3].hist(pp[yy == val], bins=np.linspace(0, 1, 11), alpha=.6, color=color, label=f'Actual {val} (n={(yy == val).sum()})')
        ax[3].axvline(.5, color='black', ls='--'); ax[3].legend(fontsize=7)
        ax[3].set(xlabel='Predicted probability', ylabel='Samples', title='Probability distribution')
        fig.suptitle(f'{name} — {scope} — {target}'); fig.tight_layout()
        fig.savefig(directory/f'{target}.png', dpi=150); plt.close(fig)
    pd.DataFrame(matrices).to_csv(directory/'confusion_matrices.csv', index=False)
    pd.DataFrame(curves).to_csv(directory/'curve_coordinates.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, max(5, len(y)*.18)), sharey=True)
    for ax, values, title in zip(axes, [y, (p >= .5).astype(int), p], ['Actual labels', 'Predicted labels (0.5)', 'Predicted probabilities']):
        im = ax.imshow(values, aspect='auto', vmin=0, vmax=1, cmap='Blues')
        ax.set(xticks=range(5), xticklabels=[t.removeprefix('warning_') for t in TARGETS], yticks=range(len(ids)), yticklabels=ids, title=title)
        ax.tick_params(axis='x', rotation=75, labelsize=8); ax.tick_params(axis='y', labelsize=6)
    fig.colorbar(im, ax=axes[-1]); fig.suptitle(f'{name} — {scope}')
    fig.tight_layout(); fig.savefig(directory/'all_labels.png', dpi=150); plt.close(fig)


def supplement_report(out):
    """Summarize selected-candidate fold variation and rare-label results."""
    tuning = pd.read_csv(out/'hyperparameter_tuning.csv')
    rows = []
    for name in SPACES:
        best = json.loads((out/f'{name}_best_params.json').read_text())['candidate']
        folds = tuning[(tuning.Model == name) & (tuning.candidate == best) & tuning.scope.str.startswith('fold_')]
        for metric in [c for c in tuning.columns if c.startswith(('MCC_', 'AUPR_', 'Macro_', 'Micro_'))]:
            values = folds[metric]
            rows.append({'Model': name, 'metric': metric, 'fold_mean': values.mean(), 'fold_std': values.std(), 'defined_folds': int(values.notna().sum())})
    pd.DataFrame(rows).to_csv(out/'cv_fold_summary.csv', index=False)
    support = pd.read_csv(out/'label_distribution.csv')
    support = support[support.scope == 'test'].set_index('target')
    test = pd.read_csv(out/'test_performance.csv')
    lines = ['\n## Per-label test comparison\n',
             '| Warning | Test positives | Highest AUPR (model) | Highest MCC (model) |',
             '| --- | --- | --- | --- |']
    for target in TARGETS:
        ap = test.sort_values('AUPR_'+target, ascending=False).iloc[0]
        mc = test.sort_values('MCC_'+target, ascending=False).iloc[0]
        lines.append(f'| {target.removeprefix("warning_")} | {int(support.loc[target, "positive"])}/{int(support.loc[target, "samples"])} | {ap["AUPR_"+target]:.4f} ({ap.Model}) | {mc["MCC_"+target]:.4f} ({mc.Model}) |')
    rare = ', '.join(f'{t.removeprefix("warning_")} ({int(support.loc[t, "positive"])})' for t in TARGETS if support.loc[t, 'positive'] <= 2)
    lines += ['\nTied leaders are listed fully in `best_model_per_label.csv`. These test leaders are descriptive.',
              f'\nLabels with at most two positive test samples: {rare or "none"}. Their scores can change substantially with a single prediction. Use the full per-label table below to compare all algorithms.\n',
              '| Model | '+ ' | '.join(t.removeprefix('warning_')+' AP / MCC' for t in TARGETS)+' |',
              '| --- | '+' | '.join(['---']*5)+' |']
    for _, row in test.iterrows():
        lines.append('| '+row.Model+' | '+' | '.join(f'{row["AUPR_"+t]:.3f} / {row["MCC_"+t]:.3f}' for t in TARGETS)+' |')
    path = out/'performance_summary.md'
    original = path.read_text().split('\n## Per-label test comparison')[0]
    path.write_text(original+'\n'.join(lines)+'\n')
    # Record every tied best model rather than arbitrarily breaking ties.
    winners = []
    for scope, file in [('CV_OOF', 'cv_performance.csv'), ('test_descriptive', 'test_performance.csv')]:
        table = pd.read_csv(out/file)
        for target in TARGETS:
            for metric in ['AUPR', 'MCC']:
                column = metric+'_'+target
                for _, row in table[table[column] == table[column].max()].iterrows():
                    winners.append({'scope': scope, 'target': target, 'metric': metric, 'Model': row.Model, 'value': row[column]})
    pd.DataFrame(winners).to_csv(out/'best_model_per_label.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/final')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'ML/results_warnings')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-iter', type=int, default=8)
    parser.add_argument('--group-map', type=Path, help='JSON mapping Site to physical site group; default joins known QC replicates')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.n_iter < 2:
        parser.error('--n-iter must be at least 2 to tune every model')
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    group_map = {'BSIb': 'BSI', 'DGYb': 'DGY'}
    if args.group_map:
        group_map.update(json.loads(args.group_map.read_text()))
    m, X, y, label_rows, structures = inspect(args.data_dir, out, group_map)
    groups = m.SiteGroup.to_numpy()
    train, test = partition(groups, y, [.8, .2], args.seed)
    Xtr, Xte, ytr, yte = X.iloc[train], X.iloc[test], y[train], y[test]
    gtr = groups[train]
    folds = partition(gtr, ytr, [.2]*5, args.seed+1)
    assignment = m[['Site', 'SiteGroup', 'Season']].copy()
    assignment['split'] = 'test'; assignment.loc[Xtr.index, 'split'] = 'train'
    assignment['cv_validation_fold'] = pd.Series(pd.NA, index=assignment.index, dtype='Int64')
    cv_splits = []
    for fold, valid in enumerate(folds):
        fit = np.setdiff1d(np.arange(len(train)), valid)
        assert not set(gtr[fit]) & set(gtr[valid])
        assert not set(groups[test]) & set(gtr[fit])
        cv_splits.append((fit, valid))
        assignment.loc[Xtr.index[valid], 'cv_validation_fold'] = fold+1
        label_rows += label_report(ytr[valid], f'cv_validation_{fold+1}')
        label_rows += label_report(ytr[fit], f'cv_training_{fold+1}')
    assert not set(groups[train]) & set(groups[test])
    assert assignment.groupby('SiteGroup').split.nunique().max() == 1
    assert assignment.dropna().groupby('SiteGroup').cv_validation_fold.nunique().max() == 1
    assignment.to_csv(out/'split_and_cv_assignment.csv')
    label_rows += label_report(ytr, 'train')+label_report(yte, 'test')
    pd.DataFrame(label_rows).to_csv(out/'label_distribution.csv', index=False)
    selector = SampleTopTaxa().fit(Xtr)
    taxa = taxa_report(selector, Xtr, Xte, 'final_training_union')
    for fold, (fit, valid) in enumerate(cv_splits):
        s = SampleTopTaxa().fit(Xtr.iloc[fit])
        taxa += taxa_report(s, Xtr.iloc[fit], Xtr.iloc[valid], f'cv_{fold+1}')
    pd.DataFrame(taxa).to_csv(out/'selected_taxa.csv', index=False)
    for block, names in selector.selected_.items():
        (out/f'selected_{block}.txt').write_text('\n'.join(c.split('::', 1)[1] for c in names)+'\n')
    selector.transform(X).to_csv(out/'final_feature_matrix.csv')
    audit_text = [
        'Detected identity columns: SampleID, Site, Season; Round/No consistency verified.',
        f'Metadata predictors: {len(CATS+NUMS)} (2 categorical, 14 numeric).',
        'Taxonomic row labels: Phylum in P(%), Genus in G(%). All remaining sheet columns are sample abundance percentages.',
        'Workbook sample IDs normalize round-number-suffix to round-number; suffix/site crosswalk exported.',
        f'Matched samples: {len(m)}; Site labels: {m.Site.nunique()}; physical grouping units: {len(set(groups))}.',
        f'Training: {len(train)} samples / {len(set(gtr))} groups; test: {len(test)} samples / {len(set(groups[test]))} groups.',
        'Train groups: '+', '.join(sorted(set(gtr))),
        'Test groups: '+', '.join(sorted(set(groups[test]))),
        'Microbiome union sizes: '+str({b: len(c) for b, c in selector.selected_.items()}),
        'LEAKAGE CHECK: explicit metadata whitelist excludes every warning and effluent column.',
        'LEAKAGE CHECK: no site group overlap in holdout or CV; each training site is validation in exactly one fold.',
        'LEAKAGE CHECK: taxa union, imputation, encoding, scaling, and resampling refit within each training fold.',
        'LEAKAGE CHECK: test set excluded from tuning and calibration; classification thresholds fixed at 0.5.',
        'Known QC replicas grouped with parents. Other Site labels remain distinct unless group-map provided.',
        'Target source missingness is exported: existing zero labels are retained as requested, not certified normal.',
        'Rare labels may lack positives in individual validation folds; AP is NA in those folds; pooled OOF AP is the tuning objective.',
        'All workbook taxa, including unidentified/off-domain annotations, remain eligible; no unsupported domain filtering.',
        'Top-k ties (including zero values) break alphabetically; no re-normalization or mean abundance ranking.',
        'MCC uses sklearn convention of 0 when the denominator is zero; class support is reported.',
    ]
    report = '\n'.join(audit_text)+'\n\n'+pd.DataFrame(label_rows).to_string(index=False)+'\n\n'+(out/'samples_per_site.csv').read_text()
    (out/'pretraining_audit.txt').write_text(report)
    print(report, flush=True)
    config = {'seed': args.seed, 'n_iter': args.n_iter, 'group_map': group_map, 'metadata_features': CATS+NUMS,
              'targets': TARGETS, 'top_k': BLOCKS, 'thresholds': {t: .5 for t in TARGETS}, 'search_spaces': SPACES,
              'selection': 'pooled training out-of-fold Macro average precision, tie break Macro MCC',
              'aupr_definition': 'non-interpolated average precision (sklearn.average_precision_score)',
              'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__, 'matplotlib': matplotlib.__version__},
              'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.data_dir/'metadata.csv', args.data_dir/'Dat_BAC.xlsx', args.data_dir/'Dat_ARC.xlsx']}}
    save_json(out/'run_config.json', config)
    if args.audit_only:
        return
    cv_results, fitted, warn_rows = [], {}, []
    # Complete all tuning/final training before any test performance is evaluated.
    for name, space in SPACES.items():
        candidates = list(ParameterSampler(space, n_iter=args.n_iter, random_state=args.seed))
        scores, candidate_oof = [], []
        for candidate, params in enumerate(candidates):
            oof = np.zeros(ytr.shape)
            for fold, (fit, valid) in enumerate(cv_splits):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model = WarningModel(name, params, args.seed).fit(Xtr.iloc[fit], ytr[fit], gtr[fit])
                    oof[valid] = model.predict_proba(Xtr.iloc[valid])
                warn_rows.extend({'model': name, 'candidate': candidate, 'fold': fold+1, 'category': w.category.__name__, 'message': str(w.message)} for w in caught)
                fold_metrics = metrics(ytr[valid], oof[valid])
                cv_results.append({'Model': name, 'candidate': candidate, 'scope': f'fold_{fold+1}', 'params': json.dumps(params), **fold_metrics})
            score = metrics(ytr, oof)
            scores.append(score); candidate_oof.append(oof)
            cv_results.append({'Model': name, 'candidate': candidate, 'scope': 'pooled_OOF', 'params': json.dumps(params), **score})
            print(f'{name} candidate {candidate+1}/{len(candidates)}: CV Macro AUPR={score["Macro_AUPR"]:.4f}, MCC={score["Macro_MCC"]:.4f}', flush=True)
        best = max(range(len(scores)), key=lambda i: (scores[i]['Macro_AUPR'], scores[i]['Macro_MCC']))
        params, oof = candidates[best], candidate_oof[best]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            final = WarningModel(name, params, args.seed).fit(Xtr, ytr, gtr)
        warn_rows.extend({'model': name, 'candidate': best, 'fold': 'final', 'category': w.category.__name__, 'message': str(w.message)} for w in caught)
        joblib.dump(final, out/f'{name}.joblib')
        joblib.dump({'selector': final.selector, 'preprocessor': final.preprocessor}, out/f'{name}_preprocessing.joblib')
        pd.DataFrame(final.preprocessor.transform(final.selector.transform(X)), index=X.index, columns=final.preprocessor.get_feature_names_out()).to_csv(out/f'{name}_encoded_feature_matrix.csv')
        save_json(out/f'{name}_best_params.json', {'params': params, 'candidate': best, 'thresholds': final.thresholds.tolist(), 'CV_pooled_OOF': scores[best]})
        predictions(out, name, 'CV_OOF', m.loc[Xtr.index], ytr, oof)
        fitted[name] = (final, scores[best], oof)
        pd.DataFrame(cv_results).to_csv(out/'hyperparameter_tuning.csv', index=False)
        pd.DataFrame(warn_rows, columns=['model', 'candidate', 'fold', 'category', 'message']).to_csv(out/'fit_warnings.csv', index=False)
    cv_table = pd.DataFrame([{'Model': name, **score} for name, (_, score, _) in fitted.items()]).sort_values(['Macro_AUPR', 'Macro_MCC'], ascending=False)
    cv_table.to_csv(out/'cv_performance.csv', index=False)
    selected = cv_table.iloc[0].Model
    save_json(out/'cv_selected_model.json', {'Model': selected, 'rule': config['selection']})
    test_rows = []
    for name, (final, _, oof) in fitted.items():
        p = final.predict_proba(Xte)
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        test_rows.append({'Model': name, **metrics(yte, p)})
        predictions(out, name, 'test', m.loc[Xte.index], yte, p)
        plots(out, name, 'test', Xte.index, yte, p)
        plots(out, name, 'CV_OOF', Xtr.index, ytr, oof)
    test_table = pd.DataFrame(test_rows).sort_values(['Macro_AUPR', 'Macro_MCC'], ascending=False)
    test_table.to_csv(out/'test_performance.csv', index=False)
    test_table.to_csv(out/'performance_summary.csv', index=False)
    winners = []
    for scope, table in [('CV_OOF', cv_table), ('test_descriptive', test_table)]:
        for target in TARGETS:
            for metric in ['AUPR', 'MCC']:
                column = metric+'_'+target
                best_row = table.sort_values(column, ascending=False).iloc[0]
                winners.append({'scope': scope, 'target': target, 'metric': metric, 'Model': best_row.Model, 'value': best_row[column]})
    pd.DataFrame(winners).to_csv(out/'best_model_per_label.csv', index=False)
    def markdown(table):
        columns = ['Model', 'Macro_AUPR', 'Macro_MCC', 'Micro_AUPR']
        lines = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
        for _, row in table.iterrows():
            lines.append('| '+str(row.Model)+' | '+' | '.join(f'{row[c]:.4f}' for c in columns[1:])+' |')
        return '\n'.join(lines)
    summary = f'''# Multi-label anaerobic digestion warnings

CV-selected model: **{selected}**. Selection uses training pooled OOF Macro AUPR, with Macro MCC as tie breaker.
The highest test Macro AUPR is **{test_table.iloc[0].Model}**; this is a descriptive comparison, not a new tuning decision.

## Training 5-fold site-grouped CV (pooled out-of-fold)

{markdown(cv_table)}

## Independent test

{markdown(test_table)}

All five classifiers are independent. Thresholds are fixed at 0.5. AUPR means average precision.
CV scores are tuning estimates for the selected hyperparameters, not nested-CV unbiased estimates.
No threshold tuning was performed. Full per-label MCC/AUPR appear in the CSV tables.

{len(train)} training samples from {len(set(gtr))} groups; {len(test)} test samples from {len(set(groups[test]))} groups.
Missing archaeal samples were excluded explicitly in sample_audit.csv. Existing warning labels were preserved,
including zeros caused by missing target-source measurements. The smallest label has very few positives:
individual-fold AP may be undefined; pooled OOF scoring includes all training samples. MCC is 0 for degenerate cases.

GLMNET is elastic-net logistic regression (scikit-learn), not the R glmnet software.
SVM probabilities use sigmoid calibration on inner site-grouped OOF margins, with preprocessing refitted inside calibration folds.
RF/SVM/GLMNET use balanced class weights, XGBoost uses negative/positive class weighting,
and KNN/NNET use random minority oversampling confined to each training fold.
Imputation is training-median for numeric features (all-missing columns use zero), most-frequent for categories;
numerical predictors including abundances are z-scaled, and categories are one-hot encoded with unknown levels ignored.
Top-k selection is per sample and per block, union fitted on training only; zero ties are alphabetical.
Workbook annotations, including unidentified and off-domain entries, are retained as supplied.
See fit_warnings.csv for convergence diagnostics and pretraining_audit.txt for grouping and label support.
'''
    (out/'performance_summary.md').write_text(summary)
    supplement_report(out)
    print(summary, flush=True)

if __name__ == '__main__':
    main()
