#!/usr/bin/env python3
"""Site-grouped joint warning comparison; see README_multilabel.md."""
from pathlib import Path
import argparse
import hashlib
import json
import warnings
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import matthews_corrcoef, precision_recall_curve, auc
from multilabel_models import TARGETS, CATS, NUMS, BLOCKS, SampleAbundanceTaxa, WarningModel, numeric

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
        for rank in (['P', 'O'] if domain == 'BAC' else ['O', 'G']):
            frame = pd.read_excel(book, sheet_name=f'{rank}(%)', index_col=0)
            expected = {'P': 'Phylum', 'O': 'Order', 'G': 'Genus'}[rank]
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
        ranks = ['P', 'O'] if domain == 'BAC' else ['O', 'G']
        if not sheets[f'{domain}_{ranks[0]}'].index.equals(sheets[f'{domain}_{ranks[1]}'].index):
            raise ValueError('Within-domain sample columns disagree')
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


def metrics(y, p, threshold=.5):
    precision, recall, _ = precision_recall_curve(y.ravel(), p.ravel())
    return {'Multilabel MCC': float(matthews_corrcoef(y.ravel(), (p >= threshold).ravel())),
            'Micro-AUPR': float(auc(recall, precision))}


def choose_threshold(y, p):
    # One shared threshold limits overfitting with only four acid-accumulation positives.
    grid = np.linspace(.01, .99, 99)
    return float(max(grid, key=lambda t: (matthews_corrcoef(y.ravel(), (p >= t).ravel()), -abs(t-.5))))


def predictions(out, name, scope, frame, y, p, threshold):
    df = frame[['Site', 'SiteGroup', 'Season']].copy()
    for j, t in enumerate(TARGETS):
        df[t+'_actual'] = y[:, j]
        df[t+'_probability'] = p[:, j]
        cutoff = float(threshold) if np.ndim(threshold) == 0 else threshold[j]
        df[t+'_predicted'] = (p[:, j] >= cutoff).astype(int)
    df.to_csv(out/f'{name}_{scope}_predictions.csv')


def main():
    parser = argparse.ArgumentParser(description='Joint five-label analysis with sample-specific abundance filtering')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/final')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'ML/results_multilabel')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-iter', type=int, default=8)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.n_iter < 2:
        parser.error('Use at least two search candidates per model')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    group_map = {'BSIb': 'BSI', 'DGYb': 'DGY'}
    m, X, y, label_rows, structures = inspect(args.data_dir, out, group_map)
    groups = m.SiteGroup.to_numpy()
    train, test = partition(groups, y, [.8, .2], args.seed)
    Xtr, ytr, gtr = X.iloc[train], y[train], groups[train]
    folds = partition(gtr, ytr, [.2]*5, args.seed+1)
    cv = [(np.setdiff1d(np.arange(len(train)), v), v) for v in folds]
    assignment = m[['Site', 'SiteGroup', 'Season']].copy()
    assignment['split'] = 'test'
    assignment.loc[Xtr.index, 'split'] = 'train'
    assignment['cv_validation_fold'] = pd.Series(pd.NA, index=m.index, dtype='Int64')
    assert not set(gtr) & set(groups[test])
    for fold, (fit, valid) in enumerate(cv, 1):
        assert not set(gtr[fit]) & set(gtr[valid])
        assignment.loc[Xtr.index[valid], 'cv_validation_fold'] = fold
        label_rows += label_report(ytr[valid], f'cv_validation_{fold}') + label_report(ytr[fit], f'cv_training_{fold}')
    assignment.to_csv(out/'split_and_cv_assignment.csv')
    label_rows += label_report(ytr, 'train') + label_report(y[test], 'test')
    pd.DataFrame(label_rows).to_csv(out/'label_distribution.csv', index=False)
    selector = SampleAbundanceTaxa().fit(Xtr)
    selector.transform(X).to_csv(out/'final_feature_matrix.csv')
    # Descriptive audit includes test-only taxa, explicitly marked as excluded from modeling.
    rows = []
    for block in BLOCKS:
        counts = selector.mask(X, block).sum()
        for c in counts.index:
            rows.append({'block': block, 'taxon': c.split('::', 1)[1], 'feature': c,
                         'samples_above_0_1_percent': int(counts[c]),
                         'training_samples_above_0_1_percent': int(selector.counts_[block][c]),
                         'included_in_final_matrix': c in selector.selected_[block]})
        (out/f'selected_{block}.txt').write_text('\n'.join(selector.selected_[block])+'\n')
    taxa = pd.DataFrame(rows)
    taxa.to_csv(out/'taxa_audit.csv', index=False)
    taxa[taxa.included_in_final_matrix].to_csv(out/'selected_taxa.csv', index=False)
    taxa.groupby('block').agg(source_taxa=('taxon', 'size'),
        observed_union_taxa=('samples_above_0_1_percent', lambda s: int((s > 0).sum())),
        final_microbiome_features=('included_in_final_matrix', 'sum')).to_csv(out/'feature_counts.csv')
    fold_taxa = []
    for fold, (fit, valid) in enumerate(cv, 1):
        s = SampleAbundanceTaxa().fit(Xtr.iloc[fit])
        fold_taxa.extend({'fold': fold, 'block': b, 'feature': c, 'training_samples_above_0_1_percent': int(s.counts_[b][c])}
                         for b, cols in s.selected_.items() for c in cols)
    pd.DataFrame(fold_taxa).to_csv(out/'cv_feature_columns.csv', index=False)
    config = {'seed': args.seed, 'n_iter': args.n_iter, 'group_map': group_map,
              'metadata_features': CATS+NUMS, 'targets': TARGETS, 'blocks': BLOCKS,
              'filter': 'Per sample, retain actual percent only if strictly >0.1; otherwise zero. No renormalization. Union fitted on training rows only.',
              'selection': 'Maximum pooled OOF Micro-AUPR; tie-break flattened MCC at threshold 0.5. Select model before test evaluation.',
              'threshold': 'Shared threshold across five labels; maximize flattened MCC on selected-candidate training OOF predictions over 0.01..0.99; tie closest to 0.5.',
              'Micro-AUPR': 'Trapezoidal area under pooled precision-recall curve from flattened probabilities (not average precision).',
              'Multilabel MCC': 'Binary MCC of flattened sample-label pairs; zero for zero denominator.',
              'search_spaces': SPACES,
              'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__, 'matplotlib': matplotlib.__version__},
              'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.data_dir/'metadata.csv', args.data_dir/'Dat_BAC.xlsx', args.data_dir/'Dat_ARC.xlsx']}}
    save_json(out/'run_config.json', config)
    excluded = [c for c in m.columns if c.startswith('eff_') or c in TARGETS]
    save_json(out/'leakage_check.json', {'excluded_effluent_and_target_columns': excluded,
        'predictor_whitelist': CATS+NUMS, 'other_identity_columns_excluded': ['SampleID','Site','SiteGroup','No','Round'],
        'checks': ['No target or effluent predictor', 'Disjoint site groups in holdout and each CV fold',
                   'Union, imputation, encoding, scaling and resampling fitted within training folds',
                   'SVM calibration uses inner site-grouped OOF margins', 'Test excluded from tuning and threshold selection']})
    audit = [f'Matched samples: {len(m)}; site labels: {m.Site.nunique()}; groups: {len(set(groups))}.',
             f'Train: {len(train)} samples, {len(set(gtr))} groups. Test: {len(test)} samples, {len(set(groups[test]))} groups.',
             'Training sites: '+', '.join(sorted(set(gtr))), 'Test sites: '+', '.join(sorted(set(groups[test]))),
             'Missing ARC: 1-32\' and 4-24; excluded explicitly in sample_audit.csv.',
             'Duplicate/missing metadata and normalized genomic IDs checked; SampleID and Site mapping validated.',
             'QC replicates BSIb→BSI and DGYb→DGY; 35 groups, not the assumed 34 complete sites.',
             'Final microbiome features: '+str({b: len(c) for b,c in selector.selected_.items()}),
             'Numeric ranges parsed to midpoint; ± values use center; other nonnumeric values become missing (logged).',
             'Median numeric imputation, most-frequent categorical imputation, unknown-safe one-hot encoding, z-score scaling; all fitted on training rows.',
             'RF/SVM/elastic-net logistic regression: balanced class weights; XGBoost: training negative/positive ratio.',
             'KNN/NNET: per-label random oversampling on training rows only. Single-class training labels use constant prediction.',
             'GLMNET denotes sklearn elastic-net logistic regression (saga), not the R glmnet package.',
             'All supplied annotations remain eligible, including unidentified and off-domain taxa.',
             'Existing warning labels retained; missing target-source measurements can produce unflagged zeros.',
             'CV scores are tuning estimates, not nested unbiased performance estimates. Rare positives limit stability.',
             'No additional ML feature selection. Test-only union taxa appear only in descriptive audit.']
    (out/'pretraining_audit.txt').write_text('\n'.join(audit)+'\n')
    print('\n'.join(audit[:6]), flush=True)
    if args.audit_only:
        return
    tuning, fit_warnings, fitted, cv_rows = [], [], {}, []
    for name, space in SPACES.items():
        candidates = list(ParameterSampler(space, n_iter=args.n_iter, random_state=args.seed))
        scores, outputs = [], []
        for candidate, params in enumerate(candidates):
            oof = np.zeros(ytr.shape)
            for fold, (fit, valid) in enumerate(cv, 1):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model = WarningModel(name, params, args.seed).fit(Xtr.iloc[fit], ytr[fit], gtr[fit])
                    oof[valid] = model.predict_proba(Xtr.iloc[valid])
                fit_warnings.extend({'Model': name, 'candidate': candidate, 'fold': fold, 'category': w.category.__name__, 'message': str(w.message)} for w in caught)
                tuning.append({'Model': name, 'candidate': candidate, 'scope': f'fold_{fold}', 'params': json.dumps(params), **metrics(ytr[valid], oof[valid])})
            score = metrics(ytr, oof)
            scores.append(score); outputs.append(oof)
            tuning.append({'Model': name, 'candidate': candidate, 'scope': 'pooled_OOF', 'params': json.dumps(params), **score})
            pd.DataFrame(tuning).to_csv(out/'hyperparameter_tuning.csv', index=False)
            print(f'{name} {candidate+1}/{len(candidates)}: {score}', flush=True)
        best = max(range(len(scores)), key=lambda i: (scores[i]['Micro-AUPR'], scores[i]['Multilabel MCC']))
        params, oof = candidates[best], outputs[best]
        threshold = choose_threshold(ytr, oof)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            final = WarningModel(name, params, args.seed).fit(Xtr, ytr, gtr)
        fit_warnings.extend({'Model': name, 'candidate': best, 'fold': 'final', 'category': w.category.__name__, 'message': str(w.message)} for w in caught)
        final.thresholds = np.full(5, threshold)
        joblib.dump(final, out/f'{name}.joblib')
        joblib.dump({'selector': final.selector, 'preprocessor': final.preprocessor}, out/f'{name}_preprocessing.joblib')
        pd.DataFrame(final.preprocessor.transform(final.selector.transform(X)), index=X.index,
                     columns=final.preprocessor.get_feature_names_out()).to_csv(out/f'{name}_encoded_feature_matrix.csv')
        save_json(out/f'{name}_best_params.json', {'params': params, 'candidate': best,
            'thresholds': dict(zip(TARGETS, final.thresholds)), 'CV_selection_at_0_5': scores[best]})
        predictions(out, name, 'CV_OOF', m.loc[Xtr.index], ytr, oof, threshold)
        cv_rows.append({'Model': name, **metrics(ytr, oof, threshold)})
        fitted[name] = final
        pd.DataFrame(fit_warnings, columns=['Model','candidate','fold','category','message']).to_csv(out/'fit_warnings.csv', index=False)
    cv_table = pd.DataFrame(cv_rows)
    cv_table.to_csv(out/'cv_performance.csv', index=False)
    # Ranking uses the predeclared 0.5 tie-break, prior to threshold fitting.
    selection = []
    for name in fitted:
        best_info = json.loads((out/f'{name}_best_params.json').read_text())
        selection.append({'Model': name, **best_info['CV_selection_at_0_5']})
    selected = sorted(selection, key=lambda r: (r['Micro-AUPR'], r['Multilabel MCC']), reverse=True)[0]['Model']
    save_json(out/'cv_selected_model.json', {'Model': selected, 'rule': config['selection']})
    pd.DataFrame([{'Model': n, **dict(zip(TARGETS, f.thresholds))} for n,f in fitted.items()]).to_csv(out/'thresholds.csv', index=False)
    evaluate(out, m, X, y, test, fitted, selected, cv_table, audit, args)


def evaluate(out, m, X, y, test, fitted, selected, cv_table, audit, args):
    test_rows = []
    fig, ax = plt.subplots(figsize=(7,6))
    curves = []
    for name, final in fitted.items():
        p = final.predict_proba(X.iloc[test])
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        score = metrics(y[test], p, final.thresholds)
        test_rows.append({'Model': name, **score})
        predictions(out, name, 'test', m.iloc[test], y[test], p, final.thresholds)
        precision, recall, _ = precision_recall_curve(y[test].ravel(), p.ravel())
        ax.plot(recall, precision, label=f'{name}: {score["Micro-AUPR"]:.3f}')
        curves.extend({'Model': name, 'Recall': r, 'Precision': pr} for r,pr in zip(recall,precision))
    ax.set(xlabel='Micro recall', ylabel='Micro precision', title='Held-out sites: pooled five-label precision–recall', xlim=(0,1), ylim=(0,1.02))
    ax.legend(); fig.tight_layout(); fig.savefig(out/'micro_precision_recall.png',dpi=180); plt.close(fig)
    pd.DataFrame(curves).to_csv(out/'micro_pr_coordinates.csv',index=False)
    test_table = pd.DataFrame(test_rows)
    test_table.to_csv(out/'test_performance.csv',index=False)
    for metric, filename in [('Multilabel MCC','multilabel_mcc'), ('Micro-AUPR','micro_aupr')]:
        fig, ax = plt.subplots(figsize=(8,5))
        bars=ax.bar(test_table.Model, test_table[metric],color='#287c8e')
        ax.bar_label(bars,fmt='%.3f',padding=3)
        ax.set(ylabel=metric,title=f'Held-out sites: {metric}')
        ax.set_ylim(min(0,float(test_table[metric].min())-.1),max(.1,float(test_table[metric].max())+.12))
        fig.tight_layout();fig.savefig(out/f'{filename}.png',dpi=180);plt.close(fig)
    def markdown(df):
        return '\n'.join(['| Model | Multilabel MCC | Micro-AUPR |','|---|---:|---:|']+
            [f'| {r["Model"]} | {r["Multilabel MCC"]:.4f} | {r["Micro-AUPR"]:.4f} |' for r in df.to_dict('records')])
    report = '\n'.join(['# Joint five-label warning prediction',f'CV-selected model: **{selected}**.',
        '\n## Held-out test performance',markdown(test_table),'\n## Training OOF tuning estimates',markdown(cv_table),
        '\nOOF MCC uses a threshold optimized on these same OOF predictions and is optimistic. Hyperparameters also use these folds; test is the independent evaluation.',
        '\nMicro-AUPR uses trapezoidal PR area; MCC flattens all five labels. These pooled metrics do not measure exact five-label-vector correctness.',
        '\n## Data and methods','\n\n'.join(audit), '\n## Reproducibility',
        f'Run `python ML/train_multilabel.py --seed {args.seed} --n-iter {args.n_iter}` with the versions in run_config.json.',
        'Search spaces, input hashes, assignments, feature lists, fitted preprocessing, models, thresholds and predictions are saved alongside this report.',
        'Test comparisons are descriptive; they do not replace the model selected from CV. No causal interpretation is supported.'])
    leader_mcc = test_table.loc[test_table['Multilabel MCC'].idxmax(), 'Model']
    leader_aupr = test_table.loc[test_table['Micro-AUPR'].idxmax(), 'Model']
    report += (f'\n\n## Interpretation\n{selected} was selected using training CV. '
               f'{leader_aupr} has the highest held-out Micro-AUPR and {leader_mcc} has the highest held-out MCC. '
               'The two metrics need not favor the same algorithm. The test set contains only 29 samples from seven groups, '
               'with very few positive examples for several warnings, so rankings have substantial uncertainty. '
               'These results support measurable pooled predictive signal in this holdout, not established deployment reliability.\n')
    warning_frame = pd.read_csv(out/'fit_warnings.csv')
    report += f'\nRecorded fit warnings: {len(warning_frame)}; see fit_warnings.csv for candidate and fold details.\n'
    (out/'report.md').write_text(report+'\n')
    print('\nCV-selected model:',selected,'\n',test_table.to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
