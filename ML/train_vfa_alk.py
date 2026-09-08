#!/usr/bin/env python3
"""Predict eff_TVFAs / eff_ALK using chemistry and top ARC/BAC phyla and genera.

Run: python ML/train_vfa_alk.py --n-iter 20 --folds 5
Top taxa are selected across ALL workbook samples as requested; this exposes
holdout feature distributions to selection. Scores are conditional on that
fixed feature panel. All learned imputation/encoding/scaling is fitted within CV.
Sites stay together in an approximately 80:20 GroupShuffleSplit and GroupKFold.
Substrate averages are Q-weighted; incomplete inputs remain missing for CV imputation.
"""
from pathlib import Path
import argparse
import json
import re
import warnings

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
PANEL = ['Q', 'pH', 'ALK', 'VS', 'COD', 'sCOD', 'TC', 'Protein',
         'Lipid', 'TVFAs', 'HAc', 'HPro']
SUBSTRATE_RAW = [f'substrate{i}_{p}' for i in range(1, 4) for p in PANEL]
AVERAGES = [f'substrate_avg_{p}' for p in PANEL if p != 'Q']
NUMERIC = ['Q_total_Tpy'] + AVERAGES + [
    'HRT_d', 'T_C', 'eff_pH', 'eff_VS', 'eff_TC', 'eff_Lipid', 'eff_COD',
    'eff_sCOD', 'eff_CH4', 'eff_H2S', 'eff_protein']
CATEGORICAL = ['substrate_type']


def temperature(value):
    """Use midpoint of a range and center of a mean ± spread."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if '±' in text:
        return float(text.split('±')[0])
    match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*[~-]\s*(\d+(?:\.\d+)?)', text)
    return (float(match[1]) + float(match[2])) / 2 if match else float(text)


def load_taxa(path, prefix, rank, top_n):
    """Rank taxa over all workbook samples and normalize sample IDs."""
    sheet = {'Phylum': 'P(%)', 'Genus': 'G(%)'}[rank]
    frame = pd.read_excel(path, sheet_name=sheet, index_col=0)
    if frame.index.has_duplicates or frame.index.isna().any():
        raise ValueError(f'{path}: invalid taxon labels')
    frame = frame.apply(pd.to_numeric, errors='raise')
    if not np.isfinite(frame.to_numpy()).all() or (frame < 0).any().any():
        raise ValueError(f'{path}: invalid abundance values')
    ranking = frame.mean(axis=1).sort_values(ascending=False, kind='stable')
    if len(ranking) < top_n:
        raise ValueError(f'{path}: fewer than {top_n} taxa')
    selected = frame.loc[ranking.index[:top_n]].T
    ids = []
    for sample in selected.index:
        match = re.fullmatch(r'(\d+-[^-]+)-.+', str(sample))
        if not match:
            raise ValueError(f'Unexpected workbook sample ID: {sample}')
        ids.append(match[1])
    selected.index = pd.Index(ids, name='SampleID')
    if selected.index.has_duplicates:
        raise ValueError(f'{path}: duplicate normalized sample IDs')
    selected.columns = [f'{prefix}__{rank}__{p}' for p in selected.columns]
    report = ranking.rename('mean_relative_abundance_percent').rename_axis('Taxon').reset_index()
    report['domain'] = prefix
    report['rank'] = rank
    report['selected'] = np.arange(len(report)) < top_n
    return selected, report


def add_substrate_averages(meta):
    """Compute sum(Q_i * value_i) / sum(Q_i) for each substrate parameter.

    Absent substrates contribute zero flow. A present substrate with missing flow
    or a positive-flow substrate with missing chemistry makes that average NaN.
    This avoids treating unmeasured chemistry as zero or silently reweighting it.
    pH uses the same descriptive arithmetic average, not a chemical mixing model.
    """
    meta = meta.copy()
    present = np.column_stack([
        meta[f'substrate{i}_name'].fillna('').astype(str).str.strip().ne('')
        for i in (1, 2, 3)])
    q = meta[[f'substrate{i}_Q' for i in (1, 2, 3)]].to_numpy(dtype=float)
    q = np.where(present, q, 0.0)
    valid_flow = np.isfinite(q).all(axis=1) & (q >= 0).all(axis=1)
    total = q.sum(axis=1)
    # Keep the supplied Q_total_Tpy as requested; use component Q for averaging.
    for field in PANEL:
        if field == 'Q':
            continue
        values = meta[[f'substrate{i}_{field}' for i in (1, 2, 3)]].to_numpy(dtype=float)
        valid = valid_flow & (total > 0) & ((q == 0) | np.isfinite(values)).all(axis=1)
        numerator = (q * np.where(q == 0, 0.0, values)).sum(axis=1)
        average = np.full(len(meta), np.nan)
        np.divide(numerator, total, out=average, where=valid)
        meta[f'substrate_avg_{field}'] = average
    return meta


def build_data(data_dir, out):
    meta = pd.read_csv(data_dir / 'metadata.csv', encoding='utf-8-sig',
                       dtype={'SampleID': str, 'No': str})
    raw_numeric = sorted((set(NUMERIC) - set(AVERAGES) - {'eff_protein'})
                         | set(SUBSTRATE_RAW) | {'eff_TKN', 'eff_TAN', 'eff_TVFAs', 'eff_ALK'})
    required = set(raw_numeric + CATEGORICAL + [f'substrate{i}_name' for i in (1, 2, 3)] + ['SampleID', 'Site', 'Season',
                    'eff_TKN', 'eff_TAN', 'eff_TVFAs', 'eff_ALK']) - {'eff_protein'}
    if required - set(meta.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(meta.columns))}')
    if meta.SampleID.isna().any() or meta.SampleID.duplicated().any():
        raise ValueError('Metadata SampleID must be unique and nonmissing')
    meta = meta.set_index('SampleID')
    meta['T_C'] = meta.T_C.map(temperature)
    conversions = []
    for col in raw_numeric:
        original = meta[col]
        converted = pd.to_numeric(original, errors='coerce')
        for sid, value in original[original.notna() & converted.isna()].items():
            conversions.append({'SampleID': sid, 'column': col, 'value': value})
        meta[col] = converted.replace([np.inf, -np.inf], np.nan)
    pd.DataFrame(conversions, columns=['SampleID', 'column', 'value']).to_csv(out / 'coerced_values.csv', index=False)
    meta['eff_protein'] = (meta.eff_TKN - meta.eff_TAN).clip(lower=0) * 6.25
    meta['target'] = meta.eff_TVFAs / meta.eff_ALK.where(meta.eff_ALK > 0)
    meta['target'] = meta.target.replace([np.inf, -np.inf], np.nan)
    for col in CATEGORICAL:
        meta[col] = meta[col].map(lambda x: str(x).strip() if pd.notna(x) else np.nan)
    meta = add_substrate_averages(meta)
    microbial, rankings = {}, []
    for domain in ('ARC', 'BAC'):
        blocks = []
        for rank, top_n in [('Phylum', 10), ('Genus', 16)]:
            block, ranking = load_taxa(data_dir / f'Dat_{domain}.xlsx', domain, rank, top_n)
            blocks.append(block)
            rankings.append(ranking)
        if set(blocks[0].index) != set(blocks[1].index):
            raise ValueError(f'{domain}: phylum/genus sample sets differ')
        microbial[domain] = blocks[0].join(blocks[1], validate='one_to_one')
    arc, bac = microbial['ARC'], microbial['BAC']
    pd.concat(rankings).to_csv(out / 'microbial_ranking.csv', index=False)
    audit = pd.DataFrame(index=meta.index)
    audit['has_arc'] = meta.index.isin(arc.index)
    audit['has_bac'] = meta.index.isin(bac.index)
    audit['valid_target'] = meta.target.notna()
    audit['included'] = audit.all(axis=1)
    audit.to_csv(out / 'sample_audit.csv')
    merged = meta.join(arc, how='inner', validate='one_to_one').join(bac, how='inner', validate='one_to_one')
    merged = merged.loc[merged.target.notna()].copy()
    features = NUMERIC + CATEGORICAL + list(arc.columns) + list(bac.columns)
    merged[features + ['Site', 'Season', 'target']].to_csv(out / 'integrated_data.csv')
    print(f'Included {len(merged)}/{len(meta)} metadata samples; {len(features)} input features', flush=True)
    return merged, features


def model_specs(seed, max_neighbors):
    return {
        'RandomForest': (RandomForestRegressor(random_state=seed, n_jobs=1), {
            'n_estimators': [200, 400], 'max_depth': [None, 4, 8],
            'min_samples_leaf': [1, 2, 4], 'max_features': [0.5, 1.0]}),
        'KNN': (KNeighborsRegressor(), {'n_neighbors': [k for k in [3, 5, 7, 11, 15] if k <= max_neighbors],
                                      'weights': ['uniform', 'distance'], 'p': [1, 2]}),
        'SVM': (SVR(), {'C': [0.1, 1, 10, 100], 'epsilon': [0.01, 0.1, 0.2],
                        'gamma': ['scale', 0.01, 0.1], 'kernel': ['rbf']}),
        'MLP': (MLPRegressor(random_state=seed, max_iter=3000, solver='lbfgs'), {
            'hidden_layer_sizes': [(32,), (64,), (32, 16)], 'alpha': [0.001, 0.01, 0.1, 1.0]}),
        'XGBoost': (XGBRegressor(random_state=seed, n_jobs=1, objective='reg:squarederror', tree_method='hist'), {
            'n_estimators': [100, 300], 'max_depth': [2, 3, 5], 'learning_rate': [0.03, 0.1],
            'subsample': [0.8, 1.0], 'colsample_bytree': [0.7, 1.0], 'reg_lambda': [1, 10]}),
        'ElasticNet': (ElasticNet(random_state=seed, max_iter=50000), {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0], 'l1_ratio': [0.1, 0.5, 0.9, 1.0]}),
    }


def export_importance(model, name, Xtest, ytest, out, seed):
    """Describe the CV-selected model using held-out permutation importance.

    Selection masks come from refitting on all training data, never the test set.
    Permutation importance is descriptive only and does not choose the model or
    its features. Scores measure increase in RMSE, not causation; correlated taxa
    can share or hide importance. Negative values are preserved in the export.
    """
    prep = model.named_steps['preprocess']
    phys = prep.named_transformers_['physicochemical'].named_steps['select'].get_support()
    cat = prep.named_transformers_['categorical']
    cat_names = cat.named_steps['encoder'].get_feature_names_out(CATEGORICAL)
    cat_support = cat.named_steps['select'].get_support()
    records = [{'feature': col, 'group': 'physicochemical', 'selected': bool(keep)}
               for col, keep in zip(NUMERIC, phys)]
    records += [{'feature': col, 'group': 'categorical_encoded', 'selected': bool(keep)}
                for col, keep in zip(cat_names, cat_support)]
    microbial = [c for c in Xtest.columns if c not in NUMERIC + CATEGORICAL]
    records += [{'feature': c, 'group': 'microbial', 'selected': True} for c in microbial]
    selection = pd.DataFrame(records)
    selection.to_csv(out / 'best_model_feature_selection.csv', index=False)
    # Permute original input columns: substrate_type is evaluated jointly across
    # its dummy columns, avoiding impossible one-hot combinations.
    result = permutation_importance(model, Xtest, ytest,
                                    scoring='neg_root_mean_squared_error',
                                    n_repeats=30, random_state=seed, n_jobs=1)
    selected_raw = set(np.array(NUMERIC)[phys]) | set(microbial)
    if cat_support.any():
        selected_raw.update(CATEGORICAL)
    importance = pd.DataFrame({'feature': Xtest.columns,
                               'mean_RMSE_increase': result.importances_mean,
                               'std_RMSE_increase': result.importances_std})
    importance['selected'] = importance.feature.isin(selected_raw)
    positive = importance.mean_RMSE_increase.clip(lower=0).where(importance.selected, 0)
    importance['relative_positive_importance_percent'] = (
        100 * positive / positive.sum() if positive.sum() > 0 else 0.0)
    importance.sort_values('mean_RMSE_increase', ascending=False).to_csv(
        out / 'best_model_permutation_importance.csv', index=False)
    shown = importance.loc[importance.selected].sort_values('mean_RMSE_increase')
    fig, ax = plt.subplots(figsize=(11, max(6, 0.25 * len(shown))))
    ax.barh(shown.feature, shown.mean_RMSE_increase,
            xerr=shown.std_RMSE_increase, color='steelblue', alpha=0.85)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set(xlabel='Increase in held-out RMSE after permutation (mean ± SD)',
           title=f'{name}: selected inputs, test-set permutation importance')
    fig.tight_layout()
    fig.savefig(out / 'best_model_feature_importance.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(NUMERIC, phys.astype(int), color=['steelblue' if k else 'lightgray' for k in phys])
    ax.set(xlim=(0, 1.1), xticks=[0, 1], xticklabels=['Dropped', 'Retained'],
           title=f'{name}: physicochemical selection after training-set refit')
    fig.tight_layout()
    fig.savefig(out / 'best_model_selected_features.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/final')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'ML/results_selected')
    parser.add_argument('--n-iter', type=int, default=20)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-jobs', type=int, default=1)
    args = parser.parse_args()
    if args.n_iter < 1 or args.folds < 2:
        parser.error('--n-iter must be positive and --folds must be at least 2')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    data, features = build_data(args.data_dir, out)
    groups = data.Site
    if groups.isna().any() or groups.astype(str).str.strip().eq('').any():
        raise ValueError('Site must be nonmissing for grouped splitting')
    if groups.nunique() < 3:
        raise ValueError('At least three sites are required for holdout and grouped CV')
    warnings.warn('Top taxa use all workbook samples as requested, including holdout feature distributions.')
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                        random_state=args.seed).split(data, groups=groups))
    train_groups, test_groups = groups.iloc[train], groups.iloc[test]
    assert set(train_groups).isdisjoint(test_groups)
    if args.folds > train_groups.nunique():
        raise ValueError(f'--folds cannot exceed {train_groups.nunique()} training sites')
    if len(test) < 2:
        raise ValueError('At least two test samples are required for R²')
    print(f'Train: {len(train)} samples / {train_groups.nunique()} sites; '
          f'Test: {len(test)} samples / {test_groups.nunique()} sites', flush=True)
    X, y = data[features], data.target
    Xtrain, Xtest, ytrain, ytest = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
    membership = data[['Site', 'Season']].copy()
    membership['split'] = 'train'
    membership.loc[Xtest.index, 'split'] = 'test'
    membership.to_csv(out / 'split_membership.csv')
    membership.groupby(['Site', 'Season', 'split']).size().rename('n').to_csv(out / 'split_coverage.csv')
    microbial = [c for c in features if c not in NUMERIC + CATEGORICAL]
    preprocessor = ColumnTransformer([
        # Selection is refitted on each CV training fold after imputation, before
        # scaling. Remove zero-variance physicochemical variables, preserving the
        # requested microbial panel in a separate branch.
        ('physicochemical', Pipeline([
            ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('select', VarianceThreshold(threshold=0.0)),
            ('scaler', StandardScaler())]), NUMERIC),
        ('microbial', Pipeline([
            ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scaler', StandardScaler())]), microbial),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing', keep_empty_features=True)),
                                  ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                                  ('select', VarianceThreshold(threshold=0.0))]), CATEGORICAL)])
    cv = GroupKFold(n_splits=args.folds)
    folds = list(cv.split(Xtrain, ytrain, groups=train_groups))
    cv_membership = []
    for fold, (fit, validation) in enumerate(folds, start=1):
        assert set(train_groups.iloc[fit]).isdisjoint(train_groups.iloc[validation])
        for role, indices in [('fit', fit), ('validation', validation)]:
            for j in indices:
                cv_membership.append({'fold': fold, 'role': role,
                                      'SampleID': Xtrain.index[j], 'Site': train_groups.iloc[j]})
    pd.DataFrame(cv_membership).to_csv(out / 'cv_membership.csv', index=False)
    results = []
    fitted_models = {}
    for name, (estimator, params) in model_specs(args.seed, min(len(a) for a, _ in folds)).items():
        print(f'Tuning {name}...', flush=True)
        # Scaling y is fitted within each CV fold; predictions/metrics retain original ratio units.
        pipeline = Pipeline([('preprocess', preprocessor), ('model', TransformedTargetRegressor(
            regressor=estimator, transformer=StandardScaler()))])
        space = {f'model__regressor__{k}': v for k, v in params.items()}
        search = RandomizedSearchCV(pipeline, space, n_iter=min(args.n_iter, int(np.prod([len(v) for v in params.values()]))),
                                   scoring='neg_root_mean_squared_error', cv=cv, refit=True,
                                   random_state=args.seed, n_jobs=args.n_jobs, error_score='raise')
        search.fit(Xtrain, ytrain, groups=train_groups)
        fitted_models[name] = search.best_estimator_
        pred_train, pred_test = search.predict(Xtrain), search.predict(Xtest)
        rmse = lambda actual, pred: float(np.sqrt(mean_squared_error(actual, pred)))
        row = {'Model': name, 'CV_RMSE': -search.best_score_,
               'Train_RMSE': rmse(ytrain, pred_train), 'Test_RMSE': rmse(ytest, pred_test),
               'Train_R2': r2_score(ytrain, pred_train), 'Test_R2': r2_score(ytest, pred_test)}
        results.append(row)
        pd.DataFrame(results).sort_values('CV_RMSE').to_csv(out / 'performance_summary.csv', index=False)
        pd.DataFrame(search.cv_results_).to_csv(out / f'{name}_cv_results.csv', index=False)
        joblib.dump(search.best_estimator_, out / f'{name}.joblib')
        (out / f'{name}_best_params.json').write_text(json.dumps(search.best_params_, indent=2))
        predictions = pd.concat([pd.DataFrame({'Actual': ytrain, 'Predicted': pred_train, 'split': 'train'}),
                                 pd.DataFrame({'Actual': ytest, 'Predicted': pred_test, 'split': 'test'})])
        predictions.to_csv(out / f'{name}_predictions.csv')
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        low = min(y.min(), pred_train.min(), pred_test.min())
        high = max(y.max(), pred_train.max(), pred_test.max())
        pad = max((high - low) * 0.05, 0.001)
        for ax, label, actual, predicted in zip(axes, ['Train', 'Test'], [ytrain, ytest], [pred_train, pred_test]):
            ax.scatter(actual, predicted, alpha=0.7, edgecolors='white')
            ax.plot([low-pad, high+pad], [low-pad, high+pad], 'k--', linewidth=1)
            ax.set(xlabel='Actual TVFAs / ALK', ylabel='Predicted TVFAs / ALK',
                   title=f'{name}: {label}\nRMSE={row[label+"_RMSE"]:.4f}, R²={row[label+"_R2"]:.3f}',
                   xlim=(low-pad, high+pad), ylim=(low-pad, high+pad))
        fig.savefig(out / f'{name}_prediction_vs_actual.png', dpi=180)
        plt.close(fig)
    summary = pd.DataFrame(results).sort_values('CV_RMSE')
    best_name = summary.iloc[0].Model
    export_importance(fitted_models[best_name], best_name, Xtest, ytest, out, args.seed)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(train_samples=len(train), test_samples=len(test), features=features,
                  train_sites=train_groups.nunique(), test_sites=test_groups.nunique(),
                  split='GroupShuffleSplit', cv='GroupKFold', group_column='Site',
                  best_model_by_cv=summary.iloc[0].Model,
                  caveats=['Global taxon selection includes holdout feature distributions.',
                           'Contemporaneous effluent predictors describe current state, not future forecasting.'])
    (out / 'run_config.json').write_text(json.dumps(config, indent=2))
    columns = ['Model', 'Train_RMSE', 'Test_RMSE', 'Train_R2', 'Test_R2']
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for _, row in summary.iterrows():
        lines.append('| ' + row.Model + ' | ' + ' | '.join(f'{row[c]:.4f}' for c in columns[1:]) + ' |')
    markdown = '\n'.join(lines) + '\n'
    (out / 'performance_summary.md').write_text(markdown)
    print(markdown)
    print(f'Outputs: {out.resolve()}')


if __name__ == '__main__':
    main()
