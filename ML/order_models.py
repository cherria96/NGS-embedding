"""Reusable training-only feature engineering and independent binary models."""
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

TARGETS = ['warning_' + x for x in ['acid_base_balance', 'buffer_capacity', 'acid_accumulation', 'ammonia_toxicity', 'biogas_quality']]
CATS = []
NUMS = []
BLOCKS = {'BAC_O': 'Order', 'ARC_O': 'Order'}

def numeric(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if '±' in value:
        value = value.split('±')[0]
    match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*[~-]\s*(\d+(?:\.\d+)?)', value)
    if match:
        return (float(match[1]) + float(match[2])) / 2
    return pd.to_numeric(value, errors='coerce')

class SampleAbundanceTaxa(BaseEstimator, TransformerMixin):
    """Fixed predefined union supplied by the data audit; no learned selection."""
    def fit(self, X, y=None):
        if not all(c.startswith(('BAC_O::', 'ARC_O::')) for c in X.columns):
            raise ValueError('Only Order abundance predictors are allowed')
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        values = X[self.columns_].astype(float)
        if not np.isfinite(values.to_numpy()).all() or (values.to_numpy() < 0).any():
            raise ValueError('Missing or invalid abundance: investigate source data')
        return values.where(values > .1, 0)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_, dtype=object)


def preprocessing(X, name):
    # Complete abundance inputs are validated; no imputation is needed.
    return Pipeline([('scale', StandardScaler() if name in ('KNN','SVM','NNET','GLMNET') else 'passthrough')])


def estimator(name, params, seed, y):
    if len(np.unique(y)) < 2:
        return DummyClassifier(strategy='constant', constant=int(y[0]))
    if name == 'RandomForest':
        return RandomForestClassifier(random_state=seed, class_weight='balanced', n_jobs=1, **params)
    if name == 'KNN':
        return KNeighborsClassifier(**params)
    if name == 'SVM':
        return SVC(random_state=seed, class_weight='balanced', probability=False, **params)
    if name == 'NNET':
        return MLPClassifier(random_state=seed, max_iter=1500, early_stopping=False, **params)
    if name == 'XGBoost':
        ratio = (y == 0).sum() / (y == 1).sum()
        return XGBClassifier(random_state=seed, n_jobs=1, tree_method='hist', eval_metric='logloss', scale_pos_weight=ratio, **params)
    if name == 'GLMNET':
        return LogisticRegression(penalty='elasticnet', solver='saga', class_weight='balanced', max_iter=15000, random_state=seed, **params)
    raise ValueError(name)


def fit_one(name, params, seed, X, y):
    model = estimator(name, params, seed, y)
    # Fold-local random oversampling for algorithms without class weights.
    if name in ('KNN', 'NNET') and len(np.unique(y)) == 2:
        rng = np.random.default_rng(seed)
        classes = [np.flatnonzero(y == c) for c in (0, 1)]
        n = max(map(len, classes))
        take = np.concatenate([np.concatenate([a, rng.choice(a, n - len(a), replace=True)]) for a in classes])
        rng.shuffle(take)
        X, y = X[take], y[take]
    return model.fit(X, y)


def probability(model, X):
    p = model.predict_proba(X)
    return p[:, list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(len(X))


def margin(model, X):
    if isinstance(model, DummyClassifier):
        return np.full(len(X), 1.0 if model.constant == 1 else -1.0)
    return model.decision_function(X)


class WarningModel:
    """Binary relevance bundle, including fixed abundance masking and preprocessing.

    SVM sigmoid calibration uses site-grouped out-of-fold margins, with feature
    preprocessing refitted within each calibration training fold.
    """
    def __init__(self, name, params, seed=42):
        self.name, self.params, self.seed = name, params, seed
        self.thresholds = np.full(5, .5)

    def fit(self, X, y, groups):
        self.selector = SampleAbundanceTaxa().fit(X)
        selected = self.selector.transform(X)
        self.preprocessor = preprocessing(selected, self.name).fit(selected)
        z = self.preprocessor.transform(selected)
        self.models = [fit_one(self.name, self.params, self.seed+j, z, y[:, j]) for j in range(5)]
        self.calibrators = None
        if self.name == 'SVM':
            scores = np.zeros(y.shape)
            for train, valid in GroupKFold(n_splits=min(3, len(set(groups)))).split(X, y, groups):
                selector = SampleAbundanceTaxa().fit(X.iloc[train])
                a, b = selector.transform(X.iloc[train]), selector.transform(X.iloc[valid])
                prep = preprocessing(a, self.name).fit(a)
                za, zb = prep.transform(a), prep.transform(b)
                for j in range(5):
                    base = fit_one('SVM', self.params, self.seed+j, za, y[train, j])
                    scores[valid, j] = margin(base, zb)
            self.calibrators = []
            for j in range(5):
                cal = (LogisticRegression(C=1, random_state=self.seed) if len(np.unique(y[:, j])) == 2
                       else DummyClassifier(strategy='constant', constant=int(y[0, j])))
                self.calibrators.append(cal.fit(scores[:, j:j+1], y[:, j]))
        return self

    def predict_proba(self, X):
        z = self.preprocessor.transform(self.selector.transform(X))
        if self.calibrators is not None:
            return np.column_stack([probability(c, margin(m, z).reshape(-1, 1)) for m, c in zip(self.models, self.calibrators)])
        return np.column_stack([probability(m, z) for m in self.models])

    def predict(self, X):
        return (self.predict_proba(X) >= self.thresholds).astype(int)
