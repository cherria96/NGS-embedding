"""Verify saved artifacts against source values; never refit or repredict test data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from train_family_point1 import TARGETS, METRICS, metrics

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'ML/results_family_point1'
x=pd.read_csv(out/'final_feature_matrix.csv',index_col='SampleID',float_precision='round_trip')
raw=[]
for domain in ['BAC','ARC']:
    f=pd.read_excel(ROOT/f'data/final/Dat_{domain}.xlsx',sheet_name='F(%)').set_index('Family').T
    f.index=['-'.join(s.split('-')[:2]) for s in f.index]
    f.columns=[domain+'_F::'+c for c in f.columns]
    raw.append(f.loc[x.index])
source=pd.concat(raw,axis=1)
expected=source.where(source > .1,0)
expected=expected.loc[:,(expected>0).any(axis=0)]
pd.testing.assert_frame_equal(x.sort_index(axis=1),expected.sort_index(axis=1),check_names=False)
assert x.shape==(138,370)
assert all(c.startswith(('BAC_F::','ARC_F::')) for c in x.columns)
a=pd.read_csv(out/'split_and_cv_assignment.csv',index_col='SampleID',float_precision='round_trip')
assert a.index.is_unique and not a.index.isna().any()
assert a.groupby('SiteGroup').split.nunique().eq(1).all()
assert a.groupby('Site').split.nunique().eq(1).all()
assert a[a.split=='train'].groupby('SiteGroup').cv_validation_fold.nunique().eq(1).all()
assert set(a[a.split=='train'].cv_validation_fold)==set(range(1,6))
assert len(a[a.split=='train'])==109 and len(a[a.split=='test'])==29
performance=pd.read_csv(out/'test_performance.csv').set_index('Model')
assert performance.columns.tolist()==METRICS and len(performance)==6
cv=pd.read_csv(out/'cv_performance.csv').set_index('Model')
thresholds=pd.read_csv(out/'thresholds.csv').set_index('Model')
for model in performance.index:
    for scope,split,table in [('test','test',performance),('CV_OOF','train',cv)]:
        p=pd.read_csv(out/f'{model}_{scope}_predictions.csv',index_col='SampleID',float_precision='round_trip')
        assert set(p.index)==set(a.index[a.split==split]) and p.index.is_unique
        truth=p[[t+'_actual' for t in TARGETS]].to_numpy()
        prob=p[[t+'_probability' for t in TARGETS]].to_numpy()
        pred=p[[t+'_predicted' for t in TARGETS]].to_numpy()
        threshold=thresholds.loc[model,TARGETS].to_numpy(float)
        assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
        np.testing.assert_array_equal(pred,prob>=threshold)
        computed=metrics(truth,prob,threshold)
        np.testing.assert_allclose([computed[m] for m in METRICS],table.loc[model,METRICS],atol=1e-12)
    assert (out/f'{model}.joblib').stat().st_size>0
for name in ['multilabel_mcc','micro_aupr','micro_roc_auc']:
    assert (out/f'{name}.png').stat().st_size>0
result={'status':'passed','checks':['Exact source F(%) values and strict >0.1% masking',
    '138 samples and 370 microbiome-only columns','Unique IDs and disjoint site groups',
    'Five complete grouped CV validation assignments','All six saved OOF/test metric tables independently recomputed',
    'Saved binary predictions match frozen thresholds','All trained bundles and three bar charts exist']}
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
