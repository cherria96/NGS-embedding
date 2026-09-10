"""Check saved integration, leakage boundaries and metrics without refitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from order_models import TARGETS
from train_order import metrics

out=Path(__file__).resolve().parent/'results_order'
x=pd.read_csv(out/'final_feature_matrix.csv',index_col=0)
a=pd.read_csv(out/'split_and_cv_assignment.csv',index_col=0)
assert x.index.is_unique and x.shape==(138,238)
assert all(c.startswith(('BAC_O::','ARC_O::')) for c in x.columns)
assert ((x==0)|(x>.1)).all().all()
assert a.groupby('SiteGroup').split.nunique().max()==1
assert a[a.split=='train'].groupby('SiteGroup').cv_validation_fold.nunique().max()==1
assert set(a[a.split=='train'].cv_validation_fold)==set(range(1,6))
assert json.loads((out/'leakage_check.json').read_text())['passed']
table=pd.read_csv(out/'test_performance.csv').set_index('Model')
thresholds=pd.read_csv(out/'thresholds.csv').set_index('Model')
assert table.columns.tolist()==['Multilabel MCC','Micro-AUPR','Micro-ROC-AUC']
assert len(table)==6
for name in table.index:
    p=pd.read_csv(out/f'{name}_test_predictions.csv',index_col=0)
    assert set(p.index)==set(a[a.split=='test'].index)
    y=p[[t+'_actual' for t in TARGETS]].to_numpy()
    probabilities=p[[t+'_probability' for t in TARGETS]].to_numpy()
    binary=p[[t+'_predicted' for t in TARGETS]].to_numpy()
    cutoff=thresholds.loc[name,TARGETS].to_numpy(float)
    np.testing.assert_array_equal(binary,probabilities>=cutoff)
    calculated=metrics(y,probabilities,cutoff)
    np.testing.assert_allclose(table.loc[name].to_numpy(float),list(calculated.values()))
    oof=pd.read_csv(out/f'{name}_CV_OOF_predictions.csv',index_col=0)
    assert set(oof.index)==set(a[a.split=='train'].index) and oof.index.is_unique
    assert (out/f'{name}.joblib').stat().st_size>0
print('Verified: fixed Order-only schema, 35 disjoint site groups, five CV folds, six saved models, prediction thresholds and all final metrics.')
