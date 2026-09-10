"""Independent artifact verification: no model fitting or test prediction."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, precision_recall_curve, auc, roc_auc_score
from class_macro_models import TARGETS

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'results_class_macro');args=parser.parse_args();out=args.output_dir
    read=lambda p:pd.read_csv(p,float_precision='round_trip')
    assign=read(out/'split_and_cv_assignment.csv').set_index('SampleID')
    X=read(out/'final_feature_matrix.csv').set_index('SampleID')
    metadata=pd.read_csv(Path(__file__).resolve().parents[1]/'data/final/metadata.csv').set_index('SampleID')
    assert assign.groupby('SiteGroup').split.nunique().max()==1
    assert assign.groupby('Site').split.nunique().max()==1
    train=assign.index[assign.split=='train'];test=assign.index[assign.split=='test']
    assert assign.loc[train].groupby('SiteGroup').cv_validation_fold.nunique().max()==1
    assert set(assign.loc[train].cv_validation_fold)==set(range(1,6))
    sources=[]
    for d in ['BAC','ARC']:
        f=pd.read_excel(Path(__file__).resolve().parents[1]/f'data/final/Dat_{d}.xlsx',sheet_name='C(%)').set_index('Class').T
        f.index=['-'.join(c.split('-')[:2]) for c in f.index];f.columns=[d+'_C::'+c for c in f.columns];sources.append(f)
    source=pd.concat(sources,axis=1).loc[X.index]
    expected=source.where(source>.1,0)
    columns=expected.loc[train].columns[(expected.loc[train]>0).any()]
    assert set(columns)==set(X.columns)
    np.testing.assert_array_equal(X.values,expected[X.columns].values)
    descriptive=read(out/'descriptive_all_sample_union.csv').set_index('SampleID')
    union=expected.loc[:,(expected>0).any()]
    np.testing.assert_array_equal(descriptive.values,union[descriptive.columns].values)
    folds=read(out/'cv_feature_columns.csv')
    for fold in range(1,6):
        fit=assign.index[(assign.split=='train')&(assign.cv_validation_fold!=fold)]
        expected_cols=set(expected.columns[(expected.loc[fit]>0).any()])
        assert set(folds[folds.fold==fold].feature)==expected_cols
    thresholds=read(out/'thresholds.csv');per=read(out/'per_warning_metrics.csv')
    taxa=read(out/'selected_taxa.csv')
    assert set(taxa.taxonomic_level)=={'Class'}
    assert all(c.startswith(('BAC_C::','ARC_C::')) for c in X.columns)
    assert set(taxa.feature)==set(X.columns)
    tuning=read(out/'hyperparameter_tuning.csv')
    config=json.loads((out/'run_config.json').read_text())
    assert tuning.groupby('Model').candidate.nunique().eq(config['n_iter']).all()
    assert tuning.groupby(['Model','candidate']).scope.nunique().eq(6).all()
    ranking=read(out/'cv_model_selection.csv')
    metric_cols=['Macro MCC','Macro AUPR','Macro ROC-AUC']
    rank=ranking[metric_cols].rank(ascending=False).mean(axis=1)
    selected=min(range(len(ranking)),key=lambda i:(rank.iloc[i],-ranking.iloc[i]['Macro AUPR'],-ranking.iloc[i]['Macro ROC-AUC'],-ranking.iloc[i]['Macro MCC']))
    assert ranking.iloc[selected].Model==json.loads((out/'cv_selected_model.json').read_text())['Model']
    checked=0
    for scope,tablefile,per_scope,ids in [('test','test_performance.csv','test',test),('CV_OOF','cv_performance.csv','CV_OOF_optimized',train)]:
        table=read(out/tablefile).set_index('Model')
        for name in ['RandomForest','KNN','SVM','NNET','XGBoost','GLMNET']:
            pred=read(out/f'{name}_{scope}_predictions.csv').set_index('SampleID')
            assert set(pred.index)==set(ids)
            scores=[]
            for t in TARGETS:
                y=pred[t+'_actual'].to_numpy();p=pred[t+'_probability'].to_numpy();b=pred[t+'_predicted'].to_numpy()
                np.testing.assert_array_equal(y,metadata.loc[pred.index,t])
                threshold=thresholds[(thresholds.Model==name)&(thresholds.target==t)].threshold.item()
                np.testing.assert_array_equal(b,p>=threshold)
                if scope=='CV_OOF':
                    grid=np.linspace(.01,.99,99)
                    expected_threshold=max(grid,key=lambda cut:(matthews_corrcoef(y,p>=cut),-abs(cut-.5)))
                    np.testing.assert_allclose(threshold,expected_threshold,rtol=0,atol=1e-15)
                precision,recall,_=precision_recall_curve(y,p)
                values=[matthews_corrcoef(y,b),auc(recall,precision),roc_auc_score(y,p)]
                saved=per[(per.Model==name)&(per.scope==per_scope)&(per.target==t)][['MCC','AUPR','ROC-AUC']].to_numpy()[0]
                np.testing.assert_allclose(values,saved,rtol=0,atol=1e-14)
                scores.append(values)
            key='Random Forest' if scope=='test' and name=='RandomForest' else name
            np.testing.assert_allclose(np.mean(scores,axis=0),table.loc[key].to_numpy(float),rtol=0,atol=1e-14)
            checked+=1
    result={'passed':True,'prediction_tables_recomputed':checked,'samples':len(X),'training_features':len(X.columns),'descriptive_union_features':len(descriptive.columns),
        'checks':['source percentages and strict sample mask','training-only final and fold unions','site disjointness','exact target matches','frozen thresholds and binary predictions','five per-label scores and arithmetic macro means'],
        'test_predictions_generated':0}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
