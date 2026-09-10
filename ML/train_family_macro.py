#!/usr/bin/env python3
"""Genomic Family binary relevance with equal-weight five-label macro metrics."""
from pathlib import Path
import argparse, hashlib, json, platform, warnings
import joblib
import numpy as np
import pandas as pd
import sklearn, xgboost, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, precision_recall_curve, auc, roc_auc_score
from sklearn.model_selection import ParameterSampler
from family_macro_models import TARGETS, BLOCKS, SampleAbundanceTaxa, WarningModel
from train_family_point1 import inspect, partition, label_report, predictions, save_json, SPACES, ROOT

METRICS = ['Macro MCC', 'Macro AUPR', 'Macro ROC-AUC']
DISPLAY = {'RandomForest': 'Random Forest'}

def per_label(y, p, threshold=.5):
    rows=[]
    binary=p >= threshold
    for j,t in enumerate(TARGETS):
        both=len(np.unique(y[:,j])) == 2
        if both:
            precision,recall,_=precision_recall_curve(y[:,j],p[:,j])
            pr=float(auc(recall,precision)); roc=float(roc_auc_score(y[:,j],p[:,j]))
        else:
            pr=roc=float('nan')
        rows.append({'target':t,'positives':int(y[:,j].sum()),'samples':len(y),
                     'MCC':float(matthews_corrcoef(y[:,j],binary[:,j])), 'AUPR':pr,'ROC-AUC':roc,
                     'both_classes_present':both})
    return pd.DataFrame(rows)

def metrics(y,p,threshold=.5):
    rows=per_label(y,p,threshold)
    # np.mean deliberately propagates undefined labels: never silently average fewer than five.
    return {macro:float(np.mean(rows[single].to_numpy())) for macro,single in zip(METRICS,['MCC','AUPR','ROC-AUC'])}

def select_joint(rows):
    table=pd.DataFrame(rows)
    if not np.isfinite(table[METRICS].to_numpy()).all():
        raise ValueError('Selection requires all five labels evaluable in pooled training OOF data')
    rank=table[METRICS].rank(ascending=False).mean(axis=1)
    return min(range(len(rows)),key=lambda i:(rank.iloc[i],-rows[i]['Macro AUPR'],-rows[i]['Macro ROC-AUC'],-rows[i]['Macro MCC']))

def choose_threshold(y,p):
    grid=np.linspace(.01,.99,99)
    return np.array([max(grid,key=lambda t:(matthews_corrcoef(y[:,j],p[:,j]>=t),-abs(t-.5)))
                     if len(np.unique(y[:,j]))==2 else .5 for j in range(5)])

def table_md(df):
    return '| Model | Macro MCC | Macro AUPR | Macro ROC-AUC |\n|---|---:|---:|---:|\n'+'\n'.join(
        '| '+DISPLAY.get(r['Model'],r['Model'])+' | '+' | '.join(f'{r[k]:.4f}' for k in METRICS)+' |' for r in df.to_dict('records'))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/final')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'ML/results_family_macro')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--n-iter',type=int,default=8)
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args(); out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    if (out/'test_performance.csv').exists():
        raise ValueError('Completed output exists: use a new output directory to preserve the frozen evaluation')
    m,X,y,labels,structures=inspect(args.data_dir,out,{'BSIb':'BSI','DGYb':'DGY'})
    g=m.SiteGroup.to_numpy()
    train,test=partition(g,y,[.8,.2],args.seed)
    Xtr,ytr,gtr=X.iloc[train],y[train],g[train]
    folds=partition(gtr,ytr,[.2]*5,args.seed+1)
    cv=[(np.setdiff1d(np.arange(len(train)),v),v) for v in folds]
    assignment=m[['Site','SiteGroup','Season']].copy(); assignment['split']='test'
    assignment.loc[Xtr.index,'split']='train'; assignment['cv_validation_fold']=pd.NA
    fold_features=[]
    assert not set(gtr)&set(g[test])
    for f,(a,b) in enumerate(cv,1):
        assert not set(gtr[a])&set(gtr[b])
        assignment.loc[Xtr.index[b],'cv_validation_fold']=f
        labels+=label_report(ytr[a],f'cv_training_{f}')+label_report(ytr[b],f'cv_validation_{f}')
        sel=SampleAbundanceTaxa().fit(Xtr.iloc[a])
        fold_features += [{'fold':f,'feature':c} for c in sel.columns_]
    assignment.to_csv(out/'split_and_cv_assignment.csv')
    assignment.groupby(['split','cv_validation_fold'],dropna=False).agg(samples=('Site','size'),site_groups=('SiteGroup','nunique'),sites=('Site',lambda v:'|'.join(sorted(set(v)))),groups=('SiteGroup',lambda v:'|'.join(sorted(set(v))))).to_csv(out/'partition_summary.csv')
    labels+=label_report(ytr,'train')+label_report(y[test],'test')
    pd.DataFrame(labels).to_csv(out/'label_distribution.csv',index=False)
    pd.DataFrame(fold_features).to_csv(out/'cv_feature_columns.csv',index=False)
    selector=SampleAbundanceTaxa().fit(Xtr)
    selector.transform(X).to_csv(out/'final_feature_matrix.csv')
    X.where(X>.1,0).loc[:,(X>.1).any()].to_csv(out/'descriptive_all_sample_union.csv')
    taxa=pd.DataFrame([{'feature':c,'taxon':c.split('::')[1],'domain':'Bacteria' if c.startswith('BAC') else 'Archaea','taxonomic_level':'Family',
                       'samples_above_0_1_percent':int((X[c]>.1).sum()),'training_samples_above_0_1_percent':int((Xtr[c]>.1).sum()),
                       'included_in_final_matrix':c in selector.columns_} for c in X])
    taxa.to_csv(out/'taxa_audit.csv',index=False)
    taxa[taxa.included_in_final_matrix].to_csv(out/'selected_taxa.csv',index=False)
    for domain in ['Bacteria','Archaea']:
        taxa[(taxa.domain==domain)&taxa.included_in_final_matrix].to_csv(out/f'selected_{domain}_families.csv',index=False)
    (out/'final_feature_list.txt').write_text('\n'.join(selector.columns_)+'\n')
    config={'seed':args.seed,'n_iter':args.n_iter,'targets':TARGETS,'search_spaces':SPACES,
        'feature_rule':'Strict per-sample >0.1% from F(%) sheets; preserve original percentages, mask <=0.1 to zero. Union fitted exclusively on training portion, including inner calibration. No other selection or transformation.',
        'protocol_conflict':'All-sample union conflicts with prohibition on test-informed feature construction. All-sample union exported descriptively; modeling uses training-only union.',
        'selection':'Equal-weight mean descending rank of pooled training OOF Macro MCC at fixed 0.5, Macro AUPR, Macro ROC-AUC. Ties favor Macro AUPR, ROC-AUC, MCC, then candidate order.',
        'threshold':'Per-label maximize MCC on chosen-candidate training OOF scores, grid 0.01..0.99 step .01, ties closest to .5. Frozen before test. Selection CV uses predefined .5; optimized OOF MCC is a tuning estimate.',
        'metrics':'Each macro is arithmetic mean of exactly five binary-label metrics. AUPR is trapezoidal precision-recall area, not average precision. MCC zero-denominator=0. AUPR/ROC-AUC undefined for single-class truth; macro propagates NaN, never skips labels. Pooled OOF is used for tuning; single-class validation folds remain recorded.',
        'preprocessing':'Training-only median imputation (empty fallback zero); z-score KNN/SVM/NNET/GLMNET, trees unscaled. Abundances validated finite; missing entire domains excluded, never zero-imputed.',
        'imbalance':'RF/SVM/GLMNET balanced class weights; XGBoost train negative/positive ratio; KNN/NNET per-label random oversampling to majority count with seed+j, training only after preprocessing. Single-class training uses constant estimator.',
        'SVM':'Inner three-fold GroupKFold OOF margin logistic calibration with refitted feature union/scaling.',
        'GLMNET':'sklearn saga elastic-net logistic regression analogue, not the R glmnet implementation; coefficients regularized without subsequent feature removal.',
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'xgboost':xgboost.__version__,'matplotlib':matplotlib.__version__,'joblib':joblib.__version__},
        'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.data_dir/'metadata.csv',args.data_dir/'Dat_BAC.xlsx',args.data_dir/'Dat_ARC.xlsx']}}
    save_json(out/'run_config.json',config)
    save_json(out/'leakage_check.json',{'all_metadata_excluded':list(m.columns)+['SampleID'],'predictors':selector.columns_,
        'passed':['Only Family percentage columns in predictors','No identities, season, effluent, operation, substrate or warning predictors','Site groups disjoint','Training-only feature union and preprocessing'],
        'timing_limitation':'Files do not establish sampling before warning onset. Results concern contemporaneous warning-state classification, not prospective early warning.',
        'target_limitation':'Supplied binary labels retained; missing chemistry may be encoded as unflagged zero. Source missingness exported.'})
    audit=f'Matched {len(m)} samples; {m.Site.nunique()} Site labels; {len(set(g))} groups after QC replicate grouping. Train {len(train)} samples/{len(set(gtr))} groups; test {len(test)} samples/{len(set(g[test]))} groups. Final features {len(selector.columns_)}.\n'
    (out/'pretraining_audit.txt').write_text(audit+json.dumps(config,indent=2))
    print(audit,flush=True)
    if args.audit_only:return
    tuning=[]; caught_rows=[]; fitted={}; selection=[]; cv_results=[]; per_rows=[]; thresholds=[]
    for name,space in SPACES.items():
        candidates=list(ParameterSampler(space,n_iter=args.n_iter,random_state=args.seed)); scores=[]; outputs=[]
        for k,params in enumerate(candidates):
            oof=np.zeros(ytr.shape)
            for f,(a,b) in enumerate(cv,1):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model=WarningModel(name,params,args.seed).fit(Xtr.iloc[a],ytr[a],gtr[a])
                    oof[b]=model.predict_proba(Xtr.iloc[b])
                caught_rows += [{'Model':name,'candidate':k,'fold':f,'category':w.category.__name__,'message':str(w.message)} for w in caught]
                tuning.append({'Model':name,'candidate':k,'scope':f'fold_{f}','params':json.dumps(params),**metrics(ytr[b],oof[b])})
            score=metrics(ytr,oof); scores.append(score); outputs.append(oof)
            tuning.append({'Model':name,'candidate':k,'scope':'pooled_OOF','params':json.dumps(params),**score})
            pd.DataFrame(tuning).to_csv(out/'hyperparameter_tuning.csv',index=False)
            print(name,k+1,len(candidates),score,flush=True)
        best=select_joint(scores); threshold=choose_threshold(ytr,outputs[best])
        selection.append({'Model':name,**scores[best]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            final=WarningModel(name,candidates[best],args.seed).fit(Xtr,ytr,gtr)
        caught_rows += [{'Model':name,'candidate':best,'fold':'final','category':w.category.__name__,'message':str(w.message)} for w in caught]
        final.thresholds=threshold; fitted[name]=final
        joblib.dump(final,out/f'{name}.joblib')
        joblib.dump({'selector':final.selector,'preprocessor':final.preprocessor},out/f'{name}_preprocessing.joblib')
        save_json(out/f'{name}_best_params.json',{'params':candidates[best],'candidate':best,'combinations_evaluated':len(candidates),'CV_selection_at_0_5':scores[best]})
        thresholds += [{'Model':name,'target':t,'threshold':float(v)} for t,v in zip(TARGETS,threshold)]
        predictions(out,name,'CV_OOF',m.loc[Xtr.index],ytr,outputs[best],threshold)
        cv_results.append({'Model':name,**metrics(ytr,outputs[best],threshold)})
        per_rows.append(per_label(ytr,outputs[best],threshold).assign(Model=name,scope='CV_OOF_optimized'))
    ranked=pd.DataFrame(selection);ranked['mean_metric_rank']=ranked[METRICS].rank(ascending=False).mean(axis=1)
    ranked.to_csv(out/'cv_model_selection.csv',index=False)
    selected=selection[select_joint(selection)]['Model']
    save_json(out/'cv_selected_model.json',{'Model':selected,'rule':config['selection']})
    pd.DataFrame(cv_results).to_csv(out/'cv_performance.csv',index=False)
    pd.DataFrame(thresholds).to_csv(out/'thresholds.csv',index=False)
    pd.DataFrame(caught_rows,columns=['Model','candidate','fold','category','message']).to_csv(out/'fit_warnings.csv',index=False)
    # All models, thresholds and selection are persisted before any test prediction.
    test_rows=[]
    for name,final in fitted.items():
        p=final.predict_proba(X.iloc[test])
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        predictions(out,name,'test',m.iloc[test],y[test],p,final.thresholds)
        test_rows.append({'Model':DISPLAY.get(name,name),**metrics(y[test],p,final.thresholds)})
        per_rows.append(per_label(y[test],p,final.thresholds).assign(Model=name,scope='test'))
    result=pd.DataFrame(test_rows);result.to_csv(out/'test_performance.csv',index=False)
    pd.concat(per_rows).to_csv(out/'per_warning_metrics.csv',index=False)
    for metric,filename in zip(METRICS,['macro_mcc','macro_aupr','macro_roc_auc']):
        fig,ax=plt.subplots(figsize=(9,5)); bars=ax.bar(result.Model,result[metric],color=['#db8c32' if v==result[metric].max() else '#287c8e' for v in result[metric]])
        ax.bar_label(bars,fmt='%.3f',padding=3); ax.set(ylabel=metric,title=f'Held-out sites: {metric}',ylim=(min(0,result[metric].min()-.08),result[metric].max()+.12))
        fig.tight_layout();fig.savefig(out/f'{filename}.png',dpi=200);fig.savefig(out/f'{filename}.svg');plt.close(fig)
    # Interpret already-fitted models on training only. Never refit/select features from importance.
    importance=[]
    for name in ['RandomForest','XGBoost']:
        final=fitted[name]
        for j,model in enumerate(final.models):
            if hasattr(model,'feature_importances_'):
                importance += [{'Model':name,'target':TARGETS[j],'feature':c,'importance':float(v),'domain':'Bacteria' if c.startswith('BAC') else 'Archaea'} for c,v in zip(final.selector.columns_,model.feature_importances_)]
    imp=pd.DataFrame(importance);imp.to_csv(out/'feature_importance_per_warning.csv',index=False)
    aggregate=imp.groupby(['Model','domain','feature'],as_index=False).importance.mean().sort_values(['Model','importance'],ascending=[True,False])
    aggregate.to_csv(out/'feature_importance_macro.csv',index=False)
    leaders={metric:result.loc[result[metric].idxmax(),'Model'] for metric in METRICS}
    report=['# Family microbiome-only multi-label warning prediction',f'CV-selected model: **{DISPLAY.get(selected,selected)}**. Selection was frozen before test evaluation.',
        '## Final test comparison',table_md(result),'Test metric leaders: '+json.dumps(leaders),
        '## Verified cohort',audit,'Training site groups: '+', '.join(sorted(set(gtr))), 'Test site groups: '+', '.join(sorted(set(g[test]))),
        'See partition_summary.csv for all five folds, sites and counts; label_distribution.csv for every label in each partition. Missing ARC samples 1-32\' and 4-24 were excluded. There are 37 Site labels and 35 groups, not 34 complete sites. BSIb/BSI and DGYb/DGY stay together.',
        '## Methods and protocol reconciliation']
    report += [f'**{k}**: {config[k]}' for k in ['feature_rule','protocol_conflict','selection','threshold','metrics','preprocessing','imbalance','SVM','GLMNET']]
    report += ['## Training OOF tuning estimates',table_md(pd.DataFrame(cv_results)),
        'These optimized-threshold OOF scores reuse tuning data and are optimistic; they are not nested unbiased validation estimates. Selection uses fixed-0.5 MCC recorded separately. Some validation folds lack rare positives: their AUPR/ROC macro is undefined, and is never replaced by a four-label mean.',
        '## Label consistency on held-out sites','| Model | Warning | Positives | MCC | AUPR | ROC-AUC |','|---|---|---:|---:|---:|---:|']
    for frame in per_rows:
        if frame.scope.iloc[0]=='test':
            report += [f'| {DISPLAY.get(r["Model"],r["Model"])} | {r["target"]} | {r["positives"]} | {r["MCC"]:.3f} | {r["AUPR"]:.3f} | {r["ROC-AUC"]:.3f} |' for r in frame.to_dict('records')]
    report += ['## Feature contributions','Training-fit RF impurity and XGBoost gain importances are predictive associations, not causal effects. Correlated compositional features can share importance; rankings may be unstable with this cohort. All features remain in the models. Below are the leading five per domain for presentation only; complete rankings are exported.']
    for (name,domain),frame in aggregate.groupby(['Model','domain']):
        report += [f'**{name}, {domain}**: '+', '.join(f'{r.feature.split("::")[1]} ({r.importance:.4f})' for r in frame.head(5).itertuples())]
    report += ['## Limits','Sampling times do not establish prospective prediction. This is contemporaneous classification. Existing zeros with missing target-source chemistry mean not flagged, not confirmed normal. The deterministic seed-42 holdout appeared in earlier project analyses: isolated in this run, but not a previously unseen external cohort. Few sites and very few rare-warning positives make all comparisons uncertain. A macro mean weights all five warnings equally but does not imply consistent performance or exact five-label-vector recovery.',f'Recorded fit warnings: {len(caught_rows)}; consult fit_warnings.csv for convergence limitations.',
        '## Reproducibility',f'Run `python ML/train_family_macro.py --seed {args.seed} --n-iter {args.n_iter} --output-dir NEW_DIRECTORY`. See run_config.json for versions, source hashes and full search spaces. Load joblib with ML on PYTHONPATH and supply raw Family percentages; preprocessing and thresholds are inside the bundle.']
    (out/'report.md').write_text('\n\n'.join(report)+'\n')
    from report_family_macro import enrich
    enrich(out)
    print(result.to_string(index=False),flush=True)

if __name__=='__main__':main()
