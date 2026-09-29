"""Matched, nested site-held-out learning curves; does not alter fusionML.

All transforms are fitted within each training partition. Candidate-independent
graphs are cached within a worker and use the same partition seed for all
families. This deliberately removes family-specific fold/bootstrap noise from
the production search. Whole training sites are subsampled in nested prefixes,
without inspecting their labels. Outer tests remain fixed across training sizes.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fusionml-investigation-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/fusionml-investigation-cache')
for env_name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[env_name] = '1'
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, fbeta_score
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier
import fusionML.pipeline as pipe
from fusionML.config import Config
from fusionML.search import Candidate, InteractionCandidate, FAMILY_SPECS, generate_candidates, generate_interaction_candidates, one_standard_error_select
from fusionML.folds import make_site_grouped_folds, sample_ids_for_sites
from fusionML.preprocessing import site_balanced_weights
from fusionML.evaluate import select_threshold_via_inner_oof_f2, per_sample_log_loss
from fusionML.io_taxonomy import _parse_lineage

OUT = Path(__file__).resolve().parent
FAMILIES = ['genus_ridge','genus_elastic_net','phylogeny_only','abundance_only',
            'shared_hybrid','domain_adaptive_hybrid','domain_adaptive_interaction']
BASELINES = ['elasticnet_RA_matched','xgboost_RA_matched','elasticnet_RA_flat','xgboost_RA_flat']
INPUTS = {}
RUN_OPTIONS = {}


def init_worker(targets, options=None):
    global RUN_OPTIONS
    RUN_OPTIONS = options or {}
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    for target in targets:
        cfg = Config.from_yaml(str(ROOT/'fusionML/configs/real_smoke_classification_phylo.yaml'))
        cfg.raw['columns']['target'] = target
        if RUN_OPTIONS.get('min_prevalence') is not None:
            cfg.raw['preprocessing']['min_sample_prevalence'] = RUN_OPTIONS['min_prevalence']
            cfg.raw['preprocessing']['min_site_prevalence'] = RUN_OPTIONS['min_prevalence']
        if RUN_OPTIONS.get('pseudocount') is not None:
            cfg.raw['preprocessing']['pseudocount'] = RUN_OPTIONS['pseudocount']
        inp = pipe.load_all_inputs(cfg)
        if RUN_OPTIONS.get('group_qc'):
            inp.metadata['site_id'] = inp.metadata.site_id.replace({'BSIb':'BSI','DGYb':'DGY'})
        flat = []
        for counts, tax, domain in [(inp.counts_arc_asv,inp.taxonomy_arc,'ARC'),(inp.counts_bac_asv,inp.taxonomy_bac,'BAC')]:
            keys = pd.Series({c: domain+'::'+_parse_lineage(tax.loc[c,'lineage']).get('g','Unclassified')
                              if c in tax.index else domain+'::Unclassified' for c in counts})
            agg = counts.T.groupby(keys).sum().T
            flat.append(agg.div(agg.sum(axis=1),axis=0))
        INPUTS[target] = (cfg, inp, pd.concat(flat,axis=1).loc[inp.cohort_ids])


def metric(y, p, sites, pred=None):
    y, p = np.asarray(y), np.asarray(p)
    ll = per_sample_log_loss(y,p)
    out = {'auc': roc_auc_score(y,p) if len(np.unique(y)) == 2 else np.nan,
           'ap': average_precision_score(y,p) if np.sum(y)>0 else np.nan,
           'logloss': float(pd.Series(ll).groupby(np.asarray(sites)).mean().mean()),
           'brier': float(np.mean((y-p)**2))}
    if pred is not None:
        out.update(mcc=matthews_corrcoef(y,pred),f2=fbeta_score(y,pred,beta=2,zero_division=0))
    return out


class Workbench:
    """Reuse exact production fitting code with partition-local graph caching."""
    def __init__(self,cfg,inp,flat):
        self.cfg,self.inp,self.flat = cfg,inp,flat
        self.fit_cache,self.graph_cache,self.inter_cache,self.prep_cache = {},{},{},{}
        self.originals = (pipe.build_phylogenetic_graph,pipe.build_abundance_graph,pipe.fit_interaction_transform)
        phylo, abundance, interaction = self.originals
        def phylo_cached(tree,domain,features,k):
            key=('p',domain,tuple(features),k)
            if key not in self.graph_cache: self.graph_cache[key]=phylo(tree,domain,features,k)
            return self.graph_cache[key]
        def abundance_cached(log_q,detection,sites,cfg,seed,domain):
            key=('a',domain,tuple(log_q.index),tuple(log_q.columns),seed)
            if key not in self.graph_cache: self.graph_cache[key]=abundance(log_q,detection,sites,cfg,seed,domain)
            return self.graph_cache[key]
        def interaction_cached(*args):
            key=tuple(args[0].index)
            if key not in self.inter_cache: self.inter_cache[key]=interaction(*args)
            return self.inter_cache[key]
        pipe.build_phylogenetic_graph=phylo_cached
        pipe.build_abundance_graph=abundance_cached
        pipe.fit_interaction_transform=interaction_cached

    def close(self):
        pipe.build_phylogenetic_graph,pipe.build_abundance_graph,pipe.fit_interaction_transform=self.originals

    def fit_fusion(self,ids,candidate):
        key=(tuple(ids),isinstance(candidate,InteractionCandidate),candidate.key())
        if key not in self.fit_cache:
            fam='domain_adaptive_interaction' if isinstance(candidate,InteractionCandidate) else 'domain_adaptive_hybrid'
            seed=pipe.derive_seed(self.cfg['seed'],'matched_graph',*sorted(ids))
            self.fit_cache[key]=pipe.fit_partition(ids,FAMILY_SPECS[fam],candidate,self.inp,'genus',self.cfg,seed)
        return self.fit_cache[key]

    def baseline_arrays(self,train_ids,eval_ids,representation):
        key=(tuple(train_ids),representation)
        if key not in self.prep_cache:
            if representation=='flat':
                data=self.flat
                cols=data.loc[train_ids].columns[data.loc[train_ids].var()>1e-14]
                data=data[cols]
            else:
                pf=self.fit_fusion(train_ids,Candidate(0,.1,0,None,None))
                a=self.inp.rank_counts_arc[pf.retained_a]
                b=self.inp.rank_counts_bac[pf.retained_b]
                data=pd.concat([a.div(a.sum(axis=1),axis=0),b.div(b.sum(axis=1),axis=0)],axis=1)
            if not np.isfinite(data.loc[train_ids].to_numpy()).all(): raise ValueError('unscorable baseline train row')
            scaler=StandardScaler().fit(data.loc[train_ids].to_numpy())
            self.prep_cache[key]=(data,scaler)
        data,scaler=self.prep_cache[key]
        return data.loc[train_ids].to_numpy(),data.loc[eval_ids].to_numpy(),scaler

    def predict(self,train_ids,eval_ids,candidate,model):
        sites=self.inp.metadata.loc[train_ids,'site_id'].tolist()
        y=self.inp.metadata.loc[train_ids,'target'].to_numpy()
        if len(np.unique(y))<2: raise ValueError('single class training partition')
        if model not in BASELINES:
            pf=self.fit_fusion(train_ids,candidate)
            preds=pipe.predict_with_partition_fit(pf,eval_ids,self.inp,'genus')
            good=preds.index[preds.scorable].tolist()
            stats=pipe.count_active(pf,1e-6)
            stats['retained_taxa']=len(pf.retained_a)+len(pf.retained_b)
            stats['n_edges']=len(pf.edges_a_idx)+len(pf.edges_b_idx)
            return good,preds.loc[good,'prediction'].to_numpy(),stats,pf
        rep='flat' if model.endswith('flat') else 'matched'
        x,xe,scaler=self.baseline_arrays(train_ids,eval_ids,rep)
        ok=np.isfinite(xe).all(axis=1)
        good=list(np.asarray(eval_ids)[ok]);xe=xe[ok]
        weights=site_balanced_weights(sites)*len(train_ids)
        if model.startswith('elasticnet'):
            clf=LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=.5,C=candidate,
                                   max_iter=5000,tol=1e-4,random_state=42)
            clf.fit(scaler.transform(x),y,sample_weight=weights)
            p=clf.predict_proba(scaler.transform(xe))[:,1]
            nactive=int((np.abs(clf.coef_)>1e-6).sum())
        else:
            depth,rate=candidate
            clf=XGBClassifier(n_estimators=200,max_depth=depth,learning_rate=rate,subsample=.8,
                              colsample_bytree=.8,eval_metric='logloss',n_jobs=1,random_state=42)
            clf.fit(x,y,sample_weight=weights)
            p=clf.predict_proba(xe)[:,1]
            nactive=int((clf.feature_importances_>0).sum())
        return good,p,dict(retained_taxa=x.shape[1],n_active_taxa=nactive,n_active_groups=nactive,
                           n_active_products=0,n_required_taxa=nactive,n_edges=0),None


def candidate_sets(cfg,fold,repeat,full):
    sets={}
    for name in FAMILIES:
        base='domain_adaptive_hybrid' if name=='domain_adaptive_interaction' else name
        seed=pipe.derive_seed(cfg['seed'],'candidates',base,fold if repeat==0 else f'{repeat}:{fold}')
        candidates=generate_candidates(FAMILY_SPECS[base],cfg['search'],seed)
        sets[name]=generate_interaction_candidates(candidates,cfg['interaction']) if name.endswith('interaction') else candidates
    for name in BASELINES:
        sets[name]=[.01,.1,1.,10.] if name.startswith('elasticnet') else [(1,.05),(1,.1),(3,.05),(3,.1)]
    if full:
        # Full additive grid (6 no-fusion + 2*3*3*9 fusion points), including
        # weaker fusion absent from the smoke grid. Chosen without outer labels.
        expanded=[]
        for ls in [0,.01]:
            for lr in [.01,.1,1.]:
                expanded.append(Candidate(ls,lr,0,None,None))
                for lf in [.1,1.,10.]:
                    for aa in [0,.5,1.]:
                        for ab in [0,.5,1.]: expanded.append(Candidate(ls,lr,lf,aa,ab))
        sets['adaptive_expanded']=expanded
    requested=RUN_OPTIONS.get('models')
    return {k:v for k,v in sets.items() if not requested or k in requested}


def run_task(task):
    target,repeat,fold,fraction=task
    cfg,inp,flat=INPUTS[target]
    ids=inp.cohort_ids
    site_map=inp.metadata.loc[ids,'site_id'].to_dict()
    outer=make_site_grouped_folds(site_map,3,cfg['seed']+repeat)
    full_train=sorted(outer.train_sites_for_outer_fold(fold))
    test=sample_ids_for_sites(site_map,outer.sites_in_fold(fold))
    rng=np.random.default_rng(pipe.derive_seed(cfg['seed'],'learning_sites',repeat,fold))
    shuffled=list(rng.permutation(full_train))
    use_sites=shuffled[:max(2,int(np.ceil(len(shuffled)*fraction)))]
    train=sample_ids_for_sites(site_map,use_sites)
    inner_map={sid:site_map[sid] for sid in train}
    inner=make_site_grouped_folds(inner_map,2,pipe.derive_seed(cfg['seed'],'matched_inner',repeat,fold,fraction))
    splits=[(sample_ids_for_sites(inner_map,inner.train_sites_for_outer_fold(k)),
             sample_ids_for_sites(inner_map,inner.sites_in_fold(k))) for k in range(2)]
    assert not set(site_map[s] for s in train)&set(site_map[s] for s in test)
    for tr,va in splits: assert not set(site_map[s] for s in tr)&set(site_map[s] for s in va)
    tags=dict(target=target,repeat=repeat,outer_fold=fold,fraction=fraction,n_train=len(train),
              n_train_sites=len(use_sites),n_train_positive=int(inp.metadata.loc[train,'target'].sum()),
              n_train_positive_sites=inp.metadata.loc[train].query('target == 1').site_id.nunique())
    split_rows=[]
    for role,values in [('train',train),('test',test)]:
        split_rows.extend([{**tags,'role':role,'sample_id':s,'site_id':site_map[s]} for s in values])
    wb=Workbench(cfg,inp,flat)
    rows,predictions,search_rows,failures=[],[],[],[]
    started=time.time()
    try:
        for model,candidates in candidate_sets(cfg,fold,repeat,fraction==1).items():
            scores=[]
            for candidate in candidates:
                yy,pp,ss,complexities=[],[],[],[]
                error=None
                for tr,va in splits:
                    try:
                        good,probs,stats,_=wb.predict(tr,va,candidate,model)
                        if len(good)/len(va)<.8: raise ValueError('validation coverage below 0.8')
                        yy.extend(inp.metadata.loc[good,'target'].tolist());pp.extend(probs);ss.extend(inp.metadata.loc[good,'site_id'])
                        complexities.append((stats['n_active_products'],stats['n_required_taxa'],stats['n_active_groups'])
                                            if model=='domain_adaptive_interaction' else (stats['n_active_groups'],stats['n_active_taxa']))
                    except Exception as exc:
                        error=f'{type(exc).__name__}: {exc}';break
                if error:
                    failures.append({**tags,'model':model,'candidate':str(candidate),'error':error})
                    continue
                per_site=pd.Series(per_sample_log_loss(np.array(yy),np.array(pp))).groupby(np.asarray(ss)).mean().to_numpy()
                complexity=tuple(np.mean(complexities,axis=0))
                scores.append(dict(candidate=candidate,loss=float(per_site.mean()),site_losses=per_site,
                                   complexity=complexity,y=np.array(yy),p=np.array(pp),sites=ss))
            if not scores:
                rows.append({**tags,'model':model,'selection':'one_se','status':'unavailable'})
                continue
            best=min(scores,key=lambda r:r['loss'])
            se=float(np.std(best['site_losses'])/np.sqrt(len(best['site_losses'])))
            if model in BASELINES:
                chosen=best # baseline model selection optimizes the same inner log-loss
            else:
                chosen=min((r for r in scores if r['loss']<=best['loss']+se),key=lambda r:(r['complexity'],r['loss'],r['candidate'].key()))
            selections={'one_se':chosen}
            if fraction==1 and model not in BASELINES: selections['min_loss']=best
            for s in scores:
                search_rows.append({**tags,'model':model,'candidate':str(s['candidate']),'loss':s['loss'],
                                    'complexity':str(s['complexity']),'selected_one_se':s is chosen,'selected_min_loss':s is best})
            for rule,selected in selections.items():
                threshold=select_threshold_via_inner_oof_f2(selected['y'],selected['p'],selected['sites'])
                try:
                    good,probs,stats,pf=wb.predict(train,test,selected['candidate'],model)
                    if pf is not None:
                        trpred=pipe.predict_with_partition_fit(pf,train,inp,'genus')
                        trgood=trpred.index[trpred.scorable].tolist();trprob=trpred.loc[trgood,'prediction'].to_numpy()
                    else:
                        trgood,trprob,_,_=wb.predict(train,train,selected['candidate'],model)
                    test_metric=metric(inp.metadata.loc[good,'target'],probs,inp.metadata.loc[good,'site_id'],probs>=threshold)
                    train_metric=metric(inp.metadata.loc[trgood,'target'],trprob,inp.metadata.loc[trgood,'site_id'],trprob>=threshold)
                    rows.append({**tags,'model':model,'selection':rule,'status':'ok','candidate':str(selected['candidate']),
                                 'threshold':threshold,'inner_loss':selected['loss'],'minimum_inner_loss':best['loss'],
                                 'inner_se':se,'coverage':len(good)/len(test),**stats,
                                 **{'train_'+k:v for k,v in train_metric.items()},**{'test_'+k:v for k,v in test_metric.items()}})
                    predictions.extend([{**tags,'model':model,'selection':rule,'sample_id':s,'site_id':site_map[s],
                                         'y':float(inp.metadata.loc[s,'target']),'p':float(p),'threshold':threshold}
                                        for s,p in zip(good,probs)])
                except Exception as exc:
                    failures.append({**tags,'model':model,'candidate':str(selected['candidate']),'error':'outer refit '+str(exc)})
                    rows.append({**tags,'model':model,'selection':rule,'status':'unavailable'})
    finally:
        wb.close()
    return dict(rows=rows,predictions=predictions,search=search_rows,failures=failures,splits=split_rows,
                seconds=time.time()-started,task=task)


def summarize(output):
    parts=[json.loads(p.read_text()) for p in sorted((output/'parts').glob('*.json'))]
    for name in ['rows','predictions','search','failures','splits']:
        pd.DataFrame([r for part in parts for r in part[name]]).to_csv(output/f'{name}.csv',index=False)
    rows=pd.read_csv(output/'rows.csv');preds=pd.read_csv(output/'predictions.csv')
    availability=rows.groupby(['target','fraction','model','selection']).agg(
        planned_fits=('status','size'),successful_fits=('status',lambda x:int((x=='ok').sum())))
    availability.to_csv(output/'availability.csv')
    pooled=[]
    for key,d in preds.groupby(['target','repeat','fraction','model','selection']):
        vals=metric(d.y,d.p,d.site_id,d.p>=d.threshold)
        pooled.append(dict(zip(['target','repeat','fraction','model','selection'],key),**vals,n_scored=len(d),
                           n_outer_folds=d.outer_fold.nunique()))
    pooled=pd.DataFrame(pooled);pooled.to_csv(output/'pooled_metrics.csv',index=False)
    agg=rows[rows.status=='ok'].groupby(['target','fraction','model','selection']).agg(
        n_fits=('outer_fold','size'),mean_train_sites=('n_train_sites','mean'),mean_train_n=('n_train','mean'),
        train_logloss=('train_logloss','mean'),test_logloss=('test_logloss','mean'),
        train_auc=('train_auc','mean'),test_fold_auc=('test_auc','mean'),
        retained_taxa=('retained_taxa','mean'),active_taxa=('n_active_taxa','mean'),active_groups=('n_active_groups','mean'))
    agg.to_csv(output/'learning_summary.csv')
    if rows.fraction.nunique()<2:
        print(pooled.to_string(index=False))
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    model_names=FAMILIES+BASELINES
    for target in pooled.target.unique():
        fig,axs=plt.subplots(3,4,figsize=(15,10),sharex=True,sharey=True)
        for ax,model in zip(axs.flat,model_names):
            d=rows[(rows.target==target)&(rows.model==model)&(rows.selection=='one_se')&(rows.status=='ok')]
            if len(d):
                stats=d.groupby('fraction').agg(n=('n_train_sites','mean'),tr=('train_logloss','mean'),te=('test_logloss','mean'),sd=('test_logloss','std'))
                ax.plot(stats.n,stats.tr,'o-',label='Training');ax.plot(stats.n,stats.te,'o-',label='Held-out sites')
                ax.fill_between(stats.n,np.maximum(0,stats.te-stats.sd),stats.te+stats.sd,alpha=.13)
                for fraction,point in stats.iterrows():
                    available=availability.loc[(target,fraction,model,'one_se')]
                    if available.successful_fits < available.planned_fits:
                        ax.annotate(f'{available.successful_fits}/{available.planned_fits} fits',
                                    (point.n,point.te),xytext=(3,6),textcoords='offset points',fontsize=7)
            ax.set_title(model.replace('domain_adaptive','adaptive'),fontsize=9);ax.grid(alpha=.2)
        axs.flat[-1].axis('off');axs.flat[-1].legend(*axs.flat[0].get_legend_handles_labels(),loc='center')
        fig.suptitle(target+' — nested learning curves\nScores conditional on successful fits. Band: ±1 SD across folds; not a confidence interval',fontsize=12)
        fig.supxlabel('Training sites (whole sites sampled; fixed test sites)');fig.supylabel('Equal-site log-loss (lower is better)')
        fig.tight_layout();fig.savefig(output/f'{target}_learning_curves.png',dpi=180);fig.savefig(output/f'{target}_learning_curves.pdf');plt.close(fig)
    print(pooled[(pooled.fraction==1)&(pooled.selection=='one_se')].to_string(index=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--targets',nargs='+',default=['warning_ammonia_toxicity','warning_acid_base_balance'])
    ap.add_argument('--repeats',type=int,default=2);ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--fractions',type=float,nargs='+',default=[.5,.75,1.])
    ap.add_argument('--output',type=Path,default=OUT/'learning');ap.add_argument('--summarize-only',action='store_true')
    ap.add_argument('--group-qc',action='store_true',help='Keep documented BSI/BSIb and DGY/DGYb QC pairs together')
    ap.add_argument('--models',nargs='+',default=None)
    ap.add_argument('--min-prevalence',type=float,default=None)
    ap.add_argument('--pseudocount',type=float,default=None)
    args=ap.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    if args.summarize_only: summarize(args.output);return
    (args.output/'parts').mkdir(exist_ok=True)
    manifest={'targets':args.targets,'repeats':args.repeats,'fractions':args.fractions,'outer_folds':3,'inner_folds':2,
              'grouping':'BSIb→BSI, DGYb→DGY' if args.group_qc else 'literal chemistry site_id (production convention)',
              'min_prevalence':args.min_prevalence,'pseudocount':args.pseudocount,'models':args.models,
              'subsampling':'nested random whole-site prefixes; no label stratification',
              'selection':'production one-SE for fusion, minimum inner site-logloss for ML; F2 threshold on inner OOF only',
              'changes_from_production':['same inner folds across families','same graph bootstrap seeds per partition',
                                         'additional full-size expanded adaptive grid and minimum-loss selection ablation'],
              'hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [ROOT/'fusionML/configs/real_smoke_classification_phylo.yaml',ROOT/'data/final/Dat_chem.xlsx',Path(__file__)]}}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    tasks=[(t,r,k,f) for t in args.targets for r in range(args.repeats) for k in range(3) for f in args.fractions]
    def filename(task): return '_'.join(str(x) for x in task)+'.json'
    todo=[task for task in tasks if not (args.output/'parts'/filename(task)).exists()]
    options=dict(group_qc=args.group_qc,models=args.models,min_prevalence=args.min_prevalence,pseudocount=args.pseudocount)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,initargs=(args.targets,options)) as executor:
        futures={executor.submit(run_task,task):task for task in todo}
        for future in as_completed(futures):
            result=future.result()
            (args.output/'parts'/filename(result['task'])).write_text(json.dumps(result,default=float))
            print('DONE',result['task'],round(result['seconds'],1),'seconds',len(result['rows']),'fits',flush=True)
    summarize(args.output)


if __name__=='__main__': main()
