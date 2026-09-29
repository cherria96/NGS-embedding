"""Descriptive regularization paths and protocol checks; no test-driven tuning."""
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from fusionML.investigation_20260921.learning_curves import (
    init_worker,INPUTS,Workbench,Candidate,make_site_grouped_folds,sample_ids_for_sites,metric,pipe,OUT)
from fusionML.preprocessing import JointPreprocessor


def main():
    targets=['warning_ammonia_toxicity','warning_acid_base_balance']
    init_worker(targets)
    rows,filter_rows,graph_rows=[],[],[]
    replay={}
    for target in targets:
        cfg,inp,flat=INPUTS[target]
        site_map=inp.metadata.loc[inp.cohort_ids,'site_id'].to_dict()
        folds=make_site_grouped_folds(site_map,3,cfg['seed'])
        wb=Workbench(cfg,inp,flat)
        try:
            for fold in range(3):
                tr=sample_ids_for_sites(site_map,folds.train_sites_for_outer_fold(fold))
                te=sample_ids_for_sites(site_map,folds.sites_in_fold(fold))
                ridge=wb.fit_fusion(tr,Candidate(0,.1,0,None,None))
                for domain,counts,features in [('ARC',inp.rank_counts_arc,ridge.retained_a),('BAC',inp.rank_counts_bac,ridge.retained_b)]:
                    mass=counts.loc[tr,features].sum(axis=1)/counts.loc[tr].sum(axis=1)
                    det=(counts.loc[tr,features]>=cfg['preprocessing']['detection_count']).mean()
                    pseudomass=(len(features)*.5)/(counts.loc[tr,features].sum(axis=1)+len(features)*.5)
                    filter_rows.append(dict(target=target,fold=fold,domain=domain,input_taxa=counts.shape[1],
                                            retained=len(features),mean_retained_mass=mass.mean(),min_retained_mass=mass.min(),
                                            mean_detection_prevalence=det.mean(),max_pseudocount_fraction=pseudomass.max()))
                for graph,alpha in [('phylogeny',1.),('abundance',0.),('shared',.5)]:
                    for strength in [0.,.01,.1,.3,1.,3.,10.,30.,100.]:
                        cand=Candidate(0,.1,strength,alpha if strength else None,alpha if strength else None)
                        good,p,stats,pf=wb.predict(tr,te,cand,'domain_adaptive_hybrid')
                        tp=pipe.predict_with_partition_fit(pf,tr,inp,'genus')
                        yy=inp.metadata.loc[good,'target'];ss=inp.metadata.loc[good,'site_id']
                        ev=metric(yy,p,ss)
                        train=metric(inp.metadata.loc[tr,'target'],tp.prediction,inp.metadata.loc[tr,'site_id'])
                        rows.append(dict(target=target,fold=fold,graph=graph,lambda_f=strength,
                                         **stats,train_logloss=train['logloss'],test_logloss=ev['logloss'],test_auc=ev['auc']))
                        if strength==1 and graph in ['phylogeny','abundance']:
                            for domain,beta,edges in [('ARC',ridge.fit_result.beta_a,pf.edges_a_idx),('BAC',ridge.fit_result.beta_b,pf.edges_b_idx)]:
                                if not edges: continue
                                a,b,w=map(np.array,zip(*edges));a=a.astype(int);b=b.astype(int)
                                obs=np.average(np.abs(beta[a]-beta[b]),weights=w)
                                rng=np.random.default_rng(42);null=[]
                                for _ in range(250):
                                    perm=rng.permutation(beta);null.append(np.average(np.abs(perm[a]-perm[b]),weights=w))
                                graph_rows.append(dict(target=target,fold=fold,graph=graph,domain=domain,n_edges=len(edges),
                                                       opposite_sign_fraction=float(np.mean(beta[a]*beta[b]<0)),
                                                       ridge_abs_difference=obs,random_mean_abs_difference=np.mean(null),
                                                       ratio_to_random=obs/np.mean(null)))
                if target==targets[0] and fold==0:
                    cand=Candidate(0,.1,1,.5,.5)
                    cached=wb.fit_fusion(tr,cand)
                    wb.close()
                    original=pipe.fit_partition(tr,pipe.FAMILY_SPECS['domain_adaptive_hybrid'],cand,inp,'genus',cfg,
                        pipe.derive_seed(cfg['seed'],'matched_graph',*sorted(tr)))
                    p1=pipe.predict_with_partition_fit(cached,te,inp,'genus').prediction.to_numpy()
                    p2=pipe.predict_with_partition_fit(original,te,inp,'genus').prediction.to_numpy()
                    replay={'maximum_absolute_prediction_difference':float(np.max(np.abs(p1-p2))),
                            'maximum_zero_sum_residual':max(original.fit_result.equality_residual_a,original.fit_result.equality_residual_b)}
                    assert np.allclose(p1,p2,atol=1e-10,rtol=1e-10)
                    wb=Workbench(cfg,inp,flat)
        finally: wb.close()
    pd.DataFrame(rows).to_csv(OUT/'regularization_paths.csv',index=False)
    pd.DataFrame(filter_rows).to_csv(OUT/'filter_audit.csv',index=False)
    pd.DataFrame(graph_rows).to_csv(OUT/'graph_prior_probe.csv',index=False)
    (OUT/'cache_validation.json').write_text(json.dumps(replay,indent=2))

    sys.path.insert(0,str(ROOT/'Phylospec'))
    from evaluate_multilabel import site_grouped_cv,split_train_val
    table=pd.read_csv(ROOT/'Phylospec/multi_models/input_for_all_models_taxlevel/Genus/Warnings_merged/table.csv')
    old=pd.read_csv(ROOT/'data/final/metadata.csv',encoding='utf-8-sig').set_index('SampleID')
    flags=['acid_accumulation','acid_base_balance','ammonia_toxicity','biogas_quality','buffer_capacity']
    y=table[flags].to_numpy();sites=table.SampleID.map(old.Site).to_numpy()
    splits=list(site_grouped_cv(y,sites,5,1,42));val_rows=[]
    for k,(tr_all,te) in enumerate(splits):
        tr,va=split_train_val(y[tr_all],sites[tr_all],seed=42+k)
        val_rows.extend(tr_all[va].tolist())
    overlap=[dict(fold=k,n_test=len(te),n_test_ids_in_pooled_threshold_labels=len(set(te)&set(val_rows)))
             for k,(_,te) in enumerate(splits)]
    pd.DataFrame(overlap).to_csv(OUT/'legacy_threshold_overlap.csv',index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=pd.DataFrame(rows)
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for i,target in enumerate(targets):
        for graph in ['phylogeny','abundance','shared']:
            g=d[(d.target==target)&(d.graph==graph)].groupby('lambda_f').mean(numeric_only=True)
            axs[i,0].plot(g.index,g.n_active_groups,'o-',label=graph)
            axs[i,1].plot(g.index,g.test_logloss,'o-',label=graph)
        axs[i,0].set_ylabel('Active coefficient groups');axs[i,1].set_ylabel('Held-out site log-loss')
        for ax in axs[i]:
            ax.set_xscale('symlog',linthresh=.01);ax.set_xlabel('Fusion penalty λF (ridge fixed at 0.1)');ax.grid(alpha=.2)
            ax.set_title(target.replace('warning_',''));ax.legend()
    fig.suptitle('Compression does not guarantee prediction improvement\nDescriptive outer-fold paths; not used to select models')
    fig.tight_layout();fig.savefig(OUT/'compression_regularization.png',dpi=180);fig.savefig(OUT/'compression_regularization.pdf')
    print('Wrote mechanism probes',replay,flush=True)


if __name__=='__main__': main()
