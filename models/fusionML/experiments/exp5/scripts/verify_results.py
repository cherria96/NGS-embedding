"""Independent integrity checks and uncached re-fits of domain-restricted models."""


# Locate the repository independently of the script's directory depth.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_REPO_ROOT = next(p for p in _RepositoryPath(__file__).resolve().parents
                  if (p / "main.py").is_file() and (p / "configs").is_dir())
if str(_REPO_ROOT) not in _repository_sys.path:
    _repository_sys.path.insert(0, str(_REPO_ROOT))

import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
import run_missing_asv as runner
from fusionML.search import Candidate, FAMILY_SPECS


def main():
    tables=runner.BASE/'results/tables'
    inventory=pd.read_csv(tables/'model_fit_inventory.csv')
    groups=pd.read_csv(tables/'group_catalog.csv')
    coefficients=pd.read_csv(tables/'coefficients_annotated.csv')
    features=pd.read_csv(tables/'feature_associations.csv')
    assert len(inventory)==135 and inventory.status.eq('ok').sum()==125
    missing=inventory[inventory.status.ne('ok')]
    assert set(missing.target)=={'warning_biogas_quality'} and set(missing.resolution)=={'asv'}
    assert (features.positive_fits+features.negative_fits==features.active_fits).all()
    assert (features.active_fits<=features.retained_fits).all()
    assert (features.retained_fits<=features.successful_fits).all()
    assert not coefficients[coefficients.genus=='Bathyarchaeia'].role.ne('unassigned').any()
    assert not coefficients[coefficients.domain_mismatch_asvs>0].role.ne('unassigned').any()
    keys=['resolution','target','model','repeat','outer_fold','domain','group_id']
    counts=coefficients.groupby(keys).size().rename('actual_size').reset_index()
    merged=groups.merge(counts,on=keys,validate='one_to_one')
    assert (merged.n_features==merged.actual_size).all()
    assert np.allclose(groups.theta,groups.n_features*groups.mean_beta)
    assert groups[groups.resolved_genus_fraction==0].genus_purity.isna().all()
    reference=features[(features.resolution=='asv')&(features.model=='shared_hybrid')]
    acid=reference[(reference.target=='warning_acid_base_balance')&(reference.feature_id=='BAC::14a311b4802d5606cbf72b2dc465ae56')].iloc[0]
    assert acid.positive_fits==6 and acid.negative_fits==0
    biogas=reference[(reference.target=='warning_biogas_quality')&(reference.feature_id=='ARC::abb98aa85e9528ef54e3078ff40ac726')].iloc[0]
    assert biogas.positive_fits==4 and biogas.successful_fits==4
    target=runner.TARGETS[0]
    cfg=runner.config(target);inp=runner.pipe.load_all_inputs(cfg)
    inp.metadata['site_id']=inp.metadata.site_id.replace({'BSIb':'BSI','DGYb':'DGY'})
    split=pd.read_csv(runner.OUT/'splits.csv')
    split=split[(split.target==target)&(split.repeat==0)&(split.outer_fold==0)]
    train=split[split.role=='train'].sample_id.tolist();test=split[split.role=='test'].sample_id.tolist()
    rows=pd.read_csv(runner.OUT/'rows.csv');preds=pd.read_csv(runner.OUT/'predictions.csv')
    checks={}
    for model,forbidden in [('fusion_arc_only','BAC'),('fusion_bac_only','ARC')]:
        row=rows[(rows.target==target)&(rows.repeat==0)&(rows.outer_fold==0)&(rows.model==model)].iloc[0]
        expr=ast.parse(row.candidate,mode='eval').body
        assert isinstance(expr,ast.Call) and expr.func.id=='Candidate'
        candidate=Candidate(**{k.arg:ast.literal_eval(k.value) for k in expr.keywords})
        fit=runner.pipe.fit_partition(train,FAMILY_SPECS[model],candidate,inp,'asv',cfg,runner.pipe.derive_seed(cfg['seed'],'matched_graph',*sorted(train)))
        assert not (fit.edges_a_idx if forbidden=='ARC' else fit.edges_b_idx)
        actual=runner.pipe.predict_with_partition_fit(fit,test,inp,'asv')
        expected=preds[(preds.target==target)&(preds.repeat==0)&(preds.outer_fold==0)&(preds.model==model)].set_index('sample_id')
        error=float(np.max(abs(actual.loc[expected.index,'prediction']-expected.p)))
        assert error<1e-8,error
        checks[model+'_uncached_prediction_error']=error
    print(json.dumps(dict(data_integrity='passed',**checks),indent=2))


if __name__=='__main__':main()
