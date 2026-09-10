import unittest
import numpy as np
import pandas as pd
from genomic_models import SampleAbundanceTaxa,WarningModel
from importance_genomic import transformed_probability

class GenomicTests(unittest.TestCase):
    def test_no_metadata_and_strict_training_union(self):
        x=pd.DataFrame({'BAC_P::a':[.1,.2,.05], 'ARC_G::heldout':[0,0,1.], 'Season':['Summer']*3,'HRT_d':[1]*3})
        s=SampleAbundanceTaxa().fit(x.iloc[:2])
        self.assertEqual(s.columns_,['BAC_P::a'])
        np.testing.assert_array_equal(s.transform(x)['BAC_P::a'],[0,.2,0])

    def test_batched_permutation_equals_raw_prediction(self):
        x=pd.DataFrame({'BAC_P::a':[0,.2,.1,.4,.5,.3], 'ARC_G::b':[1,.4,.2,.3,.5,.2]})
        y=np.array([[0,1,0,1,0],[1,0,1,0,1]]*3)
        model=WarningModel('RandomForest',{'n_estimators':5,'max_depth':2},42).fit(x,y,np.arange(6))
        z=model.preprocessor.transform(model.selector.transform(x))
        order=np.array([4,2,0,1,3,5]);raw=x.copy();raw['BAC_P::a']=x['BAC_P::a'].to_numpy()[order]
        index=list(model.preprocessor.get_feature_names_out()).index('BAC_P::a')
        perm=z.copy();perm[:,index]=z[order,index]
        np.testing.assert_allclose(transformed_probability(model,perm),model.predict_proba(raw))

if __name__=='__main__': unittest.main()
