import unittest
import numpy as np
import pandas as pd
from family_genomic_models import SampleAbundanceTaxa, WarningModel
from train_family_genomic import metrics

class GenomicFamilyTests(unittest.TestCase):
    def test_metadata_rejected(self):
        for column in ['Season','HRT_d','eff_pH','warning_acid_base_balance','Site']:
            with self.assertRaises(ValueError):
                SampleAbundanceTaxa().fit(pd.DataFrame({'BAC_F::a':[2.],column:[1]}))

    def test_threshold_prevalence_root(self):
        x=pd.DataFrame({'BAC_F::a':[16.]+[1.]*19+[81.], 'ARC_F::test_only':[0.]*20+[16.]})
        s=SampleAbundanceTaxa().fit(x.iloc[:20])
        self.assertEqual(s.columns_,['BAC_F::a'])
        np.testing.assert_array_equal(s.transform(x)['BAC_F::a'],[2.]+[0.]*19+[3.])

    def test_numeric_only_bundle(self):
        x=pd.DataFrame({'BAC_F::a':[0.,2.,3.,16.]*5,'ARC_F::b':[4.,0.,2.,0.]*5})
        y=np.tile([[0,1,0,0,0],[1,0,0,0,0]],(10,1))
        model=WarningModel('RandomForest',{'n_estimators':5,'max_depth':2}).fit(x,y,np.repeat(np.arange(5),4))
        self.assertEqual(set(model.preprocessor.get_feature_names_out()),set(x.columns))
        p=model.predict_proba(x)
        self.assertEqual(p.shape,(20,5))
        self.assertEqual(set(metrics(y,p)),{'Multilabel MCC','Micro-AUPR','Micro-ROC-AUC'})

if __name__=='__main__': unittest.main()
