import unittest
import numpy as np
import pandas as pd
from order_models import SampleAbundanceTaxa, WarningModel
from train_order import metrics, rank_candidates

class OrderTests(unittest.TestCase):
    def test_strict_mask_and_fixed_schema(self):
        x=pd.DataFrame({'BAC_O::a':[.1,.101,0],'ARC_O::heldout':[0,0,1.]})
        s=SampleAbundanceTaxa().fit(x.iloc[:2])
        np.testing.assert_array_equal(s.transform(x).to_numpy(),[[0,0],[.101,0],[0,1]])
        with self.assertRaises(ValueError): SampleAbundanceTaxa().fit(x.assign(Season='Summer'))
        with self.assertRaises(ValueError): s.transform(x.assign(**{'BAC_O::a':np.nan}))

    def test_joint_metrics_and_three_metric_selection(self):
        y=np.array([[0,1,0,1,0],[1,0,1,0,1]])
        self.assertEqual(list(metrics(y,y)),['Multilabel MCC','Micro-AUPR','Micro-ROC-AUC'])
        self.assertTrue(all(v==1 for v in metrics(y,y).values()))
        rows=[dict(zip(metrics(y,y),v)) for v in [[1,.1,.1],[.8,.8,.8],[.1,1,1]]]
        best,_=rank_candidates(rows)
        self.assertEqual(best,2)

    def test_scaler_fitted_only_on_training_and_constant_label(self):
        x=pd.DataFrame({'BAC_O::a':[1.,2.,3.,4.,100.],'ARC_O::b':[0.,0.,0.,0.,2.]})
        y=np.array([[0,0,0,0,0],[1,0,0,0,0],[0,0,0,0,0],[1,0,0,0,0]])
        model=WarningModel('GLMNET',{'C':1,'l1_ratio':.5},42).fit(x.iloc[:4],y,np.arange(4))
        np.testing.assert_allclose(model.preprocessor.named_steps['scale'].mean_,[2.5,0])
        p=model.predict_proba(x.iloc[4:]);self.assertEqual(p.shape,(1,5))
        np.testing.assert_array_equal(p[:,1:],0)

if __name__=='__main__':unittest.main()
