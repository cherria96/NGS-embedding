import unittest
import numpy as np
import pandas as pd
from class_models import SampleAbundanceTaxa, WarningModel
from train_class import metrics, partition

class ClassTests(unittest.TestCase):
    def test_threshold_and_fixed_columns(self):
        x=pd.DataFrame({'BAC_C::a':[.1,.101,0], 'ARC_C::b':[0,0,.3], 'Season':['Summer']*3})
        s=SampleAbundanceTaxa().fit(x.iloc[:2])
        self.assertEqual(s.columns_, ['BAC_C::a','ARC_C::b'])
        np.testing.assert_allclose(s.transform(x), [[0,0],[.101,0],[0,.3]])

    def test_metrics_joint_and_score_based(self):
        y=np.array([[0,1,0,1,0],[1,0,1,0,1]])
        p=.1+.8*y
        a=metrics(y,p,.5); b=metrics(y,p,.99)
        self.assertEqual(set(a),{'Multilabel MCC','Micro-AUPR','Micro-ROC-AUC'})
        self.assertTrue(all(abs(v-1)<1e-12 for v in a.values()))
        self.assertEqual(b['Multilabel MCC'],0)
        self.assertEqual(a['Micro-AUPR'],b['Micro-AUPR'])
        self.assertEqual(a['Micro-ROC-AUC'],b['Micro-ROC-AUC'])

    def test_group_partition(self):
        groups=np.repeat(np.arange(15),4)
        y=np.random.default_rng(2).integers(0,2,(60,5))
        parts=partition(groups,y,[.2]*5,42,trials=50)
        self.assertEqual(sorted(np.concatenate(parts).tolist()), list(range(60)))
        for i,a in enumerate(parts):
            for b in parts[i+1:]: self.assertFalse(set(groups[a]) & set(groups[b]))

    def test_fold_local_scaling_and_constant_label(self):
        x=pd.DataFrame({'BAC_C::a':[.2,.3,.4,.5,.6,.7], 'ARC_C::b':[0]*6})
        y=np.zeros((6,5),int); y[::2,0]=1
        m=WarningModel('GLMNET',{'C':1,'l1_ratio':.5},42).fit(x,y,np.arange(6))
        np.testing.assert_allclose(m.preprocessor.named_transformers_['numeric'].named_steps['scale'].mean_, [.45,0])
        self.assertEqual(m.predict_proba(x).shape,(6,5))
        np.testing.assert_equal(m.predict_proba(x)[:,1:],0)

if __name__=='__main__': unittest.main()
