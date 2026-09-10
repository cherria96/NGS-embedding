"""Protocol invariants for the >0.1% microbiome-only analysis."""
import unittest
import numpy as np
import pandas as pd
from family_point1_models import SampleAbundanceTaxa, WarningModel, preprocessing
from train_family_point1 import metrics, partition

class ProtocolTests(unittest.TestCase):
    def test_strict_boundary_and_fixed_schema(self):
        x = pd.DataFrame({'BAC_F::a': [0, .1, .100001, .15],
                          'ARC_F::b': [0, 0, 0, .2]})
        selector = SampleAbundanceTaxa().fit(x.iloc[:2])
        self.assertEqual(set(selector.columns_), set(x.columns))
        np.testing.assert_array_equal(selector.transform(x).values,
                                      [[0, 0], [0, 0], [.100001, 0], [.15, .2]])

    def test_metadata_rejected(self):
        for c in ['Season', 'Site', 'SampleID', 'eff_pH', 'warning_biogas_quality']:
            with self.assertRaises(ValueError):
                SampleAbundanceTaxa().fit(pd.DataFrame({'BAC_F::a':[1], c:[1]}))

    def test_scaler_training_only(self):
        x = pd.DataFrame({'BAC_F::a': [1., 3., 1000.]})
        p = preprocessing(x).fit(x.iloc[:2])
        np.testing.assert_allclose(p.transform(x).ravel(), [-1, 1, 998])
        raw = preprocessing(x, scale=False).fit(x.iloc[:2])
        np.testing.assert_array_equal(raw.transform(x).ravel(), x.iloc[:,0])

    def test_group_disjointness_and_coverage(self):
        groups = np.repeat(np.arange(15),4)
        y = np.random.default_rng(2).integers(0,2,(60,5))
        parts = partition(groups,y,[.8,.2],42,trials=20)
        self.assertFalse(set(groups[parts[0]]) & set(groups[parts[1]]))
        self.assertEqual(sorted(np.concatenate(parts)),list(range(60)))

    def test_joint_metrics_and_single_class_outputs(self):
        x=pd.DataFrame({'BAC_F::a':[0.,.2,.3,1.6]*5,'ARC_F::b':[.4,0.,.2,0.]*5})
        y=np.tile([[0,1,0,0,1],[1,0,0,0,1]],(10,1))
        model=WarningModel('RandomForest',{'n_estimators':5,'max_depth':2}).fit(x,y,np.repeat(np.arange(5),4))
        p=model.predict_proba(x)
        self.assertEqual(p.shape,(20,5))
        np.testing.assert_array_equal(p[:,2:4],0)
        np.testing.assert_array_equal(p[:,4],1)
        perfect=metrics(y,y.astype(float))
        self.assertEqual(perfect,{'Multilabel MCC':1.,'Micro-AUPR':1.,'Micro-ROC-AUC':1.})

if __name__ == '__main__': unittest.main()
