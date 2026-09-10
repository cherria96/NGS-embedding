import unittest
import numpy as np
import pandas as pd
from family_models import CATS, NUMS, SampleAbundanceTaxa
from train_family import aggregate_families, metrics, select_joint, partition

class FamilyTests(unittest.TestCase):
    def test_aggregation_precedes_threshold(self):
        rows = [['a','Bacteria','p','c','o','F','g','s',4],
                ['b','Bacteria','p','c','o','F','g','s',3],
                ['c','Bacteria','p','c','o','F','g','s',5],
                ['d','Bacteria','p','c','o','Other','g','s',988]]
        otu = pd.DataFrame(rows, columns=['Feature ID','Domain','Phylum','Class','Order','Family','Genus','Species','1-1-X'])
        f = aggregate_families(otu)
        self.assertAlmostEqual(f.loc['F','1-1-X'],1.2)
        self.assertAlmostEqual(f['1-1-X'].sum(),100)

    def test_prevalence_boundary_mask_root_and_test_isolation(self):
        x = pd.DataFrame({**{c:['a']*21 for c in CATS}, **{c:[1.]*21 for c in NUMS},
            'BAC_F::five_percent':[16.]+[1.]*19+[81.],
            'ARC_F::test_only':[0.]*20+[16.]})
        selector = SampleAbundanceTaxa().fit(x.iloc[:20])
        z = selector.transform(x)
        self.assertEqual(selector.selected_['BAC_F'],['BAC_F::five_percent'])
        self.assertEqual(selector.selected_['ARC_F'],[])
        np.testing.assert_array_equal(z['BAC_F::five_percent'], [2.]+[0.]*19+[3.])
        self.assertEqual(SampleAbundanceTaxa().fit(x.iloc[:20].reindex(range(21),fill_value=0)).selected_['BAC_F'],[])

    def test_three_pooled_metrics(self):
        y=np.array([[1,0,0,0,0],[0,1,0,0,0]])
        p=y*.8+.1
        self.assertEqual(metrics(y,p),{'Multilabel MCC':1.,'Micro-AUPR':1.,'Micro-ROC-AUC':1.})
        q=p.copy(); q[0,0]=.2; q[0,2]=.8
        self.assertAlmostEqual(metrics(y,q)['Multilabel MCC'],.375)
        for metric in ['Micro-AUPR','Micro-ROC-AUC']:
            self.assertEqual(metrics(y,p,.95)[metric],metrics(y,p)[metric])

    def test_joint_selection_uses_all_three(self):
        rows=[{'Multilabel MCC':.1,'Micro-AUPR':.9,'Micro-ROC-AUC':.5},
              {'Multilabel MCC':.7,'Micro-AUPR':.8,'Micro-ROC-AUC':.9}]
        self.assertEqual(select_joint(rows),1)

    def test_group_integrity(self):
        g=np.repeat(np.arange(10),4); y=np.zeros((40,5),int); y[::3]=1
        a,b=partition(g,y,[.8,.2],42,trials=20)
        self.assertFalse(set(g[a]) & set(g[b]))
        self.assertEqual(sorted(np.r_[a,b].tolist()),list(range(40)))

if __name__=='__main__': unittest.main()
