import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from multilabel_models import CATS, NUMS, SampleAbundanceTaxa
from train_multilabel import metrics, choose_threshold, partition, predictions

class MultilabelTests(unittest.TestCase):
    def test_strict_sample_mask_and_training_union(self):
        x = pd.DataFrame({**{c:['a','b','c'] for c in CATS}, **{c:[1.,2.,3.] for c in NUMS},
                          'BAC_P::a':[.1,.15,.05], 'BAC_P::test_only':[0,0,4.]})
        s = SampleAbundanceTaxa().fit(x.iloc[:2])
        result = s.transform(x)
        self.assertEqual(s.selected_['BAC_P'], ['BAC_P::a'])
        np.testing.assert_array_equal(result['BAC_P::a'], [0,.15,0])
        self.assertNotIn('BAC_P::test_only', result)

    def test_pooled_metrics_and_probability_area(self):
        y = np.array([[1,0,0,0,0],[0,1,0,0,0]])
        p = y*.8+.1
        self.assertEqual(metrics(y,p), {'Multilabel MCC':1.,'Micro-AUPR':1.})
        # TP=1,TN=7,FP=1,FN=1 => MCC=(7-1)/sqrt(2*2*8*8)=0.375
        q=p.copy();q[0,0]=.2;q[0,2]=.8
        self.assertAlmostEqual(metrics(y,q)['Multilabel MCC'],.375)
        self.assertEqual(metrics(y,p,.95)['Micro-AUPR'],1.)
        self.assertGreaterEqual(metrics(y,p,choose_threshold(y,p))['Multilabel MCC'],metrics(y,p)['Multilabel MCC'])

    def test_prediction_export_with_five_thresholds(self):
        frame=pd.DataFrame({'Site':['a','b'], 'SiteGroup':['a','b'], 'Season':['Summer','Fall']})
        y=np.zeros((2,5),dtype=int)
        p=np.full((2,5),.4)
        with tempfile.TemporaryDirectory() as directory:
            predictions(Path(directory),'model','test',frame,y,p,np.array([.3,.5,.3,.5,.3]))
            saved=pd.read_csv(Path(directory)/'model_test_predictions.csv')
            np.testing.assert_array_equal(saved.filter(like='_predicted').to_numpy(),[[1,0,1,0,1]]*2)

    def test_groups_do_not_cross_partitions(self):
        g=np.repeat(np.arange(10),4)
        y=np.zeros((40,5),dtype=int);y[::3]=1
        parts=partition(g,y,[.8,.2],42,trials=20)
        self.assertFalse(set(g[parts[0]]) & set(g[parts[1]]))
        self.assertEqual(len(set(g[parts[0]])),8)
        self.assertEqual(sorted(np.concatenate(parts).tolist()),list(range(40)))

if __name__=='__main__': unittest.main()
