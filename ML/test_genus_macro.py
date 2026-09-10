import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, precision_recall_curve, auc, roc_auc_score
from genus_macro_models import SampleAbundanceTaxa, WarningModel
from train_genus_macro import metrics, choose_threshold, select_joint

class MacroProtocolTests(unittest.TestCase):
    def test_training_union_strict_boundary(self):
        x=pd.DataFrame({'BAC_G::a':[.1,.10001,1.], 'ARC_G::b':[0,0,10.]})
        s=SampleAbundanceTaxa().fit(x.iloc[:2])
        self.assertEqual(s.columns_,['BAC_G::a'])
        np.testing.assert_array_equal(s.transform(x).values,[[0],[.10001],[1.]])
        with self.assertRaises(ValueError):s.fit(x.assign(eff_pH=7))

    def test_macro_is_exact_five_label_mean(self):
        y=np.array([[0,0,0,0,1],[1,0,0,1,0],[0,1,0,0,1],[1,1,1,0,0]])
        p=np.random.default_rng(21).random(y.shape)
        threshold=np.array([.1,.2,.3,.4,.5])
        got=metrics(y,p,threshold)
        mcc=[];pr=[];roc=[]
        for j in range(5):
            mcc.append(matthews_corrcoef(y[:,j],p[:,j]>=threshold[j]))
            precision,recall,_=precision_recall_curve(y[:,j],p[:,j])
            pr.append(auc(recall,precision));roc.append(roc_auc_score(y[:,j],p[:,j]))
        self.assertEqual(got,{'Macro MCC':np.mean(mcc),'Macro AUPR':np.mean(pr),'Macro ROC-AUC':np.mean(roc)})
        self.assertNotAlmostEqual(got['Macro MCC'],matthews_corrcoef(y.ravel(),(p>=threshold).ravel()))
        self.assertEqual(metrics(y,p,.9)['Macro AUPR'],got['Macro AUPR'])

    def test_undefined_not_silently_dropped(self):
        y=np.tile([[0,0,0,0,0],[1,1,1,1,0]],(3,1))
        got=metrics(y,y.astype(float))
        self.assertAlmostEqual(got['Macro MCC'],.8)
        self.assertTrue(np.isnan(got['Macro AUPR']))
        self.assertTrue(np.isnan(got['Macro ROC-AUC']))

    def test_thresholds_per_label(self):
        y=np.tile([[0]*5,[1]*5],(5,1))
        p=np.where(y==1,np.array([.1,.2,.4,.7,.9]),np.array([.02,.12,.3,.6,.8]))
        thresholds=choose_threshold(y,p)
        self.assertGreater(len(set(thresholds)),1)
        np.testing.assert_array_equal(p>=thresholds,y)

    def test_constant_training_labels(self):
        x=pd.DataFrame({'BAC_G::a':[0.,.2,.3,1.6]*5,'ARC_G::b':[.4,0.,.2,0.]*5})
        y=np.tile([[0,1,0,0,1],[1,0,0,0,1]],(10,1))
        model=WarningModel('RandomForest',{'n_estimators':5,'max_depth':2}).fit(x,y,np.repeat(np.arange(5),4))
        p=model.predict_proba(x)
        np.testing.assert_array_equal(p[:,2:4],0);np.testing.assert_array_equal(p[:,4],1)

if __name__=='__main__':unittest.main()
