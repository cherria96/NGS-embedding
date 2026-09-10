"""Check permutation scoring and whole-category feature grouping."""
import unittest
import numpy as np
from sklearn.metrics import average_precision_score
from feature_importance_warnings import ap_many, encoded_groups, predict_encoded
from warning_models import WarningModel
from test_warning_models import LeakageTests

class ImportanceTests(unittest.TestCase):
    def test_ap_matches_sklearn_with_tied_probabilities(self):
        rng=np.random.default_rng(13)
        for n in [5,24,100]:
            y=rng.integers(0,2,n);y[0]=1
            p=np.round(rng.random((20,n)),1)
            np.testing.assert_allclose(ap_many(y,p),[average_precision_score(y,r) for r in p],atol=1e-12)

    def test_no_positives_are_undefined(self):
        self.assertTrue(np.isnan(ap_many(np.zeros(5),np.ones((3,5)))).all())

    def test_encoded_feature_groups_and_predictions(self):
        X=LeakageTests().frame(12)
        X.loc[::2,'Season']='Winter'
        y=np.tile([0,1],6)[:,None]*np.ones((1,5),int)
        m=WarningModel('KNN',{'n_neighbors':3}).fit(X,y,np.repeat(range(6),2))
        z=m.preprocessor.transform(m.selector.transform(X))
        g=encoded_groups(m)
        self.assertEqual(len(g['Season']),2)
        self.assertEqual(sorted(sum(g.values(),[])),list(range(z.shape[1])))
        np.testing.assert_allclose(predict_encoded(m,z),m.predict_proba(X))
        # Joint permutation keeps one-hot vectors valid, unlike dummy-wise shuffles.
        zz=z.copy(); zz[:,g['Season']]=z[::-1][:,g['Season']]
        np.testing.assert_allclose(zz[:,g['Season']].sum(axis=1),1)

if __name__=='__main__': unittest.main()
