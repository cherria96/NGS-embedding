import unittest
import numpy as np
import pandas as pd
from importance_multilabel import permute_column, summarize

class ImportanceTests(unittest.TestCase):
    def test_raw_column_permutation_preserves_other_predictors(self):
        x=pd.DataFrame({'Season':['Summer','Fall','Winter'],'HRT_d':[1.,np.nan,3.],'BAC_P::x':[.2,.4,.8]},index=['a','b','c'])
        original=x.copy(deep=True)
        z=permute_column(x,'HRT_d',np.array([2,0,1]))
        pd.testing.assert_frame_equal(x,original)
        pd.testing.assert_frame_equal(z.drop(columns='HRT_d'),x.drop(columns='HRT_d'))
        np.testing.assert_allclose(z.HRT_d,[3.,1.,np.nan],equal_nan=True)
        cat=permute_column(x,'Season',np.array([2,0,1]))
        self.assertEqual(cat.Season.tolist(),['Winter','Summer','Fall'])

    def test_signed_importance_and_ranking(self):
        drops=pd.DataFrame({'Feature':['HRT_d','HRT_d','T_C','T_C','Season','Season'],
                            'Micro-AUPR drop':[.1,.3,-.1,-.3,.1,.1],
                            'Multilabel MCC drop':[0,.2,-.2,0,.2,.2]})
        table=summarize(drops).set_index('Feature')
        self.assertEqual(table.index.tolist(),['HRT_d','Season','T_C'])
        self.assertAlmostEqual(table.loc['T_C','Micro-AUPR mean drop'],-.2)
        self.assertEqual(table.loc['Season','Type'],'Categorical')
        self.assertEqual(table.loc['T_C','Share of positive metadata importance (%)'],0)
        self.assertAlmostEqual(table['Share of positive metadata importance (%)'].sum(),100)

if __name__=='__main__': unittest.main()
