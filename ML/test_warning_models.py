"""Regression tests for sample-specific selection and group isolation."""
import unittest
import numpy as np
import pandas as pd
from warning_models import SampleTopTaxa, BLOCKS, NUMS, CATS, WarningModel
from train_warnings import partition, metrics

class LeakageTests(unittest.TestCase):
    def frame(self, n=12):
        X = pd.DataFrame({c: np.arange(n, dtype=float) for c in NUMS})
        for c in CATS:
            X[c] = 'category'
        for block, k in BLOCKS.items():
            for j in range(k+2):
                X[f'{block}::taxon{j:02}'] = float(k+2-j)
        return X

    def test_union_ignores_evaluation_only_taxa_and_keeps_sample_ranks(self):
        X = self.frame(3)
        selector = SampleTopTaxa().fit(X.iloc[:2])
        X.loc[2, 'BAC_P::taxon11'] = 100
        result = selector.transform(X.iloc[2:])
        self.assertNotIn('BAC_P::taxon11', result)
        # Test-only winner displaces last training taxon before projection.
        self.assertEqual(result['BAC_P::taxon09'].iloc[0], 0)
        self.assertEqual(result['BAC_P::taxon00'].iloc[0], 12)
        self.assertEqual(len(selector.selected_['BAC_P']), 10)

    def test_union_includes_different_sample_top_taxa(self):
        X = self.frame(2)
        X.loc[1, 'BAC_P::taxon11'] = 100
        s = SampleTopTaxa().fit(X)
        self.assertEqual(len(s.selected_['BAC_P']), 11)
        self.assertEqual(s.transform(X).loc[0, 'BAC_P::taxon11'], 0)

    def test_group_partitions_cover_each_sample_once(self):
        groups = np.repeat(np.arange(15), 4)
        y = np.random.default_rng(2).integers(0, 2, (60, 5))
        folds = partition(groups, y, [.2]*5, 42, trials=50)
        self.assertEqual(sorted(np.concatenate(folds)), list(range(60)))
        for i, a in enumerate(folds):
            for b in folds[i+1:]:
                self.assertFalse(set(groups[a]) & set(groups[b]))
        self.assertTrue(all(np.array_equal(a, b) for a,b in zip(folds, partition(groups,y,[.2]*5,42,trials=50))))

    def test_constant_labels_and_unknown_categories(self):
        X = self.frame()
        y = np.zeros((12, 5), dtype=int); y[:, 1] = 1
        model = WarningModel('KNN', {'n_neighbors': 3}).fit(X, y, np.repeat(range(6),2))
        X.loc[0, 'Season'] = 'unknown'
        p = model.predict_proba(X)
        np.testing.assert_array_equal(p, y)
        self.assertTrue(np.isnan(metrics(y, p)['AUPR_warning_acid_base_balance']))

if __name__ == '__main__':
    unittest.main()
