"""Item 18.9: weighted decomposition. V_between + V_within equals the
declared equal-site population variance (v_all) for a toy repeated-site
dataset."""
import numpy as np
import pandas as pd
import pytest

from datasets.datapreprocess.preprocessing import site_balanced_weights


def test_between_plus_within_equals_site_balanced_variance():
    rng = np.random.default_rng(42)
    sites = {"A": 4, "B": 1, "C": 3, "D": 2}
    site_labels = []
    r = []
    site_true_mean = {"A": 0.5, "B": -1.2, "C": 0.1, "D": 2.0}
    for s, n_s in sites.items():
        site_labels.extend([s] * n_s)
        r.extend((site_true_mean[s] + rng.normal(0, 0.3, size=n_s)).tolist())
    site_labels = np.array(site_labels)
    r = np.array(r)

    w = site_balanced_weights(list(site_labels))
    mu = float(np.sum(w * r))
    v_all = float(np.sum(w * (r - mu) ** 2))

    df = pd.DataFrame({"site": site_labels, "r": r})
    site_means = df.groupby("site")["r"].mean()
    m_bar = site_means.mean()
    v_between = float(((site_means - m_bar) ** 2).mean())
    v_within = float(df.groupby("site")["r"].apply(lambda x: float(((x - x.mean()) ** 2).mean())).mean())

    assert v_all == pytest.approx(v_between + v_within, abs=1e-10)


def test_single_visit_site_contributes_zero_within_site_variance():
    site_labels = np.array(["A", "A", "B"])
    r = np.array([1.0, 3.0, 5.0])  # site B has a single visit
    df = pd.DataFrame({"site": site_labels, "r": r})
    within = df.groupby("site")["r"].apply(lambda x: float(((x - x.mean()) ** 2).mean()))
    assert within["B"] == 0.0
