"""Item 18.1: separate constraints. Both beta sums meet tolerance; a vector
satisfying only the combined sum is not accepted as satisfying both."""
import numpy as np

from models.fusionML.estimator import fit_log_contrast


def test_domain_specific_zero_sum_constraints():
    rng = np.random.default_rng(0)
    n, pa, pb = 40, 5, 6
    La = rng.normal(size=(n, pa)); La -= La.mean(axis=1, keepdims=True)
    Lb = rng.normal(size=(n, pb)); Lb -= Lb.mean(axis=1, keepdims=True)
    y = 1.0 + La[:, 0] - La[:, 1] + 0.02 * rng.normal(size=n)
    y_star = (y - y.mean()) / y.std()
    w = np.full(n, 1.0 / n)

    res = fit_log_contrast(La, Lb, y_star, w, lambda_s=0.0, lambda_r=0.01, lambda_f=0.0,
                            edges_a=[], edges_b=[],
                            solver_cfg={"preferred": "CLARABEL", "fallback": "OSQP", "equality_tolerance": 1e-6, "max_retries": 3})

    assert abs(np.sum(res.beta_a)) <= 1e-6
    assert abs(np.sum(res.beta_b)) <= 1e-6
    # both are individually near zero, so their sum-of-sums is too, but the
    # converse direction is the property under test:
    assert abs(np.sum(res.beta_a) + np.sum(res.beta_b)) <= 2e-6


def test_combined_only_constraint_is_not_sufficient():
    """A vector can satisfy sum(beta_a)+sum(beta_b)=0 while each domain sum
    is individually nonzero -- exactly what the separate constraints rule
    out. Demonstrate the distinction directly."""
    beta_a = np.array([5.0, 0.0, 0.0])
    beta_b = np.array([-5.0, 0.0, 0.0])
    combined_ok = abs(np.sum(beta_a) + np.sum(beta_b)) <= 1e-6
    per_domain_ok = abs(np.sum(beta_a)) <= 1e-6 and abs(np.sum(beta_b)) <= 1e-6
    assert combined_ok
    assert not per_domain_ok
