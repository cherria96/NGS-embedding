"""Item 18.7: solver behavior. Objective components and equality residuals
on small synthetic problems; lambda_F=0 agrees with the corresponding
constrained no-fusion formulation (edges present but unused); a symmetric
toy example demonstrates fusion pulling connected coefficients together,
without requiring perfect group recovery."""
import numpy as np
import pytest

from models.fusionML.estimator import fit_log_contrast

SOLVER_CFG = {"preferred": "CLARABEL", "fallback": "OSQP", "equality_tolerance": 1e-6, "max_retries": 3}


def _toy(seed=11, n=50, pa=4, pb=4):
    rng = np.random.default_rng(seed)
    La = rng.normal(size=(n, pa)); La -= La.mean(axis=1, keepdims=True)
    Lb = rng.normal(size=(n, pb)); Lb -= Lb.mean(axis=1, keepdims=True)
    y = 0.5 + 1.0 * La[:, 0] - 1.0 * La[:, 1] + 0.6 * Lb[:, 0] - 0.6 * Lb[:, 1] + 0.02 * rng.normal(size=n)
    y_star = (y - y.mean()) / y.std()
    w = np.full(n, 1.0 / n)
    return La, Lb, y_star, w


def test_objective_components_and_equality_residuals():
    La, Lb, y_star, w = _toy()
    res = fit_log_contrast(La, Lb, y_star, w, lambda_s=0.01, lambda_r=0.1, lambda_f=0.0,
                            edges_a=[], edges_b=[], solver_cfg=SOLVER_CFG)
    assert res.status == "optimal"
    assert res.equality_residual_a <= 1e-6
    assert res.equality_residual_b <= 1e-6
    manual_ridge = 0.1 * (np.sum(res.beta_a ** 2) + np.sum(res.beta_b ** 2))
    assert res.penalty_ridge == pytest.approx(manual_ridge, rel=1e-4)
    manual_sparsity = 0.01 * (np.sum(np.abs(res.beta_a)) + np.sum(np.abs(res.beta_b)))
    assert res.penalty_sparsity == pytest.approx(manual_sparsity, rel=1e-4)
    resid = y_star - res.b0 - La @ res.beta_a - Lb @ res.beta_b
    manual_loss = float(np.sum(w * resid ** 2))
    assert res.loss_value == pytest.approx(manual_loss, rel=1e-4)


def test_lambda_f_zero_agrees_with_no_fusion_formulation_even_with_edges_present():
    La, Lb, y_star, w = _toy()
    res_no_edges = fit_log_contrast(La, Lb, y_star, w, lambda_s=0.0, lambda_r=0.05, lambda_f=0.0,
                                     edges_a=[], edges_b=[], solver_cfg=SOLVER_CFG)
    res_with_edges_but_zero_lambda = fit_log_contrast(
        La, Lb, y_star, w, lambda_s=0.0, lambda_r=0.05, lambda_f=0.0,
        edges_a=[(0, 1, 1.0), (2, 3, 1.0)], edges_b=[(0, 1, 1.0)], solver_cfg=SOLVER_CFG,
    )
    np.testing.assert_allclose(res_no_edges.beta_a, res_with_edges_but_zero_lambda.beta_a, atol=1e-6)
    np.testing.assert_allclose(res_no_edges.beta_b, res_with_edges_but_zero_lambda.beta_b, atol=1e-6)
    assert res_with_edges_but_zero_lambda.penalty_fusion == 0.0


def test_fusion_pulls_connected_coefficients_together_without_requiring_exact_recovery():
    rng = np.random.default_rng(21)
    n, pa, pb = 60, 4, 3
    La = rng.normal(size=(n, pa)); La -= La.mean(axis=1, keepdims=True)
    Lb = rng.normal(size=(n, pb)); Lb -= Lb.mean(axis=1, keepdims=True)
    # symmetric toy: taxa 0 and 1 have a similar, slightly different true effect
    y = 1.0 + 0.9 * La[:, 0] + 0.7 * La[:, 1] - 1.6 * La[:, 2] + 0.03 * rng.normal(size=n)
    y_star = (y - y.mean()) / y.std()
    w = np.full(n, 1.0 / n)

    diffs = []
    for lf in (0.0, 1.0, 10.0, 100.0):
        res = fit_log_contrast(La, Lb, y_star, w, lambda_s=0.0, lambda_r=0.01, lambda_f=lf,
                                edges_a=[(0, 1, 1.0)], edges_b=[], solver_cfg=SOLVER_CFG)
        diffs.append(abs(res.beta_a[0] - res.beta_a[1]))
    # monotonically non-increasing as fusion strength grows, converging near 0
    assert diffs[0] >= diffs[1] - 1e-6
    assert diffs[-1] < diffs[0]
    assert diffs[-1] < 1e-3
