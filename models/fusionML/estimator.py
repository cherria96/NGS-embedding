"""Constrained convex log-contrast fusion estimator (design-spec item 13).

Objective (standardized-target scale, exactly as item 13 specifies, no
undocumented factor of n, 1/2 or 1/p):

    sum_i w_i (y*_i - b0 - l_i^A.beta_A - l_i^B.beta_B)^2
    + lambda_S (||beta_A||_1 + ||beta_B||_1)
    + lambda_R (||beta_A||_2^2 + ||beta_B||_2^2)
    + lambda_F sum_d sum_{(j,k) in E_d} W_jk |beta_{d,j} - beta_{d,k}|

subject to 1'beta_A = 0, 1'beta_B = 0. The intercept is unpenalized.

Never silently substitutes an unconstrained fit for a failed constrained
solve (task instruction, and item 13: "Never substitute an unconstrained
fit or convert failure to a favorable score"). A solve that exhausts the
retry policy raises SolverFailure; callers must not catch this to fall back
to something else without explicitly reporting that substitution -- and
this codebase never does that substitution anywhere.
"""
from __future__ import annotations


# Locate the repository independently of the script's directory depth.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_REPO_ROOT = next(p for p in _RepositoryPath(__file__).resolve().parents
                  if (p / "main.py").is_file() and (p / "configs").is_dir())
if str(_REPO_ROOT) not in _repository_sys.path:
    _repository_sys.path.insert(0, str(_REPO_ROOT))
_repository_sys.modules.setdefault("fusionML.estimator", _repository_sys.modules[__name__])
_repository_sys.modules.setdefault("models.fusionML.estimator", _repository_sys.modules[__name__])


import dataclasses
import time
from typing import List, Optional, Tuple

import numpy as np

try:
    import cvxpy as cp
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "cvxpy is required by fusionML.estimator (design-spec item 13 requires "
        "'CVXPY or another explicitly verified convex solver'). Install with "
        "`pip install cvxpy`."
    ) from e


class SolverFailure(RuntimeError):
    def __init__(self, message: str, attempts: List[dict]):
        super().__init__(message)
        self.attempts = attempts


@dataclasses.dataclass
class SolveAttempt:
    solver: str
    kwargs: dict
    status: str
    success: bool
    runtime_seconds: float
    equality_residual_a: Optional[float]
    equality_residual_b: Optional[float]
    error: Optional[str] = None


@dataclasses.dataclass
class FitResult:
    b0: float
    beta_a: np.ndarray
    beta_b: np.ndarray
    status: str
    solver_used: str
    solver_kwargs: dict
    runtime_seconds: float
    n_iterations: Optional[int]
    equality_residual_a: float
    equality_residual_b: float
    loss_value: float
    penalty_sparsity: float
    penalty_ridge: float
    penalty_fusion: float
    attempts: List[SolveAttempt]
    intercept_only: bool = False
    gamma: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    penalty_interaction: float = 0.0


def solver_environment_info() -> dict:
    return {
        "cvxpy_version": cp.__version__,
        "installed_solvers": cp.installed_solvers(),
    }


def _edge_arrays(edges: List[Tuple[int, int, float]]):
    if not edges:
        return None, None, None
    j = np.array([e[0] for e in edges], dtype=int)
    k = np.array([e[1] for e in edges], dtype=int)
    w = np.array([e[2] for e in edges], dtype=float)
    return j, k, w


def _build_penalties(beta_a, beta_b, lambda_s: float, lambda_r: float, lambda_f: float,
                      edges_a: List[Tuple[int, int, float]], edges_b: List[Tuple[int, int, float]]):
    pen_s = lambda_s * (cp.norm1(beta_a) + cp.norm1(beta_b))
    pen_r = lambda_r * (cp.sum_squares(beta_a) + cp.sum_squares(beta_b))
    ja, ka, wa = _edge_arrays(edges_a)
    jb, kb, wb = _edge_arrays(edges_b)
    fusion_terms = []
    if ja is not None:
        fusion_terms.append(cp.sum(cp.multiply(wa, cp.abs(beta_a[ja] - beta_a[ka]))))
    if jb is not None:
        fusion_terms.append(cp.sum(cp.multiply(wb, cp.abs(beta_b[jb] - beta_b[kb]))))
    pen_f = lambda_f * cp.sum(fusion_terms) if fusion_terms else cp.Constant(0.0)
    return pen_s, pen_r, pen_f, fusion_terms


def _solve_with_retries(problem: "cp.Problem", b0, beta_a, beta_b, loss, pen_s, pen_r, pen_f,
                         fusion_terms: list, lambda_s: float, lambda_f: float, solver_cfg: dict,
                         context_str: str, fallback: Optional[str] = None,
                         gamma=None, pen_i=None) -> FitResult:
    """gamma/pen_i are optional (the ARC-BAC interaction extension's
    coefficient vector and its L1+L2 penalty); when omitted this is exactly
    the item-13 additive estimator's solve loop. `fallback` overrides
    solver_cfg["fallback"] -- used for classification, whose logistic loss
    needs an exponential-cone-capable fallback (SCS), not OSQP, which
    cannot solve that formulation at all (item 13's later revision)."""
    tol = float(solver_cfg.get("equality_tolerance", 1e-6))
    preferred = solver_cfg.get("preferred", "CLARABEL")
    fallback = fallback if fallback is not None else solver_cfg.get("fallback", "OSQP")
    max_retries = int(solver_cfg.get("max_retries", 3))

    attempt_plan = [
        (preferred, {}),
        (preferred, {"tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "tol_feas": 1e-10} if preferred == "CLARABEL" else {}),
        (fallback, {}),
        (fallback, {"eps_abs": 1e-10, "eps_rel": 1e-10, "max_iter": 200000} if fallback == "OSQP" else {}),
    ][: max(1, max_retries + 1)]

    attempts: List[SolveAttempt] = []
    for solver_name, kwargs in attempt_plan:
        t0 = time.time()
        try:
            solve_kwargs = dict(kwargs)
            problem.solve(solver=getattr(cp, solver_name), **solve_kwargs)
            runtime = time.time() - t0
            status = problem.status
            ba = beta_a.value
            bb = beta_b.value
            b0v = b0.value
            gv = gamma.value if gamma is not None else np.zeros(0)
            finite = (
                ba is not None and bb is not None and b0v is not None
                and np.all(np.isfinite(ba)) and np.all(np.isfinite(bb)) and np.isfinite(b0v)
                and (gamma is None or (gv is not None and np.all(np.isfinite(gv))))
            )
            res_a = float(abs(np.sum(ba))) if ba is not None else float("inf")
            res_b = float(abs(np.sum(bb))) if bb is not None else float("inf")
            success = (status == cp.OPTIMAL) and finite and res_a <= tol and res_b <= tol
            attempts.append(SolveAttempt(
                solver=solver_name, kwargs=kwargs, status=status, success=success,
                runtime_seconds=runtime, equality_residual_a=res_a, equality_residual_b=res_b,
            ))
            if success:
                loss_val = float(loss.value)
                pen_s_val = float(pen_s.value) if lambda_s > 0 else 0.0
                pen_r_val = float(pen_r.value)
                pen_f_val = float(pen_f.value) if fusion_terms else 0.0
                pen_i_val = float(pen_i.value) if pen_i is not None else 0.0
                n_iter = getattr(problem.solver_stats, "num_iters", None) if problem.solver_stats else None
                return FitResult(
                    b0=float(b0v), beta_a=np.asarray(ba), beta_b=np.asarray(bb),
                    status=status, solver_used=solver_name, solver_kwargs=kwargs,
                    runtime_seconds=runtime, n_iterations=n_iter,
                    equality_residual_a=res_a, equality_residual_b=res_b,
                    loss_value=loss_val, penalty_sparsity=pen_s_val,
                    penalty_ridge=pen_r_val, penalty_fusion=pen_f_val,
                    attempts=attempts, gamma=np.asarray(gv), penalty_interaction=pen_i_val,
                )
        except Exception as e:  # solver may raise for unsupported kwargs/status
            runtime = time.time() - t0
            attempts.append(SolveAttempt(
                solver=solver_name, kwargs=kwargs, status="exception", success=False,
                runtime_seconds=runtime, equality_residual_a=None, equality_residual_b=None,
                error=str(e),
            ))

    raise SolverFailure(
        f"{context_str} solve failed after {len(attempts)} attempt(s). No unconstrained fallback was substituted.",
        attempts=[dataclasses.asdict(a) for a in attempts],
    )


def fit_log_contrast(
    L_a: np.ndarray, L_b: np.ndarray, y_star: np.ndarray, w: np.ndarray,
    lambda_s: float, lambda_r: float, lambda_f: float,
    edges_a: List[Tuple[int, int, float]], edges_b: List[Tuple[int, int, float]],
    solver_cfg: dict,
) -> FitResult:
    n, p_a = L_a.shape
    _, p_b = L_b.shape
    if p_a < 2 or p_b < 2:
        raise ValueError(f"fit_log_contrast requires >=2 taxa per domain, got p_a={p_a}, p_b={p_b}")

    b0 = cp.Variable()
    beta_a = cp.Variable(p_a)
    beta_b = cp.Variable(p_b)

    resid = y_star - b0 - L_a @ beta_a - L_b @ beta_b
    loss = cp.sum(cp.multiply(w, cp.square(resid)))
    pen_s, pen_r, pen_f, fusion_terms = _build_penalties(beta_a, beta_b, lambda_s, lambda_r, lambda_f, edges_a, edges_b)

    objective = cp.Minimize(loss + pen_s + pen_r + pen_f)
    constraints = [cp.sum(beta_a) == 0, cp.sum(beta_b) == 0]
    problem = cp.Problem(objective, constraints)

    context = (f"log-contrast regression (lambda_s={lambda_s}, lambda_r={lambda_r}, "
               f"lambda_f={lambda_f}, n={n}, p_a={p_a}, p_b={p_b})")
    return _solve_with_retries(problem, b0, beta_a, beta_b, loss, pen_s, pen_r, pen_f,
                                fusion_terms, lambda_s, lambda_f, solver_cfg, context)


def fit_log_contrast_logistic(
    L_a: np.ndarray, L_b: np.ndarray, y: np.ndarray, w: np.ndarray,
    lambda_s: float, lambda_r: float, lambda_f: float,
    edges_a: List[Tuple[int, int, float]], edges_b: List[Tuple[int, int, float]],
    solver_cfg: dict,
) -> FitResult:
    """Item 7's logistic-loss classification extension: same penalties and
    per-domain zero-sum constraints as fit_log_contrast, with squared-error
    loss replaced by weighted negative log-likelihood on raw {0,1} labels
    (no target standardization -- item 10's y-standardization step is a
    regression-only concern; logit-scale coefficients need no unscaling)."""
    n, p_a = L_a.shape
    _, p_b = L_b.shape
    if p_a < 2 or p_b < 2:
        raise ValueError(f"fit_log_contrast_logistic requires >=2 taxa per domain, got p_a={p_a}, p_b={p_b}")
    if not set(np.unique(y).tolist()) <= {0.0, 1.0}:
        raise ValueError(f"fit_log_contrast_logistic requires y in {{0,1}}, got values {np.unique(y)}")

    b0 = cp.Variable()
    beta_a = cp.Variable(p_a)
    beta_b = cp.Variable(p_b)

    eta = b0 + L_a @ beta_a + L_b @ beta_b
    loss = cp.sum(cp.multiply(w, cp.logistic(eta) - cp.multiply(y, eta)))
    pen_s, pen_r, pen_f, fusion_terms = _build_penalties(beta_a, beta_b, lambda_s, lambda_r, lambda_f, edges_a, edges_b)

    objective = cp.Minimize(loss + pen_s + pen_r + pen_f)
    constraints = [cp.sum(beta_a) == 0, cp.sum(beta_b) == 0]
    problem = cp.Problem(objective, constraints)

    context = (f"log-contrast logistic classification (lambda_s={lambda_s}, lambda_r={lambda_r}, "
               f"lambda_f={lambda_f}, n={n}, p_a={p_a}, p_b={p_b})")
    # Logistic loss needs cp.logistic's exponential-cone formulation; OSQP
    # cannot solve that at all, so classification gets its own fallback
    # (default SCS -- CLARABEL, the preferred solver, also supports it).
    fallback = solver_cfg.get("classification_fallback", "SCS")
    return _solve_with_retries(problem, b0, beta_a, beta_b, loss, pen_s, pen_r, pen_f,
                                fusion_terms, lambda_s, lambda_f, solver_cfg, context, fallback=fallback)


def fit_log_contrast_with_interaction(
    L_a: np.ndarray, L_b: np.ndarray, X_i: np.ndarray, y: np.ndarray, w: np.ndarray,
    lambda_s: float, lambda_r: float, lambda_f: float, lambda_i1: float, lambda_i2: float,
    edges_a: List[Tuple[int, int, float]], edges_b: List[Tuple[int, int, float]],
    task: str, solver_cfg: dict,
) -> FitResult:
    """ARC-BAC interaction extension (design-spec item 13 revision): jointly
    optimizes b0, beta_A, beta_B and gamma with
    eta = b0 + L_A@beta_A + L_B@beta_B + X_I@gamma. No zero-sum constraint
    on gamma (every product column is already built from valid within-
    domain balances); no cross-domain fusion edges. lambda_i2 must be
    strictly positive for an interaction-ON candidate (the caller is
    responsible for that; an interaction-OFF candidate should call
    fit_log_contrast/fit_log_contrast_logistic directly instead of passing
    an empty X_i here, per the "reuse exact matching additive fits"
    instruction)."""
    n, p_a = L_a.shape
    _, p_b = L_b.shape
    n_i = X_i.shape[1]
    if p_a < 2 or p_b < 2:
        raise ValueError(f"fit_log_contrast_with_interaction requires >=2 taxa per domain, got p_a={p_a}, p_b={p_b}")
    if n_i == 0:
        raise ValueError("fit_log_contrast_with_interaction requires at least one interaction column; "
                          "use the additive estimator directly for an interaction-off candidate")
    if task not in ("regression", "classification"):
        raise ValueError(f"unknown task {task!r}")

    b0 = cp.Variable()
    beta_a = cp.Variable(p_a)
    beta_b = cp.Variable(p_b)
    gamma = cp.Variable(n_i)

    eta = b0 + L_a @ beta_a + L_b @ beta_b + X_i @ gamma
    if task == "classification":
        if not set(np.unique(y).tolist()) <= {0.0, 1.0}:
            raise ValueError(f"classification requires y in {{0,1}}, got values {np.unique(y)}")
        loss = cp.sum(cp.multiply(w, cp.logistic(eta) - cp.multiply(y, eta)))
    else:
        loss = cp.sum(cp.multiply(w, cp.square(y - eta)))

    pen_s, pen_r, pen_f, fusion_terms = _build_penalties(beta_a, beta_b, lambda_s, lambda_r, lambda_f, edges_a, edges_b)
    pen_i = lambda_i1 * cp.norm1(gamma) + lambda_i2 * cp.sum_squares(gamma)

    objective = cp.Minimize(loss + pen_s + pen_r + pen_f + pen_i)
    constraints = [cp.sum(beta_a) == 0, cp.sum(beta_b) == 0]
    problem = cp.Problem(objective, constraints)

    context = (f"log-contrast+interaction {task} (lambda_s={lambda_s}, lambda_r={lambda_r}, lambda_f={lambda_f}, "
               f"lambda_i1={lambda_i1}, lambda_i2={lambda_i2}, n={n}, p_a={p_a}, p_b={p_b}, n_products={n_i})")
    fallback = solver_cfg.get("classification_fallback", "SCS") if task == "classification" else None
    return _solve_with_retries(problem, b0, beta_a, beta_b, loss, pen_s, pen_r, pen_f,
                                fusion_terms, lambda_s, lambda_f, solver_cfg, context,
                                fallback=fallback, gamma=gamma, pen_i=pen_i)


def intercept_only_result(y_star: np.ndarray, w: np.ndarray, p_a: int, p_b: int) -> FitResult:
    """Used when the training target has zero site-balanced variance
    (item 10: 'If y has zero variance, use an intercept-only fit')."""
    b0 = float(np.sum(w * y_star))
    return FitResult(
        b0=b0, beta_a=np.zeros(p_a), beta_b=np.zeros(p_b), status="intercept_only",
        solver_used="none", solver_kwargs={}, runtime_seconds=0.0, n_iterations=0,
        equality_residual_a=0.0, equality_residual_b=0.0, loss_value=float(np.sum(w * (y_star - b0) ** 2)),
        penalty_sparsity=0.0, penalty_ridge=0.0, penalty_fusion=0.0, attempts=[], intercept_only=True,
    )


def intercept_only_result_logistic(y: np.ndarray, w: np.ndarray, p_a: int, p_b: int) -> FitResult:
    """Classification analogue of intercept_only_result: used when the
    training labels are a single class (site-balanced prevalence is 0 or
    1). b0 is the site-balanced log-odds of the (clipped) prevalence."""
    prevalence = float(np.sum(w * y))
    p_clipped = min(max(prevalence, 1e-6), 1 - 1e-6)
    b0 = float(np.log(p_clipped / (1 - p_clipped)))
    loss_val = float(np.sum(w * (np.log1p(np.exp(-abs(b0))) + max(b0, 0) - y * b0)))
    return FitResult(
        b0=b0, beta_a=np.zeros(p_a), beta_b=np.zeros(p_b), status="intercept_only",
        solver_used="none", solver_kwargs={}, runtime_seconds=0.0, n_iterations=0,
        equality_residual_a=0.0, equality_residual_b=0.0, loss_value=loss_val,
        penalty_sparsity=0.0, penalty_ridge=0.0, penalty_fusion=0.0, attempts=[], intercept_only=True,
    )


def _solve_single_domain_with_retries(problem: "cp.Problem", b0, beta, loss, pen_s, pen_r, pen_f,
                                       fusion_terms: list, lambda_s: float, lambda_f: float, solver_cfg: dict,
                                       context_str: str, fallback: Optional[str] = None) -> FitResult:
    """Single-domain sibling of _solve_with_retries (ARC-only / BAC-only
    ablation of models 2-5, item 14's "domain-specific baseline"): one beta,
    one zero-sum constraint. Reuses FitResult with beta_b left empty (shape
    (0,)) so downstream code (predict/count_active) that only reads
    .beta_a needs no separate single-domain result type."""
    tol = float(solver_cfg.get("equality_tolerance", 1e-6))
    preferred = solver_cfg.get("preferred", "CLARABEL")
    fallback = fallback if fallback is not None else solver_cfg.get("fallback", "OSQP")
    max_retries = int(solver_cfg.get("max_retries", 3))

    attempt_plan = [
        (preferred, {}),
        (preferred, {"tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "tol_feas": 1e-10} if preferred == "CLARABEL" else {}),
        (fallback, {}),
        (fallback, {"eps_abs": 1e-10, "eps_rel": 1e-10, "max_iter": 200000} if fallback == "OSQP" else {}),
    ][: max(1, max_retries + 1)]

    attempts: List[SolveAttempt] = []
    for solver_name, kwargs in attempt_plan:
        t0 = time.time()
        try:
            solve_kwargs = dict(kwargs)
            problem.solve(solver=getattr(cp, solver_name), **solve_kwargs)
            runtime = time.time() - t0
            status = problem.status
            bv = beta.value
            b0v = b0.value
            finite = bv is not None and b0v is not None and np.all(np.isfinite(bv)) and np.isfinite(b0v)
            res = float(abs(np.sum(bv))) if bv is not None else float("inf")
            success = (status == cp.OPTIMAL) and finite and res <= tol
            attempts.append(SolveAttempt(
                solver=solver_name, kwargs=kwargs, status=status, success=success,
                runtime_seconds=runtime, equality_residual_a=res, equality_residual_b=0.0,
            ))
            if success:
                loss_val = float(loss.value)
                pen_s_val = float(pen_s.value) if lambda_s > 0 else 0.0
                pen_r_val = float(pen_r.value)
                pen_f_val = float(pen_f.value) if fusion_terms else 0.0
                n_iter = getattr(problem.solver_stats, "num_iters", None) if problem.solver_stats else None
                return FitResult(
                    b0=float(b0v), beta_a=np.asarray(bv), beta_b=np.zeros(0),
                    status=status, solver_used=solver_name, solver_kwargs=kwargs,
                    runtime_seconds=runtime, n_iterations=n_iter,
                    equality_residual_a=res, equality_residual_b=0.0,
                    loss_value=loss_val, penalty_sparsity=pen_s_val,
                    penalty_ridge=pen_r_val, penalty_fusion=pen_f_val,
                    attempts=attempts,
                )
        except Exception as e:
            runtime = time.time() - t0
            attempts.append(SolveAttempt(
                solver=solver_name, kwargs=kwargs, status="exception", success=False,
                runtime_seconds=runtime, equality_residual_a=None, equality_residual_b=None,
                error=str(e),
            ))

    raise SolverFailure(
        f"{context_str} solve failed after {len(attempts)} attempt(s). No unconstrained fallback was substituted.",
        attempts=[dataclasses.asdict(a) for a in attempts],
    )


def fit_log_contrast_single_domain(
    L: np.ndarray, y: np.ndarray, w: np.ndarray,
    lambda_s: float, lambda_r: float, lambda_f: float,
    edges: List[Tuple[int, int, float]], task: str, solver_cfg: dict,
) -> FitResult:
    """Single-domain log-contrast fit (ARC-only or BAC-only), for the
    separately-requested ablation of models 2-5 (item 14 table): how much
    predictive power comes from one domain alone. Same objective/constraint
    form as fit_log_contrast(_logistic) restricted to one domain's design
    matrix and its own within-domain fusion graph (e.g. a single-domain
    phylogeny_only/abundance_only variant); never a claim that this
    replaces the joint (cross-domain) model."""
    n, p = L.shape
    if p < 2:
        raise ValueError(f"fit_log_contrast_single_domain requires >=2 taxa, got p={p}")
    if task not in ("regression", "classification"):
        raise ValueError(f"unknown task {task!r}")

    b0 = cp.Variable()
    beta = cp.Variable(p)
    eta = b0 + L @ beta
    if task == "classification":
        if not set(np.unique(y).tolist()) <= {0.0, 1.0}:
            raise ValueError(f"classification requires y in {{0,1}}, got values {np.unique(y)}")
        loss = cp.sum(cp.multiply(w, cp.logistic(eta) - cp.multiply(y, eta)))
    else:
        loss = cp.sum(cp.multiply(w, cp.square(y - eta)))

    pen_s = lambda_s * cp.norm1(beta)
    pen_r = lambda_r * cp.sum_squares(beta)
    j, k, wgt = _edge_arrays(edges)
    fusion_terms = []
    if j is not None:
        fusion_terms.append(cp.sum(cp.multiply(wgt, cp.abs(beta[j] - beta[k]))))
    pen_f = lambda_f * cp.sum(fusion_terms) if fusion_terms else cp.Constant(0.0)

    objective = cp.Minimize(loss + pen_s + pen_r + pen_f)
    problem = cp.Problem(objective, [cp.sum(beta) == 0])

    context = (f"single-domain log-contrast {task} (lambda_s={lambda_s}, lambda_r={lambda_r}, "
               f"lambda_f={lambda_f}, n={n}, p={p})")
    fallback = solver_cfg.get("classification_fallback", "SCS") if task == "classification" else None
    return _solve_single_domain_with_retries(problem, b0, beta, loss, pen_s, pen_r, pen_f,
                                              fusion_terms, lambda_s, lambda_f, solver_cfg, context,
                                              fallback=fallback)


def fit_domain_only_baseline(L_d: np.ndarray, y_star: np.ndarray, w: np.ndarray,
                              lambda_s: float, lambda_r: float, solver_cfg: dict) -> Tuple[float, np.ndarray, dict]:
    """A single-domain log-contrast fit (no fusion), for the explicit,
    separately-requested "domain-specific baseline" item 10 mentions when a
    domain has <2 retained taxa and the joint fit cannot be built. Not
    wired into the default `train` CLI flow; callable directly."""
    n, p = L_d.shape
    if p < 2:
        raise ValueError(f"fit_domain_only_baseline requires >=2 taxa, got p={p}")
    b0 = cp.Variable()
    beta = cp.Variable(p)
    resid = y_star - b0 - L_d @ beta
    loss = cp.sum(cp.multiply(w, cp.square(resid)))
    objective = cp.Minimize(loss + lambda_s * cp.norm1(beta) + lambda_r * cp.sum_squares(beta))
    problem = cp.Problem(objective, [cp.sum(beta) == 0])
    problem.solve(solver=getattr(cp, solver_cfg.get("preferred", "CLARABEL")))
    if problem.status != cp.OPTIMAL:
        problem.solve(solver=getattr(cp, solver_cfg.get("fallback", "OSQP")))
    if problem.status != cp.OPTIMAL:
        raise SolverFailure(f"domain-only baseline solve failed, status={problem.status}", attempts=[])
    return float(b0.value), np.asarray(beta.value), {"status": problem.status}
