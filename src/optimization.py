import time

import numpy as np
import pandas as pd

import config
from utils import factorize_cov, normalize_weights, portfolio_objective, portfolio_return, portfolio_variance


def build_weight_constraints(x, max_weight=None):
    import cvxpy as cp

    constraints = [cp.sum(x) == 1, x >= 0]
    if max_weight is not None:
        constraints.append(x <= max_weight)
    return constraints


def get_solver(allow_fallback=False):
    import cvxpy as cp

    installed = set(cp.installed_solvers())
    for name in ["MOSEK", "GUROBI"]:
        if name in installed:
            return name
    if allow_fallback:
        for name in ["CLARABEL", "SCS", "ECOS", "OSQP"]:
            if name in installed:
                return name
    raise RuntimeError(
        "No required commercial solver is installed. Install MOSEK or Gurobi and make it visible to CVXPY."
    )


def solve_qp_utility(mu, Sigma, gamma, solver, max_weight=None):
    import cvxpy as cp

    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(mu)
    x = cp.Variable(n)
    risk = cp.quad_form(x, cp.psd_wrap(Sigma))
    ret = mu @ x
    problem = cp.Problem(
        cp.Minimize(0.5 * gamma * risk - ret),
        build_weight_constraints(x, max_weight=max_weight),
    )

    tic = time.perf_counter()
    problem.solve(solver=solver, verbose=False)
    elapsed = time.perf_counter() - tic

    if x.value is None:
        raise RuntimeError(f"QP solve failed with status {problem.status}")
    w = normalize_weights(x.value)
    return {
        "gamma": gamma,
        "max_weight": max_weight,
        "weights": w,
        "expected_return": portfolio_return(mu, w),
        "variance": portfolio_variance(Sigma, w),
        "std": np.sqrt(max(portfolio_variance(Sigma, w), 0.0)),
        "objective": portfolio_objective(mu, Sigma, w, gamma),
        "status": problem.status,
        "runtime_sec": elapsed,
    }


def solve_qp_with_linear_cost(mu, Sigma, gamma, reference_weights, cost_rate, solver):
    import cvxpy as cp

    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    reference_weights = np.asarray(reference_weights, dtype=float)
    n = len(mu)
    if reference_weights.shape != (n,):
        raise ValueError("reference_weights must have the same length as mu")

    x = cp.Variable(n)
    trade = cp.Variable(n, nonneg=True)
    risk = cp.quad_form(x, cp.psd_wrap(Sigma))
    ret = mu @ x
    constraints = build_weight_constraints(x)
    constraints.extend([
        trade >= x - reference_weights,
        trade >= reference_weights - x,
    ])
    problem = cp.Problem(
        cp.Minimize(0.5 * gamma * risk - ret + cost_rate * cp.sum(trade)),
        constraints,
    )

    tic = time.perf_counter()
    problem.solve(solver=solver, verbose=False)
    elapsed = time.perf_counter() - tic

    if x.value is None:
        raise RuntimeError(f"QP with linear cost solve failed with status {problem.status}")
    w = normalize_weights(x.value)
    turnover = float(np.sum(np.abs(w - reference_weights)))
    return {
        "gamma": gamma,
        "modeled_cost_rate": cost_rate,
        "weights": w,
        "expected_return": portfolio_return(mu, w),
        "variance": portfolio_variance(Sigma, w),
        "std": np.sqrt(max(portfolio_variance(Sigma, w), 0.0)),
        "objective": portfolio_objective(mu, Sigma, w, gamma) + cost_rate * turnover,
        "turnover_to_reference": turnover,
        "status": problem.status,
        "runtime_sec": elapsed,
    }


def solve_socp(mu, Sigma, sigma_max, solver, max_weight=None):
    import cvxpy as cp

    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(mu)
    B = factorize_cov(Sigma)
    x = cp.Variable(n)
    problem = cp.Problem(
        cp.Maximize(mu @ x),
        [cp.norm(B.T @ x, 2) <= sigma_max, *build_weight_constraints(x, max_weight=max_weight)],
    )

    tic = time.perf_counter()
    problem.solve(solver=solver, verbose=False)
    elapsed = time.perf_counter() - tic

    if x.value is None:
        raise RuntimeError(f"SOCP solve failed with status {problem.status}")
    w = normalize_weights(x.value)
    return {
        "sigma_max": sigma_max,
        "max_weight": max_weight,
        "weights": w,
        "expected_return": portfolio_return(mu, w),
        "variance": portfolio_variance(Sigma, w),
        "std": np.sqrt(max(portfolio_variance(Sigma, w), 0.0)),
        "objective": -portfolio_return(mu, w),
        "status": problem.status,
        "runtime_sec": elapsed,
    }


def run_qp_frontier(mu, Sigma, solver, gammas=None, max_weight=None, file_stem="qp_frontier"):
    gammas = config.FRONTIER_GAMMAS if gammas is None else gammas
    rows, weights = [], []
    tickers = list(mu.index) if hasattr(mu, "index") else [f"asset_{i}" for i in range(len(mu))]
    for gamma in gammas:
        out = solve_qp_utility(mu, Sigma, gamma, solver, max_weight=max_weight)
        rows.append({k: v for k, v in out.items() if k != "weights"})
        weights.append(pd.Series(out["weights"], index=tickers, name=f"gamma={gamma:g}"))
    frontier = pd.DataFrame(rows)
    weight_table = pd.DataFrame(weights)
    return frontier, weight_table


def run_qp_linear_cost_frontier(mu, Sigma, solver, reference_weights, cost_rate, gammas=None, file_stem="qp_linear_cost_frontier"):
    gammas = config.FRONTIER_GAMMAS if gammas is None else gammas
    rows, weights = [], []
    tickers = list(mu.index) if hasattr(mu, "index") else [f"asset_{i}" for i in range(len(mu))]
    reference_weights = np.asarray(reference_weights, dtype=float)
    for gamma in gammas:
        out = solve_qp_with_linear_cost(mu, Sigma, gamma, reference_weights, cost_rate, solver)
        rows.append({k: v for k, v in out.items() if k != "weights"})
        weights.append(pd.Series(out["weights"], index=tickers, name=f"gamma={gamma:g}"))
    frontier = pd.DataFrame(rows)
    weight_table = pd.DataFrame(weights)
    return frontier, weight_table


def run_socp_frontier(mu, Sigma, solver, qp_frontier=None, max_weight=None, file_stem="socp_frontier"):
    if qp_frontier is None or qp_frontier.empty:
        equal = np.ones(len(mu)) / len(mu)
        base_std = np.sqrt(portfolio_variance(Sigma, equal))
        sigma_grid = np.linspace(0.75 * base_std, 1.5 * base_std, config.SOCP_FRONTIER_POINTS)
    else:
        low = max(qp_frontier["std"].min() * 1.01, 1e-8)
        high = qp_frontier["std"].max()
        sigma_grid = np.linspace(low, high, config.SOCP_FRONTIER_POINTS)

    rows, weights = [], []
    tickers = list(mu.index) if hasattr(mu, "index") else [f"asset_{i}" for i in range(len(mu))]
    for sigma_max in sigma_grid:
        try:
            out = solve_socp(mu, Sigma, sigma_max, solver, max_weight=max_weight)
            rows.append({k: v for k, v in out.items() if k != "weights"})
            weights.append(pd.Series(out["weights"], index=tickers, name=f"sigma={sigma_max:.6g}"))
        except RuntimeError:
            continue
    frontier = pd.DataFrame(rows)
    weight_table = pd.DataFrame(weights)
    return frontier, weight_table


def evaluate_static_weights(weights, returns):
    from utils import performance_metrics

    daily = returns @ weights
    return performance_metrics(daily)


def summarize_portfolio_point(model, out):
    weights = np.asarray(out["weights"], dtype=float)
    active = int(np.sum(weights > config.ACTIVE_WEIGHT_THRESHOLD))
    gamma = out.get("gamma", np.nan)
    sigma_max = out.get("sigma_max", np.nan)
    return {
        "model": model,
        "gamma": gamma,
        "sigma_max": sigma_max,
        "annual_return": 252 * out["expected_return"],
        "annual_std": np.sqrt(252) * out["std"],
        "max_weight": float(np.max(weights)),
        "active_positions": active,
    }


def _weight_diagnostics(out, base_sigma):
    """Return compact in-sample diagnostics for a solved portfolio."""
    weights = np.asarray(out["weights"], dtype=float)
    std = float(out["std"])
    return {
        "train_expected_return": float(out["expected_return"]),
        "train_std": std,
        "implied_sigma_multiplier": std / base_sigma if base_sigma > 0 else np.nan,
        "max_weight": float(np.max(weights)),
        "active_positions": int(np.sum(weights > config.ACTIVE_WEIGHT_THRESHOLD)),
    }


def _best_parameter(df, parameter_col):
    valid = df.dropna(subset=["sharpe"])
    if valid.empty:
        return np.nan
    return float(valid.sort_values("sharpe", ascending=False).iloc[0][parameter_col])


def _run_unmatched_socp_grid(mu, Sigma, validation, solver, base_sigma, max_weight=None):
    """Legacy SOCP sensitivity over fixed multiples of equal-weight volatility."""
    rows = []
    for multiplier in config.SOCP_MULTIPLIERS:
        sigma_max = multiplier * base_sigma
        try:
            out = solve_socp(mu, Sigma, sigma_max, solver, max_weight=max_weight)
            metrics = evaluate_static_weights(out["weights"], validation)
            row = {
                "sigma_multiplier": multiplier,
                "sigma_max": sigma_max,
                "grid_type": "fixed_equal_weight_multiplier",
                **_weight_diagnostics(out, base_sigma),
                **metrics,
            }
            rows.append(row)
        except RuntimeError:
            continue
    return pd.DataFrame(rows)


def _run_socp_matched_to_qp(mu, Sigma, validation, solver, base_sigma, qp_outputs, qp_sensitivity, max_weight=None):
    """Evaluate SOCP on risk budgets induced by QP solutions.

    For a QP solution w_gamma, use sigma_max = sqrt(w_gamma' Sigma w_gamma). This makes
    the SOCP sensitivity comparable with the QP gamma sensitivity. The small slack avoids
    numerical infeasibility caused by solver tolerances.
    """
    rows = []
    qp_sensitivity = qp_sensitivity.set_index("gamma")
    for gamma, qp_out in qp_outputs.items():
        target_qp_std = float(qp_out["std"])
        sigma_max = max(target_qp_std * (1.0 + 1e-8), 1e-12)
        try:
            out = solve_socp(mu, Sigma, sigma_max, solver, max_weight=max_weight)
        except RuntimeError:
            continue

        metrics = evaluate_static_weights(out["weights"], validation)
        qp_metrics = qp_sensitivity.loc[gamma]
        rows.append({
            "gamma": gamma,
            "sigma_multiplier": sigma_max / base_sigma if base_sigma > 0 else np.nan,
            "sigma_max": sigma_max,
            "target_qp_std": target_qp_std,
            "grid_type": "matched_to_qp_risk",
            "qp_validation_sharpe": float(qp_metrics["sharpe"]),
            "sharpe_gap_vs_qp": float(metrics["sharpe"] - qp_metrics["sharpe"]),
            "train_return_gap_vs_qp": float(out["expected_return"] - qp_out["expected_return"]),
            "train_std_gap_vs_qp": float(out["std"] - qp_out["std"]),
            "weight_l1_distance_to_qp": float(np.sum(np.abs(out["weights"] - qp_out["weights"]))),
            **_weight_diagnostics(out, base_sigma),
            **metrics,
        })
    return pd.DataFrame(rows)


def select_hyperparameters(train, validation, mu, Sigma, solver, box_max_weight=None):
    # The QP gamma grid and the SOCP multiplier grid are not directly comparable.
    # We therefore select SOCP risk budgets on the risk levels induced by QP solutions.
    equal = np.ones(len(mu)) / len(mu)
    base_sigma = np.sqrt(portfolio_variance(Sigma, equal))

    gamma_rows = []
    qp_outputs = {}
    for gamma in config.GAMMA_GRID:
        out = solve_qp_utility(mu, Sigma, gamma, solver)
        qp_outputs[gamma] = out
        metrics = evaluate_static_weights(out["weights"], validation)
        gamma_rows.append({"gamma": gamma, **_weight_diagnostics(out, base_sigma), **metrics})
    gamma_df = pd.DataFrame(gamma_rows)
    best_gamma = _best_parameter(gamma_df, "gamma")

    box_gamma_df = pd.DataFrame()
    box_qp_outputs = {}
    best_box_gamma = np.nan
    if box_max_weight is not None:
        box_gamma_rows = []
        for gamma in config.GAMMA_GRID:
            out = solve_qp_utility(mu, Sigma, gamma, solver, max_weight=box_max_weight)
            box_qp_outputs[gamma] = out
            metrics = evaluate_static_weights(out["weights"], validation)
            box_gamma_rows.append({"gamma": gamma, **_weight_diagnostics(out, base_sigma), **metrics})
        box_gamma_df = pd.DataFrame(box_gamma_rows)
        best_box_gamma = _best_parameter(box_gamma_df, "gamma")

    # Main SOCP sensitivity: matched to the QP risk level for each gamma.
    socp_df = _run_socp_matched_to_qp(mu, Sigma, validation, solver, base_sigma, qp_outputs, gamma_df)
    best_multiplier = _best_parameter(socp_df, "sigma_multiplier")

    box_socp_df = pd.DataFrame()
    best_box_multiplier = np.nan
    if box_max_weight is not None and box_qp_outputs:
        box_socp_df = _run_socp_matched_to_qp(
            mu, Sigma, validation, solver, base_sigma, box_qp_outputs, box_gamma_df, max_weight=box_max_weight
        )
        best_box_multiplier = _best_parameter(box_socp_df, "sigma_multiplier")

    return best_gamma, best_box_gamma, best_multiplier, best_box_multiplier, gamma_df, socp_df, box_gamma_df, box_socp_df

