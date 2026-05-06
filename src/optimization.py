import time

import numpy as np
import pandas as pd

import config
from utils import factorize_cov, normalize_weights, portfolio_objective, portfolio_return, portfolio_variance


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


def solve_qp_utility(mu, Sigma, gamma, solver):
    import cvxpy as cp

    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(mu)
    x = cp.Variable(n)
    risk = cp.quad_form(x, cp.psd_wrap(Sigma))
    ret = mu @ x
    problem = cp.Problem(
        cp.Minimize(0.5 * gamma * risk - ret),
        [cp.sum(x) == 1, x >= 0],
    )

    tic = time.perf_counter()
    problem.solve(solver=solver, verbose=False)
    elapsed = time.perf_counter() - tic

    if x.value is None:
        raise RuntimeError(f"QP solve failed with status {problem.status}")
    w = normalize_weights(x.value)
    return {
        "gamma": gamma,
        "weights": w,
        "expected_return": portfolio_return(mu, w),
        "variance": portfolio_variance(Sigma, w),
        "std": np.sqrt(max(portfolio_variance(Sigma, w), 0.0)),
        "objective": portfolio_objective(mu, Sigma, w, gamma),
        "status": problem.status,
        "runtime_sec": elapsed,
    }


def solve_socp(mu, Sigma, sigma_max, solver):
    import cvxpy as cp

    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(mu)
    B = factorize_cov(Sigma)
    x = cp.Variable(n)
    problem = cp.Problem(
        cp.Maximize(mu @ x),
        [cp.norm(B.T @ x, 2) <= sigma_max, cp.sum(x) == 1, x >= 0],
    )

    tic = time.perf_counter()
    problem.solve(solver=solver, verbose=False)
    elapsed = time.perf_counter() - tic

    if x.value is None:
        raise RuntimeError(f"SOCP solve failed with status {problem.status}")
    w = normalize_weights(x.value)
    return {
        "sigma_max": sigma_max,
        "weights": w,
        "expected_return": portfolio_return(mu, w),
        "variance": portfolio_variance(Sigma, w),
        "std": np.sqrt(max(portfolio_variance(Sigma, w), 0.0)),
        "objective": -portfolio_return(mu, w),
        "status": problem.status,
        "runtime_sec": elapsed,
    }


def run_qp_frontier(mu, Sigma, solver, gammas=None):
    gammas = config.FRONTIER_GAMMAS if gammas is None else gammas
    rows, weights = [], []
    tickers = list(mu.index) if hasattr(mu, "index") else [f"asset_{i}" for i in range(len(mu))]
    for gamma in gammas:
        out = solve_qp_utility(mu, Sigma, gamma, solver)
        rows.append({k: v for k, v in out.items() if k != "weights"})
        weights.append(pd.Series(out["weights"], index=tickers, name=f"gamma={gamma:g}"))
    frontier = pd.DataFrame(rows)
    weight_table = pd.DataFrame(weights)
    frontier.to_csv(config.CSV_DIR / "qp_frontier.csv", index=False)
    weight_table.to_csv(config.CSV_DIR / "qp_frontier_weights.csv")
    return frontier, weight_table


def run_socp_frontier(mu, Sigma, solver, qp_frontier=None):
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
            out = solve_socp(mu, Sigma, sigma_max, solver)
            rows.append({k: v for k, v in out.items() if k != "weights"})
            weights.append(pd.Series(out["weights"], index=tickers, name=f"sigma={sigma_max:.6g}"))
        except RuntimeError:
            continue
    frontier = pd.DataFrame(rows)
    weight_table = pd.DataFrame(weights)
    frontier.to_csv(config.CSV_DIR / "socp_frontier.csv", index=False)
    weight_table.to_csv(config.CSV_DIR / "socp_frontier_weights.csv")
    return frontier, weight_table


def evaluate_static_weights(weights, returns):
    from utils import performance_metrics

    daily = returns @ weights
    return performance_metrics(daily)


def select_hyperparameters(train, validation, mu, Sigma, solver):
    gamma_rows = []
    for gamma in config.GAMMA_GRID:
        out = solve_qp_utility(mu, Sigma, gamma, solver)
        metrics = evaluate_static_weights(out["weights"], validation)
        gamma_rows.append({"gamma": gamma, **metrics})
    gamma_df = pd.DataFrame(gamma_rows)
    best_gamma = float(gamma_df.sort_values("sharpe", ascending=False).iloc[0]["gamma"])

    equal = np.ones(len(mu)) / len(mu)
    base_sigma = np.sqrt(portfolio_variance(Sigma, equal))
    socp_rows = []
    for multiplier in config.SOCP_MULTIPLIERS:
        sigma_max = multiplier * base_sigma
        try:
            out = solve_socp(mu, Sigma, sigma_max, solver)
            metrics = evaluate_static_weights(out["weights"], validation)
            socp_rows.append({"sigma_multiplier": multiplier, "sigma_max": sigma_max, **metrics})
        except RuntimeError:
            continue
    socp_df = pd.DataFrame(socp_rows)
    best_multiplier = float(socp_df.sort_values("sharpe", ascending=False).iloc[0]["sigma_multiplier"])

    gamma_df.to_csv(config.CSV_DIR / "gamma_validation_sensitivity.csv", index=False)
    socp_df.to_csv(config.CSV_DIR / "socp_validation_sensitivity.csv", index=False)
    selected = pd.DataFrame({
        "parameter": ["gamma", "sigma_multiplier"],
        "selected_value": [best_gamma, best_multiplier],
    })
    selected.to_csv(config.CSV_DIR / "selected_hyperparameters.csv", index=False)
    return best_gamma, best_multiplier, gamma_df, socp_df
