from pathlib import Path
import time

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRADING_DAYS = 252
PREFERRED_SOLVERS = ["MOSEK", "GUROBI"]
ALLOW_NONCOMMERCIAL_FALLBACK = False

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def installed_solver():
    installed = cp.installed_solvers()
    for solver in PREFERRED_SOLVERS:
        if solver in installed:
            return solver
    if ALLOW_NONCOMMERCIAL_FALLBACK:
        for solver in ["CLARABEL", "ECOS", "SCS"]:
            if solver in installed:
                return solver
    raise RuntimeError(
        "No required commercial solver is installed. Please install MOSEK or Gurobi "
        "and make sure it is visible to CVXPY."
    )


def load_inputs():
    mu = pd.read_csv(DATA_DIR / "mu_daily_train.csv", index_col=0).iloc[:, 0]
    sigma = pd.read_csv(DATA_DIR / "Sigma_daily_train_stable.csv", index_col=0)
    sigma = sigma.loc[mu.index, mu.index]
    return mu, sigma


def portfolio_metrics(x, mu, sigma):
    ret_daily = float(mu.to_numpy() @ x)
    var_daily = float(x @ sigma.to_numpy() @ x)
    return {
        "return_daily": ret_daily,
        "variance_daily": var_daily,
        "std_daily": np.sqrt(max(var_daily, 0.0)),
        "return_annual": TRADING_DAYS * ret_daily,
        "std_annual": np.sqrt(TRADING_DAYS * max(var_daily, 0.0)),
    }


def solve_qp_utility(mu, sigma, gamma, solver, upper_bound=None):
    n = len(mu)
    x = cp.Variable(n)
    constraints = [cp.sum(x) == 1, x >= 0]
    if upper_bound is not None:
        constraints.append(x <= upper_bound)

    obj = cp.Minimize(0.5 * gamma * cp.quad_form(x, sigma.to_numpy()) - mu.to_numpy() @ x)
    prob = cp.Problem(obj, constraints)

    t0 = time.perf_counter()
    prob.solve(solver=solver, verbose=False)
    elapsed = time.perf_counter() - t0

    return prob.status, prob.value, elapsed, np.asarray(x.value).reshape(-1)


def solve_min_variance(mu, sigma, solver):
    n = len(mu)
    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(x, sigma.to_numpy())),
        [cp.sum(x) == 1, x >= 0],
    )
    prob.solve(solver=solver, verbose=False)
    return np.asarray(x.value).reshape(-1)


def solve_socp(mu, sigma, sigma_max, solver):
    n = len(mu)
    x = cp.Variable(n)
    chol = np.linalg.cholesky(sigma.to_numpy())
    constraints = [cp.norm(chol.T @ x, 2) <= sigma_max, cp.sum(x) == 1, x >= 0]
    prob = cp.Problem(cp.Maximize(mu.to_numpy() @ x), constraints)

    t0 = time.perf_counter()
    prob.solve(solver=solver, verbose=False)
    elapsed = time.perf_counter() - t0

    return prob.status, prob.value, elapsed, np.asarray(x.value).reshape(-1)


def run_qp_frontier(mu, sigma, solver):
    gammas = np.logspace(0, 5, 25)
    rows, weights = [], []

    for gamma in gammas:
        status, obj, elapsed, x = solve_qp_utility(mu, sigma, gamma, solver)
        metrics = portfolio_metrics(x, mu, sigma)
        rows.append({"gamma": gamma, "status": status, "objective": obj, "time_sec": elapsed, **metrics})
        weights.append(pd.Series(x, index=mu.index, name=f"gamma={gamma:.6g}"))

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "qp_frontier_metrics.csv", index=False)
    pd.DataFrame(weights).to_csv(RESULTS_DIR / "qp_frontier_weights.csv")
    return pd.DataFrame(rows)


def run_socp_frontier(mu, sigma, solver):
    x_minvar = solve_min_variance(mu, sigma, solver)
    risk_min = portfolio_metrics(x_minvar, mu, sigma)["std_daily"]
    risk_individual_max = np.sqrt(np.diag(sigma.to_numpy())).max()
    sigma_values = np.linspace(1.02 * risk_min, risk_individual_max, 25)

    rows, weights = [], []
    for sigma_max in sigma_values:
        status, obj, elapsed, x = solve_socp(mu, sigma, sigma_max, solver)
        metrics = portfolio_metrics(x, mu, sigma)
        rows.append({"sigma_max_daily": sigma_max, "status": status, "objective": obj, "time_sec": elapsed, **metrics})
        weights.append(pd.Series(x, index=mu.index, name=f"sigma={sigma_max:.6g}"))

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "socp_frontier_metrics.csv", index=False)
    pd.DataFrame(weights).to_csv(RESULTS_DIR / "socp_frontier_weights.csv")
    return pd.DataFrame(rows)


def run_box_constraint_example(mu, sigma, solver):
    gamma = 1000.0
    upper_bound = 0.15
    status, obj, elapsed, x = solve_qp_utility(mu, sigma, gamma, solver, upper_bound=upper_bound)
    row = {"gamma": gamma, "upper_bound": upper_bound, "status": status, "objective": obj, "time_sec": elapsed}
    row.update(portfolio_metrics(x, mu, sigma))
    pd.DataFrame([row]).to_csv(RESULTS_DIR / "qp_box_constraint_metrics.csv", index=False)
    pd.Series(x, index=mu.index, name="weight").to_csv(RESULTS_DIR / "qp_box_constraint_weights.csv")


def make_plots(qp_df, socp_df):
    plt.figure(figsize=(7, 5))
    plt.plot(qp_df["std_annual"], qp_df["return_annual"], marker="o", label="QP utility")
    plt.plot(socp_df["std_annual"], socp_df["return_annual"], marker="s", label="SOCP risk budget")
    plt.xlabel("Annualized volatility")
    plt.ylabel("Annualized expected return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "qp_socp_frontier.png", dpi=200)
    plt.close()

    weights = pd.read_csv(RESULTS_DIR / "qp_frontier_weights.csv", index_col=0)
    selected = weights.iloc[[0, len(weights) // 2, -1]].T
    selected.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Portfolio weight")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "qp_selected_weights.png", dpi=200)
    plt.close()


def main():
    solver = installed_solver()
    mu, sigma = load_inputs()
    qp_df = run_qp_frontier(mu, sigma, solver)
    socp_df = run_socp_frontier(mu, sigma, solver)
    run_box_constraint_example(mu, sigma, solver)
    make_plots(qp_df, socp_df)
    print(f"Finished commercial-solver baselines with {solver}.")


if __name__ == "__main__":
    main()
