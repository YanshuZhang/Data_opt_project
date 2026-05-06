import time

import numpy as np
import pandas as pd

import config
from utils import portfolio_objective, project_simplex


def solve_linear_system(M, b):
    try:
        L = np.linalg.cholesky(M)
        y = np.linalg.solve(L, b)
        return np.linalg.solve(L.T, y)
    except np.linalg.LinAlgError:
        return np.linalg.solve(M, b)


def admm_qp(mu, Sigma, gamma, rho, max_iter=config.MAX_ITER, tol=config.TOL):
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(mu)
    x = np.ones(n) / n
    z = x.copy()
    u = np.zeros(n)
    M = gamma * Sigma + rho * np.eye(n)
    rows = []

    tic = time.perf_counter()
    for k in range(1, max_iter + 1):
        z_old = z.copy()
        x = solve_linear_system(M, mu + rho * (z - u))
        z = project_simplex(x + u)
        u = u + x - z

        primal = np.linalg.norm(x - z)
        dual = rho * np.linalg.norm(z - z_old)
        obj = portfolio_objective(mu, Sigma, z, gamma)
        rows.append({"iteration": k, "objective": obj, "primal_residual": primal, "dual_residual": dual})
        if primal < tol and dual < tol:
            break

    elapsed = time.perf_counter() - tic
    history = pd.DataFrame(rows)
    return {
        "rho": rho,
        "weights": z,
        "objective": portfolio_objective(mu, Sigma, z, gamma),
        "primal_residual": float(history.iloc[-1]["primal_residual"]),
        "dual_residual": float(history.iloc[-1]["dual_residual"]),
        "iterations": len(history),
        "runtime_sec": elapsed,
        "history": history,
    }


def run_admm_grid(mu, Sigma, gamma=config.CORE_GAMMA):
    summaries = []
    histories = {}
    for rho in config.ADMM_RHOS:
        out = admm_qp(mu, Sigma, gamma, rho)
        histories[rho] = out["history"]
        summaries.append({k: v for k, v in out.items() if k not in ["weights", "history"]})
    summary = pd.DataFrame(summaries)
    return summary, histories


def pdhg_qp(mu, Sigma, gamma, tau, sigma, max_iter=config.MAX_ITER, tol=config.TOL):
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(mu)
    x = np.ones(n) / n
    x_bar = x.copy()
    p = 0.0
    q = np.zeros(n)
    M = np.eye(n) + tau * gamma * Sigma
    rows = []

    tic = time.perf_counter()
    for k in range(1, max_iter + 1):
        x_old = x.copy()
        p = p + sigma * (np.sum(x_bar) - 1.0)
        q = np.maximum(q - sigma * x_bar, 0.0)
        x = solve_linear_system(M, x - tau * (p * np.ones(n) - q) + tau * mu)
        x_bar = x + (x - x_old)

        feasibility = np.sqrt((np.sum(x) - 1.0) ** 2 + np.linalg.norm(np.minimum(x, 0.0)) ** 2)
        step = np.linalg.norm(x - x_old)
        obj = portfolio_objective(mu, Sigma, x, gamma)
        rows.append({"iteration": k, "objective": obj, "feasibility": feasibility, "step_norm": step})
        if feasibility < tol and step < tol:
            break

    elapsed = time.perf_counter() - tic
    history = pd.DataFrame(rows)
    return {
        "tau": tau,
        "sigma": sigma,
        "weights": x,
        "objective": portfolio_objective(mu, Sigma, x, gamma),
        "feasibility": float(history.iloc[-1]["feasibility"]),
        "step_norm": float(history.iloc[-1]["step_norm"]),
        "iterations": len(history),
        "runtime_sec": elapsed,
        "history": history,
    }


def run_pdhg_grid(mu, Sigma, gamma=config.CORE_GAMMA):
    n = len(mu)
    k_norm_sq = n + 1.0
    summaries = []
    histories = {}
    for tau in config.PDHG_TAUS:
        sigma = 0.9 / (tau * k_norm_sq)
        out = pdhg_qp(mu, Sigma, gamma, tau, sigma)
        histories[tau] = out["history"]
        summaries.append({k: v for k, v in out.items() if k not in ["weights", "history"]})
    summary = pd.DataFrame(summaries)
    return summary, histories
