import numpy as np
import pandas as pd

import config
from optimization import solve_qp_utility, solve_socp
from utils import normalize_weights, performance_metrics, portfolio_variance, regularize_cov


def estimate_window(window_returns):
    mu = window_returns.mean().values
    Sigma = regularize_cov(window_returns.cov().values, config.COV_EPS)
    return mu, Sigma


def compute_weights(strategy, window_returns, solver, gamma=None, sigma_multiplier=None):
    n = window_returns.shape[1]
    if strategy == "Equal Weight":
        return np.ones(n) / n

    mu, Sigma = estimate_window(window_returns)
    if strategy == "QP":
        return solve_qp_utility(mu, Sigma, gamma, solver)["weights"]
    if strategy == "SOCP":
        equal = np.ones(n) / n
        sigma_max = sigma_multiplier * np.sqrt(portfolio_variance(Sigma, equal))
        return solve_socp(mu, Sigma, sigma_max, solver)["weights"]
    raise ValueError(f"Unknown strategy: {strategy}")


def rolling_strategy_returns(returns, strategy, solver, gamma, sigma_multiplier, cost_bps):
    cost_rate = cost_bps / 10000.0
    start_pos = returns.index.get_indexer([pd.Timestamp(config.TEST_START)], method="bfill")[0]
    daily_parts = []
    turnovers = []
    failures = 0
    prev_end_weights = None

    for pos in range(start_pos, len(returns), config.REBALANCE_FREQ):
        train_start = pos - config.TRAIN_WINDOW
        if train_start < 0:
            continue
        window = returns.iloc[train_start:pos]
        period = returns.iloc[pos:min(pos + config.REBALANCE_FREQ, len(returns))]
        if period.empty:
            continue

        try:
            weights = compute_weights(strategy, window, solver, gamma, sigma_multiplier)
        except Exception:
            failures += 1
            weights = np.ones(returns.shape[1]) / returns.shape[1] if prev_end_weights is None else prev_end_weights
        weights = normalize_weights(weights)

        base_weights = np.zeros_like(weights) if prev_end_weights is None else prev_end_weights
        turnover = float(np.sum(np.abs(weights - base_weights)))
        turnovers.append(turnover)

        period_returns = period @ weights
        period_returns = period_returns.copy()
        period_returns.iloc[0] -= cost_rate * turnover
        daily_parts.append(period_returns)

        asset_growth = (1.0 + period).prod(axis=0).values
        denom = float(weights @ asset_growth)
        prev_end_weights = weights if denom <= 0 else normalize_weights(weights * asset_growth / denom)

    if len(daily_parts) == 0:
        return pd.Series(dtype=float), [], failures
    return pd.concat(daily_parts).sort_index(), turnovers, failures


def run_backtests(returns, solver, gamma, sigma_multiplier):
    metrics_rows = []
    wealth_by_cost = {}
    failures_rows = []

    for cost_bps in config.TRANSACTION_COST_BPS:
        wealth_by_cost[cost_bps] = {}
        for strategy in ["Equal Weight", "QP", "SOCP"]:
            daily, turnover, failures = rolling_strategy_returns(
                returns, strategy, solver, gamma, sigma_multiplier, cost_bps
            )
            metrics = performance_metrics(daily, turnover)
            metrics_rows.append({"strategy": strategy, "transaction_cost_bps": cost_bps, **metrics})
            wealth_by_cost[cost_bps][strategy] = (1.0 + daily).cumprod()
            failures_rows.append({"strategy": strategy, "transaction_cost_bps": cost_bps, "solver_failures": failures})

    metrics_df = pd.DataFrame(metrics_rows)
    failures_df = pd.DataFrame(failures_rows)
    metrics_df.to_csv(config.CSV_DIR / "backtest_metrics.csv", index=False)
    failures_df.to_csv(config.CSV_DIR / "backtest_failures.csv", index=False)

    for cost_bps, curves in wealth_by_cost.items():
        wealth = pd.DataFrame(curves)
        wealth.to_csv(config.CSV_DIR / f"backtest_wealth_{cost_bps}bps.csv")
    return metrics_df, failures_df, wealth_by_cost
