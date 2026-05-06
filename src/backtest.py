import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from optimization import solve_qp_utility, solve_socp
from utils import normalize_weights, performance_metrics, portfolio_variance, regularize_cov


def estimate_window(window_returns):
    mu = window_returns.mean().values
    Sigma = regularize_cov(window_returns.cov().values, config.COV_EPS)
    return mu, Sigma


def compute_weights(strategy, window_returns, solver, gamma=None, sigma_multiplier=None, reference_weights=None):
    n = window_returns.shape[1]
    if strategy == "Equal Weight":
        return np.ones(n) / n

    mu, Sigma = estimate_window(window_returns)
    if strategy == "QP":
        return solve_qp_utility(mu, Sigma, gamma, solver)["weights"]
    if strategy == "QP Box":
        return solve_qp_utility(mu, Sigma, gamma, solver, max_weight=config.BOX_WEIGHT_CAP)["weights"]
    if strategy == "SOCP":
        equal = np.ones(n) / n
        sigma_max = sigma_multiplier * np.sqrt(portfolio_variance(Sigma, equal))
        return solve_socp(mu, Sigma, sigma_max, solver)["weights"]
    if strategy == "SOCP Box":
        equal = np.ones(n) / n
        sigma_max = sigma_multiplier * np.sqrt(portfolio_variance(Sigma, equal))
        return solve_socp(mu, Sigma, sigma_max, solver, max_weight=config.BOX_WEIGHT_CAP)["weights"]
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
            weights = compute_weights(strategy, window, solver, gamma, sigma_multiplier, prev_end_weights)
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


def rolling_returns_on_period(returns, strategy, solver, start_date, end_date, gamma=None, sigma_multiplier=None, cost_bps=0):
    cost_rate = cost_bps / 10000.0
    start_pos = returns.index.get_indexer([pd.Timestamp(start_date)], method="bfill")[0]
    end_pos = returns.index.get_indexer([pd.Timestamp(end_date)], method="ffill")[0] + 1

    daily_parts = []
    turnovers = []
    prev_end_weights = None

    for pos in range(start_pos, end_pos, config.REBALANCE_FREQ):
        train_start = pos - config.TRAIN_WINDOW
        if train_start < 0:
            continue

        window = returns.iloc[train_start:pos]
        period = returns.iloc[pos:min(pos + config.REBALANCE_FREQ, end_pos)]
        if period.empty:
            continue

        try:
            weights = compute_weights(strategy, window, solver, gamma, sigma_multiplier, prev_end_weights)
        except Exception:
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

    if not daily_parts:
        return pd.Series(dtype=float), []
    return pd.concat(daily_parts).sort_index(), turnovers


def plot_gamma_sensitivity_test(df, equal_test_sharpe, selected_gamma, best_test_gamma, path):
    plt.figure(figsize=(6.4, 4.0))
    plt.semilogx(df["gamma"], df["validation_sharpe"], marker="o", label="Validation Sharpe")
    plt.semilogx(df["gamma"], df["test_sharpe"], marker="s", label="Test Sharpe")
    plt.axhline(equal_test_sharpe, linestyle="--", label="Equal Weight Test Sharpe")
    plt.axvline(selected_gamma, linestyle=":", label="Selected gamma")
    plt.axvline(best_test_gamma, linestyle="-.", label="Test-best gamma")
    plt.xlabel("Gamma")
    plt.ylabel("Sharpe")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def generate_gamma_sensitivity_test_figure(returns, solver, cost_bps=0.0):
    val_start = pd.to_datetime(config.TRAIN_END) + pd.Timedelta(days=1)
    val_end = pd.to_datetime(config.VAL_END)
    test_start = pd.to_datetime(config.TEST_START)
    test_end = returns.index.max()

    eq_test_daily, eq_test_turnover = rolling_returns_on_period(
        returns, "Equal Weight", solver, test_start, test_end, cost_bps=cost_bps
    )
    eq_test = performance_metrics(eq_test_daily, eq_test_turnover)

    rows = []
    for gamma in config.GAMMA_GRID:
        val_daily, val_turnover = rolling_returns_on_period(
            returns, "QP", solver, val_start, val_end, gamma=gamma, cost_bps=cost_bps
        )
        test_daily, test_turnover = rolling_returns_on_period(
            returns, "QP", solver, test_start, test_end, gamma=gamma, cost_bps=cost_bps
        )
        val_metrics = performance_metrics(val_daily, val_turnover)
        test_metrics = performance_metrics(test_daily, test_turnover)
        rows.append({
            "gamma": gamma,
            "validation_sharpe": val_metrics["sharpe"],
            "test_sharpe": test_metrics["sharpe"],
        })

    df = pd.DataFrame(rows).sort_values("gamma")
    if df.empty:
        return

    selected_gamma = float(df.sort_values("validation_sharpe", ascending=False).iloc[0]["gamma"])
    best_test_gamma = float(df.sort_values("test_sharpe", ascending=False).iloc[0]["gamma"])
    plot_gamma_sensitivity_test(
        df,
        eq_test["sharpe"],
        selected_gamma,
        best_test_gamma,
        config.FIGURE_DIR / "gamma_sensitivity_test.png",
    )


def run_backtests(returns, solver, gamma, box_gamma, sigma_multiplier, box_sigma_multiplier):
    metrics_rows = []
    wealth_by_cost = {}

    gamma_by_strategy = {
        "Equal Weight": None,
        "QP": gamma,
        "QP Box": box_gamma,
        "SOCP": gamma,
        "SOCP Box": gamma,
    }
    sigma_by_strategy = {
        "Equal Weight": None,
        "QP": sigma_multiplier,
        "QP Box": sigma_multiplier,
        "SOCP": sigma_multiplier,
        "SOCP Box": box_sigma_multiplier,
    }

    for cost_bps in config.TRANSACTION_COST_BPS:
        wealth_by_cost[cost_bps] = {}
        for strategy in ["Equal Weight", "QP", "QP Box", "SOCP", "SOCP Box"]:
            daily, turnover, _ = rolling_strategy_returns(
                returns, strategy, solver, gamma_by_strategy[strategy], sigma_by_strategy[strategy], cost_bps
            )
            metrics = performance_metrics(daily, turnover)
            metrics_rows.append({"strategy": strategy, "transaction_cost_bps": cost_bps, **metrics})
            wealth_by_cost[cost_bps][strategy] = (1.0 + daily).cumprod()

    metrics_df = pd.DataFrame(metrics_rows)
    return metrics_df, wealth_by_cost
