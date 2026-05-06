import argparse
import pandas as pd

import config
from algorithms import run_admm_grid, run_pdhg_grid
from backtest import run_backtests
from data_pipeline import clean_prices_and_returns, download_prices, estimate_parameters, split_returns
from optimization import get_solver, run_qp_frontier, run_socp_frontier, select_hyperparameters, solve_qp_utility
from reporting import (
    compile_report,
    make_algorithm_tables,
    make_backtest_tables,
    make_data_tables,
    make_solver_tables,
    plot_algorithm_histories,
    plot_backtest,
    plot_frontier,
    plot_qp_weights,
)
from utils import ensure_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--allow-open-source-fallback", action="store_true")
    parser.add_argument("--compile-pdf", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()

    # Data preparation
    if args.skip_download and (config.DATA_DIR / "adjusted_close_prices.csv").exists():
        prices = pd.read_csv(config.DATA_DIR / "adjusted_close_prices.csv", index_col=0, parse_dates=True)
    else:
        prices = download_prices()
    _, returns, _, cleaning_summary = clean_prices_and_returns(prices)
    train, validation, _, split_summary = split_returns(returns)
    mu, Sigma, param_summary = estimate_parameters(train)

    # Commercial-solver baselines
    solver = get_solver(allow_fallback=args.allow_open_source_fallback)
    best_gamma, best_sigma_multiplier, gamma_sensitivity, socp_sensitivity = select_hyperparameters(
        train, validation, mu, Sigma, solver
    )
    qp_frontier, qp_weights = run_qp_frontier(mu, Sigma, solver)
    socp_frontier, _ = run_socp_frontier(mu, Sigma, solver, qp_frontier)
    baseline = solve_qp_utility(mu, Sigma, config.CORE_GAMMA, solver)

    # First-order methods
    admm_summary, admm_histories = run_admm_grid(mu.values, Sigma.values, config.CORE_GAMMA)
    pdhg_summary, pdhg_histories = run_pdhg_grid(mu.values, Sigma.values, config.CORE_GAMMA)

    # Out-of-sample backtesting
    metrics_df, failures_df, wealth_by_cost = run_backtests(returns, solver, best_gamma, best_sigma_multiplier)

    # Report artifacts
    make_data_tables(cleaning_summary, split_summary, param_summary)
    selected = pd.read_csv(config.CSV_DIR / "selected_hyperparameters.csv")
    make_solver_tables(qp_frontier, socp_frontier, selected)
    make_algorithm_tables(admm_summary, pdhg_summary, baseline["objective"])
    make_backtest_tables(metrics_df, failures_df, gamma_sensitivity, param_summary)

    plot_frontier(qp_frontier, socp_frontier)
    plot_qp_weights(qp_weights)
    plot_algorithm_histories(admm_histories, pdhg_histories)
    plot_backtest(wealth_by_cost, metrics_df, gamma_sensitivity)

    if args.compile_pdf:
        compile_report()

    print(f"Solver used: {solver}")
    print(f"Selected gamma: {best_gamma}")
    print(f"Selected sigma multiplier: {best_sigma_multiplier}")
    print("Finished. Results are in data/, results/, and report/.")


if __name__ == "__main__":
    main()
