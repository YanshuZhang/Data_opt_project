import argparse
import numpy as np
import pandas as pd

import config
from algorithms import run_admm_grid, run_pdhg_grid
from backtest import generate_gamma_sensitivity_test_figure, run_backtests
from data_pipeline import clean_prices_and_returns, download_prices, estimate_parameters, split_returns
from optimization import (
    get_solver,
    run_qp_frontier,
    run_socp_frontier,
    select_hyperparameters,
    solve_qp_utility,
    solve_socp,
    summarize_portfolio_point,
)
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


def _summary_to_dict(df):
    return dict(zip(df["item"], df["value"]))


def _fmt_pct(value):
    return f"{100.0 * float(value):.2f}%"


def _fmt_num(value, digits=6):
    return f"{float(value):.{digits}f}"


def _log_section(title):
    print(f"\n[{title}]")


def _log_data_overview(source_label, prices, cleaning_summary, split_summary, param_summary):
    cleaning = _summary_to_dict(cleaning_summary)
    params = _summary_to_dict(param_summary)
    split_counts = {row.stage: int(row.observations) for row in split_summary.itertuples(index=False)}

    print(
        f"Source={source_label}; prices={prices.shape[0]} x {prices.shape[1]}; "
        f"returns={int(cleaning['return_observations'])}; "
        f"assets={int(cleaning['assets_before_cleaning'])}->{int(cleaning['assets_after_cleaning'])}; "
        f"split train/val/test={split_counts.get('train', 0)}/{split_counts.get('validation', 0)}/{split_counts.get('test', 0)}; "
        f"cond={_fmt_num(params['condition_number'], digits=2)}"
    )


def _log_hyperparameters(solver, best_gamma, best_box_gamma, best_sigma_multiplier, best_box_sigma_multiplier):
    print(
        f"solver={solver}; gamma={best_gamma}; box_gamma={best_box_gamma}; "
        f"sigma_mult={_fmt_num(best_sigma_multiplier, digits=4)}; "
        f"box_sigma_mult={_fmt_num(best_box_sigma_multiplier, digits=4)}"
    )


def _log_portfolio_summary(modification_summary):
    if modification_summary.empty:
        return

    qp_row = modification_summary.iloc[0]
    print(
        f"QP point: return={_fmt_pct(qp_row.annual_return)}, "
        f"vol={_fmt_pct(qp_row.annual_std)}, max_weight={_fmt_pct(qp_row.max_weight)}, "
        f"active={int(qp_row.active_positions)}"
    )


def _log_algorithm_summary(name, summary, metric_name):
    if summary.empty:
        print(f"{name}: no runs completed")
        return

    best_row = summary.loc[summary[metric_name].idxmin()]
    print(
        f"{name}: {len(summary)} runs, best {metric_name}={_fmt_num(best_row[metric_name])}, "
        f"iterations={int(best_row['iterations'])}, runtime={_fmt_num(best_row['runtime_sec'], digits=3)}s"
    )


def _log_backtest_summary(metrics_df):
    zero_cost = metrics_df[metrics_df["transaction_cost_bps"] == 0].copy()
    if zero_cost.empty:
        print("No 0 bps backtest metrics")
        return

    best_row = zero_cost.sort_values("sharpe", ascending=False).iloc[0]
    print(
        f"Best backtest at 0 bps: {best_row['strategy']}, "
        f"return={_fmt_pct(best_row['annual_return'])}, "
        f"vol={_fmt_pct(best_row['annual_volatility'])}, "
        f"sharpe={_fmt_num(best_row['sharpe'], digits=3)}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--allow-open-source-fallback", action="store_true")
    parser.add_argument("--compile-pdf", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()
    print("Starting run_all...")

    # Data preparation
    _log_section("Data preparation")
    if args.skip_download and (config.DATA_DIR / "adjusted_close_prices.csv").exists():
        prices = pd.read_csv(config.DATA_DIR / "adjusted_close_prices.csv", index_col=0, parse_dates=True)
        price_source = "cached adjusted_close_prices.csv"
    else:
        prices = download_prices()
        price_source = "downloaded from yfinance"
    _, returns, _, cleaning_summary = clean_prices_and_returns(prices)
    train, validation, _, split_summary = split_returns(returns)
    mu, Sigma, param_summary = estimate_parameters(train)
    _log_data_overview(price_source, prices, cleaning_summary, split_summary, param_summary)

    # Commercial-solver baselines
    _log_section("Commercial solver optimization")
    solver = get_solver(allow_fallback=args.allow_open_source_fallback)
    best_gamma, best_box_gamma, best_sigma_multiplier, best_box_sigma_multiplier, _, _, _, _ = select_hyperparameters(
        train, validation, mu, Sigma, solver, box_max_weight=config.BOX_WEIGHT_CAP
    )
    _log_hyperparameters(solver, best_gamma, best_box_gamma, best_sigma_multiplier, best_box_sigma_multiplier)

    qp_frontier, qp_weights = run_qp_frontier(mu, Sigma, solver)
    box_qp_frontier, box_qp_weights = run_qp_frontier(
        mu, Sigma, solver, max_weight=config.BOX_WEIGHT_CAP, file_stem="box_qp_frontier"
    )
    socp_frontier, _ = run_socp_frontier(mu, Sigma, solver, qp_frontier)
    box_socp_frontier, _ = run_socp_frontier(
        mu, Sigma, solver, qp_frontier, max_weight=config.BOX_WEIGHT_CAP, file_stem="box_socp_frontier"
    )
    # Use selected hyperparameters for the reported commercial-solver portfolio points.
    # Keep a separate CORE_GAMMA baseline for the first-order-method objective comparison.
    core_baseline = solve_qp_utility(mu, Sigma, config.CORE_GAMMA, solver)
    baseline = solve_qp_utility(mu, Sigma, best_gamma, solver)
    box_baseline = solve_qp_utility(mu, Sigma, best_box_gamma, solver, max_weight=config.BOX_WEIGHT_CAP)
    equal = pd.Series([1.0 / len(mu)] * len(mu), index=mu.index if hasattr(mu, "index") else None)
    sigma_base = float(np.sqrt((equal.values if hasattr(equal, "values") else equal) @ Sigma.values @ (equal.values if hasattr(equal, "values") else equal)))
    socp_baseline = solve_socp(mu, Sigma, best_sigma_multiplier * sigma_base, solver)
    box_socp_baseline = solve_socp(mu, Sigma, best_box_sigma_multiplier * sigma_base, solver, max_weight=config.BOX_WEIGHT_CAP)
    modification_summary = pd.DataFrame(
        [
            summarize_portfolio_point("QP", baseline),
            summarize_portfolio_point(f"QP + box ({config.BOX_WEIGHT_CAP:.0%} cap)", box_baseline),
            summarize_portfolio_point("SOCP", socp_baseline),
            summarize_portfolio_point(f"SOCP + box ({config.BOX_WEIGHT_CAP:.0%} cap)", box_socp_baseline),
        ]
    )
    print(
        f"Frontiers: qp={len(qp_frontier)}, qp_box={len(box_qp_frontier)}, "
        f"socp={len(socp_frontier)}, socp_box={len(box_socp_frontier)}"
    )
    _log_portfolio_summary(modification_summary)

    # First-order methods
    _log_section("First-order methods")
    admm_summary, admm_histories = run_admm_grid(mu.values, Sigma.values, config.CORE_GAMMA)
    pdhg_summary, pdhg_histories = run_pdhg_grid(mu.values, Sigma.values, config.CORE_GAMMA)
    _log_algorithm_summary("ADMM", admm_summary, "objective")
    _log_algorithm_summary("PDHG", pdhg_summary, "objective")

    # Out-of-sample backtesting
    _log_section("Backtesting")
    metrics_df, wealth_by_cost = run_backtests(
        returns, solver, best_gamma, best_box_gamma, best_sigma_multiplier, best_box_sigma_multiplier
    )
    _log_backtest_summary(metrics_df)

    # Report artifacts
    _log_section("Artifacts")
    make_data_tables(cleaning_summary, split_summary, param_summary)
    make_solver_tables(qp_frontier, socp_frontier, box_qp_frontier, box_socp_frontier, modification_summary)
    make_algorithm_tables(admm_summary, pdhg_summary, core_baseline["objective"])
    make_backtest_tables(metrics_df)

    plot_frontier(qp_frontier, socp_frontier)
    plot_qp_weights(qp_weights)
    plot_qp_weights(box_qp_weights, file_name="box_qp_weights.png")
    plot_algorithm_histories(admm_histories, pdhg_histories)
    plot_backtest(wealth_by_cost, metrics_df)
    generate_gamma_sensitivity_test_figure(returns, solver)
    print("Saved outputs under results/ and report/.")

    if args.compile_pdf:
        _log_section("Report compilation")
        compile_report()
        print("PDF report compiled.")

    _log_section("Done")
    print("Finished. Results are in data/, results/, and report/.")


if __name__ == "__main__":
    main()
