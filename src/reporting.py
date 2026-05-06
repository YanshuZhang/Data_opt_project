import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from utils import format_float, format_pct


def escape_latex(x):
    s = str(x)
    return s.replace("_", "\\_").replace("%", "\\%")


def write_table(df, path, columns, headers, formats=None):
    formats = formats or {}
    lines = ["\\begin{center}", "\\small", "\\begin{tabular}{" + "l" + "r" * (len(columns) - 1) + "}", "\\toprule"]
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if col in formats:
                vals.append(formats[col](value))
            elif isinstance(value, float):
                vals.append(format_float(value, 4))
            else:
                vals.append(escape_latex(value))
        lines.append(" & ".join(vals) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{center}"])
    path.write_text("\n".join(lines), encoding="utf-8")


def make_data_tables(cleaning_summary, split_summary, param_summary):
    write_table(
        cleaning_summary,
        config.TABLE_DIR / "data_cleaning_summary.tex",
        ["item", "value"],
        ["项目", "数值"],
        {"value": lambda x: format_float(float(x), 4) if isinstance(x, float) else escape_latex(x)},
    )
    write_table(
        split_summary,
        config.TABLE_DIR / "time_split_summary.tex",
        ["stage", "start_date", "end_date", "observations"],
        ["阶段", "开始日期", "结束日期", "观测数"],
    )
    write_table(
        param_summary,
        config.TABLE_DIR / "parameter_estimation_summary.tex",
        ["item", "value"],
        ["项目", "数值"],
        {"value": lambda x: format_float(float(x), 6)},
    )


def make_solver_tables(qp_frontier, socp_frontier, selected):
    def summarize_frontier(df, parameter_col):
        if df.empty:
            return df
        idx = sorted(set([0, len(df) // 4, len(df) // 2, 3 * len(df) // 4, len(df) - 1]))
        out = df.iloc[idx].copy()
        out["annual_return"] = 252 * out["expected_return"]
        out["annual_std"] = np.sqrt(252) * out["std"]
        return out

    qp = summarize_frontier(qp_frontier, "gamma")
    if not qp.empty:
        write_table(
            qp,
            config.TABLE_DIR / "qp_frontier_summary.tex",
            ["gamma", "annual_return", "annual_std", "objective", "status", "runtime_sec"],
            ["$\\gamma$", "年化收益", "年化波动", "目标值", "状态", "时间(s)"],
            {"annual_return": format_pct, "annual_std": format_pct, "objective": lambda x: format_float(x, 6), "runtime_sec": lambda x: format_float(x, 4)},
        )
    socp = summarize_frontier(socp_frontier, "sigma_max")
    if not socp.empty:
        write_table(
            socp,
            config.TABLE_DIR / "socp_frontier_summary.tex",
            ["sigma_max", "annual_return", "annual_std", "objective", "status", "runtime_sec"],
            ["$\\sigma_{\\max}$", "年化收益", "年化波动", "目标值", "状态", "时间(s)"],
            {"sigma_max": lambda x: format_float(x, 6), "annual_return": format_pct, "annual_std": format_pct, "objective": lambda x: format_float(x, 6), "runtime_sec": lambda x: format_float(x, 4)},
        )
    write_table(
        selected,
        config.TABLE_DIR / "selected_hyperparameters.tex",
        ["parameter", "selected_value"],
        ["参数", "验证集选择值"],
        {"selected_value": lambda x: format_float(float(x), 4)},
    )


def make_algorithm_tables(admm_summary, pdhg_summary, baseline_objective):
    admm = admm_summary.copy()
    admm["objective_gap"] = admm["objective"] - baseline_objective
    write_table(
        admm,
        config.TABLE_DIR / "admm_summary.tex",
        ["rho", "objective", "objective_gap", "primal_residual", "dual_residual", "iterations", "runtime_sec"],
        ["$\\rho$", "目标值", "目标差", "原始残差", "对偶残差", "迭代数", "时间(s)"],
        {"objective": lambda x: format_float(x, 6), "objective_gap": lambda x: f"{x:.2e}", "primal_residual": lambda x: f"{x:.2e}", "dual_residual": lambda x: f"{x:.2e}", "iterations": lambda x: str(int(x)), "runtime_sec": lambda x: format_float(x, 4)},
    )
    pdhg = pdhg_summary.copy()
    pdhg["objective_gap"] = pdhg["objective"] - baseline_objective
    write_table(
        pdhg,
        config.TABLE_DIR / "pdhg_summary.tex",
        ["tau", "sigma", "objective", "objective_gap", "feasibility", "iterations", "runtime_sec"],
        ["$\\tau$", "$\\sigma$", "目标值", "目标差", "可行性", "迭代数", "时间(s)"],
        {"tau": lambda x: format_float(x, 3), "sigma": lambda x: format_float(x, 6), "objective": lambda x: format_float(x, 6), "objective_gap": lambda x: f"{x:.2e}", "feasibility": lambda x: f"{x:.2e}", "iterations": lambda x: str(int(x)), "runtime_sec": lambda x: format_float(x, 4)},
    )


def make_backtest_tables(metrics_df, failures_df, gamma_sensitivity, param_summary):
    zero = metrics_df[metrics_df["transaction_cost_bps"] == 0].copy()
    write_table(
        zero,
        config.TABLE_DIR / "backtest_metrics_0bps.tex",
        ["strategy", "cumulative_return", "annual_return", "annual_volatility", "sharpe", "max_drawdown", "average_turnover"],
        ["策略", "累计收益", "年化收益", "年化波动", "Sharpe", "最大回撤", "平均换手"],
        {"cumulative_return": format_pct, "annual_return": format_pct, "annual_volatility": format_pct, "sharpe": lambda x: format_float(x, 3), "max_drawdown": format_pct, "average_turnover": lambda x: format_float(x, 3)},
    )

    cost = metrics_df.pivot(index="transaction_cost_bps", columns="strategy", values="annual_return").reset_index()
    columns = ["transaction_cost_bps"] + [c for c in ["Equal Weight", "QP", "SOCP"] if c in cost.columns]
    write_table(
        cost,
        config.TABLE_DIR / "cost_sensitivity.tex",
        columns,
        ["交易成本(bps)"] + columns[1:],
        {c: format_pct for c in columns[1:]},
    )

    failures = failures_df.groupby("strategy", as_index=False)["solver_failures"].max()
    write_table(
        failures,
        config.TABLE_DIR / "solver_failures.tex",
        ["strategy", "solver_failures"],
        ["策略", "最大失败次数"],
    )

    worst = gamma_sensitivity.sort_values("sharpe", ascending=True).iloc[0]
    best = gamma_sensitivity.sort_values("sharpe", ascending=False).iloc[0]
    cond = param_summary.loc[param_summary["item"] == "condition_number", "value"].iloc[0]
    failure = pd.DataFrame({
        "item": ["worst_gamma_on_validation", "worst_validation_sharpe", "best_gamma_on_validation", "best_validation_sharpe", "covariance_condition_number"],
        "value": [worst["gamma"], worst["sharpe"], best["gamma"], best["sharpe"], cond],
    })
    write_table(
        failure,
        config.TABLE_DIR / "failure_case.tex",
        ["item", "value"],
        ["检查项", "数值"],
        {"value": lambda x: format_float(float(x), 4)},
    )


def plot_frontier(qp_frontier, socp_frontier):
    plt.figure(figsize=(6, 4))
    if not qp_frontier.empty:
        plt.plot(np.sqrt(252) * qp_frontier["std"], 252 * qp_frontier["expected_return"], marker="o", label="QP")
    if not socp_frontier.empty:
        plt.scatter(np.sqrt(252) * socp_frontier["std"], 252 * socp_frontier["expected_return"], marker="x", label="SOCP")
    plt.xlabel("Annualized standard deviation")
    plt.ylabel("Annualized expected return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / "efficient_frontier.png", dpi=200)
    plt.close()


def plot_qp_weights(weight_table):
    if weight_table.empty:
        return
    avg = weight_table.mean(axis=0).sort_values(ascending=False)
    top = list(avg.head(8).index)
    data = weight_table[top]
    gamma_values = pd.to_numeric(
        weight_table.index.to_series().astype(str).str.replace("gamma=", "", regex=False),
        errors="coerce",
    )
    x_values = gamma_values.to_numpy() if not gamma_values.isna().any() else np.arange(len(data))
    plt.figure(figsize=(7, 4))
    plt.stackplot(x_values, data.T.values, labels=top)
    if len(x_values) > 1 and np.all(np.asarray(x_values) > 0):
        plt.xscale("log")
    plt.xlabel("Gamma")
    plt.ylabel("Weight")
    plt.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / "qp_weights.png", dpi=200)
    plt.close()


def plot_algorithm_histories(admm_histories, pdhg_histories):
    plt.figure(figsize=(6, 4))
    for rho, hist in admm_histories.items():
        residual = hist["primal_residual"] + hist["dual_residual"]
        plt.semilogy(hist["iteration"], residual, label=f"rho={rho:g}")
    plt.xlabel("Iteration")
    plt.ylabel("ADMM residual sum")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / "admm_convergence.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    for tau, hist in pdhg_histories.items():
        plt.semilogy(hist["iteration"], hist["feasibility"], label=f"tau={tau:g}")
    plt.xlabel("Iteration")
    plt.ylabel("PDHG feasibility")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / "pdhg_convergence.png", dpi=200)
    plt.close()


def plot_backtest(wealth_by_cost, metrics_df, gamma_sensitivity):
    curves = wealth_by_cost.get(0, {})
    if curves:
        plt.figure(figsize=(7, 4))
        for strategy, wealth in curves.items():
            plt.plot(wealth.index, wealth.values, label=strategy)
        plt.xlabel("Date")
        plt.ylabel("Cumulative wealth")
        plt.legend()
        plt.tight_layout()
        plt.savefig(config.FIGURE_DIR / "backtest_wealth.png", dpi=200)
        plt.close()

    if not metrics_df.empty:
        plt.figure(figsize=(6, 4))
        for strategy, df in metrics_df.groupby("strategy"):
            plt.plot(df["transaction_cost_bps"], df["annual_return"], marker="o", label=strategy)
        plt.xlabel("Transaction cost (bps per turnover unit)")
        plt.ylabel("Annualized return")
        plt.legend()
        plt.tight_layout()
        plt.savefig(config.FIGURE_DIR / "cost_sensitivity.png", dpi=200)
        plt.close()

    if not gamma_sensitivity.empty:
        plt.figure(figsize=(6, 4))
        plt.semilogx(gamma_sensitivity["gamma"], gamma_sensitivity["sharpe"], marker="o")
        plt.xlabel("Gamma")
        plt.ylabel("Validation Sharpe")
        plt.tight_layout()
        plt.savefig(config.FIGURE_DIR / "gamma_sensitivity.png", dpi=200)
        plt.close()


def compile_report():
    tex = config.REPORT_DIR / "portfolio_report.tex"
    if not tex.exists():
        return
    for _ in range(2):
        subprocess.run(
            ["xelatex", "-interaction=nonstopmode", tex.name],
            cwd=config.REPORT_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )