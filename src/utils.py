import numpy as np
import pandas as pd

import config


def ensure_dirs():
    for path in [config.DATA_DIR, config.CSV_DIR, config.FIGURE_DIR, config.TABLE_DIR, config.REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def project_simplex(v):
    v = np.asarray(v, dtype=float)
    if v.ndim != 1:
        raise ValueError("project_simplex expects a one-dimensional vector")
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, len(v) + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.ones_like(v) / len(v)
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


def regularize_cov(Sigma, eps=config.COV_EPS):
    Sigma = np.asarray(Sigma, dtype=float)
    Sigma = 0.5 * (Sigma + Sigma.T)
    return Sigma + eps * np.eye(Sigma.shape[0])


def factorize_cov(Sigma):
    Sigma = regularize_cov(Sigma, 0.0)
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(np.sqrt(eigvals))


def normalize_weights(w):
    w = np.asarray(w, dtype=float).ravel()
    w[w < 1e-10] = 0.0
    s = w.sum()
    if s <= 0:
        return np.ones_like(w) / len(w)
    return w / s


def portfolio_return(mu, w):
    return float(np.asarray(mu) @ np.asarray(w))


def portfolio_variance(Sigma, w):
    w = np.asarray(w)
    return float(w @ np.asarray(Sigma) @ w)


def portfolio_objective(mu, Sigma, w, gamma):
    return 0.5 * gamma * portfolio_variance(Sigma, w) - portfolio_return(mu, w)


def max_drawdown(wealth):
    wealth = pd.Series(wealth).dropna()
    if wealth.empty:
        return np.nan
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def performance_metrics(daily_returns, turnover=None, trading_days=config.TRADING_DAYS):
    r = pd.Series(daily_returns).dropna()
    if len(r) == 0:
        return {
            "cumulative_return": np.nan,
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "average_turnover": np.nan,
        }
    wealth = (1.0 + r).cumprod()
    ann_return = wealth.iloc[-1] ** (trading_days / len(r)) - 1.0
    ann_vol = r.std(ddof=1) * np.sqrt(trading_days)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    avg_turnover = np.nan if turnover is None or len(turnover) == 0 else float(np.mean(turnover))
    return {
        "cumulative_return": float(wealth.iloc[-1] - 1.0),
        "annual_return": float(ann_return),
        "annual_volatility": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown(wealth),
        "average_turnover": avg_turnover,
    }


def format_pct(x):
    if pd.isna(x):
        return "--"
    return f"{100 * x:.2f}\\%"


def format_float(x, digits=4):
    if pd.isna(x):
        return "--"
    return f"{x:.{digits}f}"
