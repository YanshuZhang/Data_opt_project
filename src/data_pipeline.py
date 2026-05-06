import numpy as np
import pandas as pd
import config
from utils import regularize_cov


def download_prices():
    import yfinance as yf

    raw = yf.download(
        tickers=config.TICKERS,
        start=config.START_DATE,
        end=config.END_DATE,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = config.TICKERS[:1]
    prices = prices.sort_index()
    prices.to_csv(config.DATA_DIR / "adjusted_close_prices.csv")
    return prices


def clean_prices_and_returns(prices):
    prices = prices.dropna(axis=1, how="all")
    n_before = prices.shape[1]
    missing_ratio = prices.isna().mean()
    first_date = prices.apply(lambda s: s.first_valid_index())
    last_date = prices.apply(lambda s: s.last_valid_index())
    kept = missing_ratio <= config.MAX_MISSING_RATIO

    audit = pd.DataFrame({
        "ticker": prices.columns,
        "first_valid_date": [str(first_date[c].date()) if first_date[c] is not None else "" for c in prices.columns],
        "last_valid_date": [str(last_date[c].date()) if last_date[c] is not None else "" for c in prices.columns],
        "missing_ratio": missing_ratio.values,
        "kept": kept.values,
    })
    audit.to_csv(config.CSV_DIR / "data_availability_audit.csv", index=False)

    clean_prices = prices.loc[:, kept].ffill().bfill().dropna(axis=0, how="any")
    returns = clean_prices.pct_change().dropna(axis=0, how="any")

    lower = returns.quantile(config.WINSOR_LOWER)
    upper = returns.quantile(config.WINSOR_UPPER)
    returns_clean = returns.clip(lower=lower, upper=upper, axis=1)

    clean_prices.to_csv(config.DATA_DIR / "prices_clean.csv")
    returns.to_csv(config.DATA_DIR / "returns_simple.csv")
    returns_clean.to_csv(config.DATA_DIR / "returns_clean.csv")

    summary = pd.DataFrame({
        "item": [
            "assets_before_cleaning",
            "assets_after_cleaning",
            "trading_days_after_cleaning",
            "return_observations",
            "maximum_missing_ratio_allowed",
        ],
        "value": [
            n_before,
            clean_prices.shape[1],
            clean_prices.shape[0],
            returns_clean.shape[0],
            config.MAX_MISSING_RATIO,
        ],
    })
    summary.to_csv(config.CSV_DIR / "data_cleaning_summary.csv", index=False)
    return clean_prices, returns_clean, audit, summary


def split_returns(returns):
    train = returns.loc[:config.TRAIN_END]
    val = returns.loc[pd.to_datetime(config.TRAIN_END) + pd.Timedelta(days=1):config.VAL_END]
    test = returns.loc[config.TEST_START:]

    train.to_csv(config.DATA_DIR / "returns_train.csv")
    val.to_csv(config.DATA_DIR / "returns_validation.csv")
    test.to_csv(config.DATA_DIR / "returns_test.csv")

    split = pd.DataFrame({
        "stage": ["train", "validation", "test"],
        "start_date": [str(train.index.min().date()), str(val.index.min().date()), str(test.index.min().date())],
        "end_date": [str(train.index.max().date()), str(val.index.max().date()), str(test.index.max().date())],
        "observations": [len(train), len(val), len(test)],
    })
    split.to_csv(config.CSV_DIR / "time_split_summary.csv", index=False)
    return train, val, test, split


def estimate_parameters(train):
    mu = train.mean()
    Sigma = train.cov()
    Sigma_stable = pd.DataFrame(
        regularize_cov(Sigma.values, config.COV_EPS),
        index=Sigma.index,
        columns=Sigma.columns,
    )

    mu.to_csv(config.DATA_DIR / "mu_train.csv", header=["mu"])
    Sigma.to_csv(config.DATA_DIR / "Sigma_train.csv")
    Sigma_stable.to_csv(config.DATA_DIR / "Sigma_train_stable.csv")

    eigvals = np.linalg.eigvalsh(Sigma_stable.values)
    stats = pd.DataFrame({
        "item": ["number_of_assets", "mean_daily_return_average", "mean_daily_volatility", "min_eigenvalue", "condition_number"],
        "value": [
            len(mu),
            float(mu.mean()),
            float(np.sqrt(np.diag(Sigma_stable)).mean()),
            float(eigvals.min()),
            float(eigvals.max() / eigvals.min()),
        ],
    })
    stats.to_csv(config.CSV_DIR / "parameter_estimation_summary.csv", index=False)
    return mu, Sigma_stable, stats
