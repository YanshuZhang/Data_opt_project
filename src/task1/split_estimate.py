from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

TRAIN_START = "2018-01-01"
TRAIN_END = "2021-12-31"
VALID_START = "2022-01-01"
VALID_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2025-12-31"
EPSILON = 1e-6


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    returns = pd.read_csv(
        DATA_DIR / "returns_clean.csv",
        index_col="date",
        parse_dates=True,
    )

    train = returns.loc[TRAIN_START:TRAIN_END]
    valid = returns.loc[VALID_START:VALID_END]
    test = returns.loc[TEST_START:TEST_END]

    mu = train.mean()
    sigma = train.cov()
    sigma_stable = sigma + EPSILON * np.eye(sigma.shape[0])
    sigma_stable = pd.DataFrame(sigma_stable, index=sigma.index, columns=sigma.columns)

    train.to_csv(DATA_DIR / "returns_train.csv")
    valid.to_csv(DATA_DIR / "returns_validation.csv")
    test.to_csv(DATA_DIR / "returns_test.csv")

    mu.to_csv(RESULTS_DIR / "mu_train_daily.csv", header=["mu"])
    sigma.to_csv(RESULTS_DIR / "cov_train_daily.csv")
    sigma_stable.to_csv(RESULTS_DIR / "cov_train_daily_stable.csv")

    split_summary = pd.DataFrame(
        {
            "stage": ["train", "validation", "test"],
            "start_date": [train.index.min(), valid.index.min(), test.index.min()],
            "end_date": [train.index.max(), valid.index.max(), test.index.max()],
            "n_days": [len(train), len(valid), len(test)],
        }
    )
    split_summary.to_csv(RESULTS_DIR / "time_split_summary.csv", index=False)

    print(split_summary)
    print(f"Saved mu with shape {mu.shape} and covariance with shape {sigma.shape}.")


if __name__ == "__main__":
    main()
