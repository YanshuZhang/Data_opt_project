from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MAX_MISSING_RATIO = 0.05
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99


def main() -> None:
    prices = pd.read_csv(
        DATA_DIR / "adjusted_close_prices.csv",
        index_col="date",
        parse_dates=True,
    )

    missing_ratio = prices.isna().mean()
    kept_assets = missing_ratio[missing_ratio <= MAX_MISSING_RATIO].index.tolist()
    prices = prices[kept_assets].ffill().bfill().dropna(axis=0)

    returns = prices.pct_change().dropna(axis=0)

    # Winsorize each asset return series separately.
    lower = returns.quantile(WINSOR_LOWER)
    upper = returns.quantile(WINSOR_UPPER)
    returns_clean = returns.clip(lower=lower, upper=upper, axis=1)

    prices.to_csv(DATA_DIR / "prices_clean.csv")
    returns.to_csv(DATA_DIR / "returns_simple.csv")
    returns_clean.to_csv(DATA_DIR / "returns_clean.csv")

    audit = pd.DataFrame(
        {
            "ticker": missing_ratio.index,
            "missing_ratio": missing_ratio.values,
            "kept": missing_ratio.index.isin(kept_assets),
        }
    )
    audit.to_csv(DATA_DIR / "data_availability_audit_clean.csv", index=False)

    summary = pd.DataFrame(
        {
            "item": [
                "assets_before_cleaning",
                "assets_after_cleaning",
                "price_dates_after_cleaning",
                "return_dates_after_cleaning",
                "max_allowed_missing_ratio",
                "winsor_lower_quantile",
                "winsor_upper_quantile",
            ],
            "value": [
                len(missing_ratio),
                len(kept_assets),
                prices.shape[0],
                returns_clean.shape[0],
                MAX_MISSING_RATIO,
                WINSOR_LOWER,
                WINSOR_UPPER,
            ],
        }
    )
    summary.to_csv(DATA_DIR / "cleaning_summary.csv", index=False)

    print(f"Kept {len(kept_assets)} assets out of {len(missing_ratio)}.")


if __name__ == "__main__":
    main()
