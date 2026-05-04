from pathlib import Path

import pandas as pd
import yfinance as yf


TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "V", "MA",
    "UNH", "JNJ", "PG", "HD", "KO",
    "PEP", "XOM", "CVX", "ABBV", "MRK",
    "BAC", "COST", "WMT", "DIS", "NFLX",
    "ADBE", "CRM", "INTC", "CSCO", "T",
]

START_DATE = "2018-01-01"
END_DATE = "2025-12-31"
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"



def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw = yf.download(
        TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        group_by="column",
        progress=True,
        threads=True,
    )

    # Keep only adjusted close prices.
    prices = raw["Close"].sort_index()
    prices.index.name = "date"
    prices = prices.dropna(axis=0, how="all")

    prices.to_csv(DATA_DIR / "adjusted_close_prices.csv")

    audit = pd.DataFrame(
        {
            "ticker": prices.columns,
            "first_date": [prices[c].first_valid_index() for c in prices.columns],
            "last_date": [prices[c].last_valid_index() for c in prices.columns],
            "n_observations": prices.notna().sum().values,
            "missing_ratio": prices.isna().mean().values,
        }
    )
    audit.to_csv(DATA_DIR / "data_availability_audit_raw.csv", index=False)

    print(f"Saved {prices.shape[0]} dates and {prices.shape[1]} assets.")


if __name__ == "__main__":
    main()
