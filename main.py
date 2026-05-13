"""
Market Regime Classification - Data Fetching Script
=====================================================
Fetches financial time series data from Yahoo Finance as outlined in the
project proposal:
  - Daily closing prices (~50 years of history)
  - Realized/historical volatility (rolling standard deviation of log returns)
  - Implied volatility via the VIX index (CBOE Volatility Index)
  - Additional relevant series: volume, market breadth indices, bond yields

Requirements:
    pip install yfinance pandas numpy
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from fredapi import Fred
import torch

fred = Fred(api_key = '6dac8927ae66be817978bd55e16a9241')


data = {
    'unemp': fred.get_series('UNRATE'),
    'cpi': fred.get_series('CPIAUCSL'),
    'gdp': fred.get_series('GDP'),
    'spread': fred.get_series('T10Y2Y'),
    'sp500': fred.get_series('SP500'),
    'vix': fred.get_series('VIXCLS'),
    'baa': fred.get_series('BAA'),
    'aaa': fred.get_series('AAA'),
    'gs3': fred.get_series('GS3'),   # 3-Year Treasury yield for corporate spread
}



# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# Date range: ~50 years back from today
END_DATE = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=365 * 50)).strftime("%Y-%m-%d")

# Core equity indices
EQUITY_TICKERS = [
    "^GSPC",   # S&P 500
    "^DJI",    # Dow Jones Industrial Average
    "^IXIC",   # NASDAQ Composite
    "^RUT",    # Russell 2000 (small cap)
]

# Implied volatility index (proxy for option-implied vol)
VOLATILITY_TICKERS = [
    "^VIX",    # CBOE Volatility Index (implied vol, available from ~1990)
]

# Additional financial time series useful for regime detection
SUPPLEMENTAL_TICKERS = [
    "^TNX",    # 10-Year Treasury Yield
    "^TYX",    # 30-Year Treasury Yield
    "GC=F",    # Gold Futures (risk-off indicator)
    "CL=F",    # Crude Oil Futures
    "DX-Y.NYB", # US Dollar Index
    "HG=F", #Copper
    "HYG", #High Yield Bond ETF
    "^IRX", #13 week T-bill for Yield curve computation
]

OUTPUT_DIR = "market_data"
REALIZED_VOL_WINDOWS = [21, 63, 126]  # ~1 month, 1 quarter, 6 months

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def fetch_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data for a single ticker from Yahoo Finance."""
    print(f"  Fetching {ticker} ...")

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if df.empty:
        print(f"  WARNING: No data returned for {ticker}")
        return pd.DataFrame()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalize column names
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    df.index.name = "date"
    return df


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute daily log returns from a price series."""
    return np.log(prices / prices.shift(1))


def compute_realized_volatility(log_returns: pd.Series, window: int) -> pd.Series:
    """
    Annualized realized (historical) volatility over a rolling window.
    Vol = std(log_returns) * sqrt(252)
    """
    return log_returns.rolling(window).std() * np.sqrt(252)


def build_macro_features(daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    macro = pd.DataFrame(index=daily_index)

    # Unemployment: monthly, released ~30 days after reference month
    unemp = data['unemp'].copy()
    unemp.index = pd.to_datetime(unemp.index)
    unemp_change = unemp.diff()    # 1-month change
    unemp_diff3  = unemp.diff(3)   # 3-month cumulative change
    unemp_lag3   = unemp.shift(3)  # value from 3 months prior
    unemp.index += pd.DateOffset(days=30)
    for s in [unemp_change, unemp_diff3, unemp_lag3]:
        s.index = unemp.index
    macro['unemp']        = unemp.reindex(daily_index, method='ffill')
    macro['unemp_change'] = unemp_change.reindex(daily_index, method='ffill')
    macro['unemp_diff3']  = unemp_diff3.reindex(daily_index, method='ffill')
    macro['unemp_lag3']   = unemp_lag3.reindex(daily_index, method='ffill')

    # CPI: monthly, released ~12 days after reference month
    cpi = data['cpi'].copy()
    cpi.index = pd.to_datetime(cpi.index)
    cpi_yoy      = cpi.pct_change(12) * 100  # year-over-year inflation
    cpi_mom      = cpi.pct_change(1) * 100   # month-over-month inflation
    cpi_yoy_lag3 = cpi_yoy.shift(3)          # YoY inflation from 3 months prior
    cpi.index += pd.DateOffset(days=12)
    for s in [cpi_yoy, cpi_mom, cpi_yoy_lag3]:
        s.index = cpi.index
    macro['cpi']          = cpi.reindex(daily_index, method='ffill')
    macro['cpi_yoy']      = cpi_yoy.reindex(daily_index, method='ffill')
    macro['cpi_mom']      = cpi_mom.reindex(daily_index, method='ffill')
    macro['cpi_yoy_lag3'] = cpi_yoy_lag3.reindex(daily_index, method='ffill')

    # Corporate credit spread: BAA - 3YR Treasury (market data, no publication lag)
    baa = data['baa'].copy()
    baa.index = pd.to_datetime(baa.index)
    gs3 = data['gs3'].copy()
    gs3.index = pd.to_datetime(gs3.index)
    corp_spread       = baa - gs3
    corp_spread_diff1 = corp_spread.diff(1)  # 1-month change in credit stress
    corp_spread_diff3 = corp_spread.diff(3)  # 3-month change in credit stress
    macro['corp_3yr_spread']       = corp_spread.reindex(daily_index, method='ffill')
    macro['corp_3yr_spread_diff1'] = corp_spread_diff1.reindex(daily_index, method='ffill')
    macro['corp_3yr_spread_diff3'] = corp_spread_diff3.reindex(daily_index, method='ffill')

    return macro.ffill()


def build_feature_table(close_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of closing prices (one column per ticker),
    compute log returns and realized volatility for each series.

    Directly use level ticker levels and compute the difference as a feature.
    """


    level_tickers = {"^TNX", "^TYX", "^IRX", "^VIX"}

    features = pd.DataFrame(index=close_prices.index)

    for ticker in close_prices.columns:
        series = close_prices[ticker].dropna()

        if ticker in level_tickers:
            features[f"{ticker}_level"] = series
            features[f":{ticker}_change"] = series.diff()
        else:
            log_ret = compute_log_returns(series)
            features[f"{ticker}_log_return"] = log_ret
            for w in REALIZED_VOL_WINDOWS:
                col = f"{ticker}_realized_vol_{w}d"
                features[col] = compute_realized_volatility(log_ret, w)

    return features


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_tickers = EQUITY_TICKERS + VOLATILITY_TICKERS + SUPPLEMENTAL_TICKERS

    # ── 1. Download raw OHLCV data ──────────────────────────────────────────
    print("\n[1/4] Downloading raw OHLCV data from Yahoo Finance...")
    raw_data: dict[str, pd.DataFrame] = {}
    for ticker in all_tickers:
        df = fetch_ticker(ticker, START_DATE, END_DATE)
        if not df.empty:
            raw_data[ticker] = df

    # ── 2. Build combined closing-price table ───────────────────────────────
    print("\n[2/4] Assembling closing-price table...")
    close_prices = pd.DataFrame({
        ticker: df["close"]
        for ticker, df in raw_data.items()
        if "close" in df.columns
    })
    close_prices.index = pd.to_datetime(close_prices.index)
    close_prices.sort_index(inplace=True)

    # Forward-fill sporadic NaNs (e.g., holidays, missing VIX history)
    close_prices.ffill(inplace=True)

    path_prices = os.path.join(OUTPUT_DIR, "closing_prices.csv")
    close_prices.to_csv(path_prices)
    print(f"  Saved → {path_prices}  ({close_prices.shape[0]} rows × {close_prices.shape[1]} cols)")

    # ── 3. Compute log returns + realized volatility ─────────────────────────
    print("\n[3/4] Computing log returns and realized volatility...")
    features = build_feature_table(close_prices)

    macro = build_macro_features(features.index)
    features = pd.concat([features, macro], axis=1)

    path_features = os.path.join(OUTPUT_DIR, "features.csv")
    features.to_csv(path_features)
    print(f"  Saved → {path_features}  ({features.shape[0]} rows × {features.shape[1]} cols)")

    # ── 4. Save per-ticker full OHLCV CSVs ─────────────────────────────────
    print("\n[4/4] Saving per-ticker OHLCV files...")
    for ticker, df in raw_data.items():
        safe_name = ticker.replace("^", "").replace("=", "_").replace(".", "_")
        path = os.path.join(OUTPUT_DIR, f"{safe_name}_ohlcv.csv")
        df.to_csv(path)
        print(f"  Saved → {path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Data fetch complete.")
    print(f"  Date range  : {close_prices.index.min().date()} → {close_prices.index.max().date()}")
    print(f"  Trading days: {len(close_prices)}")
    print(f"  Tickers     : {list(raw_data.keys())}")
    print(f"  Feature cols: {list(features.columns)}")
    print(f"  Output dir  : ./{OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()