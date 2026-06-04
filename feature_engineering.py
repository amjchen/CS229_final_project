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

end_date = datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.today() - timedelta(days=365 * 50)).strftime("%Y-%m-%d")

equity_ticks = ["^GSPC", "^DJI", "^IXIC", "^RUT"]

vol_ticks = ["^VIX"]

macro_misc_ticks = ["^TNX","^TYX","GC=F", "CL=F", "DX-Y.NYB", "HG=F","HYG","^IRX"]

out = "market_data"
real_windows = [21, 63, 126] 

def fetch_ticker(ticker: str, start: str, end: str):
    print(f"Fetching {ticker} ...")

    df = yf.download(ticker, start=start,  end=end,auto_adjust=True, progress=False,group_by="column",)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    df.index.name = "date"
    return df


def compute_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1))


def compute_realized_volatility(log_returns: pd.Series, window: int) -> pd.Series:
    return log_returns.rolling(window).std() * np.sqrt(252)


def build_macro_features(daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    macro = pd.DataFrame(index=daily_index)

    unemp = data['unemp'].copy()
    unemp.index = pd.to_datetime(unemp.index)
    unemp_change = unemp.diff() 
    unemp_diff3 = unemp.diff(3)   
    unemp_lag3 = unemp.shift(3) 
    unemp.index += pd.DateOffset(days=30)


    for s in [unemp_change, unemp_diff3, unemp_lag3]:
        s.index = unemp.index
    macro['unemp'] = unemp.reindex(daily_index, method='ffill')
    macro['unemp_change'] = unemp_change.reindex(daily_index, method='ffill')
    macro['unemp_diff3'] = unemp_diff3.reindex(daily_index, method='ffill')
    macro['unemp_lag3'] = unemp_lag3.reindex(daily_index, method='ffill')

    cpi = data['cpi'].copy()
    cpi.index = pd.to_datetime(cpi.index)
    cpi_yoy = cpi.pct_change(12) * 100
    cpi_mom = cpi.pct_change(1) * 100   
    cpi_yoy_lag3 = cpi_yoy.shift(3)          
    cpi.index += pd.DateOffset(days=12)

    
    for s in [cpi_yoy, cpi_mom, cpi_yoy_lag3]:
        s.index = cpi.index
    macro['cpi'] = cpi.reindex(daily_index, method='ffill')
    macro['cpi_yoy'] = cpi_yoy.reindex(daily_index, method='ffill')
    macro['cpi_mom'] = cpi_mom.reindex(daily_index, method='ffill')
    macro['cpi_yoy_lag3'] = cpi_yoy_lag3.reindex(daily_index, method='ffill')

    baa = data['baa'].copy()
    baa.index = pd.to_datetime(baa.index)
    gs3 = data['gs3'].copy()
    gs3.index = pd.to_datetime(gs3.index)
    corp_spread = baa - gs3
    corp_spread_diff1 = corp_spread.diff(1)  
    corp_spread_diff3 = corp_spread.diff(3)  
    macro['corp_3yr_spread'] = corp_spread.reindex(daily_index, method='ffill')
    macro['corp_3yr_spread_diff1'] = corp_spread_diff1.reindex(daily_index, method='ffill')
    macro['corp_3yr_spread_diff3'] = corp_spread_diff3.reindex(daily_index, method='ffill')

    return macro.ffill()


def build_feature_table(close_prices: pd.DataFrame) -> pd.DataFrame:
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
            for w in real_windows:
                col = f"{ticker}_realized_vol_{w}d"
                features[col] = compute_realized_volatility(log_ret, w)

    return features


def main():
    os.makedirs(out, exist_ok=True)

    all_tickers = equity_ticks + vol_ticks + macro_misc_ticks

    raw_data: dict[str, pd.DataFrame] = {}
    for ticker in all_tickers:
        df = fetch_ticker(ticker, start_date, end_date)
        if not df.empty:
            raw_data[ticker] = df

    close_prices = pd.DataFrame({
        ticker: df["close"]
        for ticker, df in raw_data.items()
        if "close" in df.columns
    })
    close_prices.index = pd.to_datetime(close_prices.index)
    close_prices.sort_index(inplace = True)

    close_prices.ffill(inplace = True)

    path_prices = os.path.join(out, "closing_prices.csv")
    close_prices.to_csv(path_prices)

    features = build_feature_table(close_prices)
    macro = build_macro_features(features.index)
    features = pd.concat([features, macro], axis=1)

    path_features = os.path.join(out, "features.csv")
    features.to_csv(path_features)

    for ticker, df in raw_data.items():
        safe_name = ticker.replace("^", "").replace("=", "_").replace(".", "_")
        path = os.path.join(out, f"{safe_name}_ohlcv.csv")
        df.to_csv(path)
        print(f"Data here {path}")

    print(f"Data Dates: {close_prices.index.min().date()} to {close_prices.index.max().date()}")
    print(f"With our trading days number: {len(close_prices)}")
    print(f"All Tickers: {list(raw_data.keys())}")
    print(f"Transformed Features: {list(features.columns)}")


if __name__ == "__main__":
    main()
