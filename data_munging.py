import pandas as pd
import numpy as np 
import os
from sklearn.preprocessing import StandardScaler
from config import DataConfig



cfg = DataConfig() 

features = pd.read_csv(cfg.features_path, index_col = "date ", parse_dates = True)
prices = pd.read_csv(cfg.prices_path, index_col = "date", parse_dates = True)

def remove_outliers(df):
    """
    Removing the lower % and upper %% of the data and calling them outliers
    """
    for col in df.columns:
        lo = df[col].quantile(cfg.winsorize_lower)
        hi = df[col].quantile(cfg.winsorize_upper)
        df[col] = df[col].clip(lo, hi)

    return df


def standardize_data(df):
    """
    Standardizes Data
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, index = df.index, columns = df.columns), scaler

def derive_features(features, prices):
    df = features.copy()

    #Yield Curve spread
    df["yield_curve_spread"] = prices["^TNX"] - prices["^IRX"]
    df["ycs_change"] = df["yield_curve_spread"].diff()

    #Volatility Changes
    df["vix_level"]  = prices["^VIX"]
    df["vix_change"] = prices["^VIX"].diff()

    #Copper to Gold ratio
    df["copper_gold_ratio"] = np.log(prices["HG=F"] / prices["GC=F"])
    df["cpr_change"] = df["copper_gold_ratio"].diff()

    #Vol of Vol (standard deviation of the volatility)
    df["vov"] = prices["^VIX"].diff().rolling(21).std()

    #Credit Spreads
    df["credit_spread_return"] = features["HYG_log_return"] - features["TLT_log_return"]

    #Rolling equity-bond correlation
    df["equity_bond_corr_63d"] = (features["^GSPC_log_return"].rolling(63).corr(features["TLT_log_return"]))

    #Equity momentum
    df["equity_momentum_63d"]  = features["^GSPC_log_return"].rolling(63).sum()
    df["equity_momentum_126d"] = features["^GSPC_log_return"].rolling(126).sum()

    #Small cap over large cap spread
    df["sc_lc_spread"] = features["^RUT_log_return"] - features["^GSPC_log_return"]

    return df


