import pandas as pd
import numpy as np 
import os
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("market_data/features.csv", index_col = "date", parse_dates = True)

def remove_outliers(df, lower = 0.025, upper = 0.975):
    """
    Removing the lower 2.5% and upper 2.5% of the data and calling them outliers
    """
    for col in df.columns:
        lo = df[col].quantile(lower)
        hi = df[col].quantile(upper)
        df[col] = df[col].clip(lo, hi)

    return df


def standardize_data(df):
    """
    Standardizes Data
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, index = df.index, columns = df.columns), scaler

