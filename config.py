import os
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DataConfig:
    """Data config + import and cleaning parameters"""
    features_path : str = "market_data/features.csv"
    prices_path : str = "market_data/closing_prices.csv"
    output_path         : str = "market_data/model_features.cvs"
    labeled_output_path : str = "market_data/features_with_labels.csv"

    tickers: List[str] = None

    test_days: int = 200
    winsorize_lower: float = 0.005
    winsorize_upper: float = 0.995

    kmeans_start_date: str = "2000-01-01"
    kmeans_k: int = 3
    kmeans_k_range: tuple = (2, 13)
    kmeans_smooth_window: int = 42  # ~2 month rolling mode to enforce regime persistence
    # K=4 was tried: produced a 106-day "post-COVID snap-back" regime (Jun–Nov 2020) that
    # burned a cluster on an anomalous fiscal/monetary policy environment rather than a
    # structural credit cycle state. K=3 folds it into the recovery regime cleanly.

    # Features used for K-means labeling — macro/credit only to capture cycle structure,
    # not the full feature set (which includes short-term market noise).
    kmeans_label_cols: List[str] = None

    def __post_init__(self):
        if self.kmeans_label_cols is None:
            self.kmeans_label_cols = [
                "corp_3yr_spread",       # credit stress level (primary regime driver)
                "corp_3yr_spread_diff1", # spread momentum
                "unemp",                 # labor market
                "unemp_change",          # labor market turning point
                "cpi_yoy",               # inflation regime
                "yield_curve_spread",    # recession predictor (10Y - 3M)
            ]
