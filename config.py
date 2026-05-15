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
    kmeans_k: int = 6
    kmeans_k_range: tuple = (2, 13)
    kmeans_smooth_window: int = 42  # ~2 month rolling mode to enforce regime persistence
    # K=4 was tried: produced a 106-day "post-COVID snap-back" regime (Jun-Nov 2020) that
    # burned a cluster on an anomalous fiscal/monetary policy environment rather than a
    # structural credit cycle state. K=3 folds it into the recovery regime cleanly.
    #
    # Feature set history:
    #   v1 (macro/credit): corp_3yr_spread, corp_3yr_spread_diff1, unemp, unemp_change,
    #      cpi_yoy, yield_curve_spread -- correctly identified credit crises but missed
    #      shorter equity-specific stress periods (2011, 2015-16, Q4 2018) because unemp
    #      and CPI lag equity markets by months.
    #   v2 (equity-focused, current): swapped lagged macro for real-time equity signals.
    kmeans_label_cols: List[str] = None

    def __post_init__(self):
        if self.kmeans_label_cols is None:
            self.kmeans_label_cols = [
                "corp_3yr_spread",        # credit stress -- forward-looking equity risk signal
                "yield_curve_spread",     # recession expectations (10Y - 3M)
                "vix_level",              # market fear / implied volatility
                "^GSPC_realized_vol_63d", # realized equity volatility (63-day)
                "equity_momentum_126d",   # 6-month equity trend (bull/bear direction)
                "copper_gold_ratio",      # risk appetite: growth vs. safety rotation
            ]

@dataclass
class SupervisedConfig:
    opt_method : str = "L-BFGS-B"
    max_iter: int = 2000
    horizon : int = 21
    lam : float = 1.0
    penalty_type : str = "ce_standard" #Type of penalty we will be appending. Leave empty if we dont want to apply penalty
