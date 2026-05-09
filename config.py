import os
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DataConfig:
    """Data config + import and cleaning parameters"""
    features_path : str = "market_data/features.csv"
    prices_path : str = "market_data/closing_prices.csv"
    output_path : str = "market_data/model_features.cvs"

    tickers: List[str] = None

    test_days: int = 200
    winsorize_lower: float = 0.005
    winsorize_upper: float = 0.995
