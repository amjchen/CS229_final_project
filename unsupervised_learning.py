import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode as scipy_mode
from sklearn.cluster import KMeans

from config import DataConfig
from data_munging import features, prices, derive_features, remove_outliers, standardize_data

cfg = DataConfig()


def elbow_plot(X: np.ndarray, save_path: str = "market_data/elbow_plot.png") -> None:
    k_range = range(*cfg.kmeans_k_range)
    inertias = []

    print("Running elbow search...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        print(f"  K={k:2d}  inertia={km.inertia_:,.0f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(k_range), inertias, marker="o", linewidth=2)
    ax.axvline(
        x=cfg.kmeans_k, color="red", linestyle="--", alpha=0.7,
        label=f"cfg.kmeans_k = {cfg.kmeans_k}"
    )
    ax.set_xlabel("K (number of clusters)")
    ax.set_ylabel("Inertia (within-cluster SSE)")
    ax.set_title("K-Means Elbow Plot")
    ax.set_xticks(list(k_range))
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"\nElbow plot saved → {save_path}")
    plt.show()


def smooth_labels(labels: np.ndarray, index: pd.DatetimeIndex, window: int) -> np.ndarray:
    """Rolling mode over `window` days to kill spurious single-day regime flips."""
    s = pd.Series(labels, index=index, dtype=float)
    smoothed = s.rolling(window=window, center=True, min_periods=1).apply(
        lambda x: scipy_mode(x, keepdims=True).mode[0]
    )
    return smoothed.astype(int).values


def canonical_order(labels: np.ndarray, km: KMeans, label_cols: list[str]) -> np.ndarray:
    """Remap cluster IDs so label 0 = tightest credit spread → label K-1 = widest."""
    spread_idx = label_cols.index("corp_3yr_spread")
    centroid_spreads = km.cluster_centers_[:, spread_idx]
    rank_order = np.argsort(centroid_spreads)          # ascending spread = ascending risk
    remap = {int(old): int(new) for new, old in enumerate(rank_order)}
    return np.array([remap[l] for l in labels])


def build_labels(k: int = None):
    k = k if k is not None else cfg.kmeans_k

    # Build derived feature matrix (needed for yield_curve_spread etc.)
    df = derive_features(features, prices)

    # Restrict to labeling window (board: Data 2000-2026, avoids sparse pre-2000 macro)
    df = df[df.index >= cfg.kmeans_start_date]

    # Validate that all label cols are present
    missing = [c for c in cfg.kmeans_label_cols if c not in df.columns]
    if missing:
        raise ValueError(f"kmeans_label_cols missing from feature table: {missing}")

    # Cluster on macro/credit features only — full feature set captures short-term
    # market noise that fragments economically coherent cycle regimes.
    df_label = df[cfg.kmeans_label_cols].copy()

    # Drop remaining NaN rows (rolling window warmup at start of series)
    n_before = len(df_label)
    df_label = df_label.dropna()
    print(
        f"After {cfg.kmeans_start_date} filter: {n_before} rows  |  "
        f"after dropna: {len(df_label)} rows  |  effective start: {df_label.index[0].date()}"
    )

    # Winsorize then standardize (on the 2000+ sample only)
    df_label = remove_outliers(df_label)
    X, scaler = standardize_data(df_label)

    # Elbow plot so you can verify K choice before committing
    elbow_plot(X.values)

    # Fit final model
    print(f"\nFitting K-means with K={k} on {list(X.columns)}...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    raw_labels = km.fit_predict(X.values)

    # Canonically order: label 0 = expansion (tight spreads) → label K-1 = recession
    ordered_labels = canonical_order(raw_labels, km, list(X.columns))

    # Smooth with rolling mode to enforce minimum regime duration (~1 month)
    labels = smooth_labels(ordered_labels, df_label.index, cfg.kmeans_smooth_window)

    # Align labels back onto the full feature frame
    df_out = df.loc[df_label.index].copy()
    df_out["regime"] = labels

    n_transitions = (pd.Series(labels).diff().ne(0).sum() - 1)
    print(f"\nRegime distribution (0=expansion → {k-1}=recession):")
    print(pd.Series(labels).value_counts().sort_index().to_string())
    print(f"Total regime transitions: {n_transitions}")

    df_out.to_csv(cfg.labeled_output_path)
    print(f"\nLabeled features saved → {cfg.labeled_output_path}")

    return df_out, km, scaler


if __name__ == "__main__":
    df_labeled, km, scaler = build_labels()
