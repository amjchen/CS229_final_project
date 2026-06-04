import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode as scipy_mode
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import DataConfig
from data_munging import features, prices, derive_features, remove_outliers, standardize_data

cfg = DataConfig()


def elbow_plot(X: np.ndarray, save_path: str = "market_data/elbow_plot.png"):
    #Elbow plot development
    k_range = range(cfg.kmeans_k_range[0], cfg.kmeans_k_range[1])
    inertias = []
    silhouette_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        print(f"K={k:2d}inertia = {km.inertia_:,.0f}")
        labels = km.labels_
        s_score = silhouette_score(X, labels)
        silhouette_scores.append(s_score)
        print(f"K={k:2d} silhouette score= {s_score}")

    plt.figure()
    plt.plot(list(k_range), inertias, marker = "o", linewidth = 2)
    plt.axvline(x=cfg.kmeans_k, color = "red", linestyle = "--", alpha = 0.7, label = f"cfg.kmeans_k = {cfg.kmeans_k}")

    plt.xlabel("K (number of clusters)")
    plt.ylabel("Inertia (within-cluster SSE)")
    plt.title("K-Means Elbow Plot")
    plt.xticks(list(k_range))
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def smooth_labels(labels: np.ndarray, index: pd.DatetimeIndex, window: int):
    #Rolling mode over `window` days to kill spurious single-day regime flips 
    s = pd.Series(labels, index = index, dtype=float)

    ss = s.rolling(window=window, center=True, min_periods=1)
    smoothed = ss.apply(lambda x: scipy_mode(x, keepdims=True).mode[0])
    return smoothed.astype(int).values


def canonical_order(labels: np.ndarray, km: KMeans, label_cols: list[str]) -> np.ndarray:
    #Remap cluster IDs so label 0 = tightest credit spread → label K-1 = widest 
    spread_idx = label_cols.index("corp_3yr_spread")
    centroid_spreads = km.cluster_centers_[:, spread_idx]
    rank_order = np.argsort(centroid_spreads)          # ascending spread = ascending risk
    remap = {int(old): int(new) for new, old in enumerate(rank_order)}
    return np.array([remap[l] for l in labels])


def build_labels(k: int = None):
    #Builds the labels from the data frame and then returns the fully labeled data frame
    #The K means model + scaler is also returned
    if k is None:
        k = cfg.kmeans_k

    feature_data = derive_features(features, prices)

    feature_data = feature_data[feature_data.index >= cfg.kmeans_start_date]

    missing_data = []
    for i in cfg.kmeans_label_cols:
        if i not in feature_data.columns:
            missing_data.append(i)
    
    if len(missing_data) > 0:
        raise ValueError(f"kmeans_label_cols missing from feature table: {missing_data}")

    df_label = feature_data[cfg.kmeans_label_cols].copy()

    n_before = len(df_label)
    df_label = df_label.dropna()
    start_date = df_label.index[0].date()
    print(f"After {cfg.kmeans_start_date} filter: {n_before} rows")
    print(f"after dropna: {len(df_label)} rows, effective start date is {start_date}")

    df_label = remove_outliers(df_label)
    X, scaler = standardize_data(df_label)

    elbow_plot(X.values)

    print(f"\nFitting K-means with K={k} on {list(X.columns)}...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    raw_labels = km.fit_predict(X.values)

    ordered_labels = canonical_order(raw_labels, km, list(X.columns))

    labels = smooth_labels(ordered_labels, df_label.index, cfg.kmeans_smooth_window)

    df_out = feature_data.loc[df_label.index].copy()
    df_out["regime"] = labels

    name = pd.Series(labels)
    transitions = name.diff().ne(0) 
    n_transitions = transitions.sum() - 1

    counts = name.value_counts().sort_index()


    print(f"\nRegime distribution (0= expansion(bull market) → {k-1} =contraction (bear market):")
    print(counts)
    print(f"Total regime transitions: {n_transitions}")

    df_out.to_csv(cfg.labeled_output_path)
    print(f"\nLabeled features saved to {cfg.labeled_output_path}")

    return df_out, km, scaler



if __name__ == "__main__":
    df_labeled, km, scaler = build_labels()
