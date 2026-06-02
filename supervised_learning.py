import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import contextlib
import io
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from data_munging import standardize_data
from sklearn.model_selection import TimeSeriesSplit
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap
from config import DataConfig, SupervisedConfig

cfg = DataConfig()
cfg_sup = SupervisedConfig()
fold_results = []
h = cfg_sup.horizon



def build_supervised_regime_dataset(
    df: pd.DataFrame,
    cfg_sup,
    target_col: str = "regime",
):
    """
    Convert labeled regime dataframe into a supervised learning dataset
    for multinomial logistic regression / softmax regression.

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_labels()
    target_col : str
        Target column to forecast
    add_regime_lags : bool
        Whether to include lagged regime states as predictors

    Returns
    -------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Future regime labels
    supervised_df : pd.DataFrame
        Full aligned dataframe
    """
    horizon = cfg_sup.horizon
    df = df.copy()
    df = df.dropna()

    if cfg_sup.use_window:
        
        window_size = cfg_sup.window_size
        feature_cols = []
        for c in df.columns:
            if c != target_col:
                feature_cols.append(c)
        
        array = df[feature_cols].values
        n = len(df)
        labels = df[target_col]

        split_index = int(n * cfg_sup.train_split)
        train_array = array[:split_index]

        mu = train_array.mean(axis  = 0)
        stdev = train_array.std(axis = 0)
        
        array = (array - mu) / stdev


        x_vals = []
        y_vals = []
        index = []
        current_regime = []

        if cfg_sup.evaluation_metric == 2: 
            last_i = n - horizon
        else:
            last_i = n

        for i in range(window_size - 1, last_i):
            start = i - window_size + 1
            end = i + 1
            window = array[start:end, :]
            x_vals.append(window)

            if cfg_sup.evaluation_metric == 2:
                y_loc = i + horizon
            else:
                y_loc = i

            y_vals.append(int(labels.iloc[y_loc]))
            index.append(df.index[i])
            current_regime.append(labels.iloc[i])

        X = np.array(x_vals)
        y = pd.Series(y_vals, index = index, dtype = int)
        supervised_df = df.loc[index].copy()
        supervised_df["target"] = y.values
        current_regime = pd.Series(current_regime, index=index)


    else: 
        # yesterday's regime
        df["regime_lag_1"] = df[target_col].shift(1)
        # 1 trading week ago
        df["regime_lag_5"] = df[target_col].shift(5)

        # Predict future regime using current information
        if cfg_sup.evaluation_metric == 2:
            df["target"] = df[target_col].shift(-horizon)
        else:
            df["target"] = df[target_col]

        # Drop Nans created from shifting
        supervised_df = df.dropna().copy()

        # For design matrix X
        # remove current regime label + future target label
        drop_cols = [target_col, "target"]
        X = supervised_df.drop(columns=drop_cols)

        # Target
        y = supervised_df["target"].astype(int)
        current_regime = supervised_df[target_col]

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"y distribution:\n{y.value_counts().sort_index()}")

    return X, y, supervised_df, current_regime





def print_results(preds_train, supervised_df, y_train, preds, y_test, cfg_sup):
    y_test_np =  y_test.to_numpy(dtype=np.int64)
    y_train_np = y_train.to_numpy(dtype=np.int64)

    print("\nTRAIN RESULTS")
    print(classification_report(y_train_np, preds_train))

    print("\nTEST RESULTS")
    print(classification_report(y_test_np, preds))

    train_accuracy = (preds_train == y_train_np).mean()
    test_accuracy = (preds == y_test_np).mean()

    m1_acc_tr = None
    m1_acc = None
    m2_acc_tr = None
    m2_acc = None

    # if cfg_sup.evaluation_metric == 1:
    m1_mask_tr = y_train_np[1:] != y_train_np[:-1]
    m1_n_tr    = m1_mask_tr.sum()

    if m1_n_tr > 0: 
        correct_m1_tr = (preds_train[1:][m1_mask_tr] == y_train_np[1:][m1_mask_tr]).sum()
        m1_acc_tr = correct_m1_tr / m1_n_tr
        print(f"Train Metric1 true transitions: {m1_n_tr}")
        print(f"Train Metric1 correctly predicted transitions: {correct_m1_tr}")
        print(f"Train Metric1 transition accuracy: {m1_acc_tr:.4f}")
    else:
        print("No metric1 defined in this training window")

    m1_mask = y_test_np[1:] != y_test_np[:-1]
    m1_n    = m1_mask.sum()

    if m1_n > 0 : 
        correct_m1 = (preds[1:][m1_mask] == y_test_np[1:][m1_mask]).sum()
        m1_acc  = correct_m1/ m1_n
        print(f"Test Metric1 true transitions: {m1_n}")
        print(f"Test Metric1 correctly predicted transitions: {correct_m1}")
        print(f"Test Metric1 transition accuracy: {m1_acc:.4f}")
    else:
        print("No metric 1 defined in this test window")

    # else:
    if cfg_sup.evaluation_metric == 2:
        current_regime_train = supervised_df.loc[y_train.index, "regime"]
        future_regime_train = y_train
        predicted_regime_train = pd.Series(preds_train, index=y_train.index)
        
        transition_mask_train = current_regime_train != future_regime_train
        n_transitions_train = transition_mask_train.sum()

        if n_transitions_train > 0:
            correct_transition_preds_train = (predicted_regime_train[transition_mask_train] == future_regime_train[transition_mask_train]).sum()
            correct_m2_tr = (predicted_regime_train[transition_mask_train] == future_regime_train[transition_mask_train]).sum()
            m2_acc_tr = correct_m2_tr / n_transitions_train
            
            print(f"Train Metric2 true transitions: {n_transitions_train}")
            print(f"Train Metric2 correctly predicted transitions: {correct_m2_tr}")
            print(f"Train Metric2 transition accuracy: {m2_acc_tr:.4f}")
        else:
            print("No Metric 2 defined in this training window")

        current_regime = supervised_df.loc[y_test.index, "regime"] 
        future_regime = y_test 

        predicted_regime = pd.Series(preds, index=y_test.index)

        transition_mask = current_regime != future_regime
        n_transitions = transition_mask.sum()

        if n_transitions > 0:
            correct_m2 = (predicted_regime[transition_mask]== future_regime[transition_mask]).sum()
            m2_acc = correct_m2 / n_transitions
            print(f"Test Metric2 true transitions: {n_transitions}")
            print(f"Test Metric2 correctly predicted transitions: {correct_m2}")
            print(f"Test Metric2 transition accuracy: {m2_acc:.4f}")
        else: 
            print("No Metric 2 defined in this testing window")



    fold_results.append({
        "train_accuracy" : train_accuracy,
        "test_accuracy" : test_accuracy,

        "train_m1" : m1_acc_tr,
        "test_m1" : m1_acc,

        "train_m2": m2_acc_tr,
        "test_m2" : m2_acc,
    })

def CV_for_lambda(current_regime, X, y, model_type):
    n = len(X)
    cutoff = n // 2
    if model_type == "cnn":
        X = X[:cutoff]
    else:
        X = X.iloc[:cutoff]
        
    y = y.iloc[:cutoff]
    current_regime = current_regime.iloc[:cutoff]

    function_split = TimeSeriesSplit(n_splits= cfg_sup.folds)
    scores = defaultdict(list)

    print(f"\n Tuning lambda | model = {model_type} penalty = {cfg_sup.penalty_type}, using {cfg_sup.folds} folds")

    
    for fold, (taidx, teidx) in enumerate(function_split.split(X)):
        if model_type == "cnn":
            X_train_np = X[taidx]
            X_test_np = X[teidx]
        else:
            X_data_cur, scaler = standardize_data(X.iloc[taidx])
            X_train_np = X_data_cur.to_numpy(dtype = np.float64)
            X_test = scaler.transform(X.iloc[teidx])
            X_test_np = X_test.astype(dtype = np.float64)
        
        current_regime_fold = current_regime.iloc[taidx]
        current_regime_test = current_regime.iloc[teidx]
        current_regime_test_np = current_regime_test.to_numpy(dtype = np.int64)

        y_train = y.iloc[taidx]
        y_test = y.iloc[teidx]
        y_train_np = y_train.to_numpy(dtype = np.int64)
        y_test_np = y_test.to_numpy(dtype = np.int64)

        for lmd in cfg_sup.lam_values:
            cfg_sup.lam = lmd
            if model_type == "softmax":
                clf = SoftmaxRegression(current_regime=current_regime_fold, max_iter = cfg_sup.max_iter, eps = 1e-6, k = cfg.kmeans_k)
                clf.fit(X_train_np, y_train_np, learning_rate = cfg_sup.learning_rate, batch_size = cfg_sup.batch_size)
            elif model_type == "gda":
                clf = GDA_SGD(k = cfg.kmeans_k)
                clf.fit(X_train_np, y_train_np)
            elif model_type == "neuralnet":
                x_shape = X_train_np.shape[1]
                clf = Neural_Networks(torch.as_tensor(current_regime_fold.to_numpy().copy()), x_shape, cfg.kmeans_k, cfg_sup)
                clf.setup_nn()
                clf.fit(X_train_np, y_train_np)     
            elif model_type == "cnn":
                dim_in = X_train_np.shape[2]
                window_size = X_train_np.shape[1]
                input_met = torch.as_tensor(current_regime_fold.to_numpy().copy())
                clf = convolutionNN(input_met, dim_in, window_size, cfg.kmeans_k, cfg_sup)
                clf.setup_cnn()
                clf.fit(X_train_np, y_train_np)

            predicted_values = clf.predict(X_test_np)

            m2_indicator = current_regime_test_np != y_test_np
            total_m2 = m2_indicator.sum()
            if total_m2 > 0:
                summed_2 = (predicted_values[m2_indicator] == y_test_np[m2_indicator]).sum()
                m2_score = summed_2 / total_m2
            else:
                m2_score = 0.0
            

            scores[lmd].append(m2_score)
    

    avg_scores = {}
    for lmd, s in scores.items():
        avg_scores[lmd] = np.mean(s)
    
    best_lam = max(avg_scores, key = avg_scores.get)

    print(f"M2 best lambda is {best_lam} with score of {avg_scores[best_lam]:.4f}")
    
    keys = list(avg_scores.keys())
    y_list = list(avg_scores.values())

    plt.figure()
    plt.plot(keys, y_list, marker = 'o', label = "M2 (true transition accuracy)")
    plt.axvline(best_lam, color = "red", linestyle = '--', label = f"best lambda {best_lam}")
    plt.xlabel("Lambda")
    plt.ylabel("M2 Score")
    plt.title(f"CV for lambda for {model_type} using {cfg_sup.penalty_type}")
    plt.legend()
    plt.savefig("cv_lambda.png")
    plt.close()

    return best_lam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--softmax", action = "store_true")
    parser.add_argument("--gda", action="store_true")
    parser.add_argument("--neuralnet", action = "store_true")
    parser.add_argument("--tunelambda", action = "store_true")
    parser.add_argument("--cnn", action = "store_true")
    args = parser.parse_args()

    df_labeled = pd.read_csv(cfg.labeled_output_path, index_col=0, parse_dates=True)
    X, y, supervised_df, current_regime = build_supervised_regime_dataset(df=df_labeled, cfg_sup = cfg_sup)

    if args.tunelambda:
        if args.softmax: 
            model_type = "softmax"
        elif args.gda:
            model_type = "gda"
        elif args.neuralnet:
            model_type = "neuralnet"
        elif args.cnn:
            model_type = "cnn"
        cfg_sup.lam = CV_for_lambda(current_regime, X, y, model_type)
        print("updated cfg_sup.lam to be best lambda from cv = ", cfg_sup.lam)

    if cfg_sup.testing_method == "walk_forward":
        n = len(X)
        minimum_training_set = int(n / 2)
        folds = cfg_sup.folds
        fold_size = (n - minimum_training_set) // folds
        results = {}
        for i in range(folds):
            beg_test_idx = minimum_training_set + i * fold_size
            if i < folds - 1:
                end_test_idx = beg_test_idx + fold_size
            else: 
                end_test_idx = n

            if args.cnn:
                X_train_np = X[:beg_test_idx]
                X_test_np = X[beg_test_idx:end_test_idx]

            else:
                X_fold, scaler = standardize_data(X.iloc[:beg_test_idx])
                X_train_np = X_fold.to_numpy(dtype = np.float64)
                X_test_np = scaler.transform(X.iloc[beg_test_idx:end_test_idx]).astype(np.float64)
            
            y_train = y.iloc[:beg_test_idx]
            y_test = y.iloc[beg_test_idx:end_test_idx]
            y_train_np = y_train.to_numpy(dtype = np.int64)

            print(f"\n===== Fold {i+1} | train [0:{beg_test_idx}]  test [{beg_test_idx}:{end_test_idx}] =====")

            current_regime_fold = current_regime.iloc[:beg_test_idx]
            if args.softmax:
                print("SOFTMAX REGRESSION")
                clf = SoftmaxRegression(current_regime=current_regime_fold, max_iter=cfg_sup.max_iter, eps=1e-6, k=cfg.kmeans_k)
                clf.fit(X_train_np, y_train_np, learning_rate=cfg_sup.learning_rate, batch_size=cfg_sup.batch_size)
                preds_train = clf.predict(X_train_np)
                preds = clf.predict(X_test_np)
                pred_proba = clf.predict_proba(X_test_np)
                print_results(preds_train, supervised_df, y_train, preds, y_test, cfg_sup)

            if args.gda:
                print("GAUSSIAN DISCRIMINANT ANALYSIS")
                clf = GDA_SGD(k=cfg.kmeans_k)
                clf.fit(X_train_np, y_train_np)
                preds_train = clf.predict(X_train_np)
                preds = clf.predict(X_test_np)
                print_results(preds_train, supervised_df, y_train, preds, y_test, cfg_sup)

            if args.neuralnet:
                print("NEURAL NETWORK")
                dim_in = X_train_np.shape[1]
                k = cfg.kmeans_k
                clf = Neural_Networks(torch.as_tensor(current_regime_fold.to_numpy().copy()), dim_in, k, cfg_sup)
                clf.setup_nn()

                clf.fit(X_train_np, y_train_np)

                preds_train = clf.predict(X_train_np)
                preds = clf.predict(X_test_np)
                pred_proba = clf.predict_proba(X_test_np)
                print_results(preds_train, supervised_df, y_train, preds, y_test, cfg_sup)

            if args.cnn: 
                print("CONVOLUTION NEURAL NETWORK")
                dim_in = X_train_np.shape[2]
                window_size = X_train_np.shape[1]
                k = cfg.kmeans_k

                clf = convolutionNN(torch.as_tensor(current_regime_fold.to_numpy().copy()), dim_in, window_size, k, cfg_sup)
                clf.setup_cnn()
                clf.fit(X_train_np, y_train_np)
                preds_train = clf.predict(X_train_np)
                preds = clf.predict(X_test_np)
                pred_proba = clf.predict_proba(X_test_np)
                print_results(preds_train, supervised_df, y_train, preds, y_test, cfg_sup)
            results[f"fold_{i}"] = {"true" : y_test.to_numpy(dtype=np.int64), "pred" : preds, "pred_proba" : pred_proba}
        y_true_all = [item for f in results.values() for item in f["true"]]
        y_pred_all = [item for f in results.values() for item in f["pred"]]
        # y_pred_proba_all = [item for f in results.values() for item in f["pred_proba"]]
        y_pred_proba_all = np.vstack([f["pred_proba"] for f in results.values()])
        print(len(y_true_all), len(y_pred_all), y_pred_proba_all.shape)
        
        train_sum = 0
        test_sum = 0

        train_m1_sum = 0
        test_m1_sum = 0

        train_m2_sum = 0
        test_m2_sum = 0

        c1, c2, c3, c4, c5, c6 = 0, 0, 0, 0, 0, 0

        for fold in fold_results: 
            if fold['train_accuracy'] is not None:
                train_sum += fold["train_accuracy"]
                c1 += 1
            
            if fold["test_accuracy"] is not None:
                test_sum += fold["test_accuracy"]
                c2 += 1
            
            if fold["train_m1"] is not None: 
                train_m1_sum += fold["train_m1"]
                c3 += 1
            
            if fold["test_m1"] is not None:
                test_m1_sum += fold["test_m1"]
                c4 += 1 

            if fold["train_m2"] is not None:
                train_m2_sum += fold["train_m2"]
                c5 += 1
            
            if fold["test_m2"] is not None:
                test_m2_sum += fold["test_m2"]
                c6 += 1
            
        print(f"\n===== WALK-FORWARD SUMMARY ({folds} folds) =====")
        if c1 > 0: 
            print(f"Avg train accuracy : {(train_sum / c1):.4f}")
        if c2 > 0:
            print(f"Avg test accuracy  : {(test_sum / c2):.4f}")
        
        # if cfg_sup.evaluation_metric == 1: 
        if c3 > 0: 
            print(f"Avg train M1       : {(train_m1_sum / c3):.4f}")
        if c4 > 0:
            print(f"Avg test M1        : {(test_m1_sum / c4):.4f}")

        # else: 
            if c5 > 0:
                print(f"Avg train M2       : {(train_m2_sum / c5):.4f}")
            if c6 > 0:
                print(f"Avg test M2        : {(test_m2_sum / c6):.4f}")
        print(classification_report(y_true_all, y_pred_all))
        cm = confusion_matrix(y_true_all, y_pred_all, labels = [0, 1, 2])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bull', 'Recovery', 'Crisis'])
        
        # plt.show()
        disp.plot()
        plt.savefig('my_plot.png', dpi=300, bbox_inches='tight')

        # plots
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        plot_regime_timeline(
            y_true_all,
            y_pred_all,
            y_pred_proba_all,
            save_path="regime_timeline.png"
        )

    else: 
        split_idx = int(len(X) * cfg_sup.train_split)

        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        y_train_np = y_train.to_numpy(dtype=np.int64)

        if args.cnn:
            X_train_np = X[:split_idx]
            X_test_np = X[split_idx:]
        else: 
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            X_train, scaler = standardize_data(X_train)
            X_train_np = X_train.to_numpy(dtype=np.float64)
            X_test_np = scaler.transform(X_test)
            X_test_np = X_test_np.astype(np.float64)

        if args.softmax:
            print("SOFTMAX REGRESSION")
            clf = SoftmaxRegression(max_iter=cfg_sup.max_iter, eps=1e-6, k=cfg.kmeans_k)
            clf.fit(X_train_np, y_train_np, learning_rate=cfg_sup.learning_rate, batch_size=cfg_sup.batch_size)
            preds_train = clf.predict(X_train_np)
            preds = clf.predict(X_test_np)
            print_results(preds_train, supervised_df, y_train, preds, y_test, cfg_sup)

        if args.gda:
            print("GAUSSIAN DISCRIMINANT ANALYSIS")
            clf = GDA_SGD(k=cfg.kmeans_k)
            clf.fit(X_train_np, y_train_np)
            preds_train = clf.predict(X_train_np)
            preds = clf.predict(X_test_np)
            print_results(preds_train, supervised_df, y_train, preds, y_test, cfg_sup)

        if args.neuralnet:
            print("NEURAL NETWORK")
            dim_in = X_train_np.shape[1]
            k = cfg.kmeans_k
            clf = Neural_Networks(dim_in, k, cfg_sup)
            clf.setup_nn()

            clf.fit(X_train_np, y_train_np)

            preds_train = clf.predict(X_train_np)
            preds = clf.predict(X_test_np)

            print_results(preds_train, supervised_df, y_train, preds, y_test, cfg_sup)
        
        if args.cnn: 
            print("CONVOLUTION NEURAL NETWORK")
            dim_in = X_train_np.shape[2]
            window_size = X_train_np.shape[1]
            k = cfg.kmeans_k

            clf = convolutionNN(dim_in, window_size, k, cfg_sup)
            clf.setup_cnn()
            clf.fit(X_train_np, y_train_np)
            preds_train = clf.predict(X_train_np)
            preds = clf.predict(X_test_np)
            print_results(preds_train, supervised_df, y_train, preds, y_test, cfg_sup)
        cm = confusion_matrix(y_test, preds, labels = [0, 1, 2])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bull', 'Recovery', 'Crisis'])
        # plt.show()
        disp.plot()
        plt.savefig('my_plot.png', dpi=300, bbox_inches='tight')



def plot_regime_timeline(y_true, y_pred, pred_proba, save_path=None):
    """
    Parameters
    ----------
    y_true : (N,)
    y_pred : (N,)
    pred_proba : (N,3)

    Creates 5 stacked number-line style plots:
        True regime
        Predicted regime
        P(Bull)
        P(Recovery)
        P(Crisis)
    """

    N = len(y_true)
    bull_color = "#2ca02c"      # green
    rec_color = "#ffbf00"       # gold
    crisis_color = "#d62728"    # red

    regime_colors = np.array([
        plt.matplotlib.colors.to_rgb(bull_color),
        plt.matplotlib.colors.to_rgb(rec_color),
        plt.matplotlib.colors.to_rgb(crisis_color),
    ])

    true_strip = regime_colors[y_true]
    pred_strip = regime_colors[y_pred]

    bull_cmap = LinearSegmentedColormap.from_list(
        "bull_prob",
        ["white", bull_color]
    )
    rec_cmap = LinearSegmentedColormap.from_list(
        "rec_prob",
        ["white", rec_color]
    )
    crisis_cmap = LinearSegmentedColormap.from_list(
        "crisis_prob",
        ["white", crisis_color]
    )

    fig, axes = plt.subplots(5, 1, figsize=(18, 5), sharex=True, gridspec_kw={"hspace": 0.15})

    # "true" regimes
    axes[0].imshow(
        true_strip[np.newaxis, :, :],
        aspect="auto"
    )
    axes[0].set_ylabel("True")

    # predicted labels
    axes[1].imshow(
        pred_strip[np.newaxis, :, :],
        aspect="auto"
    )
    axes[1].set_ylabel("Pred")

    # bull proba
    axes[2].imshow(
        pred_proba[:, 0][np.newaxis, :],
        aspect="auto",
        cmap=bull_cmap,
        vmin=0,
        vmax=1
    )
    axes[2].set_ylabel("Bull")

    # recovery proba
    axes[3].imshow(
        pred_proba[:, 1][np.newaxis, :],
        aspect="auto",
        cmap=rec_cmap,
        vmin=0,
        vmax=1
    )
    axes[3].set_ylabel("Recovery")

    # crisis proba
    axes[4].imshow(
        pred_proba[:, 2][np.newaxis, :],
        aspect="auto",
        cmap=crisis_cmap,
        vmin=0,
        vmax=1
    )
    axes[4].set_ylabel("Crisis")

    for ax in axes:
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    axes[-1].set_xlabel("Time")

    plt.suptitle("Regime Predictions and Class Probabilities")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

class SoftmaxRegression:
    """Softmax regression with SGD as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, current_regime, max_iter=1000000, eps=1e-6,
                 theta_0=None, verbose=True, k=None):
        """
        Args:
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.k = k
        self.loss_history = []
        self.current_regime = current_regime

    def fit(self, x, y, learning_rate, batch_size):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Shape (n_examples, dim).
            y: Shape (n_examples,).
        """

        x_inter = self.add_intercept(x.copy())
        d = x_inter.shape[1]
        self.current_regime = np.asarray(self.current_regime)

        if self.theta is None:
            self.theta = np.zeros((d, self.k))

        for epoch in range(self.max_iter):
            self.gradient_descent_epoch(x_inter, y, learning_rate, batch_size)

            # Track loss
            loss = self.ce_loss(x_inter, y)
            self.loss_history.append(loss)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

            if epoch > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.eps:
                break

    @staticmethod
    def add_intercept(x):
        """Add intercept to matrix x.

        Args:
            x: 2D NumPy array.

        Returns:
            New matrix same as x with 1's in the 0th column.
        """
        new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
        new_x[:, 0] = 1
        new_x[:, 1:] = x

        return new_x

    def ce_loss(self, X, y):
        n = X.shape[0]
        logits = X @ self.theta
        prob = self.softmax(logits)

        log_loss = np.log(prob[np.arange(n), y] + 1e-12)
        loss = -log_loss.sum()

        # # transition_indicator = (y[h:] != y[:-h]).astype(float)
        transition_indicator = (self.current_regime != y).astype(float)# .to_numpy()
        if cfg_sup.penalty_type == "ce_standard":
            penalty = -cfg_sup.lam * (transition_indicator * log_loss).sum()
        # elif cfg_sup.penalty_type == "ce_pairwise":
        #     pair = log_loss[:-h]
        #     penalty = -cfg_sup.lam * (transition_indicator * (log_loss[h:] + log_loss[:-h])).sum()
        elif cfg_sup.penalty_type == "margin_bar":
            # diff_today_yesterday = np.zeros(n-h)
            diff_today_yesterday = np.zeros(n)
            # for k in range(h, n):
            for k in range(n):
                # yesterday = y[k-h]
                # today = y[k]
                current = self.current_regime[k]
                future = y[k]  
                # diff_today_yesterday[k-h] = cfg_sup.margin + prob[k, current] - prob[k, future]
                diff_today_yesterday[k] = cfg_sup.margin + prob[k, current] - prob[k, future]
            penalty = cfg_sup.lam * (transition_indicator * np.log(1 + np.exp(diff_today_yesterday))).sum()
        elif cfg_sup.penalty_type == "distance":
            # inside = np.zeros(n-h)
            inside = np.zeros(n)
            for i in range(n): #-h):
                for k in range(cfg.kmeans_k):
                    inside[i] += ((y[i] - k) ** 2) * prob[i, k] #((y[i+h] - k) ** 2) * prob[i+h, k]
            penalty = cfg_sup.lam *(transition_indicator * inside).sum()
        else:
            penalty = 0

        # reg = cfg_sup.l2reg_stren * np.linalg.norm(self.theta, "fro")**2
        reg = cfg_sup.l2reg_stren * np.sum(self.theta[1:] ** 2)

        return loss + penalty + reg

    def ce_grad(self, X, y, current_regime):
        n = X.shape[0]
        logits = X @ self.theta
        prob = self.softmax(logits)

        one_hot = self.one_hot(y)

        dZ = prob - one_hot
        dTheta = X.T @ dZ

        # transition_indicator = (y[h:] != y[:-h]).astype(float)
        transition_indicator = (current_regime != y).astype(float)
        if cfg_sup.penalty_type == "ce_standard":
            dZ_trans = np.zeros((n, self.k))
            dZ_trans = transition_indicator[:, None] * (prob - one_hot)
            penalty_grad = cfg_sup.lam * X.T @ dZ_trans
        # elif cfg_sup.penalty_type == "ce_pairwise":
        #     term1 = X[h:].T @ (transition_indicator[:, None] * cfg_sup.lam * (prob[h:] - one_hot[h:]))
        #     term2 = X[:-h].T @ (transition_indicator[:, None] * cfg_sup.lam * (prob[:-h] - one_hot[:-h]))
        #     penalty_grad = term1 + term2
        elif cfg_sup.penalty_type == "margin_bar":
            penalty_dZ = np.zeros_like(prob)

            for i in range(n):
                if not transition_indicator[i]:
                    continue

                c = current_regime[i]
                yi = y[i]
                pc = prob[i, c]
                py = prob[i, yi]

                s = cfg_sup.margin + pc - py

                sigmoid = 1 / (1+np.exp(-s))
                coeff = cfg_sup.lam * sigmoid
                
                grad_s = -(pc - py) * prob[i]
                grad_s[c] += pc
                grad_s[yi] -= py

                penalty_dZ[i] = coeff * grad_s

            penalty_grad = X.T @ penalty_dZ
        elif cfg_sup.penalty_type == "distance":
            inside = np.zeros(n)# np.zeros(n-h)
            for i in range(n): #(n-h):
                for k in range(cfg.kmeans_k):
                    inside[i] += ((y[i] - k) ** 2) * prob[i, k]# ((y[i+h] - k) ** 2) * prob[i+h, k]

            penalty_grad = np.zeros((X.shape[1], self.k))
            one_hot_y = self.one_hot(y)
            c = np.zeros(cfg.kmeans_k)
            for k in range(cfg.kmeans_k):
                c[k] = k

            for i in range(n): # - h):
                if transition_indicator[i]:
                    for k in range(self.k):
                        ctyi = np.dot(c, one_hot_y[i]) #+ h])
                        weight = cfg_sup.lam * prob[i, k] *((ctyi - c[k]) ** 2 - inside[i])  #+ h, k] *((ctyi - c[k]) ** 2 - inside[i])
                        penalty_grad[:, k] += weight * X[i] #+h]
            
        else:
            penalty_grad = 0

        # reg_grad = 2 * cfg_sup.l2reg_stren * self.theta
        reg_grad = np.zeros_like(self.theta)
        reg_grad[1:] = 2 * cfg_sup.l2reg_stren * self.theta[1:]
        return dTheta + penalty_grad + reg_grad

    def gradient_descent_epoch(self, X_shuffled, y_shuffled, learning_rate, batch_size):
        """
        Perform one epoch of gradient descent on the given training data using the provided learning rate.

        This code should update the parameters stored in params.
        It should not return anything

        Args:
            train_data: A numpy array containing the training data
            one_hot_train_labels: A numpy array containing the one-hot embeddings of the training labels e_y.
            learning_rate: The learning rate
            batch_size: The amount of items to process in each batch
            params: A dict of parameter names to parameter values that should be updated.
            forward_prop_func: A function that follows the forward_prop API
            backward_prop_func: A function that follows the backwards_prop API

        Returns: This function returns nothing.
        """

        n_samples = X_shuffled.shape[0]

        for start_idx in range(0, n_samples, batch_size):
            end_idx = start_idx + batch_size

            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            batch_size_actual = len(X_batch)
            current_regime = self.current_regime[start_idx:start_idx+batch_size_actual]

            dTheta = self.ce_grad(X_batch, y_batch, current_regime) / batch_size_actual
            self.theta -= learning_rate * dTheta

        return

    def one_hot(self, y):
        """
        Convert integer labels into one-hot vectors.
        """

        one_hot = np.zeros((len(y), self.k))
        one_hot[np.arange(len(y)), y] = 1

        return one_hot

    def softmax(self, z):
        """
        Compute softmax function for a batch of input values.
        The first dimension of the input corresponds to the batch size. The second dimension
        corresponds to every class in the output. When implementing softmax, you should be careful
        to only sum over the second dimension.

        Important Note: You must be careful to avoid overflow for this function. Functions
        like softmax have a tendency to overflow when very large numbers like e^10000 are computed.
        You will know that your function is overflow resistent when it can handle input like:
        np.array([[10000, 10010, 10]]) without issues.

        Args:
            x: A 2d numpy float array of shape batch_size x number_of_classes

        Returns:
            A 2d numpy float array containing the softmax results of shape batch_size x number_of_classes
        """
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exp_shift_z = np.exp(shift_z)
        return exp_shift_z / np.sum(exp_shift_z, axis=1, keepdims=True)

    def predict(self, x):
        x_inter = self.add_intercept(x.copy())
        logits = x_inter @ self.theta
        return np.argmax(self.softmax(logits), axis=1)
    
    def predict_proba(self, x):
        x_inter = self.add_intercept(x.copy())
        logits = x_inter @ self.theta
        return self.softmax(logits)


class GDA_MLE:          # i think this has some stability issues rn, but maybe we wont even use this 
    """Gaussian Discriminant Analysis
    y ~ Multinomial(phi)
    x | y = j ~ N(mu_j, Sigma_j) NOTE we allow different class to have different covariance 

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, verbose=True):
        """
        Args:
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            verbose: Print fitted parameters after training.
        """
        self.sigma = None
        self.mu = None
        self.phi = None
        self.k = cfg.kmeans_k
        
        # self.max_iter = max_iter
        # self.eps = eps
        # self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Shape (n_examples, dim).
            y: Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n = x.shape[0]
        d = x.shape[1]

        self.phi = np.zeros(self.k)
        self.mu = np.zeros((self.k, d))
        self.sigma = np.zeros((self.k, d, d))
        for j in range(self.k):
            x_j = x[y == j]
            n_j = len(x_j)
            
            self.phi[j] = n_j / n
            
            self.mu[j] = np.mean(x_j, axis=0)
            
            diff = x_j - self.mu[j]
            self.sigma[j] = (diff.T @ diff) / n_j

        self.cov_inv = []
        self.cov_det = []

        for sigma in self.sigma:
            self.cov_inv.append(np.linalg.inv(sigma))
            self.cov_det.append(np.linalg.det(sigma))
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Predict class that has maximum posterior P(y=j | x) = P(x|y=j)P(y=j) / P(x) 
        # <==> class with max log P(x|y=j) + log P(y=j)
        n = x.shape[0]
        d = x.shape[1]

        log_posteriors = np.zeros((n, self.k))

        for j in range(self.k):
            mu = self.mu[j]
            sigma_inv = self.cov_inv[j]
            sigma_det = self.cov_det[j]

            diff = x - mu  # shape: (n, d)

            exponent = -0.5 * np.sum((diff @ sigma_inv) * diff, axis=1)
            log_coeff = -0.5 * (
                d * np.log(2 * np.pi) + np.log(sigma_det)
            )
            log_likelihood = log_coeff + exponent

            log_posteriors[:, j] = log_likelihood + np.log(self.phi[j])

        return np.argmax(log_posteriors, axis=1)
        # *** END CODE HERE ***


class Neural_Networks(nn.Module):
    def __init__(self, current_regime, dim_in, k, cfg_sup):
        super().__init__()

        self.dim_in = dim_in
        self.k = k
        self.cfg_sup = cfg_sup
        self.current_regime = current_regime

    def setup_nn(self):
        layers = []
        last_d = self.dim_in

        for dim in self.cfg_sup.hidden_layer_neurons:
            MM = nn.Linear(last_d, dim)
            layers.append(MM)

            activation = nn.ReLU()
            layers.append(activation)

            dropout = nn.Dropout(self.cfg_sup.dropout_prob)
            layers.append(dropout)

            last_d = dim

        MM_out = nn.Linear(last_d, self.k)
        layers.append(MM_out)

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
    def loss_function(self, score, y): 
        n = score.shape[0]

        #Changing logit scores to probabilites through softmax
        numerator = torch.exp(score)
        denominator = numerator.sum(dim = 1, keepdim= True)
        prob = numerator / denominator

        ground_lab = []
        for i in range(n):
            ground_lab.append(prob[i, y[i]])
        ground_lab = torch.stack(ground_lab)

        log_loss = torch.log(ground_lab + 1e-12)
        loss = -log_loss.sum()

        # transition_indicator = (y[h:] != y[:-h]).float()
        # print(self.current_regime.shape, y.shape)
        transition_indicator = (self.current_regime != y).float()
        
        if self.cfg_sup.penalty_type == "ce_standard":
            penalty = -self.cfg_sup.lam * (transition_indicator * log_loss).sum()
        # elif self.cfg_sup.penalty_type == "ce_pairwise":
        #     penalty = -self.cfg_sup.lam * (transition_indicator * (log_loss[h:] + log_loss[:-h])).sum()
        elif self.cfg_sup.penalty_type == "margin_bar":
            diff_today_yesterday = []

            for k in range(n): #(h, n):
                # yest_prb = prob[k, y[k-h]]
                # today_prb = prob[k, y[k]]
                current_prb = prob[k, self.current_regime[k]]
                future_prb = prob[k, y[k]]

                diff = self.cfg_sup.margin + current_prb - future_prb # yest_prb - today_prb
                diff_today_yesterday.append(diff)

            stacked_diff = torch.stack(diff_today_yesterday)
            penalty = self.cfg_sup.lam * (transition_indicator * (torch.log(1 + torch.exp(stacked_diff)))).sum()

        elif self.cfg_sup.penalty_type == "distance":
            inside = []
            for i in range(n): # -h):
                temp = 0.0
                for k in range(self.k):
                    temp += (y[i] - k) ** 2 * prob[i,k]  # + h] - k) ** 2 * prob[i+h,k]
                
                inside.append(temp)
            
            inside = torch.stack(inside)
            penalty = self.cfg_sup.lam * (transition_indicator * inside).sum()

        else:
            penalty = 0

        return loss + penalty

    def fit(self, x, y):
        self.train()
        X_matrix = torch.tensor(x, dtype = torch.float32)
        y_vec = torch.tensor(y, dtype = torch.long)

        optimizer = torch.optim.AdamW(self.parameters(), lr = self.cfg_sup.learning_rate, weight_decay = self.cfg_sup.weight_decay)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                                             start_factor = 0.01, 
                                                             end_factor=1.0, 
                                                             total_iters = self.cfg_sup.warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                      T_max = self.cfg_sup.epochs - self.cfg_sup.warmup_epochs, 
                                                                      eta_min = self.cfg_sup.min_lr)

        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones = [self.cfg_sup.warmup_epochs])

        for epoch in range(self.cfg_sup.epochs):
            optimizer.zero_grad()
            score = self.forward(X_matrix)
            loss = self.loss_function(score, y_vec)
            loss.backward()
            optimizer.step()
            scheduler.step()
    
    def predict_proba(self, x_new):
        self.eval()

        with torch.no_grad():
            X_mat_new = torch.tensor(x_new, dtype=torch.float32)

            score = self.forward(X_mat_new)
            max_score = torch.max(score, dim=1, keepdim=True).values

            numerator = torch.exp(score - max_score)
            denominator = numerator.sum(dim=1, keepdim=True)

            softmax_x_prob = numerator / denominator
            return softmax_x_prob.cpu().numpy()

    def predict(self, x_new):
        self.eval()
        X_mat_new = torch.tensor(x_new, dtype = torch.float32)

        score = self.forward(X_mat_new)

        numerator = torch.exp(score)
        denominator = numerator.sum(dim = 1, keepdim = True)
        softmax_x_prob = numerator / denominator

        predicted_y = torch.argmax(softmax_x_prob, dim = 1)
        y_numpy = predicted_y.detach().numpy()
    
        return y_numpy

class convolutionNN(nn.Module):
    def __init__(self, current_regime, dim_in, window_size, k, cfg_sup):
        super().__init__()

        self.dim_in = dim_in
        self.window_size = window_size
        self.k = k
        self.cfg_sup = cfg_sup
        self.current_regime = current_regime
    
    def setup_cnn(self):
        self.conv1 = nn.Conv1d(in_channels= self.dim_in, out_channels= self.cfg_sup.num_filters, kernel_size= self.cfg_sup.kernel_size)
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(self.cfg_sup.dropout_prob)
        output_dim = self.window_size - self.cfg_sup.kernel_size + 1
        self.lin_out = nn.Linear(self.cfg_sup.num_filters * output_dim, self.k)
    
    def forward(self, x):
        #Based on the docstring I think the input for nnconv1d needs (n, channels, length)
        #Currently the data is stored with dimension (n_windows, window lenght, features)

        x = x.transpose(1,2)

        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim= 1)
        score = self.lin_out(x)
        return score
    
    def loss_function(self, score, y): 
        n = score.shape[0]

        #Changing logit scores to probabilites through softmax
        numerator = torch.exp(score)
        denominator = numerator.sum(dim = 1, keepdim= True)
        prob = numerator / denominator

        ground_lab = []
        for i in range(n):
            ground_lab.append(prob[i, y[i]])
        ground_lab = torch.stack(ground_lab)

        log_loss = torch.log(ground_lab + 1e-12)
        loss = -log_loss.sum()

        # transition_indicator = (y[h:] != y[:-h]).float()
        # print(self.current_regime.shape, y.shape)
        transition_indicator = (self.current_regime != y).float()
        
        if self.cfg_sup.penalty_type == "ce_standard":
            penalty = -self.cfg_sup.lam * (transition_indicator * log_loss).sum()
        # elif self.cfg_sup.penalty_type == "ce_pairwise":
        #     penalty = -self.cfg_sup.lam * (transition_indicator * (log_loss[h:] + log_loss[:-h])).sum()
        elif self.cfg_sup.penalty_type == "margin_bar":
            diff_today_yesterday = []

            for k in range(n): #(h, n):
                # yest_prb = prob[k, y[k-h]]
                # today_prb = prob[k, y[k]]
                current_prb = prob[k, self.current_regime[k]]
                future_prb = prob[k, y[k]]

                diff = self.cfg_sup.margin + current_prb - future_prb # yest_prb - today_prb
                diff_today_yesterday.append(diff)

            stacked_diff = torch.stack(diff_today_yesterday)
            penalty = self.cfg_sup.lam * (transition_indicator * (torch.log(1 + torch.exp(stacked_diff)))).sum()

        elif self.cfg_sup.penalty_type == "distance":
            inside = []
            for i in range(n): # -h):
                temp = 0.0
                for k in range(self.k):
                    temp += (y[i] - k) ** 2 * prob[i,k]  # + h] - k) ** 2 * prob[i+h,k]
                
                inside.append(temp)
            
            inside = torch.stack(inside)
            penalty = self.cfg_sup.lam * (transition_indicator * inside).sum()

        else:
            penalty = 0

        return loss + penalty

    def fit(self, x, y):
        self.train()
        X_matrix = torch.tensor(x, dtype = torch.float32)
        y_vec = torch.tensor(y, dtype = torch.long)

        optimizer = torch.optim.AdamW(self.parameters(), lr = self.cfg_sup.learning_rate, weight_decay = self.cfg_sup.weight_decay)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                                             start_factor = 0.01, 
                                                             end_factor=1.0, 
                                                             total_iters = self.cfg_sup.warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                      T_max = self.cfg_sup.epochs - self.cfg_sup.warmup_epochs, 
                                                                      eta_min = self.cfg_sup.min_lr)

        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones = [self.cfg_sup.warmup_epochs])

        for epoch in range(self.cfg_sup.epochs):
            optimizer.zero_grad()
            score = self.forward(X_matrix)
            loss = self.loss_function(score, y_vec)
            loss.backward()
            optimizer.step()
            scheduler.step()

    def predict_proba(self, x_new):
        self.eval()

        with torch.no_grad():
            X_mat_new = torch.tensor(x_new, dtype=torch.float32)

            score = self.forward(X_mat_new)
            max_score = torch.max(score, dim=1, keepdim=True).values

            numerator = torch.exp(score - max_score)
            denominator = numerator.sum(dim=1, keepdim=True)

            softmax_x_prob = numerator / denominator
            return softmax_x_prob.cpu().numpy()
    
    def predict(self, x_new):
        self.eval()
        X_mat_new = torch.tensor(x_new, dtype = torch.float32)

        score = self.forward(X_mat_new)

        numerator = torch.exp(score)
        denominator = numerator.sum(dim = 1, keepdim = True)
        softmax_x_prob = numerator / denominator

        predicted_y = torch.argmax(softmax_x_prob, dim = 1)
        y_numpy = predicted_y.detach().numpy()
    
        return y_numpy



# class GDA_SGD:
#     """
#     Gaussian Discriminant Analysis trained with mini-batch SGD
#     on the negative log likelihood.

#     Parameters optimized:
#         mu_j              : class means
#         L_j               : Cholesky factor
#                              S_j = L_j L_j^T
#         phi_j             : class priors

#     We optimize wrt L_j instead of S_j directly so that
#     precision matrices remain positive definite automatically.

#     ============================================================
#     GRADIENTS USED
#     ============================================================
#     Let:
#         S_j = L_j L_j^T     (cholesky)
#         diff_i = x_i - mu_j

#     NLL for class j:
#         L_j =
#             - n_j log(phi_j)
#             - 0.5 n_j log|S_j|
#             + 0.5 sum diff_i^T S_j diff_i

#     ------------------------------------------------------------
#     Gradient wrt mu_j
#     ------------------------------------------------------------
#         dL/dmu_j = - sum S_j (x_i - mu_j)
#     -----------------------------------------------------------
#     Gradient wrt S_j
#     ------------------------------------------------------------
#         dL/dS_j = 0.5 * (
#                 scatter - n_j S_j^{-1}
#               )
#         where scatter = sum diff_i diff_i^T

#     ------------------------------------------------------------
#     Gradient wrt L_j
#     ------------------------------------------------------------
#     Since S_j = L_j L_j^T chain rule gives:
#         dL/dL_j = (dL/dS_j + dL/dS_j^T) L_j

#     ------------------------------------------------------------
#     Gradient wrt phi_j
#     ------------------------------------------------------------
#         dL/dphi_j
#             = -n_j / phi_j + lambda

class GDA_SGD:
    """
    Gaussian Discriminant Analysis trained with mini-batch SGD
    on the negative log likelihood.
    """

    def __init__(self, k=3, max_iter=500, eps=1e-6,
                 learning_rate=1e-3, batch_size=cfg_sup.batch_size, reg_stren=10, 
                 verbose=True):

        self.k = k
        self.max_iter = max_iter
        self.eps = eps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose
        self.reg_stren = reg_stren

        self.mu = None
        self.L = None
        self.alpha = None

        self.loss_history = []

    def softmax(self, z):
        z = z - np.max(z)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    @property
    def phi(self):
        return self.softmax(self.alpha)

    def precision_matrix(self, j):
        return self.L[j] @ self.L[j].T

    def fit(self, x, y):
        n, d = x.shape

        if self.mu is None:
            self.mu = np.random.randn(self.k, d)
        if self.L is None:
            self.L = np.array([np.eye(d) for _ in range(self.k)])
        if self.alpha is None:
            self.alpha = np.zeros(self.k)

        for epoch in range(self.max_iter):
            # perm = np.random.permutation(n)
            x_shuffled = x #[perm]
            y_shuffled = y #[perm]

            self.gradient_descent_epoch(x_shuffled, y_shuffled)

            loss = self.nll_loss(x, y)
            self.loss_history.append(loss)

            if self.verbose and epoch % 200 == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

            if epoch > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.eps:
                break

    def nll_loss(self, x, y):
        n, d = x.shape
        loss = 0.0
        phi = self.phi

        for j in range(self.k):
            x_j = x[y == j]
            n_j = len(x_j)

            if n_j == 0:
                continue

            mu_j = self.mu[j]
            S_j = self.precision_matrix(j)
            phi_j = phi[j]

            diff = x_j - mu_j

            sign, logdetS = np.linalg.slogdet(S_j)
            quad = np.sum((diff @ S_j) * diff)

            class_loss = (
                -n_j * np.log(phi_j + 1e-12)
                -0.5 * n_j * logdetS
                +0.5 * quad
            )

            loss += class_loss

        # regularize precision matrix
        raw_reg = 0
        I = np.eye(d)
        for k in range(self.k):
            raw_reg += np.sum((self.precision_matrix(k) - I) ** 2) 
        reg = self.reg_stren * raw_reg

        if cfg_sup.penalty_type == "ce_standard":
            logits = np.zeros((n, self.k))  # (n, k)

            for k in range(self.k):
                mu_k = self.mu[k]
                S_k = self.precision_matrix(k)
                diff = x - mu_k

                sign, logdetS = np.linalg.slogdet(S_k)
                quad = np.sum((diff @ S_k) * diff, axis=1)

                logits[:, k] = (
                    np.log(phi[k] + 1e-12)
                    + 0.5 * logdetS
                    - 0.5 * quad
                )

            # stable softmax
            logits -= np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # log p(y_i | x_i)
            log_probs = np.log(probs[np.arange(n), y] + 1e-12)

            # transition indicator
            transition_indicator = (y[h:] != y[:-h]).astype(float)

            penalty = -cfg_sup.lam * np.sum(
                transition_indicator * log_probs[h:]
            )
        if cfg_sup.penalty_type == "ce_pairwise":
            logits = np.zeros((n, self.k))  # (n, k)

            for k in range(self.k):
                mu_k = self.mu[k]
                S_k = self.precision_matrix(k)
                diff = x - mu_k

                sign, logdetS = np.linalg.slogdet(S_k)
                quad = np.sum((diff @ S_k) * diff, axis=1)

                logits[:, k] = (
                    np.log(phi[k] + 1e-12)
                    + 0.5 * logdetS
                    - 0.5 * quad
                )

            # stable softmax
            logits -= np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # log p(y_i | x_i)
            log_probs = np.log(probs[np.arange(n), y] + 1e-12)

            # transition indicator
            transition_indicator = (y[h:] != y[:-h]).astype(float)

            penalty = -cfg_sup.lam * np.sum(
                transition_indicator * (log_probs[h:] + log_probs[:-h])
            )
        else:
            penalty = 0.0

        return (loss + penalty + reg) / n

    def nll_grad(self, x_batch, y_batch):
        """
        Computes gradients for:

            L_total = L_NLL + lambda_trans * L_trans_ce

        where

            L_trans_ce =
            - sum_j 1[y_j != y_{j-1}] log p(y_j | x_j)

        The NLL part is your original generative GDA objective.
        The transition term is discriminative and uses GDA posteriors.
        """

        n_batch, d = x_batch.shape

        grad_mu = np.zeros_like(self.mu)
        grad_L = np.zeros_like(self.L)
        grad_alpha = np.zeros_like(self.alpha)

        phi = self.phi

        # Gradients for NLL part only
        for k in range(self.k):
            x_k = x_batch[y_batch == k]
            n_k = len(x_k)

            if n_k == 0:
                continue

            mu_k = self.mu[k]
            L_k = self.L[k]
            S_k = self.precision_matrix(k)
            phi_k = phi[k]
            diff = x_k - mu_k

            grad_mu[k] += -np.sum(diff @ S_k, axis=0)


            # grad wrt S_k
            scatter = diff.T @ diff
            Sigma_k = np.linalg.inv(S_k + 1e-5 * np.eye(d))
            grad_S = 0.5 * (scatter - n_k * Sigma_k)

            # chain rule: S_k = L_k L_k^T
            grad_L[k] += (grad_S + grad_S.T) @ L_k # 2.0 * grad_S @ L_k

            grad_alpha[k] += n_batch * phi_k - n_k

        # Transition CE loss
        if cfg_sup.penalty_type == "ce_standard":
            # compute posterior probabilities p(y=k | x)
            logits = np.zeros((n_batch, self.k))
            for k in range(self.k):
                mu_k = self.mu[k]
                S_k = self.precision_matrix(k)
                diff = x_batch - mu_k
                sign, logdetS = np.linalg.slogdet(S_k)
                quad = np.sum((diff @ S_k) * diff, axis=1)

                logits[:, k] = (
                    np.log(phi[k] + 1e-12)
                    + 0.5 * logdetS
                    - 0.5 * quad
                )
            logits -= np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            Y = np.eye(self.k)[y_batch]
            G = probs - Y

            transition_mask = np.zeros(n_batch)
            transition_mask[h:] = (
                y_batch[h:] != y_batch[:-h]
            ).astype(float)

            G *= cfg_sup.lam * transition_mask[:, None]

            # alpha gradient
            grad_alpha += np.sum(G, axis=0)

            # mu and L gradients
            for k in range(self.k):
                S_k = self.precision_matrix(k)
                L_k = self.L[k]

                diff = x_batch - self.mu[k]

                # mu gradient
                grad_mu[k] += np.sum(
                    G[:, k][:, None] * (diff @ S_k),
                    axis=0
                )

                # S gradient
                outer = np.einsum(
                    'ni,nj->nij',
                    diff,
                    diff
                )
                weighted_outer = np.sum(
                    G[:, k][:, None, None] * outer,
                    axis=0
                )
                grad_S = (
                    0.5 * np.sum(G[:, k]) * np.linalg.inv(S_k)
                    - 0.5 * weighted_outer
                )

                grad_L[k] += (
                    grad_S + grad_S.T
                ) @ L_k

        # for regularization
        for j in range(self.k):
            I = np.eye(d)
            S_j = self.precision_matrix(j)
            grad_L[j] += 4.0 * self.reg_stren * (S_j - I) @ self.L[j]

        grad_mu /= n_batch
        grad_L /= n_batch
        grad_alpha /= n_batch

        return grad_mu, grad_L, grad_alpha

    def gradient_descent_epoch(self, x_shuffled, y_shuffled):
        n = x_shuffled.shape[0]

        for start_idx in range(0, n, self.batch_size):
            end_idx = start_idx + self.batch_size

            x_batch = x_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            grad_mu, grad_L, grad_alpha = self.nll_grad(x_batch, y_batch)

            self.mu -= self.learning_rate * grad_mu
            self.L -= self.learning_rate * grad_L
            self.alpha -= self.learning_rate * grad_alpha

            # stabilize Cholesky diagonal
            for j in range(self.k):
                self.L[j] = np.tril(self.L[j])

                diag = np.diag(self.L[j])
                diag = np.clip(diag, 1e-4, None)

                np.fill_diagonal(self.L[j], diag)

    def predict(self, x):
        n, d = x.shape
        log_posteriors = np.zeros((n, self.k))

        phi = self.phi

        for j in range(self.k):
            mu_j = self.mu[j]
            S_j = self.precision_matrix(j)
            phi_j = phi[j]

            diff = x - mu_j

            sign, logdetS = np.linalg.slogdet(S_j)
            quad = np.sum((diff @ S_j) * diff, axis=1)

            log_likelihood = (
                np.log(phi_j + 1e-12)
                + 0.5 * logdetS
                - 0.5 * quad
            )
            log_posteriors[:, j] = log_likelihood

        return np.argmax(log_posteriors, axis=1)


if __name__ == "__main__":
    main()
