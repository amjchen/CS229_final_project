import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import contextlib
import io
from sklearn.metrics import classification_report
from data_munging import standardize_data

from config import DataConfig, SupervisedConfig

cfg = DataConfig()
cfg_sup = SupervisedConfig()
fold_results = []


def build_supervised_regime_dataset(
    df: pd.DataFrame,
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
    horizon : int
        Forecast horizon in trading days
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
        feature_cols = [c for c in df.columns if c != target_col]

        array = df[feature_cols].values
        n = len(df)
        labels = df[target_col]
        x_rows, y_vals, index = [], [], []
        for i in range(window_size - 1, n - horizon):
            window = array[i - window_size + 1 : i + 1]
            # row = window.flatten()  # flattened window
            row = np.concatenate([window.mean(axis=0), window.std(axis=0), window[-1]])
            x_rows.append(row)
            y_vals.append(int(labels[i + horizon]))
            index.append(df.index[i])

        # col_names for flattened window (commented out):
        # col_names = [f"{col}_lag{lag}" for lag in range(window_size-1, -1, -1) for col in feature_cols]
        col_names = (
            [f"{col}_mean" for col in feature_cols] +
            [f"{col}_std" for col in feature_cols] +
            [f"{col}_last" for col in feature_cols]
        )

        X = pd.DataFrame(x_rows, index = index, columns = col_names)
        y = pd.Series(y_vals, index = index, dtype = int)
        supervised_df = df.loc[index].copy()
        supervised_df["target"] = y.values
    else: 
        # yesterday's regime
        df["regime_lag_1"] = df[target_col].shift(1)
        # 1 trading week ago
        df["regime_lag_5"] = df[target_col].shift(5)

        # Predict future regime using current information
        df["target"] = df[target_col].shift(-horizon)

        # Drop Nans created from shifting
        supervised_df = df.dropna().copy()

        # For design matrix X
        # remove current regime label + future target label
        drop_cols = [target_col, "target"]
        X = supervised_df.drop(columns=drop_cols)

        # Target
        y = supervised_df["target"].astype(int)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"y distribution:\n{y.value_counts().sort_index()}")

    return X, y, supervised_df





def print_results(preds_train, supervised_df, y_train, preds, y_test):
    y_test_np =  y_test.to_numpy(dtype=np.int64)
    y_train_np = y_train.to_numpy(dtype=np.int64)


    # print("\nTRAIN LABEL DISTRIBUTION")
    # print(y_train.value_counts().sort_index())
    # print("\nTEST LABEL DISTRIBUTION")
    # print(y_test.value_counts().sort_index())

    # transition_rate = (supervised_df["regime"] != supervised_df["target"]).mean()
    # print(f"Transition rate: {transition_rate:.4f}")

    print("\nTRAIN RESULTS")
    print(classification_report(y_train_np, preds_train))

    current_regime_train = supervised_df.loc[y_train.index, "regime"]
    future_regime_train = y_train
    predicted_regime_train = pd.Series(preds_train, index=y_train.index)
    transition_mask_train = current_regime_train != future_regime_train
    n_transitions_train = transition_mask_train.sum()
    correct_transition_preds_train = (
        predicted_regime_train[transition_mask_train] == future_regime_train[transition_mask_train]
    ).sum()

    print(f"True transitions: {n_transitions_train}")
    print(f"Correctly predicted transitions: {correct_transition_preds_train}")
    if n_transitions_train > 0: 
        print(f"Metric2 transition accuracy: {correct_transition_preds_train / n_transitions_train:.4f}")

    m1_mask_tr = y_train_np[1:] != y_train_np[:-1]
    m1_n_tr    = m1_mask_tr.sum()
    m1_acc_tr = None
    if m1_n_tr > 0: 
        m1_acc_tr  = (preds_train[1:][m1_mask_tr] == y_train_np[1:][m1_mask_tr]).sum() / m1_n_tr

        print(f"Metric1 transition accuracy: {m1_acc_tr:.4f} ({m1_n_tr} transitions)")

    # preds = clf.predict(X_test_np)
    print("\nTEST RESULTS")
    print(classification_report(y_test_np, preds))

    # Compute transition accuracy for test set
    current_regime = supervised_df.loc[y_test.index, "regime"]  # current regime at prediction time t
    future_regime = y_test  # future true regime at t+h

    # Predicted future regime
    predicted_regime = pd.Series(preds, index=y_test.index)

    # true transitions
    transition_mask = current_regime != future_regime
    n_transitions = transition_mask.sum()

    if n_transitions == 0:
        print("No transition in test window -- skipping")
        return


    # correctly predicted transitions
    correct_transition_preds = (
        predicted_regime[transition_mask] == future_regime[transition_mask]
    ).sum()

    transition_accuracy = correct_transition_preds / n_transitions

    print("Test set results:")
    print(f"True transitions: {n_transitions}")
    print(f"Correctly predicted transitions: {correct_transition_preds}")
    print(f"Metric2 transition accuracy: {transition_accuracy:.4f}")

    m1_mask = y_test_np[1:] != y_test_np[:-1]
    m1_n    = m1_mask.sum()

    m1_acc = None
    if m1_n > 0: 
        m1_acc  = (preds[1:][m1_mask] == y_test_np[1:][m1_mask]).sum() / m1_n
        print(f"Metric1 transition accuracy: {m1_acc:.4f} ({m1_n} transitions)")

    train_accuracy = (preds_train == y_train_np).mean()
    test_accuracy = (preds == y_test_np).mean()
    fold_results.append({
        "train_accuracy" : train_accuracy,
        "test_accuracy" : test_accuracy,

        "train_m1" : m1_acc_tr,
        "test_m1" : m1_acc,

        "train_m2": correct_transition_preds_train / n_transitions_train if n_transitions_train > 0 else None,
        "test_m2" : transition_accuracy
    })

def CV_for_lambda(X, y, model_type):
    n = len(X)
    split = n // 2

    train_part_X = X.iloc[:split]
    X_train, scaler = standardize_data(train_part_X)
    X_train_np = X_train.to_numpy(dtype = np.float64)
    X_test_np = scaler.transform(X.iloc[split:]).astype(np.float64)

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    y_train_np = y_train.to_numpy(dtype = np.int64)
    y_test_np = y_test.to_numpy(dtype = np.int64)

    global_best_lambda = cfg_sup.lam_values[0]
    global_best_score = -np.inf
    print(f"\n Tuning lambda | model = {model_type} penalty = {cfg_sup.penalty_type}")

    for lmd in cfg_sup.lam_values:
        cfg_sup.lam = lmd
        if model_type == "softmax":
            clf = SoftmaxRegression(max_iter = cfg_sup.max_iter, eps = 1e-6, k = cfg.kmeans_k)
            clf.fit(X_train_np, y_train_np, learning_rate = cfg_sup.learning_rate, batch_size = cfg_sup.batch_size)
        elif model_type == "gda":
            clf = GDA_SGD(k = cfg.kmeans_k)
            clf.fit(X_train_np, y_train_np)
        elif model_type == "neuralnet":
            x_shape = X_train_np.shape[1]
            clf = Neural_Networks(x_shape, cfg.kmeans_k, cfg_sup)
            clf.setup_nn()
            clf.fit(X_train_np, y_train_np)        

        predicted_values = clf.predict(X_test_np)
        indicator = y_test_np[1:] != y_test_np[:-1]
        total_m1 = indicator.sum()

        if total_m1 != 0:
            summed = (predicted_values[1:][indicator] == y_test_np[1:][indicator]).sum()
            score = summed / total_m1
        else:
            score = 0.0
        
        print(f"M1 value for lambda {lmd} = {score:.4f}")
        if score > global_best_score:
            global_best_score = score
            best_lam = lmd
    
    print(f"Best Lambda = {best_lam}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--softmax", action = "store_true")
    parser.add_argument("--gda", action="store_true")
    parser.add_argument("--neuralnet", action = "store_true")
    parser.add_argument("--tunelambda", action = "store_true")
    args = parser.parse_args()

    df_labeled = pd.read_csv(cfg.labeled_output_path, index_col=0, parse_dates=True)
    X, y, supervised_df = build_supervised_regime_dataset(df=df_labeled)

    if args.tunelambda:
        if args.softmax: 
            model_type = "softmax"
        elif args.gda:
            model_type = "gda"
        elif args.neuralnet:
            model_type = "neuralnet"
        CV_for_lambda(X, y, model_type)

    if cfg_sup.testing_method == "walk_forward":
        n = len(X)
        minimum_training_set = int(n / 2)
        folds = cfg_sup.folds
        fold_size = (n - minimum_training_set) // folds

        for i in range(folds):
            beg_test_idx = minimum_training_set + i * fold_size
            if i < folds - 1:
                end_test_idx = beg_test_idx + fold_size
            else: 
                end_test_idx = n
            
            X_fold, scaler = standardize_data(X.iloc[:beg_test_idx])
            X_train_np = X_fold.to_numpy(dtype = np.float64)
            X_test_np = scaler.transform(X.iloc[beg_test_idx:end_test_idx]).astype(np.float64)
            y_train = y.iloc[:beg_test_idx]
            y_test = y.iloc[beg_test_idx:end_test_idx]
            y_train_np = y_train.to_numpy(dtype = np.int64)

            print(f"\n===== Fold {i+1} | train [0:{beg_test_idx}]  test [{beg_test_idx}:{end_test_idx}] =====")


            if args.softmax:
                print("SOFTMAX REGRESSION")
                clf = SoftmaxRegression(max_iter=cfg_sup.max_iter, eps=1e-6, k=cfg.kmeans_k)
                clf.fit(X_train_np, y_train_np, learning_rate=cfg_sup.learning_rate, batch_size=cfg_sup.batch_size)
                preds_train = clf.predict(X_train_np)
                preds = clf.predict(X_test_np)
                print_results(preds_train, supervised_df, y_train, preds, y_test)

            if args.gda:
                print("GAUSSIAN DISCRIMINANT ANALYSIS")
                clf = GDA_SGD(k=cfg.kmeans_k)
                clf.fit(X_train_np, y_train_np)
                preds_train = clf.predict(X_train_np)
                preds = clf.predict(X_test_np)
                print_results(preds_train, supervised_df, y_train, preds, y_test)

            if args.neuralnet:
                print("NEURAL NETWORK")
                dim_in = X_train_np.shape[1]
                k = cfg.kmeans_k
                clf = Neural_Networks(dim_in, k, cfg_sup)
                clf.setup_nn()

                clf.fit(X_train_np, y_train_np)

                preds_train = clf.predict(X_train_np)
                preds = clf.predict(X_test_np)

                print_results(preds_train, supervised_df, y_train, preds, y_test)
        
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
        print(f"Avg train accuracy : {(train_sum / c1):.4f}")
        print(f"Avg test accuracy  : {(test_sum / c2):.4f}")
        print(f"Avg train M1       : {(train_m1_sum / c3):.4f}")
        print(f"Avg test M1        : {(test_m1_sum / c4):.4f}")
        print(f"Avg train M2       : {(train_m2_sum / c5):.4f}")
        print(f"Avg test M2        : {(test_m2_sum / c6):.4f}")

 

    else: 
        split_idx = int(len(X) * cfg_sup.train_split)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
    
        X_train, scaler = standardize_data(X_train)
        X_train_np = X_train.to_numpy(dtype=np.float64)
        X_test_np = scaler.transform(X_test)
        X_test_np = X_test_np.astype(np.float64)
        y_train_np = y_train.to_numpy(dtype=np.int64)

        if args.softmax:
            print("SOFTMAX REGRESSION")
            clf = SoftmaxRegression(max_iter=cfg_sup.max_iter, eps=1e-6, k=cfg.kmeans_k)
            clf.fit(X_train_np, y_train_np, learning_rate=cfg_sup.learning_rate, batch_size=cfg_sup.batch_size)
            preds_train = clf.predict(X_train_np)
            preds = clf.predict(X_test_np)
            print_results(preds_train, supervised_df, y_train, preds, y_test)

        if args.gda:
            print("GAUSSIAN DISCRIMINANT ANALYSIS")
            clf = GDA_SGD(k=cfg.kmeans_k)
            clf.fit(X_train_np, y_train_np)
            preds_train = clf.predict(X_train_np)
            preds = clf.predict(X_test_np)
            print_results(preds_train, supervised_df, y_train, preds, y_test)

        if args.neuralnet:
            print("NEURAL NETWORK")
            dim_in = X_train_np.shape[1]
            k = cfg.kmeans_k
            clf = Neural_Networks(dim_in, k, cfg_sup)
            clf.setup_nn()

            clf.fit(X_train_np, y_train_np)

            preds_train = clf.predict(X_train_np)
            preds = clf.predict(X_test_np)

            print_results(preds_train, supervised_df, y_train, preds, y_test)


class SoftmaxRegression:
    """Softmax regression with SGD as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, max_iter=1000000, eps=1e-6,
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

    def fit(self, x, y, learning_rate, batch_size):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Shape (n_examples, dim).
            y: Shape (n_examples,).
        """

        x_inter = self.add_intercept(x.copy())
        d = x_inter.shape[1]

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

        if cfg_sup.penalty_type == "ce_standard":
            transition_indicator = (y[1:] != y[:-1]).astype(float)
            penalty = -cfg_sup.lam * (transition_indicator * log_loss[1:]).sum()
        elif cfg_sup.penalty_type == "ce_pairwise":
            transition_indicator = (y[1:] != y[:-1]).astype(float)
            penalty = -cfg_sup.lam * (transition_indicator * (log_loss[1:] + log_loss[:-1])).sum()
        elif cfg_sup.penalty_type == "margin_bar":
            transition_indicator = (y[1:] != y[:-1]).astype(float)
            diff_today_yesterday = np.zeros(n-1)
            for k in range(1, n):
                yesterday = y[k-1]
                today = y[k]
                diff_today_yesterday[k-1] = cfg.margin + prob[k, yesterday] - prob[k, today]
            penalty = cfg_sup.lam * (transition_indicator * np.log(1 + np.exp(diff_today_yesterday))).sum()
        elif cfg_sup.penalty_type == "distance":
            transition_indicator = (y[1:] != y[:-1]).astype(float)
            inside = np.zeros(n-1)
            for i in range(n-1):
                for k in range(cfg.kmeans_k):
                    inside[i] += ((y[i+1] - k) ** 2) * prob[i+1, k]
            penalty = cfg_sup.lam *(transition_indicator * inside).sum()
        else:
            penalty = 0

        return loss + penalty

    def ce_grad(self, X, y):
        n = X.shape[0]
        logits = X @ self.theta
        prob = self.softmax(logits)

        one_hot = self.one_hot(y)

        dZ = prob - one_hot
        dTheta = X.T @ dZ

        if cfg_sup.penalty_type == "ce_standard":
            transition_indicator = (y[1:] != y[:-1]).astype(float)
            dZ_trans = np.zeros((n, self.k))
            dZ_trans[1:] = transition_indicator[:, None] * (prob[1:] - one_hot[1:])
            penalty_grad = cfg_sup.lam * X.T @ dZ_trans
        elif cfg_sup.penalty_type == "ce_pairwise":
            transition_indicator = (y[1:] != y[:-1]).astype(float)
            term1 = X[1:].T @ (transition_indicator[:, None] * (1 + cfg_sup.lam) * (prob[1:] - one_hot[1:]))
            term2 = X[:-1].T @ (transition_indicator[:, None] * cfg_sup.lam * (prob[:-1] - one_hot[:-1]))
            penalty_grad = term1 + term2
        elif cfg_sup.penalty_type == "margin_bar":
            transition_indicator = (y[1:] != y[:-1]).astype(float)
            diff_today_yesterday = np.zeros(n-1)
            for k in range(1, n):
                yesterday = y[k-1]
                today = y[k]
                diff_today_yesterday[k-1] = cfg_sup.margin + prob[k, yesterday] - prob[k, today]
            penalty = cfg_sup.lam * (transition_indicator * np.log(1 + np.exp(diff_today_yesterday))).sum()

            sigmoid = 1.0 / (1.0 + np.exp(-diff_today_yesterday))
            one_hot_difference = np.zeros((n-1, self.k))
            for i in range(n-1):
                one_hot_difference[i, y[i]] += 1.0
                one_hot_difference[i, y[i+1]] -= 1.0
            
            constant = (transition_indicator * sigmoid)[:, None]
            summing = prob[1:] * (one_hot_difference - (diff_today_yesterday - cfg_sup.margin)[:, None])

            penalty_grad = cfg_sup.lam * X[1:]. T @ (constant * summing)
        elif cfg_sup.penalty_type == "distance":
            transition_indicator = (y[1:] != y[:-1]).astype(float)
            inside = np.zeros(n-1)
            for i in range(n-1):
                for k in range(cfg.kmeans_k):
                    inside[i] += ((y[i+1] - k) ** 2) * prob[i+1, k]

            penalty_grad = np.zeros((X.shape[1], self.k))
            one_hot_y = self.one_hot(y)
            c = np.zeros(cfg.kmeans_k)
            for k in range(cfg.kmeans_k):
                c[k] = k

            for i in range(n - 1):
                if transition_indicator[i]:
                    for k in range(self.k):
                        ctyi = np.dot(c, one_hot_y[i + 1])
                        weight = cfg_sup.lam * prob[i + 1, k] *((ctyi - c[k]) ** 2 - inside[i])
                        penalty_grad[:, k] += weight * X[i+1]
            
        else:
            penalty_grad = 0

        return dTheta + penalty_grad

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

            dTheta = self.ce_grad(X_batch, y_batch) / batch_size_actual
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
    def __init__(self, dim_in, k, cfg_sup):
        super().__init__()

        self.dim_in = dim_in
        self.k = k
        self.cfg_sup = cfg_sup

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

        if self.cfg_sup.penalty_type == "ce_standard":
            transition_indicator = (y[1:] != y[:-1]).float()
            penalty = -self.cfg_sup.lam * (transition_indicator * log_loss[1:]).sum()
        elif self.cfg_sup.penalty_type == "ce_pairwise":
            transition_indicator = (y[1:] != y[:-1]).float()
            penalty = -self.cfg_sup.lam * (transition_indicator * (log_loss[1:] + log_loss[:-1])).sum()
        elif self.cfg_sup.penalty_type == "margin_bar":
            transition_indicator = (y[1:] != y[:-1]).float()
            diff_today_yesterday = []

            for k in range(1, n):
                yest_prb = prob[k, y[k-1]]
                today_prb = prob[k, y[k]]

                diff = self.cfg_sup.margin + yest_prb - today_prb
                diff_today_yesterday.append(diff)

            stacked_diff = torch.stack(diff_today_yesterday)
            penalty = self.cfg_sup.lam * (transition_indicator * (torch.log(1 + torch.exp(stacked_diff)))).sum()

        elif self.cfg_sup.penalty_type == "distance":
            transition_indicator = (y[1:] != y[:-1]).float()
            inside = []
            for i in range(n-1):
                temp = 0.0
                for k in range(self.k):
                    temp += (y[i + 1] - k) ** 2 * prob[i+1,k]
                
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

#     We omit explicit lambda and renormalize phi after updates.
#     """

#     def __init__(
#         self,
#         k=3,
#         max_iter=cfg_sup.max_iter,
#         eps=1e-6,
#         learning_rate=cfg_sup.learning_rate,
#         batch_size=cfg_sup.batch_size,
#         verbose=True,
#     ):
#         self.k = k
#         self.max_iter = max_iter
#         self.eps = eps
#         self.learning_rate = learning_rate
#         self.batch_size = batch_size
#         self.verbose = verbose

#         self.mu = None
#         self.L = None
#         self.phi = None

#         self.loss_history = []

#     def precision_matrix(self, j):
#         """
#         S_j = L_j L_j^T
#         """
#         return self.L[j] @ self.L[j].T

#     def fit(self, x, y):
#         n, d = x.shape

#         if self.mu is None:
#             self.mu = np.random.randn(self.k, d)
#         if self.L is None:
#             self.L = np.array([
#                 np.eye(d) for _ in range(self.k)
#             ])
#         if self.phi is None:
#             self.phi = np.ones(self.k) / self.k

#         for epoch in range(self.max_iter):
#             # shuffle
#             perm = np.random.permutation(n)
#             x_shuffled = x[perm]
#             y_shuffled = y[perm]

#             self.gradient_descent_epoch(x_shuffled, y_shuffled)
#             loss = self.nll_loss(x, y)
#             self.loss_history.append(loss)

#             if self.verbose and epoch % 200 == 0:
#                 print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

#             if (epoch > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.eps):
#                 break

#     def nll_loss(self, x, y):
#         n, d = x.shape
#         loss = 0.0

#         for j in range(self.k):
#             x_j = x[y == j]
#             n_j = len(x_j)

#             if n_j == 0:  # skip if no examples of class j
#                 continue

#             mu_j = self.mu[j]
#             S_j = self.precision_matrix(j)
#             phi_j = self.phi[j]

#             diff = x_j - mu_j
#             sign, logdetS = np.linalg.slogdet(S_j)
#             quad = np.sum((diff @ S_j) * diff)

#             class_loss = (
#                 -n_j * np.log(phi_j + 1e-12)
#                 -0.5 * n_j * logdetS
#                 +0.5 * quad
#             )

#             loss += class_loss

#         return loss / n

#     def nll_grad(self, x_batch, y_batch):
#         n_batch, d = x_batch.shape

#         grad_mu = np.zeros_like(self.mu)
#         grad_L = np.zeros_like(self.L)
#         grad_phi = np.zeros_like(self.phi)

#         for j in range(self.k):
#             x_j = x_batch[y_batch == j]
#             n_j = len(x_j)

#             if n_j == 0:
#                 continue

#             mu_j = self.mu[j]
#             L_j = self.L[j]
#             S_j = self.precision_matrix(j)
#             phi_j = self.phi[j]

#             diff = x_j - mu_j

#             # dL/dmu_j = - sum S_j (x_i - mu_j)
#             grad_mu[j] = -np.sum(
#                 diff @ S_j,
#                 axis=0
#             )

#             # dL/dS_j = 0.5 * (scatter - n_j S^-1)
#             scatter = diff.T @ diff
#             Sigma_j = np.linalg.inv(S_j)
#             grad_S = 0.5 * (scatter - n_j * Sigma_j)

#             # dL/dL_j = 2 grad_S L_j
#             grad_L[j] = 2.0 * grad_S @ L_j

#             # dL/dphi_j
#             grad_phi[j] = -n_j / (phi_j + 1e-12)

#         # average gradients over batch
#         grad_mu /= n_batch
#         grad_L /= n_batch
#         grad_phi /= n_batch

#         return grad_mu, grad_L, grad_phi

#     def gradient_descent_epoch(self, x_shuffled, y_shuffled):
#         n = x_shuffled.shape[0]

#         for start_idx in range(0, n, self.batch_size):
#             end_idx = start_idx + self.batch_size

#             x_batch = x_shuffled[start_idx:end_idx]
#             y_batch = y_shuffled[start_idx:end_idx]

#             grad_mu, grad_L, grad_phi = self.nll_grad(x_batch, y_batch)
#             self.mu -= self.learning_rate * grad_mu
#             self.L -= self.learning_rate * grad_L
#             self.phi -= self.learning_rate * grad_phi

#             # stabilize phi
#             self.phi = np.clip(self.phi, 1e-8, None)
#             self.phi /= np.sum(self.phi)

#             # stabilize Cholesky diagonal
#             for j in range(self.k):
#                 self.L[j] = np.tril(self.L[j])
#                 diag = np.diag(self.L[j])
#                 diag = np.clip(diag, 1e-4, None)

#                 np.fill_diagonal(
#                     self.L[j],
#                     diag
#                 )

#     def predict(self, x):
#         n, d = x.shape

#         log_posteriors = np.zeros((n, self.k))

#         for j in range(self.k):
#             mu_j = self.mu[j]
#             S_j = self.precision_matrix(j)
#             phi_j = self.phi[j]

#             diff = x - mu_j
#             sign, logdetS = np.linalg.slogdet(S_j)
#             quad = np.sum((diff @ S_j) * diff, axis=1)

#             log_likelihood = (
#                 np.log(phi_j + 1e-12)
#                 + 0.5 * logdetS
#                 - 0.5 * quad
#             )

#             log_posteriors[:, j] = log_likelihood

#         return np.argmax(log_posteriors, axis=1)

import numpy as np


class GDA_SGD:
    """
    Gaussian Discriminant Analysis trained with mini-batch SGD
    on the negative log likelihood.
    """

    def __init__(self, k=3, max_iter=1000, eps=1e-6,
                 learning_rate=1e-3, batch_size=32,
                 verbose=True):

        self.k = k
        self.max_iter = max_iter
        self.eps = eps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose

        self.mu = None
        self.L = None
        self.alpha = None

        self.loss_history = []

    # =========================================================
    # Softmax prior parameterization
    # =========================================================

    def softmax(self, z):
        z = z - np.max(z)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    @property
    def phi(self):
        return self.softmax(self.alpha)

    # =========================================================
    # Precision matrix
    # =========================================================

    def precision_matrix(self, j):
        return self.L[j] @ self.L[j].T

    # =========================================================
    # Training
    # =========================================================

    def fit(self, x, y):
        n, d = x.shape

        if self.mu is None:
            self.mu = np.random.randn(self.k, d)

        if self.L is None:
            self.L = np.array([np.eye(d) for _ in range(self.k)])

        if self.alpha is None:
            self.alpha = np.zeros(self.k)

        for epoch in range(self.max_iter):
            perm = np.random.permutation(n)
            x_shuffled = x[perm]
            y_shuffled = y[perm]

            self.gradient_descent_epoch(x_shuffled, y_shuffled)

            loss = self.nll_loss(x, y)
            self.loss_history.append(loss)

            if self.verbose and epoch % 200 == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

            if epoch > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.eps:
                break

    # =========================================================
    # Negative log likelihood
    # =========================================================

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
            transition_indicator = (y[1:] != y[:-1]).astype(float)

            penalty = -cfg_sup.lam * np.sum(
                transition_indicator * log_probs[1:]
            )
        else:
            penalty = 0.0

        return (loss + penalty) / n

    # =========================================================
    # Gradients
    # =========================================================

    # def nll_grad(self, x_batch, y_batch):
    #     n_batch, d = x_batch.shape

    #     grad_mu = np.zeros_like(self.mu)
    #     grad_L = np.zeros_like(self.L)
    #     grad_alpha = np.zeros_like(self.alpha)

    #     phi = self.phi

    #     for j in range(self.k):
    #         x_j = x_batch[y_batch == j]
    #         n_j = len(x_j)

    #         if n_j == 0:
    #             continue

    #         mu_j = self.mu[j]
    #         L_j = self.L[j]
    #         S_j = self.precision_matrix(j)
    #         phi_j = phi[j]

    #         diff = x_j - mu_j

    #         # dL/dmu_j = - sum S_j (x_i - mu_j)
    #         grad_mu[j] = -np.sum(diff @ S_j, axis=0)

    #         # dL/dS_j = 0.5 * (scatter - n_j S^-1)
    #         scatter = diff.T @ diff
    #         Sigma_j = np.linalg.inv(S_j)

    #         grad_S = 0.5 * (scatter - n_j * Sigma_j)

    #         # dL/dL_j = 2 grad_S L_j
    #         grad_L[j] = 2.0 * grad_S @ L_j

    #         # dL/dalpha_j = n_batch * phi_j - n_j
    #         grad_alpha[j] = n_batch * phi_j - n_j

    #     grad_mu /= n_batch
    #     grad_L /= n_batch
    #     grad_alpha /= n_batch

    #     return grad_mu, grad_L, grad_alpha

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

            # -------------------------------------------------
            # grad wrt mu_k
            # -------------------------------------------------
            grad_mu[k] += -np.sum(diff @ S_k, axis=0)

            # -------------------------------------------------
            # grad wrt S_k
            # -------------------------------------------------
            scatter = diff.T @ diff
            Sigma_k = np.linalg.inv(S_k)
            grad_S = 0.5 * (scatter - n_k * Sigma_k)

            # -------------------------------------------------
            # chain rule: S_k = L_k L_k^T
            # -------------------------------------------------
            grad_L[k] += 2.0 * grad_S @ L_k

            # -------------------------------------------------
            # grad wrt alpha_k
            # -------------------------------------------------
            grad_alpha[k] += n_batch * phi_k - n_k

        # Transition CE loss
        if cfg_sup.penalty_type == "ce_standard":
            # -------------------------------------------------
            # compute posterior probabilities
            # p(y=k | x)
            # -------------------------------------------------

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

            if cfg_sup.penalty_type == "ce_standard":

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

                probs = np.exp(logits)
                probs /= np.sum(probs, axis=1, keepdims=True)

                # =====================================================
                # vectorized CE gradients
                # =====================================================

                Y = np.eye(self.k)[y_batch]

                G = probs - Y

                transition_mask = np.zeros(n_batch)
                transition_mask[1:] = (
                    y_batch[1:] != y_batch[:-1]
                ).astype(float)

                G *= cfg_sup.lam * transition_mask[:, None]

                # alpha gradient
                grad_alpha += np.sum(G, axis=0)

                # mu and L gradients
                for k in range(self.k):

                    S_k = self.precision_matrix(k)
                    L_k = self.L[k]

                    diff = x_batch - self.mu[k]

                    # -----------------------------------------
                    # mu gradient
                    # -----------------------------------------

                    grad_mu[k] += np.sum(
                        G[:, k][:, None] * (diff @ S_k),
                        axis=0
                    )

                    # -----------------------------------------
                    # S gradient
                    # -----------------------------------------

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

        # =====================================================
        # normalize
        # =====================================================

        grad_mu /= n_batch
        grad_L /= n_batch
        grad_alpha /= n_batch

        return grad_mu, grad_L, grad_alpha

    # =========================================================
    # SGD epoch
    # =========================================================

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

    # =========================================================
    # Prediction
    # =========================================================

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
