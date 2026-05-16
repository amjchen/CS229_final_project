import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from data_munging import standardize_data

from config import DataConfig, SupervisedConfig

cfg = DataConfig()
cfg_sup = SupervisedConfig()


def build_supervised_regime_dataset(
    df: pd.DataFrame,
    target_col: str = "regime",
    horizon: int = 21
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

    df = df.copy()

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

    return X, y, supervised_df


def cross_validate_lam(X_train_np, y_train_np, lam_values, n_splits=5):
    """
    Time-series walk-forward CV to find best lambda.
    Each fold uses an expanding training window and evaluates on the next block.
    Scores on metric 1: consecutive transition accuracy.
    """
    n = len(X_train_np)
    fold_size = n // (n_splits + 1)

    best_lam = lam_values[0]
    best_score = -np.inf

    for lam in lam_values:
        cfg_sup.lam = lam
        fold_scores = []

        for fold in range(1, n_splits + 1):
            train_end = fold * fold_size
            val_end = min(train_end + fold_size, n)

            X_fold_train = X_train_np[:train_end]
            y_fold_train = y_train_np[:train_end]
            X_fold_val   = X_train_np[train_end:val_end]
            y_fold_val   = y_train_np[train_end:val_end]

            clf = SoftmaxRegression(max_iter=cfg_sup.max_iter, eps=1e-6, k=cfg.kmeans_k)
            clf.fit(X_fold_train, y_fold_train,
                    learning_rate=cfg_sup.learning_rate,
                    batch_size=cfg_sup.batch_size)

            preds_val = clf.predict(X_fold_val)

            if cfg_sup.transition_metric == "metric2":
                h = cfg_sup.horizon
                transition_mask = y_fold_val[h:] != y_fold_val[:-h]
                preds_shifted = preds_val[h:]
                y_shifted = y_fold_val[h:]
            else:  # metric1: consecutive
                transition_mask = y_fold_val[1:] != y_fold_val[:-1]
                preds_shifted = preds_val[1:]
                y_shifted = y_fold_val[1:]

            n_trans = transition_mask.sum()
            if n_trans > 0:
                correct = (preds_shifted[transition_mask] == y_shifted[transition_mask]).sum()
                fold_scores.append(correct / n_trans)

        avg_score = np.mean(fold_scores) if fold_scores else 0.0
        print(f"lam={lam:.3f} | avg transition accuracy: {avg_score:.4f}")

        if avg_score > best_score:
            best_score = avg_score
            best_lam = lam

    print(f"\nBest lam: {best_lam} (transition accuracy: {best_score:.4f})")
    return best_lam


def main():
    df_labeled = pd.read_csv(cfg.labeled_output_path, index_col=0, parse_dates=True)
    X, y, supervised_df = build_supervised_regime_dataset(df=df_labeled, horizon=cfg_sup.horizon)
    split_idx = int(len(X) * cfg_sup.train_split)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    print("\nTRAIN LABEL DISTRIBUTION")
    print(y_train.value_counts().sort_index())
    print("\nTEST LABEL DISTRIBUTION")
    print(y_test.value_counts().sort_index())

    transition_rate = (supervised_df["regime"] != supervised_df["target"]).mean()
    print(f"Transition rate: {transition_rate:.4f}")

    X_train, scaler = standardize_data(X_train)
    X_train_np = X_train.to_numpy(dtype=np.float64)
    X_test_np = scaler.transform(X_test)
    X_test_np = X_test_np.astype(np.float64)
    y_train_np = y_train.to_numpy(dtype=np.int64)
    y_test_np = y_test.to_numpy(dtype=np.int64)

    best_lam = cross_validate_lam(X_train_np, y_train_np, cfg_sup.lam_values)
    cfg_sup.lam = best_lam

    clf = SoftmaxRegression(max_iter=cfg_sup.max_iter, eps=1e-6, k=cfg.kmeans_k)
    clf.fit(X_train_np, y_train_np, learning_rate=cfg_sup.learning_rate, batch_size=cfg_sup.batch_size)

    preds_train = clf.predict(X_train_np)
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
    print(f"Transition accuracy: {correct_transition_preds_train / n_transitions_train:.4f}")

    preds = clf.predict(X_test_np)
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

    # correctly predicted transitions
    correct_transition_preds = (
        predicted_regime[transition_mask] == future_regime[transition_mask]
    ).sum()

    transition_accuracy = correct_transition_preds / n_transitions

    print("Test set results:")
    print(f"True transitions: {n_transitions}")
    print(f"Correctly predicted transitions: {correct_transition_preds}")
    print(f"Transition accuracy: {transition_accuracy:.4f}")


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

            if epoch % 10000 == 0:
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

    def cross_entropy_loss(self, y_true, probs):
        """
        Mean cross entropy loss with L2 regularization.
        """

        n_samples = len(y_true)
        correct_class_probs = probs[np.arange(n_samples), y_true]
        log_likelihood = -np.log(correct_class_probs + 1e-15)
        data_loss = np.mean(log_likelihood)

        return data_loss

    def ce_loss(self, X, y):
        n = X.shape[0]
        logits = X @ self.theta
        prob = self.softmax(logits)

        log_loss = np.log(prob[np.arange(n), y] + 1e-12)
        loss = -log_loss.sum()

        if cfg_sup.penalty_type == "ce_standard":
            transition_indicator = (y[1:] != y[:-1]).astype(float)
            penalty = -cfg_sup.lam * (transition_indicator * log_loss[1:]).sum()
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


if __name__ == "__main__":
    main()
