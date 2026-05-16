import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode as scipy_mode
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from config import DataConfig
# from data_munging import features, prices, derive_features, remove_outliers, standardize_data
# from unsupervised_learning import df_labeled, km, scaler

cfg = DataConfig()

import numpy as np
import os

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


# Baseline default softmax regression with CE loss
def main():
    df_labeled = pd.read_csv(cfg.labeled_output_path, index_col=0, parse_dates=True)
    X, y, supervised_df = build_supervised_regime_dataset(df=df_labeled, horizon=21)
    split_idx = int(len(X) * 0.75)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    print("\nTRAIN LABEL DISTRIBUTION")
    print(y_train.value_counts().sort_index())

    print("\nTEST LABEL DISTRIBUTION")
    print(y_test.value_counts().sort_index())

    transition_rate = (
        supervised_df["regime"] != supervised_df["target"]
    ).mean()

    print(f"Transition rate: {transition_rate:.4f}")

    clf = SoftmaxRegression(max_iter=100000, eps=1e-6, k=cfg.kmeans_k)
    X_train_np = X_train.to_numpy(dtype=np.float64)
    X_test_np = X_test.to_numpy(dtype=np.float64)
    y_train_np = y_train.to_numpy(dtype=np.int64)
    y_test_np = y_test.to_numpy(dtype=np.int64)
    clf.fit(X_train_np, y_train_np, learning_rate=1e-4, batch_size=500)
    
    preds = clf.predict(X_test_np)
    
    
    
    # model = LogisticRegression(
    #     solver="lbfgs",
    #     max_iter=50000,
    # )

    # model.fit(X_train, y_train)

    # preds_train = model.predict(X_train)
    # preds = model.predict(X_test)

    print(classification_report(y_test_np, preds))

    # Compute transition accuracy for test set 
    current_regime = supervised_df.loc[y_test.index, "regime"]  # current regime at prediction time t
    future_regime = y_test  # future true regime at t+h

    # Predicted future regime
    predicted_regime = pd.Series(preds, index=y_test.index)

    # # true transitions
    transition_mask = current_regime != future_regime
    n_transitions = transition_mask.sum()

    # # correctly predicted transitions
    correct_transition_preds = (
        predicted_regime[transition_mask] == future_regime[transition_mask]
    ).sum()

    transition_accuracy = (
        correct_transition_preds / n_transitions
    )

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
    def __init__(self, max_iter=1000000, eps=1e-5,
                 theta_0 = None, verbose=True, k = None):
        """
        Args:
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        # """
        # self.W = W_0
        # self.b = b_0
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

        # *** START CODE HERE ***
        x_inter = self.add_intercept(x.copy())
        n = x_inter.shape[0]
        d = x_inter.shape[1]

        if self.theta is None:
            self.theta = np.zeros((d, self.k))

        rng = np.random.default_rng(seed=229)

        for epoch in range(self.max_iter):
            # Shuffle every epoch
            indices = rng.permutation(n)

            X_shuffled = x_inter[indices]
            y_shuffled = y[indices]

            current_theta = self.theta.copy()
            self.gradient_descent_epoch(X_shuffled, y_shuffled, learning_rate, batch_size)

            # Track loss
            logits_full = x_inter @ self.theta
            probs_full = self.softmax(logits_full)
            loss = self.cross_entropy_loss(y, probs_full)
            self.loss_history.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

            if np.linalg.norm(current_theta - self.theta) < self.eps:
                break;

            
            
        # *** END CODE HERE ***

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

        # *** START CODE HERE ***
        # n_iter_per_epoch = X_shuffled.shape[0] // batch_size
        n_samples = X_shuffled.shape[0]

        for start_idx in range(0, n_samples, batch_size):
            end_idx = start_idx + batch_size

            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            # In case last batch not quite large enough
            batch_size_actual = len(X_batch)

            # Forward pass
            logits = X_batch @ self.theta
            probs = self.softmax(logits)

            # One-hot labels
            y_one_hot = self.one_hot(y_batch)

            # Gradient of CE loss
            dlogits = (probs - y_one_hot) / batch_size_actual
            dTheta = X_batch.T @ dlogits

            # L2 regularization
            # dW += self.reg_strength * self.W

            # SGD update
            self.theta -= learning_rate * dTheta
        # *** END CODE HERE ***

        # This function does not return anything
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
        # *** START CODE HERE ***
        # subtracting off maximum x on from all exponentials prevents overflow (all in (0, 1) now) while keeping answer the same
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exp_shift_z = np.exp(shift_z)
        return exp_shift_z / np.sum(exp_shift_z, axis=1, keepdims=True)
        # *** END CODE HERE ***
    
    def predict(self, x):
        x_inter = self.add_intercept(x.copy())
        logits = x_inter @ self.theta
        return np.argmax(self.softmax(logits), axis=1)
    # def predict(self, x):
    #     """Return predicted probabilities given new inputs x.

    #     Args:
    #         x: Shape (n_examples, dim).

    #     Returns:
    #         Outputs of shape (n_examples,).
    #     """
    #     #  if x.shape[1] < 3:   # add this so can reuse when calling on x that already has intercept
    #     x = util.add_intercept(x)
    #     # *** START CODE HERE ***
    #     # print(x.shape, self.theta.shape)
    #     return 1/(1+np.exp(-(x @ self.theta)))
    #     # *** END CODE HERE ***


if __name__ == "__main__":
    main()
