import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode as scipy_mode
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax


from config import DataConfig
from config import SupervisedConfig

# from data_munging import features, prices, derive_features, remove_outliers, standardize_data
# from unsupervised_learning import df_labeled, km, scaler

cfg = DataConfig()
cfg_sup = SupervisedConfig()


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


def ce_loss(theta, X, y, K):
    n = X.shape[0]
    theta = theta.reshape(K,-1)

    eta = X @ theta.T
    prob = softmax(eta, axis = 1)

    log_loss = np.log(prob[np.arange(n), y] + 1e-12)
    loss = -log_loss.sum()

    if cfg_sup.penalty_type == "ce_standard":
        transition_indicator = (y[1:] != y[:-1]).astype(float)
        penalty = -cfg_sup.lam * (transition_indicator * log_loss[1:]).sum()
    else:
        penalty = 0

    loss += penalty

    return loss

def ce_grad(theta, X, y, K):
    n = X.shape[0]
    theta = theta.reshape(K, -1)

    eta = X @ theta.T
    prob = softmax(eta, axis = 1)

    one_hot = np.zeros((n,K))
    one_hot[np.arange(n), y] = 1

    dZ = prob - one_hot
    dTheta = dZ.T @ X

    if cfg_sup.penalty_type == "ce_standard":
        transition_indicator = (y[1:] != y[:-1]).astype(float)
        dZ_trans = np.zeros((n,K))
        dZ_trans[1:] = transition_indicator[:, None] * (prob[1:] - one_hot[1:])
        penalty_grad = cfg_sup.lam * dZ_trans.T @ X
    else:
        penalty_grad = 0
    
    grad = dTheta + penalty_grad

    return grad.ravel()

def optimize(X, y, loss_fn, gradient_fn, opt_method, max_iter = 2000):
    K = int(y.max()) + 1
    d = X.shape[1]

    theta_0 = np.zeros(K * d)
    theta_star = minimize(loss_fn, theta_0, args = (X, y, K), method = opt_method, jac = gradient_fn, options = {"maxiter" : max_iter})

    if not theta_star.success:
        print(f"[WARN] {theta_star.message}")
    
    return theta_star.x.reshape(K,d)



# Baseline default softmax regression with CE loss
def main():
    df_labeled = pd.read_csv(cfg.labeled_output_path, index_col=0, parse_dates=True)
    X, y, supervised_df = build_supervised_regime_dataset(df=df_labeled, horizon=21)
    split_idx = int(len(X) * 0.8)

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

    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train.values)
    X_test_np = scaler.fit_transform(X_test.values)
    y_train_np = y_train.values
    y_test_np = y_test.values

    theta = optimize(X_train_np, y_train_np, ce_loss, ce_grad, cfg_sup.opt_method, cfg_sup.max_iter)

    K = int(y_train_np.max()) + 1
    preds = softmax(X_test_np @ theta.T, axis = 1).argmax(axis = 1)

    print(classification_report(y_test, preds))

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

if __name__ == "__main__":
    main()

# def main(train_path, valid_path, save_path):
#     """Problem: Logistic regression with Newton's Method.

#     Args:
#         train_path: Path to CSV file containing dataset for training.
#         valid_path: Path to CSV file containing dataset for validation.
#         save_path: Path to save predicted probabilities using np.savetxt().
#     """
#     x_train, y_train = util.load_dataset(train_path, add_intercept=False)

#     # *** START CODE HERE ***
#     # Train a logistic regression classifier
#     clf = LogisticRegression()
#     clf.fit(x_train, y_train)
#     # Plot decision boundary on top of validation set set
#     x_eval, y_eval = util.load_dataset(valid_path, add_intercept=False)
#     y_pred = clf.predict(x_eval)
#     # Use np.savetxt to save predictions on eval set to save_path as a 1D numpy array
#     np.savetxt(save_path, y_pred)
#     save_path_first_part = save_path.replace('.', '/').split('/')[-2]
#     plot_save_path = "src/linearclass/" + save_path_first_part + "_plot.png"
#     print(plot_save_path)
#     util.plot(x_eval, y_eval, clf.theta, plot_save_path, correction=1.0)
#     # *** END CODE HERE ***


# class SoftmaxRegression:
#     """Softmax regression with SGD as the solver.

#     Example usage:
#         > clf = LogisticRegression()
#         > clf.fit(x_train, y_train)
#         > clf.predict(x_eval)
#     """
#     def __init__(self, max_iter=1000000, eps=1e-5,
#                  theta_0=None, verbose=True):
#         """
#         Args:
#             max_iter: Maximum number of iterations for the solver.
#             eps: Threshold for determining convergence.
#             theta_0: Initial guess for theta. If None, use the zero vector.
#             verbose: Print loss values during training.
#         """
#         self.theta = theta_0
#         self.max_iter = max_iter
#         self.eps = eps
#         self.verbose = verbose

#     def fit(self, x, y):
#         """Run Newton's Method to minimize J(theta) for logistic regression.

#         Args:
#             x: Shape (n_examples, dim).
#             y: Shape (n_examples,).
#         """

#         # *** START CODE HERE ***
#         n = x.shape[0]
#         d = x.shape[1]

#         if self.theta is None:
#             self.theta = np.zeros(d)

#         for _ in range(self.max_iter):
#             predictions = 1/(1+np.exp(-(x @ self.theta)))          # self.predict(x)
#             gradient = -1/n * x.T @ (y - predictions)
#             # vectorized computation of hessian inspired by svd decomp into sum of rank one matrices
#             hessian = 1/n * x.T @ np.diag(predictions * (1 - predictions)) @ x
#             new_theta = self.theta - np.linalg.inv(hessian) @ gradient

#             #compute loss
#             loss = -np.mean(y * np.log(predictions + self.eps) +
#                     (1 - y) * np.log(1 - predictions + self.eps)
#                     )
#             print(loss)

#             diff = self.theta - new_theta
#             self.theta = new_theta
#             if np.linalg.norm(diff) < self.eps:
#                 break
#         # *** END CODE HERE ***

#     def predict(self, x):
#         """Return predicted probabilities given new inputs x.

#         Args:
#             x: Shape (n_examples, dim).

#         Returns:
#             Outputs of shape (n_examples,).
#         """
#         #  if x.shape[1] < 3:   # add this so can reuse when calling on x that already has intercept
#         x = util.add_intercept(x)
#         # *** START CODE HERE ***
#         # print(x.shape, self.theta.shape)
#         return 1/(1+np.exp(-(x @ self.theta)))
#         # *** END CODE HERE ***

# if __name__ == '__main__':
#     script_dir = os.path.dirname(os.path.abspath(__file__))
    
#     main(train_path=os.path.join(script_dir, 'ds1_train.csv'),
#          valid_path=os.path.join(script_dir, 'ds1_valid.csv'),
#          save_path=os.path.join(script_dir, 'logreg_pred_1.txt'))

#     main(train_path=os.path.join(script_dir, 'ds2_train.csv'),
#          valid_path=os.path.join(script_dir, 'ds2_valid.csv'),
#          save_path=os.path.join(script_dir, 'logreg_pred_2.txt'))