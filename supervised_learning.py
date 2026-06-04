import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
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

plt.rc('font', size=20)         
plt.rc('axes', titlesize=20)    
plt.rc('axes', labelsize=20)    
plt.rc('xtick', labelsize=16)   
plt.rc('ytick', labelsize=16)   
plt.rc('legend', fontsize=16)   
plt.rc('figure', titlesize=22) 

def build_supervised_regime_dataset(
    df: pd.DataFrame,
    cfg_sup,
    target_col: str = "regime",
):
    #Given the different type of method we are interested in doing alongside the custom loss
    #We formulate the data (in window form or just standard splits) to be passed down later
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
        df["regime_lag_1"] = df[target_col].shift(1)
        df["regime_lag_5"] = df[target_col].shift(5)

        if cfg_sup.evaluation_metric == 2:
            df["target"] = df[target_col].shift(-horizon)
        else:
            df["target"] = df[target_col]

        supervised_df = df.dropna().copy()

        drop_cols = [target_col, "target"]
        X = supervised_df.drop(columns=drop_cols)

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
    #Cross validation using walk foward method for the method and loss
    #A plot of the best lambda is reported


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
        
        if c3 > 0: 
            print(f"Avg train M1       : {(train_m1_sum / c3):.4f}")
        if c4 > 0:
            print(f"Avg test M1        : {(test_m1_sum / c4):.4f}")

            if c5 > 0:
                print(f"Avg train M2       : {(train_m2_sum / c5):.4f}")
            if c6 > 0:
                print(f"Avg test M2        : {(test_m2_sum / c6):.4f}")
        print(classification_report(y_true_all, y_pred_all))
        cm = confusion_matrix(y_true_all, y_pred_all, labels = [0, 1, 2])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1', '2'])
        
        # plt.show()
        disp.plot()
        plt.savefig('confusion_mat.png', dpi=300, bbox_inches='tight')

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
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1', '2'])
        disp.plot()
        plt.savefig('my_plot.png', dpi=300, bbox_inches='tight')


class SoftmaxRegression:
    def __init__(self, current_regime, max_iter=1000000, eps=1e-6,
                 theta_0=None, verbose=True, k=None):

        self.theta = theta_0
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.k = k
        self.loss_history = []
        self.current_regime = current_regime

    def fit(self, x, y, learning_rate, batch_size):
        #Fitting the softmax using inputs
        x_inter = self.add_intercept(x.copy())
        d = x_inter.shape[1]
        self.current_regime = np.asarray(self.current_regime)

        if self.theta is None:
            self.theta = np.zeros((d, self.k))

        for epoch in range(self.max_iter):
            self.gradient_descent_epoch(x_inter, y, learning_rate, batch_size)

            loss = self.ce_loss(x_inter, y)
            self.loss_history.append(loss)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

            if epoch > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.eps:
                break

    @staticmethod
    def add_intercept(x):
        new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
        new_x[:, 0] = 1
        new_x[:, 1:] = x

        return new_x

    def ce_loss(self, X, y):
        #All custom losses are defined here
        n = X.shape[0]
        logits = X @ self.theta
        prob = self.softmax(logits)

        log_loss = np.log(prob[np.arange(n), y] + 1e-12)
        loss = -log_loss.sum()

        transition_indicator = (self.current_regime != y).astype(float)
        if cfg_sup.penalty_type == "ce_standard":
            penalty = -cfg_sup.lam * (transition_indicator * log_loss).sum()

        elif cfg_sup.penalty_type == "margin_bar":
            diff_today_yesterday = np.zeros(n)
            for k in range(n):
                current = self.current_regime[k]
                future = y[k]  
                diff_today_yesterday[k] = cfg_sup.margin + prob[k, current] - prob[k, future]
            penalty = cfg_sup.lam * (transition_indicator * np.log(1 + np.exp(diff_today_yesterday))).sum()
        elif cfg_sup.penalty_type == "distance":
            inside = np.zeros(n)
            for i in range(n):
                for k in range(cfg.kmeans_k):
                    inside[i] += ((y[i] - k) ** 2) * prob[i, k] 
            penalty = cfg_sup.lam *(transition_indicator * inside).sum()
        else:
            penalty = 0

        reg = cfg_sup.l2reg_stren * np.sum(self.theta[1:] ** 2)

        return loss + penalty + reg

    def ce_grad(self, X, y, current_regime):
        n = X.shape[0]
        logits = X @ self.theta
        prob = self.softmax(logits)

        one_hot = self.one_hot(y)

        dZ = prob - one_hot
        dTheta = X.T @ dZ

        transition_indicator = (current_regime != y).astype(float)
        if cfg_sup.penalty_type == "ce_standard":
            dZ_trans = np.zeros((n, self.k))
            dZ_trans = transition_indicator[:, None] * (prob - one_hot)
            penalty_grad = cfg_sup.lam * X.T @ dZ_trans

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
            inside = np.zeros(n)
            for i in range(n): 
                for k in range(cfg.kmeans_k):
                    inside[i] += ((y[i] - k) ** 2) * prob[i, k]

            penalty_grad = np.zeros((X.shape[1], self.k))
            one_hot_y = self.one_hot(y)
            c = np.zeros(cfg.kmeans_k)
            for k in range(cfg.kmeans_k):
                c[k] = k

            for i in range(n):
                if transition_indicator[i]:
                    for k in range(self.k):
                        ctyi = np.dot(c, one_hot_y[i])
                        weight = cfg_sup.lam * prob[i, k] *((ctyi - c[k]) ** 2 - inside[i])  
                        penalty_grad[:, k] += weight * X[i]
            
        else:
            penalty_grad = 0

        reg_grad = np.zeros_like(self.theta)
        reg_grad[1:] = 2 * cfg_sup.l2reg_stren * self.theta[1:]
        return dTheta + penalty_grad + reg_grad

    def gradient_descent_epoch(self, X_shuffled, y_shuffled, learning_rate, batch_size):
        #Gradient Decsent one pass over the epoch

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
        one_hot = np.zeros((len(y), self.k))
        one_hot[np.arange(len(y)), y] = 1

        return one_hot

    def softmax(self, z):
        #Softmax function to transform logits into probabilites
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

        transition_indicator = (self.current_regime != y).float()
        
        if self.cfg_sup.penalty_type == "ce_standard":
            penalty = -self.cfg_sup.lam * (transition_indicator * log_loss).sum()

        elif self.cfg_sup.penalty_type == "margin_bar":
            diff_today_yesterday = []

            for k in range(n):
                current_prb = prob[k, self.current_regime[k]]
                future_prb = prob[k, y[k]]

                diff = self.cfg_sup.margin + current_prb - future_prb 
                diff_today_yesterday.append(diff)

            stacked_diff = torch.stack(diff_today_yesterday)
            penalty = self.cfg_sup.lam * (transition_indicator * (torch.log(1 + torch.exp(stacked_diff)))).sum()

        elif self.cfg_sup.penalty_type == "distance":
            inside = []
            for i in range(n): # -h):
                temp = 0.0
                for k in range(self.k):
                    temp += (y[i] - k) ** 2 * prob[i,k] 
                
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
        #Define loss for CNN
        n = score.shape[0]
        numerator = torch.exp(score)
        denominator = numerator.sum(dim = 1, keepdim= True)
        prob = numerator / denominator

        ground_lab = []
        for i in range(n):
            ground_lab.append(prob[i, y[i]])
        ground_lab = torch.stack(ground_lab)

        log_loss = torch.log(ground_lab + 1e-12)
        loss = -log_loss.sum()

        transition_indicator = (self.current_regime != y).float()
        
        if self.cfg_sup.penalty_type == "ce_standard":
            penalty = -self.cfg_sup.lam * (transition_indicator * log_loss).sum()
        elif self.cfg_sup.penalty_type == "margin_bar":
            diff_today_yesterday = []

            for k in range(n):

                current_prb = prob[k, self.current_regime[k]]
                future_prb = prob[k, y[k]]

                diff = self.cfg_sup.margin + current_prb - future_prb 
                diff_today_yesterday.append(diff)

            stacked_diff = torch.stack(diff_today_yesterday)
            penalty = self.cfg_sup.lam * (transition_indicator * (torch.log(1 + torch.exp(stacked_diff)))).sum()

        elif self.cfg_sup.penalty_type == "distance":
            inside = []
            for i in range(n): # -h):
                temp = 0.0
                for k in range(self.k):
                    temp += (y[i] - k) ** 2 * prob[i,k] 
                
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



def plot_regime_timeline(y_true, y_pred, pred_proba, save_path=None):
    #plot the time series representation for experimentation + conclusion

    N = len(y_true)
    bull_color = "#2ca02c"      
    rec_color = "#ffbf00"       
    crisis_color = "#d62728"    

    regime_colors = np.array([
        plt.matplotlib.colors.to_rgb(bull_color),
        plt.matplotlib.colors.to_rgb(rec_color),
        plt.matplotlib.colors.to_rgb(crisis_color),
    ])

    true_strip = regime_colors[y_true]
    pred_strip = regime_colors[y_pred]

    bull_cmap = LinearSegmentedColormap.from_list("bull_prob", ["white", bull_color])

    rec_cmap = LinearSegmentedColormap.from_list("rec_prob",["white", rec_color])
    crisis_cmap = LinearSegmentedColormap.from_list("crisis_prob",["white", crisis_color])

    fig, axes = plt.subplots(5, 1, figsize=(18, 5), sharex=True, gridspec_kw={"hspace": 0.15})

    axes[0].imshow(
        true_strip[np.newaxis, :, :],
        aspect="auto"
    )
    axes[0].set_ylabel("True")

    axes[1].imshow(
        pred_strip[np.newaxis, :, :],
        aspect="auto"
    )
    axes[1].set_ylabel("Pred")

    axes[2].imshow(
        pred_proba[:, 0][np.newaxis, :],
        aspect="auto",
        cmap=bull_cmap,
        vmin=0,
        vmax=1
    )
    axes[2].set_ylabel("0")

    axes[3].imshow(
        pred_proba[:, 1][np.newaxis, :],
        aspect="auto",
        cmap=rec_cmap,
        vmin=0,
        vmax=1
    )
    axes[3].set_ylabel("1")

    axes[4].imshow(
        pred_proba[:, 2][np.newaxis, :],
        aspect="auto",
        cmap=crisis_cmap,
        vmin=0,
        vmax=1
    )
    axes[4].set_ylabel("2")

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



if __name__ == "__main__":
    main()
