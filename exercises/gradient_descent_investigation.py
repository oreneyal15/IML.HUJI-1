import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from sklearn.metrics import roc_curve
from IMLearn.model_selection import cross_validate
from IMLearn.metrics.loss_functions import misclassification_error
import plotly.graph_objects as go



def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    # raise NotImplementedError()
    vals = []
    weights_ = []
    def callback(model, val, weights, norm, **kwargs):
        vals.append(val)
        weights_.append(weights)

    return callback, vals, weights_


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    # raise NotImplementedError()
    l1_convergence_rate, l2_convergence_rate = [], []
    l1_weights, l2_weights = [], []
    for eta in etas:
        l1 = L1(init)
        l2 = L2(init)
        callback, vals, weights = get_gd_state_recorder_callback()
        gd = GradientDescent(callback=callback, learning_rate=FixedLR(eta))
        l1_weights.append(gd.fit(l1, X=None, y=None))
        fig = plot_descent_path(L1, np.array(weights), title=f"For L1 Model With Fixed Learning Rate Of {eta}")
        fig.show()
        l1_convergence_rate.append(go.Scatter(x=list(range(len(vals))), y=vals, mode='markers', name=f"eta={eta}"))

        callback, vals, weights = get_gd_state_recorder_callback()
        gd = GradientDescent(callback=callback, learning_rate=FixedLR(eta))
        l2_weights.append(gd.fit(l2, X=None, y=None))
        fig = plot_descent_path(L2, np.array(weights), title=f"For L2 Model With Fixed Learning Rate Of {eta}")
        fig.show()
        l2_convergence_rate.append(go.Scatter(x=list(range(len(vals))), y=vals, mode='markers', name=f"eta={eta}"))

    fig1 = go.Figure(l1_convergence_rate).update_layout(xaxis_title="num iteration", yaxis_title="convergence rate",
        title=f"Convergence Rate Of L1 As A Function Of The GD Iteration")
    fig2 = go.Figure(l2_convergence_rate).update_layout(xaxis_title="num iteration", yaxis_title="convergence rate",
                                                        title=f"Convergence Rate Of L2 As A Function Of The GD Iteration")
    fig1.show()
    fig2.show()
    l1_losses = np.array([L1(w).compute_output() for w in l1_weights])
    l2_losses = np.array([L2(w).compute_output() for w in l2_weights])
    l1_loss = np.min(l1_losses)
    l2_loss = np.min(l2_losses)
    print(f"L1 min loss = {l1_loss}, with eta = {etas[int(np.argmin(l1_losses))]}")
    print(f"L2 min loss = {l2_loss}, with eta = {etas[int(np.argmin(l2_losses))]}")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    plots = []
    w = []
    for gamma in gammas:
        l1 = L1(init)
        callback, vals, weights= get_gd_state_recorder_callback()
        gd = GradientDescent(callback=callback, learning_rate=ExponentialLR(eta, gamma))
        w.append(gd.fit(l1, X=None, y=None))
        plots.append(go.Scatter(x=list(range(len(vals))), y=vals, mode='markers', name=f"gamma = {gamma}"))

    losses = np.array([L1(w_t).compute_output() for w_t in w])
    print("---------------------")
    print(f"Min loss for L1 with exponential decay = {np.min(losses)}, with gamma = {gammas[int(np.argmin(losses))]}")
    print("---------------------")
    # Plot algorithm's convergence for the different values of gamma
    # raise NotImplementedError()
    fig = go.Figure(plots).update_layout(
        xaxis_title="num iteration", yaxis_title="convergence rate",
        title=f"Convergence Rate Of L1 with exponential decay, With decay Rates of (0.9, 0.95, 0.99, 1)")
    fig.show()
    # Plot descent path for gamma=0.95
    # raise NotImplementedError()
    l1 = L1(init)
    l2 = L2(init)
    callback, vals, weights = get_gd_state_recorder_callback()
    gd = GradientDescent(callback=callback, learning_rate=ExponentialLR(eta, 0.95))
    gd.fit(l1, X=None, y=None)
    fig = plot_descent_path(L1, np.array(weights), title="Descent path of gamma=0.95 with the L1 model")
    fig.show()
    callback, vals, weights = get_gd_state_recorder_callback()
    gd = GradientDescent(callback=callback, learning_rate=ExponentialLR(eta, 0.95))
    gd.fit(l2, X=None, y=None)
    fig = plot_descent_path(L2, np.array(weights), title="Descent path of gamma=0.95 with the L2 model")
    fig.show()

def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    callback, vals, weights = get_gd_state_recorder_callback()
    solver = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000, callback=callback)
    model = LogisticRegression(solver=solver)
    model.fit(np.array(X_train), np.array(y_train))
    y_prob = model.predict_proba(np.array(X_train))
    fpr, tpr, thresholds = roc_curve(y_train, y_prob)
    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted LogisticRegression Model}}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    fig.show()
    roc_vals = tpr - fpr
    thresholds = np.round(thresholds, 2)
    best_alpha = thresholds[np.argmax(roc_vals)]
    print(f"Best alpha = {best_alpha}")
    model = LogisticRegression(alpha=best_alpha, solver=solver)
    model.fit(np.array(X_train), np.array(y_train))
    print(f"best alpha test loss = {model.loss(np.array(X_test), np.array(y_test))}")
    # Plotting convergence rate of logistic regression over SA heart disease data
    # raise NotImplementedError()

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    # raise NotImplementedError()
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

    l1_train_error, l1_validation_error, l2_train_error, l2_validation_error = [], [] , [], []
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    for lam in lambdas:
        logist_with_l1 = LogisticRegression(penalty="l1", alpha=0.5, lam=lam, solver=solver)
        logist_with_l2 = LogisticRegression(penalty="l2", alpha=0.5, lam=lam, solver=solver)
        l1_test, l1_validation = cross_validate(logist_with_l1, X_train, y_train, misclassification_error)
        l1_validation_error.append(l1_validation)
        l1_train_error.append(l1_test)

        l2_test, l2_validation = cross_validate(logist_with_l2, X_train, y_train, misclassification_error)
        l2_validation_error.append(l2_validation)
        l2_train_error.append(l2_test)
    l1_best_lam = lambdas[np.argmin(l1_validation_error)]
    l2_best_lam = lambdas[np.argmin(l2_validation_error)]
    logist_with_l1 = LogisticRegression(penalty="l1", alpha=0.5, lam=l1_best_lam, solver=solver).fit(X_train, y_train)
    logist_with_l2 = LogisticRegression(penalty="l2", alpha=0.5, lam=l2_best_lam, solver=solver).fit(X_train, y_train)
    print("-------------------")
    print(f"Best lambda for L1 model = {l1_best_lam}")
    print(f"Test loss for L1 model with lam={l1_best_lam} = {logist_with_l1.loss(X_test, y_test)}")
    print("-------------------")
    print(f"Best lambda for L2 model = {l2_best_lam}")
    print(f"Test loss for L2 model with lam={l2_best_lam} = {logist_with_l2.loss(X_test, y_test)}")

if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
