import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    # raise NotImplementedError()
    ensemble = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    train_error = [ensemble.partial_loss(train_X, train_y, i) for i in range(1, n_learners+1)]
    test_error = [ensemble.partial_loss(test_X, test_y, i) for i in range(1, n_learners+1)]
    x_vals = [i for i in range(1, n_learners+1)]
    go.Figure([go.Scatter(x=x_vals, y=train_error, mode="lines", name="train error",
                          marker=dict(color="green", opacity=.7)),
               go.Scatter(x=x_vals, y=test_error, name="test error",
                          mode="lines",
                          line=dict(color="orange")),
               ],
              layout=go.Layout(
                  title="The training and test errors as a function of the number of fitted learners",
                  xaxis_title="number of fitted learners",
                  yaxis_title="error")).show()
    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    symbols = {1:"circle", -1:"x"}
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[f"num of fitted learners: {m}" for m in
                                        T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(lambda X: ensemble.partial_predict(X, t), lims[0], lims[1],
                                         showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                                   showlegend=False,
                                   marker=dict(color=test_y, symbol=class_symbols[test_y.astype(int)],
                                               colorscale=[custom[0],
                                                           custom[-1]],
                                               line=dict(color="black",
                                                         width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.update_layout(title="Decision boundary obtained by using the ensemble of different size",
                      margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()


    # Question 3: Decision surface of best performing ensemble
    # raise NotImplementedError()
    num_of_models = test_error.index(min(test_error)) + 1
    y_pred = ensemble.partial_predict(test_X, num_of_models)
    go.Figure([decision_surface(lambda X: ensemble.partial_predict(X, num_of_models), lims[0], lims[1],
                                         showscale=False),
               go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                          showlegend=False,
                          marker=dict(color=test_y, symbol=class_symbols[test_y.astype(int)],
                                      colorscale=[custom[0],
                                                  custom[-1]],
                                      line=dict(color="black",
                                                width=1)))],
              layout=go.Layout(title=f"Ensemble size: {num_of_models}, accuracy = {float(np.sum(test_y == y_pred) / test_y.size)}")).show()

    # Question 4: Decision surface with weighted samples
    # raise NotImplementedError()
    D = (ensemble.D_ / np.max(ensemble.D_)) * 10
    go.Figure([decision_surface(ensemble.predict, lims[0], lims[1], showscale=False),
               go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                          showlegend=False,
                          marker=dict(color=train_y, symbol=class_symbols[train_y.astype(int)], size=D,
                                      colorscale=[custom[0],
                                                  custom[-1]],
                                      line=dict(color="black",
                                                width=1)))
               ], layout=go.Layout(
        title=f"A plot of the full ensemble with dot size proportionate to its weight")).update_xaxes(visible=False).update_yaxes(visible=False).show()

if __name__ == '__main__':
    np.random.seed(0)
    # raise NotImplementedError()
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
