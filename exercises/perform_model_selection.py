from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    # raise NotImplementedError()
    f = lambda x: (x+3)*(x+2)*(x+1)*(x-1)*(x-2)
    # X = np.random.uniform(-1.2, 2, n_samples)
    X = np.linspace(-1.2, 2, n_samples)
    y_noiseless = f(X)
    epsilon = np.random.normal(0, noise, n_samples)
    y = y_noiseless + epsilon
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), 2/3)
    train_X, train_y, test_X, test_y = train_X.values.ravel(), np.array(train_y), test_X.values.ravel(), np.array(test_y)
    go.Figure([go.Scatter(name="true data", x=X, y=y_noiseless, mode="markers"),
               go.Scatter(name="train data", x=train_X, y=train_y,
                          mode="markers"),
               go.Scatter(name="test data", x=test_X, y=test_y,
                          mode="markers")
               ]).update_layout(title="Plot of true(noiseless), test and train data", xaxis_title="X", yaxis_title="y").show()


    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_error, validation_error = [], []
    degrees = list(range(11))
    for k in degrees:
        train_avg, validation_avg = cross_validate(PolynomialFitting(k), train_X.flatten(), train_y, mean_square_error)
        train_error.append(train_avg)
        validation_error.append(validation_avg)
    go.Figure([go.Scatter(name="average training error", x=degrees, y=train_error, mode="markers+lines", line=dict(color="red")),
              go.Scatter(name="average validation error", x=degrees, y=validation_error, mode="markers+lines", line=dict(color="blue"))]
              ).update_layout(title="Average training and validation error of CV for polynomial fitting with degrees 0,1,...,10",
                                    xaxis_title="degree(k)", yaxis_title="avg error").show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    # raise NotImplementedError()
    best_k = int(np.argmin(validation_error))
    model = PolynomialFitting(best_k).fit(train_X.flatten(), train_y)
    print(f"Best degree: {best_k}")
    print(f"Test error: {mean_square_error(test_y, model.predict(test_X.flatten()))}")

def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    # raise NotImplementedError()
    X, y = datasets.load_diabetes(return_X_y=True)
    # X, y = np.array(X), np.array(y)
    train_X, train_y, test_X, test_y = X[:n_samples, :], y[:n_samples], X[n_samples:, :], y[n_samples:]
    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    # raise NotImplementedError()
    ridge_train_error, ridge_validation_error = [], []
    lasso_train_error, lasso_validation_error = [], []
    reg_param = np.linspace(0, 2.5, n_evaluations)
    for lam in reg_param:
        ridge = RidgeRegression(lam)
        lasso = Lasso(lam)
        ridge_errors = cross_validate(ridge, train_X, train_y, mean_square_error)
        lasso_errors = cross_validate(lasso, train_X, train_y, mean_square_error)
        lasso_train_error.append(lasso_errors[0])
        lasso_validation_error.append(lasso_errors[1])
        ridge_train_error.append(ridge_errors[0])
        ridge_validation_error.append(ridge_errors[1])
    go.Figure([go.Scatter(name="average training error", x=reg_param,
                          y=ridge_train_error, mode="markers+lines"),
               go.Scatter(name="average validation error", x=reg_param,
                          y=ridge_validation_error, mode="markers+lines")]
              ).update_layout(
        title="Training and validation error of CV for ridge regression",
        xaxis_title="lambda", yaxis_title="error").show()
    go.Figure([go.Scatter(name="average training error", x=reg_param,
                          y=lasso_train_error, mode="markers+lines"),
               go.Scatter(name="average validation error", x=reg_param,
                          y=lasso_validation_error, mode="markers+lines")]
              ).update_layout(
        title="Training and validation error of CV for lasso regression",
        xaxis_title="lambda", yaxis_title="error").show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    # raise NotImplementedError()
    ridge_lambda = reg_param[np.argmin(ridge_validation_error)]
    lasso_lambda = reg_param[np.argmin(lasso_validation_error)]
    print(f"Best ridge regularization parameter is: {ridge_lambda}")
    print(f"Best lasso regularization parameter is: {lasso_lambda}")
    ridge_model = RidgeRegression(ridge_lambda).fit(train_X, train_y)
    lasso_model = Lasso(lasso_lambda).fit(train_X, train_y)
    ls_model = LinearRegression().fit(train_X, train_y)
    print(f"Ridge model test error: {mean_square_error(test_y, ridge_model.predict(test_X))}")
    print(f"Lasso model test error: {mean_square_error(test_y, lasso_model.predict(test_X))}")
    print(f"Least Squares model test error: {mean_square_error(test_y, ls_model.predict(test_X))}")

if __name__ == '__main__':
    np.random.seed(0)
    # raise NotImplementedError()
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
