from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # raise NotImplementedError()
    full_data = pd.read_csv(filename).dropna().drop_duplicates()
    full_data = full_data.drop(full_data.index[full_data["price"] <= 0])
    full_data["date"] = full_data["date"].str[:4]
    full_data = full_data.drop(
        full_data.index[full_data["date"] == '0'])
    full_data = full_data.drop(full_data.index[full_data["bedrooms"] < 0])
    full_data = full_data.drop(full_data.index[full_data["bathrooms"] < 0])
    full_data = full_data.drop(full_data.index[full_data["floors"] < 0])
    full_data = full_data.drop(full_data.index[full_data["sqft_living"] <= 0])
    full_data = full_data.drop(full_data.index[full_data["sqft_lot"] <= 0])
    full_data = full_data.drop(full_data.index[full_data["sqft_above"] <= 0])
    full_data = full_data.drop(full_data.index[full_data["sqft_basement"] <= 0])
    full_data = full_data.drop(full_data.index[full_data["yr_built"] <= 0])
    full_data = full_data.drop(full_data.index[full_data["yr_renovated"] < 0])
    full_data = full_data.drop(full_data.index[full_data["sqft_living15"] < 0])
    full_data = full_data.drop(full_data.index[full_data["sqft_lot15"] < 0])
    full_data = pd.get_dummies(full_data, columns=["date"])
    response = full_data["price"]
    design_matrix = full_data.drop(columns=["id", "zipcode", "lat", "long", "price"])
    return design_matrix, response

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # raise NotImplementedError()
    for feature, feature_val in X.iteritems():
        pc = X[feature].cov(y) / (np.std(feature_val) * np.std(y))
        fig = go.Figure([go.Scatter(x=X[feature], y=y, mode='markers')],

                        layout=go.Layout(
                            title=f"Feature: {feature}, Pearson Correlation = {pc}",
                            xaxis_title="feature value",
                            yaxis_title="response"
                        ))
        fig.write_image(output_path + str(feature) + "_graph.png")






if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    # raise NotImplementedError()
    df, prices = load_data("../datasets/house_prices.csv")
    # Question 2 - Feature evaluation with respect to response
    # raise NotImplementedError()
    feature_evaluation(df, prices, "../ex2_graphs/")
    # Question 3 - Split samples into training- and testing sets.
    # raise NotImplementedError()
    train_X, train_y, test_X, test_y = split_train_test(df, prices, 0.75)
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # raise NotImplementedError()
    p_range = np.arange(0.1, 1.0, 0.01)
    loss_mean, loss_std = [], []
    conf_interval_minus = []
    conf_interval_plus = []
    for p in p_range:
        loss = []
        for _ in range(10):
            temp_train_X, temp_train_y, temp_test_X, temp_test_y = split_train_test(train_X, train_y, p)
            model = LinearRegression()
            model.fit(temp_train_X, temp_train_y)
            loss.append(model.loss(test_X, test_y))
        loss_mean.append(np.mean(loss))
        loss_std.append(np.std(loss))
        conf_interval_minus.append(np.mean(loss) - 2 * np.std(loss))
        conf_interval_plus.append(np.mean(loss) + 2 * np.std(loss))
    p_range *= 100
    go.Figure([go.Scatter(x=p_range, y=loss_mean, mode="markers+lines", name="Mean Prediction",
                line=dict(dash="dash"),
                marker=dict(color="green", opacity=.7)),
     go.Scatter(x=p_range, y=conf_interval_minus, fill=None, mode="lines",
                line=dict(color="lightgrey"), showlegend=False),
     go.Scatter(x=p_range, y=conf_interval_plus, fill='tonexty', mode="lines",
                line=dict(color="lightgrey"), showlegend=False)],
              layout=go.Layout(title="Mean of loss prediction over 10 sampels of diffrent test and train sample size",
                               xaxis_title="percentage %",
                               yaxis_title="loss mean")).show()
