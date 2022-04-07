import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # raise NotImplementedError()
    full_data = pd.read_csv(filename, parse_dates=["Date"])
    full_data = full_data.drop(full_data.index[full_data["Temp"] < -15])
    full_data = full_data.drop(full_data.index[full_data["Day"] > 31])
    full_data = full_data.drop(full_data.index[full_data["Day"] < 1])
    full_data = full_data.drop(full_data.index[full_data["Month"] > 12])
    full_data = full_data.drop(full_data.index[full_data["Month"] < 1])
    full_data = full_data.drop(full_data.index[full_data["Year"] <= 0])
    full_data["DayOfYear"] = full_data["Date"].dt.dayofyear
    design_mat = full_data.drop(columns=["Date", "Day"])
    return design_mat

def israel_sample_evaluation(samples):

    samples["Year"] = samples["Year"].astype(str)
    fig = px.scatter(samples, x="DayOfYear", y="Temp", color="Year",
                     title="Israel's change in average temp over a function of the DayOfYear")
    fig.show()

    data_frame = samples.groupby("Month").agg({"Temp":lambda x: np.std(x)})
    data_frame["Month"] = data_frame.index
    fig = px.bar(data_frame, x="Month", y="Temp", labels={"Temp": 'Temp std'},
                 title="The standard derivation of each month's daily temperature")
    fig.show()

def country_and_month_evaluation(data):
    data = data.groupby(["Country", "Month"]).Temp.agg(std="std", mean="mean").reset_index()
    px.line(data, x="Month", y="mean", error_y="std", color="Country", title="Evaluating avg temperature mean and std by countrie and month").show()

def fit_different_k(data):
    train_x, train_y, test_x, test_y = split_train_test(data["DayOfYear"], data["Temp"], 0.75)
    loss = []
    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(train_x, train_y)
        error = round(model.loss(test_x, test_y), 2)
        loss.append(error)
        print(f"Degree= {k}, test error= {error}")
    data_frame = pd.DataFrame({"degree": range(1, 11), "test error": loss})
    px.bar(data_frame, x="degree", y="test error", title="Calculating test error over polynomial reg with different degrees (Only Israel data)").show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    # raise NotImplementedError()
    features = load_data("../datasets/City_Temperature.csv")
    # Question 2 - Exploring data for specific country
    # raise NotImplementedError()
    israel_subset = features[features["Country"] == "Israel"]
    israel_sample_evaluation(israel_subset)

    # Question 3 - Exploring differences between countries
    # raise NotImplementedError()
    country_and_month_evaluation(features)

    # Question 4 - Fitting model for different values of `k`
    # raise NotImplementedError()
    fit_different_k(israel_subset)
    # Question 5 - Evaluating fitted model on different countries
    # raise NotImplementedError()
    model = PolynomialFitting(5)
    model.fit(israel_subset["DayOfYear"], israel_subset["Temp"])
    countries = features[features["Country"] != "Israel"]
    countries = countries["Country"].unique()
    country_loss = []
    for country in countries:
        country_subset = features[features["Country"] == country]
        country_loss.append(model.loss(country_subset["DayOfYear"], country_subset["Temp"]))
    data = pd.DataFrame({"Country": countries, "Test_error": country_loss})
    px.bar(data, x="Country", y="Test_error", title="Calculated loss of countries by a model of degree 5").show()

