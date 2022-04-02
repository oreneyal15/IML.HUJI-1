import datetime

from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
from IMLearn.base import BaseEstimator
from utils import *
import sklearn as sk
import numpy as np
import pandas as pd


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    full_data = pd.read_csv(filename).drop_duplicates()
    features = full_data[[
        # "accommadation_type_name",
                          "hotel_star_rating",
                          "no_of_room",
                          "original_selling_amount",
                          "no_of_adults",
                            "no_of_children",
                            "no_of_extra_bed",
                            "no_of_room",
                          # "cancellation_policy_code",
                          "is_first_booking"
                          ]]
    features = pd.get_dummies(features)
    labels = full_data[["cancellation_datetime"]]
    labels["cancellation_datetime"] = labels["cancellation_datetime"].where(labels["cancellation_datetime"].isnull(), 1).fillna(0).astype(int)
    return features, labels


def preprocess_cancellation_policies(X):
    data = []
    cols = ["no_show_1N", "no_show_100P"]
    for val in X.values:
        row = [0] * len(cols)
        if val[0].endswith("_100P"):
            row[1] = 1
        if val[0].endswith("_1N"):
            row[0] = 1
        # if not val:
        #     continue
        # if val[-1] == "N":
        #     row[3] = val[-2]
        # else:
        #     temp = val[-3:]
        data.append(row)
    return pd.DataFrame(data, columns=cols)


def preprocessing(X: pd.DataFrame):
    no_duplicates = X.drop_duplicates()
    index_dict = {no_duplicates.values[i][0]: i for i in
                  range(no_duplicates.size)}
    data = []
    for val in X.values:
        cur_row = [0] * no_duplicates.size
        cur_row[index_dict[val[0]]] = 1
        data.append(cur_row)
    return pd.DataFrame(data, columns=no_duplicates.values)


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray,
                        filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(
        filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data(
        "../datasets/agoda_cancellation_train.csv")
    train_X, test_X, train_y, test_y = sk.model_selection.train_test_split(df, cancellation_labels, test_size=0.25)
    train_y = np.ravel(train_y)
    test_y = np.ravel(test_y)
    # Fit model over data
    estimator = sk.linear_model.LogisticRegression(random_state=0, max_iter=20000).fit(train_X, train_y)
    full_data = pd.read_csv("../datasets/test_set_week_1.csv").drop_duplicates()
    features = full_data[[
        # "accommadation_type_name",
        "hotel_star_rating",
        "no_of_room",
        "original_selling_amount",
        "no_of_adults",
        "no_of_children",
        "no_of_extra_bed",
        "no_of_room",
        # "cancellation_policy_code",
        "is_first_booking"
    ]]
    features = pd.get_dummies(features)
    # Store model predictions over test set
    evaluate_and_export(estimator, features, "../datasets/318967049_318352820_208480921.csv")